/*
 * trimem_extract.c — "Custom Scan (TriMEM SemExtract)": batched LLM label extraction (0.3.0).
 *
 *   SELECT d.id, trimem.sem_extract('Which search strategy ...? Answer one of: ...', d.changes) AS s
 *   FROM attempt d WHERE trimem.sem_extract('...', d.changes) = 'greedy'
 *
 * trimem.sem_extract(question, input) -> text answers `question` about `input` with a short,
 * lower-case label (NULL for a NULL input or when the LLM gives no parsable answer). It is planned
 * like SemScore (ADR-0028): a SemExtract node over the relation whose column it reads (BASE) or over
 * the final scan/join relation (TOP) computes every label the query needs, in bounded batches. The
 * node keeps the labels in a per-query table and registers it before emitting each row; the
 * projection / qual that evaluates trimem.sem_extract only looks the label up.
 *
 * Restriction quals over a label (WHERE sem_extract(q, d.x) = 'greedy') are lifted off the base
 * relation (like sem_filter) and become the SemExtract node's own qual, so relational operators on
 * the label run right where the label is computed. Join conditions over labels are rejected (v1).
 *
 * Transport: the mailbox carries only completion; the worker commits the label into
 * trimem.sem_cache.extracted and the node re-reads its keys with a fresh snapshot (the worker's
 * commit happens after the query's snapshot was taken).
 */
#include "postgres.h"

#include <math.h>

#include "catalog/pg_operator_d.h"
#include "catalog/pg_type.h"
#include "commands/explain.h"
#include "executor/executor.h"
#include "executor/spi.h"
#include "nodes/extensible.h"
#include "nodes/makefuncs.h"
#include "nodes/nodeFuncs.h"
#include "optimizer/cost.h"
#include "optimizer/optimizer.h"
#include "optimizer/pathnode.h"
#include "optimizer/restrictinfo.h"
#include "optimizer/tlist.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/hsearch.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/ruleutils.h"
#include "utils/selfuncs.h"
#include "utils/snapmgr.h"

#include "trimem.h"

/*
 * Selectivity of `label = 'x'` with fewer than LABEL_STATS_MIN cached labels for the question:
 * labels come from a short closed set, so PG's generic 0.5% equality default is far too low.
 */
#define LABEL_DEFAULT_EQ_SEL 0.2
#define LABEL_STATS_MIN 10

typedef struct LabelEntry
{
    char key[65];
    char *label; /* NULL = no parsable answer */
} LabelEntry;

/* Restriction quals over labels lifted off one base relation (per planner call). */
typedef struct ExtractLift
{
    PlannerInfo *root;
    RelOptInfo *rel;
    List *quals; /* RestrictInfo * */
    double qsel;
} ExtractLift;

typedef struct CachedCount
{
    char *question;
    double n;
} CachedCount;

typedef struct ExtractPlannerState
{
    List *lifts;     /* ExtractLift * */
    List *counts;    /* CachedCount *: cached labels per question (memo for the trials) */
    int quals_lifted;
    int quals_planned;
} ExtractPlannerState;

static ExtractPlannerState pstate = {NIL, NIL, 0, 0};

static CustomPathMethods extract_path_methods;
static CustomScanMethods extract_scan_methods;
static CustomExecMethods extract_exec_methods;
static SPIPlanPtr extract_probe = NULL;

bool
trimem_extract_lookup(void *table, const char *key, char **label)
{
    LabelEntry *e = hash_search((HTAB *) table, key, HASH_FIND, NULL);

    if (!e)
        return false;
    *label = e->label;
    return true;
}

/*
 * Label tables of every SemExtract node currently executing in this backend. The active table
 * (last node to emit a row) answers almost every lookup; this registry covers nesting, e.g. a
 * correlated subquery whose own SemExtract node ran while the outer projection was evaluated.
 * Cells live in TopTransactionContext; the xact callback drops the list.
 */
typedef struct LiveTable
{
    HTAB *table;
    const char *model;
    const char *pv;
} LiveTable;

static List *live_tables = NIL;

bool
trimem_extract_has_live(void)
{
    return live_tables != NIL;
}

void
trimem_extract_forget_live(void)
{
    live_tables = NIL;
}

bool
trimem_extract_lookup_live(const char *question, const char *input, char **label)
{
    ListCell *lc;

    foreach (lc, live_tables)
    {
        LiveTable *t = lfirst(lc);
        char key[65];

        trimem_key_hex(TRIMEM_OP_EXTRACT, t->model, t->pv, question, input, key);
        if (trimem_extract_lookup(t->table, key, label))
            return true;
    }
    return false;
}

/* ---------------------------------------------------------------- planner-call state */
void *
trimem_extract_planner_enter(void)
{
    ExtractPlannerState *saved = palloc(sizeof(ExtractPlannerState));

    *saved = pstate;
    pstate.lifts = NIL;
    pstate.counts = NIL;
    pstate.quals_lifted = pstate.quals_planned = 0;
    return saved;
}

void
trimem_extract_planner_leave(void *saved)
{
    pstate = *(ExtractPlannerState *) saved;
}

/* Safety net (like SemFilter's): a lifted label qual must be applied by a planned SemExtract. */
void
trimem_extract_planner_check(void)
{
    if (pstate.quals_planned < pstate.quals_lifted)
        ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR),
                        errmsg("trimem: %d sem_extract qual(s) lifted but only %d applied by SemExtract node(s)",
                               pstate.quals_lifted, pstate.quals_planned),
                        errhint("Another set_rel_pathlist_hook replaced the paths. List tjs_pg "
                                "BEFORE trimem_pg in shared_preload_libraries.")));
}

/* ---------------------------------------------------------------- planning helpers */
typedef struct CallCollect
{
    Oid fid;
    List *calls; /* distinct FuncExpr * */
} CallCollect;

static bool
collect_calls(Node *node, CallCollect *c)
{
    if (node == NULL)
        return false;
    if (IsA(node, FuncExpr) && ((FuncExpr *) node)->funcid == c->fid)
    {
        if (!list_member(c->calls, node))
            c->calls = lappend(c->calls, node);
        return false;
    }
    return expression_tree_walker(node, collect_calls, c);
}

static bool
has_func(Node *node, Oid fid)
{
    CallCollect c = {fid, NIL};

    if (!OidIsValid(fid))
        return false;
    collect_calls(node, &c);
    return c.calls != NIL;
}

/* Every sem_extract call the query evaluates: select list / GROUP BY / ORDER BY, HAVING, quals. */
static List *
extract_calls(PlannerInfo *root)
{
    CallCollect c = {trimem_sem_extract_oid(), NIL};
    ListCell *lc;

    if (!OidIsValid(c.fid))
        return NIL;
    collect_calls((Node *) root->processed_tlist, &c);
    collect_calls(root->parse->havingQual, &c);
    foreach (lc, pstate.lifts)
    {
        ExtractLift *l = lfirst(lc);
        ListCell *lq;

        if (l->root != root)
            continue;
        foreach (lq, l->quals)
            collect_calls((Node *) ((RestrictInfo *) lfirst(lq))->clause, &c);
    }
    return c.calls;
}

/* Lifted label quals of this query level; NIL + false if one of them is not evaluable on `rel`. */
static List *
quals_for(PlannerInfo *root, RelOptInfo *rel, double *qsel, bool *ok)
{
    List *quals = NIL;
    ListCell *lc;

    *qsel = 1.0;
    *ok = true;
    foreach (lc, pstate.lifts)
    {
        ExtractLift *l = lfirst(lc);

        if (l->root != root)
            continue;
        if (!bms_is_subset(l->rel->relids, rel->relids))
        {
            *ok = false;
            return NIL;
        }
        quals = list_concat(quals, list_copy(l->quals));
        *qsel *= l->qsel;
    }
    return quals;
}

/* Cached labels for `question` under the current model / prompt version (memoized per planner call). */
static double
cached_labels(const char *question)
{
    ListCell *lc;
    CachedCount *cc;
    double n = 0;

    foreach (lc, pstate.counts)
    {
        cc = lfirst(lc);
        if (strcmp(cc->question, question) == 0)
            return cc->n;
    }
    {
        Oid t[3] = {TEXTOID, TEXTOID, TEXTOID};
        Datum v[3] = {CStringGetTextDatum(trimem_model), CStringGetTextDatum(trimem_prompt_version),
                      CStringGetTextDatum(question)};

        if (SPI_connect() != SPI_OK_CONNECT)
            elog(ERROR, "trimem: SPI_connect failed");
        if (SPI_execute_with_args("SELECT count(*)::float8 FROM trimem.sem_cache WHERE model = $1 "
                                  "AND prompt_version = $2 AND predicate = $3 AND op = '" TRIMEM_OP_EXTRACT
                                  "' AND extracted IS NOT NULL",
                                  3, t, v, NULL, true, 1) == SPI_OK_SELECT &&
            SPI_processed == 1)
        {
            bool isnull;

            n = DatumGetFloat8(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));
        }
        SPI_finish();
    }
    cc = palloc(sizeof(CachedCount));
    cc->question = pstrdup(question);
    cc->n = n;
    pstate.counts = lappend(pstate.counts, cc);
    return n;
}

/*
 * Selectivity of one lifted label qual. `sem_extract(q, x) = 'v'` / `<> 'v'` use the label
 * distribution already cached for q (share of cached labels equal to 'v'); anything else keeps
 * PG's estimate.
 */
static double
label_qual_sel(PlannerInfo *root, RestrictInfo *ri, Oid fid)
{
    Node *cl = (Node *) ri->clause;

    if (IsA(cl, OpExpr) && list_length(((OpExpr *) cl)->args) == 2)
    {
        OpExpr *op = (OpExpr *) cl;
        Oid ne = get_negator(TextEqualOperator);
        Node *a = strip_implicit_coercions(linitial(op->args));
        Node *b = strip_implicit_coercions(lsecond(op->args));

        if (IsA(b, FuncExpr))
        {
            Node *t = a;

            a = b;
            b = t;
        }
        if ((op->opno == TextEqualOperator || op->opno == ne) && IsA(a, FuncExpr) &&
            ((FuncExpr *) a)->funcid == fid && IsA(b, Const) && !((Const *) b)->constisnull &&
            ((Const *) b)->consttype == TEXTOID && IsA(linitial(((FuncExpr *) a)->args), Const) &&
            !((Const *) linitial(((FuncExpr *) a)->args))->constisnull)
        {
            Oid t[4] = {TEXTOID, TEXTOID, TEXTOID, TEXTOID};
            Datum v[4] = {CStringGetTextDatum(trimem_model), CStringGetTextDatum(trimem_prompt_version),
                          ((Const *) linitial(((FuncExpr *) a)->args))->constvalue,
                          ((Const *) b)->constvalue};
            double total = 0, match = 0, eq;

            if (SPI_connect() != SPI_OK_CONNECT)
                elog(ERROR, "trimem: SPI_connect failed");
            if (SPI_execute_with_args("SELECT count(*)::float8, (count(*) FILTER (WHERE extracted = $4))::float8 "
                                      "FROM trimem.sem_cache WHERE model = $1 AND prompt_version = $2 "
                                      "AND predicate = $3 AND op = '" TRIMEM_OP_EXTRACT
                                      "' AND extracted IS NOT NULL",
                                      4, t, v, NULL, true, 1) == SPI_OK_SELECT &&
                SPI_processed == 1)
            {
                bool isnull;

                total = DatumGetFloat8(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));
                match = DatumGetFloat8(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 2, &isnull));
            }
            SPI_finish();
            eq = total >= LABEL_STATS_MIN ? Max(match, 0.5) / total : LABEL_DEFAULT_EQ_SEL;
            return op->opno == TextEqualOperator ? eq : 1.0 - eq;
        }
    }
    return clause_selectivity(root, (Node *) ri, 0, JOIN_INNER, NULL);
}

/* ---------------------------------------------------------------- lifting label quals */
void
trimem_extract_lift(PlannerInfo *root, RelOptInfo *rel)
{
    Oid fid, ffids[2];
    List *quals = NIL, *others = NIL, *vars = NIL;
    ListCell *lc;
    double qsel = 1.0, core_sel;
    ExtractLift *l;

    if (!trimem_enable || (rel->baserestrictinfo == NIL && rel->joininfo == NIL))
        return;
    fid = trimem_sem_extract_oid();
    if (!OidIsValid(fid))
        return;
    ffids[0] = trimem_sem_filter_oid();
    ffids[1] = trimem_sem_filter3_oid();
    foreach (lc, rel->joininfo)
        if (has_func((Node *) ((RestrictInfo *) lfirst(lc))->clause, fid))
            ereport(ERROR, (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                            errmsg("trimem.sem_extract in a join condition is not supported (v1)"),
                            errhint("Compute the labels in the select list (e.g. of a MATERIALIZED CTE) "
                                    "and compare them in the outer query.")));
    foreach (lc, rel->baserestrictinfo)
    {
        RestrictInfo *ri = lfirst(lc);

        /* a clause that also holds sem_filter is left to lift_sem (which rejects non-top-level use) */
        if (has_func((Node *) ri->clause, fid) && !has_func((Node *) ri->clause, ffids[0]) &&
            !has_func((Node *) ri->clause, ffids[1]))
            quals = lappend(quals, ri);
        else
            others = lappend(others, ri);
    }
    if (quals == NIL)
        return;
    if (rel->reloptkind != RELOPT_BASEREL)
        ereport(ERROR, (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                        errmsg("trimem.sem_extract quals on inheritance/partition children are not supported (v1)")));

    /* undo the core's estimate of the quals; ours replaces it where the label cache tells more */
    core_sel = clauselist_selectivity(root, quals, rel->relid, JOIN_INNER, NULL);
    core_sel = Min(Max(core_sel, 1e-6), 1.0);
    foreach (lc, quals)
        qsel *= label_qual_sel(root, lfirst(lc), fid);
    qsel = Min(Max(qsel, 1e-6), 1.0);
    rel->baserestrictinfo = others;
    foreach (lc, rel->indexlist)
    {
        IndexOptInfo *ix = lfirst(lc);
        ListCell *lq;

        foreach (lq, quals)
            ix->indrestrictinfo = list_delete_ptr(ix->indrestrictinfo, lfirst(lq));
    }

    /* The input scan no longer applies the quals: its rows are the pre-qual rows. */
    rel->rows = clamp_row_est(rel->rows / core_sel);
    foreach (lc, rel->pathlist)
    {
        Path *p = lfirst(lc);

        p->rows = clamp_row_est(p->rows / core_sel);
    }
    foreach (lc, rel->ppilist)
    {
        ParamPathInfo *ppi = lfirst(lc);

        ppi->ppi_rows = clamp_row_est(ppi->ppi_rows / core_sel);
    }

    /* The qual's columns must reach the SemExtract node (BASE: right above; TOP: above the join). */
    foreach (lc, quals)
        vars = list_concat_unique(vars, pull_var_clause((Node *) ((RestrictInfo *) lfirst(lc))->clause,
                                                        PVC_RECURSE_AGGREGATES | PVC_RECURSE_WINDOWFUNCS |
                                                            PVC_INCLUDE_PLACEHOLDERS));
    foreach (lc, vars)
    {
        Var *v = lfirst(lc);

        if (IsA(v, Var) && v->varno == rel->relid && v->varattno >= rel->min_attr)
            rel->attr_needed[v->varattno - rel->min_attr] =
                bms_add_member(rel->attr_needed[v->varattno - rel->min_attr], 0);
        if (!list_member(rel->reltarget->exprs, v))
            add_column_to_pathtarget(rel->reltarget, (Expr *) v, 0);
    }
    set_pathtarget_cost_width(root, rel->reltarget);

    l = palloc0(sizeof(ExtractLift));
    l->root = root;
    l->rel = rel;
    l->quals = quals;
    l->qsel = qsel;
    pstate.lifts = lappend(pstate.lifts, l);
    pstate.quals_lifted += list_length(quals);
}

/* Equality between labels of two relations becomes an EquivalenceClass, not a joininfo clause. */
void
trimem_extract_check_joins(PlannerInfo *root)
{
    Oid fid = trimem_sem_extract_oid();
    ListCell *lc;

    if (!OidIsValid(fid))
        return;
    foreach (lc, root->eq_classes)
    {
        EquivalenceClass *ec = lfirst(lc);
        ListCell *lm;

        if (ec->ec_has_const || bms_membership(ec->ec_relids) != BMS_MULTIPLE)
            continue;
        foreach (lm, ec->ec_members)
            if (has_func((Node *) ((EquivalenceMember *) lfirst(lm))->em_expr, fid))
                ereport(ERROR, (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                                errmsg("trimem.sem_extract in a join condition is not supported (v1)"),
                                errhint("Compute the labels in the select list (e.g. of a MATERIALIZED CTE) "
                                        "and compare them in the outer query.")));
    }
}

bool
trimem_extract_present(PlannerInfo *root)
{
    return extract_calls(root) != NIL;
}

/* TOP placement applies lifted label quals above the joins: only valid for inner joins. */
bool
trimem_extract_top_ok(PlannerInfo *root)
{
    ListCell *lc;

    if (root->join_info_list == NIL)
        return true;
    foreach (lc, pstate.lifts)
        if (((ExtractLift *) lfirst(lc))->root == root)
            return false;
    return true;
}

double
trimem_extract_qual_sel(PlannerInfo *root, RelOptInfo *rel)
{
    double qsel;
    bool ok;

    (void) quals_for(root, rel, &qsel, &ok);
    return ok ? qsel : 1.0;
}

/* The single base relation whose columns every sem_extract input (and label qual) reads, or NULL. */
RelOptInfo *
trimem_extract_base_rel(PlannerInfo *root, List *initial_rels)
{
    List *calls = extract_calls(root);
    Relids rels = NULL;
    ListCell *lc;
    int relid;

    if (calls == NIL)
        return NULL;
    foreach (lc, calls)
        rels = bms_add_members(rels, pull_varnos(root, lsecond(((FuncExpr *) lfirst(lc))->args)));
    foreach (lc, pstate.lifts)
        if (((ExtractLift *) lfirst(lc))->root == root)
            rels = bms_add_members(rels, ((ExtractLift *) lfirst(lc))->rel->relids);
    if (!bms_get_singleton_member(rels, &relid))
        return NULL;
    foreach (lc, initial_rels)
    {
        RelOptInfo *r = lfirst(lc);

        if (r->reloptkind == RELOPT_BASEREL && bms_equal(r->relids, rels))
            return r;
    }
    return NULL;
}

/*
 * SemExtract paths over every path of `rel` (rel is not modified). Cost: per DISTINCT input, an
 * LLM call unless the label is cached — the hit rate is the cache coverage of the question
 * (cached labels / distinct inputs) — plus one round trip per batch with at least one miss.
 */
List *
trimem_extract_paths(PlannerInfo *root, RelOptInfo *rel, const char *placement)
{
    List *calls = extract_calls(root);
    List *inputs = NIL, *questions = NIL, *quals, *out = NIL;
    double *ncached;
    double qsel;
    bool ok;
    ListCell *lc;
    int batch = trimem_max_batch, i = 0;
    QualCost qc;

    if (calls == NIL || rel->pathlist == NIL)
        return NIL;
    quals = quals_for(root, rel, &qsel, &ok);
    if (!ok)
        return NIL;
    ncached = palloc(sizeof(double) * list_length(calls));
    foreach (lc, calls)
    {
        FuncExpr *fn = lfirst(lc);
        Node *q = linitial(fn->args);
        Expr *input = lsecond(fn->args);

        if (!IsA(q, Const) || ((Const *) q)->constisnull)
            ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                            errmsg("trimem.sem_extract question must be a non-NULL constant")));
        if (!bms_is_subset(pull_varnos(root, (Node *) input), rel->relids))
            return NIL; /* computed elsewhere; fails closed at execution */
        questions = lappend(questions, makeString(TextDatumGetCString(((Const *) q)->constvalue)));
        inputs = lappend(inputs, copyObject(input));
        ncached[i++] = cached_labels(strVal(llast(questions)));
    }
    cost_qual_eval(&qc, quals, root);
    foreach (lc, rel->pathlist)
    {
        Path *child = lfirst(lc);
        CustomPath *cp = makeNode(CustomPath);
        double rows = child->rows;
        double b = Min((double) batch, rows);
        double llm = 0, hit_min = 1.0, c_first = 0;
        StringInfoData model;
        ListCell *li;

        initStringInfo(&model);
        i = 0;
        foreach (li, inputs)
        {
            double distinct = Min(rows, estimate_num_groups(root, list_make1(lfirst(li)), rows, NULL, NULL));
            double hit = Min(1.0, ncached[i] / Max(distinct, 1.0));
            double c_eff = trimem_llm_cost_per_eval * (1.0 - hit) + trimem_cache_cost_per_eval * hit;

            llm += distinct * c_eff;
            c_first += Min(b, distinct) * c_eff;
            hit_min = Min(hit_min, hit);
            appendStringInfo(&model, "%sdistinct=%.0f cached=%.0f hit=%.2f c_eff=%.0f", i ? "; " : "",
                             distinct, ncached[i], hit, c_eff);
            i++;
        }
        {
            double p_trip = 1.0 - pow(hit_min, b);
            double trips = ceil(rows / Max(b, 1.0)) * p_trip;

            cp->path.startup_cost = child->startup_cost + c_first + p_trip * trimem_llm_cost_per_batch + qc.startup;
            cp->path.total_cost = child->total_cost + llm + trips * trimem_llm_cost_per_batch +
                                  rows * (cpu_tuple_cost + qc.per_tuple) + qc.startup;
        }
        if (quals != NIL)
            appendStringInfo(&model, "; label qual sel=%.3f", qsel);
        cp->path.pathtype = T_CustomScan;
        cp->path.parent = rel;
        cp->path.pathtarget = child->pathtarget;
        cp->path.param_info = child->param_info;
        cp->path.parallel_safe = false;
        cp->path.rows = clamp_row_est(rows * qsel);
        cp->path.pathkeys = child->pathkeys; /* order-preserving */
        cp->flags = CUSTOMPATH_SUPPORT_PROJECTION;
        cp->custom_paths = list_make1(child);
        cp->methods = &extract_path_methods;
        cp->custom_private = list_make5(inputs, questions, makeInteger(batch), makeString(model.data),
                                        makeString(pstrdup(placement)));
        cp->custom_private = lappend(cp->custom_private, quals);
        out = lappend(out, cp);
    }
    return out;
}

/* Replace `rel`'s paths by SemExtract paths over them. False if the labels cannot be computed here. */
bool
trimem_wrap_extracts(PlannerInfo *root, RelOptInfo *rel, const char *placement)
{
    List *paths = trimem_extract_paths(root, rel, placement);
    double qsel = trimem_extract_qual_sel(root, rel);
    ListCell *lc;

    if (paths == NIL)
        return false;
    rel->pathlist = NIL;
    rel->partial_pathlist = NIL;
    foreach (lc, paths)
        add_path(rel, lfirst(lc));
    rel->rows = clamp_row_est(rel->rows * qsel);
    if (rel->cheapest_total_path != NULL)
        set_cheapest(rel);
    return true;
}

static Plan *
extract_plan_path(PlannerInfo *root, RelOptInfo *rel, CustomPath *best, List *tlist, List *clauses,
                  List *custom_plans)
{
    CustomScan *cs = makeNode(CustomScan);
    Plan *child = linitial(custom_plans);
    List *scan_tlist = NIL, *quals = list_nth(best->custom_private, 5);
    ListCell *lc;
    AttrNumber resno = 1;

    foreach (lc, child->targetlist)
    {
        TargetEntry *te = lfirst(lc);

        scan_tlist = lappend(scan_tlist, makeTargetEntry(copyObject(te->expr), resno++, te->resname, false));
    }
    cs->scan.plan.targetlist = tlist;
    /* label quals run on the emitted row, after this node computed its label */
    cs->scan.plan.qual = extract_actual_clauses(quals, false);
    cs->scan.scanrelid = 0;
    cs->custom_plans = custom_plans;
    cs->custom_scan_tlist = scan_tlist;
    cs->custom_exprs = copyObject(linitial(best->custom_private));
    cs->custom_private = list_make5(copyObject(lsecond(best->custom_private)),
                                    makeString(pstrdup(trimem_model)),
                                    makeString(pstrdup(trimem_prompt_version)),
                                    copyObject(lthird(best->custom_private)),
                                    copyObject(lfourth(best->custom_private)));
    cs->custom_private = lappend(cs->custom_private, copyObject(list_nth(best->custom_private, 4)));
    cs->custom_private = lappend(cs->custom_private, makeString(pstrdup(trimem_score_decision())));
    cs->flags = best->flags;
    cs->methods = &extract_scan_methods;
    pstate.quals_planned += list_length(quals);
    return &cs->scan.plan;
}

/* ---------------------------------------------------------------- execution */
typedef struct ExtractState
{
    CustomScanState css;
    PlanState *child;
    List *input_states; /* ExprState * per sem_extract call */
    List *questions;    /* String * */
    char *model, *pv, *costs, *placement, *decision;
    int batch;
    HTAB *labels;
    MemoryContext batch_cxt;
    MinimalTuple *rows;
    int nrows, pos;
    bool child_done;
    int64 extracted, hits, requests, batches, nulls, unlabeled;
    double wait_ms;
} ExtractState;

static Node *
extract_create_state(CustomScan *cs)
{
    ExtractState *s = palloc0(sizeof(ExtractState));

    NodeSetTag(s, T_CustomScanState);
    s->css.methods = &extract_exec_methods;
    return (Node *) s;
}

static void
extract_begin(CustomScanState *node, EState *estate, int eflags)
{
    ExtractState *s = (ExtractState *) node;
    CustomScan *cs = (CustomScan *) node->ss.ps.plan;
    HASHCTL ctl = {0};
    ListCell *lc;

    s->child = ExecInitNode(linitial(cs->custom_plans), estate, eflags);
    node->custom_ps = list_make1(s->child);
    foreach (lc, cs->custom_exprs)
        s->input_states = lappend(s->input_states, ExecInitExpr(lfirst(lc), &node->ss.ps));
    s->questions = linitial(cs->custom_private);
    s->model = strVal(lsecond(cs->custom_private));
    s->pv = strVal(lthird(cs->custom_private));
    s->batch = intVal(lfourth(cs->custom_private));
    s->costs = strVal(list_nth(cs->custom_private, 4));
    s->placement = strVal(list_nth(cs->custom_private, 5));
    s->decision = strVal(list_nth(cs->custom_private, 6));
    ctl.keysize = 65;
    ctl.entrysize = sizeof(LabelEntry);
    ctl.hcxt = estate->es_query_cxt;
    s->labels = hash_create("TriMEM SemExtract", 256, &ctl, HASH_ELEM | HASH_STRINGS | HASH_CONTEXT);
    s->batch_cxt = AllocSetContextCreate(estate->es_query_cxt, "TriMEM SemExtract batch", ALLOCSET_DEFAULT_SIZES);
    s->rows = MemoryContextAlloc(estate->es_query_cxt, sizeof(MinimalTuple) * s->batch);
    {
        MemoryContext old = MemoryContextSwitchTo(TopTransactionContext);
        LiveTable *t = palloc(sizeof(LiveTable));

        t->table = s->labels;
        t->model = s->model;
        t->pv = s->pv;
        live_tables = lappend(live_tables, t);
        MemoryContextSwitchTo(old);
    }
}

/*
 * Cached labels for `keys`. Read with the latest snapshot: labels the worker committed while this
 * query runs (after its snapshot) must be visible, at any isolation level. sem_cache rows are
 * deterministic per key, so a newer snapshot cannot change an answer, only reveal one.
 * label[i] is allocated in `cxt`; found[i] = false when the key has no label.
 */
static void
probe_labels(char **keys, int n, char **label, bool *found, MemoryContext cxt)
{
    Datum *elems, arg;

    for (int i = 0; i < n; i++)
        found[i] = false;
    if (n == 0)
        return;
    if (SPI_connect() != SPI_OK_CONNECT)
        elog(ERROR, "trimem: SPI_connect failed");
    if (extract_probe == NULL)
    {
        Oid t[1] = {TEXTARRAYOID};
        SPIPlanPtr plan = SPI_prepare("SELECT key, extracted FROM trimem.sem_cache "
                                      "WHERE key = ANY($1) AND extracted IS NOT NULL", 1, t);

        if (plan == NULL || SPI_keepplan(plan) != 0)
            elog(ERROR, "trimem: cannot prepare label probe");
        extract_probe = plan;
    }
    elems = palloc(sizeof(Datum) * n);
    for (int i = 0; i < n; i++)
        elems[i] = CStringGetTextDatum(keys[i]);
    arg = PointerGetDatum(construct_array_builtin(elems, n, TEXTOID));
    if (SPI_execute_snapshot(extract_probe, &arg, NULL, GetLatestSnapshot(), InvalidSnapshot, true, false, 0) !=
        SPI_OK_SELECT)
        elog(ERROR, "trimem: label probe failed");
    for (uint64 r = 0; r < SPI_processed; r++)
    {
        bool isnull;
        char *k = TextDatumGetCString(SPI_getbinval(SPI_tuptable->vals[r], SPI_tuptable->tupdesc, 1, &isnull));
        Datum v = SPI_getbinval(SPI_tuptable->vals[r], SPI_tuptable->tupdesc, 2, &isnull);

        if (isnull)
            continue;
        for (int i = 0; i < n; i++)
            if (!found[i] && strcmp(keys[i], k) == 0)
            {
                found[i] = true;
                label[i] = MemoryContextStrdup(cxt, TextDatumGetCString(v));
            }
    }
    SPI_finish();
}

static void
extract_fill(ExtractState *s)
{
    ExprContext *econtext = s->css.ss.ps.ps_ExprContext;
    MemoryContext table_cxt = s->css.ss.ps.state->es_query_cxt;
    int nq = list_length(s->questions);
    int cap = s->batch * Max(nq, 1);
    char **keys, **inputs, **questions;
    int nk = 0;

    MemoryContextReset(s->batch_cxt);
    s->nrows = s->pos = 0;
    keys = MemoryContextAllocZero(s->batch_cxt, sizeof(char *) * cap);
    inputs = MemoryContextAllocZero(s->batch_cxt, sizeof(char *) * cap);
    questions = MemoryContextAllocZero(s->batch_cxt, sizeof(char *) * cap);
    while (s->nrows < s->batch)
    {
        TupleTableSlot *cslot = ExecProcNode(s->child);
        ListCell *le, *lq;

        if (TupIsNull(cslot))
        {
            s->child_done = true;
            break;
        }
        econtext->ecxt_scantuple = cslot;
        forboth (le, s->input_states, lq, s->questions)
        {
            bool isnull;
            Datum d = ExecEvalExprSwitchContext(lfirst(le), econtext, &isnull);
            MemoryContext old;
            char key[65], *in;
            bool dup = false;

            if (isnull)
            {
                s->nulls++;
                continue; /* STRICT: sem_extract(q, NULL) is NULL and never evaluated */
            }
            old = MemoryContextSwitchTo(s->batch_cxt);
            in = TextDatumGetCString(d);
            trimem_key_hex(TRIMEM_OP_EXTRACT, s->model, s->pv, strVal(lfirst(lq)), in, key);
            MemoryContextSwitchTo(old);
            if (hash_search(s->labels, key, HASH_FIND, NULL))
                continue; /* extracted earlier in this query */
            for (int k = 0; k < nk && !dup; k++)
                dup = strcmp(keys[k], key) == 0;
            if (!dup)
            {
                keys[nk] = MemoryContextStrdup(s->batch_cxt, key);
                inputs[nk] = in;
                questions[nk] = strVal(lfirst(lq));
                nk++;
            }
        }
        {
            MemoryContext old = MemoryContextSwitchTo(s->batch_cxt);

            s->rows[s->nrows++] = ExecCopySlotMinimalTuple(cslot);
            MemoryContextSwitchTo(old);
        }
        ResetExprContext(econtext);
    }
    if (s->nrows == 0)
        return;
    s->batches++;
    if (nk > 0)
    {
        char **label = MemoryContextAllocZero(s->batch_cxt, sizeof(char *) * nk);
        bool *found = MemoryContextAlloc(s->batch_cxt, sizeof(bool) * nk);
        TrimemRequest *reqs = MemoryContextAllocZero(s->batch_cxt, sizeof(TrimemRequest) * nk);
        int *req_k = MemoryContextAlloc(s->batch_cxt, sizeof(int) * nk);
        int nreq = 0;

        probe_labels(keys, nk, label, found, table_cxt);
        for (int k = 0; k < nk; k++)
        {
            if (found[k])
            {
                s->hits++;
                continue;
            }
            reqs[nreq].predicate = questions[k];
            reqs[nreq].input = inputs[k];
            reqs[nreq].op = TRIMEM_OPC_EXTRACT;
            req_k[nreq++] = k;
        }
        if (nreq > 0)
        {
            char **rkeys = MemoryContextAlloc(s->batch_cxt, sizeof(char *) * nreq);
            char **rlabel = MemoryContextAllocZero(s->batch_cxt, sizeof(char *) * nreq);
            bool *rfound = MemoryContextAlloc(s->batch_cxt, sizeof(bool) * nreq);

            trimem_mailbox_run(s->model, s->pv, reqs, nreq, &s->wait_ms);
            /* the worker committed the labels before completing: read them back */
            for (int r = 0; r < nreq; r++)
                rkeys[r] = keys[req_k[r]];
            probe_labels(rkeys, nreq, rlabel, rfound, table_cxt);
            for (int r = 0; r < nreq; r++)
            {
                label[req_k[r]] = rfound[r] ? rlabel[r] : NULL;
                if (!rfound[r])
                    s->unlabeled++; /* no parsable answer: NULL for this query, not cached */
            }
            s->requests += nreq;
        }
        for (int k = 0; k < nk; k++)
        {
            LabelEntry *e = hash_search(s->labels, keys[k], HASH_ENTER, NULL);

            e->label = label[k];
        }
        s->extracted += nk;
    }
}

static TupleTableSlot *
extract_next(ScanState *node)
{
    ExtractState *s = (ExtractState *) node;
    TupleTableSlot *slot = node->ss_ScanTupleSlot;

    for (;;)
    {
        if (s->pos < s->nrows)
        {
            ExecForceStoreMinimalTuple(s->rows[s->pos++], slot, false);
            trimem_extract_set_active(s->labels, s->model, s->pv);
            return slot;
        }
        if (s->child_done)
            return ExecClearTuple(slot);
        extract_fill(s);
        if (s->nrows == 0)
            return ExecClearTuple(slot);
    }
}

static bool
extract_recheck(ScanState *node, TupleTableSlot *slot)
{
    return true;
}

static TupleTableSlot *
extract_exec(CustomScanState *node)
{
    return ExecScan(&node->ss, extract_next, extract_recheck);
}

static void
extract_end(CustomScanState *node)
{
    ExtractState *s = (ExtractState *) node;

    ListCell *lc;

    trimem_extract_set_active(NULL, "", "");
    foreach (lc, live_tables)
        if (((LiveTable *) lfirst(lc))->table == s->labels)
        {
            live_tables = foreach_delete_current(live_tables, lc);
            break;
        }
    ExecEndNode(s->child);
}

static void
extract_rescan(CustomScanState *node)
{
    ExtractState *s = (ExtractState *) node;

    s->nrows = s->pos = 0;
    s->child_done = false;
    MemoryContextReset(s->batch_cxt);
    if (s->child->chgParam == NULL)
        ExecReScan(s->child);
    ExecScanReScan(&node->ss);
}

static void
extract_explain(CustomScanState *node, List *ancestors, ExplainState *es)
{
    ExtractState *s = (ExtractState *) node;
    CustomScan *cs = (CustomScan *) node->ss.ps.plan;
    List *ctx = set_deparse_context_plan(es->deparse_cxt, node->ss.ps.plan, ancestors);
    ListCell *le, *lq;

    forboth (le, cs->custom_exprs, lq, s->questions)
    {
        ExplainPropertyText("Extract Question", strVal(lfirst(lq)), es);
        ExplainPropertyText("Extract Input", deparse_expression(lfirst(le), ctx, es->verbose, false), es);
    }
    ExplainPropertyText("Model", s->model, es);
    ExplainPropertyText("Semantic Cost Model", s->costs, es);
    ExplainPropertyText("Placement", s->placement, es);
    if (s->decision[0])
        ExplainPropertyText("Placement Decision", s->decision, es);
    ExplainPropertyInteger("Batch Size", NULL, s->batch, es);
    if (es->analyze)
    {
        ExplainPropertyInteger("Inputs Extracted", NULL, s->extracted, es);
        ExplainPropertyInteger("Cache Hits", NULL, s->hits, es);
        ExplainPropertyInteger("Worker Requests", NULL, s->requests, es);
        ExplainPropertyInteger("NULL Inputs", NULL, s->nulls, es);
        ExplainPropertyInteger("Unlabeled", NULL, s->unlabeled, es);
        ExplainPropertyInteger("Batches", NULL, s->batches, es);
        ExplainPropertyFloat("LLM Wait", "ms", s->wait_ms, 1, es);
    }
}

void
trimem_extract_init(void)
{
    extract_path_methods.CustomName = "TriMEM SemExtract";
    extract_path_methods.PlanCustomPath = extract_plan_path;
    extract_scan_methods.CustomName = "TriMEM SemExtract";
    extract_scan_methods.CreateCustomScanState = extract_create_state;
    extract_exec_methods.CustomName = "TriMEM SemExtract";
    extract_exec_methods.BeginCustomScan = extract_begin;
    extract_exec_methods.ExecCustomScan = extract_exec;
    extract_exec_methods.EndCustomScan = extract_end;
    extract_exec_methods.ReScanCustomScan = extract_rescan;
    extract_exec_methods.ExplainCustomScan = extract_explain;
    RegisterCustomScanMethods(&extract_scan_methods);
}
