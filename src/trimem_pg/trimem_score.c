/*
 * trimem_score.c — "Custom Scan (TriMEM SemScore)": batched LLM relevance scores for ranking.
 *
 *   SELECT ... ORDER BY gain * (0.5 + coalesce(trimem.sem_score('<predicate>', d.changes), 0.5)) DESC
 *
 * trimem.sem_score(predicate, input) -> float8 in [0, 1] is a soft semantic operator: it never
 * removes a row, it weights it. The planner wraps the relation (single-relation query) or the
 * final scan/join relation in a SemScore node that computes every score the query's target list
 * needs, in bounded batches (cache probe by key, misses through the worker mailbox). The node
 * keeps the scores of the query in a table and registers it before emitting each row; the
 * projection that evaluates trimem.sem_score — the node's own or the next one up, e.g. below a
 * Sort — only looks the value up, so the LLM is never called per row from an expression.
 */
#include "postgres.h"

#include <math.h>

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
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/hsearch.h"
#include "utils/memutils.h"
#include "utils/ruleutils.h"
#include "utils/selfuncs.h"

#include "trimem.h"

typedef struct ScoreEntry
{
    char key[65];
    double score;
} ScoreEntry;

static CustomPathMethods score_path_methods;
static CustomScanMethods score_scan_methods;
static CustomExecMethods score_exec_methods;
static SPIPlanPtr score_probe = NULL;

bool
trimem_score_lookup(void *table, const char *key, double *score)
{
    ScoreEntry *e = hash_search((HTAB *) table, key, HASH_FIND, NULL);

    if (!e)
        return false;
    *score = e->score;
    return true;
}

/* ---------------------------------------------------------------- planning */
typedef struct ScoreCollect
{
    Oid fid;
    List *calls; /* distinct FuncExpr * */
} ScoreCollect;

static bool
collect_scores(Node *node, ScoreCollect *c)
{
    if (node == NULL)
        return false;
    if (IsA(node, FuncExpr) && ((FuncExpr *) node)->funcid == c->fid)
    {
        if (!list_member(c->calls, node))
            c->calls = lappend(c->calls, node);
        return false;
    }
    return expression_tree_walker(node, collect_scores, c);
}

static List *
score_calls(PlannerInfo *root)
{
    ScoreCollect c = {trimem_sem_score_oid(), NIL};

    if (!OidIsValid(c.fid))
        return NIL;
    collect_scores((Node *) root->processed_tlist, &c);
    return c.calls;
}

/* The single base relation whose columns every sem_score input reads, or NULL. */
RelOptInfo *
trimem_score_base_rel(PlannerInfo *root, List *initial_rels)
{
    List *calls = score_calls(root);
    Relids rels = NULL;
    ListCell *lc;
    int relid;

    if (calls == NIL)
        return NULL;
    foreach (lc, calls)
        rels = bms_add_members(rels, pull_varnos(root, lsecond(((FuncExpr *) lfirst(lc))->args)));
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
 * SemScore paths over every path of `rel` (rel is not modified). `placement` labels the node in
 * EXPLAIN: "base" (scores the relation's rows before the joins, possibly per nested-loop rescan),
 * "top (above join)" or "single relation".
 */
List *
trimem_score_paths(PlannerInfo *root, RelOptInfo *rel, const char *placement)
{
    List *calls = score_calls(root);
    List *inputs = NIL, *preds = NIL, *out = NIL;
    ListCell *lc;
    double c_per_row_distinct = 0;
    int batch = trimem_max_batch;
    StringInfoData model;

    if (calls == NIL || rel->pathlist == NIL)
        return NIL;
    initStringInfo(&model);
    foreach (lc, calls)
    {
        FuncExpr *fn = lfirst(lc);
        Node *pred = linitial(fn->args);
        Expr *input = lsecond(fn->args);
        TrimemPredStats st;

        if (!IsA(pred, Const) || ((Const *) pred)->constisnull)
            ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                            errmsg("trimem.sem_score predicate must be a non-NULL constant")));
        if (!bms_is_subset(pull_varnos(root, (Node *) input), rel->relids))
            return NIL; /* computed elsewhere; fails closed at execution */
        preds = lappend(preds, makeString(TextDatumGetCString(((Const *) pred)->constvalue)));
        inputs = lappend(inputs, copyObject(input));
        trimem_lookup_stats(trimem_model, trimem_prompt_version, strVal(llast(preds)), &st);
        c_per_row_distinct += st.c_eff;
        appendStringInfo(&model, "%shit=%.2f c_eff=%.0f", model.len ? "; " : "", st.hit, st.c_eff);
    }
    foreach (lc, rel->pathlist)
    {
        Path *child = lfirst(lc);
        CustomPath *cp = makeNode(CustomPath);
        double rows = child->rows;
        double distinct = 0;
        ListCell *li;

        foreach (li, inputs)
            distinct += Min(rows, estimate_num_groups(root, list_make1(lfirst(li)), rows, NULL, NULL));
        cp->path.pathtype = T_CustomScan;
        cp->path.parent = rel;
        cp->path.pathtarget = child->pathtarget;
        cp->path.param_info = child->param_info;
        cp->path.parallel_safe = false;
        cp->path.rows = rows;
        cp->path.startup_cost = child->startup_cost + Min((double) batch, rows) * c_per_row_distinct /
                                    Max(list_length(inputs), 1) + trimem_llm_cost_per_batch;
        cp->path.total_cost = child->total_cost + distinct * c_per_row_distinct / Max(list_length(inputs), 1) +
                              ceil(rows / batch) * trimem_llm_cost_per_batch + rows * cpu_tuple_cost;
        cp->path.pathkeys = child->pathkeys;
        cp->flags = CUSTOMPATH_SUPPORT_PROJECTION;
        cp->custom_paths = list_make1(child);
        cp->methods = &score_path_methods;
        cp->custom_private = list_make5(inputs, preds, makeInteger(batch), makeString(pstrdup(model.data)),
                                        makeString(pstrdup(placement)));
        out = lappend(out, cp);
    }
    return out;
}

void
trimem_wrap_scores(PlannerInfo *root, RelOptInfo *rel)
{
    bool single = rel->reloptkind == RELOPT_BASEREL;
    List *paths = trimem_score_paths(root, rel, single ? "single relation" : "top (above join)");
    ListCell *lc;

    if (paths == NIL)
        return;
    rel->pathlist = NIL;
    rel->partial_pathlist = NIL;
    foreach (lc, paths)
        add_path(rel, lfirst(lc));
    if (rel->cheapest_total_path != NULL)
        set_cheapest(rel);
}

static Plan *
score_plan_path(PlannerInfo *root, RelOptInfo *rel, CustomPath *best, List *tlist, List *clauses,
                List *custom_plans)
{
    CustomScan *cs = makeNode(CustomScan);
    Plan *child = linitial(custom_plans);
    List *scan_tlist = NIL;
    ListCell *lc;
    AttrNumber resno = 1;

    foreach (lc, child->targetlist)
    {
        TargetEntry *te = lfirst(lc);

        scan_tlist = lappend(scan_tlist, makeTargetEntry(copyObject(te->expr), resno++, te->resname, false));
    }
    cs->scan.plan.targetlist = tlist;
    cs->scan.plan.qual = NIL;
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
    cs->methods = &score_scan_methods;
    return &cs->scan.plan;
}

/* ---------------------------------------------------------------- execution */
typedef struct ScoreState
{
    CustomScanState css;
    PlanState *child;
    List *input_states; /* ExprState * per sem_score call */
    List *preds;        /* String * */
    char *model, *pv, *costs, *placement, *decision;
    int batch;
    HTAB *scores;
    MemoryContext batch_cxt;
    MinimalTuple *rows;
    int nrows, pos;
    bool child_done;
    int64 scored, hits, requests, batches, nulls;
    double wait_ms;
} ScoreState;

static Node *
score_create_state(CustomScan *cs)
{
    ScoreState *s = palloc0(sizeof(ScoreState));

    NodeSetTag(s, T_CustomScanState);
    s->css.methods = &score_exec_methods;
    return (Node *) s;
}

static void
score_begin(CustomScanState *node, EState *estate, int eflags)
{
    ScoreState *s = (ScoreState *) node;
    CustomScan *cs = (CustomScan *) node->ss.ps.plan;
    HASHCTL ctl = {0};
    ListCell *lc;

    s->child = ExecInitNode(linitial(cs->custom_plans), estate, eflags);
    node->custom_ps = list_make1(s->child);
    foreach (lc, cs->custom_exprs)
        s->input_states = lappend(s->input_states, ExecInitExpr(lfirst(lc), &node->ss.ps));
    s->preds = linitial(cs->custom_private);
    s->model = strVal(lsecond(cs->custom_private));
    s->pv = strVal(lthird(cs->custom_private));
    s->batch = intVal(lfourth(cs->custom_private));
    s->costs = strVal(list_nth(cs->custom_private, 4));
    s->placement = strVal(list_nth(cs->custom_private, 5));
    s->decision = strVal(list_nth(cs->custom_private, 6));
    ctl.keysize = 65;
    ctl.entrysize = sizeof(ScoreEntry);
    ctl.hcxt = estate->es_query_cxt;
    s->scores = hash_create("TriMEM SemScore", 256, &ctl, HASH_ELEM | HASH_STRINGS | HASH_CONTEXT);
    s->batch_cxt = AllocSetContextCreate(estate->es_query_cxt, "TriMEM SemScore batch", ALLOCSET_DEFAULT_SIZES);
    s->rows = MemoryContextAlloc(estate->es_query_cxt, sizeof(MinimalTuple) * s->batch);
}

static void
probe_scores(char **keys, int n, double *score, bool *found)
{
    Datum *elems, arg;

    for (int i = 0; i < n; i++)
        found[i] = false;
    if (n == 0)
        return;
    if (SPI_connect() != SPI_OK_CONNECT)
        elog(ERROR, "trimem: SPI_connect failed");
    if (score_probe == NULL)
    {
        Oid t[1] = {TEXTARRAYOID};
        SPIPlanPtr plan = SPI_prepare("SELECT key, score FROM trimem.sem_cache "
                                      "WHERE key = ANY($1) AND score IS NOT NULL", 1, t);

        if (plan == NULL || SPI_keepplan(plan) != 0)
            elog(ERROR, "trimem: cannot prepare score probe");
        score_probe = plan;
    }
    elems = palloc(sizeof(Datum) * n);
    for (int i = 0; i < n; i++)
        elems[i] = CStringGetTextDatum(keys[i]);
    arg = PointerGetDatum(construct_array_builtin(elems, n, TEXTOID));
    if (SPI_execute_plan(score_probe, &arg, NULL, true, 0) != SPI_OK_SELECT)
        elog(ERROR, "trimem: score probe failed");
    for (uint64 r = 0; r < SPI_processed; r++)
    {
        bool isnull;
        char *k = TextDatumGetCString(SPI_getbinval(SPI_tuptable->vals[r], SPI_tuptable->tupdesc, 1, &isnull));
        double v = DatumGetFloat8(SPI_getbinval(SPI_tuptable->vals[r], SPI_tuptable->tupdesc, 2, &isnull));

        for (int i = 0; i < n; i++)
            if (!found[i] && strcmp(keys[i], k) == 0)
            {
                found[i] = true;
                score[i] = v;
            }
    }
    SPI_finish();
}

static void
score_fill(ScoreState *s)
{
    ExprContext *econtext = s->css.ss.ps.ps_ExprContext;
    int npred = list_length(s->preds);
    int cap = s->batch * Max(npred, 1);
    char **keys, **inputs, **preds;
    int nk = 0;

    MemoryContextReset(s->batch_cxt);
    s->nrows = s->pos = 0;
    keys = MemoryContextAllocZero(s->batch_cxt, sizeof(char *) * cap);
    inputs = MemoryContextAllocZero(s->batch_cxt, sizeof(char *) * cap);
    preds = MemoryContextAllocZero(s->batch_cxt, sizeof(char *) * cap);
    while (s->nrows < s->batch)
    {
        TupleTableSlot *cslot = ExecProcNode(s->child);
        ListCell *le, *lp;

        if (TupIsNull(cslot))
        {
            s->child_done = true;
            break;
        }
        econtext->ecxt_scantuple = cslot;
        forboth (le, s->input_states, lp, s->preds)
        {
            bool isnull;
            Datum d = ExecEvalExprSwitchContext(lfirst(le), econtext, &isnull);
            MemoryContext old;
            char key[65], *in;
            bool dup = false;

            if (isnull)
            {
                s->nulls++;
                continue; /* STRICT: sem_score(NULL) is NULL and never evaluated */
            }
            old = MemoryContextSwitchTo(s->batch_cxt);
            in = TextDatumGetCString(d);
            trimem_key_hex(TRIMEM_OP_SCORE, s->model, s->pv, strVal(lfirst(lp)), in, key);
            MemoryContextSwitchTo(old);
            if (hash_search(s->scores, key, HASH_FIND, NULL))
                continue; /* scored earlier in this query */
            for (int k = 0; k < nk && !dup; k++)
                dup = strcmp(keys[k], key) == 0;
            if (!dup)
            {
                keys[nk] = MemoryContextStrdup(s->batch_cxt, key);
                inputs[nk] = in;
                preds[nk] = strVal(lfirst(lp));
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
        double *score = MemoryContextAlloc(s->batch_cxt, sizeof(double) * nk);
        bool *found = MemoryContextAlloc(s->batch_cxt, sizeof(bool) * nk);
        TrimemRequest *reqs = MemoryContextAllocZero(s->batch_cxt, sizeof(TrimemRequest) * nk);
        int *req_k = MemoryContextAlloc(s->batch_cxt, sizeof(int) * nk);
        int nreq = 0;

        probe_scores(keys, nk, score, found);
        for (int k = 0; k < nk; k++)
        {
            if (found[k])
            {
                s->hits++;
                continue;
            }
            reqs[nreq].predicate = preds[k];
            reqs[nreq].input = inputs[k];
            reqs[nreq].op = TRIMEM_OPC_SCORE;
            req_k[nreq++] = k;
        }
        if (nreq > 0)
        {
            trimem_mailbox_run(s->model, s->pv, reqs, nreq, &s->wait_ms);
            for (int r = 0; r < nreq; r++)
                score[req_k[r]] = reqs[r].score;
            s->requests += nreq;
        }
        for (int k = 0; k < nk; k++)
        {
            ScoreEntry *e = hash_search(s->scores, keys[k], HASH_ENTER, NULL);

            e->score = score[k];
        }
        s->scored += nk;
    }
}

static TupleTableSlot *
score_next(ScanState *node)
{
    ScoreState *s = (ScoreState *) node;
    TupleTableSlot *slot = node->ss_ScanTupleSlot;

    for (;;)
    {
        if (s->pos < s->nrows)
        {
            ExecForceStoreMinimalTuple(s->rows[s->pos++], slot, false);
            trimem_score_set_active(s->scores, s->model, s->pv);
            return slot;
        }
        if (s->child_done)
            return ExecClearTuple(slot);
        score_fill(s);
        if (s->nrows == 0)
            return ExecClearTuple(slot);
    }
}

static bool
score_recheck(ScanState *node, TupleTableSlot *slot)
{
    return true;
}

static TupleTableSlot *
score_exec(CustomScanState *node)
{
    return ExecScan(&node->ss, score_next, score_recheck);
}

static void
score_end(CustomScanState *node)
{
    ScoreState *s = (ScoreState *) node;

    /* ExecutorEnd: every projection above has run; the table dies with the query context */
    trimem_score_set_active(NULL, "", "");
    ExecEndNode(s->child);
}

static void
score_rescan(CustomScanState *node)
{
    ScoreState *s = (ScoreState *) node;

    s->nrows = s->pos = 0;
    s->child_done = false;
    MemoryContextReset(s->batch_cxt);
    if (s->child->chgParam == NULL)
        ExecReScan(s->child);
    ExecScanReScan(&node->ss);
}

static void
score_explain(CustomScanState *node, List *ancestors, ExplainState *es)
{
    ScoreState *s = (ScoreState *) node;
    CustomScan *cs = (CustomScan *) node->ss.ps.plan;
    List *ctx = set_deparse_context_plan(es->deparse_cxt, node->ss.ps.plan, ancestors);
    ListCell *le, *lp;

    forboth (le, cs->custom_exprs, lp, s->preds)
    {
        ExplainPropertyText("Score Predicate", strVal(lfirst(lp)), es);
        ExplainPropertyText("Score Input", deparse_expression(lfirst(le), ctx, es->verbose, false), es);
    }
    ExplainPropertyText("Model", s->model, es);
    ExplainPropertyText("Semantic Cost Model", s->costs, es);
    ExplainPropertyText("Placement", s->placement, es);
    if (s->decision[0])
        ExplainPropertyText("Placement Decision", s->decision, es);
    ExplainPropertyInteger("Batch Size", NULL, s->batch, es);
    if (es->analyze)
    {
        ExplainPropertyInteger("Inputs Scored", NULL, s->scored, es);
        ExplainPropertyInteger("Cache Hits", NULL, s->hits, es);
        ExplainPropertyInteger("Worker Requests", NULL, s->requests, es);
        ExplainPropertyInteger("NULL Inputs", NULL, s->nulls, es);
        ExplainPropertyInteger("Batches", NULL, s->batches, es);
        ExplainPropertyFloat("LLM Wait", "ms", s->wait_ms, 1, es);
    }
}

void
trimem_score_init(void)
{
    score_path_methods.CustomName = "TriMEM SemScore";
    score_path_methods.PlanCustomPath = score_plan_path;
    score_scan_methods.CustomName = "TriMEM SemScore";
    score_scan_methods.CreateCustomScanState = score_create_state;
    score_exec_methods.CustomName = "TriMEM SemScore";
    score_exec_methods.BeginCustomScan = score_begin;
    score_exec_methods.ExecCustomScan = score_exec;
    score_exec_methods.EndCustomScan = score_end;
    score_exec_methods.ReScanCustomScan = score_rescan;
    score_exec_methods.ExplainCustomScan = score_explain;
    RegisterCustomScanMethods(&score_scan_methods);
}
