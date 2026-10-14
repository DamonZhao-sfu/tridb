/*
 * trimem_path.c — cost-based placement of trimem.sem_filter as "Custom Scan (TriMEM SemFilter)".
 *
 * set_rel_pathlist_hook (chained after VSGT's), for a base relation whose restriction list holds
 * top-level trimem.sem_filter(const predicate, input) conjuncts:
 *   1. remove them from baserestrictinfo / indrestrictinfo (the input scan never evaluates them);
 *   2. order them by descending rank (1 - sel) / c_eff — the tridb_place_sem rule (ADR-0026);
 *   3. build the BASE placement: every path wrapped in a SemFilter chain.
 * Single-relation query: that is the plan. Multi-relation query: the relation also keeps its
 * unwrapped paths, the semantic inputs are marked as needed above the scan, and join_search_hook
 * enumerates both placements and keeps the cheaper:
 *   (a) BASE — SemFilter on the relation (inside a nested loop: rejudged per rescan);
 *   (b) TOP  — join search over the unfiltered relation, SemFilter above the final scan/join rel
 *              (batches over joined rows, after every cheap V/G/R operator).
 * Each placement is a full standard join search in GEQO-style saved/restored join_rel_list; the
 * winner is re-run for real. Cost model: docs/trimem_semscan_plan_v0.1.0.md §3.
 */
#include "postgres.h"

#include <math.h>

#include "catalog/pg_type.h"
#include "lib/stringinfo.h"
#include "nodes/extensible.h"
#include "nodes/makefuncs.h"
#include "nodes/nodeFuncs.h"
#include "optimizer/cost.h"
#include "optimizer/geqo.h"
#include "optimizer/optimizer.h"
#include "optimizer/pathnode.h"
#include "optimizer/paths.h"
#include "optimizer/planner.h"
#include "optimizer/restrictinfo.h"
#include "optimizer/tlist.h"
#include "parser/parsetree.h"
#include "utils/builtins.h"
#include "utils/lsyscache.h"
#include "utils/selfuncs.h"

#include "trimem.h"

typedef struct SemQual
{
    RestrictInfo *ri;
    Expr *input;
    char *predicate;
    TrimemPredStats st;
    double rank;
    int ord;
    int unknown; /* NULL input: -1 drop (2-arg form), 0 drop, 1 keep (3-arg unknown_as) */
} SemQual;

/* A base relation whose semantic quals were lifted in a multi-relation query. */
typedef struct LiftedRel
{
    PlannerInfo *root;
    RelOptInfo *rel;
    List *sems;          /* SemQual *, rank order */
    double sel_all;
    List *plain_paths;   /* unfiltered (TOP placement) */
    List *wrapped_paths; /* SemFilter chains (BASE placement) */
    double rows_plain;
    double rows_wrapped;
    char *alias;
} LiftedRel;

static set_rel_pathlist_hook_type prev_path_hook = NULL;
static join_search_hook_type prev_join_search_hook = NULL;
static planner_hook_type prev_planner_hook = NULL;
static CustomPathMethods path_methods;

/*
 * Safety net: semantic quals are removed from the restriction lists when lifted. If something
 * after us replaced the paths (e.g. tjs_pg loaded after trimem_pg), a predicate would silently
 * vanish. Count lifted vs planned SemFilter nodes per planner call and fail closed.
 */
static int lifted_count = 0;
static int planned_count = 0;
static List *lifted_rels = NIL;
static char *last_placement = NULL;

/* ---------------------------------------------------------------- helpers */
/* context: InvalidOid-terminated array of function OIDs */
static bool
contains_func_walker(Node *node, void *context)
{
    if (node == NULL)
        return false;
    if (IsA(node, FuncExpr))
        for (Oid *o = (Oid *) context; OidIsValid(*o); o++)
            if (((FuncExpr *) node)->funcid == *o)
                return true;
    return expression_tree_walker(node, contains_func_walker, context);
}

static int
cmp_rank(const ListCell *a, const ListCell *b)
{
    const SemQual *x = lfirst(a), *y = lfirst(b);

    if (x->rank > y->rank)
        return -1;
    if (x->rank < y->rank)
        return 1;
    return x->ord - y->ord; /* stable: original conjunct order breaks ties */
}

static int
batch_for(double k, double sel)
{
    double b = ceil(k / Max(sel, 1e-6));

    return (int) Min(Max(b, (double) trimem_min_batch), (double) trimem_max_batch);
}

static void
check_joininfo(RelOptInfo *rel, Oid *fids)
{
    ListCell *lc;

    foreach (lc, rel->joininfo)
        if (contains_func_walker((Node *) ((RestrictInfo *) lfirst(lc))->clause, fids))
            ereport(ERROR, (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                            errmsg("trimem.sem_filter over more than one relation is not supported (v1)"),
                            errhint("Apply the semantic predicate to a single relation's columns.")));
}

static List *
collect_sem_quals(RelOptInfo *rel, Oid *fids, List **others)
{
    List *sems = NIL;
    ListCell *lc;
    int ord = 0;

    *others = NIL;
    foreach (lc, rel->baserestrictinfo)
    {
        RestrictInfo *ri = lfirst(lc);
        Node *clause = (Node *) ri->clause;
        SemQual *q;
        Node *pred;

        if (!contains_func_walker(clause, fids))
        {
            *others = lappend(*others, ri);
            continue;
        }
        if (!IsA(clause, FuncExpr) ||
            (((FuncExpr *) clause)->funcid != fids[0] && ((FuncExpr *) clause)->funcid != fids[1]))
            ereport(ERROR, (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                            errmsg("trimem.sem_filter must be a top-level AND conjunct of WHERE"),
                            errhint("OR / NOT / nested uses cannot be planned as a SemFilter node.")));
        pred = linitial(((FuncExpr *) clause)->args);
        if (!IsA(pred, Const) || ((Const *) pred)->constisnull)
            ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                            errmsg("trimem.sem_filter predicate must be a non-NULL constant")));
        q = palloc0(sizeof(SemQual));
        q->ri = ri;
        q->input = lsecond(((FuncExpr *) clause)->args);
        q->predicate = TextDatumGetCString(((Const *) pred)->constvalue);
        q->unknown = -1;
        if (list_length(((FuncExpr *) clause)->args) == 3)
        {
            Node *u = lthird(((FuncExpr *) clause)->args);

            if (!IsA(u, Const) || ((Const *) u)->constisnull)
                ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                                errmsg("trimem.sem_filter unknown_as must be a non-NULL boolean constant")));
            q->unknown = DatumGetBool(((Const *) u)->constvalue) ? 1 : 0;
        }
        trimem_lookup_stats(trimem_model, trimem_prompt_version, q->predicate, &q->st);
        if (q->st.n_cached > 0)
        {
            /*
             * Cache coverage: the share of this relation's candidate inputs whose verdict is
             * already cached. Approximation: cached verdicts for the predicate over the
             * relation's pre-filter rows (assumes the cached inputs are drawn from these rows).
             */
            double rows_pre = Max(1.0, rel->rows / q->st.sel);
            double coverage = Min(1.0, q->st.n_cached / rows_pre);

            q->st.hit = Max(q->st.hit, coverage);
            q->st.c_eff = trimem_llm_cost_per_eval * (1.0 - q->st.hit) +
                          trimem_cache_cost_per_eval * q->st.hit;
        }
        q->rank = q->st.c_eff > 0 ? (1.0 - q->st.sel) / q->st.c_eff : INFINITY;
        q->ord = ord++;
        sems = lappend(sems, q);
    }
    return sems;
}

static List *
input_vars(List *sems)
{
    List *vars = NIL;
    ListCell *lc;

    foreach (lc, sems)
        vars = list_concat_unique(vars, pull_var_clause((Node *) ((SemQual *) lfirst(lc))->input,
                                                        PVC_RECURSE_AGGREGATES | PVC_RECURSE_WINDOWFUNCS |
                                                            PVC_INCLUDE_PLACEHOLDERS));
    return vars;
}

static PathTarget *
widen_target(PlannerInfo *root, PathTarget *base, List *vars)
{
    PathTarget *t = copy_pathtarget(base);
    ListCell *lv;

    foreach (lv, vars)
        if (!list_member(t->exprs, lfirst(lv)))
            add_column_to_pathtarget(t, lfirst(lv), 0);
    return set_pathtarget_cost_width(root, t);
}

/*
 * LIMIT feedback: the node may size its batch from the query's LIMIT only if nothing between it
 * and the Limit needs the whole input (no grouping/aggregation/window/distinct/set op, and any
 * ORDER BY already satisfied by the input's order).
 */
static double
limit_hint(PlannerInfo *root, Path *child, bool is_final)
{
    Query *parse = root->parse;

    if (!is_final || root->limit_tuples <= 0)
        return (double) trimem_max_batch;
    if (parse->groupClause || parse->groupingSets || parse->hasAggs || parse->hasWindowFuncs ||
        parse->distinctClause || parse->setOperations || parse->hasTargetSRFs)
        return (double) trimem_max_batch;
    if (root->query_pathkeys != NIL && !pathkeys_contained_in(root->query_pathkeys, child->pathkeys))
        return (double) trimem_max_batch;
    return root->limit_tuples;
}

static List *
private_of(SemQual *q, int batch, double k_hint, const char *placement)
{
    char buf[64];
    List *l = NIL;

    l = lappend(l, makeString(pstrdup(q->predicate)));
    l = lappend(l, makeString(pstrdup(trimem_model)));
    l = lappend(l, makeString(pstrdup(trimem_prompt_version)));
#define PUSH_F(v) do { snprintf(buf, sizeof(buf), "%.17g", (double) (v)); l = lappend(l, makeString(pstrdup(buf))); } while (0)
    PUSH_F(q->st.sel);
    PUSH_F(q->st.hit);
    PUSH_F(q->st.c_eff);
    PUSH_F(q->rank);
    PUSH_F(q->st.n_eval);
    PUSH_F(k_hint);
    l = lappend(l, makeInteger(batch));
    l = lappend(l, makeInteger(q->st.from_stats ? 1 : 0));
    l = lappend(l, makeString(pstrdup(placement)));
    PUSH_F(q->st.n_cached);
#undef PUSH_F
    l = lappend(l, makeInteger(q->unknown));
    return l;
}

/*
 * Wrap `child` (rows = pre-filter) in one SemFilter per predicate, bottom = highest rank.
 * Intermediate nodes keep `wide`; the top node emits `top`.
 */
static Path *
wrap_chain(PlannerInfo *root, RelOptInfo *rel, Path *child, List *sems, double sel_all,
           PathTarget *wide, PathTarget *top, bool is_final, const char *placement)
{
    double rows = child->rows;
    double above = sel_all; /* product of selectivities at and above the current node */
    double k_hint = limit_hint(root, child, is_final);
    ListCell *lc;

    foreach (lc, sems)
    {
        SemQual *q = lfirst(lc);
        CustomPath *cp = makeNode(CustomPath);
        /* rows this node must emit for the LIMIT above it to be met */
        double k_node = k_hint / Max(above / q->st.sel, 1e-6);
        int batch = batch_for(k_node, q->st.sel);

        cp->path.pathtype = T_CustomScan;
        cp->path.parent = rel;
        cp->path.pathtarget = lnext(sems, lc) == NULL ? top : wide;
        cp->path.param_info = child->param_info;
        cp->path.parallel_aware = false;
        cp->path.parallel_safe = false;
        cp->path.parallel_workers = 0;
        cp->path.rows = clamp_row_est(rows * q->st.sel);
        {
            /*
             * LLM work is per DISTINCT input: duplicates in a batch share one key and later
             * duplicates are answered by the worker from the committed cache. Round trips are
             * per batch with at least one miss; every row pays a tuple cost.
             */
            double b = Min((double) batch, rows);
            double p_trip = 1.0 - pow(q->st.hit, b);
            double trips = ceil(rows / Max(b, 1.0)) * p_trip;
            double distinct = Min(rows, estimate_num_groups(root, list_make1(q->input), rows, NULL, NULL));

            cp->path.startup_cost = child->startup_cost + b * q->st.c_eff +
                                    p_trip * trimem_llm_cost_per_batch;
            cp->path.total_cost = child->total_cost + distinct * q->st.c_eff +
                                  rows * cpu_tuple_cost + trips * trimem_llm_cost_per_batch;
        }
        cp->path.pathkeys = child->pathkeys; /* order-preserving filter */
        cp->flags = CUSTOMPATH_SUPPORT_PROJECTION;
        cp->custom_paths = list_make1(child);
        cp->custom_private = list_make2(copyObject(q->input), private_of(q, batch, k_node, placement));
        cp->methods = &path_methods;
        child = &cp->path;
        rows = cp->path.rows;
        above /= q->st.sel;
    }
    return child;
}

/* ---------------------------------------------------------------- base relations */
static void lift_sem(PlannerInfo *root, RelOptInfo *rel, RangeTblEntry *rte);

static void
path_hook(PlannerInfo *root, RelOptInfo *rel, Index rti, RangeTblEntry *rte)
{
    if (prev_path_hook)
        prev_path_hook(root, rel, rti, rte);
    if (trimem_expand_paths(root, rel, rte))
        return;
    /* label quals first: lift_sem then sees the relation's rows before them (SemExtract sits above) */
    trimem_extract_lift(root, rel);
    lift_sem(root, rel, rte);
    /* single-relation query: filter -> extract -> score right above this relation */
    if (trimem_enable && rel->reloptkind == RELOPT_BASEREL &&
        bms_membership(root->all_baserels) == BMS_SINGLETON)
    {
        trimem_wrap_extracts(root, rel, "single relation");
        trimem_wrap_scores(root, rel);
    }
}

static void
lift_sem(PlannerInfo *root, RelOptInfo *rel, RangeTblEntry *rte)
{
    Oid fid, fids[3];
    List *others, *sems, *vars, *old, *wrapped = NIL, *plain = NIL;
    ListCell *lc;
    PathTarget *wide;
    double sel_all = 1.0, rows_post;
    bool single;

    if (!trimem_enable || (rel->baserestrictinfo == NIL && rel->joininfo == NIL))
        return;
    fid = trimem_sem_filter_oid();
    if (!OidIsValid(fid))
        return;
    fids[0] = fid;
    fids[1] = trimem_sem_filter3_oid();
    fids[2] = InvalidOid;
    if (!OidIsValid(fids[1]))
        fids[1] = fid; /* extension not yet upgraded to the 3-argument form */
    check_joininfo(rel, fids);
    sems = collect_sem_quals(rel, fids, &others);
    if (sems == NIL)
        return;
    if (rel->reloptkind != RELOPT_BASEREL)
        ereport(ERROR, (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                        errmsg("trimem.sem_filter on inheritance/partition children is not supported (v1)")));

    list_sort(sems, cmp_rank);
    foreach (lc, sems)
        sel_all *= ((SemQual *) lfirst(lc))->st.sel;
    rel->baserestrictinfo = others;
    /* Index scans take their quals from indrestrictinfo, not baserestrictinfo. */
    foreach (lc, rel->indexlist)
    {
        IndexOptInfo *ix = lfirst(lc);
        ListCell *lq;

        foreach (lq, sems)
            ix->indrestrictinfo = list_delete_ptr(ix->indrestrictinfo, ((SemQual *) lfirst(lq))->ri);
    }
    lifted_count += list_length(sems);
    single = bms_membership(root->all_baserels) == BMS_SINGLETON;
    vars = input_vars(sems);
    wide = widen_target(root, rel->reltarget, vars);
    if (!single)
    {
        /* TOP placement needs the inputs above the scan: widen the rel target in place (all
         * paths share it) and mark the Vars as needed by the final output (relid 0). */
        foreach (lc, vars)
        {
            Var *v = lfirst(lc);

            if (IsA(v, Var) && v->varno == rel->relid && v->varattno >= rel->min_attr)
                rel->attr_needed[v->varattno - rel->min_attr] =
                    bms_add_member(rel->attr_needed[v->varattno - rel->min_attr], 0);
        }
        foreach (lc, vars)
            if (!list_member(rel->reltarget->exprs, lfirst(lc)))
                add_column_to_pathtarget(rel->reltarget, lfirst(lc), 0);
        set_pathtarget_cost_width(root, rel->reltarget);
        wide = rel->reltarget;
    }

    rows_post = rel->rows;
    /* distinct-input estimates below must see the pre-filter relation size */
    rel->rows = clamp_row_est(rows_post / sel_all);
    old = rel->pathlist;
    rel->pathlist = NIL;
    rel->partial_pathlist = NIL; /* trimem.sem_filter is PARALLEL UNSAFE */
    foreach (lc, old)
    {
        Path *child = lfirst(lc);

        child->pathtarget = wide;
        child->rows = clamp_row_est(child->rows / sel_all);
        plain = lappend(plain, child);
        wrapped = lappend(wrapped, wrap_chain(root, rel, child, sems, sel_all, wide,
                                              single ? rel->reltarget : wide, single,
                                              single ? "base (single relation)" : "base"));
    }
    rel->rows = rows_post;
    foreach (lc, wrapped)
        add_path(rel, lfirst(lc));
    if (!single)
    {
        LiftedRel *lr = palloc0(sizeof(LiftedRel));

        lr->root = root;
        lr->rel = rel;
        lr->sems = sems;
        lr->sel_all = sel_all;
        lr->plain_paths = plain;
        lr->wrapped_paths = rel->pathlist; /* survivors of add_path */
        lr->rows_plain = clamp_row_est(rows_post / sel_all);
        lr->rows_wrapped = rows_post;
        lr->alias = pstrdup(rte->eref->aliasname);
        lifted_rels = lappend(lifted_rels, lr);
    }
}

/* ---------------------------------------------------------------- placement over joins */
static RelOptInfo *
run_search(PlannerInfo *root, int levels, List *initial_rels)
{
    if (prev_join_search_hook)
        return prev_join_search_hook(root, levels, initial_rels);
    if (enable_geqo && levels >= geqo_threshold)
        return geqo(root, levels, initial_rels);
    return standard_join_search(root, levels, initial_rels);
}

static void
use_paths(LiftedRel *lr, bool wrapped)
{
    /* Assign directly: add_path may pfree a dominated path that a later trial still needs. */
    lr->rel->pathlist = list_copy(wrapped ? lr->wrapped_paths : lr->plain_paths);
    lr->rel->rows = wrapped ? lr->rows_wrapped : lr->rows_plain;
    set_cheapest(lr->rel);
}

/* TOP placement: SemFilter chain (predicates of `tops`) over every path of the final rel. */
static void
wrap_top(PlannerInfo *root, RelOptInfo *joinrel, List *tops, bool forced)
{
    List *sems = NIL, *old;
    ListCell *lc;
    double sel_all = 1.0;

    if (tops == NIL)
        return;
    foreach (lc, tops)
    {
        LiftedRel *lr = lfirst(lc);

        sems = list_concat(sems, list_copy(lr->sems));
        sel_all *= lr->sel_all;
    }
    list_sort(sems, cmp_rank);
    old = joinrel->pathlist;
    joinrel->pathlist = NIL;
    joinrel->partial_pathlist = NIL;
    foreach (lc, old)
        add_path(joinrel, wrap_chain(root, joinrel, lfirst(lc), sems, sel_all, joinrel->reltarget,
                                     joinrel->reltarget, true,
                                     forced ? "top (above join, forced)" : "top (above join)"));
    joinrel->rows = clamp_row_est(joinrel->rows * sel_all);
    set_cheapest(joinrel);
}

/*
 * bit i of mask = 1 -> lifted rel i is filtered TOP (above the join), else BASE.
 * sc: placement of the SemScore node of the query's sem_score calls (if any):
 *     0 = TOP (above the final scan/join rel), 1 = BASE (on `srel`, before the joins).
 * sx: same for the SemExtract node of the query's sem_extract calls (-1 = none; BASE on `erel`).
 * Stacking order on one relation / above the join: SemFilter -> SemExtract -> SemScore.
 */
static RelOptInfo *
search_with(PlannerInfo *root, int levels, List *initial_rels, List *mine, int mask, bool forced,
            RelOptInfo *srel, int sc, RelOptInfo *erel, int sx, bool keep)
{
    List *tops = NIL, *saved = NIL, *esaved = NIL, *esaved_partial = NIL;
    double esaved_rows = 0;
    ListCell *lc;
    RelOptInfo *rel;

    foreach (lc, mine)
    {
        bool top = (mask >> foreach_current_index(lc)) & 1;

        use_paths(lfirst(lc), !top);
        if (top)
            tops = lappend(tops, lfirst(lc));
    }
    if (erel && sx == 1)
    {
        List *wrapped = trimem_extract_paths(root, erel, forced ? "base (forced)" : "base");

        esaved = erel->pathlist;
        esaved_partial = erel->partial_pathlist;
        esaved_rows = erel->rows;
        if (wrapped != NIL)
        {
            erel->pathlist = wrapped;
            erel->partial_pathlist = NIL; /* an unwrapped partial path would skip the labels */
            erel->rows = clamp_row_est(erel->rows * trimem_extract_qual_sel(root, erel));
            set_cheapest(erel);
        }
    }
    if (srel && sc == 1)
    {
        List *wrapped = trimem_score_paths(root, srel, forced ? "base (forced)" : "base");

        saved = srel->pathlist;
        if (wrapped != NIL)
        {
            srel->pathlist = wrapped;
            set_cheapest(srel);
        }
    }
    rel = run_search(root, levels, initial_rels);
    wrap_top(root, rel, tops, forced);
    if (sx == 0)
        trimem_wrap_extracts(root, rel, forced ? "top (above join, forced)" : "top (above join)");
    if (srel && sc == 0)
        trimem_wrap_scores(root, rel);
    if (srel && sc == 1 && !keep && saved != NIL)
    {
        srel->pathlist = saved; /* the next trial starts from the unscored relation */
        set_cheapest(srel);
    }
    if (erel && sx == 1 && !keep && esaved != NIL)
    {
        erel->pathlist = esaved; /* ... and from the unlabeled relation */
        erel->partial_pathlist = esaved_partial;
        erel->rows = esaved_rows;
        set_cheapest(erel);
    }
    return rel;
}

static RelOptInfo *
trial(PlannerInfo *root, int levels, List *initial_rels, List *mine, int mask, RelOptInfo *srel, int sc,
      RelOptInfo *erel, int sx)
{
    int savelength = list_length(root->join_rel_list);
    struct HTAB *savehash = root->join_rel_hash;
    RelOptInfo *rel;

    root->join_rel_hash = NULL;
    rel = search_with(root, levels, initial_rels, mine, mask, false, srel, sc, erel, sx, false);
    root->join_rel_list = list_truncate(root->join_rel_list, savelength);
    root->join_rel_hash = savehash;
    return rel;
}

const char *
trimem_score_decision(void)
{
    return last_placement ? last_placement : "";
}

/* trimem.force_placement '<alias>=base|top' for the score relation: 1 base, 0 top, -1 none */
static int
forced_score(PlannerInfo *root, RelOptInfo *srel)
{
    char *alias;

    if (!srel || !trimem_force_placement || !trimem_force_placement[0])
        return -1;
    alias = planner_rt_fetch(srel->relid, root)->eref->aliasname;
    if (strstr(trimem_force_placement, psprintf("%s=base", alias)))
        return 1;
    if (strstr(trimem_force_placement, psprintf("%s=top", alias)))
        return 0;
    return -1;
}

static Cost
rel_cost(PlannerInfo *root, RelOptInfo *rel)
{
    Path *p = rel->cheapest_total_path;
    double fraction = root->tuple_fraction;

    if (fraction > 0 && fraction < 1) /* LIMIT: cost of fetching that fraction */
        return p->startup_cost + fraction * (p->total_cost - p->startup_cost);
    return p->total_cost;
}

static char *
mask_name(List *mine, int mask)
{
    StringInfoData s;
    ListCell *lc;

    initStringInfo(&s);
    foreach (lc, mine)
        appendStringInfo(&s, "%s%s=%s", foreach_current_index(lc) ? "," : "",
                         ((LiftedRel *) lfirst(lc))->alias,
                         ((mask >> foreach_current_index(lc)) & 1) ? "top" : "base");
    return s.data;
}

/* trimem.force_placement = 'alias=base,alias2=top' (per relation), else trimem.placement. */
static int
forced_mask(List *mine, bool *forced)
{
    int mask = 0;
    ListCell *lc;

    *forced = false;
    if (trimem_force_placement && trimem_force_placement[0])
    {
        foreach (lc, mine)
        {
            LiftedRel *lr = lfirst(lc);
            char *want_top = psprintf("%s=top", lr->alias);
            char *want_base = psprintf("%s=base", lr->alias);

            if (strstr(trimem_force_placement, want_top))
            {
                mask |= 1 << foreach_current_index(lc);
                *forced = true;
            }
            else if (strstr(trimem_force_placement, want_base))
                *forced = true;
        }
        if (*forced)
            return mask;
    }
    if (trimem_placement != 0)
    {
        *forced = true;
        return trimem_placement == 2 ? (1 << list_length(mine)) - 1 : 0;
    }
    return -1;
}

static RelOptInfo *
join_search(PlannerInfo *root, int levels, List *initial_rels)
{
    List *mine = NIL;
    ListCell *lc;
    int n, best = 0, best_sc = 0, best_sx = -1, chosen, chosen_sc, chosen_sx, nsc, nsx = 1;
    int sxs[2] = {-1, -1};
    Cost best_cost = 0;
    bool forced, first = true;
    StringInfoData log;
    RelOptInfo *srel = trimem_enable ? trimem_score_base_rel(root, initial_rels) : NULL;
    char *salias = srel ? planner_rt_fetch(srel->relid, root)->eref->aliasname : NULL;
    bool ex = trimem_enable && trimem_extract_present(root);
    bool ex_top = false;
    RelOptInfo *erel = NULL;
    char *ealias = NULL;

    if (trimem_enable)
        trimem_extract_check_joins(root); /* label equality across relations: plan-time error */
    if (ex)
    {
        erel = trimem_extract_base_rel(root, initial_rels);
        ealias = erel ? planner_rt_fetch(erel->relid, root)->eref->aliasname : NULL;
        ex_top = trimem_extract_top_ok(root);
        if (!erel && !ex_top)
            ereport(ERROR, (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                            errmsg("trimem.sem_extract quals over several relations cannot be applied above an outer join (v1)"),
                            errhint("Filter on labels of one relation, or use inner joins.")));
        nsx = 0;
        if (ex_top)
            sxs[nsx++] = 0;
        if (erel)
            sxs[nsx++] = 1;
    }

    foreach (lc, lifted_rels)
    {
        LiftedRel *lr = lfirst(lc);

        if (lr->root == root && list_member_ptr(initial_rels, lr->rel))
            mine = lappend(mine, lr);
    }
    if (mine == NIL && srel == NULL && erel == NULL)
    {
        RelOptInfo *rel = run_search(root, levels, initial_rels);

        if (ex)
            trimem_wrap_extracts(root, rel, "top (above join)");
        if (trimem_enable)
            trimem_wrap_scores(root, rel);
        return rel;
    }
    n = list_length(mine);
    if (n > 4)
        ereport(ERROR, (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                        errmsg("trimem: semantic predicates on more than 4 relations are not supported (v1)")));
    nsc = srel ? 2 : 1;

    /* Enumerate every filter BASE/TOP assignment x score x extract placement as a full join search. */
    initStringInfo(&log);
    for (int mask = 0; mask < (1 << n); mask++)
        for (int sc = 0; sc < nsc; sc++)
            for (int xi = 0; xi < nsx; xi++)
            {
                int sx = sxs[xi];
                Cost c = rel_cost(root, trial(root, levels, initial_rels, mine, mask, srel, sc, erel, sx));

                appendStringInfo(&log, "%s%s%sscore:%s=%s", first ? "" : " ", n ? mask_name(mine, mask) : "",
                                 n ? "," : "", srel ? salias : "-", srel ? (sc ? "base" : "top") : "-");
                if (ex)
                    appendStringInfo(&log, ",extract:%s=%s", erel ? ealias : "-", sx ? "base" : "top");
                appendStringInfo(&log, ":%.0f", c);
                if (first || c < best_cost)
                {
                    best = mask;
                    best_sc = sc;
                    best_sx = sx;
                    best_cost = c;
                }
                first = false;
            }
    chosen = n ? forced_mask(mine, &forced) : -1;
    if (!n)
        forced = false;
    if (chosen < 0)
        chosen = best;
    chosen_sc = forced_score(root, srel);
    if (chosen_sc >= 0)
        forced = true;
    else
        chosen_sc = best_sc;
    chosen_sx = ex ? forced_score(root, erel) : -1;
    if (chosen_sx == 0 && !ex_top)
        chosen_sx = -1; /* TOP is not valid here (label quals above an outer join) */
    if (chosen_sx >= 0)
        forced = true;
    else
        chosen_sx = best_sx;
    last_placement = psprintf("%s%s%s%s%s | candidates %s", n ? mask_name(mine, chosen) : "",
                              n && srel ? "," : "",
                              srel ? psprintf("score:%s=%s", salias, chosen_sc ? "base" : "top") : "",
                              ex ? psprintf("%sextract:%s=%s", n || srel ? "," : "",
                                            erel ? ealias : "-", chosen_sx ? "base" : "top")
                                 : "",
                              forced ? " (forced)" : " (cheapest)", log.data);
    elog(DEBUG1, "trimem placement: %s", last_placement);

    /* Re-run the winner for real (its join rels stay registered). */
    return search_with(root, levels, initial_rels, mine, chosen, forced, srel, chosen_sc, erel, chosen_sx, true);
}

/* ---------------------------------------------------------------- plan */
static Plan *
plan_path(PlannerInfo *root, RelOptInfo *rel, CustomPath *best, List *tlist, List *clauses,
          List *custom_plans)
{
    CustomScan *cs = makeNode(CustomScan);
    Plan *child = linitial(custom_plans);
    List *scan_tlist = NIL;
    ListCell *lc;
    AttrNumber resno = 1;

    /* The scan tuple is the child's output row. */
    foreach (lc, child->targetlist)
    {
        TargetEntry *te = lfirst(lc);

        scan_tlist = lappend(scan_tlist,
                             makeTargetEntry(copyObject(te->expr), resno++, te->resname, false));
    }
    cs->scan.plan.targetlist = tlist;
    cs->scan.plan.qual = NIL; /* the remaining quals are applied by the input plan */
    cs->scan.scanrelid = 0;
    cs->custom_plans = custom_plans;
    cs->custom_scan_tlist = scan_tlist;
    cs->custom_exprs = list_make1(copyObject(linitial(best->custom_private)));
    cs->custom_private = lappend(copyObject(lsecond(best->custom_private)),
                                 makeString(pstrdup(last_placement ? last_placement : "")));
    cs->flags = best->flags;
    cs->methods = &trimem_scan_methods;
    planned_count++;
    return &cs->scan.plan;
}

static PlannedStmt *
planner_guard(Query *parse, const char *query_string, int cursorOptions, ParamListInfo params)
{
    int save_lifted = lifted_count, save_planned = planned_count;
    List *save_rels = lifted_rels;
    PlannedStmt *result;

    char *save_placement = last_placement;
    void *save_extract = trimem_extract_planner_enter();

    lifted_count = planned_count = 0;
    lifted_rels = NIL;
    last_placement = NULL;
    PG_TRY();
    {
        result = prev_planner_hook
                     ? prev_planner_hook(parse, query_string, cursorOptions, params)
                     : standard_planner(parse, query_string, cursorOptions, params);
        if (planned_count < lifted_count)
            ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR),
                            errmsg("trimem: %d semantic predicate(s) lifted but only %d SemFilter node(s) planned",
                                   lifted_count, planned_count),
                            errhint("Another set_rel_pathlist_hook replaced the paths. List tjs_pg "
                                    "BEFORE trimem_pg in shared_preload_libraries.")));
        trimem_extract_planner_check();
    }
    PG_FINALLY();
    {
        trimem_extract_planner_leave(save_extract);
        lifted_count = save_lifted;
        planned_count = save_planned;
        lifted_rels = save_rels;
        last_placement = save_placement;
    }
    PG_END_TRY();
    return result;
}

void
trimem_path_init(void)
{
    path_methods.CustomName = "TriMEM SemFilter";
    path_methods.PlanCustomPath = plan_path;
    prev_path_hook = set_rel_pathlist_hook;
    set_rel_pathlist_hook = path_hook;
    prev_join_search_hook = join_search_hook;
    join_search_hook = join_search;
    prev_planner_hook = planner_hook;
    planner_hook = planner_guard;
}
