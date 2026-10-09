/* Native CustomScan SQL adapter: a FunctionScan fallback is deliberately an error. */
#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "executor/executor.h"
#include "commands/explain.h"
#include "commands/extension.h"
#include "catalog/dependency.h"
#include "catalog/pg_proc_d.h"
#include "nodes/extensible.h"
#include "nodes/makefuncs.h"
#include "nodes/nodeFuncs.h"
#include "nodes/supportnodes.h"
#include "optimizer/paths.h"
#include "parser/parsetree.h"
#include "optimizer/pathnode.h"
#include "optimizer/planmain.h"
#include "optimizer/optimizer.h"
#include "optimizer/restrictinfo.h"
#include "utils/builtins.h"
#include "utils/acl.h"
#include "miscadmin.h"
#include "utils/lsyscache.h"
#include "vsgt_exec.h"

typedef struct VsgtScan
{
    CustomScanState css;
    ExprState *request_expr;
    ExprState *vector_expr;
    VsgtState *exec;
    bool explain_only;
} VsgtScan;

static set_rel_pathlist_hook_type previous_path_hook;
static CustomExecMethods exec_methods;
static CustomScanMethods scan_methods;
static CustomPathMethods path_methods;

PG_FUNCTION_INFO_V1(tjs_vsgt_support);
Datum tjs_vsgt_support(PG_FUNCTION_ARGS) { PG_RETURN_POINTER(NULL); }
PG_FUNCTION_INFO_V1(tjs_vsgt);
Datum tjs_vsgt(PG_FUNCTION_ARGS)
{
    ereport(ERROR, (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
        errmsg("tjs_vsgt requires its native CustomScan FROM entry"),
        errhint("SELECT result FROM tjs_vsgt(request_json, query_vector); unsupported SRF forms fail closed.")));
    PG_RETURN_NULL();
}
PG_FUNCTION_INFO_V1(tjs_vsgt_stats);
Datum tjs_vsgt_stats(PG_FUNCTION_ARGS)
{
    Jsonb *stats = vsgt_stats(text_to_cstring(PG_GETARG_TEXT_PP(0)));
    if (!stats) PG_RETURN_NULL();
    PG_RETURN_JSONB_P(stats);
}

static Plan *
plan_path(PlannerInfo *root, RelOptInfo *rel, CustomPath *best,
          List *tlist, List *clauses, List *children)
{
    CustomScan *scan = makeNode(CustomScan);
    scan->scan.plan.targetlist = tlist;
    scan->scan.plan.qual = extract_actual_clauses(clauses, false);
    scan->scan.scanrelid = 0; /* a function RTE is not a heap relation */
    scan->custom_scan_tlist = list_make1(makeTargetEntry(
        (Expr *) makeVar(rel->relid, 1, JSONBOID, -1, InvalidOid, 0), 1, "row", false));
    scan->custom_exprs = copyObject(best->custom_private);
    {
        RangeTblEntry *rte = planner_rt_fetch(rel->relid, root);
        RangeTblFunction *rf = linitial_node(RangeTblFunction, rte->functions);
        scan->custom_private = list_make1_oid(((FuncExpr *) rf->funcexpr)->funcid);
    }
    scan->flags = CUSTOMPATH_SUPPORT_PROJECTION;
    scan->methods = &scan_methods;
    return &scan->scan.plan;
}

static void
path_hook(PlannerInfo *root, RelOptInfo *rel, Index rti, RangeTblEntry *rte)
{
    RangeTblFunction *rf;
    FuncExpr *fn;
    char *name;
    CustomPath *path;
    if (previous_path_hook) previous_path_hook(root, rel, rti, rte);
    if (rte->rtekind != RTE_FUNCTION || list_length(rte->functions) != 1) return;
    rf = linitial_node(RangeTblFunction, rte->functions);
    if (!IsA(rf->funcexpr, FuncExpr)) return;
    fn = (FuncExpr *) rf->funcexpr;
    name = get_func_name(fn->funcid);
    /* Verify the actual C entry via its support function, not a user-defined namesake. */
    if (!name || strcmp(name, "tjs_vsgt") ||
        !OidIsValid(get_func_support(fn->funcid)) ||
        strcmp(get_func_name(get_func_support(fn->funcid)), "tjs_vsgt_support")) return;
    if (getExtensionOfObject(ProcedureRelationId, fn->funcid) != get_extension_oid("tjs_pg", false)) return;
    if (root->parent_root || root->parse->sortClause || root->parse->groupClause ||
        root->parse->hasAggs || root->parse->hasWindowFuncs || root->parse->distinctClause ||
        root->parse->setOperations || root->parse->cteList || root->parse->hasSubLinks ||
        list_length(root->parse->jointree->fromlist) != 1 ||
        !IsA(linitial(root->parse->jointree->fromlist), RangeTblRef))
        ereport(ERROR,(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
            errmsg("VSGT supports a single direct FROM entry without joins, subqueries, sorting or aggregation")));
    if (rte->funcordinality || contain_var_clause((Node *) fn->args))
        ereport(ERROR, (errmsg("VSGT does not support ordinality or correlated requests")));
    path = makeNode(CustomPath);
    path->path.pathtype = T_CustomScan;
    path->path.parent = rel;
    path->path.pathtarget = rel->reltarget;
    path->path.rows = 10;
    path->path.startup_cost = 1;
    path->path.total_cost = 100;
    path->path.parallel_safe = false;
    path->flags = CUSTOMPATH_SUPPORT_PROJECTION;
    path->methods = &path_methods;
    path->custom_private = fn->args;
    /* No materializing FunctionScan alternative is legal for this contract. */
    rel->pathlist = NIL;
    rel->partial_pathlist = NIL;
    add_path(rel, &path->path);
}

static Node *create_state(CustomScan *scan)
{
    VsgtScan *s = palloc0(sizeof(*s));
    NodeSetTag(s, T_CustomScanState);
    s->css.methods = &exec_methods;
    return (Node *) s;
}
static void begin_scan(CustomScanState *node, EState *estate, int eflags)
{
    VsgtScan *s = (VsgtScan *) node;
    CustomScan *plan = (CustomScan *) node->ss.ps.plan;
    Oid funcid = linitial_oid(plan->custom_private);
#if PG_VERSION_NUM >= 160000
    if (object_aclcheck(ProcedureRelationId, funcid, GetUserId(), ACL_EXECUTE) != ACLCHECK_OK)
#else
    if (pg_proc_aclcheck(funcid, GetUserId(), ACL_EXECUTE) != ACLCHECK_OK)
#endif
        aclcheck_error(ACLCHECK_NO_PRIV, OBJECT_FUNCTION, get_func_name(funcid));
    s->explain_only = (eflags & EXEC_FLAG_EXPLAIN_ONLY) != 0;
    s->request_expr = ExecInitExpr(linitial(plan->custom_exprs), &node->ss.ps);
    s->vector_expr = ExecInitExpr(lsecond(plan->custom_exprs), &node->ss.ps);
}
static TupleTableSlot *next_row(ScanState *node)
{
    VsgtScan *s = (VsgtScan *) node;
    TupleTableSlot *slot = node->ss_ScanTupleSlot;
    Jsonb *row;
    ExecClearTuple(slot);
    if (s->explain_only) return slot;
    if (!s->exec)
    {
        bool rn, vn;
        Datum request = ExecEvalExprSwitchContext(s->request_expr, node->ps.ps_ExprContext, &rn);
        Datum vector = ExecEvalExprSwitchContext(s->vector_expr, node->ps.ps_ExprContext, &vn);
        if (rn) ereport(ERROR, (errmsg("VSGT request must not be NULL")));
        s->exec = vsgt_begin(DatumGetJsonbP(request), vector, vn,
                            exprType(lsecond(((CustomScan *) node->ps.plan)->custom_exprs)),
                            node->ps.state->es_snapshot, node->ps.state->es_query_cxt);
    }
    row = vsgt_next(s->exec);
    if (row)
    {
        slot->tts_values[0] = JsonbPGetDatum(row);
        slot->tts_isnull[0] = false;
        ExecStoreVirtualTuple(slot);
    }
    return slot;
}
static bool recheck(ScanState *node, TupleTableSlot *slot) { return true; }
static TupleTableSlot *exec_scan(CustomScanState *node)
{
    return ExecScan(&node->ss, next_row, recheck);
}
static void end_scan(CustomScanState *node)
{
    VsgtScan *s = (VsgtScan *) node;
    if (s->exec) { vsgt_end(s->exec); s->exec = NULL; }
}
static void rescan(CustomScanState *node)
{
    end_scan(node);
    ExecScanReScan(&node->ss);
}
static void explain_scan(CustomScanState *node, List *ancestors, ExplainState *es)
{
    ExplainPropertyText("Semantics", "stream_prefix_v1", es);
    ExplainPropertyText("Lifecycle", "native Begin/Next/End", es);
}
static CustomPathMethods path_methods = { .CustomName = "TriDB VSGT", .PlanCustomPath = plan_path };
static CustomScanMethods scan_methods = { .CustomName = "TriDB VSGT", .CreateCustomScanState = create_state };
static CustomExecMethods exec_methods = {
    .CustomName = "TriDB VSGT", .BeginCustomScan = begin_scan,
    .ExecCustomScan = exec_scan, .EndCustomScan = end_scan, .ReScanCustomScan = rescan,
    .ExplainCustomScan = explain_scan
};
void vsgt_init(void)
{
    RegisterCustomScanMethods(&scan_methods);
    previous_path_hook = set_rel_pathlist_hook;
    set_rel_pathlist_hook = path_hook;
}
