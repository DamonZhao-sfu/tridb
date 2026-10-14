/*
 * trimem_graph.c — "Custom Scan (TriMEM Expand)": costed, streaming typed 1-hop expansion over
 * the native graph access method (Fig. 1(b) "Expand: typed edges, 1 hop, fanout F").
 *
 *   SELECT ... FROM seeds s, LATERAL trimem.expand(s.vid, 'lineage', 16) e JOIN attempt a ON ...
 *
 * The function RTE is planned as a parameterized CustomScan (no FunctionScan tuplestore): one
 * native pull-API scan per outer row, stopped at the fanout cap. Row estimate and cost come from
 * per-edge-type degree statistics in trimem.graph_stats (trimem.analyze_graph()), so the join
 * planner sees the real expansion factor instead of the SRF default of 1000 rows.
 */
#include "postgres.h"

#include <math.h>

#include "access/htup_details.h"
#include "catalog/namespace.h"
#include "catalog/pg_type.h"
#include "commands/explain.h"
#include "executor/executor.h"
#include "executor/spi.h"
#include "fmgr.h"
#include "nodes/extensible.h"
#include "nodes/makefuncs.h"
#include "nodes/nodeFuncs.h"
#include "nodes/supportnodes.h"
#include "optimizer/cost.h"
#include "optimizer/optimizer.h"
#include "optimizer/pathnode.h"
#include "optimizer/restrictinfo.h"
#include "parser/parse_func.h"
#include "parser/parsetree.h"
#include "utils/builtins.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"

#include "gph_pull_api.h"
#include "trimem.h"

#define DEFAULT_DEGREE 10.0

typedef struct GraphStats
{
    int type_id;     /* 0 = unknown type */
    double avg_deg;  /* edges of this type per vertex (all vertices) */
    double max_deg;
    double n_edges;  /* all edges of this type: the relation's "table size" */
    bool from_stats;
} GraphStats;

static CustomPathMethods expand_path_methods;
static CustomScanMethods expand_scan_methods;
static CustomExecMethods expand_exec_methods;

/* ---------------------------------------------------------------- catalog / stats */
static Oid
expand_oid(void)
{
    Oid argtypes[3] = {INT8OID, TEXTOID, INT4OID};

    if (!OidIsValid(get_namespace_oid("trimem", true)))
        return InvalidOid;
    return LookupFuncName(list_make2(makeString("trimem"), makeString("expand")), 3, argtypes, true);
}

static void
lookup_graph_stats(const char *edge_type, GraphStats *out)
{
    Oid t[1] = {TEXTOID};
    Datum v[1] = {CStringGetTextDatum(edge_type)};

    out->type_id = 0;
    out->avg_deg = DEFAULT_DEGREE;
    out->max_deg = DEFAULT_DEGREE;
    out->n_edges = 0;
    out->from_stats = false;
    if (SPI_connect() != SPI_OK_CONNECT)
        elog(ERROR, "trimem: SPI_connect failed");
    if (SPI_execute_with_args(
            "SELECT t.id, s.n_edges::float8 / greatest(s.n_vertices, 1), s.max_deg::float8, s.n_edges::float8 "
            "FROM graph_store.edge_type t LEFT JOIN trimem.graph_stats s ON s.type_id = t.id "
            "WHERE t.name = $1",
            1, t, v, NULL, true, 1) == SPI_OK_SELECT &&
        SPI_processed == 1)
    {
        bool n1, n2, n3, n4;
        HeapTuple tup = SPI_tuptable->vals[0];
        TupleDesc td = SPI_tuptable->tupdesc;
        Datum id = SPI_getbinval(tup, td, 1, &n1);
        Datum avg = SPI_getbinval(tup, td, 2, &n2);
        Datum mx = SPI_getbinval(tup, td, 3, &n3);
        Datum ne = SPI_getbinval(tup, td, 4, &n4);

        out->type_id = n1 ? 0 : DatumGetInt32(id);
        if (!n2)
        {
            out->avg_deg = DatumGetFloat8(avg);
            out->max_deg = n3 ? out->avg_deg : DatumGetFloat8(mx);
            out->n_edges = n4 ? 0 : DatumGetFloat8(ne);
            out->from_stats = true;
        }
    }
    SPI_finish();
}

static double
fanout_of(Node *arg)
{
    if (IsA(arg, Const) && !((Const *) arg)->constisnull)
        return (double) DatumGetInt32(((Const *) arg)->constvalue);
    return 1e9; /* unknown at plan time: no cap */
}

/* Planner row estimate for a FunctionScan of trimem.expand (also used by the CustomPath). */
PG_FUNCTION_INFO_V1(trimem_expand_support);
Datum
trimem_expand_support(PG_FUNCTION_ARGS)
{
    Node *rawreq = (Node *) PG_GETARG_POINTER(0);

    if (IsA(rawreq, SupportRequestRows))
    {
        SupportRequestRows *req = (SupportRequestRows *) rawreq;
        FuncExpr *fn = (FuncExpr *) req->node;
        Node *type_arg;
        GraphStats gs;

        if (!fn || !IsA(fn, FuncExpr) || list_length(fn->args) != 3)
            PG_RETURN_POINTER(NULL);
        type_arg = lsecond(fn->args);
        if (!IsA(type_arg, Const) || ((Const *) type_arg)->constisnull)
            PG_RETURN_POINTER(NULL);
        lookup_graph_stats(TextDatumGetCString(((Const *) type_arg)->constvalue), &gs);
        req->rows = Max(1.0, Min(gs.avg_deg, fanout_of(lthird(fn->args))));
        PG_RETURN_POINTER(req);
    }
    PG_RETURN_POINTER(NULL);
}

/* Direct evaluation (FunctionScan fallback) is a materializing SRF: fail closed. */
PG_FUNCTION_INFO_V1(trimem_expand);
Datum
trimem_expand(PG_FUNCTION_ARGS)
{
    ereport(ERROR, (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                    errmsg("trimem.expand must be planned as a TriMEM Expand node"),
                    errhint("Use it as a single FROM / LATERAL item: FROM s, LATERAL trimem.expand(s.vid, 'type', F) e")));
    PG_RETURN_NULL();
}

/* ---------------------------------------------------------------- planning */
bool
trimem_expand_paths(PlannerInfo *root, RelOptInfo *rel, RangeTblEntry *rte)
{
    RangeTblFunction *rf;
    FuncExpr *fn;
    Oid fid;
    Node *type_arg;
    GraphStats gs;
    char *edge_type;
    CustomPath *cp;
    ParamPathInfo *ppi;
    double edges, rows;
    QualCost qc;
    char buf[64];

    if (rte->rtekind != RTE_FUNCTION || list_length(rte->functions) != 1)
        return false;
    rf = linitial_node(RangeTblFunction, rte->functions);
    if (!IsA(rf->funcexpr, FuncExpr))
        return false;
    fn = (FuncExpr *) rf->funcexpr;
    fid = expand_oid();
    if (!OidIsValid(fid) || fn->funcid != fid)
        return false;
    if (rte->funcordinality)
        ereport(ERROR, (errmsg("trimem.expand does not support WITH ORDINALITY")));
    type_arg = lsecond(fn->args);
    if (!IsA(type_arg, Const) || ((Const *) type_arg)->constisnull)
        ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                        errmsg("trimem.expand edge type must be a non-NULL constant")));
    edge_type = TextDatumGetCString(((Const *) type_arg)->constvalue);
    lookup_graph_stats(edge_type, &gs);
    if (gs.type_id == 0)
        ereport(ERROR, (errcode(ERRCODE_UNDEFINED_OBJECT),
                        errmsg("trimem.expand: unknown edge type \"%s\"", edge_type)));

    edges = Max(1.0, Min(gs.avg_deg, fanout_of(lthird(fn->args)))); /* edges read per call */
    /* The relation's size is the edge set, not one call's output: ndistinct of dst (Memoize,
     * join selectivity) is clamped by rel->tuples, which the SRF estimate would set to ~1. */
    rel->tuples = Max(rel->tuples, gs.n_edges);
    ppi = get_baserel_parampathinfo(root, rel, rel->lateral_relids);
    rows = ppi ? ppi->ppi_rows : rel->rows;
    cost_qual_eval(&qc, rel->baserestrictinfo, root);

    cp = makeNode(CustomPath);
    cp->path.pathtype = T_CustomScan;
    cp->path.parent = rel;
    cp->path.pathtarget = rel->reltarget;
    cp->path.param_info = ppi;
    cp->path.parallel_safe = false;
    cp->path.rows = rows;
    /* open = one adjacency head page; each edge = one slot read + qual + tuple emit */
    cp->path.startup_cost = random_page_cost + qc.startup;
    cp->path.total_cost = cp->path.startup_cost +
                          edges * (cpu_operator_cost + cpu_tuple_cost + qc.per_tuple) +
                          ceil(edges / 512.0) * seq_page_cost;
    cp->flags = CUSTOMPATH_SUPPORT_PROJECTION;
    cp->methods = &expand_path_methods;
    cp->custom_private = list_make4(fn->args, makeInteger(gs.type_id), makeString(edge_type),
                                    makeString(psprintf("avg_deg=%.2f max_deg=%.0f (%s)", gs.avg_deg,
                                                        gs.max_deg, gs.from_stats ? "stats" : "defaults")));
    (void) buf;
    rel->pathlist = NIL;
    rel->partial_pathlist = NIL;
    add_path(rel, &cp->path);
    return true;
}

static Plan *
expand_plan_path(PlannerInfo *root, RelOptInfo *rel, CustomPath *best, List *tlist,
                 List *clauses, List *children)
{
    CustomScan *cs = makeNode(CustomScan);
    List *args = linitial(best->custom_private);

    cs->scan.plan.targetlist = tlist;
    cs->scan.plan.qual = extract_actual_clauses(clauses, false);
    cs->scan.scanrelid = 0;
    cs->custom_scan_tlist =
        list_make2(makeTargetEntry((Expr *) makeVar(rel->relid, 1, INT8OID, -1, InvalidOid, 0), 1,
                                   "src", false),
                   makeTargetEntry((Expr *) makeVar(rel->relid, 2, INT8OID, -1, InvalidOid, 0), 2,
                                   "dst", false));
    cs->custom_exprs = list_make2(copyObject(linitial(args)), copyObject(lthird(args)));
    cs->custom_private = list_make3(copyObject(lsecond(best->custom_private)),
                                    copyObject(lthird(best->custom_private)),
                                    copyObject(lfourth(best->custom_private)));
    cs->flags = best->flags;
    cs->methods = &expand_scan_methods;
    return &cs->scan.plan;
}

/* ---------------------------------------------------------------- execution */
typedef struct ExpandState
{
    CustomScanState css;
    ExprState *seed_expr;
    ExprState *fanout_expr;
    uint32 type_id;
    char *edge_type;
    char *stats;
    const GphPullApi *api;
    void *scan;
    bool done;
    int64 seed;
    int64 fanout;
    int64 emitted_this;
    MemoryContext scan_cxt;
    int64 seeds, edges, inspected, capped;
} ExpandState;

static Node *
expand_create_state(CustomScan *cs)
{
    ExpandState *s = palloc0(sizeof(ExpandState));

    NodeSetTag(s, T_CustomScanState);
    s->css.methods = &expand_exec_methods;
    return (Node *) s;
}

static void
expand_begin(CustomScanState *node, EState *estate, int eflags)
{
    ExpandState *s = (ExpandState *) node;
    CustomScan *cs = (CustomScan *) node->ss.ps.plan;

    s->seed_expr = ExecInitExpr(linitial(cs->custom_exprs), &node->ss.ps);
    s->fanout_expr = ExecInitExpr(lsecond(cs->custom_exprs), &node->ss.ps);
    s->type_id = (uint32) intVal(linitial(cs->custom_private));
    s->edge_type = strVal(lsecond(cs->custom_private));
    s->stats = strVal(lthird(cs->custom_private));
    s->scan_cxt = AllocSetContextCreate(estate->es_query_cxt, "TriMEM Expand scan",
                                        ALLOCSET_SMALL_SIZES);
    if (!(eflags & EXEC_FLAG_EXPLAIN_ONLY))
    {
        /* Loading the trusted graph AM initializes its rendezvous provider (as VSGT does). */
        (void) load_external_function("$libdir/graph_store_am", "gph_visits", true, NULL);
        s->api = *(const GphPullApi **) find_rendezvous_variable(GPH_PULL_API_NAME);
        if (!s->api || s->api->version != 1)
            ereport(ERROR, (errmsg("TriMEM Expand requires graph pull API v1")));
    }
}

static void
expand_close(ExpandState *s)
{
    if (s->scan)
    {
        MemoryContext old = MemoryContextSwitchTo(s->scan_cxt);

        s->api->close(s->scan);
        MemoryContextSwitchTo(old);
        s->scan = NULL;
    }
    /* reset only after close(): no AM state may outlive the context it lives in */
    MemoryContextReset(s->scan_cxt);
}

static TupleTableSlot *
expand_next(ScanState *node)
{
    ExpandState *s = (ExpandState *) node;
    TupleTableSlot *slot = node->ss_ScanTupleSlot;
    ExprContext *econtext = node->ps.ps_ExprContext;

    ExecClearTuple(slot);
    if (s->api == NULL || s->done)
        return slot;
    if (s->scan == NULL)
    {
        bool sn, fnull;
        Datum seed = ExecEvalExprSwitchContext(s->seed_expr, econtext, &sn);
        Datum fan = ExecEvalExprSwitchContext(s->fanout_expr, econtext, &fnull);
        MemoryContext old;

        if (sn)
        {
            s->done = true;
            return slot;
        }
        s->seed = DatumGetInt64(seed);
        s->fanout = fnull ? PG_INT64_MAX : (int64) DatumGetInt32(fan);
        s->emitted_this = 0;
        s->seeds++;
        old = MemoryContextSwitchTo(s->scan_cxt);
        s->scan = s->api->open(s->seed, s->type_id, node->ps.state->es_snapshot);
        MemoryContextSwitchTo(old);
    }
    for (;;)
    {
        int64 dst;
        int r;
        MemoryContext old;

        if (s->emitted_this >= s->fanout)
        {
            s->capped++;
            s->done = true;
            expand_close(s);
            return slot;
        }
        old = MemoryContextSwitchTo(s->scan_cxt);
        r = s->api->next(s->scan, &dst);
        MemoryContextSwitchTo(old);
        if (r == 0)
        {
            s->done = true;
            expand_close(s);
            return slot;
        }
        s->inspected++;
        if (r < 0)
            continue;
        s->edges++;
        s->emitted_this++;
        slot->tts_values[0] = Int64GetDatum(s->seed);
        slot->tts_isnull[0] = false;
        slot->tts_values[1] = Int64GetDatum(dst);
        slot->tts_isnull[1] = false;
        return ExecStoreVirtualTuple(slot);
    }
}

static bool
expand_recheck(ScanState *node, TupleTableSlot *slot)
{
    return true;
}

static TupleTableSlot *
expand_exec(CustomScanState *node)
{
    return ExecScan(&node->ss, expand_next, expand_recheck);
}

static void
expand_end(CustomScanState *node)
{
    ExpandState *s = (ExpandState *) node;

    expand_close(s);
    MemoryContextDelete(s->scan_cxt);
}

static void
expand_rescan(CustomScanState *node)
{
    ExpandState *s = (ExpandState *) node;

    expand_close(s);
    s->done = false;
    ExecScanReScan(&node->ss);
}

static void
expand_explain(CustomScanState *node, List *ancestors, ExplainState *es)
{
    ExpandState *s = (ExpandState *) node;

    ExplainPropertyText("Edge Type", s->edge_type, es);
    ExplainPropertyText("Degree Model", s->stats, es);
    ExplainPropertyText("Lifecycle", "native graph pull API v1, one scan per outer row", es);
    if (es->analyze)
    {
        ExplainPropertyInteger("Seeds", NULL, s->seeds, es);
        ExplainPropertyInteger("Edges Emitted", NULL, s->edges, es);
        ExplainPropertyInteger("Slots Inspected", NULL, s->inspected, es);
        ExplainPropertyInteger("Fanout Capped", NULL, s->capped, es);
    }
}

void
trimem_graph_init(void)
{
    expand_path_methods.CustomName = "TriMEM Expand";
    expand_path_methods.PlanCustomPath = expand_plan_path;
    expand_scan_methods.CustomName = "TriMEM Expand";
    expand_scan_methods.CreateCustomScanState = expand_create_state;
    expand_exec_methods.CustomName = "TriMEM Expand";
    expand_exec_methods.BeginCustomScan = expand_begin;
    expand_exec_methods.ExecCustomScan = expand_exec;
    expand_exec_methods.EndCustomScan = expand_end;
    expand_exec_methods.ReScanCustomScan = expand_rescan;
    expand_exec_methods.ExplainCustomScan = expand_explain;
    RegisterCustomScanMethods(&expand_scan_methods);
}
