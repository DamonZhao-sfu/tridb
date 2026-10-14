/*
 * trimem_exec.c — executor for "Custom Scan (TriMEM SemFilter)".
 *
 * Streaming, bounded: each refill pulls at most `batch` child rows, resolves their verdicts
 * (cache probe by key, misses through the worker mailbox), then emits the survivors in child
 * order. Early termination is inherited: if the parent stops pulling, no further batch is judged.
 *
 * Memory: child rows and inputs are copied into a per-batch context that is reset only when the
 * next batch is pulled. ExecProcNode and SPI always run in the node's own context, never in the
 * batch context (the VSGT killedItems lesson: no executor/index state in a context we reset).
 */
#include "postgres.h"

#include "catalog/pg_type.h"
#include "commands/explain.h"
#include "executor/executor.h"
#include "executor/spi.h"
#include "nodes/extensible.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/memutils.h"
#include "utils/ruleutils.h"

#include "trimem.h"

typedef struct SemFilterState
{
    CustomScanState css;
    PlanState *child;
    ExprState *input;
    /* plan-time decisions (custom_private) */
    char *predicate;
    char *model;
    char *pv;
    double sel, hit, c_eff, rank, n_stats, k_hint;
    int batch;
    bool from_stats;
    char *placement;
    char *decision;
    int unknown; /* NULL input: -1/0 drop, 1 keep */
    double n_cached;
    /* batch buffer */
    MemoryContext batch_cxt;
    MinimalTuple *rows;
    bool *keep;
    int nrows;
    int pos;
    bool child_done;
    /* EXPLAIN ANALYZE counters */
    int64 judged, hits, calls, removed, batches, null_inputs;
    double wait_ms;
} SemFilterState;

static CustomExecMethods exec_methods;
CustomScanMethods trimem_scan_methods;
static SPIPlanPtr probe_plan = NULL;

static double
f_at(List *l, int i)
{
    return strtod(strVal(list_nth(l, i)), NULL);
}

static Node *
create_state(CustomScan *cs)
{
    SemFilterState *s = palloc0(sizeof(SemFilterState));

    NodeSetTag(s, T_CustomScanState);
    s->css.methods = &exec_methods;
    return (Node *) s;
}

static void
begin_scan(CustomScanState *node, EState *estate, int eflags)
{
    SemFilterState *s = (SemFilterState *) node;
    CustomScan *cs = (CustomScan *) node->ss.ps.plan;
    List *p = cs->custom_private;

    s->child = ExecInitNode(linitial(cs->custom_plans), estate, eflags);
    node->custom_ps = list_make1(s->child);
    s->input = ExecInitExpr(linitial(cs->custom_exprs), &node->ss.ps);
    s->predicate = strVal(list_nth(p, 0));
    s->model = strVal(list_nth(p, 1));
    s->pv = strVal(list_nth(p, 2));
    s->sel = f_at(p, 3);
    s->hit = f_at(p, 4);
    s->c_eff = f_at(p, 5);
    s->rank = f_at(p, 6);
    s->n_stats = f_at(p, 7);
    s->k_hint = f_at(p, 8);
    s->batch = intVal(list_nth(p, 9));
    s->from_stats = intVal(list_nth(p, 10)) != 0;
    s->placement = strVal(list_nth(p, 11));
    s->n_cached = f_at(p, 12);
    s->unknown = intVal(list_nth(p, 13));
    s->decision = strVal(list_nth(p, 14));
    s->batch_cxt = AllocSetContextCreate(estate->es_query_cxt, "TriMEM SemFilter batch",
                                         ALLOCSET_DEFAULT_SIZES);
    s->rows = MemoryContextAlloc(estate->es_query_cxt, sizeof(MinimalTuple) * s->batch);
    s->keep = MemoryContextAlloc(estate->es_query_cxt, sizeof(bool) * s->batch);
}

/* Look up cached verdicts for `keys` (unique). found[i] = -1 miss, 0/1 verdict. */
static void
probe_cache(char **keys, int n, int *found)
{
    Datum *elems;
    ArrayType *arr;
    Datum arg;
    int ret;

    for (int i = 0; i < n; i++)
        found[i] = -1;
    if (n == 0)
        return;
    if (SPI_connect() != SPI_OK_CONNECT)
        elog(ERROR, "trimem: SPI_connect failed");
    if (probe_plan == NULL)
    {
        Oid argtypes[1] = {TEXTARRAYOID};
        SPIPlanPtr plan = SPI_prepare("SELECT key, verdict FROM trimem.sem_cache WHERE key = ANY($1)",
                                      1, argtypes);

        if (plan == NULL)
            elog(ERROR, "trimem: cannot prepare cache probe: %s", SPI_result_code_string(SPI_result));
        if (SPI_keepplan(plan) != 0)
            elog(ERROR, "trimem: SPI_keepplan failed");
        probe_plan = plan;
    }
    elems = palloc(sizeof(Datum) * n);
    for (int i = 0; i < n; i++)
        elems[i] = CStringGetTextDatum(keys[i]);
    arr = construct_array_builtin(elems, n, TEXTOID);
    arg = PointerGetDatum(arr);
    ret = SPI_execute_plan(probe_plan, &arg, NULL, true, 0);
    if (ret != SPI_OK_SELECT)
        elog(ERROR, "trimem: cache probe failed: %s", SPI_result_code_string(ret));
    for (uint64 r = 0; r < SPI_processed; r++)
    {
        bool isnull;
        char *k = TextDatumGetCString(SPI_getbinval(SPI_tuptable->vals[r], SPI_tuptable->tupdesc, 1, &isnull));
        Datum v = SPI_getbinval(SPI_tuptable->vals[r], SPI_tuptable->tupdesc, 2, &isnull);

        for (int i = 0; i < n; i++)
            if (found[i] < 0 && strcmp(keys[i], k) == 0)
            {
                found[i] = isnull ? -1 : (DatumGetBool(v) ? 1 : 0);
                break;
            }
    }
    SPI_finish();
}

static void
fill_batch(SemFilterState *s)
{
    ExprContext *econtext = s->css.ss.ps.ps_ExprContext;
    MemoryContext node_cxt = CurrentMemoryContext;
    char **inputs, **keys, **ukeys;
    int *row_u, nu = 0, *found;
    int64 ntrue = 0, nhit = 0;
    TrimemRequest *reqs;
    int nreq = 0;

    MemoryContextReset(s->batch_cxt);
    s->nrows = 0;
    s->pos = 0;
    inputs = MemoryContextAllocZero(s->batch_cxt, sizeof(char *) * s->batch);
    while (s->nrows < s->batch)
    {
        TupleTableSlot *cslot = ExecProcNode(s->child); /* node context, not batch context */
        bool isnull;
        Datum d;

        if (TupIsNull(cslot))
        {
            s->child_done = true;
            break;
        }
        econtext->ecxt_scantuple = cslot;
        d = ExecEvalExprSwitchContext(s->input, econtext, &isnull);
        {
            MemoryContext old = MemoryContextSwitchTo(s->batch_cxt);

            s->rows[s->nrows] = ExecCopySlotMinimalTuple(cslot);
            inputs[s->nrows] = isnull ? NULL : TextDatumGetCString(d);
            MemoryContextSwitchTo(old);
        }
        ResetExprContext(econtext);
        s->nrows++;
    }
    if (s->nrows == 0)
        return;
    s->batches++;

    /* keys, de-duplicated within the batch */
    {
        MemoryContext old = MemoryContextSwitchTo(s->batch_cxt);

        keys = palloc0(sizeof(char *) * s->nrows);
        ukeys = palloc0(sizeof(char *) * s->nrows);
        row_u = palloc(sizeof(int) * s->nrows);
        found = palloc(sizeof(int) * s->nrows);
        for (int i = 0; i < s->nrows; i++)
        {
            row_u[i] = -1;
            if (inputs[i] == NULL)
                continue;
            keys[i] = palloc(65);
            trimem_key_hex(TRIMEM_OP_FILTER, s->model, s->pv, s->predicate, inputs[i], keys[i]);
            for (int u = 0; u < nu; u++)
                if (strcmp(ukeys[u], keys[i]) == 0)
                {
                    row_u[i] = u;
                    break;
                }
            if (row_u[i] < 0)
            {
                ukeys[nu] = keys[i];
                row_u[i] = nu++;
            }
        }
        MemoryContextSwitchTo(old);
    }

    probe_cache(ukeys, nu, found);
    for (int u = 0; u < nu; u++)
        if (found[u] >= 0)
            nhit++;

    reqs = MemoryContextAllocZero(s->batch_cxt, sizeof(TrimemRequest) * Max(nu, 1));
    {
        int *req_u = MemoryContextAlloc(s->batch_cxt, sizeof(int) * Max(nu, 1));

        for (int u = 0; u < nu; u++)
            if (found[u] < 0)
            {
                int first = -1;

                for (int i = 0; i < s->nrows && first < 0; i++)
                    if (row_u[i] == u)
                        first = i;
                reqs[nreq].predicate = s->predicate;
                reqs[nreq].input = inputs[first];
                req_u[nreq++] = u;
            }
        if (nreq > 0)
        {
            MemoryContextSwitchTo(node_cxt);
            trimem_mailbox_run(s->model, s->pv, reqs, nreq, &s->wait_ms);
            for (int r = 0; r < nreq; r++)
                found[req_u[r]] = reqs[r].verdict;
            s->calls += nreq;
        }
    }

    for (int i = 0; i < s->nrows; i++)
    {
        s->keep[i] = row_u[i] >= 0 ? found[row_u[i]] == 1 : s->unknown == 1;
        if (inputs[i] == NULL)
            s->null_inputs++;
        if (s->keep[i])
            ntrue++;
        else
            s->removed++;
    }
    s->judged += s->nrows;
    s->hits += nhit;
    /* planner statistics are per unique judged input: n_eval, n_true, n_hit */
    {
        int64 utrue = 0;

        for (int u = 0; u < nu; u++)
            if (found[u] == 1)
                utrue++;
        trimem_stats_add(s->model, s->pv, s->predicate, nu, utrue, nhit);
    }
    (void) ntrue;
}

static TupleTableSlot *
next_row(ScanState *node)
{
    SemFilterState *s = (SemFilterState *) node;
    TupleTableSlot *slot = node->ss_ScanTupleSlot;

    for (;;)
    {
        while (s->pos < s->nrows)
        {
            int i = s->pos++;

            if (s->keep[i])
            {
                ExecForceStoreMinimalTuple(s->rows[i], slot, false);
                return slot;
            }
        }
        if (s->child_done)
            return ExecClearTuple(slot);
        fill_batch(s);
        if (s->nrows == 0)
            return ExecClearTuple(slot);
    }
}

static bool
recheck(ScanState *node, TupleTableSlot *slot)
{
    return true;
}

static TupleTableSlot *
exec_scan(CustomScanState *node)
{
    return ExecScan(&node->ss, next_row, recheck);
}

static void
end_scan(CustomScanState *node)
{
    SemFilterState *s = (SemFilterState *) node;

    ExecEndNode(s->child);
    MemoryContextDelete(s->batch_cxt);
}

static void
rescan(CustomScanState *node)
{
    SemFilterState *s = (SemFilterState *) node;

    s->nrows = s->pos = 0;
    s->child_done = false;
    MemoryContextReset(s->batch_cxt);
    if (s->child->chgParam == NULL)
        ExecReScan(s->child);
    ExecScanReScan(&node->ss);
}

static void
explain_scan(CustomScanState *node, List *ancestors, ExplainState *es)
{
    SemFilterState *s = (SemFilterState *) node;
    CustomScan *cs = (CustomScan *) node->ss.ps.plan;
    List *ctx = set_deparse_context_plan(es->deparse_cxt, node->ss.ps.plan, ancestors);
    char *input = deparse_expression(linitial(cs->custom_exprs), ctx, es->verbose, false);
    char buf[128];

    ExplainPropertyText("Semantic Predicate", s->predicate, es);
    ExplainPropertyText("Semantic Input", input, es);
    ExplainPropertyText("Model", s->model, es);
    ExplainPropertyText("Prompt Version", s->pv, es);
    snprintf(buf, sizeof(buf), "sel=%.3f hit=%.3f c_eff=%.2f rank=%.3g (%s, n=%.0f, cached=%.0f)",
             s->sel, s->hit, s->c_eff, s->rank, s->from_stats ? "stats" : "defaults", s->n_stats,
             s->n_cached);
    ExplainPropertyText("Semantic Cost Model", buf, es);
    ExplainPropertyText("Placement", s->placement, es);
    if (s->unknown >= 0)
        ExplainPropertyText("Unknown Input", s->unknown ? "keep" : "drop", es);
    if (s->decision[0])
        ExplainPropertyText("Placement Decision", s->decision, es);
    snprintf(buf, sizeof(buf), "%d (k_hint=%.0f)", s->batch, s->k_hint);
    ExplainPropertyText("Batch Size", buf, es);
    if (es->analyze)
    {
        ExplainPropertyInteger("Rows Judged", NULL, s->judged, es);
        ExplainPropertyInteger("Rows Removed by Semantic Filter", NULL, s->removed, es);
        ExplainPropertyInteger("Cache Hits", NULL, s->hits, es);
        ExplainPropertyInteger("Worker Requests", NULL, s->calls, es);
        ExplainPropertyInteger("Batches", NULL, s->batches, es);
        ExplainPropertyFloat("LLM Wait", "ms", s->wait_ms, 1, es);
    }
}

void
trimem_exec_init(void)
{
    trimem_scan_methods.CustomName = "TriMEM SemFilter";
    trimem_scan_methods.CreateCustomScanState = create_state;
    exec_methods.CustomName = "TriMEM SemFilter";
    exec_methods.BeginCustomScan = begin_scan;
    exec_methods.ExecCustomScan = exec_scan;
    exec_methods.EndCustomScan = end_scan;
    exec_methods.ReScanCustomScan = rescan;
    exec_methods.ExplainCustomScan = explain_scan;
    RegisterCustomScanMethods(&trimem_scan_methods);
}
