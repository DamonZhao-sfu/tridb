/* Value-per-call native session cursor, no planner hook or materialized results. */
#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/lsyscache.h"
#include "utils/builtins.h"
#include "utils/snapmgr.h"
#include "executor/executor.h"
#include "vsgt_exec.h"
PG_MODULE_MAGIC;
PG_FUNCTION_INFO_V1(tjs_session_cursor);
PG_FUNCTION_INFO_V1(tjs_session_cursor_stats);
static void close_cursor(Datum arg)
{
    vsgt_end((VsgtState *) DatumGetPointer(arg));
}
Datum tjs_session_cursor(PG_FUNCTION_ARGS)
{
    FuncCallContext *f;VsgtState *s;Jsonb *row;
    if(SRF_IS_FIRSTCALL()) {
        MemoryContext old;
        if(PG_ARGISNULL(0)||PG_ARGISNULL(1))ereport(ERROR,(errmsg("request and query vector required")));
        f=SRF_FIRSTCALL_INIT();old=MemoryContextSwitchTo(f->multi_call_memory_ctx);
        s=vsgt_begin(PG_GETARG_JSONB_P(0),PG_GETARG_DATUM(1),false,
                     get_fn_expr_argtype(fcinfo->flinfo,1),GetActiveSnapshot(),f->multi_call_memory_ctx);
        RegisterExprContextCallback(((ReturnSetInfo *)fcinfo->resultinfo)->econtext,
                                    close_cursor,PointerGetDatum(s));
        f->user_fctx=s;MemoryContextSwitchTo(old);
    }
    f=SRF_PERCALL_SETUP();s=f->user_fctx;row=vsgt_next(s);
    if(row)SRF_RETURN_NEXT(f,JsonbPGetDatum(row));
    UnregisterExprContextCallback(((ReturnSetInfo *)fcinfo->resultinfo)->econtext,
                                  close_cursor,PointerGetDatum(s));
    SRF_RETURN_DONE(f);
}
Datum tjs_session_cursor_stats(PG_FUNCTION_ARGS)
{
    Jsonb *r=vsgt_stats(text_to_cstring(PG_GETARG_TEXT_PP(0)));
    if(!r)PG_RETURN_NULL();
    PG_RETURN_JSONB_P(r);
}
