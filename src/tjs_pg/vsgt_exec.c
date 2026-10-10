/* VSGT native streaming executor. No materialize SRF, Python pipeline or graph join. */
#include "postgres.h"
#include <float.h>
#include <math.h>
#include "commands/defrem.h"
#include "utils/json.h"
#include "utils/datum.h"
#include "access/genam.h"
#include "access/table.h"
#include "access/relscan.h"
#include "access/nbtree.h"
#include "catalog/namespace.h"
#include "catalog/pg_type_d.h"
#include "catalog/pg_am_d.h"
#include "executor/executor.h"
#include "executor/spi.h"
#include "fmgr.h"
#include "lib/stringinfo.h"
#include "miscadmin.h"
#include "utils/acl.h"
#include "utils/builtins.h"
#include "utils/fmgroids.h"
#include "utils/jsonb.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/rel.h"
#include "utils/snapmgr.h"
#include "utils/timestamp.h"
#include "utils/guc.h"
#include "vsgt_exec.h"
#include "extension/graph_store_am/gph_pull_api.h"

#define STAT_SLOTS 64
#define MAX_ID 64
/* Fixed bounded backend registry; never stores query resources or payloads. */
typedef struct VsgtStat
{
    char id[MAX_ID], strategy[24], reason[32];
    bool active;
    uint64 seq;
    int64 ann, anchors, edges, graph_calls, predicates, seeds, emitted, visited, dedup;
    int64 vector_calls, hydration, peak_bytes, open_work, requested_k, state_bytes, state_limit;
    double total_ms, first_row_ms, ann_ms, graph_ms, predicate_ms, hydrate_ms;
    /* Bounded provenance for consumed graph seeds, including non-emitting seeds.
     * m is validated at <=1024. No payloads or query-owned pointers are retained. */
    int64 consumed_seed_ids[1024];
    int64 seed_filtered, output_filtered;
} VsgtStat;
static VsgtStat stat_slots[STAT_SLOTS];
static uint64 stat_sequence;

typedef struct QueueNode { int64 vid, parent; int32 depth; } QueueNode;
typedef struct SeenEntry { int64 vid; uint32 epoch; } SeenEntry;
struct VsgtState
{
    MemoryContext ctx, scratch;
    MemoryContextCallback callback;
    VsgtStat *st;
    Snapshot snapshot;
    Relation heap, pk, ann_index, anchors, anchor_index;
    IndexScanDesc probe, ann_scan, anchor_scan;
    TupleTableSlot *slot, *ann_slot, *anchor_slot;
    FmgrInfo distfn;
    Datum vector;
    AttrNumber idcol, uidcol, scopecol, taskcol, sessioncol, validcol, fitcol, payloadcol, veccol, kindcol;
    char *scope, *task, *excluded_session, *excluded_artifact, *anchor_set;
    bool graph_only, vector_only, backward, seed_valid, output_valid, output_finite;
    bool seed_min_set, output_min_set;
    /* Emit the ANN seed itself, not only its descendants. The seed is the
     * highest-ranked node the vector stage found; leaving it unemitted meant a
     * request could consume five good seeds and return nothing. */
    bool emit_seeds, seed_pending;
    /* Top-k seed selection. The ANN streams by cosine similarity, so a threshold
     * only says "not terrible" while the paper asks for the highest-fitness nodes.
     * When on, ANN candidates are drained into this buffer, ranked by fitness, and
     * the best m become the seeds. */
    bool seed_topk; int ranked_n, ranked_at; int64 *ranked_ids; double *ranked_fit;
    double seed_min, output_min;
    int m, k, depth;
    int64 ann_budget, edge_budget;
    int edge_type;
    GphPullApi *api;
    void *graph_scan;
    QueueNode *queue;
    SeenEntry *seen, *emitted;
    uint32 epoch;
    uint32 hashmask;
    int capacity, tail, head, current;
    int64 seed;
    int64 *seed_ids;
    int root_status;
    int path_index;
    bool active_seed, closed;
    TimestampTz started;
};

static JsonbValue *jget(Jsonb *j, const char *key)
{
    JsonbValue k;
    k.type = jbvString; k.val.string.val = (char *) key; k.val.string.len = strlen(key);
    return findJsonbValueFromContainer(&j->root, JB_FOBJECT, &k);
}
static char *jstr(Jsonb *j, const char *key, const char *def)
{
    JsonbValue *v = jget(j,key);
    if (!v) return def ? pstrdup(def) : NULL;
    if (v->type != jbvString) ereport(ERROR,(errmsg("VSGT %s must be a string",key)));
    return pnstrdup(v->val.string.val,v->val.string.len);
}
static double jnum(Jsonb *j, const char *key, double def)
{
    JsonbValue *v = jget(j,key);
    if (!v) return def;
    if (v->type != jbvNumeric) ereport(ERROR,(errmsg("VSGT %s must be numeric",key)));
    return DatumGetFloat8(DirectFunctionCall1(numeric_float8, NumericGetDatum(v->val.numeric)));
}
static int jint(Jsonb *j, const char *key, int def, int min, int max)
{
    double n = jnum(j,key,def);
    if (!isfinite(n) || n < min || n > max || n != (int)n)
        ereport(ERROR,(errmsg("VSGT %s must be an integer in [%d,%d]",key,min,max)));
    return (int)n;
}
static bool jbool(Jsonb *j, const char *key, bool def)
{
    JsonbValue *v = jget(j,key);
    if (!v) return def;
    if (v->type != jbvBool) ereport(ERROR,(errmsg("VSGT %s must be boolean",key)));
    return v->val.boolean;
}
static double millis(TimestampTz from) { return (GetCurrentTimestamp()-from)/1000.0; }
static void quoted(StringInfo b, const char *s) { escape_json(b,s); }
static VsgtStat *new_stat(const char *id, const char *strategy)
{
    VsgtStat *choice = NULL;
    int i;
    if (!id || !*id || strlen(id)>=MAX_ID) ereport(ERROR,(errmsg("VSGT request_id requires 1..63 bytes")));
    for (i=0;i<STAT_SLOTS;i++)
    {
        VsgtStat *p=&stat_slots[i];
        if (!strcmp(id,p->id))
        {
            if(p->active) ereport(ERROR,(errmsg("VSGT request_id already active")));
            choice=p; break;
        }
        if (!p->active && (!choice || p->seq<choice->seq)) choice=p;
    }
    if (!choice) ereport(ERROR,(errmsg("VSGT has 64 active requests; close a cursor")));
    memset(choice,0,sizeof(*choice));
    strlcpy(choice->id,id,sizeof(choice->id)); strlcpy(choice->strategy,strategy,sizeof(choice->strategy));
    strlcpy(choice->reason,"open",sizeof(choice->reason));
    choice->active=true;choice->seq=++stat_sequence; choice->first_row_ms=-1;
    return choice;
}
Jsonb *vsgt_stats(const char *id)
{
    int i; StringInfoData b;
    for(i=0;i<STAT_SLOTS;i++) if(!strcmp(stat_slots[i].id,id))
    {
        VsgtStat *s=&stat_slots[i]; initStringInfo(&b);
        appendStringInfoString(&b,"{\"request_id\":");quoted(&b,s->id);
        appendStringInfoString(&b,",\"strategy\":");quoted(&b,s->strategy);
        appendStringInfoString(&b,",\"termination_reason\":");quoted(&b,s->reason);
        appendStringInfo(&b,",\"active\":%s,\"semantics_version\":\"stream_prefix_v1\","
            "\"ann_candidates\":%lld,\"anchor_candidates\":%lld,\"edge_slots\":%lld,\"graph_calls\":%lld,"
            "\"predicate_probes\":%lld,\"seeds_consumed\":%lld,\"emitted\":%lld,"
            "\"visited\":%lld,\"dedup\":%lld,\"vector_calls\":%lld,\"hydration_calls\":%lld,"
            "\"peak_owned_bytes\":%lld,\"open_work\":%lld,\"total_ms\":%.3f,"
            "\"first_row_ms\":%.3f,\"ann_ms\":%.3f,\"graph_ms\":%.3f,"
            "\"predicate_ms\":%.3f,\"hydrate_ms\":%.3f,\"requested_k\":%lld," 
            "\"underfill\":%s,\"state_bytes\":%lld,\"state_limit_bytes\":%lld}",
            s->active?"true":"false",(long long)s->ann,(long long)s->anchors,(long long)s->edges,(long long)s->graph_calls,
            (long long)s->predicates,(long long)s->seeds,(long long)s->emitted,
            (long long)s->visited,(long long)s->dedup,(long long)s->vector_calls,(long long)s->hydration,
            (long long)s->peak_bytes,(long long)s->open_work,s->total_ms,s->first_row_ms,
            s->ann_ms,s->graph_ms,s->predicate_ms,s->hydrate_ms, (long long)s->requested_k,
            s->emitted<s->requested_k?"true":"false", (long long)s->state_bytes,(long long)s->state_limit);
        /* Extend the telemetry object without consuming any more iterator rows. */
        b.len--; b.data[b.len]='\0';
        appendStringInfoString(&b,",\"consumed_seed_ids\":[");
        {
            int j;
            for(j=0;j<s->seeds;j++)
                appendStringInfo(&b,"%s%lld",j?",":"",(long long)s->consumed_seed_ids[j]);
        }
        appendStringInfo(&b,"],\"seed_filtered\":%lld,\"output_filtered\":%lld}", (long long)s->seed_filtered, (long long)s->output_filtered);
        return DatumGetJsonbP(DirectFunctionCall1(jsonb_in,CStringGetDatum(b.data)));
    }
    return NULL;
}
static void reset_callback(void *arg)
{
    VsgtState *s=arg;
    /* On ERROR, ResourceOwner frees PG handles. Never close stale handles after abort. */
    if (!s->closed && s->st && s->st->active)
    {
        s->st->active=false; strlcpy(s->st->reason,"aborted",sizeof(s->st->reason));
        s->st->total_ms=millis(s->started);
    }
}
static void require_select(Relation rel)
{
    if (rel->rd_rel->relrowsecurity)
        ereport(ERROR,(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),errmsg("VSGT RLS relations are not supported")));
    if (pg_class_aclcheck(RelationGetRelid(rel),GetUserId(),ACL_SELECT)!=ACLCHECK_OK)
        aclcheck_error(ACLCHECK_NO_PRIV,OBJECT_TABLE,RelationGetRelationName(rel));
}
static AttrNumber column(Relation rel, const char *name, Oid type)
{
    AttrNumber n=get_attnum(RelationGetRelid(rel),name);
    if(n<=0 || TupleDescAttr(RelationGetDescr(rel),n-1)->atttypid!=type)
        ereport(ERROR,(errmsg("VSGT requires column %s of type %s",name,format_type_be(type))));
    return n;
}
static Relation primary_index(Relation rel, AttrNumber col)
{
    List *ids=RelationGetIndexList(rel); ListCell *lc;
    foreach(lc,ids)
    {
        Relation idx=index_open(lfirst_oid(lc),AccessShareLock);
        if(idx->rd_rel->relam==BTREE_AM_OID && idx->rd_index->indisunique &&
           idx->rd_index->indisvalid && idx->rd_index->indnkeyatts==1 &&
           idx->rd_index->indkey.values[0]==col && RelationGetIndexPredicate(idx)==NIL)
        { list_free(ids);return idx; }
        index_close(idx,AccessShareLock);
    }
    ereport(ERROR,(errmsg("VSGT requires a valid unique btree index on id")));
    return NULL;
}
static bool same_text(TupleTableSlot *slot, AttrNumber col, const char *s)
{
    bool null; Datum d=slot_getattr(slot,col,&null);
    text *t;
    if(null) return false;
    t=DatumGetTextPP(d);
    return VARSIZE_ANY_EXHDR(t)==strlen(s) && !memcmp(VARDATA_ANY(t),s,strlen(s));
}
static bool scope_ok(VsgtState *s, TupleTableSlot *slot)
{
    return (s->kindcol<=0 || same_text(slot,s->kindcol,"node")) && same_text(slot,s->scopecol,s->scope) && same_text(slot,s->taskcol,s->task) &&
        (!s->excluded_session || !same_text(slot,s->sessioncol,s->excluded_session));
}
static bool passes(VsgtState *s, TupleTableSlot *slot, bool seed)
{
    bool null; Datum d; TimestampTz t=GetCurrentTimestamp(); bool ok=true;
    s->st->predicates++;
    if(!scope_ok(s,slot)) ok=false;
    if (ok && (seed?s->seed_valid:s->output_valid))
    { d=slot_getattr(slot,s->validcol,&null); if(null || !DatumGetBool(d)) ok=false; }
    if(ok && !seed && s->output_finite)
    { d=slot_getattr(slot,s->fitcol,&null); if(null || !isfinite(DatumGetFloat8(d)))ok=false; }
    if (ok && (seed?s->seed_min_set:s->output_min_set))
    { d=slot_getattr(slot,s->fitcol,&null); if(null || !isfinite(DatumGetFloat8(d)) ||
        DatumGetFloat8(d)<(seed?s->seed_min:s->output_min)) ok=false; }
    if(ok && s->excluded_artifact)
    {
        d=slot_getattr(slot,s->payloadcol,&null);
        if(!null)
        {
            JsonbValue *v=jget(DatumGetJsonbP(d),"artifact_uid");
            if(v && v->type==jbvString && v->val.string.len==strlen(s->excluded_artifact) &&
               !memcmp(v->val.string.val,s->excluded_artifact,v->val.string.len)) ok=false;
        }
    }
    if (!ok) { if (seed) s->st->seed_filtered++; else s->st->output_filtered++; }
    s->st->predicate_ms+=millis(t);return ok;
}
/*
 * Index AM calls run in the request context, never in the per-row scratch context.
 * btgettuple lazily pallocs so->killedItems in CurrentMemoryContext the first time it
 * meets a dead heap tuple (kill_prior_tuple); if that happens under scratch, the next
 * scratch reset frees it and index_endscan() pfrees a dangling pointer ("could not find
 * block containing chunk", or a SIGSEGV). Updated payload rows make this routine.
 * Regression: test/vsgt_dead_tuple_test.sql.
 */
static bool probe_node(VsgtState *s,int64 vid)
{
    ScanKeyData key; bool result; MemoryContext prev=MemoryContextSwitchTo(s->ctx);
    ScanKeyInit(&key,1,BTEqualStrategyNumber,F_INT8EQ,Int64GetDatum(vid));
    index_rescan(s->probe,&key,1,NULL,0);ExecClearTuple(s->slot);
    result=index_getnext_slot(s->probe,ForwardScanDirection,s->slot);
    MemoryContextSwitchTo(prev);
    return result;
}
static bool anchor_next(VsgtState *s)
{
    MemoryContext prev=MemoryContextSwitchTo(s->ctx);
    bool result=index_getnext_slot(s->anchor_scan,ForwardScanDirection,s->anchor_slot);
    MemoryContextSwitchTo(prev);
    return result;
}
static bool seen_add(SeenEntry *table,uint32 mask,uint32 epoch,int64 vid)
{
    uint32 at=(uint64)vid*11400714819323198485ull & mask;
    while(table[at].epoch==epoch)
    { if(table[at].vid==vid) return false;at=(at+1)&mask; }
    table[at].epoch=epoch;table[at].vid=vid;return true;
}
static int edge_type_id(const char *name)
{
    Oid args[1]={TEXTOID};Datum values[1]={CStringGetTextDatum(name)};bool null;int id;
    if(SPI_connect()!=SPI_OK_CONNECT) elog(ERROR,"VSGT SPI connect failed");
    if(SPI_execute_with_args("SELECT id FROM graph_store.edge_type WHERE name=$1",1,args,values,NULL,true,1)!=SPI_OK_SELECT || SPI_processed!=1)
        ereport(ERROR,(errmsg("VSGT edge type %s is not registered",name)));
    id=DatumGetInt32(SPI_getbinval(SPI_tuptable->vals[0],SPI_tuptable->tupdesc,1,&null));
    SPI_finish(); return id;
}

VsgtState *vsgt_begin(Jsonb *req,Datum vector,bool vector_null,Oid vector_type,Snapshot snapshot,MemoryContext owner)
{
    MemoryContext old,ctx;VsgtState *s;char *strategy,*table,*payload,*id;uint32 hsize=1;
    Oid relid;int memory_kb;int64 bytes;
    ctx=AllocSetContextCreate(owner,"VSGT request",ALLOCSET_DEFAULT_SIZES);old=MemoryContextSwitchTo(ctx);
    s=MemoryContextAllocZero(owner,sizeof(*s));s->ctx=ctx;s->snapshot=snapshot;s->started=GetCurrentTimestamp();
    if(!JB_ROOT_IS_OBJECT(req)) ereport(ERROR,(errmsg("VSGT request must be object")));
    {
        JsonbIterator *it=JsonbIteratorInit(&req->root);JsonbValue v;JsonbIteratorToken tok;
        const char *allowed[]={"strategy","table","request_id","scope","task","exclude_session",
            "exclude_artifact","anchor_set","m","k","d","ann_budget","edge_budget","state_kb",
            "seed_valid","output_valid","output_finite","seed_min_fitness","output_min_fitness","payload_column","emit_seeds","seed_topk"};
        while((tok=JsonbIteratorNext(&it,&v,true))!=WJB_DONE)if(tok==WJB_KEY)
        {
            int i;for(i=0;i<lengthof(allowed);i++)
                if(strlen(allowed[i])==v.val.string.len && !memcmp(allowed[i],v.val.string.val,v.val.string.len))break;
            if(i==lengthof(allowed))ereport(ERROR,(errmsg("unknown VSGT parameter: %.*s",v.val.string.len,v.val.string.val)));
        }
    }
    strategy=jstr(req,"strategy",NULL);table=jstr(req,"table",NULL);id=jstr(req,"request_id",NULL);
    if(!strategy || !table) ereport(ERROR,(errmsg("VSGT strategy and table required")));
    if(strcmp(strategy,"seed_cursor_f") && strcmp(strategy,"seed_cursor_b") &&
       strcmp(strategy,"graph_only_f") && strcmp(strategy,"graph_only_b") && strcmp(strategy,"vector_only"))
        ereport(ERROR,(errmsg("unsupported VSGT strategy")));
    s->st=new_stat(id,strategy);s->callback.func=reset_callback;s->callback.arg=s;
    MemoryContextRegisterResetCallback(ctx,&s->callback);
    s->graph_only=!strncmp(strategy,"graph_only",10);s->vector_only=!strcmp(strategy,"vector_only");
    if(s->graph_only && !vector_null) ereport(ERROR,(errmsg("GraphOnly requires NULL query vector")));
    s->backward=strategy[strlen(strategy)-1]=='b';
    s->scope=jstr(req,"scope",NULL);s->task=jstr(req,"task",NULL);
    if(!s->scope || !s->task) ereport(ERROR,(errmsg("VSGT scope and task required")));
    s->excluded_session=jstr(req,"exclude_session",NULL);s->excluded_artifact=jstr(req,"exclude_artifact",NULL);
    s->anchor_set=jstr(req,"anchor_set","default");
    s->m=jint(req,"m",3,1,1024);s->k=jint(req,"k",2,1,10000);s->depth=jint(req,"d",3,0,64);
    s->ann_budget=jint(req,"ann_budget",10000,1,10000000);
    s->st->requested_k=s->k;
    s->edge_budget=jint(req,"edge_budget",65536,1,10000000);
    memory_kb=jint(req,"state_kb",4096,64,65536);
    s->output_finite=jbool(req,"output_finite",true);
    s->seed_valid=jbool(req,"seed_valid",false);s->output_valid=jbool(req,"output_valid",true);
    s->emit_seeds=jbool(req,"emit_seeds",false);s->seed_pending=false;
    s->seed_topk=jbool(req,"seed_topk",false);s->ranked_n=0;s->ranked_at=0;
    s->ranked_ids=NULL;s->ranked_fit=NULL;
    s->seed_min_set=jget(req,"seed_min_fitness")!=NULL;s->seed_min=jnum(req,"seed_min_fitness",0);
    s->output_min_set=jget(req,"output_min_fitness")!=NULL;s->output_min=jnum(req,"output_min_fitness",0);
    if(!isfinite(s->seed_min)||!isfinite(s->output_min)) ereport(ERROR,(errmsg("VSGT fitness threshold must be finite")));
    relid=DatumGetObjectId(DirectFunctionCall1(regclassin,CStringGetDatum(table)));
    s->heap=table_open(relid,AccessShareLock);require_select(s->heap);
    s->kindcol=get_attnum(relid,"kind");
    if(s->kindcol>0)(void)column(s->heap,"kind",TEXTOID);
    s->idcol=column(s->heap,"id",INT8OID);s->uidcol=column(s->heap,"uid",TEXTOID);
    s->scopecol=column(s->heap,"scope_id",TEXTOID);s->taskcol=column(s->heap,"task_uid",TEXTOID);
    s->sessioncol=column(s->heap,"session_uid",TEXTOID);s->validcol=column(s->heap,"is_valid",BOOLOID);
    s->fitcol=column(s->heap,"fitness",FLOAT8OID);payload=jstr(req,"payload_column","metadata");
    s->payloadcol=column(s->heap,payload,JSONBOID);
    s->pk=primary_index(s->heap,s->idcol);
    s->probe=index_beginscan(s->heap,s->pk,snapshot,1,0);
    s->slot=table_slot_create(s->heap,NULL);
    /* Bounded queue and two open-addressed sets; no reachable-set collection. */
    s->capacity=Min((int64)(memory_kb*1024-32768)/160,s->edge_budget+s->m+1);
    s->capacity=Max(s->capacity,s->k+1);
    while(hsize<(uint32)s->capacity*2) hsize<<=1;
    bytes=(int64)s->capacity*sizeof(QueueNode)+2*(int64)hsize*sizeof(SeenEntry);
    if(bytes>memory_kb*1024-16384) ereport(ERROR,(errmsg("VSGT state_kb too small for requested k/budgets")));
    s->st->state_bytes=bytes+s->m*sizeof(int64);s->st->state_limit=memory_kb*1024;
    s->seed_ids=palloc0(s->m*sizeof(int64));
    s->scratch=AllocSetContextCreate(ctx,"VSGT scratch",ALLOCSET_SMALL_SIZES);
    s->queue=palloc0(s->capacity*sizeof(QueueNode));s->seen=palloc0(hsize*sizeof(SeenEntry));
    s->emitted=palloc0(hsize*sizeof(SeenEntry));s->hashmask=hsize-1;s->epoch=1;
    if(!s->vector_only)
    {
        int forward=edge_type_id("eg_lineage"),backward=edge_type_id("eg_child_of");
        s->edge_type=s->backward?backward:forward;
        /* Loading a trusted extension symbol initializes its rendezvous provider. */
        (void) load_external_function("$libdir/graph_store_am","gph_visits",true,NULL);
        s->api=*find_rendezvous_variable(GPH_PULL_API_NAME);
        if(!s->api || s->api->version!=1) ereport(ERROR,(errmsg("VSGT requires graph pull API v1")));
    }
    if(s->graph_only)
    {
        Oid aid=DatumGetObjectId(DirectFunctionCall1(regclassin,CStringGetDatum("public.tjs_vsgt_anchors")));
        ScanKeyData keys[3];
        s->anchors=table_open(aid,AccessShareLock);require_select(s->anchors);
        s->anchor_index=index_open(DatumGetObjectId(DirectFunctionCall1(regclassin,CStringGetDatum("public.tjs_vsgt_anchors_pkey"))),AccessShareLock);
        ScanKeyInit(&keys[0],1,BTEqualStrategyNumber,F_TEXTEQ,CStringGetTextDatum(s->scope));
        ScanKeyInit(&keys[1],2,BTEqualStrategyNumber,F_TEXTEQ,CStringGetTextDatum(s->task));
        ScanKeyInit(&keys[2],3,BTEqualStrategyNumber,F_TEXTEQ,CStringGetTextDatum(s->anchor_set));
        s->anchor_scan=index_beginscan(s->anchors,s->anchor_index,snapshot,3,0);
        index_rescan(s->anchor_scan,keys,3,NULL,0);s->anchor_slot=table_slot_create(s->anchors,NULL);
    }
    else
    {
        List *idxs=RelationGetIndexList(s->heap);ListCell *lc;Oid distproc=InvalidOid;ScanKeyData key;
        Oid hnsw=get_am_oid("hnsw",false);
        const char *iter=GetConfigOption("hnsw.iterative_scan",true,false);
        if(vector_null) ereport(ERROR,(errmsg("VSGT vector strategy requires query vector")));
        if(!iter || strcmp(iter,"relaxed_order")) ereport(ERROR,(errmsg("VSGT requires hnsw.iterative_scan=relaxed_order")));
        s->vector=datumCopy(vector,false,-1);
        foreach(lc,idxs)
        {
            Relation index=index_open(lfirst_oid(lc),AccessShareLock);
            if(index->rd_rel->relam==hnsw && index->rd_index->indisvalid &&
               index->rd_index->indnkeyatts==1 && index->rd_index->indkey.values[0]>0 &&
               index->rd_opcintype[0]==vector_type &&
               TupleDescAttr(RelationGetDescr(s->heap),index->rd_index->indkey.values[0]-1)->atttypid==vector_type &&
               RelationGetIndexPredicate(index)==NIL)
            {
                s->ann_index=index;s->veccol=index->rd_index->indkey.values[0];
                distproc=get_opcode(get_opfamily_member(index->rd_opfamily[0],index->rd_opcintype[0],index->rd_opcintype[0],1));break;
            }
            index_close(index,AccessShareLock);
        }
        list_free(idxs);
        if(!s->ann_index || !OidIsValid(distproc) || get_func_rettype(distproc)!=FLOAT8OID) ereport(ERROR,(errmsg("VSGT requires nonpartial HNSW index")));
        fmgr_info(distproc,&s->distfn);
        ScanKeyInit(&key,1,1,distproc,s->vector);
        s->ann_scan=index_beginscan(s->heap,s->ann_index,snapshot,0,1);
        index_rescan(s->ann_scan,NULL,0,&key,1);s->ann_slot=table_slot_create(s->heap,NULL);
    }
    s->st->peak_bytes=MemoryContextMemAllocated(ctx,true);
    MemoryContextSwitchTo(old);return s;
}

static bool next_anchor(VsgtState *s,int64 *vid)
{
    bool null;
    if(!s->vector_only && s->st->seeds>=s->m) return false;
    if(s->graph_only)
    {
        while(s->st->anchors<s->ann_budget && anchor_next(s))
        {
            MemoryContextReset(s->scratch);CHECK_FOR_INTERRUPTS();
            s->st->anchors++; /* Manifest source cap; no ANN work. */
            *vid=DatumGetInt64(slot_getattr(s->anchor_slot,5,&null));
            if(!null && probe_node(s,*vid) && passes(s,s->slot,true))
            {
                int i;for(i=0;i<s->st->seeds;i++)if(s->seed_ids[i]==*vid)break;
                if(i==s->st->seeds)return true;
            }
        }
        if(s->st->anchors>=s->ann_budget)strlcpy(s->st->reason,"anchor_budget",sizeof(s->st->reason));
        return false;
    }
    if(s->seed_topk)
    {
        if(!s->ranked_ids)
        {
            int cap=s->ann_budget<4096?s->ann_budget:4096,i,j;
            /* The drain loop resets s->scratch every iteration, so the buffer must
             * live in the state's own context. Allocating it in scratch freed it
             * mid-loop and the ranked ids became garbage block numbers. */
            MemoryContext prev=MemoryContextSwitchTo(s->ctx);
            s->ranked_ids=palloc(sizeof(int64)*cap);s->ranked_fit=palloc(sizeof(double)*cap);
            MemoryContextSwitchTo(prev);
            while(s->st->ann<s->ann_budget && s->ranked_n<cap)
            {
                ItemPointer tid;TimestampTz t=GetCurrentTimestamp();bool fnull;Datum fd;
                MemoryContextReset(s->scratch);CHECK_FOR_INTERRUPTS();
                s->st->vector_calls++;
                tid=index_getnext_tid(s->ann_scan,ForwardScanDirection);
                s->st->ann_ms+=millis(t);
                if(!tid) break;
                s->st->ann++;
                if(!index_fetch_heap(s->ann_scan,s->ann_slot)) continue;
                if(!passes(s,s->ann_slot,s->vector_only?false:true)) continue;
                fd=slot_getattr(s->ann_slot,s->idcol,&null);
                if(null) continue;
                s->ranked_ids[s->ranked_n]=DatumGetInt64(fd);
                fd=slot_getattr(s->ann_slot,s->fitcol,&fnull);
                s->ranked_fit[s->ranked_n]=fnull?-DBL_MAX:DatumGetFloat8(fd);
                s->ranked_n++;
            }
            /* insertion sort, descending: ranked_n is bounded by the ANN budget */
            for(i=1;i<s->ranked_n;i++)
            {
                int64 kid=s->ranked_ids[i];double kf=s->ranked_fit[i];
                for(j=i-1;j>=0 && s->ranked_fit[j]<kf;j--)
                { s->ranked_ids[j+1]=s->ranked_ids[j];s->ranked_fit[j+1]=s->ranked_fit[j]; }
                s->ranked_ids[j+1]=kid;s->ranked_fit[j+1]=kf;
            }
        }
        while(s->ranked_at<s->ranked_n)
        {
            int i;*vid=s->ranked_ids[s->ranked_at++];
            for(i=0;i<s->st->seeds;i++)if(s->seed_ids[i]==*vid)break;
            if(i==s->st->seeds) return true;
        }
        strlcpy(s->st->reason,"ann_budget",sizeof(s->st->reason));return false;
    }
    while(s->st->ann<s->ann_budget)
    {
        ItemPointer tid;TimestampTz t=GetCurrentTimestamp();
        MemoryContextReset(s->scratch);CHECK_FOR_INTERRUPTS();
        s->st->vector_calls++;
        tid=index_getnext_tid(s->ann_scan,ForwardScanDirection);
        s->st->ann_ms+=millis(t);
        if(!tid) {strlcpy(s->st->reason,"ann_end_unknown",sizeof(s->st->reason));return false;}
        s->st->ann++;
        if(!index_fetch_heap(s->ann_scan,s->ann_slot)) continue;
        *vid=DatumGetInt64(slot_getattr(s->ann_slot,s->idcol,&null));
        if(!null && passes(s,s->ann_slot,s->vector_only?false:true))
        {
            int i;for(i=0;i<s->st->seeds;i++)if(s->seed_ids[i]==*vid)break;
            if(s->vector_only || i==s->st->seeds)return true;
        }
    }
    strlcpy(s->st->reason,"ann_budget",sizeof(s->st->reason));return false;
}
static Jsonb *make_row(VsgtState *s,int64 vid,int depth,int64 parent)
{
    StringInfoData b;bool null;Datum d;TimestampTz t=GetCurrentTimestamp();
    initStringInfo(&b);
    appendStringInfo(&b,"{\"id\":%lld,\"seed_id\":%lld,\"depth\":%d,\"predecessor\":%lld,\"ordinal\":%lld,\"direction\":",
        (long long)vid,(long long)s->seed,depth,(long long)parent,(long long)s->st->emitted);
    quoted(&b,s->vector_only?"none":s->backward?"backward":"forward");
    appendStringInfoString(&b,",\"uid\":");d=slot_getattr(s->slot,s->uidcol,&null);
    if(null)appendStringInfoString(&b,"null");else quoted(&b,TextDatumGetCString(d));
    appendStringInfoString(&b,",\"payload\":");d=slot_getattr(s->slot,s->payloadcol,&null);
    if(null) appendStringInfoString(&b,"null");
    else appendStringInfoString(&b,JsonbToCString(NULL,&DatumGetJsonbP(d)->root,VARSIZE(DatumGetJsonbP(d))));
    /* A bounded parent chain is carried by the engine even through filtered nodes. */
    appendStringInfoString(&b,",\"path\":[");
    if (!s->vector_only)
    {
        int indices[65], n=0, at=s->path_index, i;
        while(at>=0 && n<65)
        {
            indices[n++]=at;
            if(s->queue[at].depth==0)break;
            for(i=0;i<at;i++)if(s->queue[i].vid==s->queue[at].parent)break;
            if(i==at)break;
            at=i;
        }
        for(i=n-1;i>=0;i--)
        { if(i<n-1)appendStringInfoChar(&b,',');appendStringInfo(&b,"%lld",(long long)s->queue[indices[i]].vid); }
    }
    appendStringInfo(&b,"],\"root_reached\":%s,\"path_complete\":%s,\"distance\":",
        s->root_status<0?"null":s->root_status?"true":"false",s->root_status==1?"true":"false");
    if(s->vector_only)
    {
        d=slot_getattr(s->slot,s->veccol,&null);
        if(!null)
        {
            double distance=DatumGetFloat8(FunctionCall2Coll(&s->distfn,InvalidOid,d,s->vector));
            if(isfinite(distance))appendStringInfo(&b,"%.17g",distance);
            else appendStringInfoString(&b,"null");
        }
        else appendStringInfoString(&b,"null");
    }
    else appendStringInfoString(&b,"null");
    appendStringInfoChar(&b,'}');s->st->hydration++;s->st->hydrate_ms+=millis(t);
    return DatumGetJsonbP(DirectFunctionCall1(jsonb_in,CStringGetDatum(b.data)));
}

Jsonb *vsgt_next(VsgtState *s)
{
    MemoryContext caller=CurrentMemoryContext;Jsonb *row=NULL;int64 vid=0,parent=0;int depth=0;
    if(s->closed) return NULL;
    MemoryContextSwitchTo(s->ctx);
    if(s->st->emitted>=s->k) {strlcpy(s->st->reason,"k",sizeof(s->st->reason));goto done;}
    for(;;)
    {
        CHECK_FOR_INTERRUPTS();
        MemoryContextReset(s->scratch);
        MemoryContextSwitchTo(s->scratch);
        if(s->vector_only)
        {
            if(!next_anchor(s,&vid)) goto done;
            if(!seen_add(s->emitted,s->hashmask,1,vid)) {s->st->dedup++;continue;}
            if(!probe_node(s,vid)) continue;
            s->seed=vid;break;
        }
        if(s->seed_pending)
        {
            s->seed_pending=false;
            /* probe_node loads the tuple into s->slot and applies the output
             * predicates; make_row reads uid/payload straight off that slot, so
             * emitting without it reads whatever the previous row left behind. */
            /* probe_node only loads the tuple; the output predicates (validity,
             * scope, task, excluded artifact/session) live in passes(). Emitting a
             * seed without them let a request return its own parent artifact. */
            if(seen_add(s->emitted,s->hashmask,1,s->seed) && probe_node(s,s->seed)
               && passes(s,s->slot,false))
            { vid=s->seed;depth=0;parent=s->seed;s->path_index=0;break; }
            s->st->dedup++;
        }
        if(s->depth==0 && !s->emit_seeds) {strlcpy(s->st->reason,"depth_zero",sizeof(s->st->reason));goto done;}
        if(!s->active_seed)
        {
            if(!next_anchor(s,&s->seed)) goto done;
            s->seed_ids[s->st->seeds]=s->seed;
            s->st->consumed_seed_ids[s->st->seeds]=s->seed;
            s->st->seeds++;s->active_seed=true;s->head=0;s->tail=1;s->current=-1;s->epoch++;
            s->queue[0]=(QueueNode){s->seed,0,0};seen_add(s->seen,s->hashmask,s->epoch,s->seed);
            if(s->emit_seeds) s->seed_pending=true;
        }
        if(!s->graph_scan)
        {
            while(s->head<s->tail && s->queue[s->head].depth>=s->depth)s->head++;
            if(s->head>=s->tail){s->active_seed=false;continue;}
            s->current=s->head++;
            MemoryContextSwitchTo(s->ctx);
            s->graph_scan=s->api->open(s->queue[s->current].vid,s->edge_type,s->snapshot);
            MemoryContextSwitchTo(s->scratch);
            s->st->graph_calls++;
        }
        if(s->st->edges>=s->edge_budget){strlcpy(s->st->reason,"edge_budget",sizeof(s->st->reason));goto done;}
        {
            TimestampTz t=GetCurrentTimestamp();int r=s->api->next(s->graph_scan,&vid);
            s->st->graph_ms+=millis(t);
            if(!r){s->api->close(s->graph_scan);s->graph_scan=NULL;continue;}
            s->st->edges++;if(r<0)continue;
        }
        /* Hard scope policy prunes cross-scope/transit, unlike validity/fitness. */
        if(!probe_node(s,vid) || !scope_ok(s,s->slot))continue;
        if(s->tail>=s->capacity){strlcpy(s->st->reason,"state_budget",sizeof(s->st->reason));goto done;}
        if(!seen_add(s->seen,s->hashmask,s->epoch,vid)){s->st->dedup++;continue;}
        depth=s->queue[s->current].depth+1;parent=s->queue[s->current].vid;
        s->path_index=s->tail;
        s->queue[s->tail++]=(QueueNode){vid,parent,depth};s->st->visited++;
        if(!passes(s,s->slot,false))continue;
        if(!seen_add(s->emitted,s->hashmask,1,vid)){s->st->dedup++;continue;}
        break;
    }
    s->root_status=-1;
    if(s->backward && s->st->edges<s->edge_budget)
    {
        void *probe;int r=0;int64 ignored;
        MemoryContextSwitchTo(s->ctx);
        probe=s->api->open(vid,s->edge_type,s->snapshot);s->st->graph_calls++;
        while(s->st->edges<s->edge_budget)
        {
            TimestampTz t=GetCurrentTimestamp();
            r=s->api->next(probe,&ignored);s->st->graph_ms+=millis(t);
            if(!r){s->root_status=1;break;}
            s->st->edges++;
            if(r==1){s->root_status=0;break;}
            CHECK_FOR_INTERRUPTS();
        }
        s->api->close(probe);
    }
    s->st->emitted++;
    if(s->st->first_row_ms<0)s->st->first_row_ms=millis(s->started);
    s->st->peak_bytes=Max(s->st->peak_bytes,(int64)MemoryContextMemAllocated(s->ctx,true));
    /* Returned payload lives in caller per-tuple context, not growing request state. */
    MemoryContextSwitchTo(caller);row=make_row(s,vid,depth,parent);
    s->st->total_ms=millis(s->started);return row;
done:
    if(!strcmp(s->st->reason,"open"))strlcpy(s->st->reason,"exhausted",sizeof(s->st->reason));
    MemoryContextSwitchTo(caller);vsgt_end(s);return NULL;
}
void vsgt_end(VsgtState *s)
{
    if(s->closed)return;
    if(!strcmp(s->st->reason,"open"))strlcpy(s->st->reason,s->st->emitted>=s->k?"k":"consumer_close",sizeof(s->st->reason));
    if(s->graph_scan){s->api->close(s->graph_scan);s->graph_scan=NULL;}
    if(s->anchor_scan)index_endscan(s->anchor_scan);
    if(s->ann_scan)index_endscan(s->ann_scan);
    if(s->probe)index_endscan(s->probe);
    if(s->slot)ExecDropSingleTupleTableSlot(s->slot);
    if(s->ann_slot)ExecDropSingleTupleTableSlot(s->ann_slot);
    if(s->anchor_slot)ExecDropSingleTupleTableSlot(s->anchor_slot);
    if(s->pk)index_close(s->pk,AccessShareLock);
    if(s->ann_index)index_close(s->ann_index,AccessShareLock);
    if(s->anchor_index)index_close(s->anchor_index,AccessShareLock);
    if(s->heap)table_close(s->heap,AccessShareLock);
    if(s->anchors)table_close(s->anchors,AccessShareLock);
    s->closed=true;s->st->active=false;s->st->total_ms=millis(s->started);
    MemoryContextDelete(s->ctx);
}
