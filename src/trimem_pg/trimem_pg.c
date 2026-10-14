/*
 * trimem_pg.c — GUCs, shared-memory mailbox and statistics, SQL entry points (ADR-0027).
 *
 * Mailbox protocol (one fixed slot array in shared memory, one LWLock, one CV):
 *   backend:  FREE -> POSTED (owner = its procno, seq = fresh), broadcast CV, wait on own latch
 *   worker:   trimem.take()     POSTED -> TAKEN, copies (model, pv, predicate, input) out
 *   worker:   trimem.complete() TAKEN  -> DONE  (verdict), SetLatch(owner) — seq must match
 *   backend:  DONE -> FREE after reading the verdict; any slot it still owns on error/exit is
 *             freed (PG_CATCH / on_shmem_exit), so a late complete() finds a seq mismatch.
 * The worker commits the verdict into trimem.sem_cache in its own transaction; the mailbox reply
 * is what the waiting query uses, so correctness never depends on cache visibility.
 */
#include "postgres.h"

#include <math.h>

#include "access/htup_details.h"
#include "access/xact.h"
#include "catalog/namespace.h"
#include "catalog/pg_type.h"
#include "commands/extension.h"
#include "common/cryptohash.h"
#include "common/hashfn.h"
#include "executor/spi.h"
#include "fmgr.h"
#include "funcapi.h"
#include "miscadmin.h"
#include "nodes/makefuncs.h"
#include "nodes/supportnodes.h"
#include "optimizer/cost.h"
#include "optimizer/optimizer.h"
#include "parser/parse_func.h"
#include "pgstat.h"
#include "storage/condition_variable.h"
#include "storage/ipc.h"
#include "storage/latch.h"
#include "storage/lwlock.h"
#include "storage/proc.h"
#include "storage/shmem.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/lsyscache.h"
#include "utils/timestamp.h"

#include "trimem.h"

PG_MODULE_MAGIC;

/* ---------------------------------------------------------------- GUCs */
double trimem_llm_cost_per_eval = 2125.0; /* 0.085 s LLM call / 4e-5 s per cost unit */
double trimem_cache_cost_per_eval = 0.5;  /* one index probe */
double trimem_llm_cost_per_batch = 2125.0; /* one LLM round trip, not amortized by batching */
double trimem_default_selectivity = 0.5;
int trimem_min_batch = 4;
int trimem_max_batch = 64;
int trimem_timeout_ms = 120000;
int trimem_slots = 1024;
int trimem_max_input_bytes = 32768;
char *trimem_model = NULL;
char *trimem_prompt_version = NULL;
bool trimem_enable = true;
int trimem_placement = 0;
char *trimem_force_placement = NULL;

static const struct config_enum_entry placement_options[] = {
    {"auto", 0, false}, {"base", 1, false}, {"top", 2, false}, {NULL, 0, false}};

/* ---------------------------------------------------------------- shared memory */
#define SLOT_FREE 0
#define SLOT_POSTED 1
#define SLOT_TAKEN 2
#define SLOT_DONE 3

#define TRIMEM_STATS_ENTRIES 512
#define TRIMEM_PRED_MAX 1024
#define TRIMEM_PV_MAX 64

typedef struct TrimemStatEntry
{
    uint64 hash; /* 0 = empty */
    char model[TRIMEM_NAME_MAX];
    char pv[TRIMEM_PV_MAX];
    char predicate[TRIMEM_PRED_MAX];
    int64 n_eval;
    int64 n_true;
    int64 n_hit;
} TrimemStatEntry;

typedef struct TrimemSlot
{
    int state;
    int owner; /* pgprocno */
    uint64 seq;
    int verdict;
    int op;       /* TRIMEM_OPC_* */
    double score; /* sem_score reply */
    int pred_len;
    int input_len;
    char model[TRIMEM_NAME_MAX];
    char pv[TRIMEM_PV_MAX];
    char data[FLEXIBLE_ARRAY_MEMBER]; /* predicate '\0' input '\0' */
} TrimemSlot;

typedef struct TrimemCtl
{
    LWLock *lock;
    ConditionVariable posted_cv;
    uint64 next_seq;
    int nslots;
    Size slot_size;
    int hint;
    TrimemStatEntry stats[TRIMEM_STATS_ENTRIES];
    /* slots follow, each slot_size bytes */
} TrimemCtl;

static TrimemCtl *ctl = NULL;
static shmem_request_hook_type prev_shmem_request = NULL;
static shmem_startup_hook_type prev_shmem_startup = NULL;
static bool exit_cb_registered = false;

static Size
slot_size(void)
{
    return MAXALIGN(offsetof(TrimemSlot, data) + trimem_max_input_bytes + 2);
}

static Size
ctl_size(void)
{
    return add_size(MAXALIGN(sizeof(TrimemCtl)), mul_size(trimem_slots, slot_size()));
}

static inline TrimemSlot *
slot_at(int i)
{
    return (TrimemSlot *) ((char *) ctl + MAXALIGN(sizeof(TrimemCtl)) + (Size) i * ctl->slot_size);
}

static void
trimem_shmem_request(void)
{
    if (prev_shmem_request)
        prev_shmem_request();
    RequestAddinShmemSpace(ctl_size());
    RequestNamedLWLockTranche("trimem", 1);
}

static void
trimem_shmem_startup(void)
{
    bool found;

    if (prev_shmem_startup)
        prev_shmem_startup();
    LWLockAcquire(AddinShmemInitLock, LW_EXCLUSIVE);
    ctl = ShmemInitStruct("trimem mailbox", ctl_size(), &found);
    if (!found)
    {
        memset(ctl, 0, ctl_size());
        ctl->lock = &(GetNamedLWLockTranche("trimem"))->lock;
        ConditionVariableInit(&ctl->posted_cv);
        ctl->next_seq = 1;
        ctl->nslots = trimem_slots;
        ctl->slot_size = slot_size();
    }
    LWLockRelease(AddinShmemInitLock);
}

bool
trimem_shmem_ready(void)
{
    return ctl != NULL;
}

static int
my_procno(void)
{
#if PG_VERSION_NUM >= 170000
    return MyProcNumber;
#else
    return MyProc->pgprocno;
#endif
}

static void
require_shmem(void)
{
    if (!ctl)
        ereport(ERROR, (errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
                        errmsg("trimem_pg must be loaded via shared_preload_libraries"),
                        errhint("Add trimem_pg to shared_preload_libraries and restart.")));
}

/* Free every slot this backend still owns (error path / process exit). */
static void
release_my_slots(void)
{
    int me;

    if (!ctl || !MyProc)
        return;
    me = my_procno();
    LWLockAcquire(ctl->lock, LW_EXCLUSIVE);
    for (int i = 0; i < ctl->nslots; i++)
    {
        TrimemSlot *s = slot_at(i);

        if (s->state != SLOT_FREE && s->owner == me)
            s->state = SLOT_FREE;
    }
    LWLockRelease(ctl->lock);
}

static void
exit_cb(int code, Datum arg)
{
    release_my_slots();
}

/* ---------------------------------------------------------------- keys */
void
trimem_key_hex(const char *op, const char *model, const char *pv, const char *predicate,
              const char *input, char *out_hex)
{
    static const char sep = '\x1f';
    static const char hexd[] = "0123456789abcdef";
    const char *parts[5] = {op, model, pv, predicate, input};
    uint8 digest[32];
    pg_cryptohash_ctx *h = pg_cryptohash_create(PG_SHA256);

    if (!h || pg_cryptohash_init(h) < 0)
        elog(ERROR, "trimem: sha256 init failed");
    for (int i = 0; i < 5; i++)
    {
        if (i > 0 && pg_cryptohash_update(h, (const uint8 *) &sep, 1) < 0)
            elog(ERROR, "trimem: sha256 update failed");
        if (pg_cryptohash_update(h, (const uint8 *) parts[i], strlen(parts[i])) < 0)
            elog(ERROR, "trimem: sha256 update failed");
    }
    if (pg_cryptohash_final(h, digest, sizeof(digest)) < 0)
        elog(ERROR, "trimem: sha256 final failed");
    pg_cryptohash_free(h);
    for (int i = 0; i < 32; i++)
    {
        out_hex[2 * i] = hexd[digest[i] >> 4];
        out_hex[2 * i + 1] = hexd[digest[i] & 0xf];
    }
    out_hex[64] = '\0';
}

/* ---------------------------------------------------------------- statistics */
static uint64
stat_hash(const char *model, const char *pv, const char *predicate)
{
    uint64 h = hash_bytes_extended((const unsigned char *) model, strlen(model), 0x7472696d656d);

    h = hash_bytes_extended((const unsigned char *) pv, strlen(pv), h);
    h = hash_bytes_extended((const unsigned char *) predicate, strlen(predicate), h);
    return h ? h : 1;
}

static TrimemStatEntry *
stat_find(const char *model, const char *pv, const char *predicate, bool create)
{
    uint64 h = stat_hash(model, pv, predicate);

    for (int probe = 0; probe < TRIMEM_STATS_ENTRIES; probe++)
    {
        TrimemStatEntry *e = &ctl->stats[(h + probe) % TRIMEM_STATS_ENTRIES];

        if (e->hash == h && strcmp(e->model, model) == 0 && strcmp(e->pv, pv) == 0 &&
            strcmp(e->predicate, predicate) == 0)
            return e;
        if (e->hash == 0)
        {
            if (!create)
                return NULL;
            e->hash = h;
            strlcpy(e->model, model, sizeof(e->model));
            strlcpy(e->pv, pv, sizeof(e->pv));
            strlcpy(e->predicate, predicate, sizeof(e->predicate));
            return e;
        }
    }
    return NULL; /* table full: counters are dropped, the trimem.stats table still applies */
}

void
trimem_stats_add(const char *model, const char *pv, const char *predicate, int64 n_eval,
                int64 n_true, int64 n_hit)
{
    TrimemStatEntry *e;

    if (!ctl || n_eval <= 0)
        return;
    LWLockAcquire(ctl->lock, LW_EXCLUSIVE);
    e = stat_find(model, pv, predicate, true);
    if (e)
    {
        e->n_eval += n_eval;
        e->n_true += n_true;
        e->n_hit += n_hit;
    }
    LWLockRelease(ctl->lock);
}

/*
 * Planner statistics = persisted trimem.stats row (flushed history) + live shared-memory
 * counters. With no evaluations the GUC defaults apply and from_stats is false.
 */
void
trimem_lookup_stats(const char *model, const char *pv, const char *predicate,
                   TrimemPredStats *out)
{
    double n_eval = 0, n_true = 0, n_hit = 0;

    out->n_cached = 0;
    if (ctl)
    {
        TrimemStatEntry *e;

        LWLockAcquire(ctl->lock, LW_SHARED);
        e = stat_find(model, pv, predicate, false);
        if (e)
        {
            n_eval += e->n_eval;
            n_true += e->n_true;
            n_hit += e->n_hit;
        }
        LWLockRelease(ctl->lock);
    }
    if (OidIsValid(get_namespace_oid("trimem", true)))
    {
        Oid argtypes[3] = {TEXTOID, TEXTOID, TEXTOID};
        Datum args[3] = {CStringGetTextDatum(model), CStringGetTextDatum(pv),
                         CStringGetTextDatum(predicate)};

        if (SPI_connect() != SPI_OK_CONNECT)
            elog(ERROR, "trimem: SPI_connect failed");
        if (SPI_execute_with_args("SELECT n_eval::float8, n_true::float8, n_hit::float8 "
                                  "FROM trimem.stats WHERE model = $1 AND prompt_version = $2 "
                                  "AND predicate = $3",
                                  3, argtypes, args, NULL, true, 1) == SPI_OK_SELECT &&
            SPI_processed == 1)
        {
            bool isnull;
            HeapTuple tup = SPI_tuptable->vals[0];
            TupleDesc td = SPI_tuptable->tupdesc;

            n_eval += DatumGetFloat8(SPI_getbinval(tup, td, 1, &isnull));
            n_true += DatumGetFloat8(SPI_getbinval(tup, td, 2, &isnull));
            n_hit += DatumGetFloat8(SPI_getbinval(tup, td, 3, &isnull));
        }
        if (SPI_execute_with_args("SELECT count(*)::float8 FROM trimem.sem_cache "
                                  "WHERE model = $1 AND prompt_version = $2 AND predicate = $3",
                                  3, argtypes, args, NULL, true, 1) == SPI_OK_SELECT &&
            SPI_processed == 1)
        {
            bool isnull;

            out->n_cached = DatumGetFloat8(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));
        }
        SPI_finish();
    }
    out->n_eval = n_eval;
    out->n_true = n_true;
    out->n_hit = n_hit;
    out->from_stats = n_eval >= 1;
    out->sel = out->from_stats ? n_true / n_eval : trimem_default_selectivity;
    out->sel = Min(Max(out->sel, 0.001), 1.0);
    out->hit = out->from_stats ? Min(Max(n_hit / n_eval, 0.0), 1.0) : 0.0;
    out->c_eff = trimem_llm_cost_per_eval * (1.0 - out->hit) + trimem_cache_cost_per_eval * out->hit;
}

/* ---------------------------------------------------------------- mailbox: backend side */
static int
post_one(TrimemRequest *r, const char *model, const char *pv, int me)
{
    for (int k = 0; k < ctl->nslots; k++)
    {
        int i = (ctl->hint + k) % ctl->nslots;
        TrimemSlot *s = slot_at(i);

        if (s->state == SLOT_FREE)
        {
            Size pl = strlen(r->predicate), il = strlen(r->input);

            s->state = SLOT_POSTED;
            s->owner = me;
            s->seq = ctl->next_seq++;
            s->verdict = -2;
            s->op = r->op;
            s->score = 0;
            s->pred_len = (int) pl;
            s->input_len = (int) il;
            strlcpy(s->model, model, sizeof(s->model));
            strlcpy(s->pv, pv, sizeof(s->pv));
            memcpy(s->data, r->predicate, pl + 1);
            memcpy(s->data + pl + 1, r->input, il + 1);
            r->slot = i;
            r->seq = s->seq;
            ctl->hint = (i + 1) % ctl->nslots;
            return 1;
        }
    }
    return 0;
}

/*
 * Resolve every request through the worker. Blocks (interruptibly, bounded by
 * trimem.timeout_ms) until all verdicts are back; fails closed on timeout or worker error.
 */
void
trimem_mailbox_run(const char *model, const char *pv, TrimemRequest *reqs, int n, double *wait_ms)
{
    int me = my_procno();
    TimestampTz start = GetCurrentTimestamp();

    require_shmem();
    for (int i = 0; i < n; i++)
    {
        Size need = strlen(reqs[i].predicate) + strlen(reqs[i].input) + 2;

        if (need > (Size) trimem_max_input_bytes + 2)
            ereport(ERROR, (errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
                            errmsg("trimem: semantic input of %zu bytes exceeds trimem.max_input_bytes (%d)",
                                   need, trimem_max_input_bytes)));
        if (strlen(model) >= TRIMEM_NAME_MAX || strlen(pv) >= TRIMEM_PV_MAX)
            ereport(ERROR, (errmsg("trimem: model or prompt_version name too long")));
        reqs[i].slot = -1;
        reqs[i].verdict = -2;
    }
    if (!exit_cb_registered)
    {
        on_shmem_exit(exit_cb, (Datum) 0);
        exit_cb_registered = true;
    }

    PG_TRY();
    {
        int remaining = n;

        for (;;)
        {
            int posted = 0;
            long elapsed;

            ResetLatch(MyLatch);
            LWLockAcquire(ctl->lock, LW_EXCLUSIVE);
            for (int i = 0; i < n; i++)
            {
                TrimemRequest *r = &reqs[i];

                if (r->verdict != -2)
                    continue;
                if (r->slot >= 0)
                {
                    TrimemSlot *s = slot_at(r->slot);

                    if (s->owner == me && s->seq == r->seq && s->state == SLOT_DONE)
                    {
                        r->verdict = s->verdict;
                        r->score = s->score;
                        s->state = SLOT_FREE;
                        remaining--;
                    }
                }
                else
                    posted += post_one(r, model, pv, me);
            }
            LWLockRelease(ctl->lock);
            if (posted > 0)
                ConditionVariableBroadcast(&ctl->posted_cv);
            if (remaining == 0)
                break;

            elapsed = (long) ((GetCurrentTimestamp() - start) / 1000);
            if (elapsed >= trimem_timeout_ms)
                ereport(ERROR, (errcode(ERRCODE_QUERY_CANCELED),
                                errmsg("trimem: %d semantic verdict(s) not returned within trimem.timeout_ms (%d ms)",
                                       remaining, trimem_timeout_ms),
                                errhint("Start the worker: python -m trimem.worker --dsn ...")));
            (void) WaitLatch(MyLatch, WL_LATCH_SET | WL_TIMEOUT | WL_EXIT_ON_PM_DEATH,
                             Min(1000L, trimem_timeout_ms - elapsed), PG_WAIT_EXTENSION);
            CHECK_FOR_INTERRUPTS();
        }
    }
    PG_CATCH();
    {
        release_my_slots();
        PG_RE_THROW();
    }
    PG_END_TRY();

    for (int i = 0; i < n; i++)
        if (reqs[i].verdict == -1)
            ereport(ERROR, (errcode(ERRCODE_EXTERNAL_ROUTINE_EXCEPTION),
                            errmsg("trimem: worker reported a failed semantic evaluation"),
                            errdetail("predicate: %s", reqs[i].predicate)));
    *wait_ms += (double) (GetCurrentTimestamp() - start) / 1000.0;
}

/* ---------------------------------------------------------------- SQL: worker side */
PG_FUNCTION_INFO_V1(trimem_take);
Datum
trimem_take(PG_FUNCTION_ARGS)
{
    int maxn = PG_GETARG_INT32(0);
    int timeout = PG_GETARG_INT32(1);
    ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
    TimestampTz start = GetCurrentTimestamp();
    int got = 0;

    require_shmem();
    InitMaterializedSRF(fcinfo, 0);
    for (;;)
    {
        LWLockAcquire(ctl->lock, LW_EXCLUSIVE);
        for (int i = 0; i < ctl->nslots && got < maxn; i++)
        {
            TrimemSlot *s = slot_at(i);
            Datum v[7];
            bool nulls[7] = {false};

            if (s->state != SLOT_POSTED)
                continue;
            s->state = SLOT_TAKEN;
            v[0] = Int32GetDatum(i);
            v[1] = Int64GetDatum((int64) s->seq);
            v[2] = CStringGetTextDatum(s->op == TRIMEM_OPC_SCORE     ? TRIMEM_OP_SCORE
                                       : s->op == TRIMEM_OPC_EXTRACT ? TRIMEM_OP_EXTRACT
                                                                     : TRIMEM_OP_FILTER);
            v[3] = CStringGetTextDatum(s->model);
            v[4] = CStringGetTextDatum(s->pv);
            v[5] = PointerGetDatum(cstring_to_text_with_len(s->data, s->pred_len));
            v[6] = PointerGetDatum(cstring_to_text_with_len(s->data + s->pred_len + 1, s->input_len));
            tuplestore_putvalues(rsinfo->setResult, rsinfo->setDesc, v, nulls);
            got++;
        }
        LWLockRelease(ctl->lock);
        if (got > 0)
            break;
        {
            long elapsed = (long) ((GetCurrentTimestamp() - start) / 1000);

            if (elapsed >= timeout)
                break;
            (void) ConditionVariableTimedSleep(&ctl->posted_cv, Min(1000L, timeout - elapsed),
                                               PG_WAIT_EXTENSION);
            CHECK_FOR_INTERRUPTS();
        }
    }
    ConditionVariableCancelSleep();
    return (Datum) 0;
}

PG_FUNCTION_INFO_V1(trimem_complete);
Datum
trimem_complete(PG_FUNCTION_ARGS)
{
    ArrayType *a_slot = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType *a_seq = PG_GETARG_ARRAYTYPE_P(1);
    ArrayType *a_ver = PG_GETARG_ARRAYTYPE_P(2);
    Datum *d_slot, *d_seq, *d_ver;
    bool *n_slot, *n_seq, *n_ver;
    int k1, k2, k3, done = 0;
    int *owners;
    int nowners = 0;

    require_shmem();
    deconstruct_array_builtin(a_slot, INT4OID, &d_slot, &n_slot, &k1);
    deconstruct_array(a_seq, INT8OID, 8, FLOAT8PASSBYVAL, TYPALIGN_DOUBLE, &d_seq, &n_seq, &k2);
    deconstruct_array(a_ver, BOOLOID, 1, true, TYPALIGN_CHAR, &d_ver, &n_ver, &k3);
    if (k1 != k2 || k1 != k3)
        ereport(ERROR, (errmsg("trimem.complete: array lengths differ")));
    owners = palloc(sizeof(int) * Max(k1, 1));
    LWLockAcquire(ctl->lock, LW_EXCLUSIVE);
    for (int j = 0; j < k1; j++)
    {
        int i;
        TrimemSlot *s;

        if (n_slot[j] || n_seq[j])
            continue;
        i = DatumGetInt32(d_slot[j]);
        if (i < 0 || i >= ctl->nslots)
            continue;
        s = slot_at(i);
        if (s->state != SLOT_TAKEN || s->seq != (uint64) DatumGetInt64(d_seq[j]))
            continue; /* the owner gave up (error/cancel) and the slot was recycled */
        s->verdict = n_ver[j] ? -1 : (DatumGetBool(d_ver[j]) ? 1 : 0);
        s->state = SLOT_DONE;
        owners[nowners++] = s->owner;
        done++;
    }
    LWLockRelease(ctl->lock);
    for (int j = 0; j < nowners; j++)
        SetLatch(&GetPGProcByNumber(owners[j])->procLatch);
    PG_RETURN_INT32(done);
}

/* Worker reply for sem_score requests: scores in [0, 1]; NULL = failed evaluation. */
PG_FUNCTION_INFO_V1(trimem_complete_scores);
Datum
trimem_complete_scores(PG_FUNCTION_ARGS)
{
    ArrayType *a_slot = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType *a_seq = PG_GETARG_ARRAYTYPE_P(1);
    ArrayType *a_sc = PG_GETARG_ARRAYTYPE_P(2);
    Datum *d_slot, *d_seq, *d_sc;
    bool *n_slot, *n_seq, *n_sc;
    int k1, k2, k3, done = 0, nowners = 0;
    int *owners;

    require_shmem();
    deconstruct_array_builtin(a_slot, INT4OID, &d_slot, &n_slot, &k1);
    deconstruct_array(a_seq, INT8OID, 8, FLOAT8PASSBYVAL, TYPALIGN_DOUBLE, &d_seq, &n_seq, &k2);
    deconstruct_array(a_sc, FLOAT8OID, 8, FLOAT8PASSBYVAL, TYPALIGN_DOUBLE, &d_sc, &n_sc, &k3);
    if (k1 != k2 || k1 != k3)
        ereport(ERROR, (errmsg("trimem.complete_scores: array lengths differ")));
    owners = palloc(sizeof(int) * Max(k1, 1));
    LWLockAcquire(ctl->lock, LW_EXCLUSIVE);
    for (int j = 0; j < k1; j++)
    {
        int i;
        TrimemSlot *s;

        if (n_slot[j] || n_seq[j])
            continue;
        i = DatumGetInt32(d_slot[j]);
        if (i < 0 || i >= ctl->nslots)
            continue;
        s = slot_at(i);
        if (s->state != SLOT_TAKEN || s->seq != (uint64) DatumGetInt64(d_seq[j]) ||
            s->op != TRIMEM_OPC_SCORE)
            continue;
        s->verdict = n_sc[j] ? -1 : 1;
        s->score = n_sc[j] ? 0 : Min(Max(DatumGetFloat8(d_sc[j]), 0.0), 1.0);
        s->state = SLOT_DONE;
        owners[nowners++] = s->owner;
        done++;
    }
    LWLockRelease(ctl->lock);
    for (int j = 0; j < nowners; j++)
        SetLatch(&GetPGProcByNumber(owners[j])->procLatch);
    PG_RETURN_INT32(done);
}

/*
 * Worker reply for sem_extract requests: completion only. The worker has already committed the
 * labels into trimem.sem_cache.extracted (a missing row = no parsable answer -> NULL label); the
 * waiting SemExtract node reads them back. A failed evaluation is reported through
 * trimem.complete(..., NULL), which makes the query fail closed.
 */
PG_FUNCTION_INFO_V1(trimem_complete_extracts);
Datum
trimem_complete_extracts(PG_FUNCTION_ARGS)
{
    ArrayType *a_slot = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType *a_seq = PG_GETARG_ARRAYTYPE_P(1);
    Datum *d_slot, *d_seq;
    bool *n_slot, *n_seq;
    int k1, k2, done = 0, nowners = 0;
    int *owners;

    require_shmem();
    deconstruct_array_builtin(a_slot, INT4OID, &d_slot, &n_slot, &k1);
    deconstruct_array(a_seq, INT8OID, 8, FLOAT8PASSBYVAL, TYPALIGN_DOUBLE, &d_seq, &n_seq, &k2);
    if (k1 != k2)
        ereport(ERROR, (errmsg("trimem.complete_extracts: array lengths differ")));
    owners = palloc(sizeof(int) * Max(k1, 1));
    LWLockAcquire(ctl->lock, LW_EXCLUSIVE);
    for (int j = 0; j < k1; j++)
    {
        int i;
        TrimemSlot *s;

        if (n_slot[j] || n_seq[j])
            continue;
        i = DatumGetInt32(d_slot[j]);
        if (i < 0 || i >= ctl->nslots)
            continue;
        s = slot_at(i);
        if (s->state != SLOT_TAKEN || s->seq != (uint64) DatumGetInt64(d_seq[j]) ||
            s->op != TRIMEM_OPC_EXTRACT)
            continue;
        s->verdict = 1;
        s->state = SLOT_DONE;
        owners[nowners++] = s->owner;
        done++;
    }
    LWLockRelease(ctl->lock);
    for (int j = 0; j < nowners; j++)
        SetLatch(&GetPGProcByNumber(owners[j])->procLatch);
    PG_RETURN_INT32(done);
}

/* Merge the shared-memory counters into trimem.stats and zero them (called by the worker). */
PG_FUNCTION_INFO_V1(trimem_flush_stats);
Datum
trimem_flush_stats(PG_FUNCTION_ARGS)
{
    TrimemStatEntry *snap;
    int n = 0, flushed = 0;

    require_shmem();
    snap = palloc(sizeof(TrimemStatEntry) * TRIMEM_STATS_ENTRIES);
    LWLockAcquire(ctl->lock, LW_EXCLUSIVE);
    for (int i = 0; i < TRIMEM_STATS_ENTRIES; i++)
        if (ctl->stats[i].hash != 0 && ctl->stats[i].n_eval > 0)
        {
            snap[n++] = ctl->stats[i];
            ctl->stats[i].n_eval = ctl->stats[i].n_true = ctl->stats[i].n_hit = 0;
        }
    LWLockRelease(ctl->lock);
    if (n == 0)
        PG_RETURN_INT32(0);
    if (SPI_connect() != SPI_OK_CONNECT)
        elog(ERROR, "trimem: SPI_connect failed");
    for (int i = 0; i < n; i++)
    {
        Oid t[6] = {TEXTOID, TEXTOID, TEXTOID, INT8OID, INT8OID, INT8OID};
        Datum v[6] = {CStringGetTextDatum(snap[i].model), CStringGetTextDatum(snap[i].pv),
                      CStringGetTextDatum(snap[i].predicate), Int64GetDatum(snap[i].n_eval),
                      Int64GetDatum(snap[i].n_true), Int64GetDatum(snap[i].n_hit)};

        if (SPI_execute_with_args(
                "INSERT INTO trimem.stats AS s (model, prompt_version, predicate, n_eval, n_true, n_hit) "
                "VALUES ($1, $2, $3, $4, $5, $6) ON CONFLICT (model, prompt_version, predicate) DO UPDATE "
                "SET n_eval = s.n_eval + EXCLUDED.n_eval, n_true = s.n_true + EXCLUDED.n_true, "
                "n_hit = s.n_hit + EXCLUDED.n_hit, updated_at = now()",
                6, t, v, NULL, false, 0) == SPI_OK_INSERT)
            flushed++;
    }
    SPI_finish();
    PG_RETURN_INT32(flushed);
}

/* Live counters + mailbox occupancy, for observability. */
PG_FUNCTION_INFO_V1(trimem_status);
Datum
trimem_status(PG_FUNCTION_ARGS)
{
    ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
    int counts[4] = {0};

    require_shmem();
    InitMaterializedSRF(fcinfo, 0);
    LWLockAcquire(ctl->lock, LW_SHARED);
    for (int i = 0; i < ctl->nslots; i++)
        counts[slot_at(i)->state]++;
    LWLockRelease(ctl->lock);
    {
        const char *names[4] = {"free", "posted", "taken", "done"};

        for (int i = 0; i < 4; i++)
        {
            Datum v[2] = {CStringGetTextDatum(names[i]), Int32GetDatum(counts[i])};
            bool nulls[2] = {false, false};

            tuplestore_putvalues(rsinfo->setResult, rsinfo->setDesc, v, nulls);
        }
    }
    return (Datum) 0;
}

/* Clear the live counters (tests). */
PG_FUNCTION_INFO_V1(trimem_reset_stats);
Datum
trimem_reset_stats(PG_FUNCTION_ARGS)
{
    require_shmem();
    LWLockAcquire(ctl->lock, LW_EXCLUSIVE);
    memset(ctl->stats, 0, sizeof(ctl->stats));
    LWLockRelease(ctl->lock);
    PG_RETURN_VOID();
}

/* ---------------------------------------------------------------- SQL: key + marker */
PG_FUNCTION_INFO_V1(trimem_key);
Datum
trimem_key(PG_FUNCTION_ARGS)
{
    char hex[65];

    trimem_key_hex(text_to_cstring(PG_GETARG_TEXT_PP(0)), text_to_cstring(PG_GETARG_TEXT_PP(1)),
                  text_to_cstring(PG_GETARG_TEXT_PP(2)), text_to_cstring(PG_GETARG_TEXT_PP(3)),
                  text_to_cstring(PG_GETARG_TEXT_PP(4)), hex);
    PG_RETURN_TEXT_P(cstring_to_text(hex));
}

/* Evaluated outside a SemFilter node -> fail closed (no per-row LLM call, no guess). */
PG_FUNCTION_INFO_V1(trimem_sem_filter);
Datum
trimem_sem_filter(PG_FUNCTION_ARGS)
{
    ereport(ERROR, (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                    errmsg("trimem.sem_filter must be planned as a TriMEM SemFilter node"),
                    errhint("Use it as a top-level WHERE conjunct over a single relation "
                            "(v1: base relations only; trimem.enable must be on).")));
    PG_RETURN_NULL();
}

Oid
trimem_sem_filter_oid(void)
{
    Oid argtypes[2] = {TEXTOID, TEXTOID};

    if (!OidIsValid(get_namespace_oid("trimem", true)))
        return InvalidOid;
    return LookupFuncName(list_make2(makeString("trimem"), makeString("sem_filter")), 2, argtypes,
                          true);
}

Oid
trimem_sem_filter3_oid(void)
{
    Oid argtypes[3] = {TEXTOID, TEXTOID, BOOLOID};

    if (!OidIsValid(get_namespace_oid("trimem", true)))
        return InvalidOid;
    return LookupFuncName(list_make2(makeString("trimem"), makeString("sem_filter")), 3, argtypes,
                          true);
}

Oid
trimem_sem_score_oid(void)
{
    Oid argtypes[2] = {TEXTOID, TEXTOID};

    if (!OidIsValid(get_namespace_oid("trimem", true)))
        return InvalidOid;
    return LookupFuncName(list_make2(makeString("trimem"), makeString("sem_score")), 2, argtypes,
                          true);
}

Oid
trimem_sem_extract_oid(void)
{
    Oid argtypes[2] = {TEXTOID, TEXTOID};

    if (!OidIsValid(get_namespace_oid("trimem", true)))
        return InvalidOid;
    return LookupFuncName(list_make2(makeString("trimem"), makeString("sem_extract")), 2, argtypes,
                          true);
}

/*
 * sem_score is evaluated in the projection right above (or inside) the SemScore node that
 * computed the batch. The node registers its score table here before emitting each row; the
 * function only looks the value up, so it never calls the LLM itself.
 */
static void *active_scores = NULL;
static char active_model[TRIMEM_NAME_MAX];
static char active_pv[TRIMEM_PV_MAX];

static void *active_extracts = NULL;
static char active_extract_model[TRIMEM_NAME_MAX];
static char active_extract_pv[TRIMEM_PV_MAX];

static void
score_xact_cb(XactEvent event, void *arg)
{
    if (event == XACT_EVENT_ABORT || event == XACT_EVENT_COMMIT ||
        event == XACT_EVENT_PARALLEL_ABORT || event == XACT_EVENT_PARALLEL_COMMIT)
    {
        active_scores = NULL; /* never let a freed query-context table outlive its query */
        active_extracts = NULL;
        trimem_extract_forget_live();
    }
}

void
trimem_extract_set_active(void *table, const char *model, const char *pv)
{
    active_extracts = table;
    strlcpy(active_extract_model, model, sizeof(active_extract_model));
    strlcpy(active_extract_pv, pv, sizeof(active_extract_pv));
}

/* Like sem_score: only looks up the label its SemExtract node computed; never calls the LLM. */
PG_FUNCTION_INFO_V1(trimem_sem_extract);
Datum
trimem_sem_extract(PG_FUNCTION_ARGS)
{
    char key[65];
    char *label = NULL;
    char *question = text_to_cstring(PG_GETARG_TEXT_PP(0));
    char *input = text_to_cstring(PG_GETARG_TEXT_PP(1));
    bool found = false;

    if (active_extracts != NULL)
    {
        trimem_key_hex(TRIMEM_OP_EXTRACT, active_extract_model, active_extract_pv, question, input, key);
        found = trimem_extract_lookup(active_extracts, key, &label);
    }
    if (!found)
        found = trimem_extract_lookup_live(question, input, &label); /* nested SemExtract nodes */
    if (!found && active_extracts == NULL && !trimem_extract_has_live())
        ereport(ERROR, (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                        errmsg("trimem.sem_extract must be planned under a TriMEM SemExtract node"),
                        errhint("Use it in the select list, GROUP BY / ORDER BY, or a WHERE conjunct over "
                                "one relation's columns (not in a join condition).")));
    if (!found)
        ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR),
                        errmsg("trimem.sem_extract: no label computed for this input by the SemExtract node")));
    if (label == NULL)
        PG_RETURN_NULL();
    PG_RETURN_TEXT_P(cstring_to_text(label));
}

PG_FUNCTION_INFO_V1(trimem_sem_extract_support);
Datum
trimem_sem_extract_support(PG_FUNCTION_ARGS)
{
    Node *rawreq = (Node *) PG_GETARG_POINTER(0);

    if (IsA(rawreq, SupportRequestCost))
    {
        SupportRequestCost *req = (SupportRequestCost *) rawreq;

        req->startup = 0;
        req->per_tuple = cpu_operator_cost; /* the LLM work is costed by the SemExtract node */
        PG_RETURN_POINTER(req);
    }
    PG_RETURN_POINTER(NULL);
}

void
trimem_score_set_active(void *table, const char *model, const char *pv)
{
    active_scores = table;
    strlcpy(active_model, model, sizeof(active_model));
    strlcpy(active_pv, pv, sizeof(active_pv));
}

PG_FUNCTION_INFO_V1(trimem_sem_score);
Datum
trimem_sem_score(PG_FUNCTION_ARGS)
{
    char key[65];
    double score;

    if (active_scores == NULL)
        ereport(ERROR, (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                        errmsg("trimem.sem_score must be planned under a TriMEM SemScore node"),
                        errhint("Use it in the select list or ORDER BY of a query over the relation(s) it reads.")));
    trimem_key_hex(TRIMEM_OP_SCORE, active_model, active_pv, text_to_cstring(PG_GETARG_TEXT_PP(0)),
                   text_to_cstring(PG_GETARG_TEXT_PP(1)), key);
    if (!trimem_score_lookup(active_scores, key, &score))
        ereport(ERROR, (errcode(ERRCODE_INTERNAL_ERROR),
                        errmsg("trimem.sem_score: no score computed for this input by the SemScore node")));
    PG_RETURN_FLOAT8(score);
}

PG_FUNCTION_INFO_V1(trimem_sem_score_support);
Datum
trimem_sem_score_support(PG_FUNCTION_ARGS)
{
    Node *rawreq = (Node *) PG_GETARG_POINTER(0);

    if (IsA(rawreq, SupportRequestCost))
    {
        SupportRequestCost *req = (SupportRequestCost *) rawreq;

        req->startup = 0;
        req->per_tuple = cpu_operator_cost; /* the LLM work is costed by the SemScore node */
        PG_RETURN_POINTER(req);
    }
    PG_RETURN_POINTER(NULL);
}

/*
 * Planner support: the core planner's row estimate for the relation includes the semantic
 * selectivity. Per-tuple cost is reported as one operator call because the qual is lifted out of
 * the scan and costed by the SemFilter node (counting it here too would double-charge).
 */
PG_FUNCTION_INFO_V1(trimem_sem_filter_support);
Datum
trimem_sem_filter_support(PG_FUNCTION_ARGS)
{
    Node *rawreq = (Node *) PG_GETARG_POINTER(0);

    if (IsA(rawreq, SupportRequestCost))
    {
        SupportRequestCost *req = (SupportRequestCost *) rawreq;

        req->startup = 0;
        req->per_tuple = cpu_operator_cost;
        PG_RETURN_POINTER(req);
    }
    if (IsA(rawreq, SupportRequestSelectivity))
    {
        SupportRequestSelectivity *req = (SupportRequestSelectivity *) rawreq;
        Node *pred = linitial(req->args);
        TrimemPredStats st;

        if (req->is_join || !IsA(pred, Const) || ((Const *) pred)->constisnull)
            PG_RETURN_POINTER(NULL);
        trimem_lookup_stats(trimem_model, trimem_prompt_version,
                           TextDatumGetCString(((Const *) pred)->constvalue), &st);
        req->selectivity = st.sel;
        PG_RETURN_POINTER(req);
    }
    PG_RETURN_POINTER(NULL);
}

/* ---------------------------------------------------------------- init */
void
_PG_init(void)
{
    DefineCustomRealVariable("trimem.llm_cost_per_eval",
                             "Planner cost of one uncached semantic evaluation (PG cost units).",
                             NULL, &trimem_llm_cost_per_eval, 2125.0, 0.0, 1e12, PGC_USERSET, 0,
                             NULL, NULL, NULL);
    DefineCustomRealVariable("trimem.cache_cost_per_eval",
                             "Planner cost of one cached semantic evaluation (PG cost units).",
                             NULL, &trimem_cache_cost_per_eval, 0.5, 0.0, 1e12, PGC_USERSET, 0,
                             NULL, NULL, NULL);
    DefineCustomRealVariable("trimem.llm_cost_per_batch",
                             "Planner cost of one LLM round trip per judged batch (PG cost units).",
                             NULL, &trimem_llm_cost_per_batch, 2125.0, 0.0, 1e12, PGC_USERSET, 0,
                             NULL, NULL, NULL);
    DefineCustomRealVariable("trimem.default_selectivity",
                             "Selectivity of a semantic predicate with no statistics.", NULL,
                             &trimem_default_selectivity, 0.5, 0.001, 1.0, PGC_USERSET, 0, NULL,
                             NULL, NULL);
    DefineCustomIntVariable("trimem.min_batch", "Minimum rows judged per batch.", NULL,
                            &trimem_min_batch, 4, 1, 4096, PGC_USERSET, 0, NULL, NULL, NULL);
    DefineCustomIntVariable("trimem.max_batch", "Maximum rows judged per batch (buffer bound).",
                            NULL, &trimem_max_batch, 64, 1, 4096, PGC_USERSET, 0, NULL, NULL, NULL);
    DefineCustomIntVariable("trimem.timeout_ms", "Max wait for the worker per batch.", NULL,
                            &trimem_timeout_ms, 120000, 10, INT_MAX, PGC_USERSET, GUC_UNIT_MS, NULL,
                            NULL, NULL);
    DefineCustomStringVariable("trimem.model", "Model id (part of the cache key).", NULL,
                               &trimem_model, "Qwen/Qwen3.8-27B-FP8", PGC_USERSET, 0, NULL, NULL,
                               NULL);
    DefineCustomStringVariable("trimem.prompt_version", "Prompt version (part of the cache key).",
                               NULL, &trimem_prompt_version, "trimem-filter-v1", PGC_USERSET, 0,
                               NULL, NULL, NULL);
    DefineCustomBoolVariable("trimem.enable", "Plan trimem.sem_filter as a SemFilter node.", NULL,
                             &trimem_enable, true, PGC_USERSET, 0, NULL, NULL, NULL);
    DefineCustomEnumVariable("trimem.placement",
                             "SemFilter placement in multi-relation queries: auto (cost-based), base, top.",
                             NULL, &trimem_placement, 0, placement_options, PGC_USERSET, 0, NULL,
                             NULL, NULL);
    DefineCustomStringVariable("trimem.force_placement",
                               "Per-relation SemFilter placement, e.g. 's=base,d=top' (experiments).",
                               NULL, &trimem_force_placement, "", PGC_USERSET, 0, NULL, NULL, NULL);
    if (process_shared_preload_libraries_in_progress)
    {
        DefineCustomIntVariable("trimem.slots", "Mailbox slots in shared memory.", NULL,
                                &trimem_slots, 1024, 16, 65536, PGC_POSTMASTER, 0, NULL, NULL,
                                NULL);
        DefineCustomIntVariable("trimem.max_input_bytes",
                                "Max predicate+input bytes per mailbox slot.", NULL,
                                &trimem_max_input_bytes, 32768, 1024, 1048576, PGC_POSTMASTER, 0,
                                NULL, NULL, NULL);
        prev_shmem_request = shmem_request_hook;
        shmem_request_hook = trimem_shmem_request;
        prev_shmem_startup = shmem_startup_hook;
        shmem_startup_hook = trimem_shmem_startup;
    }
    MarkGUCPrefixReserved("trimem");
    trimem_exec_init();
    trimem_graph_init();
    trimem_score_init();
    trimem_extract_init();
    RegisterXactCallback(score_xact_cb, NULL);
    trimem_path_init();
}
