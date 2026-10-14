/*
 * trimem.h — TriMEM semantic-predicate CustomScan (ADR-0027).
 *
 * trimem.sem_filter(predicate, input) is lifted out of a base relation's quals by a
 * set_rel_pathlist_hook and executed by "Custom Scan (TriMEM SemFilter)". Cache misses go to an
 * out-of-process worker through a shared-memory mailbox; no LLM call happens in the backend.
 */
#ifndef TRIMEM_H
#define TRIMEM_H

#include "postgres.h"
#include "nodes/pg_list.h"
#include "nodes/primnodes.h"
#include "nodes/pathnodes.h"

#define TRIMEM_OP_FILTER "sem_filter"
#define TRIMEM_NAME_MAX 128

/* GUCs (trimem_pg.c) */
extern double trimem_llm_cost_per_eval;
extern double trimem_cache_cost_per_eval;
extern double trimem_llm_cost_per_batch;
extern double trimem_default_selectivity;
extern int trimem_min_batch;
extern int trimem_max_batch;
extern int trimem_timeout_ms;
extern int trimem_slots;
extern int trimem_max_input_bytes;
extern char *trimem_model;
extern char *trimem_prompt_version;
extern bool trimem_enable;
extern int trimem_placement; /* 0 auto, 1 base, 2 top */
extern char *trimem_force_placement;

/* Planner statistics for one predicate: shared-memory counters + the trimem.stats table. */
typedef struct TrimemPredStats
{
    double n_eval;
    double n_true;
    double n_hit;
    double n_cached; /* verdicts already in trimem.sem_cache for this predicate */
    double sel;      /* estimate used by the planner */
    double hit;      /* estimated cache-hit rate */
    double c_eff;    /* effective per-row cost, PG cost units */
    bool from_stats; /* false -> GUC defaults */
} TrimemPredStats;

/* trimem_pg.c */
extern Oid trimem_sem_filter_oid(void);
extern Oid trimem_sem_filter3_oid(void);
extern Oid trimem_sem_score_oid(void);
extern Oid trimem_sem_extract_oid(void);
/* SemScore -> trimem.sem_score lookup (set by the executing SemScore node) */
extern void trimem_score_set_active(void *table, const char *model, const char *pv);
extern bool trimem_score_lookup(void *table, const char *key, double *score);
extern void trimem_lookup_stats(const char *model, const char *pv, const char *predicate,
                               TrimemPredStats *out);
extern void trimem_stats_add(const char *model, const char *pv, const char *predicate,
                            int64 n_eval, int64 n_true, int64 n_hit);
extern void trimem_key_hex(const char *op, const char *model, const char *pv,
                          const char *predicate, const char *input, char *out_hex /* 65 */);

/* Mailbox (trimem_pg.c). A request is one (predicate, input) whose verdict is not cached. */
typedef struct TrimemRequest
{
    const char *predicate;
    const char *input;
    int op;        /* TRIMEM_OPC_FILTER or TRIMEM_OPC_SCORE */
    int slot;      /* assigned slot, -1 = not yet posted */
    uint64 seq;
    int verdict;   /* -2 pending, -1 worker error, 0/1 (filter) or 1 = score present */
    double score;  /* sem_score result in [0, 1] */
} TrimemRequest;

#define TRIMEM_OPC_FILTER 0
#define TRIMEM_OPC_SCORE 1
#define TRIMEM_OPC_EXTRACT 2
#define TRIMEM_OP_SCORE "sem_score"
#define TRIMEM_OP_EXTRACT "sem_extract"

/* SemExtract -> trimem.sem_extract lookup (set by the executing SemExtract node) */
extern void trimem_extract_set_active(void *table, const char *model, const char *pv);
extern bool trimem_extract_lookup(void *table, const char *key, char **label);
extern bool trimem_extract_lookup_live(const char *question, const char *input, char **label);
extern void trimem_extract_forget_live(void);
extern bool trimem_extract_has_live(void);

extern bool trimem_shmem_ready(void);
extern void trimem_mailbox_run(const char *model, const char *pv, TrimemRequest *reqs, int n,
                              double *wait_ms);

/* trimem_path.c / trimem_exec.c */
extern void trimem_path_init(void);
extern void trimem_exec_init(void);
extern void trimem_graph_init(void);
extern void trimem_score_init(void);
extern void trimem_wrap_scores(PlannerInfo *root, RelOptInfo *rel);
extern RelOptInfo *trimem_score_base_rel(PlannerInfo *root, List *initial_rels);
extern List *trimem_score_paths(PlannerInfo *root, RelOptInfo *rel, const char *placement);
extern const char *trimem_score_decision(void);
extern bool trimem_expand_paths(PlannerInfo *root, RelOptInfo *rel, RangeTblEntry *rte);
extern struct CustomScanMethods trimem_scan_methods;

/* trimem_extract.c: SemExtract planning (placement is enumerated by trimem_path.c) */
extern void trimem_extract_init(void);
extern void trimem_extract_lift(PlannerInfo *root, RelOptInfo *rel);
extern bool trimem_extract_present(PlannerInfo *root);
extern bool trimem_extract_top_ok(PlannerInfo *root);
extern RelOptInfo *trimem_extract_base_rel(PlannerInfo *root, List *initial_rels);
extern double trimem_extract_qual_sel(PlannerInfo *root, RelOptInfo *rel);
extern List *trimem_extract_paths(PlannerInfo *root, RelOptInfo *rel, const char *placement);
extern bool trimem_wrap_extracts(PlannerInfo *root, RelOptInfo *rel, const char *placement);
extern void trimem_extract_check_joins(PlannerInfo *root);
extern void *trimem_extract_planner_enter(void);
extern void trimem_extract_planner_leave(void *saved);
extern void trimem_extract_planner_check(void);

#endif /* TRIMEM_H */
