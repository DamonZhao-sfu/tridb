# E0 TriDB-native plan-space result v0.3.0

## Material Passport

- Experiment: E0 / Optimization Opportunity from `haikaidocs/trevillisPlan.md`
- Date: 2026-08-18
- Status: **design-complete result obtained and independently repeated**
- Backend: `tridb_live`, one PostgreSQL process containing pgvector, relational storage,
  and the native adjacency-list graph access method
- Frozen contract: `configs/e0/plan_space_v0.3.yaml`
- Scope: STARK-PRIME 40 queries + OpenEvolve seed 42 10 queries
- Design size: 50 queries, 3,450 query-plan cells, three repetitions, **10,350
  observations per run**
- Platform: stock PostgreSQL 16.14, x86_64, 8 KiB `BLCKSZ`
- Claim boundary: **this is not a GX10, ARM64, CUDA, 32 KiB-fork, or 128 GB live
  benchmark sign-off**

## Result

The E0 stop condition does **not** trigger.  Across all 50 queries, the primary run's
median quality-equivalent plan spread is **6.05x**, above the predeclared 2x threshold.
An independent complete run gives **6.29x**.  The result supports the narrow O
(Optimization Opportunity) claim on this workload: plan choice materially affects latency,
and no single plan shape wins every query.

| Metric | Primary | Independent repeat |
|---|---:|---:|
| Observations | 10,350 | 10,350 |
| Errors | 0 | 0 |
| Censored traversals | 0 | 0 |
| Combined median spread | **6.051x** | **6.289x** |
| Combined p95 spread | 179.989x | 175.960x |
| Combined max spread | 230.780x | 239.548x |
| Wall time | 87.879 s | 89.016 s |
| Winner shape agreement | — | 48/50 queries (96%) |

The combined-median ratio between runs is 1.039x.  At the query level, the run/run
spread ratio has median 1.065x and p95 1.484x.  Individual plan p50 latency is noisier
(ratio median 1.051x, p95 1.472x), but the aggregate conclusion and 2x decision are stable.

## Dataset results and the honest negative slice

| Dataset | Queries | Median spread | p95 | Max | Winning shapes |
|---|---:|---:|---:|---:|---|
| STARK-PRIME | 40 | **18.216x** | 180.375x | 230.780x | traverse 34 / filter 3 / vector 3 |
| OpenEvolve | 10 | **1.935x** | 2.144x | 2.192x | traverse 6 / filter 4 / vector 0 |
| Combined | 50 | **6.051x** | 179.989x | 230.780x | traverse 40 / filter 7 / vector 3 |

OpenEvolve by itself is below the 2x stop threshold.  The O claim is therefore not uniform
across workloads: the strong combined result is driven by the dense, heterogeneous typed
graph and relational selectivities in STARK-PRIME.  This is a useful scope boundary, not a
result to hide.

The hard-coded default plan is quality-equivalent on only 16/50 queries.  On those queries,
its median suboptimality is 2.261x and p95 is 8.327x.  It is not assigned a latency ratio on
the other 34 queries because comparing a lower-quality default with the best-quality plans
would violate the frozen quality-equivalence rule.

## What was implemented to support E0

The public v1 query surface remains the one canonical `graph_query()` template.  E0 uses a
benchmark-only internal surface declared by `experiments/e0/tridb_schema.sql`:

1. `gph_traverse_bounded_multi` extends the native pull iterator to a set of allowed edge
   types and up to 64 anchors.  It implements union or required-from-each-anchor intersection,
   shares one explicit edge-step budget, and preserves `gs_open` / `gs_getnext` / `gs_close`.
2. `tjs_e0_open` executes vector-first, filter-first, and traverse-first inside the same
   PostgreSQL backend.  It retains only bounded candidate/top-k state; it never builds a
   relational edge table or ships an intermediate across a process boundary.
3. `tools/e0/load_tridb.py` creates one database per corpus, loads vectors and properties into
   PostgreSQL, inserts topology through typed batched native-AM calls, builds HNSW/B-tree
   indexes, and fails unless all counts and audited reachability checks reconcile.
4. `tridb_live` integrates the operator with the checkpointed E0 runner and records wall
   latency, native edge work, censoring, termination, stage timing, result quality, and zero
   cross-store round trips/bytes.

`traverse_first` is an experiment-only diagnostic plan, not a proposed second public query
surface.  Its graph input remains pull-based and budgeted, and its ranking state is `O(k)`;
as with any exact top-k over an unknown traversal stream, final ranked output is available
after its selected traversal has been consumed.  No full path or reached-row intermediate is
materialized.

## Frozen semantics

- `edge_types` is an allowed union applied at every hop.
- Reach is one through the plan's hop bound and excludes all anchor vertices.
- Multiple anchors use union unless `required_from_each_anchor` requests intersection.
- Ranking uses cosine distance.
- Relational predicates are target entity type, generation bounds, and OpenEvolve
  `same_parent`.
- `pre` belongs to filter-first; `during`/`post` belong to vector-first and traverse-first.
- Quality equivalence is best Hit@1 within 0.02, then best MRR within 0.02.  Hit@5 and
  Recall@20 are recorded only.

For traverse-first, `during` and `post` are semantically distinct placements but both can be
pipelined as each reached id is consumed; no full reached set is introduced merely to make
the labels look different.  Their observed timing differences should therefore be read as
repeat/noise evidence, not as a claimed physical materialization contrast.

## Load and correctness gates

| Gate | STARK-PRIME | OpenEvolve |
|---|---:|---:|
| Relational rows / vectors | 129,375 / 129,375 | 31 / 31 |
| Native vertices | 129,375 | 31 |
| Native directed arcs | 8,100,498 | 60 |
| Native visible arcs | 8,100,498 | 60 |
| Edge types | 18 | 2 (forward + reverse) |
| Audited queries with missing reachable answers | 0/40 | 0/10 |
| Verification traversals censored | 0 | 0 |

Additional parity checks:

- STARK real-data smoke: **18/18** plan results exactly match the Parquet oracle.
- OpenEvolve real-data smoke: **57/60** exactly match.  The only three differences are
  vector-first default-plan HNSW boundary choices on duplicated/equal-distance code
  embeddings; exact native traversal/filter shapes match, and quality is recorded rather
  than silently asserted equal.
- The stock-PG benchmark-internal SQL suite passes multi-type union, multi-anchor
  intersection, every plan shape, termination, and deterministic distance/id tie-breaking.
- Repository Python suite: **876 passed, 29 skipped**.
- Focused E0 lint and format checks pass.  Repository-wide `make lint` is not green because
  the unrelated untracked `FluctlightDB/` tree contributes 45 pre-existing Ruff errors; those
  files were not modified for E0.

The Docker/Podman stock-PG 17 harness could not be invoked on this host because neither CLI
is installed.  Both C modules compile locally against PostgreSQL 16, the internal SQL suite
runs against PostgreSQL 16, and the result must not be promoted to a PG17 or GX10 sign-off.

## System and measurement passport

- CPU: Intel Xeon w7-3445, 20 cores / 40 threads, one NUMA node
- RAM: 125 GiB visible to the host
- OS/kernel: Ubuntu x86_64, Linux 6.17.0-1012-oem
- PostgreSQL: 16.14, 64-bit, 8 KiB blocks
- Extensions: pgvector 0.8.0, graph_store_am 0.2.0, tjs_pg 0.2.0
- Query settings: `hnsw.iterative_scan=relaxed_order`, `hnsw.ef_search=100`, dense native
  graph open enabled; graph budgets 50,000,000 edges (STARK) and 1,000,000 (OpenEvolve)
- Execution: one client, sequential queries, warm persistent database, one default-plan
  warmup per query, three adjacent timed repetitions per plan
- Primary config SHA256:
  `6bc7097416deabc06b264a2e86bb807b0fd8737e4be58a26b2fa4fe3b279f1d8`
- Primary observations SHA256:
  `27c9b5864524370367950878b57019f5ed6ad7d3377c79a3789187e9cd0e8ad2`
- Repeat observations SHA256:
  `5067c4737ded1c7f8c8335bd8bed580cd8516a09d441e767bba95589c9e9f20c`
- Git base revision recorded by the manifests:
  `6badc988ad24eed1d0a320875f46d7841417d298`

## Interpretation boundary

E0 establishes an optimization opportunity; it does **not** establish that TriDB is faster
than Polyglot-Tuned, that a learned/cost-based planner selects the winning plan, or that the
result survives concurrency.  Those are E1/E3/E4 questions.  The present run is also a warm,
single-host, single-client experiment, so it is not an SLA or tail-latency claim.

The high p95/max spreads are real measured cells but should not become the headline without
query-level inspection: the stable headline is the predeclared median and the cross-run
agreement.  STARK includes 10 manually audited and 30 independently oracle-checked
auto-derived queries; their evidence levels remain separate in `metrics_query.csv`.

## Reproduction and artifacts

```bash
# Builds were performed locally against stock PostgreSQL 16 before these steps.
make e0-tridb-load
make e0-tridb-smoke
make e0-tridb-run

# Independent repeat (a distinct output directory is mandatory because checkpoints
# pin the config hash and never overwrite observations).
.venv/bin/python -m experiments.e0.plan_spread.runner \
  --backend tridb_live --config configs/e0/plan_space_v0.3.yaml \
  --output-dir results/e0/plan_space/tridb_live_v0.3_repeat
.venv/bin/python -m experiments.e0.plan_spread.analyze \
  --config configs/e0/plan_space_v0.3.yaml \
  --raw results/e0/plan_space/tridb_live_v0.3_repeat/observations.jsonl \
  --output-dir results/e0/plan_space/tridb_live_v0.3_repeat
```

Primary artifacts are under `results/e0/plan_space/tridb_live_v0.3/`:

- `observations.jsonl`: all 10,350 raw observations
- `run_manifest.json`: input hashes, environment, completeness, and run duration
- `metrics_plan.csv`: 3,450 per-plan reductions
- `metrics_query.csv`: 50 per-query quality-equivalent spreads
- `summary.json`: frozen stop-condition decision
- `repeatability.json`: cross-run stability receipt
- `figures/`: plan-spread ECDF, default suboptimality, winner shape, and hop breakdown

The independent raw run is under `results/e0/plan_space/tridb_live_v0.3_repeat/`.  Load
receipts are `data/e0/tridb_load_stark_v0.3.json` and
`data/e0/tridb_load_v0.3.json` (OpenEvolve).
