# TriMEM SemFilter CustomScan — plan (v0.1.0)

Date 2026-10-11. Branch `feat/trimem-semscan` (from `feat/figure1-semantic-ops`).
Decision: ADR-0027 (amends ADR-0026). Default model: `Qwen/Qwen3.8-27B-FP8` (vLLM 0.23).

**TriMEM** is the agent-memory system built on TriDB: tri-modal (V/G/R) storage plus
planner-controlled LLM semantic operators on both the ingest and the retrieve path. It is
a PG extension `trimem_pg` (C, `src/trimem_pg/`) plus a Python package `trimem/` (out-of-process
LLM worker, cache, data/workload code). It replaces the Python-orchestrated tail of Figure 1
(`bench/agent_memory/semantic/stream.py`) with a real plan node.

## 1. Goal and done-when

Make a semantic predicate a first-class node of the PostgreSQL plan:

```sql
EXPLAIN SELECT result->>'uid'
FROM tjs_vsgt($req::jsonb, $q::vector)
WHERE trimem.sem_filter('transferable idea for improving the parent', result->>'outcome')
LIMIT 5;
```

```
Limit  (cost=… rows=5)
  ->  Custom Scan (TriMEM SemFilter)  (cost=1.00..9531.20 rows=10 …)
        Predicate: transferable idea for improving the parent
        Model: Qwen/Qwen3.8-27B-FP8   Prompt Version: trimem-filter-v1
        Est. Selectivity: 0.400 (stats n=1286)   Est. Cache Hit: 0.31
        Cost Per Eval: 2125.00 (miss) / 0.50 (hit)   Batch: 13
        ->  Custom Scan (TriDB VSGT)  (cost=1.00..100.00 rows=25 …)
```

Done when:
1. `EXPLAIN` shows `Custom Scan (TriMEM SemFilter)` above its input, with the semantic
   predicate removed from the input's `Filter:`.
2. The node **participates in cost estimation**: its startup/total cost and row estimate come
   from the cost model in §3; changing the cost GUC or the cache/selectivity statistics changes
   the costs (and, where alternatives exist, the plan) — shown by the regression test.
3. Several semantic predicates on one relation are stacked in rank order
   `(1−sel)/c_eff` (the `tridb_place_sem` rule), cheapest-per-rejection first.
4. `EXPLAIN ANALYZE` reports per-node LLM calls, cache hits, rows removed, batches and wait ms.
5. Execution is streaming (TR-1): bounded batch buffer, early termination under `LIMIT`.
6. The LLM call stays out of process (ADR-0026 boundary): no HTTP/libcurl in the backend.
7. Regression SQL test + pytest pass; `make lint` / `make test` pass.

## 2. Architecture

```
 backend (PG16)                                         Python worker (trimem.worker)
 ┌─────────────────────────────────────┐                ┌──────────────────────────────┐
 │ planner: set_rel_pathlist_hook      │                │ loop:                        │
 │   wrap base-rel paths in SemFilter  │                │   SELECT trimem.take(n, ms)   │──┐
 │   cost = §3, stats from trimem.stats │                │   → batch of (slot, prompt,  │  │
 │ executor: SemFilter.Next()          │   shared-mem   │      input) requests          │  │
 │   pull ≤B child rows (bounded)      │   mailbox      │   cache-check / LLM (vLLM)   │  │
 │   key = sha256(op,model,pv,p,x)     │ ◄────────────► │   INSERT trimem.sem_cache     │  │
 │   hit: trimem.sem_cache index probe  │  (slots +      │     (own txn, committed)     │  │
 │   miss: post slots, WaitLatch       │   CV + latch)  │   SELECT trimem.complete(...) │◄─┘
 │   emit survivors in input order     │                │     → slot DONE, SetLatch    │
 └─────────────────────────────────────┘                └──────────────────────────────┘
```

Why a shared-memory mailbox and not a request table: rows the backend writes inside the
query's transaction are invisible to the worker until commit, and `NOTIFY` is also delivered
at commit. The mailbox is the only same-process-group channel that works mid-query without a
second transaction manager. The worker writes the durable cache in its own transaction, so a
replay with the LLM disabled reads the same verdicts (ADR-0026 determinism rule).

### 2.1 Extension `trimem_pg` (C, stock PG16/17; `shared_preload_libraries` required)

| file | content |
|---|---|
| `trimem_pg.c` | `_PG_init`: GUCs, shmem request/startup hooks, path hook registration |
| `trimem_shm.c/.h` | mailbox: `TRIMEM_SLOTS` fixed slots {state FREE/POSTED/TAKEN/DONE, owner procno, seq, verdict, prompt/input in-slot ≤ `trimem.max_input_bytes`}, LWLock, ConditionVariable for the worker's wait, owner latch for the reply |
| `trimem_path.c` | planner hook + cost model + `PlanCustomPath` |
| `trimem_exec.c` | SemFilter executor (Begin/Next/End/ReScan/Explain) |
| `trimem_pg--0.1.0.sql` | schema `trimem`: `sem_filter(text,text)` marker function (+ support fn), `sem_cache`, `stats`, `take()`, `complete()`, `key()` |

**SQL surface** (no new language, ADR rule 4): a boolean function used as a `WHERE` qual.
`trimem.sem_filter(predicate text, input text) RETURNS boolean` — executed directly (outside a
SemFilter node) it fails closed, like `tjs_vsgt`'s FunctionScan fallback. Its support function
answers `SupportRequestCost` and `SupportRequestSelectivity`, so the core planner's own row
estimate for the relation already includes the semantic selectivity.

**Planner hook** (`set_rel_pathlist_hook`, chained after VSGT's):
1. Find `RestrictInfo`s in `rel->baserestrictinfo` whose clause is a top-level `trimem.sem_filter`
   call (verified by OID + extension membership, not by name) with Vars of this rel only.
2. Remove them from `baserestrictinfo` (so the input scan never evaluates them) and order them
   by descending rank `(1−sel_i)/c_eff_i` (§3).
3. For every surviving path in `rel->pathlist` build a chain of SemFilter `CustomPath`s (one per
   predicate) whose `custom_paths` is the input path; input path rows are rescaled back to the
   pre-filter estimate. Partial paths are dropped (the function is `PARALLEL UNSAFE`).
4. v1 scope: base relations (heap scans, index scans, the VSGT function RTE). Join-level
   placement (pull-up above a join, Hellerstein) is v2 via `set_join_pathlist_hook`.

**Executor** — per `Next()`:
1. If the output queue is empty, pull up to `B` child tuples (B from `tridb_sem_batch`
   rule, §3; bounded by `trimem.max_batch`), evaluate the input expression, compute keys.
2. Probe `trimem.sem_cache` by key (SPI, prepared plan, index on PK) for the whole batch.
3. Post misses to free mailbox slots, `WaitLatch` with `CHECK_FOR_INTERRUPTS` and
   `trimem.timeout_ms`; on timeout → `ERROR` with hint (worker not running), never a guess.
4. Emit the batch's survivors in child order; repeat. End/ReScan releases owned slots.
   Memory: per-batch context reset after each batch (lesson from the VSGT killedItems bug:
   no index/SPI state allocated in the per-batch context).

### 2.2 Python package `trimem/`

| module | content |
|---|---|
| `trimem/keys.py` | `sem_key(op, model, prompt_version, predicate, input)` = sha256 hex of `op ␟ model ␟ pv ␟ predicate ␟ input` (UTF-8, `\x1f` separators) — byte-identical to the C key |
| `trimem/llm.py` | OpenAI-compatible async client (default `Qwen/Qwen3.8-27B-FP8`, `enable_thinking=False`, temp 0, seed 0, JSON verdict), `MockLLM` |
| `trimem/worker.py` | `python -m trimem.worker --dsn … --model …`: take → cache → LLM → insert cache + stats → complete |
| `trimem/schema.py` | installs/validates the extension; reads `trimem.stats` |
| `trimem/data/` | (phase N4) EvoTrace loader + GEM adapter re-created here, not imported from `bench/` |

### 2.3 Cache and statistics

* `trimem.sem_cache(key text PK, op, model, prompt_version, predicate, verdict bool,
  prompt_tokens, completion_tokens, seconds, created_at)` — written only by the worker.
* `trimem.stats(op, model, prompt_version, predicate_hash, n_eval, n_true, n_hit,
  sum_seconds)` — maintained by the worker; read by the planner (SPI, once per planning per
  predicate). Missing stats → GUC defaults (`trimem.default_selectivity` 0.5, hit 0).

## 3. Cost model (PG cost units)

For predicate *i* on an input with N rows:

```
c_miss  = trimem.llm_cost_per_eval   (default 2125 = 0.085 s / 4e-5 s per cpu_operator_cost unit)
c_hit   = trimem.cache_cost_per_eval (default 0.5: one index probe)
h_i     = n_hit / n_eval            (stats, else 0)
c_eff_i = c_miss·(1−h_i) + c_hit·h_i
sel_i   = n_true / n_eval           (stats, else trimem.default_selectivity)
rank_i  = (1−sel_i) / c_eff_i       (descending; same rule and tie-break as tridb_place_sem)

rows_out    = N · sel_i
startup     = child.startup + B · c_eff_i            (first batch must be judged before row 1)
total       = child.total   + N · c_eff_i + N · cpu_tuple_cost
B           = clamp(ceil(k_hint / sel_i), trimem.min_batch, trimem.max_batch)
              k_hint = root->limit_tuples if a LIMIT is known, else trimem.max_batch
```

`startup` matters: under `LIMIT`, PG's `compare_path_costs` uses fractional cost, so a
SemFilter over a cheap-startup input (VSGT stream, index scan) beats one over a sort/hash input
— the "participates in cost estimation" evidence beyond the node's own numbers.

## 4. Phases

| phase | deliverable | verify |
|---|---|---|
| N0 | name, branch, this plan, ADR-0027 | — |
| N1 | `trimem_pg`: SQL file, marker + support fn, path hook, cost model, EXPLAIN (no execution yet) | `EXPLAIN` shows the node and costs; GUC change moves costs |
| N2 | shmem mailbox, executor, cache probe, `take`/`complete` | SQL test with a mock worker; `EXPLAIN ANALYZE` counters; LIMIT early stop |
| N3 | `trimem/` worker + Qwen3.8 client + stats | pytest (mock); live run against vLLM |
| N4 | demo on EvoTrace (fig1 DB): VSGT + SemFilter plans, real LLM | plans + counters saved under `results/trimem_v0.1.0/` |
| N5 | GEM/EvoTrace data code in `trimem/data/` routed through PlannedIngest / RetrieveOperator | separate plan |

## 5. Risks

* **Backend waits on an external worker.** Bounded by `trimem.timeout_ms`, interruptible,
  fails closed. Documented in ADR-0027 as the accepted cost of a planner-visible node.
* **Mailbox capacity / input size.** Fixed slots × `max_input_bytes` in shmem; inputs over the
  cap fail closed with the size in the error (no silent truncation).
* **Snapshot vs cache.** Cache rows committed by the worker after the query's snapshot are not
  visible to the in-query SPI probe; the mailbox reply carries the verdict, so correctness does
  not depend on cache visibility.
* **GPU.** Both GPUs are occupied by the long-running gemma vLLM server; serving Qwen3.8-27B-FP8
  (≈29 GB weights) needs that server stopped or GPU headroom (user decision).

## 6. Addendum A (2026-10-11): target plan shape = paper Figure 1(b)

The user's target is one PG plan mixing V, G, R and semantic operators as in Fig. 1(b)
(`ANN_k(e_q) ∪ Lookup(refs) → Expand(typed edges, 1 hop, fanout F) → Filter(outcome, scope,
policy) → sem_filter → Rank → Assemble_B`). VSGT fuses V+G+R into one opaque node, so it cannot
show this. Required beyond §4:

| phase | deliverable |
|---|---|
| N1b | **Join-level SemFilter placement.** Semantic quals are lifted from base rels; in a `join_search_hook` the planner costs (a) SemFilter on the base rel (inner side of a nested loop ⇒ batch degenerates to 1 row per rescan) vs (b) SemFilter above the final scan/join rel (batches over joined rows, after all cheap filters). Run (b) with base-rel rows rescaled to the pre-filter count; run (a) in a GEQO-style saved/restored `join_rel_list`; keep the cheaper. The planner guard (lifted vs planned) stays. |
| N1c | **`Custom Scan (TriMEM Expand)`**: native graph-AM typed 1-hop expansion with fanout cap F, used as `LATERAL trimem.expand(seed.vid, 'type', F)`; streaming per outer row (no FunctionScan tuplestore), parameterized path. |
| N1d | **Seeds as standard nodes**: V = `Limit → Index Scan using hnsw` (pgvector `ORDER BY <=> LIMIT k`), R lookup = `Index Scan … = ANY(refs)`, union = `Append`. |
| N1e | **Rank / Assemble**: `Limit(cap)` before `Sort` so Rank's input is bounded (ADR-0007: re-rank a bounded prefix); token budget via a streaming running-sum window. `sem_rank` as a bounded node is v2. |
| N1f | **resolve P (strong mode)** as an InitPlan-style `Custom Scan (TriMEM Resolve)` that posts pending derivations through the same mailbox — v2. |

Done-when for the addendum: `EXPLAIN` of the EvoTrace retrieval query shows the Fig. 1(b) tree
with every operator as its own node, SemFilter above the join with its cost in the totals, and a
GUC/statistics change that flips SemFilter placement (base ↔ above join) or the join method.

## 7. Status (2026-10-11)

| phase | status | evidence |
|---|---|---|
| N1 SemFilter node + cost model | done | `tests/test_trimem_plan.py` (14 pass) |
| N1b cost-based placement (base vs above join, `join_search_hook`, 2 trial searches + winner re-run) | done | placement flips with the plan: top for Fig. 1(b), base when the join multiplies rows |
| N1c `TriMEM Expand` (native pull API, per-type degree stats via `trimem.analyze_graph()`) | done | rows from stats, not SRF default 1000 |
| N1d seeds as standard nodes (HNSW Index Scan ∪ pk lookup via Append) | done (plain SQL) | `test_fig1b_plan_shape` |
| N1e bounded Rank input | open | Sort above SemFilter is fed ≤ cap only when the query has `LIMIT cap` |
| N2 mailbox + executor + cache | done | `tests/test_trimem_exec.py` (10 pass, mock worker) |
| N3 worker + Qwen3.8 | done | real Qwen3.8-27B-FP8 smoke: 12 judged, correct verdicts, 2nd run from cache |
| N4 EvoTrace demo | open | |
| N1f resolve-P node, `sem_rank` | v2 | |

Cost model additions beyond §3: `trimem.llm_cost_per_batch` (one LLM round trip per judged batch
with ≥1 miss; `p_trip = 1 − hit^B`), so a SemFilter inside a nested loop pays one round trip per
rescan. The Expand relation's `tuples` is the edge-set size so Memoize/join ndistinct of `dst` is
not clamped to 1.
