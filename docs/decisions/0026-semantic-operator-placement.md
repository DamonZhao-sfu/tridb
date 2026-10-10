# ADR-0026: placement of LLM semantic operators (split: planner-visible, LLM out of process)

Status: **proposed**, 2026-10-10. It is a draft and has not been implemented.
The evidence and phasing are in `docs/figure1_unified_semantic_plan_v0.1.0.md` §2b and §4.

## Decision

The semantic operators are `sem_extract`, `sem_match`, `sem_conflict`, `sem_filter`, `sem_rank`, `sem_topk` and `sem_summarize`. They are split across three places.

1. **Logical operator and cost in the planner.**
   - Add a new, additive `src/planner/join_order_sem.c` and its SQL function `tridb_place_sem(...)`.
   - It orders predicate legs by the expensive-predicate rank `(1−sel)/(c·(1−hit))`.
   - It is called at the lowering site, following the ADR-0011 Option B precedent.
   - The frozen `LegStats` / `TriJoinOrder` stay untouched, as `join_order_cost.c` does.
2. **The LLM call stays out of process, in Python.**
   - This is the same boundary as the embedder and as `OpenAIChatExtractor` (`bench/agent_memory/gem/strategies/llm_mediated.py:534`).
   - No HTTP or libcurl in the backend.
3. **State lives in PG.**
   - `gem_sem_cache` (the LLM result cache) and `gem_pending` (the deferred-derivation set) are ordinary heap tables.
   - They are under the one WAL and transaction manager, with no sidecar.
4. **The streaming semantic filter reads from a server-side portal over the `tjs_vsgt` CustomScan.**
   - It calls `FETCH n`, evaluates the batch through the cache, and `CLOSE`s the portal once it has enough survivors.
   - The engine stream keeps TR-1. The batch buffer is bounded and budgeted (ADR-0020).
5. **`sem_rank` and `sem_topk` re-rank a bounded post-stream prefix.**
   - They are not a second rank authority inside the operator (ADR-0007).
   - `sem_filter` is an additional filter leg, which is the case ADR-0007 already admits.

## Rejected

- **Python only.** It gives the planner no visibility, and ADR-0023 forbids Python plan orchestration.
- **LLM calls inside a C operator** (a `passes()` predicate, or after `ExecQual` in `tjs_open`). This has no precedent:
  - the backend blocks on network I/O;
  - it would add a libcurl dependency on both stock and fork builds;
  - it breaks the byte-parity schedule gates (ADR-0023);
  - it raises the same process-model concerns as ADR-0017.

## Amends

- **ADR-0012.** Its "no external re-ranker / second round-trip" rule is relaxed for semantic operators only. Those calls send bounded text, return a verdict, and are cache-suppressed. V/G/R data never leaves the process.

## Consequences

- `src/planner` must join the stock-PG build (`scripts/pg17_graph_test.sh`, `STOCK_TESTS`) so that per-PR CI covers it.
- Planner C code is unbuilt-here until that CI run is green. The 1M-scale latency sign-off remains GX10-gated.
- Determinism comes from the cache: a replay with the LLM disabled must reproduce state byte for byte.
- Cache misses are disclosed in the receipt, never silently filled.
