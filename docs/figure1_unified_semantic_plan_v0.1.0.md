# Figure 1 execution plan: TriDB + GEM → unified V/G/R + semantic plan (v0.1.0)

Status: **plan only**. This document changes no code. Date: 2026-10-10, branch `experiment`.
Target: Figure 1 of `paper/build/main.pdf` (`paper/figures/fig_overview.tex`), with operator definitions in
`paper/sections/02_ingest_definition.tex` (ID), `03_retrieval_definition.tex` (RD) and `04_update_landscape.tex` (UL).

Abbreviations:
- G = `bench/agent_memory/gem/`, S = `G/strategies/`, E = `bench/agent_memory/gem_eg/`.
- IF = `docs/agent_memory_gem_interface_v0.1.0.md`, ST = `docs/agent_memory_gem_implementation_status_v0.1.0.md`.
- Nn = a negative search listed in Appendix A.

Every "exists" claim cites `path:line`, and every "missing" claim cites a negative search (Appendix A).

---

## 1. Summary

**Q1: Does GEM already cover Ingest / Retrieve / Update?**
Yes, but only in one fixed physical shape:
- **Ingest is always eager.** T, C and δ are fused inside `IngestOperator.ingest` (G/ingest.py:475) and run in one PG transaction (G/store.py:383-418).
- **Retrieve is a single shot.** Seed, Expand and θR run fused in one `tjs_open` call (G/retrieve.py:339-469). Rank is the engine order. Assemble has no budget.
- **Update (revise / forget) runs only when a caller invokes it.** It is rule-based and calls no LLM.

Missing entirely:
- the pending set P, deferred and on-read derivations, and strong/stale reads (N3, N4, N13);
- principal-based θpol (N8, N9);
- every retrieval and update semantic op (θsem, sem_rank, sem_conflict, sem_summarize) (N6);
- an LLM result cache (N5);
- any cost or selectivity estimate, and any logical plan object (N1, N2).

The only LLM calls are the two Ingest call sites (S/llm_mediated.py:312, S/agentic.py:166).

Two code bugs block the Figure 1 invariants and must be fixed first:
- duplicate merge re-points edges only in the relational `gem_edge` mirror, never in the native graph AM (G/revise.py:223-240, N14);
- policy-triggered maintenance is dead code, because `PolicyEngine.trigger` is never called (N12).

**Q2: Where should semantic operators live?**
In option **(c), split** — confirmed from the evidence, not assumed:
- A **logical** semantic operator with a cost and selectivity estimate becomes planner-visible through an *additive* decision function in `src/planner/`, following the `join_order_cost.c` precedent (src/planner/join_order_cost.c:4-7, 49).
- The LLM call stays **out of process** in Python, as every LLM call in the repo is today (no HTTP in `src/`, N-E1).
- The **LLM cache and P are ordinary heap tables** in the same database, so they share one WAL and one transaction manager.
- The streaming semantic filter pulls bounded batches from the native CustomScan `tjs_vsgt` through a server-side portal (`DECLARE … CURSOR` / `FETCH n`). It closes the portal once it has enough survivors, so TR-1 holds end to end.

Why not the alternatives:
- (a) Python alone gives the planner no visibility, and ADR-0023 forbids Python plan orchestration.
- (b) In-backend C HTTP has no precedent. It blocks the TR-1 loop on network latency, and it breaks the byte-parity gates (ADR-0023 gate 2, `test/tjs_filter_probe_test.sql`).

The recommendation is drafted as ADR-0026 (`docs/decisions/0026-semantic-operator-placement.md`, status *Proposed*).

---

## 2a. Coverage matrix (Q1)

### Ingest_π(x,S)→⟨S',r⟩

| element | exists? | where | LLM? | Python / engine | gap to close |
|---|---|---|---|---|---|
| State S=(R,V,G,P) | partial | R: G/schema.sql:27,51,141,164. V: HNSW G/schema.sql:96-97. G: native AM via `gph_insert_edge` G/store.py:538, plus the `gem_edge` mirror G/schema.sql:123 | – | engine storage | **P absent** (N4). The closest analogue is the `metadata.needs_revision` flag (G/ingest.py:465-468) |
| Input x (principal, outcome, refs) | partial | `InteractionEvent` G/types.py:190 | – | Python | No first-class `principal`, `outcome` or `refs`; they can only go in `metadata` |
| Atomic commit + policy | yes | G/ingest.py:475-572. Txn envelope G/store.py:383-418. Writer lock G/store.py:238. Postcondition G/store.py:404-405 | LLM runs **inside** the txn (G/ingest.py:504) | Python, 1 PG txn | Holding a txn open across LLM latency → P2 moves LLM work before the txn or into P |
| Receipt r | partial | `IngestResult` G/types.py:326, `gem_transition` G/schema.sql:164 | – | Python | No pending list, no cache-hit counts |
| T deterministic | yes | S/deterministic.py:71-123 | no | Python | – |
| T LLM-mediated (≈ sem_extract) | yes | S/llm_mediated.py:229-287, call at :312, pinned prompt :45-46,57-80 | **yes** | Python | It is conditioned on C (S/llm_mediated.py:243-244). See D1 |
| T agentic | yes | S/agentic.py:107-200, call at :166 | **yes** | Python | Staged writes are invisible to its read tools, and `search_memory` ignores `query` (S/agentic.py:225-254, G/store.py:680-686). See D8 |
| C vector top-k | partial | `find_similar` G/store.py:717-728 (k=8 via S/llm_mediated.py:55) | no | SQL (pgvector) | Only the LLM-mediated strategy uses it, and it reads *current* state, not a snapshot (D2) |
| C graph from refs | **no** | – (N-G1) | – | – | Add a `gph_traverse_typed` candidate read |
| C relational | partial | Title lookup G/ingest.py:362-366. Current-field check S/llm_mediated.py:471-476 | no | SQL | No task, time-window or outcome predicates |
| δ new / link | yes | S/llm_mediated.py:347-372, 399-433 → `store.link` G/store.py:514-556 | LLM | Python + AM | No rule-based link (EXG similar_to / fixed_by); those exist only in E/store.py:44-60 |
| δ merge | partial | LLM host choice S/llm_mediated.py:343-345. Title collision becomes an UPDATE G/ingest.py:358-389 | LLM / rule | Python + SQL | A title collision overwrites summary and embedding (G/ingest.py:384-388) |
| δ supersede | partial | `supersede_current` G/plan.py:183-210, applied at G/ingest.py:392-443 | rule (same field name) | Python | **No sem_conflict** (N6) |
| Eager vs deferred per derivation | **no** | Everything runs before COMMIT (G/ingest.py:10-19) | – | – | Add plan π plus P |
| On-read resolution / Mat(S) | **no** | Retrieve never reads `needs_revision` (N13) | – | – | Strong mode resolves P |

### Retrieve_ρ(q,S)→⟨O,S',r⟩

| element | exists? | where | LLM? | Python / engine | gap |
|---|---|---|---|---|---|
| Request q | partial | `Query` G/types.py:227-253 | – | Python | No principal, consistency mode or token/LLM budget. `RetrievalMode` means *modality* (G/types.py:209), see D4 |
| Seed V | yes | `tjs_open` seedless G/retrieve.py:402-415. VECTOR mode G/retrieve.py:258-276 | no | **engine** | – |
| Seed R (refs / anchor) | yes | G/retrieve.py:205-213, 278-291 | no | SQL | – |
| Expand G | yes | Inside `tjs_open` G/retrieve.py:350-375. GRAPH mode G/retrieve.py:293-337 | no | **engine** | One edge type per call, out-edges only (IF §6.1b, §6.4) |
| Filter θR | yes | Pushed predicate G/retrieve.py:166-181 | no | **engine** | `extra_filter` is raw SQL (G/retrieve.py:179-180) |
| Filter θpol | **no** | G/policy.py:60-138 holds integrity checks only (N8, N9) | – | – | ACL predicate pushed with `scope_id` |
| Filter θsem | **no** | N6 | – | – | P1 / P3 / P4 |
| Rank | partial | Engine order. Python score is `1-(rank/n)` (G/retrieve.py:533-537) | no | engine | sem_rank as a bounded post-stream re-rank (ADR-0007) |
| Assemble_B | partial | `build_prompt_block` G/retrieve.py:647-660 | no | Python | No token budget, no sem_summarize |
| S' side effects | yes | Salience and access writes G/retrieve.py:547-644 | no | SQL | The docstring promises new association edges, but the code only updates existing ones (G/retrieve.py:622 vs 630-635), see D9 |
| Strong vs stale | **no** | Graph reads are commit-visible (IF §6.5, G/memory.py:189-191) | – | – | P3 |
| Planner orders stages | **no** | Caller picks the route (G/retrieve.py:185-192). E plans VFWD/RREV/AIVG are hand-picked (E/physical.py:33-41) | – | – | P4 |

### Update

| element | exists? | where | LLM? | gap |
|---|---|---|---|---|
| Forget ladder / archive | yes | G/forget.py:56-93, 165-209. Thresholds G/salience.py:46-48. Edge tombstone G/forget.py:266-269 | no | "Compress" only archives; there is no LLM summary (D12) |
| Delete | no, by design | G/forget.py:14-16 | – | – |
| Conflict detect / resolve | partial (syntactic) | G/revise.py:130-171. Newest wins G/revise.py:151-160 | no | No sem_conflict |
| Duplicate merge | yes, **buggy** | G/revise.py:184-253. Edge re-point touches only the mirror, :223-240 (N14) | no | Fix in P0 (D10) |
| Revision propagation | flag-only | G/revise.py:258-328 | no | No re-derivation; flags spread with every call (D7) |
| Trigger / lazy maintenance | **no** | `PolicyEngine.trigger` G/policy.py:248-266 is never called (N12) | – | P2 scheduler (D6) |

### Cross-cutting

| element | exists? | where / search |
|---|---|---|
| Logical plan object | **no** | G/plan.py is a write-op vocabulary (G/plan.py:23-28). E/physical.py is a retrieval enum only |
| Cost / selectivity estimate | **no** (Python) | N1, N2. Engine: binary join-order decision only (src/planner/join_order.c:82, join_order_cost.c:49) |
| LLM result cache | **no** | N5. The paper assumes one exists (ID:73-76) |
| LLM determinism pinning | partial | temperature 0, seed 0, thinking off, versioned prompts (S/llm_mediated.py:45-46, 550-552, 576-579). No replay is possible without a cache |
| LLM call / token accounting | yes | S/llm_mediated.py:312-315 → G/ingest.py:575-596 → `CostMeter` G/store.py:148-182 → `gem_transition` G/schema.sql:180-189 |
| Same PG instance / txn | yes | Extensions and schema are created on one connection (G/store.py:327-346). G and E share the vid space but use different writer locks (G/store.py:57 vs E/store.py:29), and nothing enforces E/store.py:26-28 |

### Name map (GEM → paper)

| GEM | paper |
|---|---|
| `IngestOperator.ingest` + `Strategy.plan` | Ingest_π. `plan()` = T∘C∘δ fused, `apply()` = commit ΔS |
| `_host_slate` / `find_similar` | C (V leg only) |
| LLM `host.unit_id` vs `host.new_title` | δ merge vs new (≈ sem_match / sem_topk) |
| `supersede_current`, `edges[]` | δ supersede, δ link |
| `RetrieveOperator.retrieve`, `tjs_open` | Retrieve_ρ, Seed+Expand+θR fused |
| `build_prompt_block` | Assemble (no B) |
| `reinforce` / `_update_structure` | S' side effects |
| `ReviseOperator`, `ForgetOperator` + `ExponentialSalience` | Update: conflict / merge / propagation; forget = MemState ladder |
| `needs_revision` | nearest analogue to P (revision only) |
| `PolicyEngine` / `gem_policy` (Orogat P_t) | **not** θpol; the name collides with paper P (D3) |

### Paper-vs-code disagreements

| # | disagreement | change |
|---|---|---|
| D1 | The paper says T "depends on x only" (ID:33-34), but LLM extraction is conditioned on the candidate slate C (S/llm_mediated.py:243-244, 307-312) | **paper**: present T→C→δ as a logical split and allow a fused `sem_extract_decide` |
| D2 | Mat(S_π1)=Mat(S_π2) "given caching" (ID:73-76). A deferred δ reads the *current* candidate set, so its prompt and cache key differ | **paper + code**: read C against the snapshot at x's commit (logical ts), or limit the claim to order-insensitive derivations |
| D3 | Paper P = pending, but GEM's P_t = policies (G/schema.sql:139) | **paper**: rename pending to Q (cheaper than a code rename) |
| D4 | Paper `mode∈{strong,stale}` (RD:15-16) vs code `RetrievalMode` = modality | **code**: add a separate `consistency` field and keep `RetrievalMode` |
| D5 | θpol is principal access in the paper (RD:47-50) but integrity checks in code | **code** |
| D6 | IF:235 says revise runs "asynchronously, policy-triggered", but nothing calls `trigger` | **code** (P2) |
| D7 | IF:229-231 says propagation halts where the policy does not fire, but code flags the whole BFS | **code** (P2: re-derivation step, the home of sem_conflict) |
| D8 | Agentic docstring vs tool visibility | **code** |
| D9 | `_update_structure` docstring vs code | **code** (fix the docstring, or insert through the AM) |
| D10 | Duplicate merge desyncs the AM | **code** (P0) |
| D12 | UL Ours row vs "compress" without an LLM | **paper**: say non-LLM compression unless sem_summarize lands |
| D13 | ST is stale (384-dim, "unverified") | **doc**: ST addendum |

## 2b. Placement verdict (Q2)

Evidence base:
- **Planner shape.** `src/planner` is SQL-callable decision functions with **no planner hook** (src/planner/join_order.c:13-16; join_order--0.1.0.sql:2-4). Its only production caller is the PL/pgSQL lowering (src/graph_store/graph_store_am--0.2.0.sql:556-562; ADR-0011 Option B). `LegStats` / `TriJoinOrder` are frozen and parity-pinned to `join_order_ref.py` (join_order.c:31-43).
- **Operator shapes.**
  - `tjs_open` is a materialize-mode SRF (src/tjs_pg/tjs_pg.c:1186-1187). It has a bounded output and terminates internally after consecutive drops (:1571).
  - `tjs_vsgt` is the only true Begin/Next/End CustomScan (src/tjs_pg/vsgt_pg.c:80-121, 193-197). Its costs are hard-coded (:110-112), and `vsgt_next` returns one row per call (src/tjs_pg/vsgt_exec.c:547).
  - `tjs_session_cursor` (ValuePerCall) is built nowhere.
- **No HTTP anywhere in `src/`** (N-E1).
- **Stock-PG CI builds only** `graph_store`, `graph_store_ext` and `tjs_pg` (scripts/pg17_graph_test.sh:17-19, 36-43; .github/workflows/ci.yml:57-81). `src/planner` is fork-image only.

| criterion | (a) GEM Python | (b) Engine C | (c) Split |
|---|---|---|---|
| Planner visibility | None. ADR-0023 forbids Python plan orchestration, so sem ops can only post-filter one `tjs_open` result | Must be built from scratch either way; there is no optimizer to join | **Additive** `SemLegStats` + `tridb_place_sem` next to `join_order_cost.c`, called at the lowering (ADR-0011 precedent). Receipts serve as EXPLAIN |
| TR-1 | Lost across the LLM stage, because `tjs_open` returns a materialized ≤k set | A predicate in `vsgt_next/passes()` keeps O/N/C, but a synchronous HTTP call blocks the backend with no CHECK_FOR_INTERRUPTS, at ~10³× the per-candidate leg cost | **Engine stream stays pure.** Python `FETCH n` from a `tjs_vsgt` portal, batch-evaluate, `CLOSE` on enough survivors. Bounded batch (ADR-0020 allows a budgeted buffer) |
| Single process | State in PG; LLM out of process (as ingest is today) | No sidecar, but libcurl in the backend is a new code class holding locks and snapshots across the network (ADR-0017 concern) | The LLM is an external pure function (like the embedder). **Cache and P are heap tables**: one WAL, one txn manager. No second store |
| Determinism | temp 0 / seed 0, but no replay | Network inside a byte-parity-tested operator (ADR-0023 gate 2) | **A cache hit is a table read.** Replay can run entirely from cache. Misses go to P and are disclosed in r, like `scan_budget` |
| Build reality | Runs anywhere | Must build on stock 16/17 **and** the 13.4 fork. New libcurl dependency in the release image | Planner piece is pure arithmetic C plus SQL. **Must be added to `pg17_graph_test.sh` / `STOCK_TESTS`** (or live in `tjs_pg`) to get per-PR CI. Cache / P are DDL. LLM client stays Python |
| Effort / risk / tests | Lowest. `make test` pytest. But the Figure 1 "one planner" claim is not credible | Highest. No tests and no precedent. ADR-0011 rejected similar surface for TR-1 risk | Medium. Every seam has a harness: `join_order_cost_test.sh`, `vsgt_streaming_test.sql`, pytest fakes |

**Recommendation: (c).**
- **sem_filter** is a predicate leg. This fits ADR-0007: "a fourth leg… any additional *filter* leg".
- **sem_rank / sem_topk** are a bounded post-stream re-rank over a prefix of size ≤ B·c. A second rank authority would violate ADR-0007, so ADR-0026 states this explicitly.
- **ADR-0012's "external re-ranker / second round-trip" objection is accepted for semantic ops only.** That ADR rejects round-trips that move V/G/R data across systems. Here the LLM receives bounded text and returns a verdict, and repeat calls are suppressed by the cache.

---

## 3. Target architecture: Figure 1 boxes → components

| Figure 1 box | component | status |
|---|---|---|
| Agent | external caller (harnesses in `bench/`, `experiments/`) | exists |
| Memory operators: Ingest | `IngestOperator` G/ingest.py:475 + new `IngestPlan` π | extend |
| Memory operators: Retrieve | `RetrieveOperator` G/retrieve.py:98 + new `RetrievePlan` ρ | extend |
| Memory operators: Update | `ReviseOperator`, `ForgetOperator` + trigger / scheduler | extend |
| Unified cost-based planner | Python `gem/planner.py` (logical plan + cost from telemetry) **calling** new C `src/planner/join_order_sem.c` decision function; existing `join_order*.c` keeps V/G/R order | new (C additive) + exists |
| Vector ops | HNSW leg in `tjs_open` / `tjs_vsgt` | exists |
| Graph ops | graph AM (`GphPullApi` src/graph_store/gph_pull_api.h:6-14, `gph_traverse_*`) | exists |
| Relational ops | pushed predicate (ExprState tjs_pg.c:1493; typed `passes()` vsgt_exec.c:243-266) | exists |
| Semantic ops (orange) | new `G/semops.py`: sem_extract, sem_match, sem_conflict, sem_filter, sem_rank, sem_topk, sem_summarize | new |
| Store: Vector HNSW | pgvector HNSW G/schema.sql:96-97 | exists |
| Store: Graph adjacency AM | `graph_store_am` | exists |
| Store: Relational heap | `gem_*` tables | exists |
| Store: Pending P (paper symbol → Q, D3) | new `gem_pending` heap table | new |
| LLM result cache | new `gem_sem_cache` heap table | new |
| LLM serving | vLLM OpenAI endpoint via `OpenAIChatExtractor` S/llm_mediated.py:534 | exists |
| Panel b: ANN_k(e_q) [V], Lookup(refs) [R], Seed ∪ | seedless / anchored `tjs_open`; multiseed ADR-0012 | exists |
| Expand [G] | in operator | exists |
| Filter outcome/scope/policy [R] | scope exists; outcome via E predicates (E/oracle.py:149); policy ACL | extend |
| sem_filter [S] | portal-batched filter over `tjs_vsgt` | new |
| Rank | engine order + bounded sem_rank | extend |
| Assemble_B | `build_prompt_block` + token budget + optional sem_summarize | extend |
| Resolve P (strong mode, dashed) | on-read resolver in retrieve txn | new |

---

## 4. Phased work plan

**P−1 (prerequisite, not part of the figure): make the gates measurable.**

Today `make test` aborts at collection with 6 errors, and `make lint` reports ~874k errors that come almost entirely from untracked `results/` and `gem_evotrace_archive/`. Appendix B has the details.

- Files: `pyproject.toml`/ruff `extend-exclude` for result dirs, plus the 6 broken tests or the bench code they import.
- Exit:
  - `make test` collects;
  - the failing set is ≤ the 21 known baseline failures, with the list recorded;
  - `make lint` is clean on tracked files.
- GX10: no. Owner decision required (§5 R8).

| phase | files to touch | new tables / catalog | tests to add | exit criterion | GX10? |
|---|---|---|---|---|---|
| **P0 schema + correctness fixes** | G/schema.sql (additive). G/types.py (`IngestResult`/`RetrievalResult` fields: `pending_enqueued`, `pending_resolved`, `cache_hits`, `cache_misses`, `staleness`). G/revise.py:223-240 (re-point and tombstone via `gph_insert_edge` / `gph_tombstone_edge`, D10). G/retrieve.py:622 docstring (D9) | `gem_sem_cache(op, model, prompt_version, input_hash, output jsonb, prompt_tokens, completion_tokens, created_tx; PK(op,model,prompt_version,input_hash))`. `gem_pending(id, scope_id, derivation, target_ref, trigger∈{background,batch,on_read}, snapshot_lsn/txid, enqueued_tx, resolved_tx, status)`. Receipt columns on `gem_transition` | `tests/test_gem_unit.py`: DDL idempotence, receipt fields. `tests/test_gem_live.py` (skip-gated): after a duplicate merge, AM neighbours == `gem_edge` neighbours | Schema applies twice cleanly. Live check: AM ≡ mirror after merge. Existing gem suites unchanged | no (stock-PG / Python) |
| **P1 semantic ops behind the cache** | new `G/semops.py` (`SemOp` protocol, `CachedLLM` wrapping `OpenAIChatExtractor`, batched `complete_many`). Refactor S/llm_mediated.py:304-322 `_extract` to go through `CachedLLM`. Accounting via `CostMeter` G/store.py:148-182 | uses `gem_sem_cache` | new `tests/test_gem_semops.py`: hit/miss, key covers model + prompt_version + input, batch bound respected, `CostMeter` counts, **replay with the network disabled** | An LLM-mediated ingest replayed from cache with the HTTP client disabled gives byte-identical `gem_*` rows. Each op reports n_sem and tokens | no |
| **P2 ingest as a plan** | new `G/planner.py` (`IngestPlan` π: derivation → {eager, background, batch, on_read}). G/ingest.py (logical T/C/δ, fused op allowed per D1; candidate reads pinned to the commit snapshot per D2; enqueue to `gem_pending`; LLM work moved out of the writer txn). New graph candidate read via `gph_traverse_typed` from `refs`. Wire `PolicyEngine.trigger` (G/policy.py:248-266, D6). New `materialize(scope, until)` resolver. Propagation re-derivation step (D7) | `gem_pending` in use | new `tests/test_gem_ingest_plan.py`: **equivalence test** Mat(S_eager) == Mat(S_deferred) == Mat(S_on_read) on a fixed corpus with a warmed cache (fake + live). Receipt lists pending ids. Trigger fires | Equivalence test passes on fake and live. No LLM call happens while the writer lock is held (assert in test) | no |
| **P3 retrieve as a plan** | G/types.py `Query` (`principal`, `consistency∈{strong,stale}`, `budget_tokens`, `budget_llm`). G/retrieve.py: five explicit stages; θpol ACL predicate pushed with `scope_id` (G/retrieve.py:166-181); θsem over the result as an interim (bounded over-fetch k′=⌈k/ŝ⌉, capped by doubling, disclosed in r); strong mode resolves `gem_pending` rows hit by the candidate set in the same txn; Assemble with a token budget and optional sem_summarize | ACL column on `gem_unit` (or a `gem_acl` table) | new `tests/test_gem_retrieve_plan.py`: strong sees resolved state while stale reports `staleness>0`; θpol never leaks across principals; budget respected; n_sem counted | Strong == Mat(S) answer on test corpora. Stale ≤ budgeted latency. ACL leak test green | no |
| **P4 planner integration** | new `src/planner/join_order_sem.c` (+ `.sql` function `tridb_place_sem(sel_sem, c_llm, hit_rate, n_in, …)`; rank `(1−s)/(c·(1−h))` ordering among predicate legs; `LegStats` untouched). Extend `src/planner/join_order_ref.py` (Python spec). Makefile OBJS. **Add `src/planner` to `scripts/pg17_graph_test.sh` and `STOCK_TESTS`.** Streaming sem_filter: G/semops.py pulls `FETCH n` from a portal over a `tjs_vsgt` query and `CLOSE`s on enough survivors. `G/planner.py` calls `tridb_place_sem` and records the decision in r | SQL function `tridb_place_sem`; GUC `tridb.sem_cost_per_call` (calibrated from `gem_transition`) | `test/join_order_sem_test.sql` (bit-identical to the ref), `scripts/join_order_sem_test.sh`, pytest `tests/test_join_order_sem_ref.py`, live test: portal closes early and `vsgt_stats` shows a bounded pull | C ≡ ref on a grid of inputs. On a synthetic workload the planner places sem_filter after cheap legs and n_sem ≤ the naive order's. Stock CI job builds `src/planner`. **C is unbuilt-here until the stock CI run is green** | stock PG: no. 1M-scale latency sign-off on fork: **yes (GX10)** |
| **P5 downstream term** | new `G/downstream.py`: estimator Ê[#attempts \| O] from O's composition (golden / warning / fixed_by fractions, \|O\|tok). Trained on `exg/results/experience_graph_v0.1.0` (EXG 7:3 split). `G/planner.py` uses it to choose k / B | none | `tests/test_gem_downstream.py`: deterministic fit, monotonicity sanity checks | Brier / calibration on the held-out 30% beats a constant baseline. The planner's B choice changes the predicted cost | no |
| **P6 evaluation hooks** | harness adapters only (no operator code) | none | per-dataset smoke tests (skip-gated) | Each claim maps to ≥1 dataset (table below) with a runnable command | no (latency at 1M: yes) |

P6 claim → dataset map:

| claim | dataset / harness |
|---|---|
| Eager vs lazy ingest cost at equal answer quality | EvoMemBench (`EvoMemBench/`), LongMemEval-V2 (`LongMemEval-V2/`) |
| Retrieval + downstream attempts (E[#attempts\|O]) | EXG HumanEval+/MuSiQue/HotpotQA (`exg/datasets`, `exg/results/experience_graph_v0.1.0`) |
| Vector-seeded graph traversal under sem_filter | EvoTrace via `gem_eg` (E/load.py) |
| Multi-run trajectories / outcome labels | paper Table 8 corpora (SWE-bench experiments, τ²-bench); new adapters |

Phase order is P−1 → P0 → P1 → (P2 ∥ P3) → P4 → P5 → P6. P2 and P3 share only the schema from P0.

---

## 5. Risks and open decisions (recommended default first)

| # | decision / risk | default |
|---|---|---|
| R1 | D3 symbol clash (P pending vs P_t policies) | Rename pending to **Q** in the paper |
| R2 | D2 equivalence needs snapshot-pinned candidate reads | Pin C to the commit snapshot and narrow the paper claim to that |
| R3 | sem_rank vs ADR-0007's single rank authority | Bounded post-stream re-rank; record in ADR-0026 |
| R4 | ADR-0012 "no second round-trip" | Accept for semantic ops only (bounded text, cached) |
| R5 | Stream source for sem_filter | `tjs_vsgt` portal (CustomScan, pipelined). `tjs_open` over-fetch only as the P3 interim. `tjs_session_cursor` stays unbuilt |
| R6 | `src/planner` not in stock CI | Add it to `pg17_graph_test.sh` + `STOCK_TESTS` in P4 rather than moving it into `tjs_pg` |
| R7 | ADR-0023 `tjs.seedless_plan` and `tjs_open_graph_reached/relational_examined` are absent from `src/tjs_pg` and come from an out-of-tree runtime (`bench/agent_memory/evomembench/install_local_tjs_runtime.py:1-5`); `SET` on the unregistered GUC silently succeeds | Do not depend on them in Figure 1. File a follow-up to upstream or retire them |
| R8 | Pre-existing red gates (Appendix B) | Fix in P−1 before P0, so phase exit criteria are measurable |
| R9 | G and E share the vid space with different writer locks (G/store.py:57, E/store.py:29) | One database per package in tests; long term, a shared lock key |
| R10 | Cost constant c_llm drifts with model / hardware | GUC calibrated from `gem_transition` telemetry; recalibrate per model |
| R11 | Holding the writer txn across LLM latency (G/ingest.py:504) | P2 moves LLM work before the txn; the txn commits only validated ops |

## 6. Out of scope for v0.1.0

- HTTP or libcurl inside the backend, or any LLM call from C.
- A per-candidate semantic predicate inside the `tjs_open` scan loop.
- GPU work at query time (ADR-0017), BM25, and a new query language.
- A learned cost model; v0.1.0 uses analytic rank plus calibrated constants.
- A full multi-tenant ACL model; only a principal predicate is in scope.
- Hard delete / erase (GEM never deletes, G/forget.py:14-16).
- The 1M-scale GX10 latency sign-off: it is listed as gated and is not claimed here.

---

## Appendix A: negative searches (0 matches unless noted)

GEM: `G="bench/agent_memory/gem bench/agent_memory/gem_eg"`, run as `grep -rnI --include='*.py' -e '<pat>' $G`.

- N1 `selectivity`
- N2 `cost_model|costmodel|estimate|choose_plan|select_plan|pick_plan` (only an unrelated docstring at G/__init__.py:11)
- N3 `deferred|defer|background|strong` (the one hit is the policy name `reinforce-on-read`, G/policy.py:282)
- N4 `pending` (only local embed lists)
- N5 `lru_cache|functools.cache|llm_cache|semantic_cache|cache` (no LLM-related hits)
- N6 `sem_extract|sem_match|sem_conflict|sem_topk|sem_filter|sem_rank|sem_summarize`
- N8 `principal`
- N9 `ACL|acl|allowed_principal|authorize`
- N12 `grep -rn "\.trigger(" bench/agent_memory/gem bench/agent_memory/gem_eg`
- N13 `grep -rn needs_revision bench/agent_memory/gem/retrieve.py`
- N14 `grep -n "gph_insert_edge\|gph_tombstone_edge" bench/agent_memory/gem/revise.py`
- N-G1 `gph_traverse` in G/ingest.py and S/ (no write-time graph read)

Engine:

- N-E1 `grep -rnI "curl\|libcurl\|http:\|https:\|socket(" src/ --include=*.c --include=*.h --include=*.sql`, plus `grep -rniI "llm\|openai\|bgworker\|dblink\|plpython" src/` (only the word "semantics" in comments)
- N-E2 `grep -rn "planner_hook\|set_join_pathlist_hook\|CustomScan" src/` (only src/tjs_pg/vsgt_pg.c)
- N-E3 `grep -rn seedless_plan src/`
- N-E4 `grep -rn "def choose\|def estimate_cost\|cost_model\|choose_join_order" bench tools --include=*.py`

## Appendix B: gate baseline at plan time (2026-10-10, before any change)

- `make lint`: fails with ~874k ruff findings, nearly all in untracked `gem_evotrace_archive/snapshot` (318k) and `results/*`. On tracked `*.py` only, `ruff check` passes, and `ruff format --check` would reformat 17 files.
- `make test`: interrupted with 6 collection errors (ImportError against current bench code, and `StreamingChatResult(finish_reason=…)`). The files are:
  - `test_evomembench_generation_policy`
  - `test_evomembench_injection_gate`
  - `test_gem_e1_e2_candidate_policies`
  - `test_gem_oe_run_matrix_scheduler`
  - `test_leg_ablation_arms`
  - `test_native_fused_plan_contract`
- With those 6 ignored: 1894 passed, 21 failed, 96 skipped, 4 xfailed. The failures are in:
  - `test_e1_parity`
  - `test_evomembench_*`
  - `test_mlevolve_grade_nodes`
  - `test_polyglot_retriever_query_vector`
  - `test_release_coherence`
  - `test_table5_track_c_invariants`
  - `test_traceml_evaluation_v3`

This plan touches neither gate. It adds only Markdown under `docs/`.
