# Figure 1 on EvoTrace: tri-modal + semantic ingest/retrieve workload (v0.1.0)

Date 2026-10-10, branch `feat/figure1-semantic-ops`. Plan: `docs/figure1_unified_semantic_plan_v0.1.0.md`
(Addendum A); decision: ADR-0026. Raw results: `results/fig1_semantic_v0.1.0/` (see its README).
Code: `bench/agent_memory/semantic/` (operators, cache, pending set, planner glue, stream) and
`bench/agent_memory/fig1/` (workload). Hardware: x86_64 workstation, stock PG16 private cluster,
2× RTX PRO 6000 shared with other jobs. **Not** the GX10; no fork-build or 1M-scale claim.

## 1. Workload ("mock" agents)

EvoTrace = 10,672 finished attempts (nodes) of evolutionary coding agents over 18 tasks
(10 ALE, 8 math) in 121 sessions, with lineage, fitness and judge outcome, and pre-computed
Qwen3-Embedding-0.6B vectors (1024-d). No agent is re-run: the trace *is* the workload.
Sessions are interleaved round-robin on one timeline (parents always before children). For every
node x with parent p:

1. **Retrieve** for the agent about to improve p (one request per plan under test);
2. **Ingest** x.

**Ingest_π(x)**:
- **Core** (always eager): vertex row with relational columns, an HNSW vector, and lineage edges in the native graph AM plus the relational mirror.
- **Semantic derivations** (eager or deferred, per π):
  - `lesson` = sem_extract(outcome(x)), stored in the payload.
  - For a failed x, `fixed_by` = sem_match(x, c) over C(x), written as native `eg_fixed_by` edges. C(x) is an exact cosine top-5 over same-task nodes from other sessions that are valid, have fitness > 0, and have id < vid(x).

**Retrieve_ρ(p)**. The logical query: take U, the first k_max = 48 rows of the native `tjs_vsgt` stream that satisfy θR. The stream is ANN seeds (m = 3) followed by lineage descendants (d = 2), with scope, task and session pushed down; θR is "valid and fitness ≥ fitness(p)". Return the first k = 5 rows of U that satisfy θsem ("a transferable idea for improving p"). Add fix hints for a failed p (a typed graph traversal over `eg_fixed_by`), and assemble everything under an 800-token budget.

| plan | θ order | stops at | role |
|---|---|---|---|
| `trimodal` | θR only | k rows | V/G/R baseline, no LLM |
| `naive` | θsem then θR | universe boundary | semantic-first, no early close |
| `planned_mat` | `tridb_place_sem` order | universe boundary | planner order, no early close |
| `planned` | `tridb_place_sem` order | k survivors (closes portal) | Figure 1 plan |

All four plans stream through a server-side portal in bounded batches; none fetches the whole result.

**Arms**. Each arm runs in its own database, so arms never share an HNSW index:

| arm | ingest | consistency | purpose |
|---|---|---|---|
| eager | eager | strong | reference; runs all four plans on identical state |
| eager_ctl | eager | strong | control: identical to eager, measures run-to-run noise (HNSW randomness, LLM) |
| onread | deferred, resolved on read | strong | lazy ingest |
| onread_stale | deferred, never resolved during replay | stale | cost of staleness |
| batch | deferred, resolved every 64 ingests | strong | batched maintenance |

## 2. Results

### 2.1 Ingest: deferring derivations takes the LLM off the write path (real LLM, 662 events)

| arm | ingest p50 / p95 ms | LLM calls on ingest path | LLM wall s (all derivations) |
|---|---:|---:|---:|
| eager | 309.8 / 466.1 | 1,286 | 195.1 |
| onread | 5.3 / 8.9 | 411 (only items a read touched) | 100.6 |
| onread_stale | 4.9 / 6.4 | 0 | 23.0 (final catch-up) |
| batch | 5.3 / 8.9 (mean 35.4) | 1,243 off the critical path | 66.0 |

- **Write latency.** Deferral cuts ingest p50 58×.
- **LLM work actually needed.** On-read resolution needs 68% fewer LLM calls during the replay. The rest are only paid if the state is fully materialized at the end.
- **Batching.** Batch maintenance processes the same 1,286 calls in a third of the eager LLM wall time (66 s vs 195 s), because larger batches use vLLM's GPU batching.
- **Full corpus (10,672 events, mock).** The same shape holds: 19,220 eager calls versus 5,459 on read during the replay.

### 2.2 Retrieval: the planner halves semantic evaluations without changing the answer

| (eager arm) | n_sem / retrieval | LLM calls | p50 / p95 ms, real | n_sem / retrieval, full corpus (mock) |
|---|---:|---:|---:|---:|
| trimodal | 0 | 0 | 4.2 / 5.3 | 0 |
| naive (θsem first) | 6.37 | 1,877 | 85.8 / 269.0 | 8.63 |
| planned_mat | 3.20 | 898 | 7.0 / 151.1 | 3.37 |
| planned (streaming) | 3.03 | 859 | 6.9 / 152.9 | 2.88 |

- **Fewer evaluations.** `tridb_place_sem` puts θR first (pushed into the CustomScan) in every request. That gives 52% fewer semantic evaluations than the naive order with the real LLM, and 67% fewer on the full corpus.
- **Early close.** Closing the portal at k survivors saves another 5–15% (`planned` vs `planned_mat`). The early-close rate is low (4–6%) because few U-candidates beat the parent.
- **Same answer.** With the deterministic mock, naive, planned_mat and planned return identical ordered outputs and identical text for all 10,479 full-corpus retrievals (exact-ordered agreement 1.0000).
- **Real-LLM gaps are nondeterminism.** With the real LLM, naive vs planned_mat agree on 0.948 of retrievals. Each plan has its own cache namespace, so the same prompt is re-asked, and gemma at temperature 0 is still not deterministic across vLLM batches (§2.3).

### 2.3 Eager ≡ deferred, *given the cache*

| Mat(S) vs eager (after final materialization) | real LLM, separate caches | real LLM, shared cache | mock, full corpus |
|---|---|---|---|
| eager_ctl (identical pipeline) | lessons 52.4% equal, fixed_by Jaccard 0.895 | **equal** | **equal** |
| onread | 70.5%, 0.889 | **equal** | **equal** |
| onread_stale | 53.8%, 0.895 | **equal** | **equal** |
| batch | 62.1%, 0.906 | **equal** | **equal** |
| native AM edges == relational mirror | yes, every arm | yes | yes |
| offline replay (LLM disabled, cache only) | **equal**, 0 LLM calls | **equal**, 0 calls | **equal**, 0 calls |

This is the paper's correctness condition, Mat(S_π1) = Mat(S_π2) "given cached LLM outputs", observed directly:
- **With a shared cache it holds exactly**, for both the real and the mock LLM.
- **Without a cache it fails**, even between two identical eager runs: 52% lesson agreement.

So the cache is not just an optimization; the semantics depend on it. Two further conditions are needed, and the workload supplies both (plan D2):
1. the candidate reads are pinned (exact scan, id < vid(x));
2. the inputs are immutable under a single writer.

### 2.4 Staleness costs answer quality; strong reads cost latency

| planned, vs eager/planned_mat | exact-ordered agreement, real shared | exact-ordered agreement, mock full | retrieval p50 / p95 ms, real separate |
|---|---:|---:|---:|
| eager_ctl (noise floor) | 0.957 | 0.941 | 7.1 / 199.5 |
| onread (strong) | 0.952 | 0.930 | 93.9 / 614.0 |
| batch (strong) | 0.957 | 0.916 | 70.2 / 512.3 |
| onread_stale | **0.545** | **0.571** | 7.6 / 187.4 |

- **Strong reads** with deferred ingest are within noise of eager, but they pay for resolving pending derivations at read time: p50 7 ms → 94 ms (393 derivations resolved on read).
- **Stale reads** keep eager-like latency, but only 54–57% of retrievals return the same hints. They saw 1,518 still-pending items over 650 retrievals.

This eager / lazy / strong / stale trade-off is what a per-derivation plan π, chosen by a cost model, has to arbitrate.

### 2.5 Downstream term (P5)

The Ê[#attempts | O] estimator, trained on EXG retrieval logs with a 7:3 split, beats the base rate on MuSiQue:

| dataset | Brier, estimator | Brier, base rate | 95% CI of the gain |
|---|---:|---:|---|
| MuSiQue | 0.2064 | 0.2245 | [+0.013, +0.023] |
| HotpotQA | — | — | ties |
| HumanEval / HumanEval+ | — | — | too few failures to judge |

`choose_k` plugs the estimate into c_db + c_llm·n_sem + E[#attempts|O]·c_agent. The features are correlational, so treat the k it picks as illustrative (see `results/fig1_semantic_v0.1.0/downstream.json`).

## 3. Engine bug found and fixed

The workload updates payload rows, which leaves dead btree entries behind. That exposed a use-after-free in the VSGT CustomScan:
- **Cause.** `probe_node` / the anchor scan ran index AM calls under the per-row scratch memory context. `btgettuple` lazily pallocs `so->killedItems` there; the next scratch reset frees it; `index_endscan` then pfrees a dangling pointer.
- **Symptom.** "could not find block containing chunk", or a backend SIGSEGV.
- **Fix.** Index AM calls now run in the request context (`src/tjs_pg/vsgt_exec.c`).
- **Regression test.** `test/vsgt_dead_tuple_test.sql`, added to `STOCK_TESTS`. Before the fix it crashed the backend on stock PG16.
- **Not patched.** The unbuilt forked copy in `src/tjs_session_cursor/` has the same pattern.

## 4. Threats to validity

- **Small real slice.** The real-LLM runs cover 662 events (4 tasks). The full corpus was run only with a deterministic mock, which shows counts and logical equivalence, not LLM time.
- **Mock semantics are synthetic.** Mock verdicts are hash-driven at controlled rates (keep 0.4, fix 0.5). Real selectivities differ (real n_sem/retrieval 3.0 vs mock 2.9).
- **Shared GPU.** The vLLM server was shared with other jobs and was restarted once externally (the run was repeated). Latencies include that contention.
- **Agreement is not task quality.** No agent was re-run, so "quality" here means agreement with the eager / strong reference, not task success. The P5 estimator is the only link to downstream outcomes, and it is correlational.
- **Approximate ANN.** HNSW is approximate and non-deterministic across builds. The eager_ctl arm measures that noise floor (0.94–0.96).
- **Planner cost constants.** These were c_llm = 0.085 s/evaluation (measured single-call latency) and c_db = 2e-6 s. The ordering decision is insensitive to them here, because θR is always far cheaper.

## 5. Reproduce

```bash
# private PG16 with the extensions (see results/fig1_semantic_v0.1.0/README.md), then:
results/fig1_semantic_v0.1.0/run_all_v2.sh
python -m bench.agent_memory.fig1.report results/fig1_semantic_v0.1.0/real_v2
TRIDB_FIG1_DSN=postgresql://$USER@127.0.0.1:55440/fig1 pytest tests/test_semantic_layer.py tests/test_fig1_workload.py
```
