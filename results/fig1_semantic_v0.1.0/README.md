# Figure 1 EvoTrace experiment results (v0.1.0)

Write-up: [`docs/figure1_evotrace_experiment_v0.1.0.md`](../../docs/figure1_evotrace_experiment_v0.1.0.md).
Reproduce: `run_all_v2.sh` (private PG16 cluster on 127.0.0.1:55440 with vector 0.8.0,
graph_store_am 0.2.0, tjs_pg 0.3.0 + the VSGT fix, join_order 0.2.0; vLLM gemma-4-26B-A4B-it
on :8000 for the real runs).

| directory | status | what |
|---|---|---|
| `real_v2/` | **authoritative** | real LLM, 662 events (4 tasks × 3 sessions × ≤60 nodes), every arm and retrieval plan with its own cache namespace → cost numbers |
| `real_v2_shared/` | **authoritative** | same slice, all arms share one cache seeded by the eager arm → Mat(S) equivalence "given the cache" |
| `mock_full/` | **authoritative** | full corpus (10,672 events, 18 tasks, 121 sessions), deterministic content-aware mock LLM → scale counts and exact logical equivalence |
| `downstream.json` | **authoritative** | P5 estimator (EXG retrieval logs), held-out Brier per dataset |
| `real_slice/` | superseded | first real run, before the adversarial-review fixes: naive plan truncated the raw stream before θR (different candidate universe), strong reads skipped fix-hint destinations |
| `real_slice_shared/` | aborted | interrupted when the shared vLLM server was restarted externally; rerun as `real_v2_shared/` |
| `postgres.log` | log | server log of the private cluster (contains the VSGT crash that led to the fix) |

Each directory has one JSON per arm, `comparison.json`, `REPORT.md` (rendered by
`python -m bench.agent_memory.fig1.report <dir>`) and `run.log`.
