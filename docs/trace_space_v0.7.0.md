# TRACE configuration space v0.7.0 — spec addendum (2026-11-05)

Status: implemented, adversarially reviewed (fix round 1 applied), tested offline (CPU smoke, all checks true).
No live run started.
Plan: `docs/superpowers/plans/2026-11-05-trace-space-v070.md` (sections 0–14 and Amendment A, which wins on conflict).
Implementation report: `docs/superpowers/plans/2026-11-05-trace-space-v070-report.md`.

## 1. Space

| Field | v0.6.0 live | v0.7.0 live |
|---|---|---|
| `rank_impl` | ColBERT, LLM, cross-encoder (mandatory) | ColBERT, cross-encoder, **none** (optional; LLM removed) |
| `graph_direction` | Backward, Forward, Both | **Both** (fixed) |
| `retrieval_unit` | whole, actions, window | **whole** (fixed) |
| other fields | unchanged | unchanged |

- Schema `trace-pr-v070-rank-optional-v1` (`trimem/optimizer/plan_schema_v070.py`). `"none"` is appended to the
  full `rank_impl` domain, so earlier categorical codes do not change. Configuration keys differ from v0.6.0.
- `rank_impl = "none"`: no Rank operator. The (filtered) pool is ordered by query cosine (ties by session id) and
  the top `returned_k` are injected. `returned_k` stays active.
- Plan: `V(task-ann) → [F_pre(LLM)] → G(<impl>) → [F_post(LLM)] → Rank(<impl>) | TopK(query_cosine)`.
- Executor plans (6): VGR, P1, P2 (rank none) and VGR+Rank, P1+Rank, P2+Rank. The historical executor runs them
  unchanged (`agentgym_session_eval` builds no ranker outside `RANK_PLANS`).
- The stock `JointSpace` refuses the v0.7.0 domains, so a stock launcher on a v0.7.0 run fails closed.

## 2. NoMemory protected baseline

- Not a configuration. Own ledger `<run>/baselines/nomemory` (TEST: `<run>/holdout/baselines/nomemory`), so it is
  never in the GP, the latent encoding, the budget, `initial_keys`, staged pruning or the latency pruner's judged set.
- Evaluated once per dataset in the init stage (100 TRAIN episodes, c16, outside the 5,000-episode budget), imported
  into the arm, and always tested on TEST200.
- Candidate of the TRAIN Pareto set, TEST selection, OneConfig and the Opt-PerQuery router (always a router /
  OneConfig candidate, user 2026-11-05). NOT in the latency-cap front reference (`latency_front` false, user
  2026-11-05). Not a staged-pruning comparator (option `prune_comparator`, default off).
- Execution: the official AgentGym evaluator (`tools/trace_v070_nomemory_worker.py`) under the TRACE episodes' strict
  patches; same model, vLLM / environment slots, `--max_tokens 4096`, `--max_round`, `--action_format` and
  environment variables as TRACE episodes. Receipt: retrieval tokens 0, retrieval latency 0.

## 3. Per-configuration TRAIN samples (Amendment A.4)

- The manifest's `tasks` is the dataset's eligible TRAIN pool (source split `train` minus TEST200 and validation
  ids; ALFWorld 3,792, ScienceWorld 2,076, WebShop 3,930). `train_panel.json` lists the pool with strata.
- Every configuration and NoMemory draws its own 100 tasks: quotas = TEST task-type mix (largest-remainder rule of
  `tools/trace_joint_stratified_panel.py`), seed `digest([20261022, config key])`, uniform without replacement
  inside each type. Order = stratified interleave (`staged_pruning.stage_order`): every 20-task prefix follows the
  mix within one task per type (measured maximum deviation 0.6).
- Quotas: ALFWorld equals the 2026-11-04 stratified panel; ScienceWorld covers 97.5 % of TEST (type
  `measure-melting-point-unknown-substance` has no eligible TRAIN task); WebShop fashion 81 / garden 11 / electronics
  3 / grocery 3 / beauty 2.
- Each configuration's task list is stored in the ledger (`samples` table) and in `<run>/samples.json`.
- All consumers use the configuration's own tasks and compare unpaired: GP observations (mean / SE over own tasks,
  one shared panel identity), staged pruning (own prefixes; pessimistic corners with own SE), latency pruner,
  candidates / Pareto, TEST selection, router, HV input (`python -m trimem.experiments.trace_v070_candidates RUN`).

## 4. Protocol per dataset (order ALFWorld → ScienceWorld → WebShop)

1. Shared init `results/trace_v070_<ds>_init_20261105`: NoMemory 100 + 5 Sobol + 5 LLM (`sobol_then_llm`), fresh in
   the v0.7.0 space (nothing imported from v0.6.0 runs), TEST deferred, no pruning.
2. VAE v2 checkpoint (`results/trace_v070_seqvae_20261105/seqvae_v2_d16.pt` by default; d32 optional) —
   readiness R1 ≥ 0.95, R2 ≥ 0.90 (family `filter_order|graph_impl|rank_impl`), R3 = 1, R4 = 0; freeze-affine R6.
3. Arm `results/trace_v070_latent_<ds>_20261105`: latent-mobo-v1, Matérn × 3, q 4, c16, 5,000 TRAIN episodes incl.
   the 10 init configurations, at most 80 configurations, staged pruning at 20/40/60/80/100 own tasks
   (plan guard keyed by the 6 executor plans) and, at the same stage barriers on the exact own prefix, the
   latency cap (`pruning.latency_cap`: factor ALFWorld 2.0, ScienceWorld 3.0, WebShop 2.0; front = sample-mean
   Pareto front of the complete never-pruned BO configurations only, NoMemory excluded; init protected; it never
   prunes the last remaining configuration of an executor plan, `guarded: plan-last`). Decisions are committed
   in the ledger (`stage_decisions`) before `pruning_log.jsonl` is written (back-filled at launcher startup and
   before every round); a recorded decision is never recomputed.
4. Automatic TEST on (TRAIN sample-mean 3-objective Pareto ∪ BRANE fuzzy Pareto, TAU_ACC 0.02 / TAU_COST 0.10)
   of the never-pruned full BO configurations, ∪ NoMemory (always tested). Router and OneConfig candidates = the
   BRANE fuzzy set ∪ {NoMemory} (user 2026-11-05), so every candidate is TEST-measured (`holdout/selection.json`
   records `pareto_keys`, `fuzzy_keys`, `router_candidates`).
5. Preparation completes with `<run>/prepared.json` (written last); the queue and launcher check the marker.

Queue: `results/trace_v070_e2e_20261105/run_v070_queue.sh` (not started).

## 5. Implementation (additive; no hashed file changed)

| Module | Role |
|---|---|
| `trimem/optimizer/plan_schema_v070.py` | schema, `JointSpaceV070`, `EXECUTOR_PLAN`, `executor_plan`, `ordering` |
| `trimem/experiments/plan_execution_adapter_v070.py` | `compile_config`, `runner_config` (VGR / P1 / P2 for rank none) |
| `trimem/experiments/trace_v070_samples.py` | sampling block, `config_sample`, prefix diagnostics |
| `trimem/experiments/trace_v070_ledger.py` | `SampledLedger` (own sample per reservation), `samples.json` |
| `trimem/experiments/trace_v070_baseline.py` | NoMemory ledger: `BaselineSpace`, prepare / run / import |
| `trimem/experiments/trace_v070_runtime.py` | `v070_patches` (driver attributes), `latent_search_worker_v070` |
| `trimem/experiments/trace_v070_pruning.py` | `staged_evaluate_v070` (own prefixes, executor-plan guard) |
| `trimem/experiments/trace_v070_candidates.py` | candidates, Pareto, HV / plot export |
| `tools/trace_v070_nomemory.py`, `tools/trace_v070_nomemory_worker.py` | NoMemory episodes and claim recovery |
| `tools/trace_v070_holdout.py` | TEST: Pareto ∪ NoMemory |
| `tools/latent_latency_pruner_v070.py` | offline dry-run latency-cap reporter (never prunes) |
| `trimem/experiments/trace_v070_marker.py` | `prepared.json` completion marker, replay of an interrupted prepare |
| `tools/trace_v070_router.py` | sparse per-configuration router + OneConfig over BRANE fuzzy set ∪ NoMemory |
| `tools/trace_v070_live.py` | launcher (`--preflight-only`, `--baseline-only`) |
| `tools/prepare_trace_v070_campaign.py` | `shared-init` / `arm` (explicit-pr refused) |
| `tools/latent_readiness_v070.py`, `tools/train_trace_seq_vae_v070.py` | readiness gates, VAE training |
| `tools/trace_v070_hash_guard.py`, `tools/trace_v070_smoke.py` | fail-closed hash guard (discovery), CPU smoke |

Runtime patching replaces driver module attributes only while `v070_patches(manifest)` is active; the spawned
search worker re-enters the patches in the child and writes `<round>.v070.json` as proof.

## 6. Caveats

- The dense kNN router curve (`tools/trace_pr_router_curve.py`) needs one shared panel; it does not apply to
  per-configuration samples. `tools/trace_v070_router.py` replaces it with per-configuration own-sample predictors.
- NoMemory TRAIN latency is measured in a dedicated c16 stage without concurrent PG / ColBERT load (same
  concurrency, different co-load); it is not used by the latency cap.
- Cross-encoder rank tokens are not counted in `semantic_tokens` (pre-existing).

## 7. TRACE v0.7.1 (user 2026-11-05): SemFilter v2c and the stall rule

Additive: no hashed or v0.7.0 file changed. Same space, same VAE checkpoints. q stays 4; budget unchanged.

- **SemFilter v2c** (`trimem/experiments/semfilter_v071.py`; label `semfilter-v2c`; manifest `v071`, `fixed.semfilter`):
  TOPN 16; compact judge card `judge-card-compact-v1` (tag + outcome, goal, every action, observation cut to 60 chars,
  no-op observations dropped and runs collapsed, 2,000-char cap); within one episode a candidate judged by an earlier
  retrieval reuses its score (key: episode, candidate id, rubric, judge input; `useful` is recomputed per threshold).
  Each retrieval records `cache_hits`, `judged_new`, `tokens_saved_est` (receipt and `attempt/semfilter_v2c.jsonl`).
- **Worker**: `tools/trace_v071_live.py` runs the v0.7.0 launcher with the episode command pointing to
  `tools/trace_v071_episode_worker.py`. A v0.7.1 run started through the v0.7.0 launcher fails closed at its first
  SemFilter episode (the stock `settings` refuses `compact-v2c`). A manifest without `v071` behaves as v0.7.0.
- **ALFWorld mid-run switch**: the init stays on `semfilter-v2` (TOPN 32, raw cards, no reuse); exploration runs v2c.
  `--old-protocol-init` records the exemption in `v071.init_protocol`; the arm keeps the init's space manifest (same
  keys and samples); `fixed` may differ by `semfilter` only; imported receipts carry `semfilter_protocol` and pass the
  unchanged `check_receipt`. `semfilter_protocols.json` lists each configuration's protocols. ScienceWorld and WebShop
  use v2c from their init.
- **Stall rule** (`trimem/experiments/trace_v071_search.py`, `rag-stack-stall-v1`): after 3 consecutive completed rounds
  without a TRAIN HV gain above 1e-6, the next round is forced; the counter resets after it. Round HV: normalized with
  the frozen objective over complete never-pruned configurations plus NoMemory, own samples. A forced round picks all
  4 configurations by `uncertainty-novelty-v1` (max posterior log-volume, then novelty, then key), sequentially, over
  the latent proposal's pool without refinement; no acquisition call. Round files record `force_exploration`,
  `stagnant_rounds`, `round_hv`, `selection_rule`; `hv_rounds.jsonl` logs every round.
- `--semfilter v2` on `tools/prepare_trace_v071_campaign.py` is exactly the v0.7.0 command.
  `fixed.prefilter` keeps its template text; `fixed.semfilter` is authoritative for v0.7.1 runs.
- Queue: `results/trace_v070_e2e_20261105/run_v071_queue.sh`. Hash guard: `tools/trace_v071_hash_guard.py`.
