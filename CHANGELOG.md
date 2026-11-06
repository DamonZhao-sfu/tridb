## [TRACE v0.7.1] - 2026-11-05
### Features
- SemFilter v2c: TOPN 16, compact judge card (`judge-card-compact-v1`), reuse of a candidate's score across retrievals of one episode, with `cache_hits` / `judged_new` / `tokens_saved_est` per retrieval.
- ALFWorld mid-run switch: the v0.7.0 init on the old SemFilter is imported under a recorded exemption (`--old-protocol-init`, `v071.init_protocol`); exploration runs v2c; every receipt carries `semfilter_protocol`. ScienceWorld and WebShop use v2c from their init.
- RAG-Stack stall rule: after 3 rounds without a TRAIN HV gain above 1e-6, the next round is forced; all 4 picks follow `uncertainty-novelty-v1` sequentially; no acquisition call. `hv_rounds.jsonl` logs every round.
- New: `semfilter_v071`, `trace_v071_runtime`, `trace_v071_search`, `trace_v071_episode_worker`, `trace_v071_live`, `prepare_trace_v071_campaign`, `trace_v071_hash_guard`, `trace_v071_smoke`, `semfilter_v071_measure`, `run_v071_queue.sh` (not started).
### Design Rationale
- Additive modules plus runtime patches, because every v0.7.0 module is hashed by the completed ALFWorld init. The prepare option therefore lives in a new wrapper, not in `prepare_trace_v070_campaign`.
- `snapshot.with_pending` does not change posterior variance, so forced picks also condition the log-volume on earlier picks through the GP joint covariance (recorded as `conditioning`).
- The forced pool omits latent-refined candidates, because producing them calls the acquisition.
### Notes & Caveats
- Compact cards and reused scores are not validated against live judge decisions. Reused scores were judged against an earlier target of the same episode.
- On ALFWorld, init and exploration configurations differ in SemFilter protocol: their token and latency costs are not directly comparable, and the switch alone can raise HV in early rounds.

## [TRACE space v0.7.0] - 2026-11-05
### Features
- Configuration space v0.7.0 (`plan_schema_v070`): `rank_impl` optional (ColBERT / cross-encoder / none; LLM removed); `graph_direction` fixed Both, `retrieval_unit` fixed whole. Rank none = historical VGR / P1 / P2 (pool ordered by query cosine, top `returned_k` injected).
- NoMemory protected baseline: own ledger outside the BO budget, never pruned, TRAIN 100 in the init stage (same runtime as TRACE episodes, official AgentGym evaluator under the strict patches), imported into the arm, always tested on TEST200, candidate of the TRAIN Pareto set / OneConfig / router, latency-cap front reference.
- Per-configuration TRAIN samples (Amendment A): every configuration and NoMemory draws its own 100 tasks from the eligible TRAIN pool, stratified to the TEST task-type mix (largest-remainder quotas), seed digest([20261022, key]), stratified interleave order (every 20-task prefix follows the mix). Stored in the ledger `samples` table and `samples.json`; GP observations, staged pruning, latency pruner, candidates, TEST selection, router and HV export use each configuration's own tasks (unpaired).
- Staged pruning on own prefixes with the plan guard keyed by the 6 executor plans; latency-cap pruner v070 (factor ALF 2.0 / Sci 3.0 / WebShop 2.0, NoMemory in the front); TEST on Pareto(full never-pruned BO configurations U NoMemory).
- Tools: `prepare_trace_v070_campaign` (shared-init / arm), `trace_v070_live` (`--preflight-only`, `--baseline-only`), `trace_v070_holdout`, `trace_v070_router`, `latent_readiness_v070`, `trace_v070_hash_guard`, `trace_v070_smoke`; queue `results/trace_v070_e2e_20261105/run_v070_queue.sh` (ALF -> Sci -> WebShop; not started).
### Design Rationale
- Additive modules plus runtime patching of driver module attributes (`v070_patches`), restored on exit; the spawned latent worker re-enters the patches (`latent_search_worker_v070`, proof file `<round>.v070.json`). No file hashed by a live, paused or queued run changed (hash guard before / after).
- The manifest's `tasks` is the eligible pool, so the stock launcher fails closed (100-task rule, `rank_impl` none refused by the stock `JointSpace`); the stock validator still runs on a 100-task view.
- Budget (Q1): 5,000 TRAIN episodes per arm including the 10 init configurations, <= 80 configurations; NoMemory outside it.
### Notes & Caveats
- The dense kNN router curve needs one shared panel and does not apply to per-configuration samples; `trace_v070_router` fits per-configuration predictors on own samples instead.
- NoMemory TRAIN latency is measured in its own c16 stage (no concurrent PG / ColBERT load); the latency-cap reference may be tight.
- ALFWorld shared init prepared (`results/trace_v070_alfworld_init_20261105`, preflight passed); no episode has run. Spec addendum: docs/trace_space_v0.7.0.md.
- Fix round 1 (adversarial review): latency cap applied at the staged-pruning barrier on exact own prefixes (no pruner process); decisions committed in the ledger before logging, never recomputed; `prepared.json` completion marker; fail-closed hash guard with run discovery; c16 / max_tokens 4096 / NoMemory parity enforced; holdout lock before selection; only the lock owner writes status. User decision: router and OneConfig candidates = BRANE fuzzy Pareto set; TEST set = 3-objective Pareto U fuzzy set U NoMemory.
- Fix round 2: NoMemory always a router / OneConfig candidate; latency-cap front excludes NoMemory (`latency_front` false) and the cap never prunes the last configuration of an executor plan (`guarded: plan-last`); hash-guard checks aggregated independently; failure status written while holding queue.lock; committed stage decisions back-filled at launcher startup and before every round; NoMemory runtime parity enforced on complete ledgers and on import.

## [TRACE-PR three-dataset campaign v0.5.0] - 2026-11-03
### Features
- Search space: Rank mandatory, SemFilter optional (filter_order none/before/after = VGR+Rank / P1+Rank / P2+Rank, 5,040 configs; schema trace-pr-v050-optional-filter-v1).
- Multi-dataset campaign (ALFWorld / ScienceWorld / WebShop): prepare tool with per-dataset pool, env ports, max_round (30/30/10), frozen read-only memory clones (new trace_ablation_web_22467f614f8be112), 5 Sobol + 5 LLM + 10 x q=4, TRAIN100, TRAIN-Pareto on TEST200.
- Robust unattended execution: attempt identity written before launch; failures classified (not started -> released; terminated without result -> bounded retry; uncertain -> stop for operator); stale-claim recovery by process scan; receipts validated before done; runtime identity (model, max_round, memory clone, evaluator source hashes) bound into every receipt; frozen-memory write counter re-checked each round; CLI round/retry take the run's queue lock; breaker check atomic with claim.
- Search time limits removed (search phase 65 s, PR 45 s slice, proposal 60 s); step and unique-input budgets remain.
- D3 diagnostic (logging only, after selection): canonical 1-/2-field neighborhood gains and interaction g(ab)-g(a)-g(b).
- Queue unit trace-pr-three-dataset-20261103: c16 real smoke, then Sci -> ALF -> WebShop.
### Design Rationale
- Old best TRACE plans (ALF VGR, Sci P2) were outside the two-plan space; user chose filter optional, rank mandatory.
- Adversarial review (Codex, 3 lenses) found duplicate-episode and identity risks in recovery/retry; all high findings fixed except WebShop product isolation, which follows the user's 2026-10-23 option B (shared products) and is now asserted at prepare.
### Notes & Caveats
- The live path had never run a real episode (missing --task_name); the c16 smoke gates every campaign.
- Live smoke found two more defects before any campaign episode: the worker's memory-policy guard rejected top-level launcher json (moved to attempt/), and the receipt double-counted retrieval in latency (episodes.jsonl episode_ms already includes it).
- Runtime synced with historical TRACE runs (semfilter-v2, TOPN 32; retrieval slots / semantic workers at defaults).
- Runner arguments synced with historical runs: action_format react for ScienceWorld/WebShop (function_calling for ALFWorld), agent max_tokens 4096. Run suffix f.
- Analyzer JSON parsing fixed (Qwen wraps replies in ```json fences; every call had failed, so LLM initialization fell back to Sobol and the analyzer channel was empty). ScienceWorld run amended: round 1 kept, round 2 re-proposed by the LLM; amendment receipts in each run directory. Lease taken in shared mode like the other evaluation queues; the idle-GPU gate still guards the start. Current run suffix: 20261103e.
- Real smoke and campaigns wait for the shared evaluation lease (other experiments running at prepare time).

## [TRACE PR v0.4.4] - 2026-11-03
### Features
- PR defaults: no lr decay (one persistent Adam, lr 0.025, bo_pr default) and no re-anchoring; stagnation is counted (`stagnant_steps`) and never stops a run. `decay=True` and `patience=k` keep the v0.4.3 behaviours.
- v0.4.3 merged into the working tree on user request (Generalized PR discrete formula with full support, 200 steps x 128 samples).
### Design Rationale
- Without decay the gradient alone walks multi-level ordinal moves (explainer run: seed_count 4->128, enumeration rank 1 at step 145). On the round-10 GP 18/20 runs reach enumeration rank 1 and PR supplies 30 distinct candidates instead of 8.
### Notes & Caveats
- Changing the optimizer files halts the live campaign trace_pr_scienceworld_sobol5_llm5_30_20261103 at its next artifact check; relaunch needs re-frozen hashes.
- Factorized PR misses optima that need two fields to change together (toy: 5/20). Spec: docs/superpowers/specs/2026-11-03-trace-joint-plan-pr-search-design_v0.4.4.md.

## [TRACE PR v0.4.3 (staged)] - 2026-11-03
### Features
- Ordinal PR uses the Generalized PR discrete formula on GP-distance level positions (mean gap 1), extended to full support as an ordered logit; categorical PR is unchanged.
- PR optimizer budget follows bo_pr: 200 steps, 128 MC samples, theta in [0,1], Adam lr 0.025, optional lr/i^0.7 decay (on by request).
- Stagnation re-anchors the run at its best unanchored submittable draw (Sobol fallback) instead of stopping; every re-anchor is logged.
### Design Rationale
- v0.4.2 runs stopped after 5-14 steps and moved each ordinal field by at most one level (toy: 0/20 reach). v0.4.3 reaches the toy optimum in 20/20 seeds; on a real GP round its 8 candidates equal enumeration ranks 1-8.
- bo_pr's decay option defaults to off; with decay the pure gradient reaches 0/20 toy optima, so reach depends on re-anchoring.
### Notes & Caveats
- Staged in results/trace_pr_v043_staged (patch applies with git apply -p0). Not applied: the live campaign trace_pr_scienceworld_sobol5_llm5_30_20261103 re-hashes the optimizer files each round and would halt.
- All 20 runs converge to the same top 8, so PR adds only 8 distinct candidates per round. Spec: docs/superpowers/specs/2026-11-03-trace-joint-plan-pr-search-design_v0.4.3.md.

## [MemP/AdaMEM c16 evaluation v0.1.3] - 2026-11-03
### Features
- Rerun the same three fixed TEST200 panels in a separate c16 execution, with eight episode processes on each GPU/environment half; retain c1 artifacts and reuse frozen memory indexes.
- Gate AdaMEM construction on all 600 validated MemP c16 results; publish only complete c16 comparisons. Persist c16 as the policy for subsequent baseline/ablation evaluations.
### Design Rationale
- Match existing c16 comparison conditions. Fresh episode processes bound memory retention, a single resident index avoids index replication, and per-unit plus aggregate hard limits remain enforced.
### Notes & Caveats
- Source-frozen baseline code remains unchanged. The c16 wrapper has a separate seal, preserves the ScienceWorld null-description repair, and measures client-side retrieval waiting time.
- New outputs: `results/memp_adamem_c16_20261103`; the prior c1 results are not c16 latency evidence.

## [TRACE PR two-stage initialization and TEST200] - 2026-11-03
### Features
- Evaluate five legal Sobol configurations, then five LLM-designed initialization probes informed by the completed Sobol TRAIN measurements; preserve proposal provenance and resume without redispatch.
- Freeze one uniformly sampled source-TRAIN100 panel, retain 30 four-channel exploration configurations (q=4, final q=2), and evaluate the final TRAIN Pareto set on canonical TEST200 in a separate ledger.
### Design Rationale
- Match the inspected RAG-STACK five-plus-five protocol. Record Sobol fallback explicitly when the LLM supplies invalid, duplicate or too few proposals. Freeze objective normalization only after all ten initialization panels complete.
- Keep TEST evidence outside analyzer inputs, GP fitting and acquisition selection; preserve raw negative ScienceWorld scores and existing resource limits.
### Notes & Caveats
- Supersedes the queued 20+30 run before any real episode was dispatched; retain its directory unchanged apart from a supersession receipt.
- Complete 10+30 synthetic smoke and focused tests passed. Synthetic labels are not ScienceWorld results. External reviewer CLI calls failed due to a model/CLI version mismatch; no external review pass is claimed.

## [TRACE PR ScienceWorld 20+30 campaign] - 2026-11-02
### Features
- Parameterize initialization/total configuration budgets, support a final q=2 batch, and stop new episode admission after infrastructure failure.
- Add an isolated raw-score AgentGym evaluator bridge and a resource-gated background queue for 20 fresh initial plus 30 four-channel configurations on one 100-task TRAIN panel.
### Design Rationale
- Keep historical 10+20 manifests compatible while giving this run explicit 50-configuration/5,000-episode limits. Reuse the read-only memory graph and local inference services; collect new measurements.
### Notes & Caveats
- The queue waits for existing GPU jobs and the evaluation lease; CPU quota is 4 cores and worker-tree memory cap is 16 GiB. Shared inference/environment servers are separately guarded by host-memory admission checks.
- Runtime artifacts: results/trace_pr_scienceworld_20plus30_20261102. Remote analyzer usage is disabled. External review CLI remains unavailable; synthetic protocol validation is not reported as ScienceWorld evaluation.

## [TRACE staged search v0.4.3] - 2026-11-02
### Features
- Add a versioned entry with five Sobol configurations followed by five feedback-informed LLM configurations; subsequent BO batches default to four.
- Add measured-NHV stagnation detection, one uncertainty-selected escape slot, pending-aware remaining slots, and durable initialization/proposal replay.
### Design Rationale
- Freeze objective normalization after all ten panels; count stagnation from completed configurations so restarts cannot change the policy.
- Preserve the existing source-frozen 20+30 queue by adding separate modules rather than changing its hashed code.
### Notes & Caveats
- New entry: trimem.experiments.trace_staged_plan_search. The existing queue is not migrated or restarted. See docs/trace_staged_search_v0.4.3.md.
- Tests use synthetic labels; no live experimental results are produced. PR remains the existing local bo_pr-inspired implementation.

## [TRACE joint plan PR v0.4.2] - 2026-11-02
### Features
- Add a fixed-task-ANN joint plan/parameter schema (18 decisions, 24 GP features), three raw-score/cost GPs, six PR restart sources, qLogNEHVI refinement and pending-aware q=4 selection.
- Add fail-closed executor capabilities, schema-aware analyzer advice, atomic configuration/episode admission, fresh or audited initialization, guarded CPU/RSS budgets and a reproducible fixture smoke.
### Design Rationale
- Optimize acquisition over legal discrete configurations through PR distribution parameters; remove single/multi vector selection and its failure-entry slots. Preserve historical optimizer campaigns and raw negative environment scores.
### Notes & Caveats
- Entry point: trimem.experiments.trace_joint_plan_search. Verified live mapping currently uses before/after LLM filtering, native BFS and ColBERT; operator extensions remain disabled pending their own executor tests.
- Smoke labels are explicitly synthetic. Inactive ordinal initialization and checkpoint scope are documented in docs/trace_pr_v042_implementation.md. External review CLI was unavailable due model/version incompatibility.

## [MemP and AdaMEM three-dataset execution] - 2026-11-02
### Features
- Add isolated source-backed MemP offline and AdaMEM-LOW construction and evaluation adapters for ALFWorld, ScienceWorld and WebShop, with fixed TEST panels and complete per-call accounting.
- Add bounded two-GPU build workers, checkpoint identities, paused-competitor guards, hourly background status, source-frozen recovery, and complete-only HTML publication.
### Design Rationale
- Preserve upstream workflow/strategy algorithms while adapting environment interfaces and model endpoints; keep long-term TEST memory read-only and prevent task/family leakage.
- Store step values on disk and run one formal episode at a time so memory usage and latency remain interpretable.
### Notes & Caveats
- This is a shared-model adaptation, including a ScienceWorld port, not a claim to reproduce upstream paper tables. Full construction and TEST measurements are pending execution.
- DeltaMEM and G-Memory remain deliberately frozen; the new queue never thaws them automatically.

## [ScienceWorld direct Pareto TEST evaluation] - 2026-11-01
### Features
- Freeze each completed ablation arm's two-dimensional TRAIN frontier union and evaluate unique configurations on the same 200 held-out queries, with separate budget accounting and resumable receipts.
- Resume the completed Full TRACE router from validation with an exhaustive-panel seed metadata adapter.
### Design Rationale
- Separate fixed-configuration comparison from validation-selected router comparison; retain frozen sources and the original router budget.
### Notes & Caveats
- Shared configurations are measured once across arms. Dispatches conservatively charge missing scheduled pairs on retry, with at most 400 pairs per batch and 1,000 retry slots. TEST measurements do not feed configuration or router selection.

## [ScienceWorld ablation curve editor] - 2026-11-01
### Features
- Add per-arm observed Pareto curves and independent Tokens/Latency point selection with browser autosave in the ablation TRAIN section of qa-trajectory.html.
- Add separate TEST curve plots for configurations with all 200 held-out queries complete, with independent controls and selection storage.
### Design Rationale
- Compute each two-objective frontier from complete samples, including each arm's shared initialization; keep manually selected lines separate from observed fronts.
### Notes & Caveats
- Manual lines can include provisional or dominated points. Selections are scoped to the displayed snapshot; this UI update does not refresh experiment measurements.

## [ScienceWorld TRACE ablation preparation — in progress] - 2026-10-29
### Features
- Add raw-score ablation protocol, candidate-source controls, initialization recovery, split preparation, resource accounting, isolated resume entry points, and router evaluation helpers.
### Design Rationale
- Preserve negative scores and require identity-verified common initialization while keeping historical runs untouched.
### Notes & Caveats
- The execution path remains disabled: live service and memory identities, complete end-to-end execution, and independent review are not yet verified. No formal ablation has been launched.

## [Cross-method c16 predictor study] - 2026-10-29
### Features
- Add separate per-dataset prediction experiments using validated historical c16 training and test episodes across exploration methods.
### Design Rationale
- Expand training coverage while preserving disjoint task IDs and recording execution provenance and repeated measurements.
### Notes & Caveats
- ScienceWorld uses raw environment score; incompatible budgets, graph-specific or unverified runs are excluded. Historical TEST has been inspected and existing deployed routers are unchanged.

## [ScienceWorld cross-search training experiment] - 2026-10-29
### Features
- Train shared ScienceWorld quality, token and latency predictors on compatible measured outcomes from multiple historical search methods.
### Design Rationale
- Expand task and configuration coverage while retaining task-grouped validation and source provenance.
### Notes & Caveats
- Uses raw environment score and compatible c16 training campaigns; repeated task/config measurements are median aggregated. Historical TEST is diagnostic and the live router is not replaced.

## [ScienceWorld raw-score predictor] - 2026-10-29
### Features
- Add a separate ScienceWorld predictor experiment using original environment scores, including negative values, with train-only grouped model selection.
### Design Rationale
- Read original recorded scores rather than reconstructing negative values from clipped and normalized quality labels.
### Notes & Caveats
- The existing optimizer and deployed router retain their normalized quality protocol; raw-score artifacts cannot be mixed with them. Positive-score-only MAPE excludes nonpositive outcomes and reports its coverage.

## [Shared metric predictor experiment] - 2026-10-29
### Features
- Add an offline historical-data experiment comparing three configuration-aware metric predictors with a true three-output predictor, with query-grouped model selection.
### Design Rationale
- Share observations across configurations and compare candidate model families while keeping quality, token and latency measurable separately.
### Notes & Caveats
- Conflicting duplicate task/config observations and stale snapshots of the current experiment are excluded. Historical TEST results are diagnostic because that set has been inspected; the experiment does not replace the live router.

## [Configuration-aware BRANE router] - 2026-10-29
### Features
- Default router training performs separate per-configuration model selection for quality, token and latency using query embeddings and the configuration's plan and knobs.
### Design Rationale
- Preserves BRANE-style per-objective cross-validated selection while exposing the requested configuration and query inputs; legacy BRANE and joint models remain available.
### Notes & Caveats
- Each configuration's knobs are constant within its training samples. Historical frozen artifacts retain their original features and models; new results require retraining.

## [TRACE two-stage analyzer] - 2026-10-28
### Features
- New TRACE runs use a separate evidence-diagnosis call before the existing local-refinement guidance call, informed by Agentic AutoRAG's Diagnoser/Proposer workflow.
- Validate diagnosis evidence against TRAIN observation IDs and account for both LLM calls in analyzer usage. Log both stages and replay them without calling the model again.
### Design Rationale
- Keep the existing acquisition-driven evaluation selection, while separating evidence extraction from search guidance and preserving auditable cross-round hypotheses.
### Notes & Caveats
- Diagnosis currently receives aggregate configuration observations, not full per-task failures, so it cannot attribute individual failures to retrieval or execution. Existing recorded runs retain their original one-call workflow; new comparisons require fresh matched runs.

## [External baselines at chosen TRACE configs + CA-RAG + Agentic AutoRAG] - 2026-10-27
### Features
- `external_baselines.mapping.BEST`: one TRACE configuration per dataset (ALFWorld r14c1:VGR, ScienceWorld r5c3:VGR+SemFilter, WebShop r1c6:V+SemFilter+G+Rank); SemFilter plans now transfer their retrieval part only (SemFilter / Rank listed in `not_transferred`), external arms run with runner plan VGR. `external_baselines_test` uses these arms; queue `scripts/external_baselines_best_queue.sh` (G-Memory, ExpHarness, EXG) takes the shared evaluation lease, builds missing memories, smokes 4 items, then runs TEST.
- EXG memories built for all three datasets (LLM-free).
- CA-RAG (`vendor/ca-rag` @ 4861e04, MIT): `external_baselines/ca_rag_router.py` ports its prior-utility router; bundles = deployed Opt-PerQuery candidates with their TRAIN means; reported as a OneConfig arm (`<ds>/ca_rag/report.json`).
- Paper Table 4 (matched-quality savings): CA-RAG row from `paper/tools/ca_rag_test_pool.py` (bundles = all TEST-measured static configurations with TEST-mean priors, retrospective like the table); only that row was inserted, other rows untouched.
- Agentic AutoRAG (`vendor/Agentic-AutoRAG` @ e1c5590, MIT): search method `agentic-autorag-v1` (`trimem/optimizer/agentic_autorag.py`, prompts `agentic_autorag_*_v1.md`): Diagnoser over per-task TRAIN records + k sequential Proposer calls per round, upstream repair loop, two-objective (quality, tokens) frontier, `recommend()` for the whole-pipeline ablation; LLM = TRACE analyzer client (gpt-5.6-luna, effort low), per-round replay records.
### Design Rationale
- User decisions 2026-10-27: compare systems at TRACE's max-TRAIN-quality configuration; CA-RAG as OneConfig; Agentic AutoRAG as exploration-method and whole-pipeline ablation. Vendor code could not be executed in this environment, so the needed parts were ported with attribution instead of imported.
### Notes & Caveats
- CA-RAG's only per-query signal (query length / question cues) is ~0.03 on agent goals, so it routes every TEST task to one configuration (= Opt-OneConfig SLO-1 on all datasets); its latency / token references are rescaled to our units with the paper's headroom ratios (ALFWorld top-3 utilities differ by <= 0.003).
- TRACE has a full 200-item TEST arm only for ALFWorld r14c1; ScienceWorld r5c3 / WebShop r1c6 references are still open.
- Agentic AutoRAG: trial budget not yet passed to the Proposer; `recommend()` not yet wired into a campaign/TEST script; failure-attribution heuristic (goal-overlap 0.3) unvalidated. No live run yet.

## [Task-complementary DLS (trace-tc)] - 2026-10-27
### Features
- Add opt-in search design `task-complementary-v1` (`--search_design task-complementary-v1`): identical to `three-channel-v1` (channel-slots-v6 layout, analyst, crossover, coverage, stall recovery) except the `dls` slot, whose DLS walk score and slot arbitration become A_TC = member mean of sum_q p_q [H_q(S u {x}) - H_q(S)], the per-query 3-D hypervolume improvement over completed configurations + NoMemory + this batch's believers (paper eq. task-complementarity).
- New `trimem/optimizer/task_complementary.py` (exact vectorized 3-D HVI with per-query non-dominated pruning, frozen ln-cost scales, `TCObjective`, relaxed one-grid-level support, `TCScorer`) and `task_complementary_driver.py` (TRAIN rows incl. pre-pruning rows of pruned configs, frozen panel / features / scales, epoch-gated `validation_gate` in y space). `session_pool.suggest(tc=...)` keeps the ordinary dls-slot starts, re-ranks `dls-pool` by A_TC, adds 2 `dls-specialist` starts, arbitrates by unbounded A_TC (whole pool if no supported DLS endpoint) and records `slot_info["task_complementary"]`. NoMemory is required in S; scales frozen from round-0 rows; gate re-checked every round.
- Offline replay tool `tools/tc_replay.py` (no episodes) re-plans finished three-channel rounds under the new design.
### Design Rationale
- User decision (2026-10-27): no extra slots; replace only the existing DLS scoring so `trace-tc` differs from `trace-hv` in one place. Per-query HV first, aggregation second, so configurations with an unremarkable mean but new per-task trade-offs score positively. S uses model predictions of the same ensemble member (sparse per-config samples, binary ALFWorld outcomes) rather than raw cells.
- Router-free (the paper's channel does not assume a deployment router); cold-start / degraded gate falls back to the original qLogNEHVI DLS, recorded per pick.
### Notes & Caveats
- Ensemble expectation of HVI is optimistic for uncertain configurations (intended exploration bias, not a calibrated gain). Mean-based staged pruning and the deployment catalog filter are unchanged and may still drop specialists.
- No live profiling was run. Validated by 38 new tests, 19 regression files (284 passed, 6 skipped), a Codex adversarial review (`review_verdict.md`) and an offline replay of the ALFWorld trace-hv rounds 1-2 (mechanism active in both; `replay_report.md`), all under `results/task_complementary_20261027/`.

## [Hydra Local Refinement profiling] - 2026-10-27
### Features
- Add opt-in `hydra-local-v1`: TRAIN-only configuration/query response ensemble, router-relative positive utility acquisition, response-group and behavior-archive local search, and batch complementarity arbitration.
- Preserve existing deployment router and crossover/graft generation. Bound local evaluations, prediction memory and coverage replenishment; retain original HV baseline path.
- Add a separate frozen ALFWorld comparison controller: paired 8+8+8 configurations per method, 100-query TRAIN panels, fresh shared NoMemory canary/reference, and existing router on 200 held-out TEST queries after both discoveries freeze.
### Design Rationale
- Discover configurations useful to Opt-PerQuery across query groups and preferences. Model readiness is diagnosed on held-query and held-configuration folds; unsupported plan/knob regions receive novelty exploration rather than utility-based selection.
- Separate round utility scales from epoch behavior descriptors. Preserve observed outcomes and attempt ledgers, bind replay inputs and completed output hashes, and use the shared evaluator queue lease.
### Notes & Caveats
- Bootstrap uncertainty/group stability are heuristic; this combination is not yet an experimentally established novelty or efficiency result. PIAC is deferred.
- Development occurred in an isolated snapshot; the running unified-search24 frozen workspace and its existing controllers were not modified. Live comparison results are pending under `results/alfworld_hydra_local_20261027/`.

## [TRACE front-gap stall recovery] - 2026-10-26
### Features
- New development TRACE profiling runs default to `stall_recovery=front-gap-v1`, enabling stagnation-triggered forced exploration with channel-slots-v5/v6 in both AgentEnv and tau2 drivers. Ordinary layouts and explicit Sobol-only rounds retain their behavior.
- Record the recovery rule in search settings and proposal metadata. Historical resumes retain their recorded rule; changing it requires a fresh output directory. `--stall_recovery legacy` preserves the historical rule for a new run.
### Design Rationale
- Reuse the existing noise-aware relaxed-front stagnation detector and frontier-gap landing-probability acquisition, with existing crossover/local-search/coverage fills. This is not a new delta-HV stopping criterion or a router change.
### Notes & Caveats
- Main development sources only: the frozen unified-search24 workspace and its running TEST queue are unchanged. No profiling workload was launched by this change. Forced requests can fall back when no usable frontier gap exists.
- Validation: 81 related regression tests passed (1 skipped, 13 deselected), plus 4 driver forwarding cases and a real-GP legacy v6 eight-pick case. Three external static reviewers reported no concrete defects; no new Ruff diagnostics relative to the pre-change files. This is not a live profiling quality or efficiency result.

## [TEST work-conserving slots] - 2026-10-26
### Features
- Replace four configuration completion barriers with16 independently refilled single-worker slots,8 per inference endpoint. A completed shard immediately releases its slot to the next configuration; each configuration has at most4 workers.
- Drain in-flight old-runner tasks before handoff. Retain2264 WebShop and3795 ScienceWorld measured pairs, preserve manifests and prior ledgers, and keep WebShop TEST before ScienceWorld TEST.
### Design Rationale
- Fixed disjoint per-configuration task shards avoid duplicate execution and task-by-task interpreter startup. Explicit job ownership before spawn and positive cleanup receipts protect worker limits during failures.
- Maintain native10/30 step limits, model/memory, TEST membership, and total concurrency16. Record the scheduling regime in each new measurement and scheduling_change.json; unchanged latency is not claimed without measurement.
### Notes & Caveats
- All frozen engine files remain unchanged. New queue, review and probe evidence: results/unified_search24_20261024/test_slotfill_20261026/.17 tests passed and real command-construction probes passed for both datasets without extra episodes.
- Goal remains paused; detached queue handles normal continuation and bounded transient retries.

## [TEST budget validation and automatic recovery] - 2026-10-26
### Features
- Fix missing database max_round metadata being misread as30 for native10 WebShop TEST. New rows explicitly record the budget; legacy rows require matching run_args, identity and bounded episode steps.
- Resume WebShop TEST with755 measured pairs (one complete configuration), then ScienceWorld TEST with3795 retained pairs. Preserve old ledgers and record each recovery attempt separately.
- Automatically retry missing tasks after transient subprocess, stall or incomplete-file failures, with at most3 recovery attempts; identity/code/data errors still require diagnosis. Retain concurrency16 and native10/30 budgets.
### Design Rationale
- Reuse valid measurements rather than rerun completed tasks; keep historical manifests and database rows unchanged.
- Resume from durable attempt receipts; retain queue ownership through worker cleanup and separate evaluation deadlines from publication.
### Notes & Caveats
- Platform goal currently reports paused; the detached recovery queue runs independently. ALFWorld and external baselines remain held.

## [Sequential TEST continuation] - 2026-10-26
### Features
- Launch WebShop TEST before resuming ScienceWorld TEST: four configurations x four workers, total16. WebShop evaluates68 unique configurations from96 proposals,200TEST tasks each,10 steps; ScienceWorld retains its30-step limit.
- Reuse3795 measured ScienceWorld task/config pairs (17 full configurations); authorize one continuation for405 previously interrupted reservations, with a separate durable attempt ledger.
### Design Rationale
- Deduplicate full execution plans and parameters across methods while preserving all search provenance. TEST outcomes never feed TRAIN search.
- Persist verified evaluation completion separately from HTML publication so display failures cannot block the next dataset.
### Notes & Caveats
- No automatic evaluation retries. ALFWorld and external baselines remain held. Frozen runner code is unchanged; launch review and command probes are saved under results/unified_search24_20261024/test_followup_20261026/.

## [WebShop native10 priority and outcome reporting] - 2026-10-26
### Features
- User-paused ScienceWorld TEST; replace134TRAIN editorpoints with27provenance points from17complete200-task TEST configurations. Partial task results remain on disk; TRAIN NHV and saved selection file are preserved.
- Release the reviewed WebShop pipeline: native10 NoMemory reference, then MO-TPE/GP-BO/LHS/TRACE, seed20261023,3rounds x8configurations,100TRAIN tasks/config and20-task pruning checkpoints.
- Add read-only native10 reward/TaskScore/SR reporting with per-task CSV, configuration CSV/JSON and existing HTML metric-selector support for completed search points.
### Design Rationale
- Only complete TEST measurements replace ScienceWorld TRAIN points. WebShop raw reward is the optimization quality; SR is separately computed as fraction of reward==1, not conflated with reward.
### Notes & Caveats
- ScienceWorld controller stopped gracefully and retained completed files and rows; interrupted reservations remain charged. Future continuation must explicitly reconcile unfinished tasks and already-archived TRAIN editor exports before publishing again.
- The reporting watcher reads results and updates display metadata only; no automatic evaluation retries and no changes to frozen search/runner sources. External memory baseline processes remain paused.

## [ScienceWorld TEST shared concurrency v2] - 2026-10-26
### Features
- Increase to four simultaneous TEST configurations with four workers each (two per inference endpoint), total cap16; drain the current two configurations before handover.
- Record every method/seed/round/position for each unique configuration in JSON/CSV, active status and final displayed point provenance.
- Persist task-level TEST quality under an evaluation identity; reuse completed measurements across identical proposals and reject conflicting cache values. An on-demand CSV/JSON progress report combines metrics and all search origins.
### Design Rationale
- Shared configuration quality is measured once across the144 proposals (104unique configurations). Retain historical worker allocations instead of relabeling previous16/8-worker measurements.
### Notes & Caveats
- Historical TRAIN quality or results without matching TEST membership, memory, model, code and step-limit identity are not substituted. This cache does not bypass missing cost measurements; already complete compatible measurements are reused by the scheduler.
- The previous scheduler and frozen execution sources are retained unchanged; the new reviewed handover and scheduling plan have separate files.

## [ScienceWorld TEST shared concurrency] - 2026-10-26
### Features
- Schedule two TEST configurations concurrently with eight workers each, keeping the global worker cap at 16 (eight per inference endpoint); refill a configuration slot after completion.
- Preserve the first serial configuration's completed measurements and no-retry ledger via a reviewed, exact-PID drain-and-handover controller.
### Design Rationale
- Share available capacity across configurations without doubling inference load or restarting in-flight episodes. Validate each completed configuration before admitting its replacement.
### Notes & Caveats
- The initial configuration used 16 workers; subsequent configurations use eight each under the shared total cap. Exported point provenance records this scheduling difference; latency is not claimed to be invariant to allocation.
- All TEST task IDs, memory/model identity, step limits and total episode budget remain frozen. The parallel amendment is separate from the original manifest, and no frozen search engine files were changed.

# Changelog

## [ScienceWorld search TEST reporting v0.1.0] - 2026-10-25
### Features
- Queue a 200-task TEST evaluation after the current ScienceWorld TRACE search and before WebShop, covering all proposals from the six authorized method/seed runs, including TRAIN-pruned proposals.
- Deduplicate configurations across runs, preserve search provenance and point IDs, and replace their curve-editor coordinates with measured TEST quality, tokens, latency and steps.
### Design Rationale
- Keep TEST separate from search, pruning and TRAIN NHV. Use the frozen search execution workspace with ScienceWorld transport, max_round 30, temperature 0 and 16 concurrent episodes; never reuse TEST feedback for optimization.
### Notes & Caveats
- A shared configuration has one TEST measurement, even when multiple methods/seeds proposed it. Such duplicated display points are not independent TEST repeats.
- No automatic retry. A failed TEST stage leaves the WebShop waiter held. Publication requires complete outcomes and a browser-checked staged HTML; original TRAIN archives remain preserved outside the curve publisher's input directory.

## [WebShop outcome reporting v0.1.0] - 2026-10-25
### Features
- Reconstruct Task Score and Success Rate for the 49 historical WebShop curve-editor points from their 200-task TEST outcomes; export point summaries and task-level evidence as CSV/JSON.
- Add a Task Score / SR selector for the existing tokens and latency scatter plots, with metric-specific observed Pareto fronts and both metrics in the point table.
### Design Rationale
- Preserve original point IDs, selections, costs and mean-reward optimization objectives; compute SR from individual full-success indicators rather than aggregate reward.
### Notes & Caveats
- Reuse the existing max-round-10 reporting view, including truncated historical references. The two views share selected point IDs. New points without measured SR are omitted from the SR view.

## [Unified TRAIN search protocol v0.2.0] - 2026-10-25
### Features
- Queue ScienceWorld → WebShop → ALFWorld, with TRACE last within each dataset; retain completed two-seed ScienceWorld MO-TPE/GP-BO results and use only seed20261023 for remaining searches.
- Add reviewed continuation of user-interrupted TRACE with preserved task results, attempt charges and resource history; WebShop gets a separately billed native-10 reference before search.
- Publish completed TRAIN candidates in the existing curve-point editor with method/seed/split provenance, Pareto flags, and portable selections; render two-seed NHV means and min/max bands only where measured.
- Audit checkpoint objectives and costs against retained evidence; export full/pruned configuration records, raw task steps, resource ledgers and Pareto/NHV archives.
### Design Rationale
- Background handoffs verify process identities and successful terminal receipts. Frozen evaluation code stays unchanged; reporting operates outside the hashed engine tree and locks HTML writes.
- Keep historical protocols separate and use additional cost after round0 with fixed normalization; avoid inventing missing costs or uncertainty bands for single seeds.
### Notes & Caveats
- Four ScienceWorld baseline runs have completed; the remaining searches are pending/running. Queue preparation and review do not establish full experiment completion.
- Historical TRACE has different initialization and incomplete cost accounting; its dashed curves are contextual comparisons only. Interrupted TRACE token coverage may remain unavailable.

## [External memory baselines v0.1.0] - 2026-10-25
### Features
- Add G-Memory (NeurIPS'25), AriGraph (IJCAI'25) and Graphiti as TEST baselines (`trimem/experiments/external_baselines/`, pinned clones in `vendor/`): one HTTP retrieval service per system in its own venv, the TRAIN pool exported once (`pool.py`), and an `--external` retriever branch in `agentgym_session_eval`.
- Transfer every TRACE SLO-selected OneConfig knob that has a counterpart (`mapping.py`: top_k, outcome policy, seed count, depth, fanout, sim gate, Rank) and run 3 systems x 6 configurations x 200 TEST tasks with `external_baselines_test` under controlled-c16-v1.
- `scripts/external_baselines_queue.sh` waits for the shared GPU queue, builds the full memories (construction cost recorded separately), then runs and reports each dataset.
### Design Rationale
- Only the retriever is swapped: actor, AgentGym evaluator, env servers, budgets, admissibility (own task / instance family / outcome, verified identical to `AgentEnvTaskAdapter.allowed`) and cost accounting are TRACE's, so quality, tokens and latency are comparable.
- Services isolate the systems' dependencies (py3.12, langchain/chroma, torch, graphiti/Neo4j) from the runner venv.
### Notes & Caveats
- Cross-task adaptations for cost: AriGraph extracts triplets once per stored session (not per observation) with session provenance; Graphiti ingests with `add_episode_bulk`. All deviations are listed in each `construction.json`.
- Full runs are queued behind unified_search24; only smoke runs (20-30 stored sessions, 2-4 tasks) exist so far.

## [Unified TRAIN search protocol v0.1.0] - 2026-10-24
### Features
- Prepare isolated TRACE, MO-TPE, GP-BO and LHS runs for seeds 20261023/20261024 with three eight-configuration rounds and 20-task pruning stages.
- Add paired-panel preflight, real profile-entry probes, and a serial supervisor with exclusive evaluation locking and completion checks.
- Publish completed ALFWorld LHS TEST results while preserving existing curve-editor IDs.
### Design Rationale
- Keep experimental changes outside the active evaluator source tree; share defaults plus Sobol initialization and exclude historical auxiliary observations from fresh comparisons.
- Preserve archived manifests and verify completed historical measurements against their frozen helpers.
### Notes & Caveats
- New searches have not started. Runtime review and reference compatibility checks remain; preparation is not evidence of completed experiments.
- Existing ScienceWorld/WebShop discovery occupies the shared evaluation queue and is left untouched.

## [TRAIN search-cost preview v0.1.0] - 2026-10-24
### Features
- Add interactive NHV curves over rounds, additional tokens, time and completed episodes for three datasets; ALFWorld includes MO-TPE, GP+LogNEHVI and LHS alongside historical TRACE.
- Rebase each run to its first recorded checkpoint (ALFWorld TRACE source round 6 → displayed round 0); retain original costs and round numbers in CSV.
### Design Rationale
- Preserve measured initial NHV and fixed log-cost normalization; show missing token receipts as unavailable.
- Keep presentation scripts in paper/tools, outside the frozen evaluation-code tree; serialize publication and preserve other HTML sections.
### Notes & Caveats
- Historical warm starts, search domains and cost accounting differ; these single-run traces are descriptive, not matched-budget superiority evidence.
- ScienceWorld and WebShop currently have TRACE only; MO-TPE has no complete token-cost trace.

## [ALFWorld validation-to-TEST evaluation v0.4.0] - 2026-10-24
### Features
- Stop the MO-TPE discovery chain at the user's early-stop decision; freeze the first 20 proposed configurations for a common 200-task validation evaluation alongside the historical TRACE catalog.
- Freeze the union of validation Pareto projections before independent 200-task TEST evaluation; export all selected TEST points, descriptive Pareto plots, and the Paper figure point-selector update.
- Cite the original MO-TPE paper and the subsequent GP+LogNEHVI baseline in the manuscript's experiment section.
### Design Rationale
- Validation membership and proposal-prefix selection do not use TEST outcomes. Shared configuration identities are deduplicated; existing browser point selections keep stable IDs.
- Keep evaluation in one configuration at a time with two halves of eight workers; preserve measured LLM-span latency separately from episode wall latency.
### Notes & Caveats
- The comparison is conditional on existing BO catalogs, which may have seen validation tasks; TRACE's historical discovery budget is unmatched. Later MO-TPE proposals and their costs remain preserved outside the primary 20-proposal checkpoint.
- No new NoMemory evaluations or automatic failed-attempt retries. The prior paired discovery launch must not be resumed; GP follows after current deployment evaluation under a newly frozen scope.

## [Paper curve editor v0.1.0] - 2026-10-24
### Features
- Add a six-panel paper-figure editor to qa-trajectory.html with independent point checkboxes, clickable markers, browser autosave and portable JSON import/export.
- Render the PDF from exported point IDs while retaining all measured sweep points; distinguish selected display curves from strict Pareto frontiers.
- Add an isolated WebShop quality-first sweep over 22 frozen-router weights, reusing compatible episodes and preserving historical sweeps.
### Design Rationale
- Keep author display choices separate from measured values and optimizer decisions. Gray background points preserve the full measured distribution.
### Notes & Caveats
- Manual curves can include dominated observations. TEST weight exploration is descriptive; final deployment selection uses validation.
- Browser autosave is origin/browser-specific. Downloaded JSON is the portable input to the figure renderer.

## [ALFWorld E2 asynchronous staging v0.3.0] - 2026-10-24
### Features
- Each configuration makes its own stage-50 pruning decision and, when retained, continues to 100 tasks as capacity becomes available. Total episode concurrency remains at most 16.
- Apply the same persistent admission and NoMemory reuse-only guards to synchronous and asynchronous launches; record stage-50 verification independently for each configuration.
### Design Rationale
- Reuse the existing asynchronous scheduler. Disable automatic drain/rebalance only for E2 so scheduling alone does not consume retry reservations.
- Continue the existing MO-TPE run with its experiment ID, samples, completed observations, attempt ledger and cumulative wall budget intact; archive the old identity before freezing the authorized protocol transition. GP uses the same asynchronous mode.
### Notes & Caveats
- Pruning uses the completed comparison configurations available at each decision time. Earlier evaluations remain labeled as the synchronous segment; already recorded decisions are preserved.
- A retained configuration can wait for a free slot under the 16-episode cap; it does not wait for every configuration to finish its checkpoint.

## [Single validation holdout v0.2.0] - 2026-10-24
### Features
- Freeze one reproducible validation holdout per dataset from original TRAIN: ALFWorld, ScienceWorld and WebShop each reserve 200 tasks, matching independent TEST size.
### Design Rationale
- Apply the user's replacement of five-fold selection with a single holdout; select OneConfig / PerQuery Pareto points on validation before evaluating frozen choices on TEST.
### Notes & Caveats
- Existing BO catalogs are reused, so validation is conditional on those catalogs. Predictor fitting and validation measurements remain pending. Earlier five-fold artifacts are retained as superseded history.

## [ALFWorld E2 historical NoMemory reuse v0.1.0] - 2026-10-24
### Features
- Freeze and reuse the historical c16 NoMemory TRAIN sample and result files; reject new NoMemory episode admissions in this mode. Candidate sampling and search quotas are unchanged.
- Record the future five-fold validation-first Pareto-selection protocol and prepare ALFWorld fold IDs, conditional on an existing BO catalog.
### Design Rationale
- Reuse the actual historical task sample instead of treating a new random sample as already measured; keep historical reference costs separate from new candidate search costs.
### Notes & Caveats
- Historical NoMemory lacks an exact serving revision and evaluator hash; reuse is explicitly authorized and identified as legacy reference reuse. Previously spent warmup costs remain recorded separately.
- The validation split is prepared; fold training, validation measurements and selection remain future work before TEST. BO is not rerun within folds.

## [Paired router comparison atlas v0.1.0] - 2026-10-24
### Features
- Add a C-only versus S+P+E section with measured quality/cost plots, shared TRAIN-budget selection, paired confidence intervals and explicit pending states.
### Design Rationale
- Collapse identical TEST routings while retaining every shared budget comparison; preserve existing atlas data and refresh the section as completed reports arrive.
### Notes & Caveats
- Smoke intervals condition on frozen models. Reported costs include recorded feature/scoring overhead; precomputed SPE embeddings exclude cold-query vector provisioning.

## [Router weight rescan and C-only smoke v0.1.0] - 2026-10-23
### Features
- Add an isolated 25-point logarithmic Q-T/Q-L weight sweep and prefer its complete reports in the HTML sweep panels.
- Add reproducible TEST subsets for paired SPE/C-only smoke runs, preserving full TRAIN data and frozen calibration.
### Design Rationale
- Reuse compatible query/configuration measurements, with separate experiment IDs and cache exclusions for historical batches that exceeded the 16-worker budget. Bound new small-configuration batches to eight arms.
- Resolve memory-policy-tagged experiment IDs when collecting C-only TEST outcomes so completed episodes are recovered correctly.
### Notes & Caveats
- TRAIN calibration does not guarantee monotone or budget-feasible TEST curves. A 32-query smoke comparison is preliminary evidence; measured feature and routing costs remain included.

## [Router characteristics paired experiment v0.1.0] - 2026-10-23
### Features
- Add isolated `router_characteristics` preparation, eight-question goal-only characterizer, resumable labeling, S+P+E / C-only training and TRAIN calibration, frozen TEST routing, deduplicated measured-pair execution and paired reports.
- Add explicit quality semantics and token overhead support to sparse router fitting/replay; existing callers retain their defaults. Register ALFWorld, ScienceWorld and WebShop in a local executable experiment queue.
### Design Rationale
- Share frozen profiling outcomes/catalog and the 231-weight calibration protocol while restricting treatment predictors to C. Keep historical routers and live profiling unchanged; do not silently substitute missing labels or incompatible TEST caches.
- Recover five missing ScienceWorld TEST goals from audited pre-action user instructions, without importing outcomes. Bind resume to stable model fields/server process identity, avoiding vLLM's volatile response timestamps and permissions.
### Notes & Caveats
- Preparation and synthetic validation do not constitute experiment results. Real labeling/episodes remain queued; WebShop requires a compatible completed profiling snapshot.
- Existing SPE vectors were precomputed: vector provisioning latency is not available. Report episode measurements plus separately measured routing overhead, conditional on the frozen feature archive, with this limitation explicit.
- Queue JSON records runnable commands; it does not start a scheduler. Code/input/service identity drift rejects resume and requires a new campaign directory.

## [WebShop episode budget max-round-v1 v0.1.0] - 2026-10-23
### Features
- `trimem/experiments/episode_budget.py`: opt-in `POOL_EPISODE_MAX_ROUND=10` (+ reason; WebShop only) recorded once in state.json `episode_budget` from the next round to be proposed; WebShop runner `--max_round 10` for profiling rounds >= from_round and every test / NoMemory / fixed-plan / PerQuery-sweep arm (`Arm.max_round`, `bo_pool_driver.test_max_round`); opt rows carry `protocol.max_round`.
- Derived-truncation-10 view of every 30-turn episode (rounds 0-1, NoMemory prefix, noise repeat, controlled-v1 references) used by GP points, HV/NoMemory anchor, prune comparators, profile table / router catalog, test reports; audit snapshot `<run>/episode_budget/derived-truncation-10_round<r>.json`.
- Runner quality reuse at max_round != 30 matches only same-budget sources; aux observations off from from_round; runtime label `controlled-c16-v1+max-round-10`.
### Design Rationale
- AgentGym returns the last step's reward and WebShop rewards only Buy Now, so at temperature 0 the first 10 turns of a 30-turn episode are the 10-turn episode; latency keeps the measured head and subtracts the tail (estimated final env step error |8| ms mean).
### Notes & Caveats
- ALF / Sci / tau2 argv, SQL and rows are unchanged (no env -> no block). vLLM batching nondeterminism and the different load mix of native 10-turn runs are not reproduced by the derivation. Page renderers other than `alf_pool_page.train_panel` were not switched to the view.

## [WebShop pool v7 preparation v0.1.0] - 2026-10-23
### Features
- WebShop joins the pool pipeline: `webshop.panels --panel pool` (data/external/panels/webshop_pool.json, scope all-train = user 2026-10-23 option B: all 3,930 stored train goals, sharing 72 of the 91 test products, no test goal in the pool; strata = product category; the product-disjoint 845-goal build is archived; recorded in state.json `pool_provenance` and the test report), `bo_pool_driver --dataset webshop` (c16 default concurrency, WebShop `goals` game count), aux observations from `webshop:train200:bo3obj-v2`, SLO-3 = median config reward.
- round0-design-v1 opt-ins for fresh runs: `POOL_ROUND0_SOBOL=1` (round 0 = 10 Sobol, no plan defaults) and `POOL_FROM_ROUND0=1` (sample-size-v1, plan-p1-v1 and semfilter-v2 recorded from round 0). WebShop runs the syftr-style pruner without a run tag (its own fresh run dir).
- `webshop_test_v7` (deployed arms only at c16, as Sci v7 -- user 2026-10-23; the outer-CV path stays behind `OUTER_CV`), `webshop_pool_v7` (dry_init on a copy root, estimates), WebShop in `fixed_plans_test` / `perquery_sweep`, `webshop_pool_page` + gated renderer hooks, scripts `run_webshop_v7.sh`, `chain_sweep_then_webshop.sh`, `refresh_webshop_page.sh`.
### Design Rationale
- Every shared-module change is behind `ds == "webshop"`, a new opt-in env, or the WebShop run's own profile_complete.json, so ALF / Sci / τ² behaviour and page output stay byte-identical (checked by rendering a page copy before / after).
- The pool is the stored goal set (task rows with embeddings are what the session pipeline can query), as for ALF / Sci pools.
### Notes & Caveats
- Synthetic goals without a store task row are not in the pool (not queryable). The WebShop store holds 3,929 successful + 1 failed session, so success+failure-v1 ≈ success-only there.
- render_atlas_results currently fails in `prepare_html` (untranslated Chinese UI, pre-existing); the page refresh step of the WebShop script would hit the same error.

## [τ² pool-v6 re-profile preparation v0.1.0] - 2026-10-23
### Features
- `tau2.sfv2_revert` (`--dry_run` / `--apply`): reverts the paused τ² SemFilter-v2 train rerun of `results/tau2_pool_v5_20261020` (rerun rows / partial v2 episodes deleted; v1 episodes, superseded opt rows, archived selection / router / test artifacts and state.json restored; audit outside the run dir).
- `tau2.fixed_plans_test` carries a frozen copy of the τ² semfilter-v2 knobs, so the τ² fixed-plan stage works with or without the state.json `semfilter` block.
- `TAU2_POOL_DESIGN=explore-v6` (`tau2.design`, `session_space_tau2_v6`, `tau2.pool_v6`, `run_tau2_v6.sh`): isolated τ² re-profile, six plans incl. P1, round 0 = 10 Sobol without plan defaults, round 1 = 10 sobol-only-v1, channel-slots-v6 batches of 8 from round 2, 50 configurations, semfilter-v2 / hv-ref-anchored-v1 / gp-fix-v1 / reuse / async staging from round 0, deployed-only test.
### Design Rationale
- The design is selected by one env var read by both the driver and its runner subprocesses, so the -v5 run and every other caller keep `session-space-tau2-v5` byte-for-byte; the v6 identity (space, run tag, test experiment, `design` block) is refused on the -v5 directory.
### Notes & Caveats
- Opt-row `updated_at` of the restored rows is reconstructed from the result-file mtimes (evaluated rows) or created_at (+ retrieval_ms); the originals were not recorded by the supersede.
- Page sections with a hard-wired -v5 path (profiling-budget-hv, dataset-splits, optimizer-replay) keep showing the -v5 run after a v6 refresh.

## [ALFWorld E2 preparation v0.1.0] - 2026-10-22
### Features
- Add an isolated ALFWorld MO-TPE / GP+LogNEHVI campaign entrypoint with paired 100-task samples, explicit 5,000-evaluation budgets, dry-run, bounded execution/resume, and OneConfig-only TEST planning.
- Export audited historical TRACE catalogs, fixed-normalization NHV checkpoints, resource ledgers, SLO policy mappings, deduplicated TEST coverage and measured-data plots.
### Design Rationale
- Fresh-panel settings govern initialization and admission together; historical 200-task runs retain their original behavior. Sampling uses a method-independent namespace while storage identities remain isolated.
- Historical TRACE remains a warm-start legacy/Sobol/staged reference. Incomplete historical cost axes are missing, with surviving task counts reported separately as lower bounds.
### Notes & Caveats
- Preparation does not authorize real episodes. Execution requires the frozen manifest's knobs, limits, serving revision and memory snapshot to be confirmed, plus exclusive evaluation capacity.
- No shared services are restarted. Actual ALFWorld transport uses the existing `.venv-agentgym` evaluator. Real discovery/TEST results are still pending.

## [profiling search and pruning v0.2.1] - 2026-10-22
### Features
- New TRACE (`three-channel-v1`) runs default to the syftr-style Pareto pruner across ALFWorld, ScienceWorld and tau2. Explicit `--pruning_policy staged|none` remains available for ablation.
### Design Rationale
- Change only profiling allocation: keep TRACE's three channels, randomized LHS and analyzer. Freeze the resolved pruning policy in each new run; omitted CLI options on resume retain the recorded policy, including historical staged runs without a policy field.
### Notes & Caveats
- Legacy search and baseline defaults remain staged. For the matched four-searcher comparison, explicitly pass `--pruning_policy syftr-style` to every method; baseline isolation requirements remain in force.
- Existing state files and running campaigns are not migrated. This default follows the author's choice after the exploratory replay, not a proven adaptive-search performance advantage.

## [profiling search and pruning v0.2.0] - 2026-10-22
### Features
- Add official Optuna 4.2.1 MO-TPE ask/tell proposals alongside GP+LogNEHVI and SMAC-ParEGO, using the common measured initialization and frozen conditional space.
- Add frozen `--pruning_policy staged|syftr-style|none` to both profiling drivers; alternative policies require isolated run identities and cannot replace a policy on resume.
- Download the MIT-licensed syftr reference at commit 7174a26; add a fixed-candidate TRAIN pruning audit with full-outcome checks, cost accounting, HV retention, and plan coverage.
- Add experiment execution addendum v0.5.0 for the four-searcher comparison and separate TRACE pruning ablation.
### Design Rationale
- Separate proposal-policy effects from evaluation-allocation effects. Keep historical staged defaults and shared profiling semantics; adapt syftr's optimistic-bound/strict-dominance rule explicitly to three mean objectives.
- Full audit outcomes remain hidden from decisions until evaluation completion. Missing futures of censored configurations are excluded, counted and never replaced with predictions.
### Notes & Caveats
- Historical audit is conditional on completed configurations and does not establish adaptive TRACE or TEST deployment improvements; summed episode latency is not search wall time.
- The syftr-style adapter uses z=1.645, full searched-configuration comparators only, no stage comparators and no plan guard. It is not the original two-objective RAG experiment.
- No new live profiling campaign was launched; MO-TPE has no explicit heteroscedastic Yvar model, and conditional parameters can use Optuna's independent TPE path.

## [profiling search baselines v0.1.0] - 2026-10-22
### Features
- Add isolated `gp-lognehvi-v1` and `smac-parego-v1` proposal backends to ALFWorld/ScienceWorld and tau2 profiling.
- Use shared defaults/LHS initialization, fixed search manifests, real-observation-only updates, proposal audits and existing staged execution/checkpoints.
- Add CPU TRAIN-fixture smoke, deterministic reconstruction and driver regression tests; document candidate baseline systems.
### Design Rationale
- Compare TRACE with channel-free GP acquisition search and official SMAC3 RF/ParEGO/EI while sharing evaluation machinery.
- GP uses sequential X_pending; SMAC keeps one scalarization per unevaluated batch and reconstructs from the measured archive each round.
### Notes & Caveats
- No live experiment or comparative NHV result. Standalone performance-blind LHS and Greedy adapters are not added here.
- Tau2 fresh baseline runs use five fixed plans from initialization; historical changing-space runs are not fair fresh-run counterparts.
- SMAC batching/archive reconstruction is an explicit adaptation; see docs/trimem_search_baselines_v0.1.0.md.
- Stabilize SMAC local-search start order across Python hash seeds; audit native random-design and exhaustion-fallback admissions separately. Reject delayed P1 extension before profiling.

## [router replay selfnorm-v1 + ALFWorld v7 re-test] - 2026-10-22
### Features
- `sparse.replay` / `outer_cv.heldout_eval`: self-normalised (Hajek) routed-policy estimator `selfnorm-v1`
  (`sparse.REPLAY_ESTIMATOR`, `sparse.routed_mean`): each metric's per-candidate own-sample sum is divided by the
  summed routed-share estimates; SE by linearisation. `sum-v0` (legacy) stays selectable. Recorded in
  router_report.json / selection.json / routing tables / cv_report.json and in the CV fold fingerprint.
- `bo_pool_driver` catalog-regime-v1 (`POOL_CATALOG_FROM_ROUND` + `POOL_CATALOG_REASON`, recorded once in
  state.json by the router stage): router / selection / SLO bounds / outer CV use only rounds >= from_round.
  Router-backend change after profiling (`POOL_ROUTER_BACKENDS_CHANGE=1` + reason, history kept).
- `alf_test_v7` (archive / plan / run / report) + `run_alf_v7_test.sh`; page block `alf-v7-retest`
  (`alf_v7_page`, `render_alf_v7`).
### Design Rationale
- On sparse profiles (own 100-200 task sample per configuration) the implied routed shares summed to 1.06-1.6 on
  the ALFWorld v7 catalog, giving replay success > 1 and biasing the SLO weight choice; dense profiles are unchanged.
### Notes & Caveats
- Sci / tau2 router or CV reruns now also use selfnorm-v1 (CV fold caches refit once: fingerprint changed).
- The ALFWorld v7 rounds 0-5 selection / routers / reports were renamed `*_v7r0-5_20261022T162337Z`
  (manifest archive_v7r0-5.json); the old page sections read them (and the minmax-raw tables they were built from).

## [router utility zscore-log-v1] - 2026-10-22
### Features
- Opt-PerQuery three-weight utility rescaled so that quality, tokens and latency have equal influence:
  `U = w_Q q/s_Q - w_T ln tok/s_T - w_L ln lat/s_L` (cost_norm scale `zscore-log-v1`, `runtime.utility_norm`).
  Each s is the median over training queries of the std across candidates of the honest (OOF where available)
  predictions. It is floored at the metric's measurement SE (the median per-candidate stratified SE). The
  spread of the candidate means (MAD / IQR) and which floor was active are recorded too.
- Weights may now be 0 for w_Q too (corners valid; all-zero rejected). The weight grid includes w_Q = 0, giving
  231 points. Router protocol `perquery-utility3-querymetrics-v3`.
- router_report.json `utility_diagnostics`: equal-weight variance shares of the three terms, decision-flip
  rates under a one-s perturbation, and corner picks and shares. Fold records carry their frozen `cost_norm`.
### Design Rationale
- Min-max over all observed raw costs (long-tail episodes: tokens 8.6k-399k, latency 14 s-2193 s) shrank real
  candidate cost gaps to about 0.01, so cost weights barely moved the decision. Log costs plus a per-query
  spread make one unit of each term comparable.
### Notes & Caveats
- `minmax-raw` artifacts stay readable and reproducible (utility branches on `cost_norm.scale`;
  `sparse.fit(..., utility_norm="minmax-raw")`, `brane.fit(..., utility_norm=...)`). The rule name
  `utility3-minmax` in routing blocks is kept for artifact compatibility.
- s depends on the fitted predictors, so different backends (BRANE / LimeQO) freeze different scales.

## [channel slots v2] - 2026-10-21
### Features
- `channel-slots-v2`: in every BO round >= 1, the batch of 5 is qfirst / llm-slot (analyst rows + dls-agent endpoints) /
  xover-slot / dls-slot (incumbent, random, top-pool and new-plan DLS endpoints) / sobol-slot. The kriging
  believer follows the same order.
- Forced rounds with k >= 4 keep the forced picks but reserve trailing fills, so crossover, DLS and Sobol each
  contribute at least one pick.
- `bo_pool_driver --no_limeqo` / `POOL_NO_LIMEQO=1`: runs router, select, cv and test with BRANE only. The choice
  is recorded in state.json `router_backends`.
### Design Rationale
- Each candidate channel gets a guaranteed batch slot, so no single channel monopolises the hv picks. Slot
  membership uses multi-label sources.
- The protocol is applied only when the state.json block is first recorded, so paused runs (ALFWorld, τ²) keep
  `channel-slots-v1`.
### Notes & Caveats
- An empty slot falls back to the whole-pool hv argmax and is logged. The LLM slot runs the remaining DLS starts
  first, so the fallback is the true whole-pool argmax.
- See the docs/trimem_pool_profiling_plan_v0.1.0.md addendum of 2026-10-21. Tests: tests/test_channel_slots_v2.py.

## [shared query router v0.1.0] - 2026-10-21
### Features
- Select `--model brane` (default) or `--model limeqo` in the offline snapshot trainer.
  The new model jointly learns Q/T/L responses with one inductive low-rank query encoder.
- Whole-query, all-candidate OOF predictions; NumPy-only deployment through existing Router;
  isolated output artifacts and independent outer CV. Pool router stages export TRAIN snapshots.
- Official MIT LimeQO checkout pinned to d560b2fe7448f246d4cc7a7f8c5b20bebafc06b0.
### Design Rationale
- Share statistical information across configurations while predicting unseen queries from features;
  preserve BRANE and the existing utility/SLO selector for comparisons.
### Notes & Caveats
- LimeQO-inspired adaptation, not exact censored/transductive LimeQO; no profiling-policy replacement.
- Fixed rank/regularization, squared quality loss, log-cost predictions; performance superiority is unproven.
- CLI CV excludes feature/router overhead and reports observed coverage. No live experiment is switched.
- See docs/trimem_shared_router_v0.1.0.md for schema, invocation and validation scope.

## [three-channel search analyzer v0.2.0] - 2026-10-21
### Features
- Fresh runs use randomized plan-stratified LHS for coverage, local restarts and sampled initialization;
  Sobol remains selectable. Both sampler settings are frozen in run state.
- A dedicated shadow-replay entrypoint exercises real Luna guidance, persistent memory, Local Refinement
  and GP/acquisition selection against the frozen tau2 TRAIN178 archive in an isolated output directory.
- Fig. 3 uses randomized LHS and removes the GP-update arrow and label; paper method wording is synchronized.
### Design Rationale
- Preserve existing runs and legacy Sobol initialization while making the new design executable.
- Reveal historical rounds exogenously to test memory updates without inventing outcomes for new candidates.
### Notes & Caveats
- Two shadow rounds produced ten unexecuted candidates. This validates integration, not search effectiveness.
- Discrete mapping, deduplication and initialization top-ups can break exact Latin-hypercube structure.
- The snapshot replay uses own-quality SE with a fixed baseline, not the unavailable paired-gain variance.
- Model settings remain gpt-5.6-luna / low effort / no reasoning summary; posterior Sobol QMC is unchanged.

## [three-channel search analyzer v0.1.0] - 2026-10-21
### Features
- Global Coverage, Local Refinement and Frontier Recombination share the BO candidate pool.
- A persistent GPT-5.6 Luna analyzer supplies validated starts and knob priorities only to local search;
  measured endpoint feedback updates evidence-linked memory. Defaults: low effort, no reasoning summary.
- Sobol remains the default; plan-stratified LHS is an explicit comparison option. Fig. 3, method text,
  prompt and design contract document the new flow.
### Design Rationale
- Keep language-model advice out of GP labels and final evaluation selection. Acquisition-scored local
  refinement consumes the guidance, with full-neighborhood fallback and independent baseline starts.
- Freeze the search design in state.json: new runs default to three-channel-v1; historical runs stay legacy.
### Notes & Caveats
- No live experiment was migrated or restarted. New comparisons require separate output directories.
- Low effort retains internal reasoning; `--analyzer_effort none` disables it. No model substitution is used.
- LHS equivalence, persistent-memory benefit and optimizer overhead require controlled experiments;
  existing results are not evidence for the revised protocol.

## [router outer 5-fold CV v0.3.0] - 2026-10-21
### Features
- `trimem/optimizer/query_router/outer_cv.py` (`outer-cv5-v1`): stratified, seeded outer folds (`FOLD_SEED`
  20261021) over the training queries. Per fold, the feature transform, candidates, predictors (with nested
  selection), cost normalisation, Opt-OneConfig points and SLO bounds, and Opt-PerQuery SLO weights are fitted
  on the other 4 folds. The fold is then evaluated on held-out queries with measured outcomes only, and the
  fold's router also routes the test set.
- `trimem/experiments/router_outer_cv.py`: per-fold cache, `router_cv/cv_report.json`, frozen
  `test/routing_cv.json` with an offline pair budget (deployed-pair reuse, NoMemory reuse, completed pairs,
  expected quality-reuse discount), and the report's CV section with mean ± std across folds.
- New `cv` stage in `tau2.pool_profile` and `bo_pool_driver`. `all` now runs profile → router → select → cv →
  test → report. The test stage runs the union of the deployed and CV pairs, each pair once. On ALFWorld and
  ScienceWorld the CV test arms run only up to `--cv_test_max_pairs` (2,000) new pairs. The report adds
  `outer_cv` to test_report.json (τ²: also report.md; agent-env: report_cv.md). Reported numbers are the TEST
  set mean ± std across the 5 fold routers (user 2026-10-21). The held-out-fold evaluation is kept only as the
  internal `router_cv/cv_report.json`.
### Design Rationale
- Until now the outer partitions only produced OOF predictions for the training replay. The deployed router is
  one final fit, so no reported number was a held-out estimate of the whole pipeline. User decision: CV is the
  primary protocol, and every number is reported as mean ± std across folds.
- With sparse per-config samples, the held-out estimate reuses the replay estimator restricted to the fold. It is
  unbiased and needs no imputation. Coverage and the plug-in prediction are reported as flagged diagnostics.
### Notes & Caveats
- The deployed single-final-fit router, routing.json and its test arms are unchanged and kept for continuity.
- Test std across folds measures router-training variability only, because all folds share the test tasks. The
  per-task SE is reported separately (`se_tasks_mean`).
- τ²: the CV-only test pairs run under concurrent ALFWorld profiling load, so their latency is not strictly
  like-for-like with the deployed pairs. The live ALFWorld driver predates the stage and needs a restart to run it
  (prepared: results/pool_v5_20261020/alf_cv_restart.sh). ScienceWorld picks the stage up automatically.
- docs/trimem_query_router_plan_v0.1.0.md, addendum "outer 5-fold CV v0.3.0".

## [agent-interaction-steps-v1] - 2026-10-20
### Features
- Record generated agent turns, environment interactions and Tau2 agent tool calls separately from LLM calls; preserve counts on quality reuse and expose a read-only trajectory backfill.
### Design Rationale
- Match EXPHARNESS-style interaction-length reporting while keeping Tau2 user/tool/agent orchestrator transitions distinct; exclude fixed greetings and prompt examples.
### Notes & Caveats
- Existing running workers keep their loaded code and complete trajectories; explicit fields appear on subsequent process starts. Historical Tau2 orchestrator counts remain null. See docs/agent_step_metrics_v0.1.0.md.


## [pool profiling: plan P1+Rank + async staging v0.9.0] - 2026-10-20
### Features
- New retrieval plan `P1+Rank`, displayed V+SemFilter+G+Rank. It applies SemFilter to the vector seeds
  before graph expansion, then expands, caps the pool, ranks with ColBERT and takes the top-k. `P2+Rank`
  (VGR+SemFilter+Rank) is unchanged. Receipts record `plan_order` and `filtered_seed_ids`.
- ALF/Sci session space `session-space-agentenv-v7` with 5 plans. Round 0 is still 10 configurations
  (5 plan defaults + 5 Sobol).
- τ² live run: `P1+Rank` can be proposed from round 7 (state.json `space_extension`, plan-p1rank-v1). The
  run id is unchanged.
- `session_pool.use_plans`, new-plan DLS starts (`dls-newplan`), and per-plan pool composition (`plan_counts`).
- Asynchronous staged pruning `staged-async-v1` in `tau2.pool_profile` and `bo_pool_driver`. Each
  configuration is decided as soon as its own stage completes, the comparators are logged per decision, and
  concurrency is rebalanced as arms finish. Recorded in state.json `staging.async`.
### Design Rationale
- `P1+Rank` gives a cheaper SemFilter (judge calls only on the seeds) next to the pool-filtering plan. The
  category is appended so that existing GP encodings and τ² keys do not change.
- Gating by round keeps τ² rounds 0-6 reproducible without bumping the τ² space version.
- Async staging removes the synchronous barrier, where slow SemFilter arms used to hold back fast VGR arms
  at every checkpoint.
### Notes & Caveats
- In async mode, a decision sees only the comparators that exist when its stage completes, so decision order
  can affect the outcome. τ² round 6 mixes a synchronous stage 50 with asynchronous stages 100 / 178.
- The GP has no observation of `P1+Rank` until it is measured. Its prior comes from the categorical kernel.
- `docs/trimem_pool_profiling_plan_v0.1.0.md`, addendum "plan P1+Rank and asynchronous staged pruning".

## [TRACE calibrated routing figure v0.2.2] - 2026-10-20
### Features
- Fig. 3 uses solid curved proposal arrows, dashed map-to-pool arrows and
  straight box-to-box connectors in panel (a). Removed posterior and
  profiling-budget annotations.
- Both deployment paths share profiling. The fixed branch is labeled in
  After profiling; per-query routing reuses its results for three query-dependent
  predictors and cost--quality calibration. Removed the candidate-catalog box
  and the duplicated profiling matrix and headers in panel (b).
- Explicitly show fuzzy Pareto pruning and best-per-plan retention before
  router training; fixed selection continues to use all completed configurations.
- Replaced the SLO filter box with a schematic quality--token plot showing
  quality-first, token-first and latency-first choices; point color encodes latency.
- Removed the transfer table and retained all six directions in prose.
### Design Rationale
- BRANE Fig. 2 provides the visual structure; calibration maps SLOs to utility
  weights before the frozen router chooses configurations for new queries.
### Notes & Caveats
- The calibration curve and surrogate contours are schematic. No runtime
  behavior or experimental results were changed.

## [TRACE figure and evaluation design v0.2.1] - 2026-10-20
### Features
- Fig. 3 now connects Sobol/LLM proposals to configuration points, DLS to a
  local region, and crossover to a green candidate; every pool point has an
  explicit incoming map connection. Renamed evaluation/results boxes and
  removed the training-input box.
- Added the shared evaluation design with three questions: end-to-end value,
  search efficiency/ablations, and all six directed transfers among ALFWorld,
  ScienceWorld and tau2-bench. Removed the separate predictor-accuracy RQ.
- Specified hypervolume-versus-budget plots and exported historical curves
  with PDF, SVG, PNG, CSV and provenance metadata.
### Design Rationale
- Channel arrows distinguish point proposals, local search regions and the
  common candidate pool. Transfer comparisons separate configuration warm
  starts from predictor adaptation under equal target evaluation budgets.
### Notes & Caveats
- Historical curves use single-trial archives and are separate diagnostics;
  primary pass^4 and six-way transfer comparisons remain planned experiments.
- No agent episodes, optimizer runs or production state were changed.

## [TRACE query-dependent predictors v0.2.0] - 2026-10-20
### Features
- Dense and sparse router training now default to three independently selected
  query-dependent predictors per configuration: quality, raw latency and raw tokens.
- Five-fold query cross-fitting reruns three-fold, 30-trial model selection inside
  each outer training partition. Preference replay uses OOF predictions for all
  three objectives; final deployment models are refitted on all observed rows.
- Updated the shared manuscript, routing equation and editable workflow diagram
  with supervision, model families, CV, normalization and deployment details.
### Design Rationale
- Raw-cost regression targets arithmetic conditional means for mean-cost SLOs.
  Normalization uses observed training-query costs so equal configuration means
  do not erase query-dependent cost differences.
- Model-selection folds are separated from preference-replay folds. Sparse
  configurations train only on their own measured queries.
### Notes & Caveats
- New training protocol: `perquery-utility3-querymetrics-v2`. Mean-cost and
  log-ridge ablations remain explicit options; old JSON artifacts remain readable.
  New outer artifacts use schema version 2; the runtime also reads version 1.
  Frozen artifacts and experiment results are not retrained or overwritten.
- Training replay is conditional on the training-selected catalog and feature
  representation and full-training cost bounds. Independent test episodes remain necessary for final claims.
- Inner selection, outer prediction and serving share cost fallback semantics;
  fallback means respect workload strata. Dense log-cost routing remains a legacy ablation.
- Nested model selection adds offline fitting work; production training latency
  has not been benchmarked by this change.
- Quality labels retain the profiling protocol. In particular, single-trial
  training labels do not become pass^4 labels through this change.

## [TRACE optimizer workflow figures v0.1.2] - 2026-10-20
### Features
- Added two editable workflow panels to the shared optimizer figure: BO exploration
  followed by SLO-based fixed selection, and per-query router training/deployment.
- Added a dedicated four-source candidate-pool subsection and clarified that the
  empirical Pareto set is a subset of completed configurations. The fixed-selection
  figure branch ends at Fixed configuration.
- Exported a standalone vector PDF, an SVG with embedded fonts and editable text,
  and a reproducible TikZ source package.
### Design Rationale
- Follow RAG-Stack Figure 6 for the candidate-pool, acquisition, profiling and GP
  feedback loop, and BRANE Figure 2 for the offline/online router split.
- Distinguish acquisition-based exploration from final SLO selection and show that
  both deployment paths use the same completed profiles.
### Notes & Caveats
- The contour map is schematic. LLM proposals are optional. Router quality is
  query-dependent; costs remain frozen configuration means.
- The paper-fig presentation runtime was unavailable, so the editable deliverables
  are TikZ/SVG rather than PPTX. This change does not modify or launch experiments.

## [TriMEM pool BO GP protocol gp-fix-v1] - 2026-10-20
### Features
- `session_pool.fit` gp-fix-v1: Gamma(2, 1) outputscale priors on every ScaleKernel, a seeded MLL fit and
  per-output degeneracy diagnostics (`model.gp_diagnostics`). gp-v0 can still be selected with `use_gp`.
- τ² `pool_profile.observed_points`: the quality GP noise is the paired SE² minus the common-mode covariance
  (`unpaired_stats.common_mode_cov`; floor 25%).
- `state.gp` records the protocol and its `from_round` (`session_bo_agentenv.gp_block` / `use_gp_for`, both
  drivers). Every proposal logs the posterior spread.
### Design Rationale
- The τ² round-3 quality GP was constant (+0.0354, SD ≈ 0). About 53% of each paired SE² is shared by every point
  (same tasks, same NoMemory arm), so the fixed noise exceeded the observed spread (Cochran Q 12.8 / 19 df) and
  ML-II zeroed the outputscale. Diagnosis and LOO CV are in docs/trimem_pool_profiling_plan_v0.1.0.md, addendum
  gp-fix-v1.
### Notes & Caveats
- Not retroactive: the live -v5 τ² run applies it from round 4, and rounds 0–3 stay as proposed. The quality
  signal remains weak: LOO Spearman is +0.17 (gp-v0 was constant).

## [TRACE design-review documents v0.1.1] - 2026-10-20
### Features
- Replaced the detailed TRACE overview with an offline profiling/search loop and an online
  characterization/selection/execution flow. Added TriMEM optimizer (TRACE) to Figure 1(a).
- Aligned routing Equation 2 with `utility3-minmax`: independent quality/latency/token weights,
  frozen raw-cost min--max normalization, and zero penalty for a constant cost dimension.
- Explained profiling inputs and outputs, GP-based point selection, sparse router training and
  SLO-based preference selection; synchronized the shared companion draft.
### Design Rationale
- Keep the supervisor's idea-review draft focused on profiling, optimization and routing.
  Fold the separate proposal-channel diagram into prose, following METIS's clear selection flow.
### Notes & Caveats
- BO retains log-cost objectives; the router uses normalized raw mean costs. SLO selection
  estimates workload means and does not enforce per-query latency guarantees.
- This is a documentation update; it does not change experiment code or launch profiling.
  Sources, original PDF backup, code alignment and validation are in `paper/sigmod/review_20261020/`.

## [TriMEM qfirst slot + test arms v2] - 2026-10-20
### Features
- Quality-first slot (`qfirst-slot-v1`, `state.qfirst_slot`, from_round 3): in every BO round >= 3 of both pool
  drivers, the first batch slot is UCB_1.5(gain) x P(tokens, latency within budget) (`session_pool.suggest`
  acq_mode="qfirst", quality champion as an extra DLS start `dls-qchamp`). The believer conditions on it before
  the 4 hv picks. Picks are labelled `channel="qfirst"` with beta / budget / champion recorded.
- Opt-PerQuery test weights v2: default (0.6,0.2,0.2), quality (0.8,0.1,0.1), tokens (0.1,0.1,0.8), latency
  (0.1,0.8,0.1). TEST_VERSION τ² `pool-v2`, ALF/Sci `pool-strat200-v4`.
- ALF/Sci test concurrency 16 -> 40, recorded in the routing table and the test protocol. The reused arms ran at 16.
### Design Rationale
- Budget = max observed cost x 1.1, so the slot can exploit around the expensive τ² champion (about 348k tok /
  964 s, 2-3x the medians). With this budget the slot is effectively quality-UCB with a feasibility guard.
- Quality target "gain", not "improve": on panel-aggregated pool points, "improve" (gain > 0) is degenerate.
### Notes & Caveats
- Applied to the live τ² run before round 3 (graceful drain + restart of the trimem-pool-v5 scope). Docs:
  `docs/trimem_pool_profiling_plan_v0.1.0.md`, last addendum.

## [TriMEM staged pruning v2 + sim_gate none] - 2026-10-20
### Features
- `staged_pruning` v2 (`staged-pareto-prune-v2`; τ² `tau2-pool-staged-v2`): round 0 is staged and pruned like every
  round (only NoMemory protected); prune when the optimistic corner is weakly dominated by a full configuration's
  point OR by the pessimistic corner of NoMemory / any configuration at >= the stage on the same subset (τ²
  paired: gains vs NoMemory on the same tasks); plan guard keeps >= 1 full configuration per plan; decisions log
  corners and every dominator; round-0 savings enter the BO budget (budget counts episodes actually run).
- Live τ² state migrated v1 -> v2 at the restart (`staging.migrated_from_v1`); 50-prefix filled first.
- sim_gate = none for every new configuration: τ² proposal-side constraint (`PROPOSE_FIXED`,
  `tau2-propose-simgate-none-v1`, `state.proposal`) in every `session_pool` channel and the analyst view;
  ALF/Sci space `session-space-agentenv-v6` fixes it (run ids / dirs `*-v6`, NoMemory copied from `-v5`).
- The 5 gated τ² round-0 configurations stopped by the user (`staging.stopped_by_user`): GP observations only.
### Design Rationale
- Corner-vs-pessimistic-corner works before anything is fully evaluated; per-configuration SEs (not the A-B paired
  SE) keep the two-corner rule conservative.
### Notes & Caveats
- Docs: `docs/trimem_pool_profiling_plan_v0.1.0.md`, addendum "round 0 staged too (prune rule v2), sim_gate = none".

## [TriMEM pool profiling space v5] - 2026-10-20
### Features
- memory_form fixed per dataset, not searched (session spaces `-v5`): τ² `compact`, ALFWorld / ScienceWorld `raw`;
  recorded like `edges` (`FIXED`), dropped from grid / GP features; new experiment ids and dirs (`*-v5`,
  `results/tau2_pool_v5_20261020`).
- τ² profiling 1 trial per train task (`train178-1trial`, pass^1, budget 30 x 178 x 1); test stays 4 trials / pass^4.
- LLM analyst candidate channel (`analyzer_pool`, `analyst-pool-v1`) in BO rounds >= 1 of both pool drivers: local
  Qwen, temperature 0, strict in-space parsing, never fatal, per-round logs under `<run dir>/analyst/`.
- Queue `results/pool_v5_20261020/run_queue.sh` (τ² -> ALFWorld -> ScienceWorld) supersedes the -v4 supervisor.
### Design Rationale
- Offline token measurement: τ² compact cuts injected memory ~80%; ALF/Sci compact ~= raw, distilled longer.
- 1-trial profiling quarters the τ² profiling cost; the t0 episode is the 4-trial run's trial 0 (same seed).
### Notes & Caveats
- -v4 memory episodes discarded (kept as evidence); NoMemory reused (ALF 49; τ² 43 t0 files re-keyed to 1-trial
  with provenance). The ALF/Sci NoMemory sample namespace excludes the analyst tag so reuse stays valid.

## [TriMEM pool profiling space v3] - 2026-10-20
### Features
- Session spaces (agent-env / τ²) fix edges to `similar+similar_up` (was `similar_up`); versions bumped to
  `session-space-agentenv-v3` / `session-space-tau2-v3`; new experiment ids and dirs (`*-v3`, `tau2_pool_20261020`).
### Design Rationale
- User correction; a version bump keeps every void similar_up row / episode out of the corrected experiments.
- NoMemory's sample is space-independent, so its protocol-identical episodes are reused (copied).
### Notes & Caveats
- The noise repeat (default VGR) depends on the edges and is rerun; void dirs kept untouched as evidence.

## [TriMEM pool staged profiling v0.1.0] - 2026-10-19
### Features
- `bo_pool_driver` (ALFWorld / ScienceWorld pool round) stages BO rounds >= 1 with 3-D Pareto pruning
  (`staged_pruning`, `staged-pareto-prune-v1`): 50 -> 100 -> 200 of each configuration's own stratified
  sample, pruned when the optimistic corner is dominated by a full configuration or NoMemory.
- Budget = the unpruned 50 x 200 configuration episodes; savings buy more configurations (<= 80).
- Arms finishing early free their slots: the others are drained (in-flight episodes finish) and relaunched.
- Test stage (`pool-strat200-v3`): Opt-OneConfig SLO-1/2/3 + Opt-PerQuery at 4 fixed weight vectors (no slice
  or Opt-PerQuery SLO arms); 16 concurrent episodes = the controlled arms used as the paired baseline.
### Design Rationale
- Same rule as the τ² round; quality (not gain) in the corner: NoMemory is a common constant.
- Pruned configurations stay GP observations with partial-n noise; selection uses full ones only.
### Notes & Caveats
- The paused ALFWorld round-0 state resumes unchanged (staging block added, round 0 in full);
  `--staging none` restores the old fixed rounds. A state with unstaged BO rounds is refused.

## [TriMEM tau2 request routing v0.1.0] - 2026-10-19
### Features
- Opt-in SemFilter request routing uses each vLLM replica's running/waiting load and local in-flight
  reservations. Agent, user and official judge conversations keep their episode endpoint.
- Explicit transport continuation preserves original completed files and their immutable protocol,
  records the new source/routing protocol separately, and validates both sets of retrieval receipts.
- Serving-segment digests on each new result and database receipt distinguish batch-limit trials.
  Continuation binds the original result file hashes and rejects silent protocol rollback.
- A drain file stops new episode admission and lets in-flight episodes finish before serving changes.
  Endpoint failure also stops admission when request routing is enabled.
### Design Rationale
- An episode can issue four concurrent semantic calls, so episode-count balancing alone underutilizes
  one replica while the other queues. Request routing changes transport without changing judge payloads.
### Notes & Caveats
- Transport authorization rejects task, seed, model, corpus and evaluation-setting changes. Different
  batches/endpoints may still produce different numeric outputs. Throughput measurements remain segmented.

## [TriMEM difficulty reporting v0.1.0] - 2026-10-19
### Features
- ALFWorld / ScienceWorld reports now include all / easy / hard / unknown groups using the frozen
  ExpRAG v2 Appendix B.2 task-type partition. Report coverage, score, full-success rate, paired NoMemory
  gain and SE, tokens and latency. Pool SLO / lambda arms retain their frozen routing and shared episodes.
- Atlas and profiling-page refreshes write `results/difficulty_20261019/report.{json,md}` and update a
  separate easy/hard table in `qa-trajectory.html`. Queued pool test reports include the same breakdown.
### Design Rationale
- Group only at reporting time, so labels do not affect BO, SLO selection, retrieval or environment runs.
### Notes & Caveats
- These are paper-defined task-type groups, not official environment difficulty labels. Incomplete groups
  show observed-sample means with coverage. Missing labels remain unknown, singleton SE is undefined.
- Pool and older controlled runs have different cost protocols. The report preserves baseline provenance.

## [TriMEM pool profiling round v0.1.0: ALFWorld + ScienceWorld, session space v2] - 2026-10-19
### Features
- Profiling pool = every non-test task the env server can address (`build_agentenv_pool`): ALFWorld 4,066
  (1,646 games appended to the AgentGym server by `env_server_managed`, append-only, `n_games` in
  `/managed_stats`), ScienceWorld 4,144 (`--sciworld_scope all` = 6,712 incl. the 7 task types AgentGym excludes).
- `bo_pool_driver` (profile | router | select | test | report | all): 50 configurations, each on its OWN
  stratified (task_type) random 200-sample with a recorded per-config seed; NoMemory and the VGR noise repeat
  on their own samples; unpaired stratified means / SEs (`unpaired_stats`,
  `session_bo_agentenv.load_points_unpaired`); `--sampling shared --estimator paired` keeps the old design.
- Session space v2 (`session_space_agentenv_v2`, 69,120 configurations): edges fixed similar_up; seed
  4–128, depth 2–4, fanout 4–64, top_k 3–10; sim_gate, memory_form (raw / distilled via
  `agentenv_sem_extract` / success_only_raw), outcome_policy (success_only / success_failure / _capped),
  semfilter_threshold (SemFilter plans). `SessionConfigV2`: candidate pool cap 256 by query cosine,
  expansion guard 4,096, similarity gate, failed-card cap; receipts record pool size / cap hits / gating.
- Sparse BRANE router (`query_router/sparse.py`): per-config predictors on each config's own sample, lambda
  chosen per SLO on an unbiased out-of-fold replay estimate. Opt-OneConfig / Opt-PerQuery for SLO-1/2/3
  frozen before the test; test reruns NoMemory and the 4 plan defaults under the v2 protocol.
- Queue `results/bo_agentenv_20261016/pool_strat200/run_pool_strat200.sh` (after the τ² 4-trial run and the
  λ-fill queue; ALFWorld then ScienceWorld). Max concurrency (user 2026-10-19): vLLM `MAX_NUM_SEQS` 64 (serve
  script default still 16), parallel arms per BO round / test (`run_parallel`), env `MAX_ENVS`, SemFilter
  judge threads 8, host-wide v2 retrieval slots 16 (PG 32 connections), 30-minute throughput measurement.
  Test = Opt-OneConfig / Opt-PerQuery arms only; NoMemory + plan defaults = optional `--test_fixed`.
- Opt-PerQuery TEST λ curve (λ 0 … 1.0 + first all-cheapest λ, w 0.5/0.5, frozen): offline routing table,
  each distinct (configuration, query) pair executed once across all λ / Opt arms, per-arm assembly
  (`bo_pool_driver.routing_table / pairs_of / assemble`); train-OOF vs test curves on the profiling-points
  page (`render_profiling_points.pool_round`). λ arms share episodes, so they are correlated.
### Design Rationale
- Independent per-config samples cover the pool's heterogeneity (50 x 200 ≈ 10,000 distinct-ish tasks
  instead of one 100/500-task panel) at the price of unpaired noise; the GP gets each config's own SE², the
  NoMemory SE is common-mode. Redundant v2 combinations are canonicalized so BO never evaluates a duplicate.
- The replay estimator needs no counterfactual: each config's random sample is an unbiased sample of the pool.
### Notes & Caveats
- v1 behaviour is byte-identical (SessionConfig receipts, adapter seeds, SemFilter threshold 0.6, BO engine
  neighbours); v2-only paths: `--space_v2`, `widen_seeds`. ALFWorld has 0 failed sessions (outcome_policy
  mostly matters for ScienceWorld). Plan: docs/trimem_pool_profiling_plan_v0.1.0.md.

## [TriMEM τ² controlled rerun v0.2.0: 4 trials + endpoint pool] - 2026-10-19
### Features
- `--num_trials N` (runner + supervisor): every task runs N times with the official per-trial
  seeds (`random.seed(300)`, `randint` per trial); one result file per (task, trial)
  (`…__t<k>.json`), resumable per episode; memory arms pair each trial with the same-trial
  NoMemory episode. Reports pass^1 (mean reward), official pass^k, pass@k, all-trials agreement
  and pairwise trial agreement; W/L and the sign test are per task on trial-mean reward.
- `--api_bases` endpoint pool: each episode is pinned to the least-loaded server (8030/8031) for
  agent, user simulator, official NL judge and SemFilter calls; endpoint recorded per episode.
- Per-stage concurrency: `--max_concurrency` (episodes, supervisor cap 16 -> 64),
  `--max_retrieval_concurrency` (retrievals; PG 55450 max_connections = 32), `--semantic_workers`;
  the supervisor now passes `--max_rss_gb` to each arm.
### Design Rationale
- Diagnosis of the w16 run: all agent/user/judge traffic went to 8030 (GPU0 100 %, GPU1 0 %);
  per-call latency doubled from 4 to 16 workers (3.6 s -> 7.5 s per agent call), so throughput
  rose only 229 -> 321 episodes/h. The judge already ran inside each episode (about 1 % of time).
- Episode pinning (not per-call balancing) keeps each conversation in one server's prefix cache.
### Notes & Caveats
- Single-trial runs keep the old file names, protocol seed and behaviour. The protocol now
  records `num_trials`, `trial_seeds` and `endpoint_pool`, so old markers are not resumable with this code.

## [TriMEM τ² controlled rerun v0.1.1] - 2026-10-19
### Features
- Remove the per-source-task session cap from shared retrieval, τ² runner CLI/injection, per-query
  BRANE configuration, comparison commands, validation, metrics and current report text.
- Restore ordinary ranked session top-k. Several distinct sessions from one task may be selected
  together; source-task identities remain available as observational evidence.
### Design Rationale
- Follow the user's explicit reversal of the earlier constraint. Preserve session-ID deduplication,
  train/test isolation, existing ranking/SemFilter behavior and immutable-run verification.
### Notes & Caveats
- Stop the capped run and retain it as superseded. The uncapped rerun uses a fresh output root and
  protocol identity, with no relabelling or reward reuse from the capped experiment.
- Remove the configuration field entirely, restoring its earlier serialization shape. Old cap
  CLI/JSON options are rejected instead of silently enabling a hidden limit.

## [TriMEM τ² controlled rerun v0.1.0] - 2026-10-19
### Features
- Add an optional per-source-task session limit after final ranking across all session retrieval
  plans, preserving full candidate evidence and filling top-k from later distinct tasks.
- Add five-arm τ² smoke/full comparison orchestration, explicit frozen BRANE transfer metadata,
  protocol and native-corpus fingerprints, actual vendor import verification, exclusive arm locks,
  result/opt-receipt consistency checks and automatic progress reports.
- Record selected source task identities and diagnose incomplete official NL judge outputs without
  changing official rewards. Audit official agent/user/evaluator settings and historical task revisions.
### Design Rationale
- Keep official environment/evaluator behavior and common local Qwen settings fixed across arms.
  Separate the current success+failure memory policy and one-session cap from historical results.
- Retain the historical BO configuration and BRANE model unchanged for a clearly labelled transfer
  evaluation. Reuse each query embedding between the router probe and selected retrieval.
### Notes & Caveats
- Shared retrieval defaults preserve uncapped behavior for other datasets. The new comparison
  explicitly enables one session per task and forbids historical reward reuse.
- The additional configuration field changes serialized config hashes. Legacy strict receipt
  resume can reject configurations written before this field existed; this rerun uses fresh
  experiment IDs/directories and does not migrate or reuse those historical receipts.
- This protocol is a one-trial local Qwen comparison, not the released four-trial GPT-4.1 benchmark.
  The old router was not retrained under the new memory policy or task cap.

## [TriMEM corpus graph explorer v0.2.0] - 2026-10-19
### Features
- Add complete Task/Session/native-hub overviews for CodeActInstruct, ToRA-MATH and τ² inside
  `qa-trajectory.html`, with pan/zoom, split and edge-type filters, node search, directed neighbor
  inspection, full graph JSON download and navigation to the original Session/Step records.
- Export native AM topology in the same read-only snapshot as the memory census. Keep held-out
  archive membership edges explicitly separate from persisted graph edges and retain reverse indexes.
- Document actual edge meanings, same-task versus cross-task similarity counts, and research-only
  Skill-α adaptations grounded in local τ² evidence and the current success+failure memory policy.
### Design Rationale
- Use Canvas for the full unsampled projection and a focused neighbor layout for readable local
  relationships. Layout coordinates do not encode vector distance. Steps remain in the existing
  task detail view, while counts account for all omitted Step edges.
- Preserve source hashes, database load identity and immutable asset publication. Similarity edges
  describe candidate relationships, not historical memory usage or demonstrated transfer success.
### Notes & Caveats
- τ² held-out Task vertices have existing domain/issue hub edges; their historical Session vertices
  and membership links remain archive-only. This change performs no graph mutation or model call.
- Skill-α experiment protocols are proposals only. No retrieval strategy, policy, editor or agent
  evaluation was changed or launched by this visualization update.

## [TriMEM multi-session corpus explorer v0.1.0] - 2026-10-19
### Features
- Reuse local CodeActInstruct and ToRA-MATH source files, preserve complete published messages and
  provenance, and normalize existing independent attempts into Task → Session → Step archives.
- Add deterministic task-level memory splits, train-only PostgreSQL stores with real 1,024-dimensional
  Qwen embeddings, native graph memberships/ordering/similarity edges, resumable embedding batches,
  import receipts and idempotent verification. Reuse the existing τ² store without reimporting it.
- Integrate an offline three-dataset explorer inside `qa-trajectory.html`, preserving its original
  datasets and embedded JSON. A lazy embedded view provides complete action/observation details,
  split filters, task search and JSON downloads. Keep the standalone view as its backing page.
  Match archive manifests and import receipts to a consistent database census before showing verified
  memory counts. Verify τ²'s original train projection separately from its full four-trial archive.
- Publish immutable asset generations and atomically switch the viewer entry after successful validation.
  Main-page integration respects both existing locks and persists through the atlas template/generator.
- Recompute historical τ² rates, paired outcomes and cost from 500 test episodes and audit 400 retrieval
  receipts. Add a source-backed Skill-α mechanism/ablation report, following the user's research-only scope.
### Design Rationale
- Keep source membership, custom experimental splits and retrieval eligibility distinct. Published ToRA
  outputs come from MATH test tasks, so the custom split must not masquerade as an official train/test split.
- Keep actual graph topology in `graph_store_am`; the browser exports archived hierarchy and identifies
  train-only memory scope independently. Large archives load per task rather than as one browser payload.
- Retain missing observations as null and label outcome provenance. Do not synthesize trials or infer
  ToRA correctness from missing source reports. Reject duplicate MATH problem identities across indices.
### Notes & Caveats
- CodeAct has 1,093 eligible tasks / 2,291 sessions; ToRA-MATH has 4,553 / 9,106; τ² has 278 / 1,112.
  A session here is a published independent attempt, not a longitudinal user conversation.
- CodeAct's split is task-ID-disjoint but has 16 identical-prompt groups across train/test, all from
  distinct ALFWorld trial worlds. ToRA's original 16k training trajectories were not available in the
  checked official release. Neither corpus supplies an official multi-session trajectory test split.
- Skill-α is literature/code research only. The unadopted migration draft is archived outside runtime;
  no effect claim from that draft is part of this release. The existing τ² retrieval implementation is unchanged.
- Verification targets stock PostgreSQL 16 on this machine, not the GX10-only fork build.

## [SWE memory v0.3.0: CoderForge eval-agent replica, full CoderForge store, SemExtract] - 2026-10-19
### Features
- `swebench/oh/`: replica of CoderForge's official eval agent (OpenHands 0.52.1 CodeAct, 4 tools, native tool
  calling, byte-exact system / instance prompts from the published eval trajectories, T 0.7 / top_p 0.8, 32k
  output, 100 iterations, 120 s/command, 7,200 s/episode) driven from the host against the action server of
  the `qingyangwu/swesmith.*` images (`oh_episode.py`, `AttachedRuntime`); `.venv-oh` via
  `scripts/setup_openhands_venv.sh` + `patches/openhands-0.52.1-trimem.patch` (thinking off + context guard,
  `oh/ctx_guard.py`); patch always `git add -A && git diff --cached <base>`.
- Store `coderforge_full` (= trimem_swe_coderforge, read-only) + `cf_full_leakage` (cf-full-leakage-v1:
  test holdout, own task, canonical-id match, same-function across all three sources on each task's own
  pre-image) and `cf_verified_leakage` (cf-verified-leakage-v1) for SWE-bench Verified queries.
- `sem_extract.SemExtract` (sem-extract-v1): Open/Next/Close query-time distillation of the selected past
  tasks + their sibling sessions into a <= 2,000-token experience block; `retrieve_memory(extract=...)`,
  arms `VGR+SemExtract`, `VGR+Rank+SemExtract`, `VGR+SemFilter-pre+SemExtract`; `store_adapter.task_siblings`.
- `cf_latency_replay` (per-stage retrieval latency, `InstrumentedAdapter`), `cf_oh_run` (CoderForge test,
  5 arms), `cf_verified_run` + `verified_grade` + `Dockerfile.ohv_thin` (Verified 500, streamed Epoch images,
  bind-mounted OpenHands runtime, SWE-bench harness 5.0.2).
### Design Rationale
- Replicating the official agent removes the biggest memory-specific confound of v0.2.0: cards were
  OpenHands tool-call JSON shown to a bash-only agent.
- `store_adapter` resolves vids through `graph_store.gph_vid_map` (indexed) instead of the unindexed
  `vid` columns (sequential scans of 4.4 GB / 33 GB tables at full-CoderForge scale).
- SemExtract injects a short distilled block instead of ~15k tokens of raw cards; it reads only the
  bounded selection (TR-1) and its siblings, asserted to stay within the selected tasks.
### Notes & Caveats
- Seed kNN on the full store runs as an exact seq scan at hnsw.ef_search=1000 (~300 ms; HNSW unused).
  `store_adapter.FORCE_INDEX_STORES` exists but is empty (user: report only).
- OpenHands' stock stuck-loop detector ends a sizeable share of episodes (AgentStuckInLoopError); kept as
  in stock OpenHands; reported per arm.
- The v0.2.0 mini-swe-agent cf_smith run is paused at 46/100 tasks (resumable; `interim_summary_paused46.md`).
- vLLM watchdog in `cf_oh_run` / `cf_verified_run`: a server whose generation_tokens_total does not advance for
  300 s while it reports running requests is no longer routed to (logged to `logs/watchdog.log`; servers are
  never restarted by the runner); episodes take the least-loaded healthy server when their slot opens. Added
  after 8031 hung on 2026-10-19 (~02:58-03:45; restarted by the operator).
- Container labels are per experiment (`trimem-cfoh=1` CoderForge, `trimem-cfv=1` Verified). A shared label made
  the first Verified smoke's exit sweep remove 20 running CoderForge episodes; all 20 were detected as
  "container lost", set aside (`result.container_lost*.json`) and are rerun on resume - none was graded.
- Agent profile `oh-replica-v2-thinking` (Qwen3.8 card thinking-mode sampling, enable_thinking, 32k output incl.
  thinking; `--agent_profile`), reasoning capture + thinking-token counts (`ctx_guard.after`), upstream-fetch
  detector (`oh/fetch_detect.py`), per-episode stuck / cap / empty / fetch / thinking flags in the reports;
  `cf_verified_run --out/--ids_file/--agent_profile`. OpenHands 0.52.1 drops prior-turn reasoning from the
  history (documented in plan v0.3.0 section 6).
- `scripts/chain_cfoh_then_verified.sh`: when the CF run has 100 tasks done (resuming it up to 3 times), writes
  its final report and starts Verified 500 x 5 arms (user-approved).

## [CoderForge SWE_Smith memory experiment: split, store, renderer v2, runner] - 2026-10-18
### Features
- `cf_smith_split`: split `coderforge_smith_400_100_v1` (seed 20261018; CoderForge SWE_Smith partition only,
  user decision): 10 repos (top by eligible mixed-outcome tasks) x (10 test + 40 train), same repos on both
  sides, mixed-outcome-first SHA256 order, exact-duplicate skip, `canonical_repo` alias map, `--verify`
  (reproduces the written keys). Opt-in `--gate` variant (official gold/empty check) exists but is NOT used
  (user decision: no gating; known-bad tasks are flagged in the report).
- Store `trimem_coderforge_smith_train400` (PG 55450; 400 tasks / 1,600 sessions / 109,564 steps; 462,256 native
  edges) via `swe5_train_store --dataset cf_smith`; vectors from a private GPU0 embedding server (fp32,
  cosine >= 0.9997 vs CPU).
- `memory_render` renderer `swe-card-v2` (default): same/related-repo header, no task ids ("past issue #k"),
  collapsed file views, parsed commands; `swe-card-v1` via `style="v1"`; version recorded in receipts.
- `retrieval`: configs now carry their edges; new default `swe-vgr-v3` = v2 + edges `similar` + `similar_up`
  (user decision); v1 / v2 reproducible via `CONFIGS`. NOTE: runners that relied on the implicit default
  (e.g. `rebench_run`, `swe5_replay`) now get v3 unless they pass `config=`.
- `swe5_leakage` / `swe5_replay`: dataset `cf_smith`, outputs under `results/coderforge_smith_20261018/`.
- `cf_smith_run` (preflight / run / report; 5 arms, 16 concurrent, resumable, label `trimem-cfsmith=1`,
  RSS / RAM / disk sampler with abort), `cf_smith_grade` (official harness in `.venv-swesmith`, one process
  per grade, harness errors never cached as verdicts), `cf_smith_report` (paired McNemar, per repo, all tasks
  and excluding known-bad-gold tasks, unresolved-by-every-arm, step-cap / autosubmit counts). `swe5_harness`
  gained a job-file CLI.
- `agent_driver.run_episode(autosubmit=True)`: SWE-agent-style autosubmission on any non-submit exit (step /
  wall cap, errors): `git add -A && git diff --cached <HEAD after startup>` collected before the container is
  removed; `autosubmitted`, `autosubmit_reason`, `base_commit` recorded. Default off (other runners unchanged).
  The cf_smith run uses step_limit 100 / wall 3,600 s (official CoderForge eval: OpenHands max_iterations 100);
  the stopped 50-step v0 run is kept in `episodes_v0_step50/`.
- `cf_smith_run`: each arm is graded right after its own episode (grading semaphore `--max_grades`, default 8,
  independent of the 16 agent slots; `--task_workers` 8); before, the 5 arms were graded sequentially after the
  slowest arm. Shutdown safety: SIGTERM / abort first sets the abort flag, so in-flight episodes never write
  `result.json` after the containers are swept (they are written aside as `result.discarded*.json`); an episode
  whose container vanished (`agent_driver.container_lost`: docker "No such container") is set aside
  (`result.container_lost*.json`) and rerun on resume. Bug found: the first per-arm restart's SIGTERM sweep
  produced 3 empty LimitsExceeded results ("collect failed: No such container"); set aside and rerun
  (`logs/set_aside_container_lost.json`). Empty-autosubmit reason now reads "no changes in /testbed".
### Design Rationale
- SWE-smith repos need one image each (~5 GB); R2E_Gym / SWE_Rebench need one per task, so they are deferred.
- `swe5_harness` pre-seeds the mirror-privacy flag: the vendored harness calls the GitHub REST API once per
  process (60/h unauthenticated); past ~60 grades every grade failed with "403 rate limit exceeded" and was
  reported as status `error` / unresolved. The swesmith mirrors are public (cloned anonymously).
- `swe5_harness` test-suite timeout floor 600 s (profiles default 90 s; under 16 agents + 8 graders sqlfluff's
  suite took 90.05 s -> status `timeout` = unresolved regardless of the patch); recorded as `timeout_s` /
  `profile_timeout_s`. A prediction whose patch does not apply is now the verdict `patch_apply_failed`
  (unresolved) instead of a harness `error` that was retried forever.
### Notes & Caveats
- Official gold fix does not resolve for 12 of the 54 checked test tasks (all 10 checked tkrajina/gpxpy - bug
  injected into its root `test.py` - and 2 pydata/patsy); kept in the run, flagged in `known_bad_gold.json`.
  The earlier "gold failed" sqlglot/conan results were GitHub rate-limit harness errors, not real failures.

## [SWE 5-repo memory: preflight, SWE-smith train-only store, llm-budget-v3] - 2026-10-18
### Features
- `llm_budget`: versioned protocols (`Budget`, `PROTOCOLS`, `get`); new `llm-budget-v3-2026-10-18` =
  context 262,144 / memory 131,072 / output 8,192, used by every SWE runner via `llm_budget.SWE`. Module
  defaults stay v2, so non-SWE runners reproduce unchanged.
- `swebench.retrieval`: versioned configs `swe-vgr-v1` (depth 1, top_k 3) and default `swe-vgr-v2`
  (seed 8, depth 3, fanout 4, top_k 5); `exclusions=` applies a precomputed per-query exclusion list and
  asserts it on every call (`swe5_leakage.assert_no_leak`); `plan=` override for the LLM-free replay.
  `store_adapter`: stores `swesmith_5r` / `coderforge_5r`, records blocked task keys.
- `swe5_leakage` (protocol `swe5-leakage-v2`): test holdout + own task + SAME-FUNCTION/class exclusion
  (ast entities on the base commit, every bug patch checked with `git apply --check`); near-duplicate
  issue texts (cosine >= 0.95) reported only (user decision 2026-10-18). Lists in
  `results/swe_memory_5repo_20261018/exclusions_<dataset>.json`.
- `swe5_actions` + `swe5_train_store`: train-only store `trimem_swe_smith_5r_train` (PG 55450) with
  SWE-smith re-render dedup (1,641 -> 856 sessions), canonical editor verbs, action-derived patch summaries.
- `swe5_split`: re-implements the swesmith_5repo_500_100_v1 rule (reproduces its 100/500 keys exactly)
  and built `coderforge_5repo_500_100_v1` (CoderForge work paused by the user afterwards).
- `swe5_harness` (vendored SWE-smith @ 9b74ac0, `.venv-swesmith`), `swe5_preflight`, `swe5_env`
  (instance-branch checkout + git-history wipe, container label `trimem-swe5=1`), `swe5_loadtest`,
  `swe5_replay`. `cpu_embedding_server --device` (default cpu; GPU instance fp32) for bulk store vectors.
### Design Rationale
- SWE-smith-trajectories' `patch` column belongs to other rows (15/2,364 rows of tool-00000 contain their own
  patch's first added line), so card patch summaries and the gave-up label come from the session's edit
  actions; the same run appears under several traj_ids (tool/xml/ticks and repeats), deduplicated by the
  per-step operation + argument + observation sequence.
- Same-function uses exact class-chain + function entities from `ast` (hunk headers over-match whole
  classes: 55 vs 41 test queries); ancestor-inclusive and hunk-header counts are kept as sensitivity.
### Notes & Caveats
- `trimem_coderforge_5r_train` was built before the CoderForge pause; it is not used. CoderForge's split
  crosses the SWE-smith split (canonical ids overlap 89 tasks), pending the user's decision.
- The agent environment hides git history; the harness grades in a fresh container (official flow).

## [SWE trajectory expansion v1] - 2026-10-18
### Features
- Add pinned SWE-smith, raw CoderForge-Preview, and R2E-Gym SFT downloads and adapters for native Task → Session → Step graphs.
- Deduplicate SWE-smith format exports, merge CoderForge interrupted-run timestamp suffixes into their original task, and recover R2E-Gym task identities only through unambiguous exact issue matches.
- Extend the experience atlas with task and session similarity edges, source split and nonempty-session counts, and a concise evaluation resource report.
### Design Rationale
- Retain every canonical session and step, including failures and unknown outcomes. Keep full raw text with pinned revision and SHA-256 provenance while storing bounded text previews.
- Use the existing dense native batch edge API and stage structural edges from imported records, avoiding Python lists of millions of step edges and scalar linear vertex lookups.
- Bind embedding caches and assembled arrays to ordered identities, content, and the pinned model. Keep the three new databases separate from existing corpora.
### Notes & Caveats
- These releases contain training trajectories without an official held-out trajectory split. R2E-Gym SFT omits per-row success labels and some task environments cannot be recovered from the available task metadata.
- Image size samples are registry compressed-transfer sizes, not measured disk usage. Agent evaluation duration is a stated planning scenario, not a benchmark result. HTML exports bounded subgraphs from the full native stores.

## [Trajectory memory policy success+failure-v1] - 2026-10-18
### Features
- Trajectory memory now also injects FAILED sessions, clearly labelled (user decision 2026-10-18), for
  ALFWorld / ScienceWorld / WebShop (AgentGym session + router evaluators), AppWorld and τ² (runners and
  router runners), and every plan built on them (VGR, VGR+Rank, P2, P2+Rank, BO / router configurations).
- One versioned switch `trimem/experiments/memory_policy.py`: `success+failure-v1` (default) or
  `success-only-v0`, via `$TRIMEM_MEMORY_POLICY` or `--memory_policy`. Under v1 every card starts with
  `SUCCESSFUL ATTEMPT — ...` or `FAILED ATTEMPT — this approach did NOT solve the task (outcome: ...); do not
  repeat it; use it to avoid the same mistake.`, the memory header explains both kinds.
- `semantic_rubric`: new `semantic-rubric-v2` (`SYSTEM_V2`, `system_for`, judges take `version=`): a failed
  episode is useful only if it reveals a concrete pitfall that applies to the new task. `VERSION` / `SYSTEM`
  (v1) unchanged.
- Policy in experiment ids (`:mem=success+failure-v1`), memory-arm result dirs / BO driver state / router
  artifacts (`__mem-sf1`), opt `protocol.memory_policy`, and every identical-memory / cross-experiment
  reuse key. Per-episode successful vs failed card counts in opt `receipt.card_outcomes`,
  `generation.card_outcomes` and the per-episode result / memory files.
- Offline replay `trimem.experiments.memory_policy_replay` (no LLM) and tests `tests/test_memory_policy.py`
  plus policy cases in the AppWorld / τ² / WebShop / reuse tests.
### Design Rationale
- No quota for failures: the pool and selection are unchanged, only admissibility widens, so a failed card
  appears exactly when it is among the best candidates. Unknown outcomes (`success IS NULL`) stay out.
- Success-only stays byte-identical (cards, headers, receipts, ids, dirs), so finished runs remain
  reproducible and resumable; v1 results can never be mixed with them (tagged ids/dirs, `.memory_policy`
  marker, policy in reuse keys, routers deployed only under their training policy).
- ColBERT (Rank) scores the card without its outcome label: with no failed candidate the v1 top-k equals
  the v0 top-k; SemFilter/SemRank judges see the label.
### Notes & Caveats
- Failed sessions per store: ScienceWorld 526, τ² 345, WebShop 1, ALFWorld 0 (+239 unknown, excluded),
  AppWorld 0, SWE-QA-Pro none (ungraded teacher sessions, no outcome; code unchanged). Replay (20 train
  queries): failed cards in top-3 for ScienceWorld 9/20 (VGR) and 2/20 (VGR+Rank) queries, τ² 10/20 and
  15/20; memory ≤ 6.2k tokens. Design note: `docs/trimem_multidataset_trajectory_memory_eval_addendum_v0.1.2.md`.
- No-Memory baselines are shared (never tagged). v1 BO/test runs start fresh experiments; cross-experiment
  reuse from the success-only train100 experiment no longer matches.
- SWE-bench (`trimem/experiments/swebench/`) is not changed here; it can adopt `memory_policy` and
  `semantic-rubric-v2` (`RubricSemFilter(..., version=...)`).

## [TRACE design-review documents v0.1.0] - 2026-10-18
### Features
- Recast the cooptimization and SIGMOD PDFs as concise design-review documents without experimental results.
- Add shared TRACE method text: four-channel profiling with an active LLM analyzer, BRANE-based per-query model training, and frozen deployment routing.
- Redraw cooptimization Figures 1/2 and incorporate SIGMOD system/trajectory Figures 1/2 as Figures 3/4.
### Design Rationale
- Separate acquisition-driven exploration from query-dependent deployment, and stage diagnostics from pre-execution features.
- Preserve BRANE/RAG-PE attribution while naming the combined optimizer TRACE.
### Notes & Caveats
- Documentation update only. Runtime code and existing model artifacts retain their identifiers.
- Original manuscript sources and PDFs are archived under paper/trajectory_cooptimization/versions/.

## [SWE trajectory experience graphs v0.1.0] - 2026-10-18
### Features
- Three SWE agent-trajectory datasets imported as TriMEM experience graphs on PG 55450: `trimem_swe_rebench_oh`
  (6,306 tasks / 67,074 sessions), `trimem_swe_agent_nebius` (3,591 / 80,036), `trimem_swe_gym_oh` (2,438 / 6,055),
  with native Task/Session/Step vertices, repo/model hubs, kNN similarity edges and Qwen3 task/session vectors.
- Normalizers + loader (`trimem/data/swe_traj_normalize.py`, `swe_traj_load.py`), 10 unit tests.
- Atlas page: three new dataset tabs with stats and native-retrieval subgraphs; relational-tables view lists the DBs.
  Corpus layout no longer overlaps labels when a task has several sessions ("N sessions" group note instead).
### Design Rationale
- Steps loaded for a deterministic resolved/unresolved-alternating subset (≤400k per DB): one graph-AM page per
  vertex would make all 6.4M steps ~50 GB.
- Overlap with SWE-bench Verified 500 checked (0 tasks, 0 repos) so the stores are usable as memory for Verified.
### Notes & Caveats
- Do not run a database-wide `ANALYZE` on TriMEM DBs: it reads graph_store.gstore pages as heap tuples and fails
  with bogus-xid / MultiXact errors (not corruption; amcheck clean).

## [HotpotQA experience corpus view v0.2.0] - 2026-10-18
### Features
- Default KBevo visualization to a 24-task corpus subgraph with Task → Session membership
  and native cross-task similarity edges. Provide 12 / 24 / 100 / all 5,743 task views,
  pan / zoom, node inspection, directed edge details, and displayed-subgraph downloads.
- Expand all 18,012 steps to show all 29,498 native vertices and 37,697 forward edges.
  Preserve trajectory drilldown and provide an explicitly labeled Session projection of Task similarity.
### Design Rationale
- Canvas handles the full corpus, including isolated tasks, without creating tens of thousands of DOM nodes.
  Weakly connected components group related tasks for display. Layout coordinates are not semantic distances.
- Verify rendered forward topology against the existing native graph export before publishing.
  Collapse inverse edges for readability. The optional Session projection is display-only and is not counted
  as independently stored topology or evidence of retries, improvement, or causal memory use.
### Notes & Caveats
- At the existing cosine >= 0.65 threshold, 847 tasks have a similarity neighbor and 4,896 do not.
  Small subgraphs deliberately prioritize connected tasks, while the full view includes every task.
- Snapshot export and browser rendering only. No database schema, inference service, or graph operator changes.

## [Agent-env WebShop v0.1.0] - 2026-10-18
### Features
- WebShop (AgentGym `webshop`) is the fourth agent-env dataset, wired through the ScienceWorld pipeline:
  `bo_agentenv_driver` `DS["webshop"]` (ReAct, env 36003 / 36013, quality reuse on train100), new stage
  `test-fixed` (No-Memory + VGR / VGR+Rank / P2 / P2+Rank plan defaults on AgentEval 200),
  `query_router_agentenv --dataset webshop`, summaries / renderers / xlsx include WebShop.
- Managed env server `trimem.experiments.webshop.env_server` (`env_server_managed --env webshop`,
  `scripts/env_servers.sh webshop`): official FastAPI app, one shared seeded SimServer, private per-env
  sessions released on reset/close, `/close`, idle reaper, 32-env cap (HTTP 503), in-process JVM capped
  at `-Xmx1g` SerialGC, `/managed_stats` (live envs, sessions, process RSS / threads, child JVMs).
- Store `trimem_webshop` (`trimem.experiments.webshop.store_build`; same DDL / edges / Qwen3-Embedding vectors
  as `trimem_agentenv`): 3,930 AgentTraj-L sessions (success from an official-client replay,
  `webshop.replay_trajs`), 200 held-out AgentEval tasks. Adapter: `agentenv_task_adapter.dsn_for`,
  WebShop family = `task.scene`, WebShop memory header; rubric knowledge for WebShop.
- Frozen panel `data/external/panels/webshop_train100.json` and test shards `webshop_{a,b}.json`
  (`webshop.panels`); chain `results/bo_agentenv_20261016/run_webshop.sh`.
### Design Rationale
- Separate database instead of rows in `trimem_agentenv`: its loader has no incremental mode and the store
  backs running / finished ALFWorld / ScienceWorld experiments (same choice as `trimem_appworld`).
- AgentGym ids index the 6,910 *synthetic* goals of the 1,000-product set (verified on all 3,930 AgentTraj-L
  ids), so the 1.18M-product index is not needed (3 MB Lucene index, ~1.1 GB server RSS).
- `generate_product_prices` draws ranged prices from the unseeded global `random` before the goals are
  built, so upstream the `price lower than X` clause of a goal changes between server starts and between the
  official per-env SimServers; seeding one shared SimServer makes every arm / GPU half / restart see the same
  goals (paired gains need identical tasks). The replay shows the env is otherwise unchanged (15,933 /
  15,935 logged observations byte-identical, 3,929 / 3,930 trajectories reach reward 1.0).
- L1 family = instruction with the price clause stripped (user definition); `--family-mode product` exists.
### Notes & Caveats
- Product-level overlap: 3,085 / 3,930 training trajectories are on an ASIN that also has a test goal
  (different options). Allowed under the instruction family, excluded under `--family-mode product`.
- `scripts/env_servers.sh stop_port` matched any process whose command text mentioned
  `env_server_managed ... --port N` (it killed a calling shell); it now requires the server binary as argv[0].
- In-process JVM: `/managed_stats.jvms` counts JVM child processes (always 0 for WebShop).

## [HotpotQA experience graph v0.1.0] - 2026-10-17
### Features
- Import the pinned KBevo HotpotQA release into isolated `trimem_kbevo`: 5,743 QA sessions,
  5,743 tasks, 18,012 steps, 29,498 native vertices and 75,394 directed edges including inverse edges.
- Store complete source QA records and parse recorded KB lookups. Embed complete questions with
  Qwen3-Embedding-0.6B, index with pgvector HNSW, and add top-5 question-similarity edges at cosine >= 0.65.
- Add a searchable, keyboard-accessible graph explorer to `qa-trajectory.html` with source text,
  malformed-call inspection, answer F1, native-graph downloads and a file-level public-source census.
### Design Rationale
- Task/session/step topology uses the existing graph AM and `trimem.expand`. A separate database
  avoids mutations to other experiment stores. The relational edge mirror holds properties only.
- Pin source revision and SHA256, preserve malformed spans without inferring missing tool results,
  and verify every native adjacency plus raw-record and observation round trips before publishing.
### Notes & Caveats
- Actual release contains 5,743 QA records plus 6,003 auxiliary extraction records, contradicting
  its dataset card's 11,746-trajectory / 5,744-question interpretation. Each QA question occurs once.
- 82 malformed lookup-like spans occur in 63 sessions. Some source `triplets` fields are unrelated
  to their questions, so they are retained as unverified metadata and excluded from graph topology.
- Similarity links are derived, not retries or causal memory-use evidence. F1 is source-supplied.
  This is a corpus construction/exploration experiment, not held-out memory-quality evaluation.
- Native storage verified on stock PG16 x86_64. No GX10 fork or benchmark sign-off is implied.

## [Agent-env BO train500 v0.1.0] - 2026-10-17
### Features
- Panels `data/external/panels/{alfworld,sciworld}_train500.json` (hash 43d56effa3a90b5c / d3268af138a92011):
  the 100 train100 items first + 400 new training tasks drawn with the train100 rules (same pool, strata,
  seed 20261016 and per-stratum shuffles, nested; test items and test task keys excluded; server-checked,
  0 unloadable). `build_agentenv_panels --extend 500`.
- `bo_agentenv_driver --panel_name train500 --batch 10 --rounds 9`: experiment `<ds>:train500:bo3obj-v2`
  (`session_bo_agentenv.experiment_id(ds, panel_name)`; the default still returns the train100 id), results
  in `results/bo_agentenv_20261016/<ds>_train500/`, 100 configurations (round 0 = 4 defaults + 6 Sobol,
  9 x 10 kriging-believer picks), No-Memory of the shared 100 items copied from train100, test arms
  `BO500-Qbest` / `BO500-Knee`.
- Quality reuse for train500 on both datasets, plus `agentgym_session_eval --reuse_from/--reuse_from_dirs`:
  reuse ORIGINAL train100 episodes of the same query with byte-identical memory, same llm_budget and action
  format (never noise repeats, never test arms; rows record `reused_from_experiment`).
- `query_router_agentenv --panel_name train500` (suffixed features / CV / `router_brane-paper_train500.json`).
- Reporting: `render_bo_results_section` per (dataset, panel) blocks with per-round hypervolume, strict and
  relaxed (z = 1) fronts with configurations; BO500 arms in `summarize_controlled` / `render_controlled_section`;
  `export_agentenv_xlsx` panel / experiment columns and cross-experiment reuse counts.
- Chain `results/bo_agentenv_20261016/run_train500.sh` (waits for "NEWENV BO DONE"; not started).
- Tests: `tests/test_train500_panels.py`, `tests/test_train500_profiling.py`, `tests/test_session_eval_reuse_from.py`.
### Design Rationale
- Separate experiment id / directory / test-arm names keep the finished train100 artifacts and commands
  unchanged; the driver lock stays per dataset because it guards the shared env servers.
- Nested draw: train100 is exactly the head of each stratum's shuffled pool, so extending the same
  permutation keeps the 500-panel a superset drawn under identical rules.
- ALFWorld train100 rows predate `memory_sha256` / `result_file`; the source episode is identified by the
  retrieval id written into each memory file, so cross-experiment reuse still requires identical memory.
### Notes & Caveats
- Estimated ~2.2-3 GPU-days with reuse (~5 without); see `docs/trimem_train500_plan_v0.1.0.md`.
- The analyzer channel is on in the chain (as in train100); drop `--analyzer` for pure BO.

## [AppWorld memory arm v0.2.0] - 2026-10-16
### Features
- The runner drives the LATEST official AppWorld agent by default (user decision 2026-10-16):
  `trimem/experiments/appworld/agent_head.py` + `configs/simplified_react_qwen38_27b_fp8.jsonnet` run the
  `vendor/appworld` HEAD (42b5bcf) `appworld_agents` simplified ReAct code agent with the official evaluator
  (`appworld.evaluator.evaluate_task`). `run.py --agent_version head|0.1.3` (default `head`) keeps the
  v0.1.3.post1 recoma path; `APPWORLD_ROOT` defaults to the matching data root and its `version.txt` is checked.
- New venv `vendor/appworld-venv-0.2` (appworld 0.2.0.dev0 + appworld-agents from `vendor/appworld`, pydantic
  2.12.5, psycopg, numpy, httpx) and data 0.2.0 in `data/external/appworld-0.2` (gitignored, encrypted-
  redistribution only). Frozen panel `data/external/panels/appworld_traindev_v02.json` (147 = train 90 + dev 57,
  hash d0c95ab66e344d80, ids identical to the 0.1.0 panel). `common.DEFAULT_ROOT` is now the 0.2.0 root.
- Tests: `tests/test_appworld_head_agent.py` (HEAD venv + data 0.2.0 only: config vs the official gpt-4o
  test_normal config, a full episode with a fake OpenAI client = memory on the `Task:` line, `extra_body`,
  `max_tokens`, official evaluation pass / fail, and the `--replay_gt` runner path); pure glue tests in
  `tests/test_appworld_memory.py` (agent flag, data-version guard, steps pairing, `extra_body` binding).
### Design Rationale
- Everything of the official agent is unchanged (prompt, loop, no truncation as in the official configs,
  max_steps 50, usage tracker, evaluator); only the model endpoint (official self-hosted vLLM convention,
  thinking off, `max_tokens` 8192 from llm_budget) and the memory suffix of the task instruction differ.
- `extra_body` is bound to the raw OpenAI call by the glue: the official `non_cached_lm_call` raises on unknown
  kwargs, so the official vLLM model entries' `extra_body` cannot reach the client (verified offline).
- Opt experiment ids gain `:agent-<version>` so rows of the two agents never pair or reuse each other.
### Notes & Caveats
- Data 0.2.0 = 0.1.0 except `db_version`, solution import lines and train scenario b0a8eae (supervisor /
  datetime / ground-truth API calls changed); instructions and split ids are identical.
- pydantic >= 2.13 breaks `import appworld` at HEAD (`GenerateSchema.match_type` signature); pinned 2.12.5.
- Nothing live has run; the store is not created. `llm_calls` on the HEAD path = agent steps (retries uncounted).

## [AppWorld memory arm v0.1.0] - 2026-10-16
### Features
- `trimem/experiments/appworld/` (plan `docs/trimem_three_dataset_plan_v0.1.0.md` A1-A3): store builder for
  `trimem_appworld` (agentenv schema + native edges; train-90 ground-truth API-call sessions, hook for the
  official agent's own successful No-Memory train runs; embedder injected), `AppWorldTaskAdapter` (L1: own
  task, held-out, same scenario excluded; successful sessions only), a runner around the official ReAct agent
  (retrieve once, memory appended to the task instruction, official `world.evaluate()`, `opt.session_retrieval`
  via `ScalarSessionStore`, per-task result files for resume, No-Memory mode, shards / workers, quality reuse),
  the frozen panel `data/external/panels/appworld_traindev147.json` (hash d0c95ab66e344d80), and
  `tests/test_appworld_memory.py` (mocks only).
### Design Rationale
- The installed evaluator is `appworld==0.1.3.post1` with data 0.1.0; `vendor/appworld` HEAD agents need
  appworld 0.2 / data 0.2.0, so the version-matched official v0.1.3.post1 ReAct (recoma) is used, cloned to
  `vendor/appworld-v0.1.3.post1`. Only the generator (local vLLM, thinking off) and the loader (memory) differ.
- Memory is additive to the official prompt / history budgets, so No-Memory runs are the official agent.
### Notes & Caveats
- Nothing live has run (GPUs reserved); the store is not created. `max_tokens` is the shared llm_budget 8192
  (official 400). litellm 1.37.19 (recoma pin) was built from its GitHub tag; it is no longer on PyPI.
- AppWorld plaintext (results, memory, agent code) is only written under `data/` (gitignored).

## [Per-query router v0.1.0] - 2026-10-16
### Features
- `trimem/optimizer/query_router/` (plan `docs/trimem_query_router_plan_v0.1.0.md`): dense `config x query`
  outcome matrix from the session-BO training rows (`matrix`), pre-episode query features (task structure,
  one cheap VGR seed probe, PCA of the task vector, optional BRANE-style LLM yes/no labels; `features`),
  BRANE-style router (fuzzy Pareto candidates + per-configuration logistic / ridge predictors; `brane`),
  contextual GP router (one joint GP per objective over [config (+) query] features; `contextual_gp`),
  repeated K-fold CV with router overhead charged and a headroom gate (`evaluate`), numpy-only JSON artifact
  and routing rule `argmax_c q_c(x) - lambda (w_T ln tokens_c(x) + w_L ln latency_c(x))` (`runtime`).
- CLI `trimem/experiments/query_router_agentenv.py` (`features`, `cv`, `train`, `route-test`) and the deployment
  evaluator `trimem/experiments/agentgym_router_eval.py` (patches `agentgym_session_eval` in its own process).
### Design Rationale
- The BO keeps deciding which configurations to measure; its dense matrix is exactly the BRANE training data,
  so the router costs no extra episodes. Deployment needs only numpy because `.venv-agentgym` has no sklearn/torch.
- The running BO's evaluator and driver files are untouched (each arm re-imports them).
### Notes & Caveats
- First offline result (ALFWorld, 39 configurations, 10 x 5-fold CV): router - Q-best single = +0.02 +/- 0.02
  quality; within noise. No routed test episodes have been run.
- NoMemory can be a CV column (`--nomemory`) but deployment routers must not contain it.

## [Experience graph atlas v0.1.0] - 2026-10-16
### Features
- Unified six-dataset interactive atlas at `triMEM_optimizer/it20_summary/qa-trajectory.html`, with standalone embedded data, searchable Task/Session/Step inspection, graph pan/zoom, recorded retrieval playback, semantic decisions, and dark/mobile layouts.
- Reuses ALFWorld, ScienceWorld and SWE-QA exports and historical retrieval receipts. Imports AppWorld's 90 train API demonstrations, WebShop's 3,930 AgentTraj-L sessions and 712 historical τ²-bench train-ID sessions into an isolated native graph-store prototype.
### Design Rationale
- User requires no GPU use or interference with background experiments. New vectors are explicitly CPU-only hashed unigram TF-IDF (512d), with graph traversal in a separate Unix-only PG16 process restricted to one CPU and stopped after export.
- New SemFilter decisions are author-written offline illustrations, as explicitly requested. Historical model decisions and offline annotations are labeled separately. The original QA metrics page is backed up.
### Notes & Caveats
- The HTML contains representative subgraphs and all example retrieval candidates, not every node from each full store. Source counts and exported counts are separate.
- τ²-bench is a train-ID subset of GPT-4.1 default runs, not the entire trajectory archive. Shared entities and near-duplicate tasks remain, so these are modeling examples, not leakage-clean evaluation results.
- New lexical vectors are not neural embeddings. Unknown rewards remain unknown. AppWorld plaintext is for local viewing only.
- Browser and data validation uses Chromium with GPU disabled and verifies standalone offline rendering.

## [Agent-env three-objective BO v0.1.0] - 2026-10-16
### Features
- Frozen 100-query training panels `data/external/panels/{alfworld,sciworld}_train100.json`
  (`trimem/experiments/build_agentenv_panels.py`, seed 20261016, stratified, server-checked, test ids excluded).
- `trimem/optimizer/session_space_agentenv.py` (plans VGR / VGR+Rank / P2 / P2+Rank, edges knob, 1,728 configs)
  and `session_pool.use_space()`; SWE-QA space and behaviour unchanged.
- `trimem/experiments/session_bo_agentenv.py`: reuses the session-pool engine (three GPs, qLogNEHVI), round 0,
  kriging-believer batches of 5, hypervolume / Pareto, frozen Q-best / Knee selection.
- `trimem/experiments/bo_agentenv_driver.py`: resumable controlled profiling + held-out test driver;
  `agentgym_session_eval` gains `--panel / --edges / --protocol_extra`.
- 2026-10-16 (during the ALFWorld run): stagnation rule from round 2; LLM analyzer channel (RAG-PE R^LLM,
  `trimem/optimizer/analyzer_session.py`, Codex `gpt-5.6-luna`; `gpt-5.5-luna` is rejected for ChatGPT accounts) with
  per-stage sub-metrics (`trimem/experiments/session_submetrics.py`) from round 3; single-point-front trade-off-arm
  gap; noise-aware relaxed Pareto front (z = 1 pooled SE) for stagnation, gaps and the analyzer from round 4;
  driver kill-group fix (an unreaped zombie leader kept the group id alive).
### Design Rationale
- Reuse over rewrite: the session-pool engine already models quality, ln tokens and ln latency.
- The optimizer reads only split=train rows of `<ds>:train100:bo3obj-v1`; test results never return.
### Notes & Caveats
- Adversarial review (codex, 3 lenses): REJECT -> 10 findings fixed before launch
  (results/bo_agentenv_20261016/review/VERDICT.md).
- ScienceWorld's AgentGym training list covers 7 of 10 topics; the panel stratifies over those 7.

## [Experiment resource safety v0.1.0] - 2026-10-16
### Features
- `trimem-eval.slice` (systemd user slice): every experiment process runs inside it with
  MemoryMax 100G (80% of RAM), MemoryHigh 94G, no swap, CPUQuota 32/40 cores, TasksMax 6000,
  lower CPU/IO weight than login sessions. Launch with `scripts/in_eval_slice.sh`; `scripts/mem_guard.sh`
  pauses eval clients when host RAM reaches 80%.
- `trimem/experiments/env_server_managed.py`: official AgentGym ScienceWorld/ALFWorld servers plus
  leak guards: JVM `-Xmx1g -XX:+UseSerialGC -XX:CICompilerCount=2`, idle reaper (30 min), live-env cap
  (32, evicting idle envs), ALFWorld `/close`, `GET /managed_stats`. `scripts/env_servers.sh` starts
  them in the slice with per-server MemoryMax (ScienceWorld 24G, ALFWorld 16G).
- ALFWorld server leak fixed in the managed wrapper (+3.6 MB/episode, never released): each reset left a
  `tw-vN` entry in TextWorld's module registry holding the whole env chain, and each PddlEnv dlopened a
  private copy of libdownward.so that was never unloaded. Fix: drop the reset's registry entries (resets
  serialized), close the replaced env, pool cleaned-up libdownward copies. 400-episode stress: 698 -> 807 MB
  and flat (was 692 -> 2,114 MB, linear); ScienceWorld 400 create/close episodes end at 57 MB, 0 JVMs.
- `agentgym_official_eval`: pool workers close their env clients on normal exit
  (multiprocessing finalizer) and on SIGTERM.
### Design Rationale
- 2026-10-16 07:10-11:18Z: the official evaluator never closes env clients, so each launch and
  watchdog restart leaked one ScienceWorld JVM per worker (153 JVMs, 105 GB, ~8,000 threads):
  100% RAM and swap, load ~1,000, SSH unusable, both vLLM engines OOM-killed.
- Three independent layers: clients close (normal path), the server reaps/caps (client killed with
  SIGKILL or crashed), cgroups cap whatever is left (host and SSH stay up; only a slice process dies).
- Measured: capped JVM 413 MB vs 1,185 MB RSS and 16 vs 53 threads over 30 episodes, same wall time.
### Notes & Caveats
- The idle TTL (30 min) is far above one episode step; an env reaped under a still-alive client makes
  that client's episode fail (official runner retries), which the watchdog then restarts.
- JVM flags do not change ScienceWorld semantics (GC/heap only); earlier ScienceWorld arms ran
  with default JVM flags.

## [Session trajectory retrieval v0.1.1] - 2026-10-15
### Features
- Shared session full-pool retrieval core and a native-graph SWE-QA adapter with conservative
  same-repository exclusion, P1/P2 filtering and a full-pool semantic-ranking interface.
- Deterministic historical-experience injection into the official SWE-QA system prompt,
  exact-token memory packing, explicit truncation receipts and Qwen session-level SemFilter.
- Durable `opt.session_retrieval` records link retrieval, generation and three Luna judge
  rounds. The training view excludes test, smoke, failed and incomplete observations.
### Design Rationale
- Coherent session histories provide more context than isolated tool-result steps. Topology
  stays in native graphAM and candidate ranking stays outside blocking engine operators.
- Teacher answers are explicitly silver references for profiling. The official 260-question
  test remains independent and never updates the optimizer.
### Notes & Caveats
- The five-query pilot measures filtering behavior, not answer quality. Empty results are
  retained without forced-pass fallbacks. Long records are explicitly excerpted.
- Training repository/commit resolution, session BO integration and the other dataset
  adapters remain pending. This release does not claim completed optimized evaluation.

## [Trajectory retrieval full-pool selection v1] - 2026-10-15
### Features
- `trajectory_v7` (Math, ALE and router cells): every seed is expanded, the whole result pool is
  drained, distinct attempts are ranked by SemRank score (SemRank plans) or gain (all other
  plans), then the top-k are packed. The SQL result `LIMIT 64` and stream-stop-at-k are removed.
- Multi-hop expansion de-duplicates each hop's frontier (`SELECT DISTINCT`), so work is bounded
  by the reachable set instead of the number of paths.
- Receipts record `selection: full-pool-v1` and `pool_size {distinct, rows_streamed, truncated}`;
  blocking plan nodes are recorded (`blocking: true`) instead of failing.
- ALFWorld / ScienceWorld trajectory store (`trimem_agentenv` on 55450): 11,178 tasks, 15,018
  sessions, 545,016 steps, 2,512,332 native edges; canvas experience graph, VGR traversal page
  with a reconstructed real P2 retrieval (cached SemFilter verdicts) and agent prompts.
### Design Rationale
- User decision: streaming early termination biased context to the nearest seeds and the
  lineage branch (second_autocorr expanded only 16 of 24 seeds). Ranking the full pool makes the
  choice depend on quality signals, not on stream position.
- Without per-hop DISTINCT, seeds 32 / depth 4 / fanout 16 produced 1.88M path rows (181
  distinct attempts, 74 s); with it 5,390 rows in 1.4 s (seeds 48: 8,038 rows, 2.2 s).
### Notes & Caveats
- Breaks comparability with every earlier profiling / BO / A/B point (measured under streaming
  semantics); receipts without `selection` are old semantics. Re-profiling is needed before the
  BO is retrained on mixed data.
- P1 now judges every seed and P2 every distinct pool row with SemFilter (previously only until
  k rows passed): semantic LLM calls per retrieval rise (worst case about 181 distinct rows).
- Departs from TR-1 at the selection layer only; recorded as an addendum in CLAUDE.md.

## [SWE-QA evaluation v0.1.0] - 2026-10-15
### Features
- Add the official 260-question No Memory benchmark runner with full-call token and
  latency accounting, repository-isolated official tools, and a standalone QA report.
- Add a second GPU worker that publishes completed question directories atomically
  without replacing the first worker's claimed questions.
- Queue the user-selected Qwen3.8 judge after answer generation, with three independent
  official-prompt evaluations per answer and separate judge costs.
- Visualize a verified native VGR query with selectable return paths, edge types,
  step-by-step navigation, SQL and execution-plan evidence.
### Design Rationale
- Keep scoring off the generation GPU until generation ends. Preserve official agent
  and judge logic while exposing every call and missing-score coverage.
### Notes & Caveats
- Generation is ongoing. No full-benchmark quality result is claimed yet.
- The answerer and judge use the same model. Single- and dual-GPU execution phases
  are recorded separately because resource concurrency changes latency conditions.
- The VGR demonstration runs against the separate SFT corpus and is not injected
  into No Memory answers. Any speculative worker collision is retained as extra work.

## [SWE-QA explorer v0.2.0] - 2026-10-15
### Features
- Add a Task → Session → Step explorer to the English dashboard with eight selectable
  directed edge types, session expansion, task search, similarity components and record inspection.
### Design Rationale
- Render the native-validated export directly. Load full tool evidence on node selection
  and collapse large graph scopes to keep the 25,963-node corpus navigable.
### Notes & Caveats
- This corpus has one session per task. Similarity connects sessions, with no inferred
  cross-session step edges or causal memory-use claims. All 34 browser checks pass.

## [SWE-QA import v0.1.0] - 2026-10-15
### Features
- Import all 1,000 SWE-QA-Pro SFT trajectories into the isolated `trimem_sweqa` database,
  preserving raw messages and exposing task/session/step, calls, observations, and tool schemas.
- Build full-question embeddings and HNSW seeds, native hierarchy/sequence/similarity edges,
  and an interactive explorer embedded in the English experiment dashboard.
### Design Rationale
- Keep source sequence distinct from program lineage and inferred similarity distinct from
  causal memory reuse. Preserve 74 tool responses embedded in user control messages with provenance.
- Use a separate database in the existing Postgres instance to avoid changing Math/ALE topology.
### Notes & Caveats
- Question vectors only in this first import. Similarity uses top-5 with cosine >= 0.65 across
  sessions, an explicit initial policy that is not calibrated against downstream QA quality.
- Source answers have no independent correctness labels, generation costs, or repository commits.
  The SWE-QA semantic prompt is a proposed adapter contract, not an evaluated semantic pipeline.

## [0.14.0] - 2026-10-14
### Features
- Harness search space v7 (`trimem/experiments/trajectory_v7.py`; cells `trajectory_param_cell_v7`,
  `ale_param_cell_v7`): six post-retrieval knobs — edge set, pool ordering (stream / gain /
  score / gain×score), dedup (id / code hash / one per seed / MMR 0.7), render (full code /
  diff vs. own parent / intent only), retrieval cadence (1 / 5) and SemFilter fallback to VGR.
  77,761 canonical configurations (ordering is inactive under MMR).
- Agentic analyzer (stage `trimem/optimizer/analyzer.py`): RAG-PE-style LLM proposals from
  history + per-stage sub-metrics (Claude Sonnet 5.5 via the headless `claude` CLI; DeepSeek
  optional). Every call is logged to `opt.analyzer_call`; sub-metrics go to
  `opt.segment_submetric` (375 probe windows backfilled).
- Four-channel candidate pool + arbitration (stage `harness_pool.py`): Sobol breadth, analyzer
  rows, discrete local search, Pareto-tension crossover; qLogNEHVI with a same-context
  baseline (qLogEHVI posterior-mean front otherwise); forced gap exploration after three
  rounds off the three-objective front; full-enumeration oracle.
- Offline profiling driver (stage `harness_profile.py`): panel rounds, i.e. one configuration
  on one start state per task, one iteration each; every state is stored as its own window;
  the acquisition is averaged over the panel contexts. Experiments `pe:math_v7_profile_v1`
  (25 rounds) and `pe:ale_v7_profile_v1` (12 rounds).
### Design Rationale
- Every v7 knob defaults to the v6 behaviour, and a v7 action stores only its non-default
  knobs, so all earlier windows keep their `opt.candidate` identity and become the default
  points of the larger space (unit-tested selection equivalence).
- Sub-metrics only steer candidate generation and are never GP inputs, as in RAG-PE. Panel
  members are not averaged into one observation (unlike RAG-Stack's per-config mean over
  queries), so the contextual GPs keep the state information.
### Notes & Caveats
- The legacy v6 decoder (`harness_bo.config_of`) rejects actions with v7 knobs, so the v6
  online BO never trains on mis-decoded v7 windows.
- An adversarial review (Codex) found ten issues, all fixed before launch: analyzer failure
  containment, the ALE private end iteration for cadence 5, v6 contamination, concurrent runs
  from one checkpoint, the drifting stall reference, the NEHVI baseline, interruption
  accounting, the fallback trigger, MMR/ordering canonicalization, and the double GP fit.
- Stage-tree code is not yet synced to the repo.

## [0.13.11] - 2026-10-14
### Features
- Add VGR+SemRank: native SemScore over the capped VGR candidate stream, followed by bounded agent-side score ordering and Top-4 budget packing.
- Add isolated four-task Math reruns and an ALE transition that archives legacy score-only runs before restarting that arm.
### Design Rationale
- Preserve streaming native graph operators and the old score-only protocol. Reranking is an explicit agent-side step over at most 64 VGR rows.
### Notes & Caveats
- SemRank ties preserve VGR stream order. Full-code memories that exceed the existing inspiration budget are skipped without truncation.
- Revised ALE and Math experiments share the existing semantic worker and retain 32K input/output limits.

## [0.13.10] - 2026-10-13
### Features
- Permit the strict no-reference control to use the latest trajectory prompt, original task
  system message, current evaluation feedback and full-input token accounting.
- Add a convex_13 no-memory rerun coordinator with separate 20/50 stages and ten-round checkpoints.
- Allow ALE scheduling to wait for explicitly added Math controls.
### Design Rationale
- Match current generation settings without accidentally injecting own-run history or retrieved
  programs into a no-memory arm. Preserve historical baseline measurements separately.
### Notes & Caveats
- No-memory retains the current parent program and its feedback. All three reference sections
  are empty. The main sixteen-cell Math matrix remains independently scheduled.

## [0.13.9] - 2026-10-13
### Features
- Queue AHC008, AHC011, AHC024, AHC025 and AHC026 behind completion of all Math
  trajectory runs at iteration 50, using VGR, SemRankOnly, P1 and P2.
- Save and resume exact ten-round checkpoints, independently evaluate each public-selected
  winner on the held-out private set, and retain score, rank, performance and seed-relative change.
- Add task-selectable Math/ALE quality curves and cumulative quality/token scatter plots
  to the existing HTML dashboard, with checkpoint-aligned generation and semantic API usage.
### Design Rationale
- Private outcomes are reporting-only and never inform program selection or agent feedback.
  Separate stage directories preserve provenance and prevent future checkpoint leakage.
### Notes & Caveats
- Token accounting includes observed API retries. Missing or ambiguous ownership is marked
  partial rather than charged twice. Unobserved private outcomes remain pending.
- Private evaluation uses cached ALE-Bench Lite held-out cases. These are single-seed runs.

## [0.13.8] - 2026-10-13
### Features
- Added circle packing to the four-task comparison, a fixed historical-best dashed reference,
  method legends with visibility toggles, and observed-only curves for checkpoints 10/20/30/40/50.
- Added a staged coordinator: finish all sixteen task-plan cells at 20, then resume each saved
  checkpoint into a separate directory for iterations 21–50, saving checkpoints every ten rounds.
- Designated experiment-trajectory-retrieval-v4 as the default code branch in tracked experiment
  defaults, with branch/commit/source hashes recorded in launch manifests.
### Design Rationale
- Preserve completed first-stage measurements and compare continuation quality without rerunning
  iterations 1–20. Historical corpus scores provide a stable reference rather than a moving target.
### Notes & Caveats
- Pending checkpoints remain unavailable until observed. Historical best is not an upper bound.
- Runtime data, local database directories and model/corpus files remain external experiment inputs.

## [0.13.7] - 2026-10-13
### Features
- Added task-priority parallel scheduling for the Math trajectory matrix, with live concurrency
  control, per-cell status, completed-cell reuse, and CPU/GPU embedding placement.
- Dedicated semantic worker concurrency is configurable. Parallel semantic usage with ambiguous
  context ownership is marked unavailable instead of charged repeatedly to multiple cells.
### Design Rationale
- Fill both existing generation-service GPU replicas while prioritizing within-task plan comparisons.
  Experiment semantics, prompt budgets, seeds and iteration counts remain fixed.
### Notes & Caveats
- The first convex13/VGR cell ran serially. Later parallel latency measurements include contention.
- The user stopped the optimizer experiment and retained its archive before this resource change.

## [0.13.6] - 2026-10-13
### Features
- Trajectory generation input budget is 32768 tokens including the local Qwen chat template.
  Retrieved inspiration context has a 16384-token section budget and a hard maximum of four programs.
- Budget accounting records original/final input tokens and whole-record removals. Mandatory current
  code and feedback are never clipped. An oversized mandatory prompt is rejected explicitly.
- Added the Codex execution plan for three Math tasks, four retrieval plans and 20 iterations,
  with checkpoints at 10/20 and isolated new-store experiment outputs.
### Design Rationale
- Complete source programs and evaluation feedback need more room than summary-only context.
  Separate input, reference-section and generation-output limits keep accounting unambiguous.
### Notes & Caveats
- The top-program section retains its independent 8192-token limit. The 32768 input cap excludes output.
- SemRankOnly still annotates scores without reranking. Experiment reports must state that limitation.
- Existing legacy experiments and their budgets remain unchanged.

## [0.13.5] - 2026-10-13
### Features
- Added Current Evaluation Feedback and per-attempt Evaluation Feedback to trajectory prompts.
  These render recorded metrics, diagnostics and artifacts without additional clipping or LLM summarization.
- The trajectory snapshot explicitly loads selected-parent and recent-attempt artifacts beyond the
  default OpenEvolve snapshot cap. Missing feedback remains explicit.
- SemFilter now judges contextual usefulness using complete candidate code. Shared SemScore evidence
  also includes complete code. The semantic prompt version is trajectory-context-v4-candidate-code.
### Design Rationale
- The agent needs its own observed failures in generation context, and semantic selection needs the
  candidate implementation rather than relying only on historical intent claims.
### Notes & Caveats
- Source evaluators may already have shortened diagnostics. This change cannot reconstruct lost text.
- Native semantic request limits reject oversize input instead of clipping candidate code.
- Semantic scoring remains pointwise annotation; no ranking/sorting behavior changes in this release.

## [0.13.4] - 2026-10-13
### Features
- Trajectory-store prompts preserve each task's original system message and show complete
  retrieved code plus stored intent. Default target: four inspirations and one own-run top program.
- Added actual recent-attempt history, with Changes/Metrics/Outcome based on combined_score
  versus the recorded parent. History excludes external memories and future iterations.
- Exported original-system templates for all 18 Math/ALE tasks, raw prompt examples, provenance,
  system variants and measured historical reference-program counts.
### Design Rationale
- Recent attempts, own-run best programs and retrieved experiences serve different purposes.
  Keep all three explicit. Template assembly makes no summarization model call.
### Notes & Caveats
- Opt-in trajectory-store worker wrapper only. Installed OpenEvolve and legacy runs are unchanged.
- Inspiration and top-program sections each have an 8192-token limit. Oversize programs are skipped
  whole and counted, never clipped. Thus the returned count can be below the configured target.
- Top programs are selected from retained, evaluated current-run records across islands, excluding
  explicitly invalid records and the current parent's ID/code. A missing alternative is shown honestly.

## [0.13.3] - 2026-10-13
### Features
- Added a versioned full-code Qwen3 embedding contract and resumable isolated-corpus rebuild.
  Corpus/query inputs share the same renderer, checkpoint and pooling. Oversize code is rejected,
  never silently cut. Similarity adjacency is rebuilt as same-task, cross-session cosine top-16.
- Added an opt-in OpenEvolve trajectory-store adapter with full task/parent code, actual evaluator
  feedback, objective and remaining generation budget on every retrieval. Full-context semantic
  requests replace v2 excerpts. Context files and hashes are recorded per round.
- Added deterministic inspiration evidence cards with provenance, intent, framework changes,
  fitness gain, validity/error and optional complete historical code. No post-retrieval summary
  LLM is called. Whole cards exceeding the 8192-token memory section budget are skipped and counted.
### Design Rationale
- Code suffixes and current failures must affect retrieval. A 2000-character prefix omitted code
  in 95.5% of stored Math/ALE attempts. Code vectors and derived graph edges must migrate together.
- Existing summaries/intent are evidence from history, not instructions or newly generated summaries.
### Notes & Caveats
- New-schema path is explicitly enabled with --trajectory-store and the rebuilt DB on port 55450.
  Active legacy ALE and frozen BO experiment paths retain their original embedding contracts.
- Semantic scores annotate a streaming candidate sequence. This is not a global sort by score.
- Remaining-token accounting currently covers generation calls reported in that run's telemetry,
  not semantic retrieval calls. It remains null when no total budget is configured.
- Native semantic mailbox capacity is 524288 bytes in the isolated cluster. Full current state
  is capped at 393216 bytes with explicit rejection on overflow, not truncation.

## [0.13.2] - 2026-10-13
### Features
- New-schema semantic retrieval now binds a `CurrentAgentContext` containing the actual
  problem, immediate objective, program state, failure evidence, constraints and provenance.
  Seed filters, endpoint filters, pointwise scores and lowered semantic joins receive it.
- Added `trajectory-context-v2` input version and context digest. Bounded UTF-8 context views
  report field omissions and source hashes. Unknown historical failures remain explicit.
- Updated English prompt previews and added measured graph-degree statistics, traversal
  directions and proposed seed/fanout/hop profiles to the explorer.
### Design Rationale
- Task identity alone cannot establish usefulness for the current agent state. Current-state
  evidence is supplied independently of the retrieved candidate to avoid conflating the two.
- Contexts are capped at 12,000 bytes, with request-size validation against the native
  semantic input limit. This is separate from the final memory token budget.
### Notes & Caveats
- Historical contexts are reconstructed for inspection, not claimed to be recorded semantic
  calls or leakage-free retrieval examples. Live callers must supply current state per request.
- Active frozen BO runs and their old memory formatter remain unchanged. The new query path
  does not automatically migrate their runtime. Proposed graph budgets are not quality optima.

## [0.13.1] - 2026-10-13
### Features
- Loaded the complete EvoTrace corpus into the isolated `trimem_traj` database on port 55450.
- Added optional `--merge-as-context` graph policy: lineage, context and similar, each with
  reverse adjacency. Merge Parent B remains a context edge with its original label.
- Preserved ambiguous response-to-attempt links as `llm_call.link_candidates` without asserting
  an exact attempt ID. Added graph-policy reload guards.
- Added `trimem/trajectory_query.py` for the new schema. Semantic score/filter evidence now
  includes intent provenance, framework changes and parent/child fitness instead of the old
  `changes` field. Native SemJoin is not claimed: a bounded join plus SemFilter expresses it.
- Added offline full-table, native-graph and prompt explorers in
  `triMEM_optimizer/trajectory_storage_v020_20261013/`, embedded in the English it20 report.
### Design Rationale
- Preserve active optimizer experiment hashes and snapshots. New-schema query compilation is
  a separate entrypoint, with explicit SQL/EXPLAIN artifacts and pointwise LLM score semantics.
- Native topology is exported directly and checked against the property mirror. All 11
  application tables retain every row and field in paginated files and compressed JSONL.
### Notes & Caveats
- This is a full offline corpus, not a leakage-free online evaluation snapshot.
- Intent embeddings use the pinned Qwen3-Embedding-0.6B checkpoint. Eleven long inputs use
  the documented 6000-character limit. Full intent text is retained in the relation.
- Existing BO runs retain their old scorer input. No new downstream quality claim is made.

## [0.13.0] - 2026-10-13
### Features
- TriMEM trajectory store v0.2 (`docs/trimem_trajectory_storage_v0.2.0.md`): the EvoTrace loader
  now loads every relation in the dataset, into a NEW cluster (`scripts/traj_cluster.sh`:
  `.tridb-pgdata-traj`, port 55450, database `trimem_traj`). Relational tables:
  - `task`, with 18 task_spec vectors;
  - `session`;
  - `attempt`, gaining `framework_changes`, `intent`, `intent_source`, `intent_emb`,
    `backend`, `program_id`, `generation` and `artifact_sha`;
  - `prompt` + `prompt_system`, holding full prompt and response text;
  - `llm_call`, with full content and reasoning, `attempt_id` exact links and a `purpose`;
  - `search_membership` (run-length intervals), `search_scalar` and `search_checkpoint`;
  - `edge_prop`, a property mirror and idempotency guard.
- New native edge types `context` / `context_up` (inspirations shown to the LLM) and
  `merge_parent` / `merge_parent_up` (gepa Merge Parent B). `lineage`, `similar` and their
  reverses are kept.
- `not_selected(attempt_id)` reconstructs the pool at the previous checkpoint, minus the parent
  and the context sources, which it reads from the native graph.
- `trimem/data/embed_intent.py` embeds the LLM's stated change intent through the 8011 server.
### Design Rationale
- llm_call → attempt is an exact sha1 match of the response text within a session, not a
  heuristic. Rows that need a heuristic link are left unlinked. Topology is native; properties,
  text and time are relational.
- The LLM-derived `changes` is dropped in this store in favour of the agent's own intent.
### Notes & Caveats
- The loader refuses ports 5432, 55440 and 55441 and the `.tridb-pgdata-trimem` data
  directory. Production `trimem_evo` is untouched.
- The replay code still reads `attempt.changes`, so it stays on 55441.

## [0.12.0] - 2026-10-12
### Features
- TriMEM SemExtract (`trimem_pg` 0.3.0, ADR-0028): `trimem.sem_extract(question, input) -> text`
  answers a question about a text with a short lower-case label, planned as
  `Custom Scan (TriMEM SemExtract)` like SemScore. Labels feed ordinary relational operators:
  WHERE label = ... (lifted into the node's own qual), GROUP BY / HAVING, per-label caps with
  window functions, comparison in an outer query.
- Placement enumeration covers SemFilter × SemScore × SemExtract (BASE on the extract relation vs
  TOP above the join, `trimem.force_placement`), cache-coverage-aware cost, label-qual
  selectivity from the cached label distribution; EXPLAIN shows Placement, Placement Decision,
  Inputs Extracted, Cache Hits, Worker Requests, Unlabeled, LLM Wait.
- Worker / LLM: `OP_EXTRACT`, `EXTRACT_PROMPT` + strict `parse_label`, `extract_many` on
  `OpenAIJudge` and `MockJudge` (deterministic `extract_rule`).
- Upgrade script `trimem_pg--0.2.0--0.3.0.sql`: `sem_cache.extracted text`, `sem_extract`,
  `sem_extract_support`, `complete_extracts(slots, seqs)` (REVOKEd from PUBLIC).
### Design Rationale
- The mailbox stays numeric: the worker commits the label to `sem_cache` and only signals
  completion; the node re-reads its keys with the latest snapshot (the worker commits after the
  query's snapshot). One LLM call per distinct input replaces one `sem_filter` per category.
### Notes & Caveats
- `sem_extract` inside a join condition is rejected (v1); compare labels of two relations in an
  outer query over a MATERIALIZED CTE. Label quals are never placed above an outer join.
- Unparsable replies return NULL and are not cached, so they are asked again on the next query.
- Production needs the new `.so` installed and a restart (shared_preload_libraries), then
  `ALTER EXTENSION trimem_pg UPDATE TO '0.3.0'` in each database.

## [0.11.0] - 2026-10-12
### Features
- Objective O4 (v3 space): ĝ/ĝ_ref − 0.1·L̂/L_ref with L̂ = expected LLM inference seconds per valid
  child, from per-candidate output-token and validity models and a fitted latency law
  (`trimem/objective/costs.py` joins OpenEvolve telemetry to iterations by prompt hash).
- Mixed traversal path (similar → lineage, two chained TriMEM Expand nodes) in the graph space
  (27 configurations, 325 candidates); online models scaled to Var(Δ_best).
- Workbook `triMEM_optimizer/trimem_experiments.xlsx` holds every plan (fixed, random, A1–A6,
  B0–B3, C0–C3); the summary page hides the random and v1 arms by default (`--hide`).
### Design Rationale
- Input tokens cost almost no time (0.00018 s vs 0.029 s per output token, R² 0.85); penalising
  them made the optimizers inject too little (docs/trimem_downstream_objective_plan_v0.4.0.md).
### Notes & Caveats
- Output-token prediction is weak (LOTO Spearman 0.04–0.59); latency still dominated by the
  model's output length, which context barely changes.

## [0.10.0] - 2026-10-12
### Features
- Downstream-aware plan selection for TriMEM in OpenEvolve (`trimem/objective/`,
  `bench/agent_memory/gem_oe/trimem_adaptive.py`, `--arm trimem-adaptive`): per-iteration choice of
  plan family × k (v1) and, in v2, × graph traversal configuration (edge lineage/similar,
  direction down/up via native reverse edge types, 1/2 chained Expand hops, fanout 4/8/16; 289
  candidates). Every decision and its child outcome is logged in `trimem_outcome` with propensity.
- Estimators: hurdle model (calibrated logistic × gradient boosting) as a leave-one-task-out offline
  prior; Bayesian linear online model (cold / warm start, Thompson sampling); v2 trains a separate
  "inject nothing" estimator. Objectives O1 (tokens per expected gain), O2 (utility − λ·tokens),
  O3 (cheapest candidate predicted at least as good as rank:5).
- OpenEvolve cells can be resumed from a checkpoint into a new directory (`--resume-from`,
  `oe_matrix --resume-dir`), restoring the adaptive policy state (`policy.pkl`).
- Summary page `triMEM_optimizer/it20_summary/` (fixed plans to 50 iterations, optimizer section).
### Design Rationale
- v2 targets improvement over the run's best (the reported quality) instead of over the sampled
  parent; v1 arms saved 25–38% tokens but chose "inject nothing" 21–40% of the time and lost
  1–7% quality (docs/trimem_downstream_objective_plan_v0.3.0.md).
### Notes & Caveats
- OpenEvolve's evolution trace drops ~3.5% of children (parent evicted before the child returns);
  the analysis now merges the checkpoint database. The v1 offline priors were trained before this
  fix. Single seed (42) so far.

## [0.9.0] - 2026-10-11
### Features
- TriMEM (`src/trimem_pg`, `trimem/`): LLM semantic predicates as a planner CustomScan
  (`TriMEM SemFilter`) and a costed native graph expansion node (`TriMEM Expand`) in one PG plan;
  cost-based SemFilter placement over joins (every base/top assignment is a full join search);
  shared-memory mailbox to an out-of-process LLM worker; cache-coverage-aware cost model.
- trimem_pg 0.2.0 soft semantic operators: `sem_filter(predicate, input, unknown_as)` (keep or
  drop rows whose input is NULL) and `sem_score(predicate, input)` in [0, 1], planned as
  `TriMEM SemScore` (batched, cached) so an LLM relevance score can weight an ORDER BY instead of
  filtering. Replay arms A5 (keep unknown) and A6 (score-weighted rank).
- EvoTrace loader (attempt table, HNSW, lineage + cross-session similarity graphs) and the
  plan-choice and end-to-end agent replay experiments (`trimem/experiments`, `trimem/replay`).
### Design Rationale
- ADR-0027: the LLM stays out of process, but the semantic step is a real plan node, so the
  optimizer can cost and place it against vector, graph and relational operators.
### Notes & Caveats
- Requires `shared_preload_libraries = 'tjs_pg,trimem_pg'` (order matters). Stock PG16 only; the
  PG13.4 fork / GX10 build is unbuilt here. The objective does not yet include downstream agent
  cost (P5); the replay shows that term is material.

## [0.8.0] - 2026-10-10

### Features
- **Figure 1: semantic operators under one plan** (branch `feat/figure1-semantic-ops`, ADR-0026).
  `bench/agent_memory/semantic/`: LLM result cache and pending set as PG heap tables,
  sem_extract/filter/match/conflict/rank/topk/summarize with batching and offline replay,
  predicate placement via the new `tridb_place_sem` (`src/planner/join_order_sem.c`,
  join_order 0.2.0, bit-identical Python spec), streaming θsem over a `tjs_vsgt` portal.
- GEM: deferred ingest (`gem/ingest_plan.py`), strong/stale reads, ACL predicate, token
  budget, cache-backed strategy client, D10 graph-AM merge fix; P5 downstream estimator.
- EvoTrace ingest/retrieve replay experiment (`bench/agent_memory/fig1/`), write-up in
  `docs/figure1_evotrace_experiment_v0.1.0.md`.

### Fixes
- **VSGT use-after-free on updated rows** (`src/tjs_pg/vsgt_exec.c`): index AM calls ran in
  the per-row scratch context, so btree's lazily allocated `killedItems` was freed before
  `index_endscan` (error or SIGSEGV). Regression test `test/vsgt_dead_tuple_test.sql`.

### Design Rationale
- LLM stays out of process (no HTTP in the backend; TR-1 and byte-parity gates), but its
  verdicts and the pending set live in the same database and WAL; the planner sees the
  semantic leg through an additive decision function, never by changing the frozen core.
- Eager ≡ deferred requires the cache: with a real LLM two identical runs disagree on half
  the extracted lessons; with a shared cache every plan materializes the same state.

### Notes & Caveats
- Planner C built and tested on stock PG16 only; PG13.4 fork / GX10 unbuilt here.
- P−1: `make test` (2039 passed, 0 failed) and `make lint` (clean) now pass. Result trees,
  archives and vendored code are excluded from lint; graph_store_am bumped to 0.3.0
  (no SQL change) to match tjs_pg for release coherence; `candidate_policy_hash` format
  cannot reproduce hashes stored in old `results/gem_eg_e1_e2_v0.1.0` rows.
- `test_gem_live.py::TestC6Write::test_the_c6_write_is_HOT` is flaky (pre-existing).

## [0.7.1] - 2026-09-02

### Fixes
- **Selecting "No memory" on a math task blanked every panel.** The iter20 runs contain
  no control cell for the math tasks, so all four panels 409'd at once — which reads as
  a broken demo rather than as missing data. A task now declares which arms it actually
  recorded (`arms_selectable`), and the selector disables and labels the ones it does
  not have instead of letting the request fail.
- **ALE and math were showing one unlabelled number.** They do not score the same way.

### Features
- `bench/agent_memory/gem_viz/scoring.py`: the score, in each domain's own terms.
  - **math** — one evaluator, one score, no split; the panel says so.
  - **ALE** — search optimises a **public** score (the cell's `ahcNNN_n50.py`, 50 public
    cases) and the reportable number is the **private** one: ALE-Bench Lite's official
    held-out split, re-scored afterwards by `tools/evotrace/ale_private_rescore.py` and
    never fed back into search. Both are shown, separately labelled, the private one set
    off in its own column.

### Design Rationale
- **Private scores exist only on the public-best-so-far chain.** That is what the
  protocol requires: the held-out split is re-scored after the fact, so it is computed
  at the seed and at the iterations where the public best advanced, and nowhere else.
  Most iterations therefore have a public score and no private one. The panel says which
  iterations are on the chain and shows the cell's endpoint instead; it never carries a
  private number onto an off-chain iteration, and never relabels the public number as
  private. Doing either would claim exactly the held-out result the split exists to
  protect.

### Notes & Caveats
- Measured on `ale:ahc011` (`gem`, graph_expand, p=1): the chain is iterations
  [0, 2, 3, 4]; at iteration 2 the private performance is 1660 (Δseed +8, Δvs-nocontext
  +10, 300 cases); at iteration 1 there is no private score and the cell endpoint
  (1652 → 1660 @it4) is shown instead.
- The private chain files are discovered under the replay roots
  (`private_chain_scores.csv`), not hardcoded: which run holds the rescore varies.

## [0.7.0] - 2026-09-02

### Features
- **A memory selector: GEM vs No memory.** The `arm` axis is threaded through every
  panel and every route. In replay it selects a DIFFERENT recorded cell — the corpus's
  `true_nocontext` control, whose receipt has all four slot counts at 0. In live it
  skips retrieval entirely and generates with no inspirations.
- **The evaluator's verdict is the headline of panel 5.** parent fitness → child fitness
  → delta, with the judge result and the number of injected memories, above the timing
  bar instead of buried in a JSON blob.

### Design Rationale
- **The control is a different run, not a filtered view.** Hiding the injected rows from
  the GEM cell would fake an experiment nobody performed; switching arms switches which
  recorded run the panels describe. `true_nocontext` — no external inspirations AND no
  own top/diverse programs — is the strict control, because the weaker one still shows
  the run its own output and so answers "which SOURCE of references is better" rather
  than "do references help at all".
- **The no-memory arm issues no query.** Retrieving and then discarding would spend the
  latency and make "0 injected" mean something different from what the recorded control
  means by it. The three legs say "not run" rather than showing empty results.
- **The score is what the chain exists to produce.** memory → prompt → code → SCORE, and
  the score was the hardest thing on the panel to find. A zero is now labelled as a real
  outcome rather than looking like a missing one.

### Notes & Caveats
- Measured on `ale:ahc011` iteration 1 (replay): GEM injects 4 and scores 429,289.48
  (Δ −4,910.06); the control injects 0 and scores 430,640.76 (Δ −1,640.90). Two
  different runs with two different parents — this is one iteration of one task, not a
  result.
- Live generation already ran the sha256-verified evaluator; this change makes its
  verdict visible rather than adding it.

## [0.6.2] - 2026-09-02

### Fixes
- **The second live generation always failed.** `llm_telemetry.install()` patches
  `OpenAILLM` once and refuses a second, different path — it is a process-wide
  singleton, not a per-call hook. `generate.py` handed it a fresh `TemporaryDirectory`
  per iteration, so the first generation succeeded and every one after it raised
  `RuntimeError: LLM telemetry already targets /tmp/.../llm_telemetry.jsonl, not
  /tmp/.../llm_telemetry.jsonl`. The first directory was also deleted on the way out,
  leaving the hook pointed at a path that no longer existed.

### Design Rationale
- **One stable file, sliced by byte offset.** The telemetry file now lives for the
  process; each call records the file size before the LLM call and reads only what was
  appended after it. That is the shape the hook actually supports, and it makes the
  slice — not the file — the unit that belongs to an iteration.
- **If the hook is already owned elsewhere, report no tokens.** Reusing whatever file
  another owner points at would silently attribute someone else's counts to this
  iteration, which is worse than reporting none; the panel says why instead.

## [0.6.1] - 2026-09-02

### Features
- **Work in flight is visible.** Every panel shows a spinner and a ticking elapsed
  counter while its request is running; the stream emits a `stage_start` event BEFORE
  each stage, so a 30-second stage reads as running rather than as a hung server. The
  Run button disables itself and, in live mode, says what it costs
  ("~60s: incl. one real iteration"). The live-generation button counts up too.
- **The Experience Graph opens one level at a time.** It starts at Task -> Sessions —
  which is the first hop of the W1.a rollup — and expands on demand: clicking a SESSION
  draws its attempts with the lineage edges among them; clicking a NODE draws the whole
  lineage TREE that attempt belongs to, root and all. Vertices with more behind them
  have a dashed outline and say what a click will open; a "collapse all" link resets.

### Design Rationale
- **A texture is not a picture.** A task has up to 1,769 attempts and the corpus has
  10,672. Drawn at once, nothing is legible and every vertex looks the same — so the
  disclosure follows the query's own shape rather than a scroll bar.
- **Clicking a node opens its TREE, not its neighbours.** "Which tree is this attempt
  in" is the question a node click asks, so the server walks up to the root and back
  down through every descendant (a recursive CTE over `parent_node_uid`) and returns
  the connected component.
- **Expansion state lives in the client and travels on every request.** The server holds
  no per-viewer state, so a reload cannot disagree with what is on screen.
- **The stream does not light up a stage card.** Its stage names and the three legs'
  names overlap only by accident ("graph" is both), so a card highlight would point at
  the wrong thing half the time.

### Notes & Caveats
- `expand=["*"]` means "everything": the vertex inspector and the stage merge must
  resolve any vertex the panels can name, not only the ones opened on the canvas.
- Measured on `ale:ahc011` (live): level 0 is 5 vertices; opening one session draws 116
  attempts; opening one attempt draws its whole tree.

## [0.6.0] - 2026-09-02

### Features
- **Live generation.** `bench/agent_memory/gem_viz/generate.py` runs one real iteration:
  live retrieval -> `PromptSampler` -> a real LLM call -> `apply_diff` -> the cell's own
  evaluator. Panel 5 in live mode now shows a prompt, a diff and a fitness that were
  produced just now, with measured tokens and latency.
- **`seed_layer="node"` (W1.b) is the default.** The vector leg finally has real work:
  it returns several similar historical attempts with their cosine distances, and the
  graph leg walks `eg_lineage` for 4 (ALE) / 6 (math) hops instead of the 2-hop `hier`
  rollup.
- **Endpoint discovery.** The LLM and embedding endpoints are probed across known ports
  rather than taken from the receipts; `/api/health` reports which answered.
- **Replay's vector leg highlights the query anchor** — the parent program whose code
  becomes the query vector — and separately names the ANN entries where the plan
  recorded them.

### Design Rationale
- **Nothing about the experiment is restated.** Sampling parameters come from
  `run_arm.FROZEN`, the prompt from openevolve's own `PromptSampler`, the memory
  rendering from `GemMemoryDatabase`, the token counts from `gem_oe.llm_telemetry`, and
  the score from the cell's `wrapped_evaluator.py`. If this module and `run_arm` ever
  disagreed about what an iteration IS, the demo would have stopped describing the
  system.
- **The evaluator is sha256-verified before anything is scored.** A swapped evaluator
  does not error, it just returns different numbers — and a fitness that merely LOOKS
  comparable is worse than no fitness at all.
- **The anchor is the cell's own recorded seed.** Anchoring on an arbitrary high-reward
  corpus node (the first attempt) put the query above almost everything the predicate
  could admit and returned nothing: a correct answer to the wrong question.
- **"seed" means two different things and they are now separated.** The QUERY ANCHOR
  (the parent, always recorded) is not the ANN ENTRIES (what the scan returned, recorded
  only by the host plans). Replay highlights the first and names the second only where
  it exists.
- **Degradation is loud.** With `seed_layer="node"` and the embedding service down, the
  adapter still runs — as task-seeded — but `health()` reports `degraded: true` and says
  why. The old `auto` default made that substitution silently, so the same demo computed
  a different query on a machine whose embedding service happened to be down.
- **One generation at a time.** Two concurrent runs share one GPU; both would be slower
  than the recorded runs and neither latency number would be comparable, so the second
  request is refused rather than queued.

### Fixes found while verifying this feature
- **Live generation is never automatic.** Panel 5 used to load on every task switch;
  with generation wired up that fired a real ~50s GPU job nobody asked for, and
  concurrent ones collided on the single-flight lock. It is now behind an explicit
  button that says what it costs.
- **DB access is serialised.** `EgStore` holds one psycopg connection and FastAPI serves
  from a thread pool; interleaving a generation with a retrieval on the same connection
  produced sporadic 500s that looked like query bugs.
- **`retrieve()` reported the wrong query family.** It printed "W1.a" regardless of the
  seed layer; with node seeding the default that label was wrong on every live call.
- **A leg that names zero vertices now says why.** With `ale:ahc008`'s anchor at fitness
  3,669,080.6, all 10 lineage-reached candidates fail `fitness > anchor` — the reuse
  query correctly declining to inject anything. A bare "0" read as a broken panel.

### Notes & Caveats
- One iteration, never a run: a full 20-iteration cell is ~27 minutes.
- Live generation needs a cell — the evaluator, its sha256 and the frozen config live in
  the cell directory and nowhere else.
- The evaluator executes LLM-generated code. It runs through the cell's existing wrapper
  (own process group, hard deadline), with a shorter outer timeout (default 300s).
- Endpoints found on this box: LLM `127.0.0.1:8003` (`qwen3.8-262k`), embeddings
  `127.0.0.1:8021` (`Qwen3-Embedding-0.6B`, dim 1024) — the receipts record 8005/8011,
  which is where they were, not where they are.

## [0.5.3] - 2026-09-02

### Fixes
- Live reads no longer leave the Postgres backend `idle in transaction`. Every public
  read on `EvoTraceLiveAdapter` is wrapped by `_releases_snapshot`, which rolls back on
  the way out however the call exits.

### Design Rationale
- `EgStore.connect` uses `autocommit=False`, so a bare SELECT opens a transaction that
  nothing closes. A long-lived demo server then pins the oldest xmin, stalls vacuum, and
  will eventually block DDL on the tables it is reading. The failure is silent for hours,
  which is exactly why it is a decorator on the methods and a test on the class rather
  than a `finally` someone can forget to add to the next method.

### Notes & Caveats
- Confirmed read-only: a full live run leaves `gem_eg_vertex` / `gem_eg_edge` /
  `gem_eg_plan_stat` row counts unchanged.

## [0.5.2] - 2026-09-02

### Features
- The execution graph now draws **both** relations: `hier` (task → session → node) and
  `lineage` (parent attempt → child attempt). The relation the query traversed is solid;
  the other is dashed and labelled "present in the graph, not walked by this query".
- A layout switch: **traversal hops** (BFS distance from the ANN entries) or **lineage
  tree** (depth in the session's own lineage). Within a column, vertices are ordered by
  parent so siblings sit together.
- Each vertex carries `lineage_depth_in_subgraph`, `parent_vid`, `children_in_subgraph`
  and `siblings_in_subgraph`; the graph leg reports edges per relation, lineage roots and
  max lineage depth.

### Design Rationale
- **Drawing one relation was wrong, not merely incomplete.** W1.a traverses `hier`, which
  is two hops and renders as a two-level star — so the picture implied a session's
  attempts were unrelated siblings. The corpus says otherwise: 406 of `ale:ahc011`'s 414
  nodes have a parent, one parent has 38 children, and the busiest session is 9 levels
  deep. The tree was in the data and absent from the screen.
- **The untraversed relation stays visible.** Hiding it would trade one wrong picture for
  another; dashing it says "this structure exists, this query did not use it", which is
  the actual state of affairs.
- **Lineage depth is scoped and labelled.** It is computed over the induced subgraph, so
  a vertex whose parent was not reached looks like a root; the docs say so rather than
  letting "root" be read as a corpus fact.

### Notes & Caveats
- Hop bounds differ by relation: `hier` is fixed at 2; `lineage` is 4 (ALE) / 6 (math),
  pinned against `gem_oe.retrievers.W1B_HOPS` by a test.
- `island_id` is NULL for every node in this corpus, so island membership is not shown
  anywhere.
- Replay's execution graph still carries only the run's own lineage (from
  `evolution_trace`), not the corpus lineage — that needs the database.

## [0.5.1] - 2026-09-02

### Features
- Replay now **names** the legs the trace actually recorded, instead of reporting all
  three as counted-only: `retrieval_telemetry.entries` gives the ANN entry uids on the
  host plans, and every returned vertex is a recorded sample of the relational leg's
  passing side. Only the graph leg stays counted-not-named.
- The result leg carries **the retrieved content**: the injected fingerprint in replay,
  the full `gem_eg_artifact.payload` in live. Rendered inline under "what came back",
  and on any row you click.
- Selecting a leg highlights it in the corpus layout too, so replay's graph responds to
  a click rather than silently relaying out. A leg with no named vertices now says so on
  the graph and shows a dashed card instead of appearing to do nothing.
- Static assets are served `Cache-Control: no-cache`.

### Design Rationale
- **A named vertex must be drawn.** Highlighting a set whose members are absent from the
  subgraph dims the whole picture and highlights nothing, which reads as a broken panel
  rather than a correct one. Vertices a leg names but the trace-derived subgraph lacks
  are merged in, tagged with the leg that named them.
- **Hop columns only where hops were measured.** Only the live probe measures BFS depth;
  laying replay's subgraph out in hop columns would invent a depth we never measured and
  collapse every vertex into one meaningless column. Replay keeps the corpus layout.
- **"Seeded but not recorded" ≠ "no vector leg".** The native fused plans seed inside the
  operator and never write the uids out; the panel now distinguishes that from a plan
  with no ANN stage, and from a leg the trace simply does not cover.
- **The passing sample is labelled as a sample.** Every returned vertex passed the
  filter, but they are not the whole passing set and nothing that failed is recorded.
  The note states both limits rather than letting the row count imply completeness.
- **`no-cache` on static.** Without it a browser keeps a stale `app.js` after a redeploy
  and the demo silently runs old code — indistinguishable from the new code being broken.

### Notes & Caveats
- Replay's payload is a ~100-character fingerprint, not the program; the note now says
  how many characters it is and points at live mode for the full artifact.

## [0.5.0] - 2026-09-02

### Features
- New panel: **the three legs**. Vector search, graph traverse and relational scan are
  executed separately and each leg's vertex set is highlighted on the Experience Graph,
  with the operator's own fused answer shown alongside as a fourth card.
- The execution graph lays vertices out by **BFS hop** — ANN entries, hop 1, hop 2 … —
  computed by calling `gph_traverse_bounded` once per depth and differencing the reaches.
- `GET /api/stages`, `ExecutionTrace` / `ExecutionStage` / `StageNode` in the adapter
  protocol, implemented by the replay, live and stub adapters.
- The live adapter now defaults to the 8005 LLM endpoint and an 8011 embedding endpoint,
  reports both in `/api/health`, and picks node- vs task-seeded retrieval from whether
  the embedding service is actually reachable.

### Design Rationale
- **The legs are probed, not intercepted.** `tjs_open` answers the whole query in one
  call and reports counters, so the only way to name the vertices each leg touched is to
  run the separable two-stage plan and query the graph AM directly. Every payload says
  this is a second execution, and stage 4 states how many returned vertices fell outside
  the probed reach rather than hiding the discrepancy.
- **Hops are measured, not inferred.** Placing a vertex at "hop 2" because it looked far
  away would be a drawing decision dressed as a fact; differencing `reach(d)` against
  `reach(d-1)` makes the column a real BFS level.
- **Replay refuses to highlight.** A recorded run's telemetry names only `result_uids`,
  so the other three legs render as "counted, not named". Highlighting a plausible set
  would be the most convincing lie this demo could tell.
- **Schema-qualified AM calls.** `graph_store_am` installs into its own schema, which is
  not on the default `search_path`; the schema is looked up at runtime rather than
  hardcoded so a differently-installed instance still works.

### Notes & Caveats
- Under the `same_task` split with a task-level entry kind there is exactly one eligible
  task vertex, so the ANN leg is degenerate — it confirms the entry rather than choosing
  among candidates. The panel says so. Node-seeded (W1.b) runs give that leg real work,
  and live mode uses them whenever the embedding service is up.
- The stage tables cap at 200 rows; highlighting always covers the full vertex set.

## [0.4.1] - 2026-09-02

### Features
- `gem_viz` now reads `results/e2/` and shows only the cells that ran **20 iterations** —
  the recent sweeps. `--replay-root` is repeatable, `--iterations N` moves the filter and
  `--any-iterations` drops it; `bench/out/oe` still works through the same flag.
- Curated tasks are now `ale:ahc008`, `ale:ahc011`, `ale:ahc025` (requested) plus
  `math:circle_packing` and `math:heilbronn_triangle` for cross-domain contrast.
- Cells are labelled by **variant** (physical plan · seed layer · top-m · injection
  frequency), and panel 5 states the axis it is actually diffing.
- Panel 4 reports **measured** LLM tokens and API latency from `llm_telemetry.jsonl`,
  and the timing bar splits into retrieval / LLM api / evaluator+harness.
- The UI opens on each cell's first iteration whose injection gate opened, and reports a
  shut gate as sampling policy rather than as an empty result.

### Design Rationale
- **Cell discovery is layout-agnostic.** `results/e2/<run>/cells/<cell>/` and
  `bench/out/oe/<run>/<cell>/` differ by a directory level, so a cell is defined by
  holding a `run_receipt.json` rather than by a fixed glob depth.
- **The iteration filter is on by default.** `results/e2` mixes 20- and 100-iteration
  sweeps; a browser that silently offered both would invite comparisons between cells
  that never ran the same experiment.
- **`arm` was no longer the axis.** Six `gem` cells per ALE task differ by plan and
  injection frequency, so collapsing them into "gem" would have hidden what the
  experiment sweeps and made the diff compare a cell with itself.
- **Tokens are joined, not assumed.** `llm_telemetry.jsonl` has no iteration field, so
  calls are attributed to the iteration whose `[start, next start)` window on the same
  pid contains them. If any call falls outside every window the per-iteration view is
  withheld and the panel falls back to a labelled estimate — a mis-attributed token
  count is worse than none.
- **A shut gate and an empty diff are outcomes, not bugs.** Both now render as explicit
  statements; a blank panel would have read as a rendering failure and invited the wrong
  conclusion about the retrieval system.

### Notes & Caveats
- These runs contain no polyglot or cognee cells, so panel 5 is a **plan** diff, not a
  memory-system diff, and says so.
- `predicate_sql` reconstruction now handles node-seeded (W1.b) runs: the reward bound is
  `fitness_gt` (strictly beat the parent) rather than `min_fitness`, plus the finite-
  fitness clause. It still cannot recover the target session's exclusion clause or the
  parent's duplicate-code exclusions; the panel names that gap.

## [0.4.0] - 2026-09-02

### Features
- `bench/agent_memory/gem_viz/`: an interactive four-panel walkthrough of GEM on
  EvoTrace — task browser, Experience Graph with a vertex inspector, retrieval trace,
  generate-and-evaluate, plus a three-arm diff. FastAPI backend, dependency-free single
  page front end, SSE run stream.
- Two adapters behind one `DatasetAdapter` protocol: `replay` (reads the cell
  directories `gem_oe.run_arm` already wrote — no Postgres, no LLM, no network) and
  `live` (real `tjs_open` execution against TriDB, real `gem_eg_vertex` rows).
- `stub_adapter.StubAdapter`: a deliberately EvoTrace-unlike second implementation, so
  "the UI is not coupled to EvoTrace" is a test rather than a claim.
- Five curated tasks (`curated.py`), each with the reason it was picked rendered in the
  UI: 4 five-arm math tasks plus `ale:ahc015` as the cross-domain check.

### Design Rationale
- **Provenance is mandatory, not decorative.** Every payload carries `{mode, source,
  note}` and every panel renders it. A number that cannot name its file line or its SQL
  does not get shown. The mode badge means a replayed panel can never be mistaken for a
  live one.
- **Live retrieval reuses the real arm.** `EvoTraceLiveAdapter` constructs the actual
  `GemRetriever`, asks it for its own `QuerySpec`, and prints the predicate string
  `tjs_open` was handed. A parallel re-implementation would drift and the demo would
  quietly stop describing the system.
- **Rejection attribution is a labelled diagnostic.** The operator reports counters, not
  the identity of what it discarded, so live mode re-evaluates the same pushdown clauses
  over the task's node set and attributes each miss to the first failing clause. The
  clause list is rebuilt and then checked verbatim against `Predicate.to_sql()`; on any
  mismatch the panel reports attribution as unavailable instead of guessing.
- **The default cell is chosen by richness.** A curated task whose default cell injects
  nothing at iteration 1 looks broken; `injection_frequency` is part of the ranking for
  exactly that reason, and a test pins it.
- **Panel 4 shows retrieval to scale against the LLM.** On the recorded runs retrieval is
  ~0.1-0.5% of an iteration's wall clock. Since the end-to-end bottleneck is the LLM, the
  UI has to say so rather than let a retrieval-latency panel imply otherwise.

### Notes & Caveats
- Live mode needs `.venv-e0` (it reuses `gem_oe`'s retriever, which imports openevolve).
  Replay mode runs in `.venv` with no database.
- Live **generation** needs the OpenEvolve LLM endpoint; when it is down panel 4 reports
  it and points at replay. Live **three-arm diff** needs the polyglot and cognee stacks,
  so it stays a replay panel.
- Replay cannot attribute per-candidate rejection, holds no problem statement, records no
  real token counts (`est_*` is `chars/4`, basis printed), and reconstructs the predicate
  SQL from the recorded bound — it cannot recover the target session's exclusion clause.
  All four limits are stated in the UI, not just in this file.
## [Joint query-router predictors] - 2026-10-29
### Features
- New router training defaults to three separate shared predictors using configuration parameters and projected query embeddings, with observed quality, tokens, and latency as training labels.
### Design Rationale
- Sharing observations across configurations exposes plan and knob differences directly, while whole-query cross-fitting prevents a held-out task's labels from appearing under another configuration.
### Notes & Caveats
- Historical router artifacts remain readable; the `brane` backend is available explicitly. New predictors require retraining, and past experimental metrics are unchanged. The fixed forest is an initial model choice and has not yet been validated against the historical router on held-out workloads.

## [Joint-selected query router] - 2026-10-29
### Features
- Default new router training to three configuration-aware predictors, independently selecting a model family for quality, tokens and latency using query-grouped training folds.
- Restore ScienceWorld's original environment scores for this router and preserve negative scores through prediction and training replay.
### Design Rationale
- Pool observed task/configuration pairs while keeping metric models separate, and freeze selected models and utility scales before routing.
### Notes & Caveats
- Existing frozen router artifacts are unchanged. The new backend trains on the current profile, not automatically on all historical campaigns. ScienceWorld raw-score routing does not support derived episode-budget views; it defaults to no LimeQO comparison. Existing campaigns with a different recorded LimeQO setting require explicit compatibility handling. Historical diagnostic metrics do not establish the accuracy of newly trained routers.

## [DeltaMem / G-Memory recovery queue] - 2026-11-02
### Features
- Resume remaining TRAIN construction and all three datasets' TEST evaluations in durable systemd services, checking every 30 minutes and retrying failed workers after 30 minutes.
### Design Rationale
- DeltaMem resumes committed checkpoints; G-Memory rebuilds interrupted stores in fresh directories because its adapter has no safe insertion cursor. Completed stores and evaluated TEST pairs are reused.
- Hold the shared evaluation lease across construction and evaluation; worker lifetimes are bound to the coordinator to prevent orphaned work from overlapping a later measurement.
### Notes & Caveats
- Historical paused processes have exited. Their artifacts and pause records remain preserved. Existing frozen experiment sources and the Matérn patch remain unchanged.

