# TRACE space v0.7.0 — implementation report (2026-11-05)

Plan: `docs/superpowers/plans/2026-11-05-trace-space-v070.md` (incl. Amendment A). Spec addendum:
`docs/trace_space_v0.7.0.md`. CHANGELOG entry `[TRACE space v0.7.0] - 2026-11-05`.
No live run was started. No GPU was used. Nothing was committed. No subagent was spawned.

## 1. Files created

Modules (all additive, each <= 256 lines):

| File | Lines | Role |
|---|---|---|
| `trimem/experiments/plan_execution_adapter_v070.py` | 79 | `compile_config` / `runner_config`: rank none -> VGR / P1 / P2 |
| `trimem/experiments/trace_v070_samples.py` | 137 | per-configuration stratified samples (quota, seed, interleave) |
| `trimem/experiments/trace_v070_ledger.py` | 136 | `SampledLedger` (own sample per reservation, `samples` table), `samples.json` |
| `trimem/experiments/trace_v070_baseline.py` | 199 | NoMemory: `BaselineSpace`, prepare / run / import / entry |
| `trimem/experiments/trace_v070_runtime.py` | 219 | `v070_patches`, `space_patches`, `validate_manifest_v070`, `latent_search_worker_v070` |
| `trimem/experiments/trace_v070_pruning.py` | 119 | `staged_evaluate_v070` (own prefixes, executor-plan guard, optional NoMemory comparator) |
| `trimem/experiments/trace_v070_candidates.py` | 110 | candidates (BO U NoMemory, own tasks), Pareto, HV / plot export CLI |
| `tools/trace_v070_nomemory.py` | 202 | `evaluate_nomemory`, receipt builder, command / env, claim recovery |
| `tools/trace_v070_nomemory_worker.py` | 29 | official AgentGym evaluator + strict TRACE patches |
| `tools/trace_v070_holdout.py` | 170 | `run_pruned_holdout_v070` (Pareto U NoMemory; `--now`, `--select-only`) |
| `tools/latent_latency_pruner_v070.py` | 119 | latency cap with the NoMemory front; per-dataset factors |
| `tools/trace_v070_router.py` | 186 | sparse per-configuration classifier router + OneConfig incl. NoMemory |
| `tools/trace_v070_live.py` | 148 | launcher (`--preflight-only`, `--baseline-only`) |
| `tools/prepare_trace_v070_campaign.py` | 249 | `shared-init` / `arm` (explicit-pr refused) |
| `tools/latent_readiness_v070.py` | 67 | readiness R1-R6 in the v0.7.0 space, family incl. rank |
| `tools/trace_v070_hash_guard.py` | 85 | hash guard (CLI, used by test / smoke / queue) |
| `tools/trace_v070_smoke.py` | 256 | CPU synthetic smoke |

Already present and not rewritten: `trimem/optimizer/plan_schema_v070.py`, `tools/train_trace_seq_vae_v070.py`.

Tests (`tests/v070_fixtures.py` + 13 files): `test_v070_hash_guard` (T11), `test_v070_schema` (T1),
`test_v070_runner_config` (T2), `test_v070_runtime` (T3), `test_v070_nomemory` (T4), `test_v070_baseline` (T5),
`test_v070_pruning` (T6), `test_v070_latency_pruner` (T7), `test_v070_holdout` (T8), `test_v070_prepare` (T9),
`test_v070_vae` (T10), `test_v070_samples` (per-config samples), `test_v070_live` (launcher + router).
T12 = `tools/trace_v070_smoke.py` (run directly; section 3).

Run artifacts:
- `results/trace_v070_alfworld_init_20261105` — prepared for real (see section 4).
- `results/trace_v070_smoke_20261105` — smoke output.
- `results/trace_v070_e2e_20261105/run_v070_queue.sh` — queue (not started);
  `hash_guard_before.json` — snapshot of all 128 hashed files taken before any v0.7.0 file was written.
- Docs: `docs/trace_space_v0.7.0.md`, this report, CHANGELOG entry.

## 2. Test results

| Suite | Command | Result |
|---|---|---|
| v0.7.0 tests (T1-T11 + samples + launcher / router) | `.venv-optimizer/bin/python -m pytest -q tests/test_v070_*.py` | **74 passed** (35.5 s) |
| Regression (plan section 11) | `pytest -q tests/test_trace_latent_*.py tests/test_trace_joint_*.py tests/test_staged_pruning.py tests/test_trace_pr_staged_pruning.py` | **334 passed** (28 min 47 s, CPU) |
| Hash guard before | `results/trace_v070_e2e_20261105/hash_guard_before.json` (snapshot, 128 files) | 5 guarded manifests: 0 mismatches |
| Hash guard after | `python -m tools.trace_v070_hash_guard` -> `hash_guard_after_implementation.json` | `ok: true` (0 mismatches, no new drift, no hashed file imports a v0.7.0 module) |
| Lint | `ruff check` on all new files | clean (one pre-existing F401 in `tools/train_trace_seq_vae_v070.py`) |

Per-test-file counts: hash_guard 10, schema 14, runner_config 6, runtime 6, nomemory 4, baseline 4, pruning 3,
latency_pruner 3, holdout 5, prepare 3, vae 5, samples 6, live 5 (total 74).

Per-configuration sample tests (Amendment A): two configurations get different samples, both exactly on the TEST
mix quota; samples are deterministic in (seed, key, pool); every 20-task prefix follows the mix (deviation <= 1;
measured max 0.6 on the real ALF / Sci / WebShop pools over 200 keys each); the ledger reserves each
configuration's own tasks and records them (table + `samples.json`); observations equal the stock aggregate over
own tasks and share one panel identity; staged pruning runs own prefixes (T6); the latency pruner, candidates,
holdout and router use own rows (T7, T8, launcher / router tests). T9 runs the real hashed `base.prepare`
offline (memory identity stubbed) for ALFWorld: shared init -> completion with identity-checked non-synthetic
receipts -> arm import (10 init + NoMemory, budget 80, 5,000 episodes, sizes 20..100).

## 3. Smoke (T12) — `results/trace_v070_smoke_20261105`

Command: `.venv-optimizer/bin/python -m tools.trace_v070_smoke --out results/trace_v070_smoke_20261105 --new 8
--stage-step 20` (CPU, 48.8 s). Tiny v0.7.0 space (rank ColBERT | none), 600-task synthetic pool, 100-task own
samples, tiny CPU VAE v2 (d 4) + provisional affine, real `guarded_search` spawn.

| Check | Result |
|---|---|
| space_v070 | true |
| rank_none_present | true |
| no_llm_rank | true |
| nomemory_train_complete | true |
| nomemory_not_in_main_ledger | true |
| nomemory_never_pruned | true |
| nomemory_test_candidate | true |
| test_summary_has_nomemory | true |
| v070_worker_used | true (3 `round_*.v070.json` markers, every patch flag true) |
| hashed_files_unchanged | true |
| per_config_samples_distinct | true |
| samples_follow_test_mix | true |
| prefixes_stratified | true (max deviation <= 1 task per type at 20..100) |
| staged_checkpoint_logged | true (32 decisions, 2 pruned: at 20 and at 60 tasks) |
| bo_rounds_ge_2 | true (3 BO rounds: 4 + 4 + 1 configurations) |

TEST: 4 BO configurations + NoMemory (in the TRAIN Pareto set, tested), 1,000 TEST episodes; selection:
18 candidates (17 full BO + NoMemory), 2 pruned excluded. `latent_latency_pruner_v070` ran once (no
partial configuration left to judge at that point; its decision rule is covered by T7).

## 4. Real ALFWorld preparation and preflight

`python -m tools.prepare_trace_v070_campaign shared-init --dataset alfworld --template
results/trace_pr_three_dataset_prepare/template_v060.json --run-dir results/trace_v070_alfworld_init_20261105
--memory-db trace_ablation_sci_48d26f8d18eba20f --seed 20261022` (read-only Postgres access; no ANALYZE).

- Protocol `trace_v070_alfworld_init_20261105-v1`, `fixed.protocol_version` `trace-v070-rank-optional-nomemory`,
  96 hashed artifacts (incl. every v0.7.0 module), search backend `shared-init-v1`, TEST deferred, budget 10.
- Pool 3,792 eligible TRAIN tasks; quotas look 9 / pick_and_place 25 / clean 18 / cool 14 / heat 12 / two 22
  (equal to the 2026-11-04 stratified panel). 5 Sobol configurations reserved, each with its own 100 tasks
  (500 episodes). NoMemory key `ed97cd92…`, own 100-task sample, 0 / 100 done.
- `python -m tools.trace_v070_live --run-dir results/trace_v070_alfworld_init_20261105 --preflight-only`:
  `status passed` (services up: both vLLM bases list the model, env servers, rank bases, embedding);
  `vae_training_running false` (the VAE job finished 11:31, d16 and d32 rc 0); GPU utilization at that moment
  99 / 100 % (another job uses vLLM; the launcher's resource gate waits for idle GPUs).

Readiness pre-check (CPU, informational; `results/trace_v070_e2e_20261105/readiness_precheck_alfworld_d16_d32.json`,
`python -m tools.latent_readiness_v070 run --space-run results/trace_v070_alfworld_init_20261105`):

| Checkpoint | R1 | R2 (family filter_order / graph_impl / rank_impl) | R3 | R4 | R5 | passes | seconds |
|---|---|---|---|---|---|---|---|
| d16 | 1.000 | 1.000 | 1.000 | 0 | 1.000 | yes | 937 |
| d32 | 1.000 | 1.000 | 1.000 | 0 | 1.000 | yes | 555 |

The queue writes its own `readiness_<ds>.json` for the chosen checkpoint (the provisional-affine cache of this
pre-check in `results/trace_v070_e2e_20261105` is reused).

## 5. Deviations from the plan (with reasons)

1. **Manifest `tasks` = the eligible TRAIN pool** (not a 100-task panel). Reason: per-configuration samples
   (Amendment A.4) need the pool for `evaluate`'s panel check and for the worker's `--panel` file
   (`train_panel.json` now lists the pool with strata). Consequence: the stock validator alone refuses a live
   v0.7.0 manifest (fail closed); `validate_manifest_v070` runs the stock checks on a 100-task view plus the v0.7.0
   checks. The pruning block's `order` is a description string (own prefixes), not one frozen order.
2. **Extra driver patches** beyond plan section 5: `aggregate_panel` (own-task observations, one shared panel
   identity), `SearchLedger` -> `SampledLedger`, `pruning_budget` (full cost = sample size), `validate_manifest`.
   Reason: Amendment A.4. All are restored on exit and re-entered by the spawned worker.
3. **Router**: `tools/trace_v070_router.py` does not wrap `trace_pr_router_curve` (dense kNN needs one shared
   panel). It reimplements the classifier router's `fit` / `route` on per-configuration own samples (honest OOF on
   own tasks, plain model prediction on the other union tasks) and reuses its `report` with TEST outcomes unioned
   with the NoMemory TEST ledger. Not exercised on live data.
4. **`--preflight-only`** does not refuse while VAE training runs (it launches nothing); it reports
   `vae_training_running`. A real run still calls `check_no_vae_training`.
5. **Readiness inside the queue** runs on the prepared init space (needs a v0.7.0 run manifest), so its order is
   init -> readiness R1-R5 -> freeze-affine (R6) -> arm, instead of readiness before the init.
6. **Queue** script name `run_v070_queue.sh` (per the task) and the latency pruner runs as a child process of the
   queue (stopped by PID), not as a separate unit.
7. Hash guard: the three v0.6.0c manifests (2026-11-03c) already had 3 mismatching files each
   (`plan_surrogate.py`, `joint_search_ledger.py`, `trace_joint_plan_search.py`) BEFORE this work; the guard
   checks them against the pre-work snapshot (no new drift) instead of their manifests.
8. Not done: the `adversarial-review` step of plan section 12 (it spawns reviewer agents; this task forbade
   subagents). Recommended before starting the queue.

## 6. Start the queue (NOT started)

```bash
systemd-run --user --unit=trace-v070-e2e-20261105 --slice=trimem-eval.slice --collect \
  --setenv=PATH=/local-scratch/localhome/hza214/local/node-v22.14.0-linux-x64/bin:/usr/local/bin:/usr/bin:/bin \
  -p WorkingDirectory=/localhome/hza214/tridb \
  /localhome/hza214/tridb/results/trace_v070_e2e_20261105/run_v070_queue.sh
```
Options: `DATASETS="alfworld"` (subset), `CK=results/trace_v070_seqvae_20261105/seqvae_v2_d32.pt` (d32), passed with
`--setenv=`. Progress: `results/trace_v070_e2e_20261105/queue.log`, `<run>/status.json`.

## 7. Open risks

1. NoMemory has never run a real episode through `tools/trace_v070_nomemory_worker.py`; the first real episodes
   are in the ALFWorld init stage (`--baseline-only` first is a cheap live check). ALFWorld pool items with server
   `appended` (1,372 of 3,792) rely on the managed env server serving them by index for the official evaluator, as
   in the historical pool runs.
2. The latency-cap reference includes NoMemory (Q2): with NoMemory much faster than retrieval configurations, a
   factor 2.0 cap may prune most BO configurations at 20 tasks on ALFWorld / WebShop. Check
   `latency_prune_log.jsonl` after the first arm rounds.
3. Rank-none configurations have not run live: first check `memory/<idx>.json` (`rank_impl` none, selected by
   descending cosine) and `worker.log` (`--plan VGR|P1|P2`).
4. The manifest is ~4k tasks + strata; every ledger open hashes it (a few ms) — negligible next to episodes.
5. Process-global patches: a code path that builds a stock `JointSpace` or reads the pool as a panel fails
   closed (ValueError) but can stop a run late.
6. Editing any v0.7.0 module after preparation changes its hash; the prepared ALFWorld init then fails
   `verify_artifacts` and must be re-prepared (it has no episodes yet, so this is safe now).
7. `tools/train_trace_seq_vae_v070.py` has an unused import (`evaluate_v2`, ruff F401); left unchanged (pre-existing,
   was running). New files follow the repository's 120-column TRACE style (not `ruff format`-ed).

## 8. Fix round 1 (adversarial review verdict `review_v070/verdict.md` + user decision 2026-11-05)

Sections 1–7 above describe round 0. Where they conflict with this section, this section holds (e.g. the queue
no longer starts a pruner process; the router candidate set changed).

| # | Fix | Files | Tests |
|---|---|---|---|
| 1 | Latency cap moved INTO `staged_evaluate_v070`: same stage barrier, exact own n-prefix, front = sample-mean Pareto front of complete never-pruned BO configurations ∪ complete NoMemory TRAIN, `cap = factor × fastest front latency`, factor from `pruning.latency_cap` (ALF 2.0 / Sci 3.0 / WebShop 2.0, enforced for live manifests), init protected, NoMemory never judged, logged with the staged decisions (`prune_rule` pareto / latency-cap). The separate pruner process is removed from the queue; `latent_latency_pruner_v070` is an offline dry-run reporter only (refuses `dry=False`, writes `latency_prune_report.jsonl`). | `trimem/experiments/trace_v070_pruning.py`, `tools/prepare_trace_v070_campaign.py` (`pruning_block`), `tools/latent_latency_pruner_v070.py`, `results/trace_v070_e2e_20261105/run_v070_queue.sh` | `test_v070_pruning::test_latency_cap_at_the_stage_barrier`; `test_v070_latency_pruner` (reporter never prunes, non-dry refused); smoke `latency_cap_at_stage_barrier` (19 latency-cap prunes) |
| 2 | **User decision (supersedes verdict fix 2):** router + OneConfig candidates = BRANE fuzzy Pareto set on TRAIN (`brane.fuzzy_pareto`, TAU_ACC 0.02, TAU_COST 0.10; acc = own-sample mean quality, Sci `max(score,0)/100`; cost = `0.5 ln mean tokens + 0.5 ln mean latency_ms`) over complete never-pruned BO ∪ NoMemory. Automatic TEST set = 3-objective Pareto ∪ fuzzy set ∪ NoMemory; `selection.json` records `pareto_keys`, `fuzzy_keys`, `router_candidate_keys`, per-candidate `brane` acc / cost, NoMemory `in_pareto` / `in_fuzzy` / `tested`. The router refuses (fit, route, report) if any candidate lacks complete TEST receipts. | `trimem/experiments/trace_v070_candidates.py` (`brane_acc_cost`, `fuzzy_set`, export flag), `tools/trace_v070_holdout.py`, `tools/trace_v070_router.py` (`router_candidates`) | `test_v070_holdout::test_fuzzy_retained_config_is_tested_and_router_candidate` (a fuzzy-retained, 3-objective-dominated config is TESTed and is a router candidate; a fuzzy-pruned config is neither; refusal on a candidate without TEST receipts); smoke `test_set_has_pareto_and_fuzzy` |
| 3 | Commit first, then log: one ledger transaction records every decision of a checkpoint (`stage_decisions` table, key × stage) and applies the prunes; `pruning_log.jsonl` is appended afterwards and missing lines are back-filled from the ledger. On resume a recorded (key, stage) decision is never recomputed; with the async pruner gone, decisions are a pure function of the ledger at the barrier. | `trimem/experiments/trace_v070_ledger.py` (`apply_stage_decisions`, `stage_decisions`), `trimem/experiments/trace_v070_pruning.py` | `test_v070_pruning::test_decisions_commit_first_and_are_not_recomputed` (crash after commit, before log -> resume does not call `decide_checkpoint`, log back-filled); smoke `decisions_durable` |
| 4 | `prepared.json` completion marker written LAST (after manifest, ledgers, imports, `samples.json`, post-checks); the queue checks the marker; the launcher refuses a live run without it (or with a changed manifest); the arm refuses an unprepared shared init; an interrupted preparation (no marker, no evaluation evidence) is set aside as `<dir>.incomplete-<epoch>` and replayed; with evaluation evidence it is refused. | `trimem/experiments/trace_v070_marker.py` (new, 81 lines), `tools/prepare_trace_v070_campaign.py`, `tools/trace_v070_live.py`, queue script | `test_v070_prepare::test_prepared_marker_and_replay`, `::test_arm_refuses_unprepared_shared_init`; arm test asserts `prepared.json` |
| 5 | Hash guard fails closed: a missing protected manifest, the missing pre-work snapshot or the missing discovery baseline is a violation. Protected runs = discovery (every `results/*/manifest.json` with non-empty `live.artifacts` that is not v0.7.0: 36 runs, 202 files) ∪ the explicit floor. Discovered runs: each file matches its manifest or its frozen pre-existing value (`hash_guard_discovered_baseline.json`, frozen once; evidence: 0 of 202 hashed files modified after the work start 11:20); new runs must match exactly. | `tools/trace_v070_hash_guard.py` (`--freeze-baseline`) | `test_v070_hash_guard` (13: fail closed without snapshot / baseline / guarded manifest; drift rules; discovery skips v0.7.0 runs) |
| 6 | Runtime parity enforced at the manifest boundary (live): `task_concurrency` 16, `fixed.concurrency` 16, agent `max_tokens` 4096 (live, fixed, runtime), paired LLM / env slots, latency-cap factor of the dataset, `latency_front` true (all manifests). NoMemory: `parity_violations` (model, LLM / env / rank bases, env vars, max_round, action format, max_tokens, concurrency, fixed runtime) checked by `run_baseline` and by `--preflight-only` (`nomemory_parity: ok`). | `trimem/experiments/trace_v070_runtime.py` (`validate_parity`), `trimem/experiments/trace_v070_baseline.py`, `tools/trace_v070_live.py` | `test_v070_prepare::test_runtime_parity_enforced` (on the real prepared ALF manifest) |
| 7 | `holdout/driver.lock` is taken before TRAIN checks, selection and any holdout write. | `tools/trace_v070_holdout.py` | `test_v070_holdout::test_selection_waits_for_the_holdout_lock` |
| 8 | Only the `queue.lock` owner writes `status.json`; a rejected launcher exits (SystemExit) without touching it. | `tools/trace_v070_live.py` | `test_v070_live::test_non_owner_never_writes_status` |

Verification after fix round 1:
- `pytest -q tests/test_v070_*.py`: **85 passed** (hash_guard 13, schema 14, runner_config 6, runtime 6, nomemory 4,
  baseline 4, pruning 5, latency_pruner 3, holdout 7, prepare 6, vae 5, samples 6, live 6).
- Hash guard: `ok: true`, 0 violations (`results/trace_v070_e2e_20261105/hash_guard_after_fix_round1.json`).
- `ruff check` on all v0.7.0 files: clean (except the pre-existing F401 / import order in
  `tools/train_trace_seq_vae_v070.py`, which I left as delivered; an accidental `ruff --fix` on it was reverted).
- CPU smoke `results/trace_v070_smoke_20261105` (round-0 output kept as `..._round0`): **18 / 18 checks true**,
  8 BO rounds, 118.6 s; 35 stage decisions (19 latency-cap, 4 Pareto, 12 kept); TEST set 3 BO + NoMemory
  (3-objective Pareto 4 keys, fuzzy 4 keys, NoMemory in both); 23 pruned configurations excluded.
- ALFWorld init re-prepared (the round-0 preparation had no episodes and no marker, so the prepare command set it
  aside as `results/trace_v070_alfworld_init_20261105.incomplete-1793910976`; it can be deleted). New preparation:
  97 hashed artifacts, `prepared.json` present, same quotas and NoMemory key `ed97cd92…`.
  `--preflight-only`: `status passed`, `nomemory_parity ok`, no `status.json` written.
- The regression suites of plan section 11 were not rerun after fix round 1 (no hashed or stock module changed;
  round 0: 334 passed).

Start command (unchanged; queue NOT started): see section 6.

Open risks added by fix round 1:
1. In the smoke, the latency cap pruned 19 of 23 pruned configurations at 20 tasks. With NoMemory as the fastest
   front member, factor 2.0 may prune most retrieval configurations on ALFWorld / WebShop live; this follows the
   user's rule (Q2) but deserves a look after the first arm rounds (`pruning_log.jsonl`, `prune_rule`).
2. The latency cap is not subject to the executor-plan guard (a plan can lose all its configurations to the cap).
3. The BRANE fuzzy set may enlarge the TEST set beyond the 3-objective Pareto set (more TEST episodes).

## 9. Fix round 2 (scoped re-review `review_v070/rereview_round1.md` + user decisions 2026-11-05)

Sections 1–8 describe earlier rounds; where they conflict with this section, this section holds. Re-review item 2
("NOT ADDRESSED") was a false positive on the fuzzy rule: the BRANE fuzzy design is kept; only its NoMemory part
is fixed (A).

| # | Fix | Files | Tests |
|---|---|---|---|
| A | NoMemory is ALWAYS a router and OneConfig candidate, even when BRANE fuzzy pruning drops it: `selection.json["router_candidates"]` = fuzzy set ∪ {NoMemory} (`router_candidate_keys` kept as the same list). The router refuses a selection whose candidates lack NoMemory or any complete TEST receipt. | `tools/trace_v070_holdout.py`, `tools/trace_v070_router.py` | `test_v070_holdout::test_nomemory_forced_when_dominated` (NoMemory not in fuzzy, still a router candidate), `::test_fuzzy_retained_config_is_tested_and_router_candidate` (refuses without NoMemory); smoke `test_set_has_pareto_and_fuzzy` |
| B | Hash guard: guarded / older-snapshot / discovered checks aggregated independently (violations prefixed by check); any violation from any check fails the guard. | `tools/trace_v070_hash_guard.py` | `test_v070_hash_guard::test_guarded_mismatch_is_not_hidden_by_the_discovered_baseline` |
| C | The failure status (`stopped`) is written inside the `with own` block, while `queue.lock` is still held. | `tools/trace_v070_live.py` | `test_v070_live::test_failure_status_written_while_holding_the_lock` (a probe flock at the `stopped` write fails: lock held) |
| D | Committed stage decisions are back-filled into `pruning_log.jsonl` at launcher startup (after taking `queue.lock`) and before every `run_round` (`backfill_from_ledger`), not only inside staged evaluation. | `trimem/experiments/trace_v070_pruning.py`, `tools/trace_v070_live.py` | `test_v070_pruning::test_decisions_commit_first_and_are_not_recomputed` (crash between commit and log, all batch configurations pruned, nothing pending -> the launcher back-fill writes the line, idempotent, no recomputation); `test_v070_live::test_round_loop_backfills_committed_decisions_first` |
| E | NoMemory runtime parity is enforced also when the baseline ledger is already complete (launcher `baseline_stage`) and on import (source baseline manifest vs destination manifest: model, LLM / env / rank bases, env vars, max_round, action format, max_tokens, concurrency, fixed runtime identity, and every source receipt's runtime); mismatch refuses before any receipt is restated. | `trimem/experiments/trace_v070_baseline.py`, `tools/trace_v070_live.py` | `test_v070_baseline::test_parity_refused_on_import_and_on_a_complete_ledger` |
| F | **User decision:** the latency-cap front EXCLUDES NoMemory: front = sample-mean Pareto front of the complete never-pruned BO (retrieval) configurations only; cap = factor × fastest latency on that front (ALF 2.0, Sci 3.0, WebShop 2.0). Manifest `baselines.NoMemory.latency_front` = false, required by `validate_v070_manifest`. NoMemory stays a Pareto / TEST / router / OneConfig candidate and is never pruned. The offline reporter keeps a what-if `baseline_front` option. | `trimem/experiments/trace_v070_pruning.py`, `trimem/experiments/trace_v070_baseline.py`, `trimem/experiments/trace_v070_runtime.py`, `tools/latent_latency_pruner_v070.py` | `test_v070_pruning::test_latency_cap_at_the_stage_barrier` (NoMemory at 1 s is ignored: cap = 2 × 8 s of the BO front; NoMemory not in the recorded front); `test_v070_prepare::test_runtime_parity_enforced` (`latency_front` true refused) |
| G | **User decision:** the latency cap respects the executor-plan guard: it never prunes the last remaining never-pruned configuration of an executor plan (6 plans); if all survivors of a plan would leave at this checkpoint, the latency-capped one with the lowest prefix latency is kept with `guarded: "plan-last"` (top level and in `latency_cap`). | `trimem/experiments/trace_v070_pruning.py` (`_latency_plan_guard`) | `test_v070_pruning::test_latency_cap_keeps_the_last_config_of_an_executor_plan` |

Verification after fix round 2:
- `pytest -q tests/test_v070_*.py`: **90 passed** (hash_guard 14, schema 14, runner_config 6, runtime 6, nomemory 4,
  baseline 5, pruning 6, latency_pruner 3, holdout 7, prepare 6, vae 5, samples 6, live 8).
- Hash guard: `ok: true`, 0 violations (`results/trace_v070_e2e_20261105/hash_guard_after_fix_round2.json`).
- `ruff check` on all v0.7.0 files (except the delivered VAE script): clean.
- CPU smoke `results/trace_v070_smoke_20261105` (earlier outputs kept as `..._round0`, `..._round1`): **18 / 18
  checks true**, 3 BO rounds, 53 s; 33 stage decisions (2 Pareto, 1 latency-cap, 30 kept), NoMemory never in a
  latency front; TEST set 3 BO + NoMemory; router candidates 4 (fuzzy set incl. NoMemory). The smoke landscape now
  gives beam width 4 a larger latency (12 s per beam unit, was 7 s), so the cap still fires with the NoMemory-free
  front.
- ALFWorld init re-prepared: the fix-round-1 preparation (marker present, no evaluation evidence) was moved by hand
  to `results/trace_v070_alfworld_init_20261105.superseded-fixround1`; the new preparation has `prepared.json`,
  97 hashed artifacts, `latency_front` false, same quotas and NoMemory key `ed97cd92…`. `--preflight-only`:
  `status passed`, `nomemory_parity ok`, no `status.json` written.
- The plan section 11 regression suites were not rerun (no hashed or stock module changed; round 0: 334 passed).

Notes:
1. With NoMemory excluded and the plan guard on, the latency cap prunes much less than in fix round 1 (smoke: 1
   latency prune vs 19). It fires only for configurations slower than factor × the fastest BO front member.
2. Leftover directories that can be deleted (no episodes): `results/trace_v070_alfworld_init_20261105.incomplete-1793910976`,
   `results/trace_v070_alfworld_init_20261105.superseded-fixround1`, `results/trace_v070_smoke_20261105_round0`,
   `results/trace_v070_smoke_20261105_round1`.

## 10. SemFilter v0.7.1 + stall rule (user decisions 2026-11-05)

Nothing was started. No hashed file changed; `run_v070_queue.sh` is untouched. Decisions as finally given: q stays 4,
stall patience 3, a forced round has 4 exploration picks. The hold has fired: the ALF init ended with `test_deferred`
and `trace-v070-e2e-20261105` is inactive.

### Measurement (offline, Qwen3.8 tokenizer; `results/trace_v070_e2e_20261105/semfilter_v071_card_tokens_*.json`)

| Judge cards | Sample | Raw tokens / card | Compact tokens / card | Reduction |
|---|---|---|---|---|
| ALFWorld, candidates the init actually judged | 28,689 judgements, 5,614 cards | 301.7 | 253.1 | **16.1%** |
| ALFWorld, uniform sample of the clone | 400 | 295.6 | 249.6 | 15.6% |
| ScienceWorld, uniform sample | 400 | 1,088.8 | 429.3 | **60.6%** |
| WebShop, uniform sample | 400 | 315.1 | 197.3 | **37.4%** |

ALFWorld saves least because its observations are mostly under 60 characters. Replay of the ALF init receipts (800
SemFilter episodes, read-only `opt.session_retrieval`), candidates judged:

| Policy | Candidates judged | vs stock |
|---|---|---|
| Stock (TOPN 32, no reuse) | 28,689 | 100% |
| TOPN 16 | 18,691 | 65.2% |
| TOPN 16 + periodic reuse | 14,822 | **51.7%** (periodic hit rate 52.8%: 3,869 of 7,324) |

Estimate, not a live measurement: ALF judge prompt tokens about **54% lower** (0.517 candidates x 0.882 tokens per
candidate; 412 tokens per candidate today, of which about 110 are shared prompt). Stock SemFilter episodes spend
15,095 semantic tokens on average, 40.2% of them in periodic retrievals.

### Files

New: `trimem/experiments/semfilter_v071.py`, `trimem/experiments/trace_v071_runtime.py`,
`trimem/experiments/trace_v071_search.py`, `tools/trace_v071_episode_worker.py`, `tools/trace_v071_live.py`,
`tools/prepare_trace_v071_campaign.py`, `tools/trace_v071_hash_guard.py`, `tools/trace_v071_smoke.py`,
`tools/semfilter_v071_measure.py`, `results/trace_v070_e2e_20261105/run_v071_queue.sh`, `tests/test_v071_semfilter.py`
(13), `tests/test_v071_stall.py` (8), `tests/test_v071_prepare.py` (9).

| Decision | Implementation | Test |
|---|---|---|
| TOPN 16 | stock knob in `live.environment`, recorded in `fixed.semfilter` (fingerprint changes for v2c inits) | `test_topn_16_judged_count`, `test_shared_init_v2c_manifest` |
| Compact card | `compact_card`: tag + outcome, goal, all actions, observation 60 chars, no-ops dropped / collapsed, cap 2,000 | `test_compact_card_*` |
| Periodic reuse | episode cache around `sem.judge`; `cache_hits`, `judged_new`, `tokens_saved_est` in the receipt and `attempt/semfilter_v2c.jsonl` | `test_reuse_across_periodic_retrievals`, `test_reuse_key_has_rubric_and_reuse_can_be_off` |
| Versioning | `v071` block; `--semfilter v2c` default; `--semfilter v2` = the v0.7.0 command; launcher prints the protocol | `test_semfilter_v2_is_the_v070_command`, `test_stock_path_unchanged_and_fail_closed`, `test_worker_command_rewritten_only_for_v071` |
| ALF mid-run switch | `--old-protocol-init`: `v071.init_protocol`, same space as the init, `fixed` differs by `semfilter` only, receipts tagged and passed through the unchanged `check_receipt`; `semfilter_protocols.json` | `test_midrun_*`; smoke `exemption_recorded`, `init_receipts_old_protocol`, `protocol_table_separates` |
| Stall rule | `stall_state` (pure), `stagnation_status` patience 3, epsilon 1e-6; round file fields; `hv_rounds.jsonl` | `test_counter_improve_stall_forced_reset`, `test_pruned_round_counts_as_stalled_and_is_pure`; smoke `replay_gives_same_flags` |
| Forced picks | 4 sequential `uncertainty-novelty-v1` picks, no acquisition call | `test_forced_picks_sequential_deterministic_differs_from_greedy`, `test_forced_tie_breaks_novelty_then_key`, `test_conditioning_lowers_a_neighbour_of_an_earlier_pick` |

### Verification

- `pytest tests/test_v070_*.py tests/test_v071_*.py`: **122 passed**. Plan section 11 regression suites not rerun (no
  stock module changed; round 0: 334 passed).
- v0.7.1 smoke `results/trace_v071_smoke_20261105`: **14 / 14 checks true**, 12 rounds, 196.9 s. HV stalled for 3 rounds,
  `round_042` was forced, the counter reset, and picks 2-4 used `conditioning: schur`. Mean proposal time: forced 1.6 s,
  ordinary 10.7 s (tiny space; not a live estimate).
- v0.7.0 smoke (scratchpad): 18 / 18 true. Hash guards v0.7.0 and v0.7.1: `ok` (`hash_guard_v070_after_v071.json`,
  `hash_guard_v071_after_v071.json`).
- `.venv-agentgym` (Python 3.10): the wrapper installs and `agentgym_session_eval` then uses the v2c settings, build and
  view (import only; no episode ran).
- Read-only check against the real ALF init: memory identity, runtime identity, regenerated panel sha and sampling
  block all equal the init's, so the real mid-run arm passes the exemption check. Its ledger was not opened.

### Start

(a) Normal case: ALFWorld mid-run, then ScienceWorld and WebShop on v0.7.1 from their init. This is the default. The
script refuses to start while `trace-v070-e2e-20261105` is active.

```bash
systemd-run --user --unit=trace-v071-e2e-20261105 --slice=trimem-eval.slice --collect \
  --setenv=PATH=/local-scratch/localhome/hza214/local/node-v22.14.0-linux-x64/bin:/usr/local/bin:/usr/bin:/bin \
  -p WorkingDirectory=/localhome/hza214/tridb \
  /localhome/hza214/tridb/results/trace_v070_e2e_20261105/run_v071_queue.sh
```

Only the remaining datasets: add `--setenv=DATASETS="sciworld webshop"`.

(b) A restarted ALFWorld (a fresh init on v0.7.1, not the mid-run path): add `--setenv=ALF_RESTART=1
--setenv=DATASETS=alfworld`. It uses `results/trace_v071_alfworld_restart_init_20261105` and
`results/trace_v071_latent_alfworld_restart_20261105`, and never reuses the mid-run arm's files.

Run directories: `results/trace_v071_latent_alfworld_20261105`, `results/trace_v071_{sciworld,webshop}_init_20261105`,
`results/trace_v071_latent_{sciworld,webshop}_20261105`. Progress: `results/trace_v070_e2e_20261105/queue_v071.log`.
Use `tools.trace_v071_live`, never `tools.trace_v070_live`, on these runs.

### Deviations

1. `prepare_trace_v070_campaign` and `validate_manifest_v070` are hashed by the ALF init and could not change. The
   option is in the new wrapper. `validate_manifest_v070` already accepts both forms (unknown keys pass);
   `validate_v071_manifest` checks the block and `semfilter_mode` reports which protocol a manifest runs.
2. `snapshot.with_pending` does not change posterior variance. Each pick's snapshot comes from it as specified, and
   the log-volume is also conditioned on earlier picks through the GP joint covariance. Without this, "sequential"
   would equal the top 4 by marginal variance.
3. The forced pool omits latent-refined candidates, because producing them calls the acquisition. It holds Sobol, LLM,
   recombination and anchors. The analyzer LLM call still runs in a forced round.
4. The compact label drops the advice sentence ("do not repeat it ..."); the rubric's FAILED-ATTEMPT clause states it.
   This goes beyond the suggested design and gives about 2 of ALFWorld's 16 percentage points.

### Open risks

1. Compact cards and reused scores are not validated against live judge decisions, so quality may shift. Reused scores
   were judged against an earlier target of the same episode, while periodic retrievals re-target on recent state.
2. ALFWorld mixes protocols: init configurations carry TOPN 32 costs and exploration configurations carry v2c costs,
   in one GP and one frozen objective. The switch alone can raise HV in early rounds, which delays the stall counter
   and favours exploration SemFilter configurations over init ones. Separate them with `semfilter_protocol` /
   `semfilter_protocols.json`; the execution fingerprint does not separate them on ALFWorld.
3. No real v2c episode has run. The first ones are in the first ALFWorld arm round. Check
   `<episode>/attempt/semfilter_v2c.jsonl` and `receipt.semfilter` there.
4. `fixed.prefilter` keeps its template text ("top_n=32"), because the exemption requires equality on ALFWorld.
   `fixed.semfilter` is authoritative.
