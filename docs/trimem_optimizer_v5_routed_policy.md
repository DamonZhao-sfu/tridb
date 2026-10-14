# TriMEM optimizer v5, Branch B: per-round routed policy

Status: implemented and unit/private-PG tested in the v5 stage. No run config exists yet,
because a router needs Branch A profiling data first.

## What changes at run time

`evaluation.application_mode = "per_round_routed_policy"` (the default
`first_decision_then_frozen_policy` is unchanged).

- The trial candidate is a `RoutedPolicy`: a router artifact plus a fixed λ.
  `action_hash = sha256({"kind": "per_round_routed_policy", "router_artifact_sha256", "lambda"})`.
- `ContinuationAdapter(retrieve_each_step=True)` calls `retrieve` before every round, not
  once before round 1.
- Each round: build state features → `policy.choose(features, cached)` → retrieve the
  chosen `CandidateSpec` on the branch's single private PG cluster. The cluster is started
  lazily once and reused, and the "retrieve only once" guard is relaxed only in this mode.
  No-memory still never starts PG.
- Memory is cached per `action_hash`, because the query is the frozen snapshot query. If a
  repeated action's cached context no longer fits the current prompt window (because
  `best_code` grew), it is re-retrieved and marked `cache_refit`.
- `history[i]["routing"]` records `action`, `action_hash`, `features`, the router `scores`
  for every catalog action (gain, generation_tokens, retrieval_tokens, cost, score),
  `cache_hit`, `cache_refit`, `execution_id`, `packed_ids`, the `receipt` (`null` on a cache
  hit), `retrieval_request_ids` and `retrieval_tokens`. Every objective ledger event belongs
  to exactly one round: it is either in that round's generation `request_ids` or in its
  `retrieval_request_ids`.
- Unit telemetry also has `policy` and `retrievals` (one entry per actual execution).
  `retrieval` is `null`.
- `application_mode` is part of the runtime config hash, so the routed protocol_id differs
  from Branch A's. The two are different protocols, so compare objectives, not ids.

## Router (`trimem/optimizer/router.py`)

The router follows BRANE: `a*(x) = argmax_a p̂_a(x) − λ·ĉ_a(x)`.

- **Samples**: one sample per (observed Branch A trial, unit, round). Only `completed` units
  with complete usage are used.
- **State features** (one function shared offline and online):
  - `round_index`: completed rounds.
  - `best_before`.
  - `gap_to_corpus_best`: direction × (best valid fitness of the task in the frozen
    snapshot − best_before).
  - `tokens_so_far`: the ledger total at decision time.
  - `last_valid` and `last_improved`.

  These are concatenated with `SearchSpace(catalog).encode(action)`.
- **Targets**:
  - `gain = direction × (best_fitness[i] − best_before)`.
  - `generation_tokens`: the input and output tokens of that round's generation requests.
- **Cost**: ĉ = predicted generation tokens + the measured mean retrieval tokens of the
  action (from round 1 of its profiling trials). The retrieval term is dropped when the
  action's memory is already cached in the branch. Retrieval cost is a measured table, not
  a learned target, because in frozen-mode data retrieval happens only in round 1, which
  would confound it with `round_index`.
- **Models**: `sklearn.ensemble.GradientBoostingRegressor` (sklearn 1.9.1 in
  `.venv-optimizer`) with 200 trees, depth 3, learning rate 0.05, `subsample=1` and a fixed
  `random_state`. CPU only and deterministic: the same data and seed give the same artifact
  sha.
- **Catalog**: the distinct profiled actions only. An unprofiled action (no_memory included)
  would be scored by pure extrapolation, and in a synthetic test it won spuriously, so it is
  excluded. Branch A reserves no_memory first, so it is normally present;
  `no_memory_in_catalog` records whether.
- **λ calibration**:
  - The grid is `scale × {0, .01, .03, .1, .3, 1, 3, 10, 30}`, where
    `scale = pstdev(gain) / mean(round cost)`.
  - For each held-out trial, models are trained on the other trials and route every logged
    state of the held-out trial.
  - A round's (gain, generation tokens) is the realized value when the router picks the
    logged action. Otherwise it is the held-out model's prediction (direct method).
  - Retrieval cost uses a branch cache simulated per (λ, unit) from the router's own
    choices, exactly as online: an action pays retrieval the first time it is chosen and is
    free afterwards. The logged action's first pick pays that unit's realized retrieval.
    With perfect predictions, the offline cost equals the online routed cost (tested).
  - λ maximizes the hypervolume of the mean per-trial (gain, −cost) point against
    (0, 1.1 × the largest mean cost on the grid). Ties go to the smaller λ.
  - The full table is stored in the artifact.

### Artifact (JSON)

`schema_version`, `kind`, `seed`, `sklearn_version`, `state_features`, `feature_names`,
`model` {estimator, params, `gain` and `generation_tokens` as `{pickle_b64, sha256}`,
target/cost definitions}, `catalog` [{action, action_hash, retrieval_tokens, profiled_units}],
`lambda` {method, is_offline_proxy, scale, multipliers, grid, reference, table, chosen,
caveats}, `no_memory_in_catalog`, `training_data` {archive, archive_manifest_sha256,
protocol_id, cohort_id, trial_ids, n_samples, snapshots, corpus_best_fitness,
quality_direction}, `artifact_sha256` (stable hash of all other fields).

Loading checks, in order:

1. The artifact sha256.
2. The optional expected sha.
3. Each model's pickle sha.
4. That the sklearn version is exactly equal. `scikit-learn==1.9.1` is pinned in
   `requirements-trimem-optimizer.txt`.
5. Catalog identities.
6. The feature layout.

Then it unpickles. Only load artifacts this code produced.

`training_data.deployment_invariants` stores:

- Protocol fields: model, prompt and tokenizer versions, token budget, quality kind and
  direction, horizon, max_steps, task weights and quality scales.
- The unique (task, checkpoint, snapshot sha, initial_fitness) units. Seeds are excluded.
- `execution.config`: the normalized evaluation config. This covers generation and
  semantic settings (model, endpoint, temperatures, token limits, context window),
  checkpoint code/spec/evaluator sha256s and initial_fitness, snapshot shas, scorer and
  postgres options, and dependency file shas. Path-valued inputs are replaced by their
  content shas.
- `execution.code`: sha256s of the execution-chain modules (`EXECUTION_MODULES` in
  `router.py`).

`train-router` refuses a config whose hash is not the one bound into the archive's
`baseline_policy_version`. `run-policy` refuses any mismatch and names the differing
fields. Only `application_mode`, `seeds`, `trace_directory`, the protocol `name` and
`baseline_policy_version` may differ. A code edit to any listed module therefore requires
retraining the router.
`train-router` publishes atomically: it writes a temp file, then hard-links it
to a new destination and fails if that destination exists.

### CLI

```bash
python -m trimem.optimizer train-router --archive <branchA_archive> --config <branchA_config.json> \
    --out router.json [--seed 0]
python -m trimem.optimizer run-policy --factory trimem.optimizer.runtime:math_ale_factory \
    --config <branchB_config.json> --router router.json [--router-sha256 <sha>] \
    [--lambda <λ>] --archive <policy_archive_dir> [--repeats 3]
python -m trimem.optimizer status|fail --archive <policy_archive_dir> ...
```

`run-policy` does not go through the search controller. `TrialArchive` validates every
action as a `CandidateSpec`, and the controller checks `space.contains`, so a policy cannot
be a search proposal. `PolicyArchive` subclasses `TrialArchive`, keeps the same trial record
format, and has its own manifest:

- `kind=routed_policy_evaluation`.
- `router_artifact_sha256`.
- `allow_repeats=true` for noise-control repeats. λ may vary across trials in one archive.

Each trial's `proposal` is `{"strategy": "fixed_routed_policy_evaluation"}`, with no fake
acquisition value. Observations pass `validate_observation` before `tell`.

## Limits (read before quoting Branch B numbers)

1. **State × config is confounded with the trajectory.** Every profiling trial applies one
   config for all 10 rounds. The router never sees a config switch, so it never sees a state
   reached by config X followed by config Y. Per-round gains for a config are conditioned on
   states that config produced itself.
2. **Coverage is thin.** With uniform-random Branch A, most configs appear in exactly one
   trial (10 rounds, one seed). In leave-one-trial-out, holding out a trial usually removes
   its config from training entirely, so its counterfactual rounds are pure extrapolation.
3. **λ calibration is an offline proxy, not an estimate of online performance.**
   Counterfactual rounds use model predictions, which makes the proxy optimistic for the
   model. States come from the logging policy, not the router, so no distribution-shift
   correction is applied. Importance weighting is infeasible because the logging policy is
   uniform over about 865 actions with about 20 trials.
4. **Single-seed noise.** v4 showed no_memory at 0.548 vs 0.705 across runs. Per-round
   Δbest is mostly zero with rare jumps, so p̂ is a high-variance regression on a small
   sample. Run the λ\*, λ\*/2 and 2λ\* policies with repeats before claiming a difference.
5. **Memory caching assumes the same config gives the same memory.** That holds for the
   frozen graph and query, but a nondeterministic SemFilter judge could have returned
   different rows. Caching also hides that variation within a branch. Within a branch, the
   PG-side semantic cache can make a second semantic config cheaper than its profiled
   retrieval cost. The ledger records the actual cost.
6. **Gen-token prediction ignores the router's own effect on prompt length** beyond the
   action encoding and features.
