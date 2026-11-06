# TRACE Latent-Space BO (BayesQO-style) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a TRACE search arm that encodes plan+configuration into a learned latent space (BayesQO-style grammar + VAE), proposes batches by maximizing qLogNEHVI over the latent space with multi-start L-BFGS-B inside a trust region, and decodes each proposal into a valid, canonical, deduplicated plan+configuration.

**Architecture:** A fixed-slot grammar maps every TRACE configuration to a token vector, and decodes ANY token vector to a valid canonical configuration (BayesQO §4.1 "decoding validity", modulo mapping). A small VAE (per-slot embeddings → MLP → z ∈ R^d → per-slot categorical heads) is trained once per space on enumerated configurations (no episodes). Per round: measured configs are encoded (posterior mean μ(x)); one GP per objective on z (same ObjectiveProtocol normalization and fixed per-config noise as the native GP); qLogNEHVI is maximized with `botorch.optim.optimize_acqf` (multi-start L-BFGS-B, fixed Sobol base samples) inside a TuRBO-style box; the endpoint z* is decoded (argmax), snapped to z̃ = μ(decode(z*)), deduplicated, and added as pending for the next of q = 4 sequential-greedy picks. The arm plugs in at the existing seam `trace_joint_plan_search.guarded_search` via a new live wrapper, so hashed files of live runs stay untouched.

**Tech Stack:** Python 3.13, torch 2.14 (CPU), botorch 0.17.2, gpytorch 1.15.2, pytest; `.venv-optimizer`.

**Spec:** this plan's Design Decisions section; user note "方法 B：连续 latent 上直接优化 qLogNEHVI——多起点梯度搜索" (2026-11-04); `research_notes/paper_reading_20261104/bayesqo.md` (BayesQO grammar/VAE/TuRBO details and its §7 TRACE caveats).

## Global Constraints

- Do NOT modify any file hashed in a live run's `manifest.live.artifacts` or `manifest.sources` (the stratified runs `results/trace_pr_*_v060_strat_20261104` are live): `trimem/experiments/trace_joint_plan_search.py`, `trimem/optimizer/{plan_schema,plan_codec,plan_surrogate,probabilistic_plan_search,joint_search_controller,joint_search_ledger,staged_pruning}.py`, `tools/trace_joint_live.py`, `tools/prepare_trace_joint_campaign*.py`. New behaviour lives in new files only.
- Objective = qLogNEHVI over (quality, −log1p tokens, −log1p latency) with the run's frozen `objective.json` (`ObjectiveProtocol`), ref point `(-0.1,-0.1,-0.1)`, `SobolQMCNormalSampler(64, seed)` fixed during optimization.
- q = 4 per round; total budget and staged pruning exactly as the stratified runs (`manifest.pruning`, max 80 configs, 5,000 episodes); initialization = the same 5 Sobol + 5 LLM configs.
- Same runtime profile as other arms (c16, `trimem-eval.slice`, 80% RAM rule); report agent steps with every metric.
- Decoding is deterministic (argmax), never stochastic (BayesQO's categorical sampling adds noise; rejected).
- VAE training uses only enumerated/sampled configurations — never TEST or episode outcomes.

## Design Decisions (spec)

| # | Decision | Value |
|---|---|---|
| D1 | Grammar | One slot per `plan_schema.FIELD_NAMES` field, in `FIELDS` order; slot vocabulary = `space.domains[name]` + `PAD`. Encode: inactive field → `PAD`. Decode: slot token t → `domains[name][t % len(domains[name])]` (PAD → default), then `space.canonicalize`. Every token vector decodes to a valid canonical config; canonical configs round-trip exactly (injective, unlike BayesQO). |
| D2 | VAE | Embedding 16/slot → MLP 256-256 → (μ, logσ²) ∈ R^d; decoder MLP 256-256 → per-slot logits. β = 0.1 (BayesQO `kl_factor`). Loss: per-slot CE (masked for inactive slots) + β·KL. |
| D3 | Latent dim | Smallest d ∈ {4, 6, 8, 12, 16} with exact canonical reconstruction ≥ 99.9% on a 200k held-out sample; record in checkpoint. |
| D4 | Training data | 1,000,000 uniform canonical samples (`space.sample`, dedup) + all measured configs of finished v060c runs; seed 20261104; 20 epochs, Adam 1e-3, batch 4096, CPU. |
| D5 | Latent box | Per-dim [min, max] of μ over the training set, widened 5%; stored in the checkpoint as `bounds` (2×d). |
| D6 | Surrogate | `ModelListGP` of 3 `SingleTaskGP` on z (Matérn-5/2 ARD, `Standardize`), `train_Yvar` = the observation noise from `ObjectiveProtocol.tensors` (partial-panel pruned configs included, as in the native GP). |
| D7 | Trust region | Box centered at the Pareto point with max HV contribution; side `L·(bounds_hi−bounds_lo)`; L₀ = 0.8, L ∈ [0.05, 1.6]; after a round: HV improved → L×2, else L×0.5; L < 0.05 → reset to 0.8 at a new center. State persisted in `round_NNN.json["latent_tr"]`. |
| D8 | Acquisition optimization | `optimize_acqf(acq, bounds=TR box, q=1, num_restarts=20, raw_samples=512, return_best_only=False, options={"maxiter": 200, "batch_limit": 5})`; extra starts = encoded current Pareto configs clipped into the box (≤ 8). |
| D9 | Decode + select | For restart endpoints in descending acq value: decode → canonical → skip if key in archive ∪ selected; snap z̃ = μ(decoded); keep first non-duplicate. If all 20 endpoints duplicate: Sobol 2,048 points in the box, decode, take best acq among non-duplicates; if still none: `space.sample(1, avoid=…)` and log `fallback="sobol-space"`. Add z̃ to `X_pending`; repeat for q picks. |
| D10 | Integration | New module replaces `guarded_search` only; ledger, staged evaluation, pruning, TEST deferral unchanged. Manifest field `search = "latent-bo-v1"` and `latent = {"checkpoint", "sha256", "d"}`. |
| D11 | Gates before any live run | G1 reconstruction ≥ 99.9%; G2 latent smoothness: Spearman(‖z_i−z_j‖, |q_i−q_j|) > 0 on measured v060c configs; G3 LOO GP quality on z within 10% of the native mixed GP's LOO log-likelihood; G4 offline replay on ALF/Sci v060c archives: 4 picks valid, distinct, novel, posterior HV gain ≥ the PR picks' median. If G2 or G3 fails, stop and report — do not launch. |

## Review Focus

1. Endpoint decodes to an already-measured config (collapse of many z to one config) — expected: next endpoint, then Sobol-in-box, then space sample; never a duplicate reservation.
2. Snapped z̃ far from z* (decoder region mismatch) — expected: picks are ranked by acq(z̃) (`ell_snapped`), not acq(z*); both are logged.
3. Pruned partial-panel observations in the latent GP — expected: included with their larger noise, never as TR center.
4. Checkpoint/space mismatch (space manifest digest differs from the checkpoint's) — expected: hard error before any proposal.
5. Trust region collapse after repeated non-improving rounds — expected: reset per D7, logged, never an empty box.

---

### Task 1: Grammar (`latent_grammar`)

**Files:**
- Create: `trimem/optimizer/latent_grammar.py`
- Test: `tests/test_trace_latent_grammar.py`

**Interfaces:**
- Produces: `slot_sizes(space) -> list[int]` (vocab incl. PAD per slot); `encode_tokens(space, config) -> list[int]`; `decode_tokens(space, tokens: Sequence[int]) -> dict` (canonical, valid); `PAD = 0` (value i ↦ token i+1).

- [ ] **Step 1: Write failing tests**
  - `test_roundtrip_canonical`: for 20,000 `space.sample(..., seed=1)` configs `c`, `decode_tokens(space, encode_tokens(space, c)) == space.canonicalize(c)`.
  - `test_any_tokens_decode_valid`: 20,000 random int vectors in `[0, 10**6)` of length `len(slot_sizes)` → `space.key(decode_tokens(...))` does not raise and the result equals its own canonicalization.
  - `test_inactive_slots_are_pad`: a config with `filter_order="none"` encodes `sem_threshold_pre`, `sem_threshold_post`, `filter_pre_impl`, `filter_post_impl` slots as `PAD`.
- [ ] **Step 2: Run** `PYTHONPATH=. .venv-optimizer/bin/python -m pytest tests/test_trace_latent_grammar.py -q` — expect FAIL (module missing).
- [ ] **Step 3: Implement** per D1 using `space.active_mask` and `space.canonicalize`; space = `JointSpace(manifest["space"]["domains"], ...)` from `get_space`.
- [ ] **Step 4: Run tests** — expect PASS.
- [ ] **Step 5: Commit** `feat(trace): fixed-slot latent grammar with always-valid decoding`.

### Task 2: VAE + training CLI

**Files:**
- Create: `trimem/optimizer/latent_vae.py`
- Create: `tools/train_trace_latent_vae.py`
- Test: `tests/test_trace_latent_vae.py`

**Interfaces:**
- Consumes: Task 1 `slot_sizes`, `encode_tokens`, `decode_tokens`.
- Produces: `class TraceVAE(torch.nn.Module)(slot_sizes: list[int], d: int)` with `encode_mean(tokens: LongTensor[N,S]) -> Tensor[N,d]`, `decode_argmax(z: Tensor[N,d]) -> LongTensor[N,S]`; `train_vae(space, d: int, *, n_samples: int, epochs: int, seed: int, extra_configs=()) -> tuple[TraceVAE, dict]` (metrics: `recon_exact`, `kl`, `bounds`); `save_checkpoint(path, model, space, metrics)` / `load_checkpoint(path, space) -> tuple[TraceVAE, dict]` (raises `ValueError` on space-digest mismatch); CLI `python -m tools.train_trace_latent_vae --run-dir <run> --out <dir> [--dims 4,6,8,12,16]` writing `vae_d{d}.pt` + `vae_metrics.json` and choosing d per D3.

- [ ] **Step 1: Write failing tests**
  - `test_small_vae_reconstructs`: `train_vae(space, 8, n_samples=20000, epochs=5, seed=0)` → `metrics["recon_exact"] >= 0.95` on its held-out split (CPU, < 60 s).
  - `test_decode_is_deterministic`: same z decoded twice → identical tokens.
  - `test_checkpoint_space_mismatch`: loading with a space whose manifest digest differs raises `ValueError`.
- [ ] **Step 2: Run** — expect FAIL.
- [ ] **Step 3: Implement** per D2/D4/D5 (masked CE on inactive slots; bounds from μ over training set +5%).
- [ ] **Step 4: Run tests** — expect PASS.
- [ ] **Step 5: Train real checkpoints**: `PYTHONPATH=. .venv-optimizer/bin/python -m tools.train_trace_latent_vae --run-dir results/trace_pr_sciworld_v060_strat_20261104 --out results/trace_latent_vae_20261104` — expect `vae_metrics.json` with chosen d and `recon_exact >= 0.999` (Gate G1). If no d meets G1, stop and report.
- [ ] **Step 6: Commit** `feat(trace): latent VAE over TRACE configurations`.

### Task 3: Latent BO proposer

**Files:**
- Create: `trimem/optimizer/latent_bo.py`
- Test: `tests/test_trace_latent_bo.py`

**Interfaces:**
- Consumes: Task 2 `TraceVAE`, `load_checkpoint`; Task 1 `encode_tokens`/`decode_tokens`; `plan_surrogate.ObjectiveProtocol`, `Observation`.
- Produces: `fit_latent_model(Z: Tensor[n,d], obs: list[Observation], objective: ObjectiveProtocol) -> ModelListGP`; `@dataclass TrustRegion(center: list[float], length: float)` with `update(improved: bool) -> TrustRegion` and `box(bounds: Tensor[2,d]) -> Tensor[2,d]`; `propose_latent_batch(space, vae, obs, objective, *, q: int, seed: int, tr: TrustRegion | None, archive_keys: set[str], pruned_keys: set[str] = frozenset()) -> dict` returning `{"selected": [{"config", "key", "ell", "ell_snapped", "z", "z_snapped", "restart", "fallback"}], "latent_tr": asdict(tr_next), "stats": {"restarts", "duplicates", "decode_collisions"}}`.

- [ ] **Step 1: Write failing tests**
  - `test_batch_valid_distinct_novel`: synthetic 12 observations (quality = f(seed_count, rank_impl), costs = f(fanout)) with a d=6 VAE trained on 20k samples → `propose_latent_batch(q=4)` returns 4 configs, all `space.canonicalize`-fixed points, distinct keys, none in `archive_keys`.
  - `test_duplicate_endpoints_fall_back`: monkeypatch `decode_argmax` to always return one archived config → result still has 4 novel picks with `fallback in {"sobol-box", "sobol-space"}`.
  - `test_trust_region_update`: `TrustRegion(c, 0.8).update(True).length == 1.6`; repeated `update(False)` from 0.8 resets to 0.8 once below 0.05; `box()` stays within bounds and has positive width.
  - `test_pending_changes_second_pick`: with q=2 the second pick's `ell` is computed with the first pick's `z_snapped` in `X_pending` (assert acquisition called with non-empty `X_pending`).
  - `test_ranked_by_snapped_value`: every pick has `ell_snapped` set, and among the endpoints the chosen one has the highest `ell_snapped` of the non-duplicates.
  - `test_pruned_obs_never_center`: observations flagged pruned (partial panel) are used by the GP but `TrustRegion.center` is the encoding of a full-panel Pareto config.
- [ ] **Step 2: Run** — expect FAIL.
- [ ] **Step 3: Implement** per D6–D9 (`qLogNoisyExpectedHypervolumeImprovement`, `optimize_acqf(..., return_best_only=False)`, fixed `SobolQMCNormalSampler(64, seed=seed)`).
- [ ] **Step 4: Run tests** — expect PASS.
- [ ] **Step 5: Commit** `feat(trace): latent-space qLogNEHVI proposer with trust region and decode/dedup`.

### Task 4: Offline gates on finished archives

**Files:**
- Create: `tools/latent_offline_gates.py`
- Test: run on real data (no unit test; the gate report is the deliverable)

**Interfaces:**
- Consumes: Tasks 1–3; finished runs `results/trace_pr_alfworld_v060_20261103c`, `results/trace_pr_sciworld_v060_20261103c` (`observations`, `objective.json`, `round_0*.json` PR picks).
- Produces: `results/trace_latent_vae_20261104/gates.json` with keys `G1_recon`, `G2_spearman`, `G3_loo_latent`, `G3_loo_native`, `G4_replay` (per dataset per round: picks, novelty, posterior HV gain vs PR picks), and `pass: bool`.

- [ ] **Step 1: Implement** `main(run_dirs, checkpoint) -> dict` computing G1–G4 per D11 (G3 = mean LOO predictive log-likelihood per objective; native = `plan_surrogate.fit_model`).
- [ ] **Step 2: Run** `PYTHONPATH=. .venv-optimizer/bin/python -m tools.latent_offline_gates --checkpoint results/trace_latent_vae_20261104/vae_d<d>.pt results/trace_pr_alfworld_v060_20261103c results/trace_pr_sciworld_v060_20261103c` — expect `gates.json` written; `pass` decides whether Task 5 proceeds.
- [ ] **Step 3: Report** G1–G4 to the user with numbers; stop here if `pass` is false.
- [ ] **Step 4: Commit** `feat(trace): offline gates for latent BO`.

### Task 5: Live arm wrapper and prepare

**Files:**
- Create: `trimem/experiments/trace_latent_search.py`
- Create: `tools/trace_latent_live.py`
- Create: `tools/prepare_trace_latent_campaign.py`
- Test: `tests/test_trace_latent_live.py`

**Interfaces:**
- Consumes: Task 3 `propose_latent_batch`; `trace_joint_plan_search.{observations, get_space, atomic_json}`; `prepare_trace_joint_campaign_stratified` (wrapped, not modified).
- Produces: `latent_guarded_search(directory, manifest, entries, *, agent=(), seed=0, round_file="round") -> tuple[dict, None]` (same contract as `guarded_search`: writes `round_file.json` with `selected` [{config,key,ell,sources:["latent"]}], `archive_keys`, `latent_tr`); `tools.trace_latent_live` = `tools.trace_joint_live.main` with `trace_joint_plan_search.guarded_search` patched to `latent_guarded_search` when `manifest["search"] == "latent-bo-v1"` (refuses otherwise); `tools.prepare_trace_latent_campaign --dataset D --run-dir R --checkpoint P` = stratified prepare + `search`, `latent` block, and checkpoint + new modules added to `live.artifacts`.

- [ ] **Step 1: Write failing test** `test_latent_round_end_to_end` (fixture manifest as `tests/test_trace_pr_staged_pruning.py`, `search="latent-bo-v1"`, tiny checkpoint from Task 2): three `run_round` calls with `guarded_search` patched → each BO round reserves 4 novel configs with `sources == ["latent"]`, `round_NNN.json["latent_tr"]` present, pruning log still written.
- [ ] **Step 2: Run** — expect FAIL.
- [ ] **Step 3: Implement** the three files (TR state read from the previous `round_*.json`; HV-improvement flag from the archive's HV before/after the last batch).
- [ ] **Step 4: Run** the new test and the full suite `pytest -q tests/test_trace_joint_*.py tests/test_trace_pr_staged_pruning.py tests/test_trace_latent_*.py` — expect all PASS.
- [ ] **Step 5: Adversarial review** (`adversarial-review` skill, codex, prompt-inlined diff) — fix accepted findings, re-run suite.
- [ ] **Step 6: Commit** `feat(trace): latent-BO live arm`.

### Task 6: Campaign (only if Task 4 gates pass)

**Files:**
- Create: `results/trace_latent_<dataset>_20261105/` via Task 5 prepare
- Modify: nothing hashed

- [ ] **Step 1: Prepare** one dataset first (ScienceWorld, same stratified panel/seed as `trace_pr_sciworld_v060_strat_20261104`) with `tools.prepare_trace_latent_campaign`; run `tools.trace_latent_live --preflight-only` — expect `passed`.
- [ ] **Step 2: Queue** after the running stratified chain (systemd unit in `trimem-eval.slice`); TEST via `tools.trace_pr_test_now`, then `trace_pr_final_select`, `trace_pr_router_curve`, `publish_trace_pr_test_points --label "TRACE latent-BO"`.
- [ ] **Step 3: Compare** with the PR arm on the same panel: HV vs episodes (TRAIN), TEST Pareto points (quality, tokens, latency, agent steps), picks' novelty/collisions; report to the user.
