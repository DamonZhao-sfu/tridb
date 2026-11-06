# TRACE Latent-MOBO (latent refinement channel) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the backend `latent-mobo-v1` to TRACE. In this backend, a "latent refinement" channel replaces the PR channel. The channel runs multi-start L-BFGS-B on qLogNEHVI over a frozen sequence-VAE latent, then decodes each endpoint to a valid config, re-encodes it and re-scores it. Only canonical-config scores compete with the Sobol, LLM and Recombination candidates for the q≤4 profiling slots.

**Architecture:** New modules only. The masked deterministic decoder, the frozen affine map `u = A(E(c))`, the latent GP, the trust-region state and the batch proposer live in new `trimem/optimizer/latent_*.py` files. They reuse `SeqGrammar`/`SeqVAE`. They also reuse `AcquisitionSnapshot` through a duck-typed codec. A new worker replaces `trace_joint_plan_search._search_worker` through a monkeypatch at the documented seam. The monkeypatch is in a new live wrapper, the same pattern as `tools/prepare_trace_joint_campaign_stratified.py`. A new shared-initialization driver produces the 2×(5 Sobol + 5 LLM) = 20 initial panels. Both arms (latent and an explicit-PR control) import these panels.

**Tech Stack:** Python 3.13, `.venv-optimizer` (torch 2.14.1+cpu, botorch 0.17.2, gpytorch 1.15.2, scipy 1.18.1), pytest. VAE training used `miniconda3/envs/vllm17` (CUDA). This plan does not train a VAE.

**Spec:** `docs/superpowers/specs/2026-11-04-trace-latent-mobo-design_v0.1.0.md` (authoritative). Also read `docs/superpowers/specs/reviews/2026-11-04-trace-latent-mobo/{architect,skeptic,verdict}.md`. This plan supersedes `docs/superpowers/plans/2026-11-04-trace-latent-bo.md`.

## User decisions (2026-11-04) — override anything below that conflicts

- **Initialization: 10 configs = 5 Sobol + 5 LLM** (one stage, the existing `sobol_then_llm`), not 2×(5+5). Total budget stays 50 configs per arm: **10 init + 40 exploration = 10 batches × q=4** (same shape as the PR runs). Consequence: Task 8 shrinks to "run the existing 5+5 init once as a shared-init run (budget 10) and import it into both arms with `import_from`"; no new init-stage driver and no `run_round` patch for init. The objective normalization (`objective.json`) is frozen from these 10.
- **TEST = TRAIN sample-mean Pareto only** (spec §12.5). Use the existing `holdout.selection = "train_pareto"` (`tools/trace_joint_protocol.run_holdout`), run automatically by the live driver at budget completion; do NOT use `trace_pr_test_now` (union) for these arms.
- **TRAIN panel stratified by the TEST task-type mix** (`tools/trace_joint_stratified_panel.make_stratified_panels`, as the `*_v060_strat_20261104` runs).
- **Concurrency c16** for TRAIN and TEST, the same runtime profile as all historical runs.
- Q1 gate margins approved as proposed (Spearman −0.10, log-likelihood −0.5 nats). Q2 dataset order (updated by the user): **ALFWorld → ScienceWorld → WebShop**. Q4: the **explicit-PR control arm is approved** (same shared init, same panel, no pruning). Q6: if every checkpoint fails the gates, **stop and report** (no retraining).
- VAE checkpoints: d ∈ {16, 32, 64} (d8 dropped by the user), `results/trace_latent_seqvae_20261104/seqvae_d{16,32,64}.pt`.

## Global Constraints

- **Hashed files — never modify:** `trimem/experiments/trace_joint_plan_search.py`, `trimem/optimizer/{plan_schema,plan_codec,plan_completion,plan_surrogate,probabilistic_plan_search,joint_search_controller,joint_search_ledger,joint_plan_analyzer,staged_pruning}.py`, `tools/trace_joint_live.py`, `tools/prepare_trace_joint_campaign*.py`, `tools/trace_pr_test_now.py`, `tools/trace_pr_final_select.py`. New behaviour goes in new modules. The new modules monkeypatch only these module attributes: `trace_joint_plan_search._search_worker`, `trace_joint_plan_search.run_round`, `prepare_trace_joint_campaign.{prepare_run,import_archive,legacy_to_v060,INITIAL,NEW}`.
- **Tests:** `PYTHONPATH=. .venv-optimizer/bin/python -m pytest <file> -q`. Each task: failing test first, then the code. The `.venv-optimizer` torch is CPU-only, so all VAE inference runs on CPU (`map_location="cpu"`, ≤4 threads).
- **Budget (user decision, supersedes spec §1):** 10 initial configs = 5 Sobol + 5 LLM, then 40 exploration configs = 10 batches × q=4, on one frozen 100-task TRAIN panel stratified by TEST type. After TRAIN, the TRAIN-Pareto configs run once on the fixed TEST-200 panel. TEST never enters the VAE, the GP, gates, thresholds or the candidate selection.
- **No staged pruning in either arm (spec §7):** the manifest has no `pruning` key. `aggregate_panel(..., allow_partial=False)` admits only complete 100-task panels. The latent worker refuses a manifest that has `pruning`.
- **Objective (spec §7):** `target = (Q, -log1p(T), -log1p(L))`. `y_norm = (target-lo)/span`, frozen in `objective.json` from the 20 initial configs. Span floor is `(1, 1e-3, 1e-3)`. `ref = (-0.1,-0.1,-0.1)` in y_norm units. `train_Yvar = var_target/span²`, where var is the per-config task variance/n, log-propagated (existing `aggregate_panel`). `Standardize(m=1)` is GP-internal only. The ref is never standardized. NHV = HV/1.1³.
- **GP (spec §6):** three independent `SingleTaskGP` on canonical `u(c) ∈ R^d`, Matérn-5/2 ARD (`get_covar_module_with_dim_scaled_prior(ard_num_dims=d, use_rbf_kernel=False)`), `Standardize(m=1)`, fixed `train_Yvar`, no input transform (A is already frozen to [0,1]^d), `ModelListGP`. Fit timeout 120 s.
- **Score contract (spec §6, §8):** every candidate is a valid canonical config before it gets a score. The score is `ell(u(c))` from a pending-aware `qLogNoisyExpectedHypervolumeImprovement` with `X_pending = u(selected-this-batch)`, 64 Sobol QMC samples and seed fixed per round. An endpoint score `ell(u*)` is a diagnostic only. It never claims a slot. Gradients go through the GP/acquisition only, never through the decoder.
- **Proposal compute (spec §8):** anchors ≤ 25. Box `R(a)=[a-h/2,a+h/2]∩[0,1]^d`. h₀=0.4, h∈[0.05,1.0]. Per anchor: L-BFGS-B ≤50 iterations and ≤100 objective calls. Per batch: ≤300 s and ≤10,000 calls. Backfill: legal Sobol ≤4096 draws.
- **Resources (spec §11):** the reused `guarded_search` caps the worker at 4 CPU threads, RSS 4 GiB and 125 s for startup/GP. Agent runs use the same runtime profile as other arms: c16, `trimem-eval.slice`, 80 % RAM cap, `scripts/env_servers.sh`. Report agent steps with every metric.
- **Identity (spec §11):** config identity = `JointSpace.key`. The VAE version, the checkpoint sha256 and the affine sha256 are optimizer fingerprints in `manifest["latent"]` and `live.artifacts`. They do not change the config identity.
- **Commits:** the working tree has unrelated uncommitted changes. Stage only the files that a task names. Commit only if the user authorized commits for this execution. Message format: `type(scope): summary`, then the session attribution lines.
- **Spec ⇄ brief/code disagreements and their resolution in this plan:**
  1. *Representation (spec §3–§5 vs existing code).* The spec asks for a fixed field order with INACTIVE tokens, a 2-layer/64-hidden encoder, d=8 (16 fallback), β ramp to 0.01, a 10k stratified corpus with combination holdout, and init-20 in the training partition. The existing `SeqVAE` uses a BayesQO variable-length grammar, 6+6 layers, d_model 128, β=0.1, a 300k uniform Sobol corpus and a 20k disjoint held-out set, with d ∈ {8,16,32,64} trained before the init exists. **Resolution:** reuse the existing checkpoints (user instruction). Apply the spec §5 gates as acceptance (Task 11). Report "no held-out combination test established". Replace the training-partition membership of init-20 with the init round-trip gate R6 (Task 13).
  2. *Decoder (spec §4 vs `SeqGrammar.decode`).* The spec forbids default-filling. The repair rule `valid[t % len(valid)]` is a default fill. **Resolution:** a new masked greedy decoder (Task 1). The live path never repairs: it asserts that every decoded token is allowed.
  3. *20-point init (spec §1).* `validate_manifest` allows only one 5+5 stage. **Resolution:** a new shared-init stage driver (Task 8) runs under a `run_round` patch. The arms import the 20 panels.
  4. *TEST set (spec §12.5).* The spec tests all TRAIN sample-mean non-dominated configs. `tools/trace_pr_test_now.py` tests the union of sample-mean Pareto, GP Pareto and OneConfig. **Resolution:** run `trace_pr_test_now` unchanged for both arms. It is the same rule for both arms and a superset of the spec set. Report the spec subset separately (Task 16). Open question Q5.
  5. *Recovery (spec §11 state machine).* The ledger is hashed. **Resolution:** map the spec states onto the existing ledger. `reserved` = config reserved. `submitted` = episode `claimed` + `attempt.json`. `completed_raw` = attached receipt. `archived` = `complete_configs`. `submission_unknown` = `recover_claims` "uncertain". The proposal is persisted in `round_NNN.json` before reservation. "Atomic reservation retries ≤8" cannot conflict under `driver.lock`, so record it as N/A. The retry policy is the existing one: `max_attempts=2` (one retry per task) and `failure_breaker=3`. The spec's "retries ≤10 % of nominal episodes" cap is not enforced by code. `trace_latent_compare` reports the retry count and the retry cost.
  6. *Pending (spec §8).* The spec says "running + reserved". `_run_round` proposes only when nothing is pending, so `X_pending` = this batch's earlier picks.
  7. *RSS (spec §11 8 GiB).* The reused guard is 4 GiB. That is stricter, and it is the same as the PR arm.
  8. *Comparison arm (spec §12.5).* The live stratified PR runs (10 init, pruning, ≤80 configs) are not comparable. **Resolution:** a new explicit-PR control arm with the same 20 init, the same panel and no pruning. It costs 30×100 extra episodes. Open question Q4.
  9. *Brief "noise: partial panels if pruning".* Per the spec there is no pruning, so the noise comes from full panels only.

## Review Focus

1. **A v0.6.0 config with `graph_direction="Forward"` is imported from the shared init.** `import_archive` applies `legacy_to_v060` (Forward→Both). Expected: the arm wrapper imports configs unchanged. Test: `test_arm_import_keeps_forward` (Task 9).
2. **A decoded endpoint re-encodes outside [0,1]^d.** Expected: it is scored and can be submitted. Its u is never clipped. It is never used as an anchor. Tests: `test_affine_does_not_clip` (Task 2), `test_out_of_domain_anchor_skipped` (Task 6).
3. **Many endpoints collapse to one key (decoder holes), or all endpoints decode to measured keys.** Expected: at most one pool entry per key. The collapse ratio is logged. The slot is still filled from the other sources. Test: `test_collapsed_endpoints_fill_from_other_sources` (Task 6).
4. **The wall-time budget runs out in the middle of a slot.** Expected: only validated endpoints enter the pool. Unvisited anchors are logged `unvisited`, not "no improvement". The later slots select from the existing legal pool. Test: `test_budget_exhaustion_marks_unvisited` (Task 6).
5. **The checkpoint or the affine file changes after prepare, or the space domains differ from the checkpoint domains.** Expected: a hard error before any GP fit or proposal. Test: `test_worker_refuses_changed_checkpoint` (Task 7).

---

### Task 0: Baseline the existing latent code and pin CPU loading

**Files:**
- Test: `tests/test_trace_latent_seq_vae.py` (create)
- Create: `tests/latent_fixtures.py`
- Commit (already written, untracked): `trimem/optimizer/latent_seq_grammar.py`, `trimem/optimizer/latent_seq_vae.py`, `trimem/optimizer/latent_grammar.py`, `trimem/optimizer/latent_vae.py`, `tools/train_trace_seq_vae.py`, `tools/latent_offline_gates.py`, `tests/test_trace_latent_seq_grammar.py`, `tests/test_trace_latent_grammar.py`

**Interfaces:**
- Produces (`tests/latent_fixtures.py`): `TINY_DOMAINS: dict`. Values: filter_order (none, before, after); graph_impl (BFS, beam); filter_pre/post_impl (LLM); rank_impl (ColBERT, LLM); retrieval_unit (whole); injection_timing (start, periodic); seed_count (8, 32); depth (2, 3); fanout (8); edge_threshold (off); beam_width (2, 4); sem_threshold_pre/post (0.6); retrieval_interval (5); returned_k (3, 5); trajectory_category (All); graph_direction (Forward, Both). This gives 576 canonical configs. Also produces: `tiny_space() -> JointSpace`; `enumerate_space(space) -> list[dict]` (product, canonicalize, dedupe by key); `tiny_model(d=4) -> SeqVAE` (`functools.lru_cache`; `train_seq_vae(space, d, n_samples=192, holdout=16, epochs=3, seed=0, layers=1, batch=64)`); `synthetic_receipt(config, task, manifest) -> dict` (labelled `synthetic: True`; quality rises with seed_count=32 and rank LLM; tokens rise with depth; latency rises with beam).

- [ ] **Step 1: Write the failing tests**

```python
def test_checkpoint_roundtrip_cpu(tmp_path):
    m = tiny_model(); p = tmp_path / "t.pt"
    save_checkpoint(p, m, tiny_space(), {"d": 4})
    m2, _ = load_checkpoint(p, tiny_space())
    x = torch.as_tensor(to_matrix(SeqGrammar(tiny_space()), enumerate_space(tiny_space())[:8]))
    assert not m2.training and torch.equal(m.encode_mean(x[:, 1:]), m2.encode_mean(x[:, 1:]))

def test_checkpoint_domain_mismatch_raises(tmp_path):  # load with JointSpace.live() -> ValueError

def test_enumerate_tiny_space():
    assert len(enumerate_space(tiny_space())) == 576
```

- [ ] **Step 2: Run** `PYTHONPATH=. .venv-optimizer/bin/python -m pytest tests/test_trace_latent_seq_vae.py tests/test_trace_latent_seq_grammar.py -q`. Expected: FAIL (`ModuleNotFoundError: tests.latent_fixtures`).
- [ ] **Step 3: Write `tests/latent_fixtures.py`.** Recompute the count with `enumerate_space`. If the count is not 576, correct the literal in the test, not the domains.
- [ ] **Step 4: Run the same command.** Expected: all PASS.
- [ ] **Step 5: Commit** the listed files: `feat(trace-latent): baseline sequence grammar/VAE and CPU loader tests`.

### Task 1: Masked deterministic decoder (spec §4)

**Files:**
- Create: `trimem/optimizer/latent_seq_decode.py`
- Test: `tests/test_trace_latent_seq_decode.py`

**Interfaces:**
- Consumes: `SeqGrammar` (`vocab`, `index`, `decode`, `_pick`), `SeqVAE._decode_logits`, `MAX_LEN`, `BOS`, `EOS`.
- Produces: `allowed_next(grammar: SeqGrammar, prefix: Sequence[int]) -> tuple[int, ...]`. Also `decode_masked(model: SeqVAE, grammar: SeqGrammar, z: torch.Tensor) -> list[list[int]]` (tokens without BOS, ending at EOS) and `decode_configs(model, grammar, z) -> list[dict]` (canonical; raises `ValueError` on non-finite z; raises `RuntimeError` if a token outside `allowed_next` would be needed).

- [ ] **Step 1: Write the failing tests**

```python
def test_encoded_sequences_are_accepted_step_by_step():
    for c in enumerate_space(S):
        toks = G.encode(c)
        for i, t in enumerate(toks):
            assert t in allowed_next(G, toks[:i])
        assert allowed_next(G, toks) == ()

def test_masked_decode_needs_no_repair():          # 256 z ~ N(0, 25I) incl. |z|=1e3 rows
    for toks in decode_masked(M, G, z):
        assert all(t in allowed_next(G, toks[:i]) for i, t in enumerate(toks))
        assert G.decode(toks) == S.canonicalize(G.decode(toks)) and toks[-1] == G.index[EOS]

def test_decode_is_deterministic():                # same z twice -> identical configs
def test_nonfinite_latent_rejected():              # NaN / inf row -> ValueError
def test_beam_width_only_after_beam():             # decoded BFS configs never contain a "beam=" symbol
```

- [ ] **Step 2: Run** `PYTHONPATH=. .venv-optimizer/bin/python -m pytest tests/test_trace_latent_seq_decode.py -q`. Expected: FAIL (`ModuleNotFoundError`).
- [ ] **Step 3: Implement.** For `allowed_next`, use a private `SeqGrammar` subclass that overrides `_pick`. The override raises an internal `_Need(valid)` when the token is `None`. Map `valid` symbols to indices. A prefix that parses completely returns `(EOS,)`. A prefix that ends in EOS returns `()`. This reuses the grammar's own state machine, so no parser is duplicated. For `decode_masked`: eval mode, `torch.no_grad()`, batched autoregressive steps of ≤`MAX_LEN-1`. Set logits outside each row's allowed set to −inf. Take argmax with the lowest index on ties. Stop each row at EOS.
- [ ] **Step 4: Run the same command.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(trace-latent): masked greedy decoder without repair`.

### Task 2: Frozen affine map and `LatentCodec` (spec §6)

**Files:**
- Create: `trimem/optimizer/latent_plan_codec.py`
- Test: `tests/test_trace_latent_codec.py`

**Interfaces:**
- Consumes: Task 1 `decode_configs`; `SeqVAE.encode_mean`; `to_matrix`; `load_checkpoint`.
- Produces:
  - `@dataclass(frozen=True) Affine(lo: tuple[float, ...], hi: tuple[float, ...])` with `to_u(z: Tensor) -> Tensor` and `to_z(u: Tensor) -> Tensor` (double, no clipping).
  - `fit_affine(model, grammar, train_configs, init_configs=()) -> Affine`. Rule: std floor 1e-3; the initial bounds are mean±4·std over the train encodings; widen them to cover all train encodings and all init encodings; then add 5 % of the *original* width on each side.
  - `save_affine(path, affine, meta: dict)` and `load_affine(path) -> tuple[Affine, dict]`.
  - `class LatentCodec(space, model, affine, *, checkpoint_sha256: str)` with `.space`, `.d`, `.identity` (digest of space.identity, checkpoint sha and affine), `.key(c)`, `.encode(c) -> list[float]` (u, cached by key), `.encode_many(configs) -> Tensor[N,d]` (double, fills the cache), `.decode(U: Tensor[N,d]) -> list[dict]` (`to_z` then `decode_configs`) and `.roundtrip(c) -> bool` (`decode(encode(c)) == canonicalize(c)`).
  - `load_codec(checkpoint: Path, affine: Path, space) -> LatentCodec`. It verifies that the checkpoint sha256 equals `meta["checkpoint_sha256"]`. On a mismatch it raises `ValueError`.

- [ ] **Step 1: Write the failing tests**

```python
def test_affine_rule():                        # synthetic Z: bounds ⊇ all train/init rows; margin = 5% of mean±4std width
def test_affine_does_not_clip():
    a = Affine((0.0,), (1.0,)); u = a.to_u(torch.tensor([[2.0]], dtype=torch.double))
    assert u.item() == 2.0 and a.to_z(u).item() == 2.0
def test_train_encodings_inside_unit_box():    # tiny model, all 576 configs as train -> 0 <= u <= 1
def test_encode_cached_and_deterministic():    # encode(c) twice -> same list; encode_many fills cache
def test_codec_rejects_changed_checkpoint(tmp_path):  # edit one byte of the .pt -> load_codec ValueError
```

- [ ] **Step 2: Run** `PYTHONPATH=. .venv-optimizer/bin/python -m pytest tests/test_trace_latent_codec.py -q`. Expected: FAIL (`ModuleNotFoundError`).
- [ ] **Step 3: Implement** the interfaces above. Encode in batches of 4096 with `torch.no_grad()` and cast to double. `torch.set_num_threads` is the caller's job.
- [ ] **Step 4: Run the same command.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(trace-latent): frozen affine latent codec`.

### Task 3: Latent surrogate and coordinate contract (spec §6–§7)

**Files:**
- Create: `trimem/optimizer/latent_plan_surrogate.py`
- Test: `tests/test_trace_latent_surrogate.py`

**Interfaces:**
- Consumes: Task 2 `LatentCodec`; `plan_surrogate.{Observation, ObjectiveProtocol, AcquisitionSnapshot, aggregate_panel}` (import only).
- Produces:
  - `fit_latent_model(codec, observations, objective, *, seed=0, maxiter=150, timeout=120) -> ModelListGP`. It raises `ValueError` on mixed `panel_identity` or repeated keys, the same checks as `fit_model`.
  - `latent_snapshot(codec, model, observations, objective, *, seed, pending=()) -> AcquisitionSnapshot`. This is the existing class with `budget=None`. The duck-typed codec is accepted because the class needs only `.space.identity`, `.encode` and `.key`.
  - `acquisition_u(snapshot, U: Tensor[N,d]) -> Tensor[N]`. It returns `snapshot.acquisition(U.unsqueeze(-2))` with gradients enabled.
  - `measured_hv(observations, objective) -> float`. It computes the HV of the y_norm means that are strictly above ref, with botorch `Hypervolume`.

- [ ] **Step 1: Write the failing tests**

```python
def test_hv_hand_computed():                  # y_norm points (1,0,0),(0,1,0), ref -0.1 -> 0.021 (abs 1e-12)
def test_point_below_ref_adds_zero_hv():      # adding (-0.5, 2, 2) leaves HV unchanged
def test_posterior_in_ynorm_units():          # 12 obs, tiny noise: |posterior mean - train_Y| < 0.05 at train u
def test_yvar_is_scaled_by_span_squared():    # model.models[i].likelihood noise == v[:, i] / span[i]**2
def test_gradient_matches_finite_difference():# interior u; central FD eps 1e-4; rtol 2e-2
def test_pending_lowers_duplicate_score():
    s = latent_snapshot(...); c = candidate
    assert s.with_pending([c]).score([c])[0] < s.score([c])[0]
def test_mixed_panels_rejected():             # ValueError
```

- [ ] **Step 2: Run** `PYTHONPATH=. .venv-optimizer/bin/python -m pytest tests/test_trace_latent_surrogate.py -q`. Expected: FAIL (`ModuleNotFoundError`).
- [ ] **Step 3: Implement.** Mirror `fit_model`, but use `SingleTaskGP(X, Y_i, train_Yvar=V_i, covar_module=<Matérn-5/2 ARD per Global Constraints>, outcome_transform=Standardize(m=1))`. Use `fit_gpytorch_mll(..., optimizer_kwargs={"options": {"maxiter": maxiter}, "timeout_sec": timeout})` under `torch.random.fork_rng` with the seed.
- [ ] **Step 4: Run the same command.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(trace-latent): latent GP, gradient acquisition and measured HV`.

### Task 4: Trust region, compute budget and anchor order (spec §8)

**Files:**
- Create: `trimem/optimizer/latent_region.py`
- Test: `tests/test_trace_latent_region.py`

**Interfaces:**
- Produces:
  - Constants: `H0, H_MIN, H_MAX, GAIN_EPS, STREAK = 0.4, 0.05, 1.0, 1e-4, 3`.
  - `@dataclass(frozen=True) TrustState(h=H0, up=0, down=0, resets=0)`.
  - `step(s: TrustState, gain: float) -> TrustState`. Rule: if gain > GAIN_EPS, set up+=1 and down=0, else set down+=1 and up=0. When up reaches STREAK: h=min(2h, H_MAX) and up=0. When down reaches STREAK: if h == H_MIN, reset h=H0 and resets+=1, else h=max(h/2, H_MIN); in both cases down=0.
  - `replay(gains: Sequence[float]) -> TrustState`.
  - `box(anchor: Tensor[d], h: float) -> Tensor[2,d]` (clamped to [0,1]).
  - `batch_gains(rounds: Sequence[tuple[set[str], list[str]]], observations, objective) -> list[float]`. Each round is (archive_keys, selected_keys). Only rounds with all selected keys complete count. gain = measured_hv(archive ∪ selected) − measured_hv(archive).
  - `class BatchBudget(seconds: float | None = 300.0, calls: int = 10_000)` with `slot_share(remaining_slots) -> tuple[float | None, int]`, `anchor_share(slot_share, remaining_anchors) -> tuple[float | None, int]` (calls are capped at 100 per anchor), `charge(seconds, calls)` and `exhausted() -> bool`.
  - `anchor_order(n: int, *, seed: int, round_index: int, slot: int) -> list[int]` (`random.Random(digest([seed, round_index, slot])).shuffle`).

- [ ] **Step 1: Write the failing tests** (table-driven)

```python
@pytest.mark.parametrize("gains,h,resets", [
    ([1, 1, 1], 0.8, 0), ([1] * 6, 1.0, 0), ([0, 0, 0], 0.2, 0),
    ([0] * 9, 0.05, 0), ([0] * 12, 0.4, 1), ([1, 0, 1, 0, 1, 0], 0.4, 0)])
def test_replay(gains, h, resets): ...
def test_box_clamped_and_positive_width():
def test_batch_gains_skip_incomplete_rounds():
def test_budget_shares_carry_unused_forward():   # slot gets remaining/remaining_slots; unused flows to next anchor
def test_anchor_order_frozen_and_slot_dependent():
```

- [ ] **Step 2: Run** `PYTHONPATH=. .venv-optimizer/bin/python -m pytest tests/test_trace_latent_region.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** the interfaces above. All state is a pure function of the round history, so resume needs no mutable state file.
- [ ] **Step 4: Run the same command.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(trace-latent): HV-driven trust region and proposal budget`.

### Task 5: Endpoint refinement and projection (spec §6, §8)

**Files:**
- Create: `trimem/optimizer/latent_plan_search.py` (first half)
- Test: `tests/test_trace_latent_refine.py`

**Interfaces:**
- Consumes: Task 3 `acquisition_u`/snapshot; Task 4 `box`; Task 2 `LatentCodec`.
- Produces:
  - `@dataclass(frozen=True) Endpoint(anchor_key: str, anchor_source: str, u_start: tuple, u_end: tuple | None, ell_start: float | None, ell_end: float | None, status: str, calls: int, seconds: float)`. `status ∈ {"converged","maxfun","timeout","nan","error","unvisited"}`.
  - `refine_anchor(snapshot, u0: Tensor[d], bounds: Tensor[2,d], *, max_calls: int, seconds: float | None, anchor_key: str, anchor_source: str) -> Endpoint`. It wraps `snapshot.acquisition` in a call-counting `AcquisitionFunction` and runs `botorch.generation.gen.gen_candidates_scipy(u0[None,None], acq, lower_bounds=bounds[0], upper_bounds=bounds[1], options={"maxiter": 50, "maxfun": max_calls}, timeout_sec=seconds)`. A non-finite value or gradient gives `status="nan"` with `ell_end=None`. The score is never fabricated.
  - `project(codec, snapshot, endpoints) -> list[dict]`. For each usable endpoint, it returns `{config, key, u_end, u_canonical, ell_end, ell_canonical, delta, distance, roundtrip, anchor_key, anchor_source}`. Here `ell_canonical = snapshot.score([config])[0]`, `delta = ell_canonical - ell_end` and `distance = ||u_end - u_canonical||₂`. An endpoint that fails decode or validation gets `{"status": "rejected", "reason": ...}`.

- [ ] **Step 1: Write the failing tests**

```python
def test_endpoint_stays_in_box():
def test_refine_does_not_decrease_ell():          # ell_end >= ell_start - 1e-6
def test_calls_counted_and_bounded():             # max_calls=20 -> 0 < e.calls <= 20 + 25 (one line search slack)
def test_nan_acquisition_reports_nan(monkeypatch):# patched acquisition returns NaN -> status "nan", ell_end None
def test_projection_rescores_at_canonical_u():
    r = project(codec, snap, [e])[0]
    assert r["ell_canonical"] == snap.score([r["config"]])[0]
    assert r["u_canonical"] == tuple(codec.encode(r["config"]))
```

- [ ] **Step 2: Run** `PYTHONPATH=. .venv-optimizer/bin/python -m pytest tests/test_trace_latent_refine.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `Endpoint`, `refine_anchor` and `project`.
- [ ] **Step 4: Run the same command.** Expected: PASS. If the 25-call slack fails, record the observed overshoot in the test comment and set the slack to it. Do not remove the bound.
- [ ] **Step 5: Commit** `feat(trace-latent): bounded L-BFGS-B refinement with canonical rescoring`.

### Task 6: Four-source batch proposal with pending-aware greedy selection (spec §8–§9)

**Files:**
- Modify: `trimem/optimizer/latent_plan_search.py` (add the batch proposer)
- Test: `tests/test_trace_latent_propose.py`

**Interfaces:**
- Consumes: Tasks 2–5. From `joint_search_controller` (import only): `recombine`, `frontier_parents`, `select_starts`.
- Produces:
  - `propose_latent_batch(space, codec, snapshot, measured: list[dict], *, agent=(), seed: int, round_index: int, q: int, trust: TrustState, budget: BatchBudget) -> dict`.
  - The return value is the `_run_round` contract plus diagnostics: `{"selected": [{"config","key","ell","snapshot","sources","plan","latent": dict | None}], "starts", "anchors": [{"key","source","u","status"}], "endpoints": [Endpoint as dict], "projections", "pool_count", "provenance", "rejected_agent", "fallbacks", "budget": {"seconds","calls","per_slot"}, "trust": asdict(trust), "codec": codec.identity, "seconds", "seed", "q"}`.
- Algorithm (fixed by spec §8; the signature does not determine it):
  1. Build the pool. `sobol = space.sample(256, seed=seed, avoid=forbidden)`. LLM = first 4 canonicalizable `agent` configs. `cross = recombine(space, frontier_parents(snapshot, measured), seed=seed+10000)` (≤64). Merge the three with provenance `sobol`/`llm`/`recombination`. `forbidden` = the measured keys.
  2. Select anchors. `starts = select_starts(space, snapshot, measured, pool, agent=llm, cross=cross, seed=seed)` (≤25). Drop a start if `codec.roundtrip` fails or if u ∉ [0,1]^d (log the reason). Refill from `space.sample(..., seed=seed+300000+1000*trust.resets)` until 25 anchors or until the draws run out. Merge the start configs into the pool (measured starts stay forbidden).
  3. For slot j in range(q): `snap_j = snapshot` if no picks yet, else `snapshot.with_pending(picks)`. If budget remains, visit the anchors in `anchor_order(...)` with `budget.anchor_share`. Call `refine_anchor` in `box(u_a, trust.h)`. Call `project`. Add ≤25 unique legal keys with source `latent`. `legal = pool − forbidden − picks − rejected`. If `legal` is empty, backfill with `space.sample(4, seed=seed+400000+j, avoid=…, max_draws=4096)` and record `fallbacks`. If `legal` is still empty, stop the batch and record the reason. Pick `argmax snap_j.score(legal)` with the key as tie-break (as `greedy_batch`).
  4. Mark each anchor that a slot did not visit because the budget ran out as `unvisited` in that slot's record.

- [ ] **Step 1: Write the failing tests** (tiny space, tiny model, 20 synthetic observations, `synthetic` label)

```python
def test_batch_valid_distinct_novel():            # q=4 -> 4 canonical, distinct, unmeasured keys
def test_final_batch_q2():                        # q=2 -> 2 picks
def test_endpoint_score_never_claims_slot(monkeypatch):
    # refine_anchor patched: endpoint ell_end=+1e9 decoding to config X; an explicit Sobol config Y has higher ell(u(Y))
    assert out["selected"][0]["key"] == key(Y)
def test_collapsed_endpoints_fill_from_other_sources(monkeypatch):  # all endpoints decode to one measured key
def test_second_pick_uses_pending_snapshot():
    assert out["selected"][1]["snapshot"] == snap.with_pending([out["selected"][0]["config"]]).identity
def test_roundtrip_failed_anchor_skipped(monkeypatch):
def test_out_of_domain_anchor_skipped(monkeypatch):
def test_budget_exhaustion_marks_unvisited():     # BatchBudget(seconds=None, calls=30) -> some anchors "unvisited"; 4 picks
def test_backfill_when_pool_nearly_exhausted():   # 572 of 576 measured -> <=4 picks, no measured key, fallbacks logged
def test_deterministic_with_call_budget():        # seconds=None -> identical selected keys across two calls
```

- [ ] **Step 2: Run** `PYTHONPATH=. .venv-optimizer/bin/python -m pytest tests/test_trace_latent_propose.py -q`. Expected: FAIL (`AttributeError: propose_latent_batch`).
- [ ] **Step 3: Implement** `propose_latent_batch` per the algorithm.
- [ ] **Step 4: Run** `PYTHONPATH=. .venv-optimizer/bin/python -m pytest tests/test_trace_latent_*.py -q`. Expected: all PASS.
- [ ] **Step 5: Commit** `feat(trace-latent): four-source pending-aware batch proposal`.

### Task 7: Search worker at the `_search_worker` seam (spec §7, §11)

**Files:**
- Create: `trimem/experiments/trace_latent_search.py`
- Test: `tests/test_trace_latent_worker.py`

**Interfaces:**
- Consumes: Tasks 2–6. From `trace_joint_plan_search` (import only): `get_space`, `aggregate_panel`, `atomic_json`, `analyzer_client`, `guarded_search`, `run_round`.
- Produces:
  - `BACKEND = "latent-mobo-v1"`.
  - `validate_latent_manifest(manifest) -> None`. It requires `search_backend == BACKEND`, no `pruning`, `selection == "greedy"` and `q == 4`. For a non-smoke run it also requires `initial_configs == 20` and `budget == 50`. It requires a `latent` block `{checkpoint, checkpoint_sha256, affine, affine_sha256, d, decode: "masked-greedy-v1", proposal_budget: {"seconds": 300, "calls": 10000}, anchors: 25, h0: 0.4}` whose sha256 values match the files.
  - `latent_search_worker(directory, manifest, entries, agent, seed, round_file) -> None`. It has the same signature as `_search_worker`. Phases go to `worker_phase.json` (`analyzer`, `gp`, `search`) with the same analyzer block as `_search_worker`. It replays `TrustState` from the earlier `round_*.json` files via `batch_gains`. It writes `round_file.json` (the Task 6 dict + `archive_keys`) and `round_file.pt` (`{"codec": identity, "gp_state": state_dict, "objective", "trust", "seed", "round_index", "sources": manifest["sources"], "latent": manifest["latent"]}`). On an exception it writes `round_file.error.json` and re-raises.
  - `install(module=trace_joint_plan_search) -> contextlib.AbstractContextManager`. It patches `_search_worker` and restores it on exit.

- [ ] **Step 1: Write the failing tests** (tiny space; checkpoint and affine saved in `tmp_path`; `smoke: True` manifest with 10 tasks, `initial_configs: 20`, `budget: 32`; the 20 initial panels are imported with `import_initial` from synthetic receipts)

```python
def test_rounds_reserve_and_finish(tmp_path):     # with install(): run_round x4 -> 4,4,4 picks; status budget_complete at 32
def test_round_file_contract(tmp_path):           # round json: archive_keys == complete keys; every pick has "sources"; .pt exists
def test_worker_refuses_changed_checkpoint(tmp_path):  # flip a byte after prepare -> RuntimeError from guarded_search, error.json names sha
def test_worker_refuses_pruning(tmp_path):        # manifest with pruning -> error.json
def test_replay_after_crash_reuses_round_file(tmp_path):  # round json present, no reservation -> run_round reserves the same keys, worker not spawned
def test_install_restores_original():
```

- [ ] **Step 2: Run** `PYTHONPATH=. .venv-optimizer/bin/python -m pytest tests/test_trace_latent_worker.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.** The child process of `guarded_search` (spawn) must import the worker by reference. Keep `latent_search_worker` a module-level function. Call `validate_latent_manifest` and `load_codec` before the GP phase, so that a mismatch fails within the 125 s startup limit.
- [ ] **Step 4: Run the same command.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(trace-latent): latent search worker behind the guarded_search seam`.

### Task 8: Shared 2×(5 Sobol + 5 LLM) initialization driver (spec §1, §5)

**Files:**
- Create: `trimem/experiments/trace_latent_init.py`
- Modify: `tests/latent_fixtures.py` (add `make_shared_init`)
- Test: `tests/test_trace_latent_init.py`

**Interfaces:**
- Consumes (import only): `trace_joint_plan_search.{validate_manifest, get_space, observations, initial_advice, evaluate_pending, atomic_json}`, `SearchLedger`, `ObjectiveProtocol`.
- Produces:
  - Test helper (add to `tests/latent_fixtures.py`): `make_shared_init(tmp_path, *, forward: bool = False) -> Path`. It runs the 4 stages on the tiny space with synthetic receipts and returns the run dir. Tasks 9, 15 and 16 use it.
  - `INIT_BACKEND = "shared-init-2x5s5l-v1"` and `STAGES = (("sobol", 0), ("llm", 5), ("sobol", 10), ("llm", 15))`. Each stage adds 5 configs and is entered when the complete count equals the second value.
  - `prepare_shared_init(directory, manifest) -> dict`. It replaces `prepare_run`. It sets `initial_configs=20`, `budget=20`, `new_configs=0`, `initialization=None` and `initial_keys=[]`. It reserves stage-A Sobol: `space.sample(5, seed=seed)`.
  - `run_init_round(directory, *, evaluator=None, agent=()) -> dict`. It has the same contract as `run_round` and takes `driver.lock`. Pending configs are evaluated first. At 5 or 15 complete configs, it writes `init_stage_{n}.json` (LLM proposal; Sobol fallback with seed+777+n) **before** reservation. A replay reads the file and never calls the LLM again. At 10 complete configs: `space.sample(5, seed=seed+1, avoid=10 keys)`. At 20 complete configs, it does three things once and idempotently: (a) freeze `objective.json` from all 20; (b) write `shared_init.json` `{keys, stages: [{stage, source, keys}]}`; (c) rewrite `manifest.json` `initial_keys` to the 20 keys only if the value is still `[]`. It then returns `{"status": "budget_complete"}`.

- [ ] **Step 1: Write the failing tests** (tiny space, synthetic receipts, `initial_advice` patched)

```python
def test_stage_order_and_sources(tmp_path):        # 4 stages -> 20 distinct keys; sources sobol,llm,sobol,llm
def test_llm_stage_persisted_before_reserve(tmp_path, monkeypatch):  # 2nd call after crash -> advice called once
def test_invalid_llm_falls_back_to_sobol(tmp_path, monkeypatch):     # 3 duplicates -> 3 "sobol_fallback"
def test_finalize_is_idempotent(tmp_path):         # extra run_init_round -> same objective.json digest, initial_keys unchanged
def test_second_llm_stage_sees_15_observations(tmp_path, monkeypatch):
```

- [ ] **Step 2: Run** `PYTHONPATH=. .venv-optimizer/bin/python -m pytest tests/test_trace_latent_init.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `prepare_shared_init` and `run_init_round`.
- [ ] **Step 4: Run the same command.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(trace-latent): shared two-stage 20-config initialization driver`.

### Task 9: Prepare wrapper for the shared init and the arms

**Files:**
- Create: `tools/prepare_trace_latent_campaign.py`
- Test: `tests/test_trace_latent_prepare.py`

**Interfaces:**
- Consumes: `tools.prepare_trace_joint_campaign` as `base` (patched attributes only); Task 8; Task 7 `BACKEND`, `validate_latent_manifest`.
- Produces:
  - CLI `shared-init --dataset D --template T --run-dir R [--memory-db M] [--panel random|stratified] [--seed 20261022]`. It patches `base.INITIAL=20`, `base.NEW=0` and `base.prepare_run=shared_prepare_run`. The latter sets `search_backend=INIT_BACKEND`, `selection="greedy"` and `holdout.selection="deferred"`, adds the new modules to `live.artifacts`, then calls `prepare_shared_init`.
  - CLI `arm --backend latent-mobo-v1|explicit-pr --shared-init R0 --template T --run-dir R [--checkpoint P --affine A]`. It patches `base.INITIAL=20`, `base.NEW=30`, `base.legacy_to_v060=lambda c: dict(c)` and `base.import_archive=arm_import_archive`. The latter sets `search_backend`, `selection="greedy"`, `holdout.selection="deferred"` and no pruning. For the latent arm it also adds the `latent` block and adds the checkpoint, affine and new modules to `live.artifacts`. It asserts that the arm's `space.identity` equals that of the shared-init run. Then it calls the original `import_archive`. The panel option must match the shared-init run.
  - Python functions: `shared_prepare_run(directory, manifest) -> dict`, `arm_import_archive(directory, manifest, source_dir) -> dict` (the backend comes from the closure set by the CLI) and `latent_block(checkpoint, affine, d) -> dict`.

- [ ] **Step 1: Write the failing tests** (no Postgres: call `shared_prepare_run`/`arm_import_archive` directly with a fixture manifest and a Task 8 fixture shared-init dir)

```python
def test_arm_import_keeps_forward(tmp_path):      # shared init with graph_direction="Forward" -> arm ledger has Forward
def test_arm_has_20_initial_keys_and_budget_50(tmp_path):
def test_arm_manifest_passes_validators(tmp_path):# validate_manifest + validate_latent_manifest
def test_arm_rejects_space_identity_mismatch(tmp_path):
def test_explicit_pr_arm_has_no_latent_block(tmp_path):
```

- [ ] **Step 2: Run** `PYTHONPATH=. .venv-optimizer/bin/python -m pytest tests/test_trace_latent_prepare.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** the wrapper. Write the module docstring in the same style as `prepare_trace_joint_campaign_stratified.py`: what is patched and why.
- [ ] **Step 4: Run the same command.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(trace-latent): campaign prepare wrapper for shared init and arms`.

### Task 10: Live dispatcher and interleaved arm scheduler (spec §12.4–§12.5)

**Files:**
- Create: `tools/trace_latent_live.py`
- Test: `tests/test_trace_latent_live.py`

**Interfaces:**
- Consumes: `tools.trace_joint_live.{main, evaluate, verify_artifacts, verify_memory, service_preflight, recover_claims, resource_snapshot, eligible_resources, ROOT}`; Task 7 `install`; Task 8 `run_init_round`.
- Produces:
  - `backend_patches(manifest) -> ContextManager`. `INIT_BACKEND` patches `trace_joint_plan_search.run_round`. `latent-mobo-v1` patches `_search_worker`. `explicit-pr` patches nothing. Any other value raises `ValueError`. The originals are restored on exit.
  - CLI `--run-dir R [--preflight-only]` runs `trace_joint_live.main()` under `backend_patches`.
  - CLI `--interleave R1 R2 --order-seed S` alternates rounds. At step k the arm order is `random.Random(digest([S,k])).shuffle`. Each arm round does `verify_artifacts`, `verify_memory`, `service_preflight`, `recover_claims`, then `run_round(evaluator=evaluate)` under that arm's patches. Before and after each round it appends `resource_snapshot()` and the arm/round to `<R>/interleave_log.jsonl`. It holds the shared `results/pool_evaluation_queue.lock` lease like `main`. It refuses to start while `pgrep -f train_trace_seq_vae` matches. After each arm reaches `budget_complete`, the arm drops out.

- [ ] **Step 1: Write the failing tests**

```python
def test_backend_patches_restore():
def test_unknown_backend_raises():
def test_interleave_order_deterministic():        # pure helper interleave_order(seed, step, arms)
def test_interleave_runs_until_both_complete(tmp_path, monkeypatch):  # fake run_round per arm; log has both arms each step
def test_refuses_during_vae_training(monkeypatch):# patched pgrep -> SystemExit with message
```

- [ ] **Step 2: Run** `PYTHONPATH=. .venv-optimizer/bin/python -m pytest tests/test_trace_latent_live.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `backend_patches`, `interleave_order` and the two CLI modes.
- [ ] **Step 4: Run** `PYTHONPATH=. .venv-optimizer/bin/python -m pytest tests/test_trace_latent_*.py tests/test_trace_joint_*.py tests/test_trace_pr_staged_pruning.py -q`. Expected: all PASS. This proves that the hashed paths still behave the same.
- [ ] **Step 5: Commit** `feat(trace-latent): backend dispatcher and interleaved arm scheduler`.

### Task 11: Representation readiness gates, offline G1–G3 rerun and the d-choice rule (spec §5, §12.2)

**Files:**
- Create: `tools/latent_readiness.py`
- Modify: `tools/latent_offline_gates.py` (add `--kind seq`; keep the `slot` path unchanged as the default)
- Test: `tests/test_trace_latent_readiness.py`

**Interfaces:**
- Produces:
  - `readiness(checkpoint: Path, space) -> dict` with the keys below. Validation = the checkpoint's own held-out set, re-derived as `space.sample(n_train+holdout, seed)[n_train:]`. Test = 20,000 fresh keys from `space.sample(..., seed=seed+1)` that exclude the train and validation keys, evaluated once. All exact-match values use the masked decoder (Task 1). The repaired-decoder value is reported next to it.
    - `R1_val_exact`
    - `R2_family_min` (family = (filter_order, graph_impl))
    - `R3_probe_valid` (10,000 Sobol u in [0,1]^d with a provisional `fit_affine(train)`; failures stay in the denominator)
    - `R4_collisions` (`scipy.spatial.cKDTree.query_pairs(r=1e-6)` over train ∪ val u)
    - `R5_test_exact`
    - `per_field_exact`, `probe_unique_rate`, `seconds`
    - `combination_holdout: "not established (uniform corpus)"`
  - `passes_readiness(r) -> bool`: `R1 ≥ 0.95`, `R2 ≥ 0.90`, `R3 == 1.0`, `R4 == 0`.
  - `tools/latent_offline_gates.py --kind seq --checkpoints … --runs … --out …`. It encodes with `LatentCodec` (provisional affine) and fits LOO with `fit_latent_model`, the same GP as live. It adds `pass` per d with this rule: G2 Spearman > 0 on every run, and on every run and every objective, latent LOO Spearman ≥ native − 0.10 and latent LPD ≥ native − 0.5 nats.
  - `choose_d(readiness: dict[int, dict], gates: dict[int, dict]) -> int | None`. It returns the smallest d whose checkpoint exists and that passes both `passes_readiness` and the gates. An absent checkpoint (for example a d64 OOM) is reported `missing`.
  - CLI `run --checkpoints …` writes `readiness.json`. CLI `choose` writes `chosen.json` `{d, reason, readiness, gates}`. CLI `freeze-affine --checkpoint P --shared-init R0 --out A` (used in Task 13) writes `fit_affine(train, init_20)`, the `R6_init_roundtrip` count and `checkpoint_sha256`.

- [ ] **Step 1: Write the failing tests**

```python
def test_choose_smallest_passing_d():   # {8: fail R1, 16: pass, 32: pass} -> 16
def test_choose_none_when_all_fail():   # -> None
def test_missing_checkpoint_reported():
def test_family_rate_on_fixture():      # hand-built hits per (filter_order, graph_impl) -> min rate
def test_collision_count():             # two identical rows + one distant -> 1
def test_gate_rule_uses_margins():      # latent spearman native-0.11 -> fail; -0.09 -> pass
```

- [ ] **Step 2: Run** `PYTHONPATH=. .venv-optimizer/bin/python -m pytest tests/test_trace_latent_readiness.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** the interfaces above.
- [ ] **Step 4: Run the same command.** Expected: PASS.
- [ ] **Step 5: Check that the training jobs have ended.** Run `pgrep -f train_trace_seq_vae`. Expected: no output. Then run `ls results/trace_latent_seqvae_20261104/seqvae_d*.pt`.
- [ ] **Step 6: Run the readiness gates on every existing checkpoint (CPU, 4 threads).** Run `PYTHONPATH=. .venv-optimizer/bin/python -m tools.latent_readiness run --space-run results/trace_pr_sciworld_v060_strat_20261104 --checkpoints results/trace_latent_seqvae_20261104/seqvae_d*.pt --out results/trace_latent_seqvae_20261104/readiness.json`. Expected: one entry per d with R1–R5.
- [ ] **Step 7: Rerun the G1–G3 gates for the sequence VAE.** Run `PYTHONPATH=. .venv-optimizer/bin/python -m tools.latent_offline_gates --kind seq --checkpoints results/trace_latent_seqvae_20261104/seqvae_d*.pt --runs results/trace_pr_alfworld_v060_20261103c results/trace_pr_sciworld_v060_20261103c --out results/trace_latent_seqvae_20261104/gates_G1_G3_seq.json`. Expected: a `pass` value per d.
- [ ] **Step 8: Choose d.** Run `PYTHONPATH=. .venv-optimizer/bin/python -m tools.latent_readiness choose --readiness results/trace_latent_seqvae_20261104/readiness.json --gates results/trace_latent_seqvae_20261104/gates_G1_G3_seq.json --out results/trace_latent_seqvae_20261104/chosen.json`. Expected: `{"d": <int or null>}`. **If d is null: stop and report the R/G numbers to the user.** Do not retrain. Do not fall back to PR (spec §5, §12).
- [ ] **Step 9: Commit** the code and the three JSON files: `feat(trace-latent): readiness gates and d selection for sequence VAE`.

### Task 12: Synthetic end-to-end smoke (spec §12.3)

**Files:**
- Create: `tools/latent_synthetic_smoke.py`
- Output: `results/trace_latent_smoke_20261104/smoke.json` (labelled `"synthetic": true`)

**Interfaces:**
- Consumes: Tasks 0–9 on the tiny space. The smoke uses the real `run_round`, `guarded_search` and ledger with the `synthetic_receipt` evaluator.
- Produces: `main(out: Path) -> dict` with the checks `{gp_three_models, fd_gradient_ok, rescoring_used, no_measured_or_pending_dup, batch_sizes == [4]*7 + [2], unique_configs == 50, proposal_budget_respected, restart_consistent}`. To check restart consistency, it kills the driver after `round_NNN.json` is written and before the reservation, then resumes. The resumed driver must reserve the same keys.

- [ ] **Step 1: Run** `PYTHONPATH=. .venv-optimizer/bin/python -m tools.latent_synthetic_smoke --out results/trace_latent_smoke_20261104`. Expected: FAIL (module missing).
- [ ] **Step 2: Implement** `main` with the checks above.
- [ ] **Step 3: Run the same command.** Expected: every check `true`. Reaching all checks `true` is not evidence that the method improves agent search (spec §12.3).
- [ ] **Step 4: Commit** `test(trace-latent): synthetic 20+30 smoke`.

### Task 13: Adversarial review before any live run

**Files:**
- Output: `docs/superpowers/plans/reviews/2026-11-04-trace-latent-mobo/{skeptic,architect,verdict}.md`
- Modify: `CHANGELOG.md` (add an entry: version-level feature)

- [ ] **Step 1: Make the diff file.** Run `git diff --no-index /dev/null <each new file>` together with `git diff -- tools/latent_offline_gates.py`, and write the result to `$SCRATCH/latent_mobo.diff`. Use the scratchpad directory.
- [ ] **Step 2: Invoke the `adversarial-review` skill.** Use the Skeptic and Architect lenses. The codex sandbox cannot read files, so put the spec and the diff in the prompt: `{ cat docs/superpowers/specs/2026-11-04-trace-latent-mobo-design_v0.1.0.md; cat $SCRATCH/latent_mobo.diff; } | codex exec -m gpt-5.6-terra --sandbox read-only "<lens prompt: check score contract, pending semantics, budget/unvisited accounting, hashed-file seams, Forward import, resume>"`. Expected: two non-empty review files.
- [ ] **Step 3: Fix the accepted findings.** Add one test per fix. Write `verdict.md` with an accept or reject decision for each finding.
- [ ] **Step 4: Run** `PYTHONPATH=. .venv-optimizer/bin/python -m pytest tests/test_trace_latent_*.py tests/test_trace_joint_*.py tests/test_trace_pr_staged_pruning.py -q`. Expected: all PASS.
- [ ] **Step 5: Add a `CHANGELOG.md` entry** `[latent-mobo-v1] - 2026-11-04` (Features / Design Rationale / Notes & Caveats; list the 9 resolutions from Global Constraints).
- [ ] **Step 6: Commit** `docs(trace-latent): adversarial review and changelog`.

### Task 14: Shared initialization campaign (live; needs explicit user go)

**Files:**
- Output: `results/trace_latent_init_<dataset>_<date>/` (new run directory)

- [ ] **Step 1: Ask the user for go.** Get the dataset (Q2), the panel (Q3) and permission to start agent profiling: 2,000 TRAIN episodes.
- [ ] **Step 2: Prepare.** Run `PYTHONPATH=. .venv-optimizer/bin/python -m tools.prepare_trace_latent_campaign shared-init --dataset sciworld --template results/trace_pr_three_dataset_prepare/template_v060.json --run-dir results/trace_latent_init_sciworld_<date>`. Expected: a JSON line, and `ledger.sqlite` with 5 reserved configs.
- [ ] **Step 3: Preflight.** Run `PYTHONPATH=. .venv-optimizer/bin/python -m tools.trace_latent_live --run-dir results/trace_latent_init_sciworld_<date> --preflight-only`. Expected: `{"status": "passed", ...}`.
- [ ] **Step 4: Start the run.** Start the same command without `--preflight-only` as a systemd unit in `trimem-eval.slice`, with environment servers from `scripts/env_servers.sh`. Expected: `status.json` reaches `test_deferred`, and `shared_init.json` lists 20 keys.
- [ ] **Step 4a: Run the agent smoke (spec §12.4) on stage A.** After the first 5 Sobol panels complete, check two things. (a) Every receipt has `config_key == space.key(config)`, the frozen `execution_fingerprint`, `quality_source == "environment_raw_score"` and an integer `agent_steps`. (b) The median latency per episode is within the range of the stage-A panels of `results/trace_pr_sciworld_v060_strat_20261104` for the same runtime profile. If either check fails, stop the unit and report.
- [ ] **Step 5: Freeze the affine and check R6.** Run `PYTHONPATH=. .venv-optimizer/bin/python -m tools.latent_readiness freeze-affine --checkpoint results/trace_latent_seqvae_20261104/seqvae_d<d>.pt --shared-init results/trace_latent_init_sciworld_<date> --out results/trace_latent_seqvae_20261104/affine_d<d>.json`. Expected: `R6_init_roundtrip == 20/20`. **If R6 is below 20/20: stop and report** (spec §5).

### Task 15: Arms and proposal preflight (spec §12.3 projection gate)

**Files:**
- Create: `tools/latent_proposal_preflight.py`
- Test: `tests/test_trace_latent_preflight.py`
- Output: `results/trace_latent_arm_<dataset>_<date>/`, `results/trace_pr_control_<dataset>_<date>/`, `<latent arm>/proposal_preflight.json`

**Interfaces:**
- Produces:
  - `preflight(run_dir: Path, *, seeds=(1, 2, 3)) -> dict`. On the frozen GP of the 20 imported observations, it runs one full-budget `propose_latent_batch` per seed with no reservation and no agent run.
  - Gate (spec §12.3): `pass = (#unmeasured legal unique configs with source "latent" across the seeds ≥ 4) and (some projection has ell_canonical > ell_start of its anchor + 1e-6)`.
  - Report fields: the delta distribution (`ell_canonical − ell_end`), the endpoint/unique ratio, the Spearman correlation of the pre- and post-projection ranks, and the best `ell_canonical` against the best ell of a legal Sobol pool with the same number of objective calls (capped at 4096). The preflight wall time is charged to the arm.

- [ ] **Step 1: Write the failing tests** (tiny space)

```python
def test_preflight_gate_counts_only_latent_unmeasured(tmp_path):
def test_preflight_fails_when_no_projection_improves(tmp_path, monkeypatch):  # project patched: ell_canonical = ell_start
def test_preflight_writes_no_ledger_rows(tmp_path):
```

- [ ] **Step 2: Run** `PYTHONPATH=. .venv-optimizer/bin/python -m pytest tests/test_trace_latent_preflight.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `preflight` and a CLI `--run-dir`.
- [ ] **Step 4: Run the same command.** Expected: PASS.
- [ ] **Step 5: Prepare the latent arm.** Run `PYTHONPATH=. .venv-optimizer/bin/python -m tools.prepare_trace_latent_campaign arm --backend latent-mobo-v1 --shared-init results/trace_latent_init_sciworld_<date> --template results/trace_pr_three_dataset_prepare/template_v060.json --run-dir results/trace_latent_arm_sciworld_<date> --checkpoint results/trace_latent_seqvae_20261104/seqvae_d<d>.pt --affine results/trace_latent_seqvae_20261104/affine_d<d>.json`. Expected: 20 complete configs and `objective.json` equal to the one in the shared-init run.
- [ ] **Step 6: Prepare the PR control arm.** Run the same command with `--backend explicit-pr --run-dir results/trace_pr_control_sciworld_<date>` and no checkpoint. Do this only if the user approved Q4.
- [ ] **Step 7: Run the proposal preflight.** Run `PYTHONPATH=. .venv-optimizer/bin/python -m tools.latent_proposal_preflight --run-dir results/trace_latent_arm_sciworld_<date>`. Expected: `proposal_preflight.json` with a `pass` value. **If `pass` is false: stop new profiling, keep the diagnostics and report.** Do not switch to PR. Do not tune until the gate passes (spec §12.3).
- [ ] **Step 8: Commit** the preflight code and the tests: `feat(trace-latent): proposal preflight gate`.

### Task 16: Interleaved live arms, TEST-200 and comparison report (live; needs explicit user go)

**Files:**
- Create: `tools/trace_latent_compare.py`
- Test: `tests/test_trace_latent_compare.py`
- Output: `results/trace_latent_compare_<dataset>_<date>/{compare.json,curves.csv}`; one sheet in `triMEM_optimizer/trimem_experiments.xlsx`

**Interfaces:**
- Produces: `compare(arms: dict[str, Path]) -> dict`. For each arm it reports the following, from TRAIN ledgers, round files, resource files and TEST ledgers. TEST is read last.
  - TRAIN NHV against unique profiles and against total search wall time (proposal + GP + LLM + profiling).
  - The raw TRAIN Pareto set: Q, tokens, latency and mean agent steps.
  - The spec TEST subset (sample-mean non-dominated) and the `trace_pr_test_now` union, each with TEST Q/tokens/latency/agent steps.
  - Latent diagnostics: projection deltas, latent-source selection rate, unique proposals per acquisition-second, fallbacks, unvisited anchors, failures.
  - Separate cost lines: representation training (`metrics.seconds` of the checkpoint), readiness, preflight, init, retries, TEST.
  - Flags for rounds whose `interleave_log.jsonl` shows contention.

- [ ] **Step 1: Write the failing tests** (fixture arms with synthetic ledgers)

```python
def test_nhv_curve_monotone_and_normalized():   # NHV = HV / 1.1**3 per unique-profile count
def test_spec_test_subset_is_sample_mean_pareto():
def test_agent_steps_reported_for_every_metric_row():
def test_test_rows_never_change_train_selection():
```

- [ ] **Step 2: Run** `PYTHONPATH=. .venv-optimizer/bin/python -m pytest tests/test_trace_latent_compare.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `compare`. It writes `compare.json`, `curves.csv` and one xlsx sheet (`TRACE latent vs PR`).
- [ ] **Step 4: Run the same command.** Expected: PASS.
- [ ] **Step 5: Ask the user for go to start the interleaved arms.** Then run `PYTHONPATH=. .venv-optimizer/bin/python -m tools.trace_latent_live --interleave results/trace_latent_arm_sciworld_<date> results/trace_pr_control_sciworld_<date> --order-seed 20261104` as a systemd unit in `trimem-eval.slice`. Expected: both arms reach `test_deferred` with 50 complete configs. The latent arm's `round_*.json` show batch sizes 4×7 and 2.
- [ ] **Step 6: Run TEST-200 for each arm.** Run `PYTHONPATH=. .venv-optimizer/bin/python -m tools.trace_pr_test_now --run-dir <arm>` once for each arm. Expected: a holdout ledger with complete 200-task panels for every selected key.
- [ ] **Step 7: Write the comparison.** Run `PYTHONPATH=. .venv-optimizer/bin/python -m tools.trace_latent_compare --arms latent=results/trace_latent_arm_sciworld_<date> pr=results/trace_pr_control_sciworld_<date> --out results/trace_latent_compare_sciworld_<date>`. Expected: `compare.json`, `curves.csv` and an updated xlsx sheet.
- [ ] **Step 8: Report to the user.** Label the result a single-seed feasibility case study (spec §12.5). Include agent steps with each metric. Do not claim that the method is superior.
- [ ] **Step 9: Commit** the compare tool and its tests: `feat(trace-latent): arm comparison report`.

---

## Open questions for the user (resolve before Task 14)

- **Q1:** The G2/G3 gate rule in Task 11 is my proposal: Spearman margin −0.10 and LPD margin −0.5 nats against the native mixed GP, on the ALF and Sci v060c archives. Do you approve it?
- **Q2:** Which dataset goes first? The plan uses ScienceWorld as an example.
- **Q3:** Which TRAIN panel: random fixed (spec §7, the default here) or stratified (the current PR campaigns)?
- **Q4:** Do you approve the new explicit-PR control arm (+3,000 TRAIN episodes + TEST) that shares the same 20 init?
- **Q5:** Which TEST set: the `trace_pr_test_now` union for both arms (the default here), or only the spec's sample-mean Pareto set?
- **Q6:** If no existing checkpoint passes Task 11: retrain to spec §3–§5 (2-layer, d=8, 10k corpus), or stop?
