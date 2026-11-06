# Goal prompt — TRACE latent-MOBO v2: VAE v2 + NumPyro SAAS (parallel, 512/256) end-to-end on three datasets

> Queued by the user on 2026-11-04 (~20:00). Execute top to bottom. Binding constraints:
> `.superpowers/sdd/2026-11-04-trace-latent-mobo-implementation/{global-constraints.md,user-decisions.md}`.
> Answer the user in Chinese, STE style, concise. No git commits.

## User decisions this goal implements (verbatim intent)

- "训练完之后 用新的 vector F 方案再测试一下在三个数据集上是否拟合达标" — after VAE v2 training, run the
  G1–G3 gates with surrogate F (SAAS, NumPyro NUTS 512/256/16, the 3 objectives fitted in parallel) on
  ALFWorld, ScienceWorld and WebShop.
- "那之后就决定用并行拟合的 NumPyro SAAS 512/256 + VAE v2 来进行 end-to-end 的三个数据集的探索采样(50 预算)
  + 测试集测试" — then run the latent arm end to end on all three datasets: 10 shared init + 40 explored
  (10 batches × q=4), budget 50, then TEST on the TRAIN sample-mean Pareto set.
- "测试环境要和之前的配置一样" — identical runtime profile to all historical TRACE-PR runs: c16 TRAIN and
  TEST, the same agent model/vLLM, AGENT_MAX_TOKENS 4096, the same memory DBs
  (`TRIMEM_{SCIWORLD,ALFWORLD}_MEMORY_DB=trace_ablation_sci_48d26f8d18eba20f`,
  `TRIMEM_WEBSHOP_MEMORY_DB=trace_ablation_web_22467f614f8be112`), the same fixed TEST panel
  (`triMEM_optimizer/test_panels/fixed_test_v1*`), the TRAIN panel stratified by the TEST task-type mix,
  `trimem-eval.slice`, the same env servers (`scripts/env_servers.sh`). Never compare latency across regimes.
- Earlier decisions still in force: dataset order ALF → Sci → WebShop; TEST = TRAIN sample-mean Pareto
  (`holdout.selection="train_pareto"`, run automatically at budget completion); no pruning in latent arms;
  pause WebShop for GPU VAE training and resume right after; never modify files hashed in a live run.

## Step 1 — VAE v2 (in progress when this goal was queued)

1. Wait for the v2 implementer (task-vae-v2-brief.md / task-vae-v2-report.md). Review its diff (task reviewer,
   SDD rules); fix loop until approved.
2. Pause WebShop: `systemctl --user stop trace-pr-strat-chain-20261104`; append
   `PAUSED by user (GPU for VAE v2 training)` to `results/trace_pr_three_dataset_strat_20261104/chain_status.txt`.
   Confirm no `trace_joint_live` process remains and GPU utilisation drops.
3. Train v2 d16/d32/d64 on GPU, one job per GPU (≈3.5 GB free each; never two per GPU), with the command
   line from the v2 report, output `results/trace_latent_seqvae_v2_20261104/seqvae_v2_d{16,32,64}.pt/.json`.
   Use `ps … | grep "[t]rain…"` patterns only (never pkill/pgrep that matches the shell).
4. Resume WebShop immediately after the last training job ends: start the chain again with
   `systemd-run --user --unit=trace-pr-strat-chain-20261104 --slice=trimem-eval.slice
   --property=StandardOutput=append:<strat_chain.log> --property=StandardError=append:<strat_chain.log>
   --working-directory=/localhome/hza214/tridb results/trace_pr_three_dataset_strat_20261104/run_strat_chain_resume.sh`
   (same transient unit as before; the driver resumes from the ledger). Verify status.json returns to running.
5. Report to the user: held-out exact reconstruction, order metrics (aux Spearman per numeric field,
   collapsed dims), training time.

## Step 2 — readiness + F gates on v2 (CPU, eval slice)

1. `tools.latent_readiness run --space-run results/trace_pr_alfworld_v060_strat_20261104 --checkpoints
   results/trace_latent_seqvae_v2_20261104/seqvae_v2_d*.pt --out …/readiness.json`.
2. Snapshot the WebShop strat ledger read-only (`sqlite3 … ".backup …"` + manifest + objective) — refresh the
   snapshot (more configs now).
3. Gates: `tools.latent_offline_gates --kind seq --variants F --checkpoints …/seqvae_v2_d{16,32,64}.pt --runs
   results/trace_pr_alfworld_v060_20261103c results/trace_pr_sciworld_v060_20261103c <webshop snapshot>
   --out …/gates_F_v2.json` with `TRIMEM_JAX_CPUS=12`. Run all three d (no stop-at-first-pass) so the user sees
   the full table; also report the v1 F numbers next to them (gates_F_parallel_d16.json).
4. Choose d: smallest d that passes readiness (R1 ≥ 0.95, R2 ≥ 0.90, R3 = 1.0, R4 = 0) and the F gates on all
   three datasets. If none passes all three, use the d that passes readiness with the highest mean CV LPD over
   the three datasets and objectives, and mark the choice "gates not passed" in the report (the user decided
   to proceed with F + v2). If no d passes readiness R3/R4 (decode validity / collisions): STOP and report.
5. Report the gate table (per dataset × objective, LPD/Spearman vs native, fit seconds) to the user.

## Step 3 — wire NumPyro SAAS into the live path (code; SDD review)

1. `fit_latent_model(..., kernel="saas-numpyro")` → `latent_numpyro_saas.fit_saas_numpyro` (parallel=True,
   512/256/16, compilation cache + warm start if those landed and passed equivalence). The snapshot must
   return mixture moments for posterior consumers (same as the Pyro SAAS fix in Task 12).
2. `validate_latent_manifest` accepts `surrogate.kernel = "saas-numpyro"`; `prepare_trace_latent_campaign`
   `--kernel saas-numpyro` writes it and adds `latent_numpyro_saas.py` and `latent_seq_vae_v2.py` to
   `live.artifacts` (the `latent_*.py` glob may already do this — verify).
3. CPU pinning: the live worker must pin JAX explicitly (opt-in env set by the live dispatcher, e.g.
   `TRIMEM_JAX_CPUS=12`), never as a hidden import side effect for torch threads.
4. The worker's `{"phase":"gp"}` refresh before each fit must stay under the 125 s guard (parallel fit = one
   phase; measured ≈10–15 s).
5. Rerun the synthetic smoke with `--kernel saas-numpyro` and the chosen v2 checkpoint on the real space:
   all 8 checks true. Fix the Task 12 minor first: `rescoring_used` must not pass silently when the refit is
   not bit-identical (NumPyro).
6. Task 13: adversarial review (codex, inline diff) of the whole latent branch + CHANGELOG entry; final
   whole-branch review on the most capable model. Then tell the user the code is done.

## Step 4 — end-to-end runs (queued behind the current strat chain)

For each dataset in order ALF → Sci → WebShop:
1. Shared init (10 = 5 Sobol + 5 LLM, `sobol_then_llm`), stratified TRAIN panel, same seed (20261022):
   `tools/prepare_trace_latent_campaign.py shared-init --dataset D --template <the strat run's template>
   --run-dir results/trace_latent_v2_<D>_init_20261104 --memory-db <same as strat>`.
   Ruling allowed: if `arm` can import the 10 init configs of the existing stratified TRACE-PR run
   (`results/trace_pr_<D>_v060_strat_20261104`, same panel and space digest — the importer checks), import
   them instead of re-evaluating, so the latent arm and the TRACE-PR strat run share the same init; log the
   choice. Otherwise run the shared init.
2. Latent arm: `arm --backend latent-mobo-v1 --kernel saas-numpyro --structure independent --checkpoint
   <chosen v2> --affine <frozen affine>` where the affine is `tools.latent_readiness freeze-affine
   --checkpoint <chosen> --shared-init <init run> --out …/affine_<D>.json` (report R6 init round trip).
3. Run with `tools/trace_latent_live.py --run-dir <arm>` (or `--interleave` only if a control arm is added);
   budget 50, q 4, c16; TEST = train_pareto runs automatically at budget completion.
4. Queue: one transient service `trace-latent-v2-e2e-20261104` in `trimem-eval.slice` whose script waits until
   `results/trace_pr_three_dataset_strat_20261104/chain_status.txt` contains `all done` (or the strat chain
   unit is inactive), then runs the three datasets in order; on any failure write the status and stop.
   Status file: `results/trace_latent_v2_e2e_20261104/queue_status.json`.
5. Control: the existing stratified TRACE-PR runs (same panel; pruning differs — state this). An explicit-PR
   control arm with the same shared init is approved but NOT queued by default (it doubles GPU time); offer it.

## Step 5 — results

Per dataset: TRAIN Pareto (configs, sample means), TEST metrics of the Pareto configs (quality as success rate
AND reward for WebShop, tokens, latency, agent steps — ALF/Sci environment_steps, τ²-style agent_calls where
applicable), HV on TEST vs the TRACE-PR stratified run, source mix (latent / Sobol / LLM / recombination),
per-round fit and proposal seconds. Plot data → CSV + one sheet in `triMEM_optimizer/trimem_experiments.xlsx`.
Update the "choose curve points" HTML only if the user asks.
