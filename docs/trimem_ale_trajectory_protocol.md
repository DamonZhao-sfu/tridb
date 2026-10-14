# Latest-trajectory ALE continuation protocol

Default branch: `experiment-trajectory-retrieval-v4`.
Entrypoint: `.venv-e0/bin/python -m trimem.experiments.ale_trajectory_run --run`.
Coordinator authorization file is required before starting the durable waiter.

The coordinator waits for all four Math tasks × four retrieval plans to complete iteration 50 and for the Math supervisor to finish. Waiting starts no GPU model or semantic worker. A failed Math run is displayed as `blocked_math50` and polled for recovery. The ALE queue cannot start early.

Tasks: ahc008, ahc011, ahc024, ahc025, ahc026. Plans: VGR, SemRankOnly, P1, P2. SemRankOnly retains its current score-only behavior. Each cell runs five consecutive stages ending at 10/20/30/40/50. Later stages load only the exact preceding checkpoint and write fresh stage directories. Original stages are never overwritten. Each stage must contain exactly 10 generation dispatches with the expected absolute iteration indices.

Generation keeps the latest task-specific system/prompt, complete code plus intent, at most 4 inspirations, 16384-token inspiration budget, 32768-token full input, 32768-token output, thinking=false, seed 42. ALE uses C++, the existing 50-public-case scoring contract, 600-second evaluation timeout, no simplification nudge, and exact diff matching. The experiment copies each existing evaluator into its cell and changes only `WORKERS` from 8 to 1. Shared evaluators and their scoring code remain intact.

At each checkpoint, the private scorer reads that checkpoint's `best_program_info.json` and the exact winning program JSON. It checks iteration identity, forbids external memory programs and future iterations, verifies metrics, and records full source/hash. It never uses a glob for the latest checkpoint. Seed and public-selected program are evaluated by the existing `ale_private.score` / `Session.private_eval` with ALE-Bench Lite. Private outcomes are saved under `private/`, never fed into OpenEvolve, the trajectory store, prompts, or winner selection. Private failures are explicit missing metrics and do not alter public search. Private scoring may lag public stages and continues after public 50 until all checkpoint jobs finish.

There are at most 6 public cells, 1 evaluator worker per cell, and 1 private job with 4 workers. Public processes use a 12-CPU affinity pool and private scoring a separate 4-CPU pool when available, with recorded fallback on smaller hosts. Embedding instances alternate GPU 0/1. Generation uses the existing shared vLLM service. Private cache writes are serialized and isolated from earlier experiments. The semantic worker uses concurrency 8/max_take 8.

Data is under `results/trimem_v0.1.0/ale_trajectory_v4_20261013/`. Stage outputs are `itN/task__plan__s42/`. `summary.json` exposes per-cell checkpoint public quality, private raw score/rank/performance/delta, selected program identity, all stage paths, and semantic log paths. Token accounting sums actual API usage across non-overlapping stages. Private evaluations use no LLM tokens. Missing/ambiguous usage remains unavailable. `status.json`, `preflight.json`, and `manifest.json` preserve process and dataset provenance.

Read-only preflight checks ALE-Bench installation, rootless Podman API/images, model context window, cached per-task dataset archives, seeds/system/evaluator hashes, and the branch contract. Cached private-lite sizes are 200/300/150/500/150 cases respectively. Public evaluation continues to generate seeds 0..49, matching the existing experiment protocol rather than using only the five lite-public seeds.

Public seeds 0–49 are disjoint from each cached private-lite seed set. This checks seed identity, not byte-for-byte case equivalence. The private metric selector shows AtCoder performance, raw private score or performance change from the seed. Quality/token scatter plots retain separate task axes and mark partial token attribution.
