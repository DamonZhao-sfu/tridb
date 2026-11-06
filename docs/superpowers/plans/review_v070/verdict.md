# Adversarial review — TRACE v0.7.0 (reviewer_cli=codex; skeptic, architect, minimalist; 2026-11-05)

## Verdict: REJECT (high findings with reviewer consensus)

## Findings → Lead judgment and required fix
1. [high, all 3] Latency pruner runs as an unsupervised async process, judges arbitrary completed subsets
   (not the stored 20-task prefix), races the staged barrier (can raise "Stage incomplete" or yield timing-dependent
   partial panels), and the queue never checks it is alive. ACCEPT. Fix: move the latency-cap rule INTO
   `staged_evaluate_v070` at each stage checkpoint (same barrier, exact own prefix of the stage size, front =
   complete never-pruned configs ∪ NoMemory, factor per dataset 2/3/2, init and NoMemory protected); log with the
   staged decisions; remove the separate pruner process from the queue (keep the tool only as an offline/dry-run
   reporter). This also removes findings "queue does not supervise pruner".
2. [high, all 3] Router candidates vs TEST set mismatch. ACCEPT; superseded by USER decision 2026-11-05: router + OneConfig candidates = BRANE fuzzy-Pareto set (∪ NoMemory); TEST = 3-obj Pareto ∪ fuzzy set ∪ NoMemory.
   USER decision 2026-11-05 ("还是在pareto point上训练"): router + OneConfig candidates = the TEST-measured set
   (TRAIN sample-mean Pareto ∪ NoMemory); refuse to route if any candidate lacks complete TEST receipts.
3. [high, architect] Pruning decision logged before `prune_many`; resume recomputes. ACCEPT (lower after fix 1):
   commit prune_many first, then log; on resume, a decision for (key, stage) already applied is not recomputed;
   decisions are a pure function of the ledger state at the barrier.
4. [high, minimalist] Arm prepare publishes the manifest before importing NoMemory; an interrupted prepare looks
   complete. ACCEPT: write `prepared.json` completion marker last; queue checks the marker, not the manifest;
   prepare is replayable/idempotent when the marker is absent.
5. [medium, all 3] Hash guard silently skips missing manifests / missing snapshot. ACCEPT: fail closed when a guarded
   manifest or the before-snapshot is missing; discover protected runs (every results/*/manifest.json with
   live.artifacts whose run is not v0.7.0) instead of a hard-coded list, plus the explicit list as a floor.
6. [medium, architect] Runtime parity not enforced. ACCEPT: validate_manifest_v070 enforces task_concurrency 16,
   agent max_tokens 4096, same model/vLLM bases and env servers for NoMemory and TRACE, latency_front true.
7. [medium, skeptic] Holdout selection written before acquiring driver.lock. ACCEPT: take the lock first.
8. [medium, skeptic] A rejected second launcher writes status "stopped" into the owner's run. ACCEPT: only the
   lock owner writes status; lock failure exits without touching status.

## What went well
No reviewer found a leak of v0.7.0 patches into hashed files, a NoMemory path into the GP/budget, or a sample
mis-assignment; per-config sampling, ledger and schema were not challenged.
