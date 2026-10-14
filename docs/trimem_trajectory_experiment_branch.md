# Default code branch for trajectory experiments

The user designated `experiment-trajectory-retrieval-v4` as the default code branch
on 2026-10-13. Use this branch for subsequent trajectory retrieval experiments.
The tracked policy is `trimem/experiments/defaults.json`. Each experiment manifest
records the actual branch, commit and relevant source hashes. A dirty workspace
can contain unrelated local research artifacts and must not be cleaned implicitly.

## Current experiment

Four tasks: convex 13, second autocorrelation, Heilbronn triangle, circle packing.
Four plans: VGR, SemRankOnly (score-only), P1 and P2. Six parallel cells, task-priority
scheduling, seed 42, thinking disabled. At most four retrieved full-code + intent
programs, 16384 inspiration tokens, 32768 full input tokens, 32768 output tokens.

Complete all 16 task-plan cells at iteration 20 first. Then resume each saved
checkpoint into a fresh continuation output directory, generating iterations 21–50.
Save checkpoints at 30, 40 and 50. Existing first-stage results stay immutable.

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate base
.venv-e0/bin/python -m trimem.experiments.math_trajectory_extend --preflight
```

The extension coordinator exposes its CLI with `--help`. Do not start a second
coordinator if its lock/status indicates an existing live run. Its output root is
`results/trimem_v0.1.0/math_trajectory_extension_20261013/`, with a data interface
mapping each task-plan to stage20 and continuation directories. The dashboard is
`triMEM_optimizer/it20_summary/trimem_it20_summary_en.html`.

## Code and data boundaries

The branch stores the TriMEM implementation, its local execution dependencies,
experiment/rendering code, tests, plans and configuration. Model checkpoints,
PostgreSQL data directories, external corpora and generated experiment logs remain
local inputs/outputs. Set up the existing database/model service and Python
requirements before reproducing. Tests run here do not certify the GX10 ARM fork.
