# TriMEM downstream-aware plan selection — v0.3.0 (graph space, Δ_best target, O3)

Date 2026-10-12. Extends `docs/trimem_downstream_objective_plan_v0.2.0.md` (kept). Branch
`feat/trimem-semscan`. Motivation: the v0.2.0 arms (A1–A6, O1/O2) used 25–38% fewer tokens than
the best fixed plan (rank) but lost 1–7% quality at @20. Diagnosis (seed 42, iterations 1–20):

| policy | "inject nothing" chosen | mean k | child > parent |
|---|---:|---:|---:|
| fixed rank | 0% | 4.7 | 67% |
| O1 offline | 40% | 2.3 | 51% |
| O2 online warm | 26% | 2.9 | 53% |

1. *Wrong target*: Δ = improvement over the sampled parent; the reported quality is best-so-far.
2. *"Inject nothing" over-predicted*: one shared linear/hurdle model let early-iteration gains of
   the fixed-`none` cells (weak parents) leak into the zero-token candidate, which O1 favours most.
3. *Token-weighted objectives* trade quality for tokens by construction.

## Changes

| change | v0.2.0 | v0.3.0 |
|---|---|---|
| target | Δ = max(0, child − parent) | **Δ_best = max(0, child − run best at decision time)** (new-best probability × size) |
| "inject nothing" | same estimator as k > 0 | **separate estimator** on (parent fitness, progress, best − parent, parent tokens) |
| objective | O1 min Tok/ĝ, O2 utility − λ·Tok | **O3: min Tok(c) s.t. ĝ(c) ≥ ĝ(rank:5 @ lineage/down/1/8)** (tie → larger ĝ) |
| search space | family × k (13) | **graph config × family × k + none = 24 × 12 + 1 = 289** |

Graph configuration (`trimem/objective/space.py`), all native TriMEM Expand plan nodes:

| parameter | values | implementation |
|---|---|---|
| edge type | lineage (successful parent→child edits), similar (cross-session kNN, k=16) | stored edge types |
| direction | down, up | `lineage_up` / `similar_up` reverse edge types added to the graph store (182,400 edges) |
| hops | 1, 2 | 2 = two chained `trimem.expand` (union of both frontiers) |
| fanout | 4, 8, 16 | per-seed cap of each Expand |

Pool per configuration: top-8 rows by improvement over the row's own parent (equal to v1's
"over the seed" for lineage/down/1). One SemScore and one SemFilter pass over the union of the 24
pools (measured: 38 rows, ~10k tokens first time, cached afterwards; retrieval 100 ms). The
semantic tokens are charged to hard/rank candidates in proportion to their pool's share of the union.

## Estimation and training

* Labels: Δ_best. Rows logged before v0.3.0 have no recorded best; it is rebuilt from the session
  log (seed, then the best child of earlier iterations) — exact up to OpenEvolve's ~4-iteration
  concurrency.
* Offline prior: two hurdle models (memory / none), leave-one-task-out, from all logged decisions
  (fixed cells, v1 random, v1 A1–A6, v2 random). v1 rows enter the memory model with the default
  graph features (lineage/down/1/8).
* Online: two Bayesian linear models (memory / none), cold (zero mean, logged scaling) or warm
  (ridge fit of the prior's data); Thompson sampling; 10% exploration, uniform over the 13
  (family, k) choices then over the 24 graph configurations, propensities logged.

## Experiment (seed 42)

1. v2 random exploration, 5 tasks × 20 iterations (training data covering the graph dimensions).
2. Retrain the v2 priors (`trimem.objective.train --space v2`).
3. B1–B3 = O3 × {offline, online cold, online warm}, 5 tasks × 20 iterations.
4. Resume all v2 cells (with saved policy state) to 50 iterations.
5. Page: `triMEM_optimizer/it20_summary/trimem_it20_summary.html` (B0–B3 next to the fixed plans,
   v1 random and A1–A6; graph-parameter choice table).

The v1 A1–A6 resume to 50 was cancelled (superseded); their @20 results stay on the page.
