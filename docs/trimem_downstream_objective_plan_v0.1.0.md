# TriMEM downstream-aware optimisation objective — implementation plan (v0.1.0, draft for discussion)

Date 2026-10-11. Status: **discussion draft, objective not frozen.** Builds on
`docs/trimem_semscore_plan_v0.1.0.md` and `paper/sections/06_cost_model.tex`.

## 0. Where we are

`J_impl(ρ) = c_db(ρ) + Σ_q (D_q c_q + T_q c_trip)` prices retrieval only. It chooses the
placement of semantic operators inside one logical query (result-preserving, so the downstream
term is constant there). It cannot choose the plan family (tri-modal / hard filter / rank), the
context size k or the injection rate p, because those change the context O handed to the agent.

Measured downstream signal we already log (OpenEvolve × TriMEM, seed 42, partial, per generated
child): invalid rate (child score ≤ 0), improvement rate (child > parent), tokens per valid child.

| plan | invalid | improves parent | tokens / valid child |
|---|---:|---:|---:|
| nocontext | 64% | 18% | 8.1k |
| own population only | 47% | 31% | 9.3k |
| tri-modal | 7% | 66% | 18.3k |
| rank (SemScore) | 6% | 70% | 15.3k |
| hard filter (optimizer / sem-first / trav-first) | 22–28% | 48–57% | 11.3–15.0k |

Memory cuts invalid generations by ~6–10× but doubles tokens per call, i.e. exactly the
trade-off the objective must price.

## 1. Downstream quantities (definitions)

For a plan ρ with retrieved context O at evolution step t:
* `T(O)`   injected tokens (Σ stored token counts of the retrieved programs + template).
* `S_out`  output tokens of the agent call (regressed on T(O), task, parent length).
* `q(O)`   P(child is valid | O)   → expected attempts to a valid child `A(O) = min(1/q(O), R)`.
* `g(O)`   E[Δfitness⁺ | O] (expected positive improvement over the parent).
* `Anchor(O)` mean similarity of O to recent parents / archive (diversity guard, per Experience
  Graphs: heavy injection collapses the strategy space).

## 2. Candidate objectives (to discuss — not chosen)

| ID | form | inspired by | pros / cons |
|---|---|---|---|
| O1 | `min J_impl + A(O)·[c_in(T₀+T(O)) + c_out S_out] + λ·Anchor(O)` | our §3 objective + Experience Graphs | one scalar; needs price constants and λ |
| O2 | `max U(O) − λ_T·tokens − λ_L·latency`, `U = w₁·q(O) + w₂·g(O)` | AgentQE reward `0.5EX+0.3F1+0.2F1−0.2ρ`, CA-RAG `q̂ − w_L L̂ − w_T T̂` | matches the two papers; weights are policy choices |
| O3 | `min tokens per valid child = (T₀+T(O)+S_out)/q(O)` s.t. `q(O) ≥ q_min` | Experience Graphs' metric; FrugalGPT constraint | interpretable, no prices; ignores Δfitness |
| O4 | Pareto set over (q or g, tokens, latency), pick by SLO | Abacus, MOAR | best for the paper's figures; not a single decision |

## 3. Decision variables the planner would gain

plan family ∈ {tri-modal, hard filter, rank}, k ∈ {0,1,3,5,8}, injection rate p ∈ {0, 0.1, 0.5, 1},
plus the existing placement choice (still decided by J_impl, since it does not change O).

## 4. Estimators

* `T(O)`: exact at plan time — add `tokens int` to `attempt`, sum over the candidate ids
  (metadata-only pass of the retrieval query, no semantic operator execution needed for tri-modal;
  for semantic plans use the cached verdicts/scores, else the plan's expected rows).
* `q(O)`, `g(O)`: logistic / GBDT on features of O — k, similarity max/mean/gap, fraction valid,
  mean & max fitness of O vs parent, pairwise diversity, same-task share, plan family, semantic
  score mean. Calibrated (isotonic); use the lower credible bound of q for A(O) (1/q is convex).
  Training data: every logged (iteration, plan, O features, child valid, Δfitness, tokens) row
  from greedy + OpenEvolve runs (already in `triMEM_optimizer/trimem_experiments.xlsx`).
* Online: per (task, plan-arm) Beta posteriors / contextual bandit (LinUCB or Thompson) updated
  after every iteration with the observed outcome; ε share of k=0 arms kept for unbiased
  counterfactuals (Experience Graphs' p=0 baseline).

## 5. Feedback loop (answer to "does each iteration feed the next?")

Two layers, both proposed:
1. **Offline prior** (AgentQE-style): models fitted on logged runs, loaded at plan time.
2. **Online update** (bandit, per session): after iteration t, the harness writes
   `(plan, O-features, valid, Δfitness, tokens, latency)` into a `trimem.outcome` table in the same
   database; the planner's estimates for iteration t+1 read the updated posteriors. So yes — each
   iteration's realised downstream cost and quality steer the next decision, while the prior
   keeps early iterations sensible.

## 6. Implementation phases

| phase | deliverable |
|---|---|
| D0 | `trimem.outcome` table + harness logging (OpenEvolve and greedy); `attempt.tokens` |
| D1 | offline dataset builder from logs; fit + calibrate q̂, ĝ, Ŝ_out; report AUC / calibration on held-out runs |
| D2 | `trimem.choose_plan(...)` (Python first, SQL function later): enumerate (family, k, p), evaluate T(O) via metadata query, score with the chosen objective, return the plan; EXPLAIN-style receipt of every candidate |
| D3 | online bandit updates from `trimem.outcome`; ε-exploration |
| D4 | experiments: fixed-plan baselines vs adaptive objective (O1–O4 ablations), ≥3 seeds, matched budgets; metrics: tokens per valid child, invalid rate, steps-to-threshold, best fitness, #distinct strategies |
| D5 | paper: Related Work (DB optimizers + RAG/routing/memory) and Cost Model §8 extension |

## 7. Open questions (for discussion)

1. Quality signal: validity only, Δfitness, or reaching a task threshold?
2. Scalar (O1/O2) vs constrained (O3) vs Pareto (O4) as the planner's decision rule.
3. Decision granularity: per iteration, per session, or per task family?
4. How strongly to penalise anchoring (λ), and how to measure strategy diversity.
5. Does the optimizer own k / p / family, or does the user fix the family and the optimizer only k / p?
