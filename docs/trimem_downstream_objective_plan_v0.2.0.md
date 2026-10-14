# TriMEM downstream-aware plan selection — implementation & experiment plan (v0.2.0)

Date 2026-10-11. Supersedes the discussion draft v0.1.0 (kept for history). Branch
`feat/trimem-semscan`. Agent: OpenEvolve 0.3.2 via `bench/agent_memory/gem_oe` + TriMEM.

## 1. Decisions taken (2026-10-11)

| question | decision |
|---|---|
| quality signal | **score improvement** Δ = max(0, f_child − f_parent) (combined_score is already normalised per task) |
| decision rule | test **both O1 and O2** (§3) |
| decision granularity | **every evolution iteration** |
| decision variables | **plan family × k** (joint); injection rate p not optimised (memory injected every iteration) |
| estimation modes | test **offline prior** and **online update** (cold and warm-started) |
| anchoring / diversity penalty | not decided → off by default, run as an ablation (λ ∈ {0, 0.1}) |

## 2. Per-iteration decision

At iteration t with parent π (code, fitness f_π):

1. **Candidate pool** — one tri-modal retrieval for π (ANN seeds → lineage Expand → telemetry
   filter), ordered by improvement, top `M = 16` rows: `P_t`.
2. **Semantic passes over the pool** (shared, cached, charged as retrieval cost):
   `sem_score(q, d.changes)` for every row of `P_t` (SemScore), `sem_filter(q, d.changes)`
   verdicts for every row (SemFilter). Cache hits are free; each pass is ≤ 16 rows.
3. **Candidates** `c = (family, k)`, family ∈ {tri-modal, hard filter, rank}, k ∈ {1, 3, 5, 8},
   plus `none` (k = 0, OpenEvolve's own population only) → 13 candidates.
   `O_c` = the top-k of `P_t` under the family's rule (tri-modal: improvement order; hard: rows
   with verdict true; rank: improvement × (0.5 + score)).
4. **Score** every candidate with the objective (§3) using the estimators (§4); pick
   `c* = argmax/argmin`; inject `O_{c*}` into k of OpenEvolve's inspiration slots.
5. **Observe** after the child is evaluated: (Δ, child valid, prompt/output tokens, generation
   latency, retrieval tokens/latency) → `trimem.outcome`; the online mode updates its model.

Placement of each semantic operator inside its pass stays with the existing cost model
(`J_impl`), because placement does not change O.

## 3. Objectives (both tested)

Notation for candidate c: `ĝ(c) = Ê[Δ | O_c, π]`, `T(c)` injected tokens (exact, §4),
`Ŝ(c)` predicted output tokens, `Tok(c) = T₀ + T(c) + Ŝ(c) + R_tok(c)` total tokens
(R_tok = semantic-operator tokens attributable to the family; shared passes split evenly over
the families that use them), `Lat(c)` predicted generation + retrieval latency.

* **O1 — cost per unit of expected improvement** (end-to-end cost form, extends §3 of the paper):
  `c* = argmin_c  [ α·Tok(c) + β·Lat(c) ] / max(ĝ(c), ε)`, with α = 1 token-unit, β = 0
  (tokens only) by default, β > 0 as a variant; ε = 1e-3 keeps "no improvement expected" finite
  but very expensive. Ties → smaller Tok.
* **O2 — utility minus normalised cost** (AgentQE / CA-RAG form):
  `c* = argmax_c  ĝ(c)/ĝ_max − λ_T·norm(Tok(c)) − λ_L·norm(Lat(c))`, where `norm` is min–max over
  the iteration's candidate set (AgentQE's group normalisation) and `ĝ_max` the largest ĝ in the
  set. Default λ_T = 0.2, λ_L = 0 (AgentQE uses 0.2 for its cost term); sensitivity λ_T ∈ {0.1, 0.2, 0.4}.
* **Ablation term** (both): `+ λ_A · Anchor(c)` (mean cosine similarity of O_c to the last 5
  parents), λ_A ∈ {0, 0.1}.

## 4. Estimators

* **T(c)** exact: add `attempt.tokens` (Qwen3.8 tokenizer, computed once per row; new rows at
  ingest); `T(c) = Σ_{o∈O_c} tokens(o) + per-item template tokens`.
* **ĝ(c)** — hurdle model: `ĝ = P̂(Δ>0 | x) · Ê[Δ | Δ>0, x]` (logistic regression + gradient
  boosted regressor, scikit-learn in `.venv-e0`), isotonic calibration of P̂.
  Features x(c, π): family one-hot, k, parent fitness f_π, iteration t/T, mean/max of
  (d.fitness − f_π) over O_c, mean/max improvement-over-own-parent in O_c, mean/max/gap of the
  cosine similarity to π, mean semantic score, fraction with verdict true, pairwise diversity of
  O_c, mean code tokens, fraction from other sessions. No task id (leave-one-task-out transfer).
* **Ŝ(c)** linear regression on (T(c), parent tokens, family) from llm telemetry.
* **Lat(c)** = Ŝ(c)·(s/token from telemetry) + measured retrieval latency of the passes.

### Modes
* **Offline prior (OP)** — fit once on logged data, frozen during the run.
* **Online cold (ON-C)** — Bayesian linear model per target (Thompson sampling for ĝ) with an
  uninformative prior; updated after every iteration of the session.
* **Online warm (ON-W)** — same, prior = the offline model's coefficients (Bayesian update from OP).
* Exploration: ON-* sample from the posterior (Thompson); ε = 0.1 uniformly random candidate in
  every mode to keep off-policy data unbiased (logged with propensities).

## 5. Data for the offline prior

The existing runs fix k = 5 and one family per cell, so they cannot identify the effect of k.
**Phase D1 collects an exploration dataset**: OpenEvolve with a uniformly random candidate
(13 choices, propensity 1/13) at every iteration, 5 tasks × 2 seeds × 20 iterations
(≈ 200 decisions), plus the 7 fixed-plan cells already running (≈ 700 decisions, k = 5 only).
Training uses leave-one-task-out: the prior used on task τ never saw τ.

## 6. Implementation

| phase | deliverable | files | done when |
|---|---|---|---|
| D0 | `attempt.tokens`; `trimem.outcome(session, iteration, candidate, features jsonb, propensity, chosen, delta, valid, tok_in, tok_out, gen_ms, ret_tok, ret_ms)` | `trimem/data/evotrace.py`, `src/trimem_pg/trimem_pg--0.2.0--0.3.0.sql` | table populated by a 2-iteration smoke run |
| D1 | pool + candidates + features + random policy in OpenEvolve; child-outcome hook (`GemMemoryDatabase.add` → `retriever.observe`) | `trimem/objective/{pool,features}.py`, `bench/agent_memory/gem_oe/trimem_adaptive.py` | exploration dataset of ≈ 200 decisions with outcomes |
| D2 | estimators (hurdle ĝ, Ŝ, Lat) + offline training + LOTO report (AUC of P̂, Spearman of ĝ vs Δ, calibration) | `trimem/objective/models.py`, `trimem/objective/train.py` | report in `results/trimem_v0.1.0/objective/estimators.json` |
| D3 | objectives O1/O2 + policies OP / ON-C / ON-W (+ε) with per-iteration receipt (all 13 candidates, scores, choice) | `trimem/objective/{objective,policy}.py` | unit tests on synthetic candidates; smoke run writes receipts |
| D4 | experiment matrix (§7), scheduler reuse (`trimem/experiments/oe_matrix.py`, new plan ids) | — | all cells complete |
| D5 | analysis + page/PDF (run2 style) + xlsx export + paper §8 extension | `trimem/experiments/render_compare.py`, `export_xlsx.py` | page, PDF and workbook in `triMEM_optimizer/` |

## 7. Experiments

Tasks: circle_packing, heilbronn_convex_13, second_autocorr_ineq, third_autocorr_ineq,
heilbronn_triangle. 20 iterations first (checkpoints @10, @20), then 100 iterations for the best
configurations. Seeds: 2 (42, 43), 3 for the final table.

| arm | policy | objective | mode |
|---|---|---|---|
| F-none / F-tri5 / F-hard5 / F-rank5 | fixed (existing cells) | – | – |
| R | uniform random candidate | – | – (exploration data) |
| A1 | adaptive | O1 | offline prior |
| A2 | adaptive | O1 | online cold |
| A3 | adaptive | O1 | online warm |
| A4 | adaptive | O2 | offline prior |
| A5 | adaptive | O2 | online cold |
| A6 | adaptive | O2 | online warm |
| ablations | best of A1–A6 with λ_A = 0.1; O2 with λ_T ∈ {0.1, 0.4}; O1 with β > 0 | | |

Metrics (all at the same iteration): best fitness @10/@20(/@100); cumulative tokens (retrieval +
generation); tokens per unit improvement; invalid-child rate; number of improving iterations;
wall-clock per iteration and planner overhead; choice distribution over (family, k) per
iteration; estimator quality online (predicted vs realised Δ); regret against the best fixed
arm per task.

Acceptance questions: (a) does an adaptive arm match the best fixed arm's fitness at lower
tokens (or exceed it at equal tokens)? (b) O1 vs O2? (c) does online updating beat the frozen
prior, and does warm start beat cold start in the first 10 iterations?

## 8. Risks

* Small data: ≈ 200 exploratory decisions per run; mitigate with a low-dimensional model and
  hierarchical pooling over tasks.
* Δ is sparse (most iterations do not improve) → hurdle model; report P̂(Δ>0) separately.
* Per-iteration semantic passes add retrieval cost even for unchosen families → charged in Tok;
  cached across iterations.
* OpenEvolve runs several iterations concurrently → outcomes arrive out of order; online updates
  apply when the child of iteration t is evaluated (logged with timestamps).
