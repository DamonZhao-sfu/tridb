# TriMEM downstream-aware plan selection — v0.4.0 (mixed path, O4 latency objective)

Date 2026-10-12. Extends v0.3.0. Decision (user, 2026-10-12): weighted objective with λ_L = 0.1;
do not penalise input tokens; penalise output tokens and LLM inference time; add a cross-session
mixed traversal path; no `mixed_up` for now.

## Why v0.3.0 (O3) under-performed rank

* O3 minimised Tok subject to ĝ ≥ ĝ(rank:5). Base prompt and output length were run-level
  constants, so only the injected (input) tokens separated candidates: O3 picked the fewest
  injected tokens (k ≈ 1.2, "inject nothing" ≈ 25% online).
* Input tokens barely cost time. Joined telemetry (2,322 requests):
  `seconds ≈ 2.58 + 0.00018·prompt + 0.0292·output`, R² 0.85; corr(prompt, output) = 0.18.
  B2/B3 cut prompts by ~58% vs rank yet took the same time per request, and lost 5–10% quality.
* ĝ ranks candidates weakly (Spearman ~0.3), and Thompson samples with τ² = 1, σ² = 0.01 dwarfed
  Δ_best (~0.03), so cheap candidates passed the constraint by noise.

## v0.4.0

**O4 (weighted, λ_L = 0.1)**
`score(c) = ĝ(c)/ĝ_ref − 0.1 · L̂(c)/L_ref`, ref = rank:5 on lineage/down/1/8 (same iteration).
`L̂(c) = (a + b·prompt(c) + d·Ŝ_out(c)) / max(P̂(valid | c), 0.2)` = expected inference seconds per
valid child. Ŝ_out and P̂(valid) are predicted per candidate (gradient boosting / calibrated logistic
on the candidate features); a, b, d from the latency regression. Input tokens enter only through b.

**Mixed path** — edge `mixed` = two chained TriMEM Expand nodes: `similar` (kNN into another
session) then `lineage` (that session's successful children); the pool is the second frontier.
Fanout 4/8/16 → 27 graph configurations, 325 candidates.

**Online models** — prior and noise scaled to the data: τ² = σ² = Var(Δ_best) (measured 0.0123).

## Experiment (seed 42)
C0 random exploration (v3 space, 20 it) → train v3 prior (Δ_best estimators + output-token,
validity and latency models; leave-one-task-out) → C1–C3 = O4 × {offline, online cold, online warm}
20 it → resume C0–C3 to 50 → page + workbook. Page hides random arms (R, B0, C0) and A1–A6.
