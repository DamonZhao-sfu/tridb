# TRACE latent MOBO design review — 2026-11-04

## Intent
Deliver a reviewable design for joint plan/parameter encoding, legal decoding and continuous-acquisition proposals for multiobjective TRACE search. No implementation, model training, profiling or RL changes were authorized by this design task.

## Review execution and limitations
Used the `adversarial-review` skill and actual `codex exec` CLI calls, with Skeptic and Architect lenses plus builder questions. Initial inherited model was unavailable to the local account; subsequent filesystem-mode review hit a bubblewrap networking restriction. Those attempts were not counted as successful reviews. Retried using the locally advertised `gpt-5.6-terra` model and the complete draft supplied through stdin, without tool access. Both resulting review files are nonempty and stored alongside this verdict.

The skill's `brain/principles.md` and linked principle files were absent after filesystem search. Reviewers explicitly acknowledged this and applied the supplied lens text; no full-principles compliance is claimed. Reviews were of the supplied design, not independent runtime verification. The primary agent inspected the relevant optimizer source separately.

## Verdict
**Initial draft: REVISE. Revised design: ready for user review by lead judgment.** No claim of implementation readiness or empirical superiority. The revised draft has not received a second independent review; accepted issues were incorporated and checked by the lead.

## Findings and lead judgment

1. **High — surrogate/projection gap (both lenses). Accept the operational concern.** A gradient optimum of ell(u) need not be good after decoding. §§1,6,8,12 now explicitly call this relaxed continuous candidate generation, use canonical rescoring, and require proposal preflight with unique legal outputs and post-projection improvement over anchors. The design never asserts exact optimization of the discontinuous decoder-composed acquisition. Whether this improves real search remains an experiment question; silently switching to PR is prohibited.
2. **High — objective/noise coordinate ambiguity (Skeptic). Accept.** §7 specifies raw→transformed→normalized values, variance propagation and BoTorch Standardize's internal/output behavior. It identifies the population-expectation estimand and acknowledges correlated common-panel noise approximated by diagonal variance. Finite-panel noise is not falsely described as execution-repeat noise.
3. **High — pending semantics (Skeptic). Accept clarification.** §8 explicitly supplies canonical coordinates via qLogNEHVI X_pending; no synthetic labels or key-only exclusion substitute for posterior treatment. Restart reconstructs pending from durable state.
4. **Medium — TEST scope and crash windows (Skeptic). Accept.** §§11–12 specify all complete TRAIN Pareto points, held-out TEST-200, separate TEST cost, a durable evaluation state machine, uncertain submission handling, idempotent receipts, failure/retry accounting and unique-configuration charging.
5. **High — single seed cannot establish superiority (Architect). Accept reporting limit; do not expand the user's budget.** §12 now labels the first comparison a single-seed feasibility case study. No repeated expensive campaign was launched or silently added.
6. **Medium — latency comparability and compute-order bias (Architect). Accept.** §§8,12 specify seeded anchor scheduling with per-anchor budgets and telemetry, shared execution/caching/concurrency contracts, arm interleaving and handling of external contention without selective deletion.
7. **High — all future observations must dominate the fixed reference (Architect). Reject as a mathematical requirement.** Hypervolume above a fixed reference gives zero contribution to points not better than it in all objectives; such observations can still train the GP. BoTorch supports baseline pruning by posterior probability of being Pareto-optimal *and better than the reference*. The reference need not be worse than every future observation. §7 now makes the contributing region explicit, preserves bad/negative data and freezes reference across arms. A reference can emphasize some tradeoffs over others; this is a reporting/policy limitation, not an invalid-HV condition. Evidence: [BoTorch hypervolume and NoisyExpectedHypervolumeMixin documentation](https://botorch.readthedocs.io/en/stable/utils.html).

## What was retained
- Canonical config/evaluation identity remains independent of representation version.
- Constrained deterministic decoder separates legality from learned geometry.
- Three measured-metric GPs, pending-aware acquisition, and projection rescoring have distinct interfaces.

## Remaining user-review choices
Suggested defaults, not approved runtime changes: latent dimension 8 (16 fallback), 10k reconstruction corpus (50k cap), readiness thresholds, moving-box policy and proposal compute limits. The user's selected BO direction and prior 20+30/100 TRAIN/200 TEST protocol are preserved. Actual runtime capability must still be audited before training or profiling. No user-only ambiguity prevents reviewing this design.
