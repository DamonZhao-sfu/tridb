# Math: seed SemFilter + SemRank execution plans (P1 / P2 / optimizer) — experiment plan v0.1.0

Date 2026-10-12. Branch `feat/trimem-semscan`. Requested by the user (verbatim intent): run joint
SemFilter + SemRank plans on math first (ALE paused), k = 10, hops = the task's mean lineage depth,
lineage + similar edges together, plus the optimizer branch as an arm that only enumerates execution
plans (no parameter choice); inputs and outputs must never be truncated.

## Tasks and readouts
heilbronn_convex_13, second_autocorr_ineq, heilbronn_triangle. Seed 42. Iterations 1-20 in
`results/trimem_v0.1.0/math_sem/it20/<cell>`, resumed from `checkpoint_20` to 50 in `.../it50/<cell>`;
best-so-far read at 10 / 20 / 30 / 40 / 50. Private nothing (math evaluators are exact).

## Retrieval (identical for every memory arm)
| knob | value |
|---|---|
| seeds | top-24 attempts of the task **that carry a change summary**, by cosine to the parent's code embedding (corpus only, `cell IS NULL`) |
| traversal | every hop expands the frontier along BOTH `lineage` (within session) and `similar` (cross-session kNN), fanout 8 per node per edge type, frontier DISTINCT per hop; pool = union of all hop frontiers |
| hops | task mean lineage depth: heilbronn_convex_13 = 4, second_autocorr_ineq = 4, heilbronn_triangle = 5 |
| telemetry | `d.is_valid AND d.fitness > p.fitness AND d.fitness >= parent - 0.2|parent|` (p = d's own parent) |
| k | 10 memories, all 10 inspiration slots (`--injection-rate 1.0 --num-diverse 10`) |
| injection form | LLM change summaries (`--inject-as changes`); rows without a summary are not retrievable |
| SemFilter predicate (on the SEED s) | `SEED_FILTER_PREDICATE`: "The change alters the method itself -- the algorithm, optimizer or search procedure, the objective or constraint formulation, the parametrization or representation, or the construction or initialization strategy -- rather than only tuning numeric constants or hyperparameters (step counts, learning rates, restart counts, tolerances), reformatting code, or changing input/output handling." |
| semantic cache | one per task shared by the five msem arms (`prompt_version = msem-<task>`): identical verdicts for P1 / P2 / opt; semantic cost compared by `Rows Judged` |
| SemRank (on the candidate d) | rank = (d - parent(d)) × (0.5 + sem_score("concrete transferable algorithmic idea", d.changes)) |

Every frontier row carries the id of the seed it was reached from, so filtering on the seed before
or after the traversal selects the same candidates: P1 and P2 are two physical plans of one logical
query (same injected rows), differing in traversal and semantic cost.

## Arms
| arm | plan |
|---|---|
| `msem-vgr` | Vector → Graph → Telemetry → Assemble (no semantic operator; reference) |
| `msem-rank` | Vector → Graph → Telemetry → SemRank(d) → Assemble (reference) |
| `msem-p1` | Vector → **SemFilter(s)** → Graph → Telemetry → SemRank(d) → Assemble (seeds pruned before traversal) |
| `msem-p2` | Vector → Graph → Telemetry → **SemFilter(s)** → SemRank(d) → Assemble (filter after traversal + telemetry; `trimem.force_placement = 'sa=top,d=top'`) |
| `msem-opt` | optimizer branch, plan enumeration only: per retrieval, `EXPLAIN` both P1 and P2 and execute the one with the lower TriMEM cost estimate (cache-coverage-aware semantic cost + traversal cost); decision logged per iteration |

3 tasks × 5 arms = 15 cells.

## No truncation (inputs and outputs)
* vLLM generator context raised to 262,144 tokens (the model maximum; previous math runs: 32,768).
* Output cap `max_tokens` = 32,768 for these runs (history on these tasks: max 5,192, 0 hits at 8,192).
* Input: 10 change summaries (≤ 1,121 chars each) + the current program + previous/top programs;
  history max prompt 24,561 tokens at k = 5 full code. Prompt guard `--prompt-budget-tokens 220000`
  (drops whole own programs, never cuts text) as a safety net only.
* Retrieved programs are injected as summaries, never cut (`code_chars` only bounds the SQL column that
  is not rendered); OpenEvolve does not truncate program code in prompts.
* Verification gate before launching the matrix: a 2-iteration smoke cell per plan must show
  `finish_reason = stop` (no `length`) and prompt tokens well below budget; after the runs the
  truncation counts (finish_reason = length, prompt > budget, dropped_for_budget) are reported per
  cell and must be zero.

## ALE settings recorded for the next ALE round (not run now)
`max_tokens` 32,768 (×4), context 262,144 (×2), prompt budget 220,000.

## Outputs
Cells under `results/trimem_v0.1.0/math_sem/`; summary page section and xlsx sheets; run telemetry
written back (`trimem.data.ingest_runs`).

## Addendum A (2026-10-12, before launch) -- SemFilter selectivity check
First launch stopped after iteration 1 because SemFilter removed (almost) every seed:

| task | seeds | roots (no summary) | kept, old predicate |
|---|---|---|---|
| heilbronn_convex_13 | 24 | 8 | 3-5 |
| second_autocorr_ineq | 24 | 24 | 0 (P1/P2 returned 0 rows) |
| heilbronn_triangle | 24 | 22 | 0 (P1/P2 returned 0 rows) |

Causes and fixes: (1) early parents are close to other sessions' initial programs, which have no
summary -> seeds now come from summary-bearing attempts only; (2) the ALE predicate names local-search
parts and rejects every gradient-optimisation change -> domain-neutral `SEED_FILTER_PREDICATE`;
(3) 502 corpus summaries were the placeholder "Full rewrite" (nodiff runs) -> re-derived from the diff;
(4) per-cell caches let borderline verdicts flip between cells (vLLM batch nondeterminism at
temperature 0), so P1 and P2 differed -> one shared cache per task. With the fixes, the seeds kept
are 22/24, 18/24 and 10/24.

## Addendum B (2026-10-12) -- same arms on ALE after math reaches 50
Arms `msem-{vgr,rank,p1,p2,opt}` on ahc008 / ahc011 / ahc025 (`scripts/ale_sem.sh`, out
`results/trimem_v0.1.0/ale_sem/it20` -> `it50`), started automatically when both math chains exit
(`scripts/after_math_run_ale.sh`). Hops = mean lineage depth 3 / 4 / 4. ALE settings from oe_matrix:
cpp, change summaries, 50 public cases, eval timeout 600 s, max_tokens 32,768, prompt budget 220,000.
Pre-launch: 863 ALE summaries (stubs + missing) re-derived (diff limit 60,000) -> every non-root attempt
has a summary; SemFilter keeps 17 / 11 / 14 of 24 seeds; smoke (ahc011, p1/p2/opt, 2 it): 10 rows,
P1 = P2 = opt rows, opt chose P1 (115 vs 2,202 rows judged), finish_reason = stop, max prompt 14,810.

## Addendum C (2026-10-12 20:11) -- OpenEvolve simplification nudge turned off for ALE
Compile-error analysis of 60 ALE children (local g++ -std=gnu++20, each undeclared identifier traced to
parent / diff / injected summaries): 0 identifiers came from memory; 30/60 children deleted a definition
still referenced (43 identifiers), 14 identifiers were helpers called but never added, 10 were wrong names,
11 children had syntax/type errors. Cause: OpenEvolve's default `suggest_simplification_after_chars = 500`
adds "Consider simplifying - code length exceeds 500 characters" to every ALE prompt (291/291).
Change: `run_arm --no-simplify-nudge` (receipt field `simplify_nudge`), on for every ALE cell via oe_matrix.
* ahc008 / ahc011 / ahc025: iterations 1-20 ran WITH the nudge; iterations 21-50 restarted from
  checkpoint_20 WITHOUT it (interrupted 21-50 runs archived in `ale_sem/archive_nudge_on/it50`).
* ahc024 / ahc026: restarted from iteration 0 WITHOUT the nudge (interrupted runs archived in
  `ale_sem/archive_nudge_on/it20`).
* Math runs are unchanged (nudge on, as before).

## Addendum D (2026-10-12 21:40) -- silent diff-block drops; fix staged for the next round
OpenEvolve 0.3.2 applies a SEARCH/REPLACE block only on an exact line match (trailing whitespace
included) and silently drops it otherwise. ALE seeds carry trailing spaces on ~20% of lines; 508 / 2,969
ALE blocks (17%) were dropped, 77% of them would match ignoring trailing whitespace. Children with a
dropped block: 36% compile errors vs 12% when every block applied. `run_arm --diff-ws-tolerant`
(`bench/agent_memory/gem_oe/diff_patch.py`, receipt `diff_match`) matches on `rstrip()`; replay: 83% ->
97% of blocks applied. NOT enabled for the current round (per the user); enable for the next round.

ahc008 no-gain analysis: the corpus beats this seed with one idea (single bottom wall per strip + larger
safe region, 3.67M -> 3.82-3.98M). 17 retrievable corpus memories score above the seed; across 181
ahc008 retrievals only 27 distinct memories were injected, 1 above the seed, none with that idea. For the
seed parent the telemetry pool has 55 rows: 9 of the 17 are reachable but rank 43-55 under the
gain = d - parent(d) key (big jumps from weak parents rank first); 8 are not reachable within 3 hops.
