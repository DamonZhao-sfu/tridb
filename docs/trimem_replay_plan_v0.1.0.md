# TriMEM end-to-end replay: plans vs task quality and phase cost (v0.1.0)

Date 2026-10-11. Branch `feat/trimem-semscan`. Builds on `docs/trimem_semscan_plan_v0.1.0.md`.

## Question

How do TriMEM execution plans (retrieval placement of the semantic filter, eager vs deferred
construction) change (a) downstream agent task quality and (b) the latency and input/output
tokens of construction, retrieval and generation?

## Current optimisation target (answer to "does it include downstream cost?")

No. The planner minimises PG cost units = c_db (native V/R costs, Expand degree model) +
LLM cost of semantic operators (distinct inputs × c_eff + per-batch round trips). The paper's
third term E[#attempts | O]·c_agent (P5 estimator, `bench/agent_memory/gem/downstream.py`) is not
wired into the planner. This experiment measures the downstream effect the planner ignores.

## Setup

* Memory: `trimem_evo` (EvoTrace: 10,672 attempts, code embeddings with HNSW, native `lineage`
  and `similar` edges, telemetry). Code attached for 4 math tasks. Each replay cell writes its own
  attempts with `cell = <cell id>`; reads see `cell IS NULL OR cell = <own cell>`.
* Tasks (pure-Python evaluators from the corpus runs, reproduce corpus scores exactly):
  `math:circle_packing`, `math:heilbronn_convex_13`, `math:second_autocorr_ineq`,
  `math:third_autocorr_ineq`. ALE tasks are out: podman is unavailable on this host and their
  prompts (61–109k tokens) exceed the 32k context of the served model.
* Agent: greedy hill-climbing replay (parent = best program so far), K iterations, start from
  the corpus's majority root program. Generator: Qwen3.8-27B-FP8 (vLLM, thinking off,
  temperature 0.7, per-request seed derived from (task, iteration, context hash) so identical
  prompts give identical programs).
* Construction (ingest of each new attempt): embed code (Qwen3-Embedding-0.6B, :8011), insert
  row + native lineage edge, and an LLM derivation `changes` = summary of the parent→child
  change (temperature 0), which the semantic filter later reads.
* Retrieval for parent p (Fig. 1(b) shape): ANN top-24 seeds by p's code embedding → Expand
  lineage (fanout 8) → telemetry (valid, improved over its parent, ≥ 0.8·fitness(p)) →
  `trimem.sem_filter` on the descendant's change text → rank by improvement → top 3 → context
  (fitness delta, change summary, code excerpt).

## Arms (execution plans)

| arm | retrieval | semantic placement | construction |
|---|---|---|---|
| A0 no-memory | none | – | ingest only (for parity of logs) |
| A1 tri-modal | V+G+R | – | eager derivation |
| A2 trimem | V+G+R+Sem | planner (auto) | eager derivation |
| A3 trimem-alt | V+G+R+Sem | forced to the placement auto did NOT choose | eager derivation |
| A4 trimem-deferred | V+G+R+Sem | planner (auto) | deferred derivation, resolved before each read (strong) |

A2 vs A3 isolates placement (same logical query → same context → same programs; cost only).
A2 vs A4 isolates construction timing (same state given the derivation cache; cost moves
from construction to retrieval). A0/A1/A2 measure what memory and the semantic filter buy.

## Metrics per iteration

quality (fitness, best-so-far), construction (embed ms/tokens, derivation ms/tokens, insert ms),
retrieval (SQL ms, rows judged, LLM calls/tokens of the semantic filter, deferred resolutions),
generation (LLM ms, prompt/completion tokens), evaluation ms.

Outputs: `results/trimem_v0.1.0/replay/<run>/` (per-iteration JSONL per cell, summary JSON) and
a comparison page.

## Results (run2, 2026-10-11) — `results/trimem_v0.1.0/replay/run2/` (page: replay.html)

run1 is kept for cost only: its quality numbers are invalid (seed = hash(prompt) regenerated the
same failing program after every failure; 20 concurrent CPU-bound evaluations hit the 300 s
timeout). run2 fixes both (seed includes the iteration; evaluator thread budget + slots; 600 s).

| arm | mean gain (4 tasks) | best on | constr. time / LLM tok | retr. time / LLM tok | gen. time / prompt tok |
|---|---:|---:|---|---|---|
| A0 no memory | +0.188 | 0 | 9 s / 0 | 0 s / 0 | 7348 s / 105k |
| A1 tri-modal | +0.426 | 1 | 551 s / 57k | 0.7 s / 0 | 6010 s / 201k |
| A2 TriMEM (planner placement) | +0.405 | 0 | 386 s / 52k | 48 s / 15k | 5715 s / 158k |
| A3 TriMEM (other placement) | +0.438 | 2 | 451 s / 57k | 282 s / 21k | 5818 s / 185k |
| A4 TriMEM (deferred construction) | +0.365 | 1 | 16 s / 0 | 476 s / 62k | 4566 s / 160k |

* Memory matters most where the seed is weak: heilbronn_convex_13 0.063 (no memory) vs
  0.41–0.72, circle_packing 0.61 vs 0.94–0.97. On second_autocorr no-memory (1.002) beats A3/A4
  (0.995); on third_autocorr all arms end within 0.983–0.990.
* Quality differences among A1–A4 are within single-run noise: trajectories diverge by
  iteration 1–3 even where the retrieved context is identical (sampling at T=0.7 is not
  batch-invariant in vLLM), so plan-equivalence of quality is not observable from one run.
* Placement (A2 vs A3): same logical query, 5.9× less retrieval time and 27% fewer semantic
  tokens with the planner's choice (SemFilter above the join).
* Construction timing (A2 vs A4): deferral moves ~50k derivation tokens from ingest (386 s → 16 s)
  to the read path (48 s → 476 s retrieval).
* The semantic filter shortens the agent's prompt: generation prompt tokens 201k (A1) vs 158k (A2),
  −43k for 15k semantic-filter tokens. This downstream effect is exactly the term the planner's
  objective does not yet contain.
* Generation dominates wall time (≈90%); retrieval is ≤8% even with LLM calls in the query.
