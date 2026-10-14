# SemanticRank on a reranker model (cross-encoder / ColBERT) — plan v0.1.0

Date 2026-10-12. Branch `feat/trimem-semscan`. Requested by the user: keep the operator name
**SemanticRank** (`trimem.sem_score`, plan node `TriMEM SemScore`) and its place in every plan; replace only
the backend that produces the score, from a 27B LLM call per row to a reranker model.

## 1. Why
| | today (LLM) | target (reranker) |
|---|---|---|
| per row | one Qwen3.8-27B-FP8 call (~300 prompt tokens, `{"score": n}` 0-10) | one 0.6B cross-encoder forward, or ColBERT MaxSim |
| cold retrieval | 21 s for ~120 rows (math smoke); 116 s for 4,605 rows | target < 1 s for 500 rows |
| GPU | the same vLLM instance as generation -> a cold SemRank slows every running cell | own small instance on GPU 1 (~14 GB free) |
| planner | `trimem.llm_cost_per_eval = 2125` (0.085 s) for every semantic op | per-op cost: SemRank ≈ 1-3 % of SemFilter |

SemFilter stays on the LLM (a yes/no natural-language predicate; a reranker would need a tuned threshold).

## 2. What stays the same (operator contract)
* SQL: `trimem.sem_score(predicate text, input text) -> double precision in [0, 1]`; placement, `force_placement`,
  EXPLAIN fields (`Inputs Scored`, `Worker Requests`, `LLM Wait`) unchanged.
* Rank formula unchanged: `(d - parent) × (0.5 + sem_score(SEM_PREDICATE, d.changes))`.
* Cache: `trimem.sem_cache` keyed by sha256(op␟model␟pv␟predicate␟input); the reranker gets its own `model`
  string, so LLM and reranker scores never mix, and the existing ~5k LLM scores remain as labels.

## 3. Candidate backends
| id | model | how the score is produced | notes |
|---|---|---|---|
| `qwen3-reranker` (primary) | Qwen/Qwen3-Reranker-0.6B (same family as our embedder) | vLLM `/score`: instruction "Judge whether the code-change summary has the property", query = predicate, document = `d.changes`; P(yes) ∈ [0, 1] | used as-is, no calibration needed |
| `colbert` | jina-colbert-v2 (or colbert-ir/colbertv2.0) | MaxSim(query tokens, doc tokens); doc token vectors computed on the fly for the candidates (a few hundred) | raw MaxSim is unbounded -> fixed calibration `σ((s-μ)/τ)`, μ/τ fitted once offline, so a row's score is the same in every query and can be cached |

The query is the fixed predicate, so the reranker scores "how well the summary has this property". The
LLM grades that property directly. Phase 1 measures how closely the reranker matches it. A parent-aware
query (current parent summary as the query) is a separate design and out of scope here.

## 4. Phases
**Phase 1 — offline agreement study (no DB change, runs now).** `trimem/experiments/rerank_agreement.py`.
* Labels: every LLM `sem_score` row in `trimem.sem_cache` for `SEM_PREDICATE` (math + ALE, ~5k) joined to
  `attempt.changes`, plus the retrievals logged in `retrieval_telemetry.jsonl` (candidate lists).
* Metrics per backend: Spearman ρ against the LLM score; top-10 overlap and NDCG@10 of the **final rank
  formula** on the logged candidate lists (what actually changes the injected memories); latency for 500
  rows (cold); GPU memory.
* Gate: ρ ≥ 0.6 and top-10 overlap ≥ 7/10 on median -> go; if Qwen3-Reranker-0.6B fails, use ColBERT; report both.

**Phase 2 — backend in the worker (Python only).**
* `trimem/rerank.py`: `RerankJudge` implementing `Judge.score_many` (`qwen3-reranker`: vLLM `/score`; `colbert`:
  local encoder + MaxSim + calibration). `judge_many`/`extract_many` delegate to the LLM judge.
* `trimem/worker.py`: route by request model: `--backend <model>=llm:<url>` / `=rerank:<url>` (repeatable),
  so one worker serves the LLM for SemFilter and the reranker for SemanticRank. Invalid -> 0.5 as today.
* Serve: `vllm serve Qwen/Qwen3-Reranker-0.6B --task score --port 8012 --gpu-memory-utilization 0.08` on
  GPU 1.
* Tests: unit tests with a mock `/score` server; worker routing test.

**Phase 3 — per-operator model in the extension (C, version 0.4.0).**
* GUC `trimem.score_model` (default = `trimem.model`): the SemScore node posts its requests and computes
  its cache keys with this model; SemFilter / SemExtract keep `trimem.model`.
* GUC `trimem.score_cost_per_eval` (default = `trimem.llm_cost_per_eval`) used by the SemScore path
  cost, so the planner and the opt arm see a cheap SemanticRank.
* `trimem.stats` / cache coverage are already keyed by model -> no schema change.
* Build and test on the scratch cluster (port 55491) first.
* **Deploying to production 55441 needs a PG restart.** That happens only in a deployment window when no
  experiment uses the cluster, together with SemExtract 0.3.0 (same procedure: install `.so.new` then mv,
  restart, `ALTER EXTENSION trimem_pg UPDATE`).
* *Interim without a restart* (only if experiments must start before the window): give the reranker arms
  their own `prompt_version` (`msem-rr-<task>`), and have the worker answer SemScore requests of those
  prompt versions with the reranker. The keys stay unique because pv is part of the key. Cost: the cache
  row's `model` column is not part of the key, and the planner still prices SemScore at the LLM rate.
  Phase 3 removes both.

**Phase 4 — experiment (math first, 20 iterations).** Arms that differ from the running `msem-*` arms only
in the SemanticRank backend: `msem-rank-rr`, `msem-p1-rr`, `msem-opt-rr` on heilbronn_convex_13 /
second_autocorr_ineq / heilbronn_triangle (same seeds, k = 10, hops = mean depth, change summaries).
Readouts: best@10/@20 vs the LLM-backend arms, retrieval latency (cold and median), semantic tokens on the
generator (should drop to SemFilter only), top-10 overlap with the LLM-backend ranking per retrieval.
ALE follows if math shows no quality loss.

## 5. Risks
* The reranker could agree poorly with the LLM on a judgment rather than a relevance task. Phase 1 gates
  on this before any run, and both backends are reported either way.
* ColBERT calibration drift across tasks: fit μ/τ per domain (math / ALE), store in the plan doc.
* GPU 1 memory: the reranker must fit beside the embedder and the generator; vLLM util capped at 0.08.
* Mixed caches: prevented by distinct model strings (Phase 3) or distinct prompt versions (interim).

## 6. Deliverables
`trimem/rerank.py`, worker routing, `trimem/experiments/rerank_agreement.py` + report
(`results/trimem_v0.1.0/rerank/agreement.json`), extension 0.4.0 (`trimem.score_model`,
`trimem.score_cost_per_eval`) + ADR `docs/decisions/0029-semrank-reranker-backend.md`, oe_matrix arms
`msem-*-rr`, CHANGELOG entry.
