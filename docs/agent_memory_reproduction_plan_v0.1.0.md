# Reproducing Omri et al. on TriDB — Gap Register and Phased Plan

> **Version:** 0.1.0
> **Date:** 2026-07-29
> **Status:** Plan. Contains **no measurements**. Every TriDB number referenced
> here is either a prior committed result (cited) or explicitly marked NOT RUN.
> **Supersedes:** §16 ("Implementation gaps and recommended order") of
> [`agent_memory_workload_characterization_v0.1.0.md`](agent_memory_workload_characterization_v0.1.0.md),
> which stated the order before the LongMemEval end-to-end pipeline landed
> (`7e81630`). That document remains authoritative for the paper summary,
> metric definitions (§10), event schema (§9), and the pre-publication
> checklist (§17).
> **Primary source:** Y. Omri, Z. Gan, Z. Broveak, R. Geens, Z. He,
> A. Pentland, M. Verhelst, T. Weissman, T. Tambe, *Agent Memory:
> Characterization and System Implications of Stateful Long-Horizon Workloads*,
> arXiv:2606.06448v1 (2026-06-04).

## 0. Source-access disclosure

> **Resolved (2026-07-29, same day).** The PDF was supplied directly and read
> first-hand after this document was written. Every paper figure cited here and
> in the characterization doc was checked and **matches**; the flagged
> prose/table inconsistencies are the paper's own, and one further prose/figure
> mismatch was found (§4.6 staleness: prose lists five systems, Figure 8b marks
> six, including HippoRAG v2). The "second-hand" caveat below is therefore
> **lifted** — the paper column is now first-hand verified. Details the earlier
> reconstruction did not carry, and a stage-by-stage capability gap analysis of
> the seven-stage pipeline this plan's experiments presuppose, are in
> [`agent_memory_system_design_v0.1.0.md`](agent_memory_system_design_v0.1.0.md),
> which should be read **before** this plan: it establishes which pipeline
> stages exist at all. The original disclosure is retained below for the record.

The paper PDF was **not re-read for this document**. `arxiv.org` is blocked by
this session's egress policy (proxy returned 403 to CONNECT for
`arxiv.org:443`), as is `alphaxiv.org`. What was independently verified here is
the paper's identity and contribution list (title, all nine authors,
2026-06-04 submission, the four-axis taxonomy / phase-aware harness / ten
systems / two suites / ten recommendations structure) via web search metadata.

Everything else in this plan takes its paper facts from the repo's existing
1,191-line reconstruction,
[`agent_memory_workload_characterization_v0.1.0.md`](agent_memory_workload_characterization_v0.1.0.md),
whose AI disclosure states its numerical claims were checked against the arXiv
PDF and which already flags the paper's internal Table 3/prose inconsistencies.
**Before any TriDB result is published against a paper number, one person with
PDF access must re-verify the specific paper cell being compared.** Treat this
plan's paper column as second-hand until then.

## 1. What this plan answers

Three questions, in the order asked:

1. What has the repo already produced toward this paper? (§2, §3)
2. What features are we missing to reproduce its experiments? (§4 — the gap
   register)
3. What is the plan, given that we want an agent-memory *system*, not only a
   benchmark row? (§6–§8)

The short version: **we have one of the paper's ten systems, roughly half of
its profiling harness, none of its energy measurement, and none of its
lifecycle experiments — but we have an engine surface the paper cannot
measure at all.** The honest target is not a replication of the paper's H100
absolute numbers; it is a *methodological* reproduction on our hardware with
TriDB added as an eleventh system, plus the database axis (one transaction
manager, one WAL, fused retrieval) that the paper's ten systems all lack.

## 2. Inventory: what exists today

Verified against the working tree at `7e81630`.

### 2.1 The benchmark path — Paradigm II (embedRAG), vector only

| Component | File | Lines | What it does |
|---|---|---:|---|
| Storage backend | `bench/agent_memory/backend.py` | 341 | Scoped `agent_memory_units` table: `vector(384)` + cosine HNSW + `(scope_id)` and `(scope_id, session_id)` B-trees; `replace_scope()` writes one corpus in a single transaction; `search()` sets `hnsw.iterative_scan = strict_order` so a selective scope predicate cannot silently return fewer than *k*; `isolated=False` retains all scopes for filtered-ANN work |
| LongMemEval retrieval | `longmemeval_adapter.py` | 359 | Session- and turn-granularity corpora, official flat-index JSONL contract, recall-any/all + NDCG at 1/3/5/10/30/50, `has_answer` read only after retrieval |
| LongMemEval end-to-end | `longmemeval_pipeline.py` | 1,481 | The paper's 5 histories × 60 questions = 300-question path. Emits `run_manifest.json`, `events.jsonl`, `predictions.jsonl`, `judge_results.jsonl`, `call_ledger.jsonl`, `summary.json` |
| LoCoMo retrieval | `locomo_adapter.py` | 243 | Per-conversation corpus, `dia_id` preservation, prompt emission for an external answer model |
| LoCoMo end-to-end | `locomo_pipeline.py` | 719 | Resumable retrieval → generation → binary LLM judge → category metrics (Single/Multi/Temporal/Open/Overall) |
| Dataset fetcher | `tools/fetch_memoryagentbench_longmemeval.py` | 64 | Pulls the five-history `ai-hyz/MemoryAgentBench` `Accurate_Retrieval` / `longmemeval_s*` slice, records the HF revision |
| vLLM launcher | `scripts/serve_longmemeval_vllm.sh` | 54 | Answer endpoint (Qwen3-32B-FP8 artifact advertised as `Qwen/Qwen3-32B`, port 8000) + embedding endpoint (`Qwen3-Embedding-0.6B`, port 8001); refuses to replace an occupied port |
| Tests | `tests/test_agent_memory_adapters.py`, `test_longmemeval_pipeline.py`, `test_locomo_pipeline.py` | 299 / 282 / 89 | Parsing, ids, roles, metric shapes, ground-truth isolation, output contracts, scope pushdown |

The LongMemEval pipeline is further along than the characterization doc's §6.1
implies. It already measures, per query: **effective TTFT from query admission
to the first streamed token**, and separately `query_embedding_seconds`,
`tridb_retrieval_seconds`, `prompt_assembly_seconds`,
`vllm_queue_prefill_seconds`, `decode_seconds`, with p50/p95/p99 summaries
(`build_summary`, `longmemeval_pipeline.py:797`), a Wilson 95% interval on
judged accuracy, an exact call ledger separating embedding/answer/judge/insert/
retrieval calls, judge time excluded from serving walltime, a
`paper_reference` block carrying the paper's embedRAG row (39.8%, 14.4 min,
610 calls, TTFT p50 1.96 s, total 2.78 s), and a `comparability` block that
hard-codes `paper_hardware_match: false`.

That is a real slice of the paper's phase-aware profiler. What it is missing is
listed in §4.

### 2.2 The system path — TriDB *is* already a tri-modal memory store

`tools/tridb_mcp.py` (advisor plan 098, `docs/mcp_agent_memory_v0.1.0.md`) is
the closest thing in the repo to the product goal, and it is **not** wired into
the benchmark path:

| MCP tool | Engine mechanism |
|---|---|
| `store_memory(text, kind, embedding?)` | relational row + `vector` + `graph_store.gph_upsert_vertex()` in **one transaction**, with `ext_id == vid` enforced so the graph leg addresses the right memory |
| `connect(src, dst, rel)` | `graph_store.register_edge_type()` + `gph_insert_edge()` — typed native edges, auto-registering relation names |
| `recall(query, k, mode, anchor_id?)` | `mode='fused'` = seedless `public.tjs_open` under the ADR-0021 PPR default (connection-weighted recall); `anchor_id` lowers to the filter-first bounded-traversal path; `mode='vector'` = plain HNSW. Response carries `tjs_open_graph_censored()` and the termination reason |
| `neighbors(id, rel?, hops)` | `gph_traverse_typed` (1 hop) / `gph_traverse_bfs` (multi-hop) |
| `memory_stats()` | `gph_vertex_count()`, `gph_visible_edge_count()`, edge types, extension versions |

### 2.3 Engine surface available to the memory layer

- Graph AM (`src/graph_store*/*.sql`): typed edges, `gph_insert_edges` batch,
  `gph_tombstone_edge/_vertex`, `gph_freeze`, `gph_upsert_vertex` external-id
  mapping, `gph_traverse_bounded` + `gph_traverse_bounded_censored`,
  `gph_visits()`, `gph_page_reads()`.
- Fused operator (`src/tjs_pg/tjs_pg--0.2.0.sql`): `tjs_open(tbl, k, term_cond,
  m_seeds, hops, id_col, filter, query, src, edge_type)` plus six honesty
  probes — `candidates_examined`, `termination_reason`, `budget_capped`
  (NULL = right-censored, never fabricated TRUE), `bridges_injected`,
  `graph_examined`, `graph_censored`.
- Prior measured results reusable as controls: fusion speed win at 200k
  (`benchmark_wiki_fusion_v0.1.0.md`), cross-modal consistency demo
  (`benchmark_wiki_consistency_v0.1.0.md`), Gate A/B filter-first wins, the
  Milvus+Neo4j+Postgres baseline (`baseline/`), SM-1…SM-5 machinery
  (`bench/harness.py`, `bench/metrics.py`), and `bench/graphrag_report.py`'s
  vector-only-vs-mention-graph expansion harness.

## 3. Honest status: authored, not measured

**No agent-memory experiment has produced a committed result.** `bench/out/`
does not exist in the tree and is gitignored; no agent-memory artifact appears
under `bench/results/`; `docs/` has no `benchmark_agent_memory_*` report. The
LongMemEval and LoCoMo pipelines are authored and unit-tested, never run
end-to-end in a recorded run.

So the accurate answer to "we have already produced some of the experiments" is:
**we have produced the harness for one experiment (P1/P2-shaped serving
measurement of one system), not the experiments.** The first real deliverable
is a single recorded TV run, not more code.

Two blockers found while verifying this, both fixed or filed below:

- `requirements-agent-memory.txt` was referenced by `README.md`,
  `bench/agent_memory/README.md`, **and by the pipeline's own error messages**
  (`"tiktoken is required; install requirements-agent-memory.txt"`) but did not
  exist. Created in this change.
- No `make` target exists for either pipeline (`Makefile` has `mcp-demo` and
  38 other targets, none for agent memory). Filed as G0.

## 4. Gap register

Severity: **P0** blocks any credible reproduction · **P1** blocks a specific
paper experiment · **P2** quality/honesty debt.

| ID | Missing feature | Sev | Blocks | Evidence |
|---|---|---|---|---|
| G0 | Runnable entry points: `requirements-agent-memory.txt` (now added), `make agent-memory-*` targets, a smoke path that does not need two vLLM servers | P0 | every experiment | `grep agent.memory Makefile` → only `mcp-demo` |
| G1 | **GPU energy measurement.** No power sampler anywhere in `bench/` (no `pynvml`, no `nvidia-smi` sampling, no trapezoidal integration, no phase-aligned power series) | P0 | P2, P3, and every J/correct or kJ number — i.e. the paper's headline 26.7× energy and 47× J/correct spreads | `grep -ril "pynvml\|nvidia-smi\|joule\|energy" bench/` → 1 README mention |
| G2 | **LLM-mediated construction.** TriDB construction is deterministic chunk+embed. Nothing extracts entities/facts/triples, and **nothing in `bench/agent_memory/` touches the graph at all** (zero hits for `gph_`, `tjs_open`, `graph_store`) | P0 | the paper's central claim (construction dominates the lifecycle), P3, P4, and every Paradigm III/IV comparison | `grep -rin "gph_\|graph_store\|tjs_open" bench/agent_memory/` → no matches |
| G3 | **Comparison systems.** Paper compares ten; we have one (TriDB-vector). Not even the two cheap anchors exist: `long_context` passthrough and BM25 | P0 | P1, P2, P5 — no frontier, no spread, no ranking | `bench/agent_memory/` contains only the TriDB backend |
| G4 | **Workload coverage.** Only the MemoryAgentBench `Accurate_Retrieval` / `longmemeval_s*` slice has a fetcher. Missing: test-time learning (BANKING77, CLINC150, NLU, TREC, MovieRec), long-range understanding (InfinityBench-Sum, DetectiveQA), selective forgetting (FactConsolidation-SH/MH), EventQA, and **all of MemoryArena** | P1 | P5 (macro-average frontier), P6 (freshness has no dataset), and the paper's "BM25's aggregate lead is workload-dependent" caveat | `tools/fetch_memoryagentbench_longmemeval.py` filters `source == "longmemeval_s*"` |
| G5 | **Freshness / arrival driver.** No session-arrival replay, no sync-vs-async construction scheduling, no staleness accounting | P1 | P6, T7 | `grep -ril "staleness\|arrival\|poisson" bench/ tools/` → no matches |
| G6 | **Growth sweep.** No 64K→1M history-length sweep, no per-user footprint or WAL/index-bytes capture in the memory path | P1 | P7, T8 | no sweep driver; `pg_relation_size`/`pg_stat_wal` exist only in `bench/driver.py` for other benches |
| G7 | **Construction-model ladder + validity gates.** No construction model at all, so no capability floor, and no schema/tool-contract/referential-integrity gates to distinguish "worse answers" from "corrupted store" | P1 | P4 | consequence of G2 |
| G8 | **Database telemetry in the memory path.** `EXPLAIN (ANALYZE, BUFFERS, WAL, FORMAT JSON)`, relation/index sizes, `pg_stat_wal` deltas, and all six TJS probes are available in the engine but collected by no agent-memory pipeline | P1 | the TriDB-specific axis the paper cannot measure; SM-1/SM-3 on this workload | probes exist at `src/tjs_pg/tjs_pg--0.2.0.sql:68-117`; unused in `bench/agent_memory/` |
| G9 | **Maintenance / mutability.** No dedup, consolidation, conflict resolution, forgetting, re-embedding, tombstone-driven compaction, or post-maintenance recall check. The engine has `gph_tombstone_*` and `gph_freeze`; the memory layer never calls them | P1 | T10, the paper's Paradigm III.b/IV rows, and the product goal in §7 | `grep tombstone bench/` → no matches |
| G10 | **Interference and amortization drivers.** No co-located-construction-vs-QA experiment, no queries-per-write sweep, no break-even *Q\** calculation | P1 | P3, T2 | no driver |
| G11 | LoCoMo pipeline reports **means only** (`mean_answer_seconds`, `mean_judge_seconds`) — no per-query raw timing rows, no quantiles, no phase split, unlike its LongMemEval sibling | P2 | any tail claim on LoCoMo | `locomo_pipeline.py:387` |
| G12 | Judge protocol: the paper-equivalent path needs GPT-4o (`OPENAI_API_KEY`); the local self-judge is a protocol variant that must never be reported as paper-equivalent, and no independent-judge sensitivity run exists | P2 | comparability of every accuracy number | `bench/agent_memory/README.md` already documents the hazard |

### 4.1 Hardware reality — an unfixable gap, to be stated not closed

The paper's local regime is **one H100 80 GB** with six Xeon 8480C cores, vLLM,
Qwen3 32B/14B/8B/1.7B (FP8 for 32B/14B, BF16 below), 75%/15% GPU-memory split
for LLM/embedding servers, thinking mode off. We do not have an H100. Per
`CLAUDE.md`, our targets are the **GX10 (ARM64 + CUDA, 128 GB)** and the DGX
Spark; the MSVBASE fork builds only on the GX10, while the graph AM and
`tjs_open` run as stock-PG extensions on x86_64 PG 16/17.

Consequences, non-negotiable:

1. **Never** compare a TriDB wall-clock, TTFT, or joule figure directly to a
   paper cell as if replicated. The pipeline's existing
   `comparability.paper_hardware_match: false` flag must survive into every
   report.
2. Cross-system claims require running **every compared system on our own
   hardware** (§ Phase C). This is the real cost driver of the whole plan.
3. Paper cells remain useful for two things only: *shape* checks (does
   construction dominate for LLM-mediated systems here too?) and
   *ordering* checks (does the accuracy/energy ranking survive a hardware
   change?).

## 5. Reproduction triage

| Paper experiment | Reproducible on our stack? | Blocking gaps |
|---|---|---|
| P1 long-context vs memory | **Yes, after LC + one more system** | G3, G0 |
| P2 lifecycle price (wall, calls, energy, J/correct) | **Partly** — wall/calls yes today, energy no | G1, G3 |
| P3 construction traffic + interference | No | G1, G2, G10 |
| P4 construction-model floor | No | G2, G7 |
| P5 build–serve–accuracy frontier | No (only 1 of 4 task families, 1 of 10 systems) | G3, G4 |
| P6 freshness–latency | No (no dataset, no scheduler) | G4, G5 |
| P7 per-user growth | **Partly** — needs a sweep driver + size capture | G6, G8 |
| P8 wait decomposition + tails | **Yes for TriDB-vector today**; cross-system no | G3 (G11 for LoCoMo) |

Two of eight are reachable in weeks; the rest are gated on construction (G2),
comparison systems (G3), and energy (G1) — in that order of leverage.

## 6. Phased plan

Each phase ends with a committed artifact and a stated gate. Phases A, B, D, E,
F are stock-PG/x86-runnable; only the fork sign-off and 128 GB headline runs are
GX10-gated, and none of this plan depends on the fork.

### Phase A — make the documented path runnable (closes G0, G11)

- **A1** `requirements-agent-memory.txt` (**done in this change**) and `make`
  targets: `agent-memory-test` (the three unit files + ruff),
  `agent-memory-lme-smoke` (1 sample × 1 question, `--skip-judge`),
  `agent-memory-lme` (300 questions), `agent-memory-locomo`.
- **A2** Lift the LongMemEval pipeline's event/phase recorder into a shared
  `bench/agent_memory/telemetry.py` implementing the four-artifact schema of
  characterization §9 verbatim (`run_manifest` / `events` / `queries` /
  `summary`), with the phase vocabulary fixed at `ingestion`,
  `construction_llm`, `construction_embedding`, `db_write`, `maintenance`,
  `query_embedding`, `db_retrieval`, `prompt_assembly`, `answer_generation`,
  `judging`. Retrofit both pipelines onto it.
- **A3** Give LoCoMo the LongMemEval treatment: per-query raw rows, phase
  split, p50/p95/p99, Wilson interval.
- **Gate:** `make test` and `make lint` stay green; the smoke run produces all
  four artifacts with a schema-validating test.

### Phase B — one real, complete TriDB-vector row (closes G1, G8)

- **B1** Phase-aligned energy: a `pynvml` sampler thread writing
  `(t, power_W, util, VRAM, SM%, HBM BW)` on the same monotonic clock as
  `events.jsonl`, integrated trapezoidally per phase per characterization
  §10.3. Record sampling frequency and idle-power policy. Report **GPU-only**
  energy and say so.
- **B2** Database telemetry collector: `EXPLAIN (ANALYZE, BUFFERS, WAL, FORMAT
  JSON)` digest per query shape, `pg_stat_wal` and LSN deltas around each
  construction transaction, heap/index/total relation bytes, and — on the same
  connection, immediately after the call — all six TJS probes plus
  `gph_visits()` / `gph_page_reads()` deltas and
  `graph_store.last_join_order()`. Record every relevant GUC
  (`hnsw.iterative_scan`, `hnsw.ef_search`, `tjs.graph_work_budget`,
  `tjs.vector_scan_budget`, `tjs.graph_scoring`, `tjs.ppr_alpha`,
  `tjs.ppr_rmax`), writing `unsupported` where a GUC is absent rather than
  guessing a default.
- **B3** Run configuration **TV** (TriDB flat dense) on the 300-question
  LongMemEval workload, three repetitions, warm/cold cache labeled. First
  recorded TriDB row: accuracy + Wilson CI, construction wall, QA wall, call
  ledger, GPU energy, J/query, J/correct, effective TTFT p50/p95/p99, storage
  bytes.
- **Gate:** publish no accuracy without construction cost, p95 effective TTFT,
  storage per user, and J/correct alongside it (characterization §15). Label
  the row `TriDB-vector` / Paradigm II — **not** tri-modal.

### Phase C — comparison systems on our hardware (closes G3, starts G4)

Ordered by credibility-per-hour:

- **C1 LC** (`long_context`): full-history prompting through the same vLLM
  answer endpoint. Cheapest, and it is the paper's P1 control.
- **C2 BM25**: deterministic, sub-minute construction, and the paper's
  aggregate accuracy winner (47.0% on LongMemEval, 55.8% macro). If our BM25
  does not land near the paper's ordering, our harness is wrong — this is the
  single best sanity check available.
- **C3** One append-only structured system, **GraphRAG or HippoRAG v2**,
  implemented *on TriDB* (native graph AM for topology, never a relational
  edge table) rather than vendored. `bench/graphrag_report.py` already has the
  seed-then-expand-then-rerank machinery to lift.
- **C4** *(optional, expensive)* one consolidating system (Mem0/SimpleMem) or
  one agentic system (A-Mem) to touch Paradigms III.b/IV. Do **not** attempt
  all seven remaining systems; state the omission instead.
- **Gate:** identical answer model, identical judge, identical question set,
  paired question-level bootstrap for every pairwise claim. Report which of
  the paper's ten systems we did **not** run, in the report body, not a
  footnote.

### Phase D — the tri-modal thesis, which is why TriDB is here (closes G2, G8)

This is the phase that turns a benchmark into a contribution.

- **D1 Construction service.** Lift `tools/tridb_mcp.py`'s store/connect
  pattern into a library (`bench/agent_memory/trimodal.py`) used by both the
  MCP server and the benchmarks. It must: chunk → extract (pinned prompt +
  schema version) → embed → write **memory row + vector + graph vertices +
  typed edges + provenance in one transaction**, with provenance (source
  memory ids, extractor model, prompt version, confidence, creating xid,
  validity interval) as relational metadata and topology only in the graph AM.
  Rollback on malformed extraction. Representation per characterization §8.
- **D2 Retrieval configurations** with stable IDs, semantics frozen per ID:
  **TS** (source-anchored canonical TJS — the v1 publication config), **TO-M**
  and **TO-P** (seedless `tjs_open`, membership vs PPR scoring, labeled
  research/operator configs), **TV-SHARED** / **TJS-SHARED** (multi-user
  retained scopes). Set `tjs.graph_scoring` explicitly on every seedless run;
  never inherit a version-dependent default.
- **D3 T5 modality/execution ablation:** vector only · graph only ·
  relational+vector · vector+graph · fused TJS · materialized three-leg
  composition · Milvus+Neo4j+Postgres baseline. The materialized composition
  and the multi-system stack are **baselines, never alternative TriDB
  implementations**; the measured TJS path must preserve Open/Next/Close and
  early termination (TR-1).
- **D4** SM-1…SM-5 on this workload plus per-returned-result work metrics:
  candidates examined, graph edge-steps, page reads, censor fraction,
  termination-reason distribution, bridges injected, join-order distribution,
  buffer hits/reads.
- **Gate:** every tri-modal number carries an exact-oracle recall check (SM-4)
  and its censor/termination status. A censored run is a different operating
  point, not a faster exact run. Never mix latency from one budget with recall
  from another.

### Phase E — the lifecycle experiments (closes G4–G7, G10)

- **E1 (P4)** Construction-model ladder over locally servable models with the
  answer/embedding/judge models fixed, and **validity gates before QA**: JSON
  schema parse rate, entity/fact/edge referential validity, no dangling vertex
  mapping, rollback on malformed units, duplicate/conflict policy conformance,
  extraction precision/recall on a hand-checked subset. A structural failure
  is a failed configuration, not an accuracy datapoint.
- **E2 (P7/T8)** Growth sweep: 64K → 1M tokens for one user, then 1/10/1K
  users; isolated-scope vs shared-index; append-only vs delete-reinsert vs
  compaction vs forgetting. Fit linear *and* log-log slopes and report the
  observed range — do not declare super-linearity from a curved plot. Label
  any 100K-user figure **projected**.
- **E3 (P6/T7)** Freshness: replay session arrivals at fixed 5 s (paper
  protocol) and Poisson means of 0.1/1/5/30/60 s, synchronous vs asynchronous
  construction, with staleness defined per characterization §10.6. Requires
  MemoryArena (G4) or — if unavailable — a **clearly-labeled substitute** built
  from LoCoMo/LongMemEval session boundaries; a substitute is never presented
  as MemoryArena. TriDB-specific addition: concurrent readers verify no torn
  cross-modal state while relational + vector + graph commit together.
- **E4 (P3)** Interference: QA alone · construction alone · co-located without
  admission control · co-located with separate concurrency limits · separate
  embedding endpoint. Reject any scheduling policy that hides construction
  latency by serving unreported stale state.
- **E5 (P2/T2)** Queries-per-write sweep (1, 3, 10, 30, 60, 300, 1000) and
  break-even *Q\** computed independently for wall time, energy, and dollars.
- **E6 (P8/T9)** Tail characterization under explicit budgets: p50/p95/p99/max,
  p95/p50, censor fraction, termination distribution, timeout rate. State the
  distinction the paper draws: TJS is *algorithm-bounded* when its budgets are
  active; external LLM construction loops remain *LLM-bounded* and need their
  own iteration and time caps.

### Phase F — maintenance, i.e. the actual memory system (closes G9)

- **F1** Implement and measure `append`, `update`, `tombstone`, conflict
  correction, graph-edge replacement, re-embedding, `compact`
  (`gph_freeze`-aware), and `forget`, each as a separately measured policy —
  the paper's recommendation 8 ("construction cadence must be system-aware").
- **F2** Failure injection: every abort must leave vector + graph + relational
  state atomically old or atomically new. Reuse `bench/wiki_consistency.py`'s
  torn-state methodology; the prior committed result there (TriDB 0 torn vs
  multi-store 42/42 injected) is the template, and the known residual — the
  native graph leg is commit-visible, not yet snapshot-isolated (DEV-1166) —
  must be restated, not quietly dropped.
- **F3** Report bytes reclaimed, dead tuples, WAL per operation, transaction
  latency, abort rate, and **recall before vs after maintenance**.

### Phase G — report and publication gates

Report layout per characterization §15 (ten sections, claim-and-scope first),
and every box in the §17 pre-publication checklist ticked with evidence.
Paper results and TriDB results stay in **separate tables**.

## 7. From reproduction to an agent-memory system

The stated goal is a system, not a paper reply. The mapping is:

| Product capability | Comes from | Status |
|---|---|---|
| Store a memory as text + vector + typed links, atomically | `tools/tridb_mcp.py` `store_memory`/`connect` | **exists** |
| Connection-weighted recall | seedless `tjs_open` + PPR default (ADR-0021) | **exists** |
| Any agent as a client, zero integration code | MCP stdio server, `make mcp-demo` | **exists** |
| Build memory *from* an interaction stream (extraction, not manual `store_memory` calls) | Phase D1 | **missing — the gap between a store and a memory system** |
| Consolidate, dedup, resolve conflicts, forget | Phase F1 | **missing** |
| Freshness under continuous sessions | Phase E3 | **missing** |
| Bounded, disclosed tails | engine budgets + Phase E6 | probes exist, unmeasured on this workload |
| Evidence it is better than a vector store | Phases B, C, D3 | **missing (no recorded run)** |

Read that column and the priority is unambiguous: **D1 and F1 are the product;
B and C are what make anyone believe it.** The rest of the paper's experiment
list is how we find out where it breaks.

## 8. Recommended immediate order

1. **A1–A3** — plumbing and one shared telemetry schema. Days, no GPU.
2. **B1–B3** — energy sampler, DB telemetry, then *actually run TV once* and
   commit the artifacts. This converts "we have a harness" into "we have a
   measurement."
3. **C1 + C2** — LC and BM25. Cheapest possible credibility, and the BM25
   ordering check validates the harness against the paper.
4. **D1 + D2** — construction into the native graph + TS retrieval. First
   genuinely tri-modal agent-memory result.
5. **D3** — the modality ablation, which is the claim nobody else in the
   paper's ten systems can make.

Everything after that (E, F) is ordered by which recommendation we want to
support with evidence, and can be resequenced freely.

## 9. Risks and standing honesty gates

| Risk | Mitigation |
|---|---|
| Paper facts here are second-hand (§0) | Re-verify each cited cell against the PDF before publication; keep paper and TriDB tables separate |
| Absolute numbers get compared across hardware | `paper_hardware_match: false` propagates into every report; H100 vs GX10/Spark stated in every claim |
| "Tri-modal" applied to the vector-only adapter | Configuration IDs are frozen; TV is labeled Paradigm II embedRAG everywhere |
| Graph modeled as a relational edge table for convenience | CLAUDE.md golden rule 3; `tests/test_tjs_no_bfs_in_operator.py` is the existing precedent for enforcing this in CI |
| A blocking operator sneaks into the memory path | TR-1: Open/Next/Close + early termination, verified per configuration |
| Censored or right-censored runs reported as exact | `graph_censored` / `termination_reason` / `budget_capped` (NULL = unknown) recorded per query; budget-shaped headlines refused |
| Self-judged accuracy presented as paper-equivalent | Judge model and protocol recorded per run; independent-judge sensitivity analysis before any published accuracy |
| Nine of ten paper systems never run | Stated in the report body as a scope limit, with the ordering-check argument (C2) doing the harness validation instead |

## 10. Reference

Omri, Y., Gan, Z., Broveak, Z., Geens, R., He, Z., Pentland, A., Verhelst, M.,
Weissman, T., & Tambe, T. (2026). *Agent memory: Characterization and system
implications of stateful long-horizon workloads* (Version 1). arXiv.
https://doi.org/10.48550/arXiv.2606.06448

## AI disclosure

This plan was produced with AI-assisted repository inspection and drafting. The
paper PDF was inaccessible from the authoring environment (§0); paper facts are
taken from the repo's prior reconstruction and the paper's identity and
contribution list were verified against public search metadata. The inventory
in §2, the status finding in §3, and every gap in §4 were verified against the
working tree at `7e81630` and are cited to files. No measurements were produced
or implied.
