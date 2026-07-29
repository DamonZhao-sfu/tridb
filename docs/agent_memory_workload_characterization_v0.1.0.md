# Agent-Memory Workload Characterization with TriDB

> **Version:** 0.1.0  
> **Date:** 2026-07-27  
> **Status:** Methodology and execution guide; paper results are reported, while
> proposed TriDB experiments are explicitly marked as not yet run.  
> **Primary source:** Y. Omri et al., *Agent Memory: Characterization and
> System Implications of Stateful Long-Horizon Workloads*, arXiv:2606.06448v1,
> submitted 2026-06-04
> ([abstract](https://arxiv.org/abs/2606.06448),
> [PDF](https://arxiv.org/pdf/2606.06448),
> [DOI](https://doi.org/10.48550/arXiv.2606.06448)).  
> **TriDB scope:** stock PostgreSQL 16/17 is sufficient for the native graph AM
> and `tjs_open`; GX10 is required for the PG 13.4 MSVBASE-fork sign-off and
> 128 GB headline runs.

## 1. Purpose

This document has three goals:

1. summarize every section of Omri et al.'s paper;
2. reconstruct the paper's experiments and reported results without mixing
   results from different hardware or model regimes; and
3. turn its methodology into a concrete workload-characterization protocol for
   TriDB.

The main conclusion is operational: answer accuracy alone does not characterize
an agent-memory system. A useful TriDB study must measure the complete lifecycle:
memory construction, database writes, retrieval, prompt assembly, answer
generation, maintenance, storage growth, freshness, tail latency, and answer
quality. Each phase must be observable on a common monotonic timeline.

TriDB adds a database-specific question that the paper does not isolate: when
vector search, native graph traversal, and relational filtering execute inside
one PostgreSQL process, how much work is avoided relative to materializing and
joining results across separate systems? TriDB's existing SM-1 through SM-5
metrics answer part of this question. The paper's lifecycle measurements add
construction cost, energy, growth, staleness, and tail behavior.

## 2. Evidence status and reading cautions

> **Verification update (2026-07-29).** The PDF has since been read first-hand
> and every figure reproduced in this document was checked against it: Table 3
> (all nine rows), the Figure 10 TTFT pairs, the Figure 11 tail ratios, the §4.5
> frontier points, the §4.7 growth numbers, and the §4.3 traffic shape all match.
> The two prose/table inconsistencies flagged below are confirmed as the
> paper's own. **One additional discrepancy found on that read:** §4.6's prose
> lists five systems as accumulating staleness, but Figure 8b carries a
> `max stale:` marker on six rows — including **HippoRAG v2 (max stale: 2)**,
> which the prose omits. Cite the figure, not the sentence, for that claim; §4.6
> below inherits the prose list. Details the reconstruction did not carry, plus
> a stage-by-stage capability analysis, are in
> [`agent_memory_system_design_v0.1.0.md`](agent_memory_system_design_v0.1.0.md).

The paper is an arXiv v1 preprint. As of its 2026-06-04 submission, the arXiv
record does not link an official code repository, and the paper describes its
profiling harness as "to be open-sourced." The results are therefore primary
experimental evidence but not independently reproduced here.

Several constraints matter when interpreting the numbers:

- Ten systems are compared, but their native execution models differ. The
  authors standardize generation and embedding backbones within a configuration
  while retaining system-specific buffering, parallel ingestion, consolidation,
  and tool flow.
- The remote regime uses OpenAI models and `text-embedding-3-small`; the local
  regime uses one H100 80 GB with Qwen3 models and
  `Qwen3-Embedding-0.6B`. Absolute timings from one regime must not be compared
  as if they came from the other.
- The principal LongMemEval systems study uses only five histories, although
  each history has roughly 360K tokens and 60 questions, for 300 questions
  total.
- Energy is raw GPU energy, not whole-node energy. CPU, DRAM, storage, network,
  and cooling energy are outside that measurement.
- Some adaptations change prompts or chunking to make systems compatible with
  MemoryAgentBench. These changes are reasonable but complicate strict
  implementation equivalence.
- The full MemoryAgentBench frontier is macro-averaged over heterogeneous task
  families. A strong aggregate score can hide poor multi-hop or temporal
  performance.
- Table 3 and its accompanying prose contain minor numerical inconsistencies.
  Table 3 reports 4,128 J/correct for BM25, while the prose says 4,145. More
  materially, Table 3 and Figure 4 report approximately 116.1, 144.6, and
  185.9 kJ/correct for A-Mem, MIRIX, and Letta, respectively, while the prose
  associates A-Mem and MIRIX with 115 and 197 kJ. This document treats the table
  and figure as authoritative and flags the prose mismatch rather than silently
  reconciling it.

## 3. Paper summary, section by section

### Abstract

The paper frames agent memory as a systems workload rather than only a quality
mechanism. It contributes a four-axis taxonomy, a phase-aware profiler, a study
of ten memory systems across MemoryAgentBench and MemoryArena, and ten
deployment recommendations. Its core finding is that design choices move large
amounts of work between the write and read paths.

### Section 1: Introduction

Long-horizon agents create mutable state from their own interaction streams.
Unlike conventional RAG over a static corpus, agent memory has a write path,
search path, and maintenance policy. Full-history prompting eventually fails on
capacity, prefill cost, and effective recall. External memory bounds the context
presented per query, but memory algorithms differ sharply in construction cost,
query latency, storage, and update behavior.

The paper asks three questions:

1. What tradeoffs distinguish emerging memory paradigms?
2. What computational costs and infrastructure demands do they expose?
3. How do construction, storage, retrieval, and update choices affect
   utilization, bandwidth, latency, and scalability?

For TriDB, the introduction supplies the correct unit of analysis: not a single
ANN query, but the lifecycle of state written and read by an agent.

### Section 2.1: Agent Memory Execution Pipeline

The paper decomposes execution into seven stages:

| Stage | Meaning | Primary cost surface |
|---|---|---|
| Ingestion | Receive dialogue, tool output, documents, traces, and feedback; choose turn, chunk, window, or session granularity | input bytes/tokens, units, arrival rate |
| Construction | Transform raw history into chunks, facts, summaries, triples, or agent-selected records | LLM/embedding calls, prefill and completion tokens, latency, energy |
| Storage | Persist raw or constructed memories | bytes, index size, WAL, write amplification, update cost |
| Retrieval | Select memory records for the current query | calls, candidates, graph work, latency, tail behavior |
| Prompt assembly | Serialize retrieved records into model context | selected entries, prompt tokens, assembly latency |
| Generation | Produce the answer | TTFT, decode latency, tokens, energy, quality |
| Maintenance | Deduplicate, consolidate, resolve conflicts, forget, compress, or re-embed | background work, stale state, reclaimed bytes, quality change |

The paper then abstracts these stages into a two-tier hierarchy: persisted
long-term memory and active working memory. Retrieval moves a bounded subset
from the former into the latter.

### Section 2.2: Taxonomy of Agent Memory Paradigms

The paper classifies systems along construction, storage, retrieval, and
mutability:

| Paradigm | Systems studied | Construction | Retrieval | Mutability |
|---|---|---|---|---|
| I. Long context | `long_context` | none | full-history passthrough | append |
| II. Flat RAG | BM25, embedRAG | deterministic indexing | lexical or dense top-k | append |
| III.a. Append-only structured RAG | GraphRAG, HippoRAG v2 | LLM extraction plus embeddings | dense/graph expansion or PPR/rerank | append |
| III.b. Consolidating structured RAG | Mem0, SimpleMem | LLM extraction and consolidation | fact top-k or iterative hybrid retrieval | consolidate |
| IV. Agentic control flow | A-Mem, Letta, MIRIX | LLM-controlled writes and mutation | graph, tools, or typed-memory routing | mutate |

TriDB is not itself one row in this taxonomy. It is a tri-modal storage and
execution substrate on which multiple rows can be implemented. The current
`bench.agent_memory` adapters implement Paradigm II embedRAG. TriDB becomes a
Paradigm III.a substrate only when a workload actually constructs native graph
topology and retrieves through graph-aware TJS. An external controller can build
Paradigm III.b or IV behavior while TriDB remains the only database and the only
transaction/WAL authority.

### Section 3.1: Workload Suite

MemoryAgentBench converts long contexts into incremental 4,096-token streams and
tests:

- accurate retrieval: SH-Doc QA, MH-Doc QA, LongMemEval_S_*, EventQA;
- test-time learning: BANKING77, CLINC150, NLU, TREC, MovieRec;
- long-range understanding: InfinityBench-Sum and Detective QA; and
- selective forgetting: FactConsolidation-SH/MH.

The systems characterization centers on five LongMemEval_S_* samples, each with
about 360K history tokens and 60 queries. Retrieval is capped at ten entries;
local-model runs may use five when context overflows. MemoryArena supplies 20
multi-session physics tasks for the freshness experiment.

The paper preserves native system behavior but makes targeted compatibility
adaptations. Examples include task-aware extraction prompts, smaller Letta
chunks, tool-call caps, and an ICL-aware Mem0 prompt.

### Section 3.2: Serving and Hardware Setup

Two model-serving regimes are studied:

- **Remote:** GPT-4o-mini or GPT-4.1-mini and
  `text-embedding-3-small`.
- **Local:** one H100 80 GB, six Intel Xeon Platinum 8480C cores, vLLM,
  Qwen3-32B/14B/8B/1.7B, and `Qwen3-Embedding-0.6B`.

The 32B and 14B models use FP8; smaller models use BF16. The LLM server receives
75% of GPU memory and the embedding server 15%. Thinking mode is disabled.

TriDB studies on GX10 must reproduce the *method*, not import the H100 absolute
values. Hardware, quantization, GPU memory fractions, vLLM version, model
revision, and co-location policy must be recorded in every TriDB run manifest.

### Section 3.3: Profiling Harness

The profiler attributes cost to construction, retrieval, and generation on one
monotonic timeline.

API telemetry includes:

- request type and source label;
- start/end timestamps and latency;
- prompt, completion, and embedding input tokens;
- embedded sequence count; and
- the chunk, window, turn, or query responsible for the call.

Hardware telemetry includes power, utilization, VRAM, SM activity, tensor-core
activity, and HBM bandwidth. Phase energy is the time integral of device power.

This phase-aware record is the minimum model for a TriDB workload harness.
Database-specific telemetry should be added to it, not placed in a separate
report that cannot be aligned with model calls.

### Section 4.1: Why Agent Memory

The experiment compares full-history prompting with external memory on
LongMemEval_S_*. Construction is excluded. Most external-memory systems retain
similar or better accuracy at a fraction of long-context serving time. One
reported remote example is Mem0 below 0.1 seconds per query versus approximately
38 seconds for GPT-4.1-mini long-context prompting. On identical local hardware,
per-query serving latency spans about two orders of magnitude across memory
systems ([paper §4.1, Figure 2](https://arxiv.org/pdf/2606.06448#page=5)).

The result justifies measuring memory, but also warns against calling every
external-memory design equivalent. Query latency alone hides the construction
bill.

### Section 4.2: Construction Dominates the Agent Lifecycle

This experiment includes construction plus 300 LongMemEval queries, using local
Qwen3-32B and `Qwen3-Embedding-0.6B`. Deterministic BM25 and embedRAG
construction finishes in under a minute. LLM-mediated construction ranges from
hours to more than half a day.

The paper's complete Table 3 is reproduced below because it is the most useful
quantitative reference for a TriDB lifecycle study
([paper §4.2, Table 3](https://arxiv.org/pdf/2606.06448#page=6)):

| System | Accuracy (%) | Construct + 300 QA wall time | Calls | Total GPU energy (kJ) | J/correct |
|---|---:|---:|---:|---:|---:|
| BM25 | 47.0 | 16.3 min | 300 | 582 | 4,128 |
| GraphRAG | 46.0 | 1.83 h | 3,215 | 2,082 | 15,084 |
| HippoRAG v2 | 44.3 | 44.2 min | 2,743 | 1,339 | 10,079 |
| A-Mem | 42.7 | 11.76 h | 19,230 | 14,864 | 116,116 |
| embedRAG | 39.8 | 14.4 min | 610 | 495 | 4,144 |
| SimpleMem | 36.0 | 3.92 h | 4,447 | 5,481 | 50,749 |
| Mem0 | 32.0 | 4.02 h | 4,538 | 4,878 | 50,813 |
| Letta | 27.7 | 14.36 h | 18,394 | 15,429 | 185,873 |
| MIRIX | 20.0 | 6.03 h | 7,655 | 8,678 | 144,629 |

End-to-end energy spans more than 26.7×, and energy per correct answer more than
47×. The system with the lowest query latency need not have the lowest lifecycle
cost.

### Section 4.3: Construction Is Embedding- and Prefill-Dominated

Construction repeatedly reads long chunks and emits short structured records.
Across systems, the median decode share of construction tokens is only 4.6%,
ranging from 0.9% for Letta to 28.5% for SimpleMem
([paper §4.3, Figure 5](https://arxiv.org/pdf/2606.06448#page=7)).

Embedding traffic is bimodal:

- GraphRAG and HippoRAG v2 issue large offline batches, approximately 2,300 and
  125 sequences per embedding call, respectively.
- Mem0 and agentic write loops issue latency-sensitive per-record embeddings,
  often at a 1:1 call-to-sequence ratio.

The systems implication is interference: background construction wants
throughput and KV-cache capacity, while QA wants low TTFT. The paper recommends
admission control, batching or deferral, separate service treatment for batch
versus sequential embedding traffic, and prefix/chunk reuse.

### Section 4.4: Construction-Model Choice Is Algorithm-Constrained

The construction LLM is swept from Qwen3-1.7B through Qwen3-32B and
GPT-4o-mini while holding the QA model and embedding model fixed. Systems with
soft output contracts degrade gradually as the construction model shrinks.
GraphRAG remains around 47–48% across the ladder. MIRIX fails completely with
Qwen3-1.7B because the model cannot reliably satisfy its tool and JSON
contracts ([paper §4.4, Figure 6](https://arxiv.org/pdf/2606.06448#page=8)).

The minimum viable construction model is therefore an algorithm-specific
capability floor. A cheaper model below that floor can corrupt the memory store
rather than merely reduce answer quality.

### Section 4.5: Construction–Serve–Accuracy Frontier

The experiment macro-averages all MemoryAgentBench datasets and plots build
time, per-query latency, and accuracy:

- BM25: under one second construction, 55.8% accuracy, about 7.4 s/query.
- HippoRAG v2: about 277 s construction, 47.4% accuracy.
- GraphRAG: about 2,850 s construction, 47.0% accuracy.
- A-Mem: about 17,666 s construction, 42.1% accuracy.
- Mem0: about 4,108 s construction, 26.8% accuracy, but the lowest query
  latency at about 2.2 s.
- SimpleMem: approximately 18.4 s/query, the slowest query path in this
  comparison.

No system dominates all three axes
([paper §4.5, Figure 7](https://arxiv.org/pdf/2606.06448#page=9)). BM25's
aggregate lead is workload-dependent: its advantage narrows or reverses on
paraphrase, multi-hop, and temporal tasks.

### Section 4.6: Freshness–Latency Tradeoff

The authors replay MemoryArena session traces with a fixed five-second
inter-session gap. A write is fresh when all prior sessions are committed
before the next query. Synchronous scheduling preserves freshness but places
construction on the user-visible critical path. Asynchronous scheduling hides
write latency but may serve stale memory.

BM25 and embedRAG remain fresh in both schedules. SimpleMem, MIRIX, Letta,
Mem0, and A-Mem accumulate one or more stale sessions. Per-session construction
spans roughly five orders of magnitude: around \(10^{-3}\) seconds for BM25,
around 10 seconds for Mem0 and MIRIX, and above 100 seconds in some
consolidation tails
([paper §4.6, Figure 8](https://arxiv.org/pdf/2606.06448#page=9)).

If construction plus retrieval exceeds the arrival interval, a system cannot
simultaneously guarantee zero staleness and low user-visible latency. The paper
recommends treating this inequality as a feasibility constraint.

### Section 4.7: Per-User Footprint Growth

One user's history is scaled from about 64K to 1M tokens. Most stores grow
roughly linearly in bytes, with about a 9× spread at 1M tokens. HippoRAG v2
reaches approximately 62 MB; Mem0 approximately 12 MB. Projected to 100K users,
the endpoints are approximately 0.7 TB for embedRAG and 6.2 TB for HippoRAG v2.

Agentic systems show a more serious effect: construction token cost grows
super-linearly because each new write queries and updates a growing store.
Retrieval latency stays relatively flat because indexed reads are sub-linear.
None of the evaluated systems prunes by default
([paper §4.7, Figure 9](https://arxiv.org/pdf/2606.06448#page=10)).

The growth *slope*, not only the initial footprint, determines long-lived cost.

### Section 4.8: Serving Latency Structure

The paper splits user wait into retrieval/pre-answer delay and post-first-token
streaming:

| System | Effective TTFT (s) | Total time (s) |
|---|---:|---:|
| BM25 | 2.41 | 3.20 |
| embedRAG | 1.96 | 2.78 |
| GraphRAG | 0.22 | 1.42 |
| HippoRAG v2 | 3.56 | 5.03 |
| Mem0 | 0.10 | 0.57 |
| SimpleMem | 22.58 | 24.68 |
| MIRIX | 1.23 | 2.15 |
| Letta | 9.17 | 12.76 |
| A-Mem | 1.20 | 1.92 |

The two-order-of-magnitude TTFT range is caused mainly by retrieval pipeline
depth, not by the answer model
([paper §4.8, Figure 10](https://arxiv.org/pdf/2606.06448#page=10)).

The paper distinguishes:

- **algorithm-bounded phases**, whose work is capped by a static algorithm; and
- **LLM-bounded phases**, whose tool calls or reflection rounds continue until
  a model decides to stop.

BM25 and embedRAG have p95/p50 near 1.3×; HippoRAG v2 is near 1.6×. Wider
examples include GraphRAG at 5.9× and Letta at 3.9×
([paper §4.8, Figure 11](https://arxiv.org/pdf/2606.06448#page=11)).
LLM-bounded systems therefore need explicit iteration and time limits.

### Section 5: Discussion and Conclusion

The paper concludes that construction, not query serving, often dominates
agent-memory cost. Construction is principally an embedding and prefill
workload, and co-locating it with interactive QA creates resource contention.
The taxonomy predicts cost shape, model sensitivity, batching behavior, and
latency tails.

The authors identify multi-node consistency, multi-agent coordination, and
multimodal memory as open problems. Their study is single-node and
text-centered, which makes it especially relevant to a first TriDB
characterization while leaving distributed and multimodal claims out of scope.

## 4. Paper experiments: designs and results

The paper's experimental program can be reconstructed as eight experiments:

| ID | Experimental idea | Independent variables | Main dependent variables | Reported result |
|---|---|---|---|---|
| P1 | Compare full-history prompting with external memory | memory system, remote/local backbone | answer accuracy, retrieval + generation latency | external memory usually preserves quality while reducing serving latency; local systems span about 100× |
| P2 | Price the entire lifecycle | memory system | construction/query wall time, API calls, GPU energy, J/correct | 26.7× total-energy and >47× J/correct spreads; construction dominates LLM-mediated systems |
| P3 | Characterize construction traffic | memory paradigm | LLM/embed calls, prompt/decode/embed tokens, GPU traces, embedding batch size | median construction decode share 4.6%; traffic splits into batch-heavy and sequential write-loop regimes |
| P4 | Sweep construction LLM capability | construction model size/provider | QA accuracy, structural failures | most systems degrade gradually; GraphRAG is stable; MIRIX fails at Qwen3-1.7B |
| P5 | Measure the build–serve–accuracy frontier | memory system and MAB category | build time, per-query time, macro accuracy | no global winner; BM25 wins aggregate accuracy but not every task family or latency axis |
| P6 | Replay inter-session arrivals | synchronous/asynchronous scheduling, five-second gap | user wait, staleness, per-session construction | slow writers must choose blocking or stale reads; construction spans five orders of magnitude |
| P7 | Scale one user's history | 64K, 128K, 256K, 512K, 1M tokens | construction, token cost, storage, retrieval latency | storage spreads about 9×; agentic token cost can grow super-linearly; retrieval remains fairly flat |
| P8 | Decompose user wait and tails | memory system | effective TTFT, total time, p50/p95 | retrieval depth dominates TTFT; adaptive/LLM-bounded flows have wider tails |

These are reported paper results. They are not TriDB measurements.

## 5. The paper's ten recommendations, translated for TriDB

| Paper recommendation | TriDB interpretation |
|---|---|
| 1. Select memory as a systems decision | Report quality, construction, serving, storage, and maintenance together |
| 2. Price lifecycle energy | Integrate PostgreSQL, embedding, and generation phases; do not report only vLLM decode energy |
| 3. Treat construction as controlled background throughput | Separate or prioritize construction and QA queues; run an interference experiment |
| 4. Reuse overlapping construction inputs | Cache chunk embeddings and construction prefixes; record cache state |
| 5. Validate the construction-model floor | Check schema/tool validity and store integrity before comparing quality |
| 6. Match build/query split to arrival pattern | Sweep queries per write and report amortized cost |
| 7. Treat freshness feasibility as a hard constraint | Require \(T_{construct}+T_{retrieve} \le T_{arrival}\) for zero-stale asynchronous service |
| 8. Make construction cadence system-aware | Measure append, consolidate, compact, and rebuild policies separately |
| 9. Measure growth slope and prune | Sweep history length and user count; report bloat, WAL, and bytes reclaimed |
| 10. Provision for tails | Report p50/p95/p99 and maximum; retain TJS work/censor signals and external LLM timeouts |

## 6. What TriDB can characterize today

### 6.1 Current agent-memory path

The code under `bench/agent_memory/` currently implements:

1. turn- or session-level memory construction units;
2. local batched embeddings using `BAAI/bge-small-en-v1.5`;
3. a scoped PostgreSQL table with JSON metadata, `vector(384)`, a B-tree scope
   index, and cosine HNSW;
4. dense top-k retrieval with `hnsw.iterative_scan = strict_order`;
5. LoCoMo prompt assembly, vLLM generation, LLM judging, and metrics; and
6. LongMemEval retrieval output and official-shaped retrieval metrics.

This is a **Paradigm II embedRAG configuration**. The output must be labeled
`TriDB-vector` or `TriDB-flat-dense`; calling it tri-modal would be inaccurate.

The LoCoMo pipeline currently reports:

- answer and judge token usage;
- mean answer and judge request latency;
- LLM-judge accuracy overall and by LoCoMo category;
- diagnostic lexical F1; and
- evidence recall, hit-any, hit-all, MRR, and NDCG at available \(k\).

It does not yet report construction latency, database retrieval latency,
prompt-assembly latency, storage growth, PostgreSQL I/O/WAL, energy, or latency
quantiles.

### 6.2 Existing TriDB engine probes

The stock-PG fused operator exposes per-backend probes that must be read on the
same database connection immediately after each `tjs_open` call:

| Probe | Meaning |
|---|---|
| `tjs_open_candidates_examined()` | vector candidates consumed, or filter-first qualifying rows examined before top-k |
| `tjs_open_graph_examined()` | native graph-AM edge steps |
| `tjs_open_graph_censored()` | whether `tjs.graph_work_budget` truncated graph work |
| `tjs_open_termination_reason()` | `filter_first`, `term_cond`, `scan_budget`, or `stream_end_unknown`, depending on the installed build |
| `tjs_open_budget_capped()` | observable vector-budget status; `NULL` means the underlying pgvector ending is unknown |
| `tjs_open_bridges_injected()` | graph bridges offered by the last seedless call |
| `graph_store.last_join_order()` | lowering decision for the most recent canonical query |
| `gph_visits()` | cumulative native graph visits in the backend |
| `gph_page_reads()` | cumulative native adjacency page reads in the backend |
| `gph_vertex_count()` / `gph_edge_count()` | graph size |

Additional PostgreSQL surfaces include `EXPLAIN (ANALYZE, BUFFERS, WAL, FORMAT
JSON)`, relation/index sizes, `pg_stat_wal`, and optional
`pg_stat_statements`.

All TJS censor and termination fields are part of the result's epistemic status.
A censored run is a different operating point, not a faster exact run.

## 7. TriDB configurations to compare

Use stable configuration IDs. Never change retrieval semantics under an
existing ID.

| ID | Configuration | Purpose |
|---|---|---|
| LC | Full-history vLLM prompt | Paradigm I quality and prefill baseline |
| TV | TriDB flat dense retrieval | Current Paradigm II adapter |
| TS | TriDB source-anchored canonical TJS | v1 vector + native graph + relational path |
| TO-M | Seedless `tjs_open`, membership scoring | open-query tri-modal research operator |
| TO-P | Seedless `tjs_open`, PPR scoring | graph-graded research operator |
| TV-SHARED | Dense retrieval with all users/scopes retained | multi-user filtered ANN and bloat characterization |
| TJS-SHARED | Tri-modal shared store | multi-user filter, graph, and ANN interaction |
| MS | Neo4j + Milvus + PostgreSQL baseline | materialize-transfer-prune comparison for SM-1/SM-2/SM-5 |

`TS` is the v1 publication configuration. `TO-M` and `TO-P` must be labeled
research/operator configurations because seedless `tjs_open` is not the single
source-anchored canonical query.

Set `tjs.graph_scoring` explicitly for every seedless run. Do not rely on a
version-dependent default.

## 8. Tri-modal memory representation

A valid tri-modal experiment uses the three stores for distinct roles:

- **Vector:** rank memory units or entities by semantic similarity.
- **Native graph AM:** store topology and traverse relationships. Never use a
  relational edge table as the measured graph path.
- **Relational:** enforce scope, time, role, type, validity interval, privacy,
  or other structured predicates.

A practical representation is:

| Object | Relational/vector row | Native graph vertex | Native graph edges |
|---|---|---|---|
| memory unit | content, scope, session, time, role, embedding | yes | belongs-to-session, mentions-entity, follows |
| entity/fact | canonical text, type, validity, embedding | yes | typed semantic relations |
| session | scope, start/end time, ordinal | optional | contains-memory, precedes-session |

Construction may use an external LLM or embedding service, but persistence must
remain inside one PostgreSQL transaction. The memory row, vector, graph
vertices/edges, and relational metadata must commit or abort together. This
preserves TriDB's one-transaction-manager and one-WAL contract.

Graph construction must record provenance: source memory IDs, extractor model,
prompt/version, confidence, creation transaction, and validity interval.
Provenance is relational metadata; topology itself remains in the native graph
AM.

## 9. Required event and artifact schema

Write four artifacts per run:

```text
bench/out/agent_memory/<run_id>/
  run_manifest.json
  events.jsonl
  queries.jsonl
  summary.json
```

### 9.1 `run_manifest.json`

At minimum:

```json
{
  "schema_version": "tridb_agent_memory_workload_v0.1.0",
  "run_id": "20260727-lme-ts-qwen3",
  "git_commit": "<commit>",
  "dirty_worktree": true,
  "dataset": {
    "name": "LongMemEval_S",
    "path": "<path>",
    "sha256": "<sha256>",
    "samples": 5
  },
  "configuration_id": "TS",
  "tridb": {
    "postgres_version": "<version>",
    "extensions": {},
    "block_size": 8192,
    "settings": {},
    "query_template": "canonical-v1"
  },
  "models": {
    "construction": {},
    "embedding": {},
    "answer": {},
    "judge": {}
  },
  "hardware": {},
  "schedule": {
    "mode": "batch-then-query",
    "workers": 1
  },
  "warmup": {},
  "repetitions": 3
}
```

Record exact model revisions and quantization, not only display names.

### 9.2 `events.jsonl`

Every phase event uses a monotonic clock:

```json
{
  "run_id": "...",
  "sample_id": "...",
  "session_id": "...",
  "query_id": null,
  "phase": "construction",
  "operation": "embedding_request",
  "unit_id": "...",
  "start_ns": 0,
  "end_ns": 0,
  "latency_ms": 0.0,
  "input_tokens": 0,
  "output_tokens": 0,
  "sequences": 0,
  "status": "ok"
}
```

Allowed phases should include `ingestion`, `construction_llm`,
`construction_embedding`, `db_write`, `maintenance`, `query_embedding`,
`db_retrieval`, `prompt_assembly`, `answer_generation`, and `judging`.

### 9.3 `queries.jsonl`

One row per query should include:

- all phase timestamps and latencies;
- TTFT and total generation time;
- result IDs, scores, and prompt tokens;
- answer and judge output;
- quality metrics;
- `EXPLAIN` plan digest and buffer/WAL fields;
- every TJS probe listed in Section 6.2; and
- corpus, candidate, graph, and prompt sizes.

### 9.4 `summary.json`

Aggregate by configuration, sample, task category, history length, and arrival
schedule. Preserve raw observations; a summary must never be the only artifact.

## 10. Metric definitions

### 10.1 Phase cost

For phase \(p\):

\[
T_p = t_{p,end} - t_{p,start}
\]

Report sum, mean, p50, p95, p99, maximum, and a query-bootstrap 95% confidence
interval. Do not add overlapping asynchronous phase durations to derive
wall-clock time; wall-clock is measured independently.

### 10.2 Effective TTFT

\[
TTFT_{effective}
= T_{query\ embedding}
+ T_{db\ retrieval}
+ T_{prompt\ assembly}
+ T_{model\ queue}
+ T_{model\ prefill\ to\ first\ token}
\]

The current LoCoMo pipeline measures the entire HTTP answer request, not true
TTFT. Until streaming timestamps are added, label the field
`answer_request_latency`, not TTFT.

### 10.3 Energy

For sampled GPU power \(P_i\) at times \(t_i\), use trapezoidal integration:

\[
E_{GPU} = \sum_i \frac{P_i + P_{i+1}}{2}(t_{i+1}-t_i)
\]

Align samples with phase markers and report sampling frequency and idle-power
policy. Report raw GPU energy separately from whole-node energy.

### 10.4 Quality-normalized cost

\[
J/correct =
\frac{E_{construct} + Q(E_{retrieve}+E_{generate})}
     {Q \times accuracy}
\]

If accuracy is zero, report the metric as undefined/infinite, not zero.

### 10.5 Query-volume amortization

For systems \(A\) and \(B\), the query count at which a more expensive build
breaks even is:

\[
Q^* =
\frac{C_{construct,A}-C_{construct,B}}
     {C_{query,B}-C_{query,A}}
\]

Calculate this independently for wall time, energy, and monetary cost. A
break-even point exists only when the numerator and denominator imply a
positive \(Q^*\).

### 10.6 Freshness

For query \(q_j\), define:

\[
staleness(q_j) =
|\{s_i : i < j \land commit(s_i) > admit(q_j)\}|
\]

Report stale-query fraction, mean/max stale sessions, write backlog, and
synchronous user wait. For dependency-sensitive tasks, report answer quality
conditioned on staleness.

### 10.7 Storage growth

At each history length \(H\), record:

- table, HNSW, and other index bytes;
- native graph relation and index bytes;
- total database bytes;
- live/dead tuples and bloat estimate;
- vertices, visible edges, and graph pages;
- WAL bytes per ingestion unit; and
- construction tokens per new input token.

Fit both linear and log-log slopes. Report the observed range instead of
declaring super-linearity from a visually curved plot alone.

### 10.8 TriDB work efficiency

Retain the existing success metrics:

- SM-1: intermediate-result reduction;
- SM-2: latency-win fraction;
- SM-3: corpus fraction examined;
- SM-4: answer-set parity/recall against an exact oracle; and
- SM-5: transaction atomicity.

Add:

- vector candidates examined per returned result;
- graph edge-steps and page reads per returned result;
- graph censor fraction;
- termination-reason distribution;
- bridge injection count;
- chosen join-order distribution; and
- PostgreSQL shared/local/temp block hits and reads.

### 10.9 Retrieval and answer quality

For LongMemEval, report recall-any, recall-all, and NDCG at the official cutoffs,
plus official answer scoring after generation.

For LoCoMo, report evidence recall/hit/MRR/NDCG, category-level judged accuracy,
and the exact judge model and prompt. If the answer model judges itself, label
the result as self-judged and run an independent-judge sensitivity analysis
before publication.

## 11. Runnable baseline commands

### 11.1 Capture the environment

```bash
cd /local-scratch/localhome/hza214/tridb

export PGPORT=55432
export PGUSER="$(id -un)"
export TRIDB_DSN="postgresql://${PGUSER}@127.0.0.1:${PGPORT}/postgres"
export VLLM_BASE_URL="http://127.0.0.1:8000/v1"
export VLLM_API_KEY="EMPTY"

mkdir -p bench/out/agent_memory/environment

git rev-parse HEAD > bench/out/agent_memory/environment/git_commit.txt
git status --short > bench/out/agent_memory/environment/git_status.txt
python --version > bench/out/agent_memory/environment/python.txt
curl -sS "${VLLM_BASE_URL}/models" \
  > bench/out/agent_memory/environment/vllm_models.json

psql "$TRIDB_DSN" -X -v ON_ERROR_STOP=1 \
  -c "SELECT version(), current_setting('block_size');" \
  -c "SELECT extname, extversion FROM pg_extension ORDER BY extname;" \
  -c "SELECT name, setting, unit FROM pg_settings
      WHERE name IN (
        'shared_buffers', 'work_mem', 'effective_cache_size',
        'track_io_timing', 'max_parallel_workers_per_gather'
      )
      ORDER BY name;" \
  > bench/out/agent_memory/environment/postgres.txt
```

Hash benchmark inputs and record the hash in the manifest:

```bash
sha256sum /path/to/locomo10.json /path/to/longmemeval_s.json
```

### 11.2 Current end-to-end LoCoMo vector baseline

```bash
python -m bench.agent_memory.locomo_pipeline \
  --input /path/to/locomo/data/locomo10.json \
  --retrieval-output bench/out/locomo_tridb_top20.json \
  --output bench/out/locomo_tridb_qwen.json \
  --metrics-output bench/out/locomo_tridb_qwen_metrics.json \
  --top-k 20 \
  --workers 4 \
  --checkpoint-every 20 \
  --request-timeout 300
```

If the retrieval file is missing, the pipeline runs retrieval first. It is
resumable. Use a separate endpoint/model for judging when possible:

```text
--judge-base-url ... --judge-api-key ... --judge-model ...
```

This command produces answer quality and retrieval quality, but it is not yet a
full paper-style workload profile because construction, PostgreSQL retrieval,
and prompt assembly are not separately timed.

### 11.3 Current LongMemEval retrieval baselines

Session-level:

```bash
python -m bench.agent_memory.longmemeval_adapter \
  --input /path/to/longmemeval_s.json \
  --output bench/out/longmemeval_tridb_session.jsonl \
  --granularity session \
  --retrieve-k 50
```

Turn-level:

```bash
python -m bench.agent_memory.longmemeval_adapter \
  --input /path/to/longmemeval_s.json \
  --output bench/out/longmemeval_tridb_turn.jsonl \
  --granularity turn \
  --retrieve-k 50
```

Run these as separate configurations. Session versus turn and user-only versus
`--include-assistant` change the corpus and must not be pooled.

The adapter rebuilds an isolated corpus for each LongMemEval question to match
official flat-retrieval semantics. That repeated construction is not a normal
deployment pattern. Report both:

1. official isolated mode for benchmark comparability; and
2. a persistent/shared variant for deployment characterization.

## 12. Database measurement procedure

### 12.1 Construction

Split construction into:

1. unit formation;
2. construction LLM, if any;
3. embedding;
4. PostgreSQL transaction;
5. index/graph maintenance; and
6. commit.

Before and after the transaction, capture:

```sql
SELECT clock_timestamp(),
       pg_current_wal_lsn(),
       wal_bytes
FROM pg_stat_wal;
```

After construction:

```sql
SELECT
  pg_relation_size('locomo_units') AS heap_bytes,
  pg_indexes_size('locomo_units') AS index_bytes,
  pg_total_relation_size('locomo_units') AS total_bytes;

SELECT gph_vertex_count(), gph_edge_count(), gph_visible_edge_count();
```

Use `COPY` or the production bulk path in a bulk-construction configuration,
and transactional incremental inserts in a streaming configuration. Do not
compare their throughput without identifying the write mode.

### 12.2 Retrieval

For the dense baseline, collect JSON `EXPLAIN` around the same scoped query used
by the adapter:

```sql
EXPLAIN (ANALYZE, BUFFERS, WAL, TIMING OFF, FORMAT JSON)
SELECT id, external_id, content, event_time,
       embedding <=> $1::vector AS distance
FROM locomo_units
WHERE scope_id = $2
ORDER BY embedding <=> $1::vector
LIMIT $3;
```

Use prepared parameters from the harness; do not interpolate vectors or scope
IDs into executable SQL.

For TJS, execute `EXPLAIN ANALYZE`, retrieve results, and read the probes on the
same persistent connection:

```sql
SELECT tjs_open_candidates_examined(),
       tjs_open_graph_examined(),
       tjs_open_graph_censored(),
       tjs_open_termination_reason(),
       tjs_open_budget_capped(),
       tjs_open_bridges_injected(),
       graph_store.last_join_order(),
       gph_visits(),
       gph_page_reads();
```

These functions describe the most recent call or cumulative backend state.
Connection pools must not move the follow-up probe to another backend. Record
deltas for cumulative graph counters.

Explicitly record:

```sql
SHOW hnsw.iterative_scan;
SHOW hnsw.ef_search;
SHOW tjs.graph_work_budget;
SHOW tjs.vector_scan_budget;
SHOW tjs.graph_scoring;
SHOW tjs.ppr_alpha;
SHOW tjs.ppr_rmax;
```

If a setting does not exist in the installed extension version, record
`unsupported`; never substitute a guessed default.

### 12.3 Cache regimes

Run and label:

- cold PostgreSQL cache, where feasible and safely controlled;
- warm database cache with a cold query;
- repeated warm query; and
- warm database plus warm vLLM prefix/cache.

Cache flushing is invasive and platform-specific. Use a dedicated benchmark
host and record the procedure. Never present a warm run as cold.

## 13. Proposed TriDB experiment matrix

The following experiments are designs, not completed results.

### T1. Long context versus TriDB memory

**Question:** At what history length does external TriDB memory dominate
full-history prompting in latency, energy, and quality?

**Sweep:** 64K, 128K, 256K, 512K, and 1M history tokens; configurations LC, TV,
TS, TO-M, and TO-P; fixed answer model and prompt budget.

**Metrics:** construction, effective TTFT, total query latency, prompt tokens,
energy/query, evidence recall, answer quality, storage.

**Hypothesis grounded in the paper:** LC prefill grows with history while
indexed TriDB retrieval stays flatter. This is not a result until run on TriDB.

### T2. Queries-per-write amortization

**Question:** When does expensive graph or LLM-mediated construction pay back?

**Sweep:** 1, 3, 10, 30, 60, 300, and 1,000 queries per construction epoch.

**Metrics:** lifecycle wall time, GPU energy, database work, J/correct, and
break-even \(Q^*\).

**TriDB-specific value:** compare native graph/TJS construction cost against
repeated savings from fused retrieval.

### T3. Construction traffic and co-location interference

**Question:** How do embedding batches, graph writes, and LLM prefill interfere
with interactive QA?

**Conditions:** QA alone; construction alone; co-located without admission
control; co-located with separate concurrency limits; optional separate
embedding endpoint.

**Metrics:** QA TTFT p50/p95/p99, construction throughput, GPU power/utilization,
PostgreSQL commit latency, WAL rate, HNSW and graph-write throughput.

**Stop condition:** reject any scheduling policy that hides construction latency
by serving unreported stale state.

### T4. Construction-model capability floor

**Question:** What is the smallest model that produces a valid TriDB memory
store?

**Sweep:** available local construction models, fixed embedding/answer/judge
models.

**Validity gates before QA:**

- JSON/schema parse success;
- entity/fact/edge referential validity;
- no dangling graph vertex mapping;
- transaction rollback on malformed units;
- duplicate/conflict policy conformance; and
- extraction precision/recall on a hand-checked subset.

**Interpretation:** a structural failure is not an accuracy datapoint. It is a
failed configuration.

### T5. Tri-modal modality and execution ablation

**Question:** Which modality and which fusion mechanism produce the gain?

**Configurations:**

1. vector only;
2. native graph only;
3. relational filter plus vector;
4. vector plus graph;
5. fused vector + native graph + relational TJS;
6. materialized three-leg composition; and
7. multi-system baseline.

**Metrics:** exact-oracle recall, evidence recall, answer quality, latency,
intermediate rows, candidates, graph edge-steps, page reads, and transfer bytes.

**Invariant:** the materialized composition is a baseline, never an alternative
TriDB implementation. The measured TJS path must preserve Open/Next/Close and
early termination.

### T6. Selectivity, topology, and join order

**Question:** When should filter-first, vector-first, or seedless graph expansion
drive the plan?

**Sweep:**

- relational selectivity: 0.01%, 0.1%, 1%, 10%, 50%, 100%;
- graph degree and skew;
- hops: 1, 2, 3;
- top-k: 1, 5, 10, 20, 50;
- graph work budget and vector scan budget; and
- membership versus PPR scoring.

**Metrics:** chosen order, intermediate rows, candidates, graph work, censoring,
latency, recall, and parity.

**Reporting rule:** never combine latency from one budget/termination setting
with recall from another.

### T7. Freshness and session arrivals

**Question:** Can TriDB meet both freshness and latency SLOs under continuous
sessions?

**Schedules:** fixed and Poisson arrivals with mean gaps of 0.1, 1, 5, 30, and
60 seconds; synchronous and asynchronous construction.

**Metrics:** staleness, stale-query fraction, queue depth, construction/commit
time, user wait, answer quality conditioned on staleness.

**TriDB-specific test:** write relational attributes, vectors, and native graph
edges in one transaction while concurrent readers verify that no torn
cross-modal state is visible.

### T8. Per-user and fleet growth

**Question:** How do bytes, WAL, construction cost, and retrieval cost grow with
history and user count?

**Sweep:** 64K to 1M tokens/user and 1, 10, 1K, and projected 100K users.

**Modes:** isolated tables/scopes, shared index, append-only, delete/reinsert,
compaction, and forgetting.

**Metrics:** growth slopes, dead tuples, index size, graph pages, WAL/input byte,
write latency, retrieval tails, recall after maintenance.

**Caution:** a 100K-user number should be labeled projected unless actually
loaded.

### T9. Work-bound and tail-latency characterization

**Question:** Do TriDB's explicit work bounds produce predictable tails without
unacceptable recall loss?

**Sweep:** query type, graph density, budgets, `term_cond`, seed count, hops, and
concurrency.

**Metrics:** p50/p95/p99/max, p95/p50, candidates, graph work, censor fraction,
termination distribution, recall, and timeout rate.

**Expected distinction:** TJS is algorithm-bounded when its explicit budgets are
active. External LLM construction/retrieval loops remain LLM-bounded and need
their own iteration/time caps.

### T10. Transaction and maintenance workload

**Question:** What is the cost and quality impact of mutation, conflict
resolution, and forgetting?

**Operations:** append, update, delete/tombstone, conflict correction, graph edge
replacement, re-embedding, compact, forget.

**Metrics:** transaction latency, WAL, locks, abort rate, space reclamation,
freshness, answer quality, and pre/post-maintenance recall.

**Correctness gate:** every failure injection must leave vector, graph, and
relational state atomically old or atomically new.

## 14. Experimental controls

For every comparison:

1. Pin dataset hash and sample IDs.
2. Pin all model revisions, prompts, token limits, temperature, and seeds.
3. Use the same memory construction units unless granularity is the variable.
4. Use the same answer model and independent judge.
5. Fix top-k and prompt serialization unless they are explicit sweep variables.
6. Validate exact-oracle retrieval before timing approximate configurations.
7. Warm up each model and query path; exclude warmups and report their count.
8. Randomize or counterbalance configuration order to reduce thermal and cache
   drift.
9. Run at least three full repetitions; use enough queries for tail estimates.
10. Report failures, timeouts, invalid stores, censoring, and missing metrics.
11. Separate remote-model network time from local compute time.
12. Separate isolated component runs from co-located interference runs.

For stochastic judged accuracy, report the number of questions and a bootstrap
confidence interval. For paired configuration comparisons, bootstrap paired
question-level differences. Do not use unpaired averages when the same questions
are answered by both systems.

## 15. Recommended report layout

Each result report should contain:

1. **Claim and scope:** configuration, workload, hardware, and what was not run.
2. **Quality table:** retrieval and answer metrics by task category.
3. **Phase table:** construction, retrieval, assembly, generation, maintenance.
4. **Tail table:** p50/p95/p99/max and p95/p50.
5. **Work table:** candidates, edge-steps, page reads, buffers, censoring.
6. **Lifecycle table:** total energy, J/query, J/correct, queries-per-write.
7. **Growth plots:** bytes, WAL, and construction cost versus history/user count.
8. **Freshness plot:** user wait versus stale sessions at each arrival rate.
9. **Pareto plot:** build cost, query cost, and quality.
10. **Limitations:** unavailable counters, projections, self-judging, cache
    state, failed configurations, and hardware gates.

Never promote a single aggregate accuracy value to the headline. At minimum,
pair it with construction cost, p95 effective TTFT, storage per user, and
quality-normalized energy.

## 16. Implementation gaps and recommended order

> **Superseded in detail by
> [`agent_memory_reproduction_plan_v0.1.0.md`](agent_memory_reproduction_plan_v0.1.0.md)
> (2026-07-29).** That document re-verified this list against the working tree
> after the end-to-end LongMemEval pipeline landed — steps 1–3 below are largely
> in place for LongMemEval (phase timing, quantiles, and streamed-TTFT
> measurement all ship in `bench/agent_memory/longmemeval_pipeline.py`), while
> step 2 remains open for LoCoMo. It carries the current gap register (G0–G12,
> with severities and blocked paper experiments) and the phased plan. The rest of
> this document — the paper summary, §9 event schema, §10 metric definitions, and
> §17 checklist — remains authoritative.

The fastest path from the current repository to a credible characterization is:

1. **Add phase telemetry to `TriDBMemoryBackend`.** Time embedding, transaction,
   and search separately; emit one event schema.
2. **Add quantiles and raw per-query records to the LoCoMo pipeline.** Means are
   insufficient for the paper's tail claims.
3. **Add streamed vLLM TTFT measurement.** Preserve total request latency.
4. **Add PostgreSQL plan, buffer, WAL, and size snapshots.**
5. **Add GPU sampling aligned to phase markers.**
6. **Build a native graph-AM memory loader.** Do not model topology as relational
   joins.
7. **Add source-anchored canonical TJS retrieval and then seedless variants.**
8. **Add the history-length, queries-per-write, and arrival-schedule sweep
   driver.**
9. **Add exact-oracle and censor gates to every tri-modal report.**
10. **Run x86/stock-PG development measurements, then repeat GX10-gated
    measurements before making GX10 claims.**

Steps 1–5 characterize the current flat-dense baseline. Steps 6–9 are required
for a genuine TriDB tri-modal agent-memory study.

## 17. Pre-publication checklist

- [ ] Paper results and TriDB results are in separate tables.
- [ ] Every result identifies remote versus local serving.
- [ ] Hardware and model revisions are recorded.
- [ ] Current vector-only adapters are not called tri-modal.
- [ ] Native graph AM, not a relational edge table, backs graph results.
- [ ] Vector, graph, and relational writes share one transaction.
- [ ] TJS work, termination, and censor probes are saved per query.
- [ ] Warm/cold cache state is identified.
- [ ] Construction is included in lifecycle cost.
- [ ] Effective TTFT is measured from query admission, not only vLLM admission.
- [ ] p50, p95, p99, maximum, failures, and timeouts are reported.
- [ ] Quality is broken down by task family.
- [ ] The judge is independent or self-judging is clearly labeled.
- [ ] Energy scope is identified as GPU-only or whole-node.
- [ ] Projections are not presented as loaded measurements.
- [ ] GX10-only claims come from GX10 runs.
- [ ] Raw events, manifests, queries, and summaries are retained.

## 18. Reference

Omri, Y., Gan, Z., Broveak, Z., Geens, R., He, Z., Pentland, A., Verhelst, M.,
Weissman, T., & Tambe, T. (2026). *Agent memory: Characterization and system
implications of stateful long-horizon workloads* (Version 1).
arXiv. https://doi.org/10.48550/arXiv.2606.06448

## AI disclosure

This guide was produced with AI-assisted source extraction, repository
inspection, evidence synthesis, and drafting. Numerical paper claims were
checked against the arXiv PDF and source. Proposed TriDB experiments are
methodological recommendations and have not been represented as completed
measurements.
