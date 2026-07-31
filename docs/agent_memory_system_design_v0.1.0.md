# What TriDB Needs to Be a Complete Agent Memory System

> **Version:** 0.1.0
> **Date:** 2026-07-29
> **Status:** Capability gap analysis + component design. Contains **no
> measurements**.
> **Companion documents:**
> [`agent_memory_workload_characterization_v0.1.0.md`](agent_memory_workload_characterization_v0.1.0.md)
> (paper summary, metric definitions, event schema) and
> [`agent_memory_reproduction_plan_v0.1.0.md`](agent_memory_reproduction_plan_v0.1.0.md)
> (experiment-level gap register G0–G12, phased plan). This document sits
> *before* both: it asks what capabilities the system must have at all, since
> the paper's experiments are measurements **of** a seven-stage pipeline and
> cannot be run against a pipeline with missing stages.
> **Source:** Omri et al., *Agent Memory: Characterization and System
> Implications of Stateful Long-Horizon Workloads*, arXiv:2606.06448v1
> (2026-06-04, cs.AI). **Read first-hand from the PDF for this document** —
> Stanford / KU Leuven / MIT; correspondence yomri@stanford.edu,
> ziyu.gan@outlook.com.

## 1. Paper verification, and corrections to our earlier docs

The prior two documents took their paper facts second-hand because arxiv.org is
blocked from the authoring environment. The PDF has now been read directly. The
result of the check:

**Confirmed exactly** — Table 3 (all nine systems × accuracy / wall / calls /
kJ / J-per-correct), the Figure 10 effective-TTFT pairs, the Figure 11 tail
ratios, the §4.5 frontier points, the §4.7 growth numbers (~9× spread, HippoRAG
v2 ~62 MB, Mem0 ~12 MB, 0.7 TB → 6.2 TB at 100K users), the §4.3 traffic shape
(median decode share 4.6%, 0.9% Letta → 28.5% SimpleMem; GraphRAG ~2,300 and
HippoRAG v2 ~125 sequences per embedding call), and the ten recommendations.
Our reconstruction was accurate.

**The two prose/table inconsistencies our doc flagged are real**, and are the
paper's, not our transcription's:

- §4.2 prose says "BM25 establishes a floor of 4,145 J per correct answer",
  but Table 3 gives BM25 **4,128** and embedRAG **4,144**. The prose figure
  appears to be embedRAG's, attributed to BM25.
- The same sentence says "A-Mem and MIRIX reach 115 kJ and 197 kJ", while
  Table 3 and Figure 4 give A-Mem **116.1**, MIRIX **144.6**, Letta **185.9**.
  The "197" matches nothing; the nearest value is Letta's. Treat Table 3 and
  Figure 4 as authoritative, as our doc already does.

**One new discrepancy found on this read**, not previously recorded: §4.6 prose
lists "SimpleMem, MIRIX, Letta, Mem0, and A-Mem" as accumulating staleness
under asynchronous scheduling, but Figure 8b carries a `max stale:` marker on
**six** rows — including **HippoRAG v2 (max stale: 2)**, which the prose omits.
Our characterization doc §4.6 inherited the five-system prose list; anyone
citing that sentence should cite the figure instead.

**Details the reconstruction did not carry, and which change what we build:**

| Detail | Where | Why it matters to us |
|---|---|---|
| Retrieval capped at **10 entries for every system**; reduced to **5** only when a local context window overflows; Letta additionally uses **512-token** ingestion chunks | §3.1 | Fixes our retrieval budget as protocol, not a tuning knob. Our LME pipeline's `top_k=10` + 5-chunk prompt cap already matches |
| BM25 construction uses **no LLM and no embedding model** (Table 1: `Embed ✗`) | Table 1 | The BM25 baseline needs no GPU at all — cheapest possible comparison system |
| The four taxonomy axes are literally Table 1's columns: **construction pipeline** (agent-control / LLM / embed / struct), **agent DB**, **retrieval pipeline** (LLM / flow), **mutability** (append / consolidate / mutate) | Table 1 | This is the classification form our configurations must fill in |
| Construction has exactly **four forms**: absent, deterministic, LLM-mediated, agentic | §2.1 | Our construction is *deterministic*. Three of four forms are unimplemented |
| Hardware telemetry uses **NVML + DCGM-class GPM counters**: power, GPU util, VRAM, **SM activity, tensor-core (HMMA) activity, HBM bandwidth**; phase energy = integrating device power per interval | §3.3 | Our energy gap (G1) is wider than "no joules" — the paper reads tensor-core and HBM occupancy too |
| Each system runs in an **isolated SLURM job**, 1 GPU + 6 Xeon 8480C cores | §3.2 | Isolation is part of the protocol; co-location is a separate experiment (§4.3's interference argument) |
| The 64K→1M growth sweep uses **LongMemEval's own arbitrary-context generation** | §4.7 | We do not need to invent a history scaler — the generator exists upstream |
| MemoryArena is **arXiv:2602.16313**, physics split, 20 multi-session tasks, each subtask a **retrieve-act-write cycle**, replayed at a controlled **5 s** inter-session gap | §3.1, §4.6 | The freshness dataset is identifiable and citable; staleness = number of prior sessions not yet persisted at query admission |
| Figure 7b's fourth category is labeled **"Conflict Resolution"** while Table 2 calls the same datasets (FactConsolidation-SH/MH) **"selective forgetting"** | Table 2 vs Fig 7b | Use the dataset names, not the category label, when reporting |
| "**Energy is invariant** ... **latency, by contrast, is schedulable**" | §4.2 | The cleanest statement of why construction scheduling is a systems problem: you can hide *when* you pay, never *that* you pay |

## 2. The capability checklist the paper implies

The paper decomposes agent memory into **seven runtime stages** (§2.1):
ingestion → memory construction → storage → retrieval → prompt assembly →
generation → maintenance. Every experiment in §4 is an instrumented measurement
across those stages. So the user's premise is correct: **the stages are
prerequisites, not outputs.** Concretely:

| Paper experiment | Stages it must be able to instrument |
|---|---|
| §4.1 why memory | retrieval, assembly, generation |
| §4.2 construction dominates | **construction**, storage, retrieval, generation, + energy |
| §4.3 construction traffic shape | **construction** (LLM + embed call/token/batch structure) |
| §4.4 construction-LLM floor | **construction** (with output-contract validity) |
| §4.5 build–serve–accuracy frontier | all of construction + retrieval + generation, across four task families |
| §4.6 freshness | **ingestion as a stream**, construction scheduling, **commit visibility** |
| §4.7 growth | ingestion, construction, **storage accounting**, retrieval |
| §4.8 tails | retrieval, generation |
| (Rec 8/9) | **maintenance** — compaction, pruning, forgetting |

A system that only chunks-and-embeds can be measured on four of the eight rows
and can occupy exactly one cell of Table 1 (embedRAG, Paradigm II).

## 3. Stage-by-stage gap analysis

Legend: **✅** present and usable · **⚠️** partial or present in the wrong place
· **❌** absent.

### 3.0 The headline structural gap: we have two half-stores, not one memory store

Before the per-stage detail, the finding that dominates everything else. TriDB
has **two disjoint memory schemas**, and neither one is an agent memory store:

| | `memories` (`tools/tridb_mcp.py`) | `agent_memory_units` (`bench/agent_memory/backend.py`) |
|---|---|---|
| Columns | `id, kind, text, created_at, embedding` | `id, scope_id, external_id, session_id, kind, role, content, event_time, event_order, metadata jsonb, embedding, created_at` |
| Vector index | HNSW **`vector_l2_ops`** | HNSW **`vector_cosine_ops`** |
| Graph vertices/edges | **yes** — `gph_upsert_vertex` in the same txn, `ext_id == vid`, typed edges via `connect` | **none** |
| Fused retrieval | **yes** — seedless `tjs_open`, PPR default, anchored filter-first | **no** — plain HNSW `ORDER BY <=>` |
| Multi-tenancy | **none** — no user, no scope | `scope_id` + `(scope_id, session_id)` B-trees |
| Interaction-stream model | **none** — no session, role, event time or order | **yes** |
| Provenance | **none** | `metadata jsonb` (unstructured) |

**The store that has the graph has no notion of a user, a session, a turn, or
time. The store that models the interaction stream has no graph.** The paper's
Paradigm III.a "graph store" row requires both *simultaneously* — an entity
extracted from turn 47 of session 3 of user A must be a graph vertex *and*
carry its scope, session, role, and timestamp. Neither table can represent
that today.

They also disagree on the distance operator (L2 vs cosine), so a memory written
through one path and ranked through the other would order differently. This is
a correctness trap, not only duplication.

**Everything in §4 flows from fixing this first.**

### 3.1 Ingestion — ⚠️ batch-only, one source type

| Capability | State | Detail |
|---|---|---|
| Unit = fixed chunk | ✅ | 4,096-token chunks, tiktoken + nltk sentence-aware (`longmemeval_pipeline.py`), matching the MAB protocol |
| Unit = turn | ✅ | LoCoMo dialogue turns; LongMemEval turn granularity |
| Unit = session | ✅ | `--granularity session` |
| Unit = sliding window | ❌ | SimpleMem's construction unit; no windowing primitive |
| Source = user/assistant dialogue | ✅ | with `role`, and a documented `--include-assistant` corpus variant |
| Source = tool output, documents, execution traces, environment feedback | ❌ | the paper's ingestion stage explicitly covers all four; we model dialogue only |
| **Streaming arrival** | ❌ | every path is "read a benchmark JSON, loop over it". There is no `ingest(event)` entry point, no arrival clock, no queue. §4.6 freshness is unrunnable by construction |
| Ingestion of feedback / reward | ❌ | — |

The MCP server's `store_memory(text, kind)` is the only online write API, and it
accepts an opaque blob: no session, no role, no timestamp, no user.

### 3.2 Memory construction — ❌ one of four forms

This is the largest functional gap in the system.

| Construction form (paper §2.1) | State | What is missing |
|---|---|---|
| absent (long-context passthrough) | ❌ | trivial to add; it is the P1 control (LC) |
| **deterministic** (chunk/index, no LLM) | ✅ | chunk + batched embed + one-transaction insert. This is our entire construction path |
| **LLM-mediated** (extract facts / summaries / triples; optionally ADD/UPDATE/DELETE consolidation) | ❌ | no extractor, no output schema, no schema-validity gate, no consolidation decision, no entity resolution, no triple → graph-edge writer, no provenance record |
| **agentic** (LLM-controlled writes, tool invocation, record mutation) | ❌ | no tool surface for writes, no iteration cap, no mutation loop |

Consequences, stated plainly:

- We cannot build **any** Paradigm III or IV configuration, so the paper's
  central finding — construction dominates the lifecycle — has nothing to
  measure on TriDB.
- The native graph is never populated from an interaction stream. Graph
  topology in the benchmark path is not "sparse", it is **absent** (zero
  `gph_*` / `tjs_open` references under `bench/agent_memory/`).
- §4.4's capability-floor experiment needs a construction model *and* output
  contracts to violate. We have neither.
- Recommendation 3 and 4 (batching policy, prefix/chunk reuse) have no knob to
  turn: our embedding batch size is fixed at 64 with no III.a-vs-III.b regime
  distinction.

### 3.3 Storage — ✅ the strongest stage, with two holes

| Capability | State | Detail |
|---|---|---|
| Dense vector store | ✅ | pgvector HNSW, `hnsw.iterative_scan = strict_order` so a selective scope filter cannot silently return fewer than *k* |
| Native graph store | ✅ | graph AM: typed edges, `gph_insert_edges` batch, external-id vertex mapping, tombstones, freeze, bounded + censored traversal, visit/page-read counters |
| Structured metadata / filters | ✅ | relational columns + B-tree indexes; filter pushdown into the fused operator |
| **Multi-store coupling in one transaction** | ✅ | **the differentiator.** One Postgres process, one transaction manager, one WAL — the paper's ten systems all couple stores across process or service boundaries |
| Lexical / inverted index | ❌ | needed for the BM25 comparison and for a SimpleMem-style hybrid store. See §4.4 — this touches a golden rule and needs a decision, not a silent commit |
| Typed memory partitions (semantic / episodic / procedural / resource) | ❌ | MIRIX-style routing; also Figure 1's own taxonomy |
| Footprint accounting API | ❌ | no per-scope bytes, index size, or growth reporting from the memory layer (the SQL exists; nothing calls it) |
| Unified schema | ❌ | see §3.0 |

### 3.4 Retrieval — ⚠️ the best operator is not wired to the memory path

| Capability | State | Detail |
|---|---|---|
| Dense top-k | ✅ | benchmark path |
| Scope/session/time filtering | ✅ | pushed into the ANN query |
| **Fused vector + native graph + relational** | ⚠️ | `tjs_open` exists and works — but only from the MCP server, over the schema that has no scope or session. The benchmark path never calls it |
| Graph-graded scoring (PPR) | ⚠️ | ADR-0021 default; same wiring gap |
| Bounded work + disclosed censoring | ✅ | `graph_work_budget`, `graph_censored`, `termination_reason`, `budget_capped` (NULL = right-censored, never fabricated) |
| Reranking stage | ❌ | HippoRAG v2 reranks after PPR; we have no reranker |
| Lexical leg / hybrid fusion | ❌ | follows from §3.3 |
| Iterative / multi-round retrieval (plan → retrieve → reflect) | ❌ | SimpleMem's read path; also what makes a system LLM-bounded rather than algorithm-bounded |
| Retrieval as an LLM-callable tool | ❌ | Paradigm IV read path |

TriDB's honest position: **our retrieval stage is the most advanced of the seven
and the least connected.** Closing §3.0 converts an existing, tested operator
into an agent-memory capability at near-zero engine cost.

### 3.5 Prompt assembly — ✅ essentially complete

Serialization, prompt fitting, the paper's 10-retrieved / 5-in-prompt overflow
rule, and per-query assembly latency all exist in `longmemeval_pipeline.py`.
Missing only: a shared serializer across configurations (LoCoMo has its own),
separate token accounting for the memory block vs the question, and
prefix-reuse-friendly ordering (Recommendation 4).

### 3.6 Generation — ✅ strong, except energy

| Capability | State |
|---|---|
| Streamed generation, true TTFT from query admission to first token | ✅ |
| Decode-phase timing, prompt/completion token accounting | ✅ |
| p50/p95/p99 + max, Wilson interval on judged accuracy | ✅ |
| Judge (MemoryAgentBench GPT-4o protocol; local judge as a labeled variant) | ✅ |
| **GPU energy, power, SM/tensor-core activity, HBM bandwidth** | ❌ |

The energy gap is a *generation-and-construction* gap, not a generation gap
alone, and it is wider than "no joules": the paper reads NVML **plus**
DCGM-class GPM counters and integrates power per phase interval.

### 3.7 Maintenance — ❌ engine primitives only, no policy layer

The paper is blunt that most systems are weak here ("memory accumulates
indefinitely with no freshness or pruning policy"), and Recommendation 9 says
operators must add pruning themselves. TriDB has the *primitives* and none of
the *policy*:

| Operation | Engine primitive | Memory-layer driver |
|---|---|---|
| Deduplication | — | ❌ |
| Consolidation (merge/distill records) | MVCC update + edge rewire | ❌ |
| Conflict resolution (ADD/UPDATE/DELETE) | one-txn multi-modal write | ❌ |
| Forgetting / pruning | `gph_tombstone_vertex/_edge`, DELETE | ❌ |
| Compression / compaction | `gph_freeze`, VACUUM | ❌ |
| Re-embedding | UPDATE + index maintenance | ❌ |

This is simultaneously our biggest missing capability and our strongest
architectural claim: a consolidation that rewrites a fact, rewires three edges,
and re-embeds the record is **one transaction** in TriDB. In Mem0 or A-Mem it
is a multi-service dance with no atomicity. Nobody in Table 1 can make that
claim — and we cannot demonstrate it until the policy layer exists.

### 3.8 Cross-cutting: the phase-aware profiler — ⚠️

The paper's second contribution is a *single* harness that instruments every
system identically. We have good per-pipeline instrumentation in one pipeline,
weaker in the other (LoCoMo: means only), and no shared component. For
cross-system comparison the profiler must be a separate module that any
configuration plugs into — otherwise every new system re-implements timing and
the numbers stop being comparable.

### 3.9 Summary scorecard

| Stage | State | Blocking for |
|---|---|---|
| Ingestion | ⚠️ batch-only, dialogue-only | §4.6 |
| **Construction** | ❌ 1 of 4 forms | §4.2, §4.3, §4.4, §4.5, and all of Paradigm III/IV |
| Storage | ✅ (holes: lexical, typed partitions, accounting, unified schema) | §4.7 |
| Retrieval | ⚠️ fused operator unwired | the tri-modal thesis |
| Prompt assembly | ✅ | — |
| Generation | ✅ except energy | §4.2, §4.4 |
| **Maintenance** | ❌ policy layer absent | Rec 8/9, Paradigm III.b/IV |
| Profiler | ⚠️ per-pipeline, not shared | every cross-system claim |

## 4. What to build

Five components. Ordered by dependency, not by size.

### 4.1 M1 — the unified memory core (`bench/agent_memory/core.py`)

One schema serving the MCP server, the benchmarks, and any future
configuration. Requirements:

- **Records**: `memory` rows carrying `scope_id` (user/tenant), `session_id`,
  `kind` (`turn` / `chunk` / `window` / `fact` / `entity` / `summary`),
  `role`, `content`, `event_time`, `event_order`, `embedding`, `metadata`, and
  a stable `external_id` for benchmark id preservation.
- **Graph identity**: every record that participates in topology gets a graph
  vertex with `ext_id == memory.id`, upserted in the same transaction, as
  `tridb_mcp.py` already does. This invariant is what lets `tjs_open`'s graph
  leg address memories correctly; it must be enforced, not assumed.
- **One distance operator**, chosen once. Recommend **cosine** (the benchmark
  path's existing choice, and the convention for normalized sentence
  embeddings); migrate the MCP server's `vector_l2_ops` index. Record the
  choice in the run manifest.
- **Provenance table**: source memory ids, extractor model + revision, prompt
  version, confidence, creating xid, validity interval. Relational metadata —
  topology stays in the graph AM (CLAUDE.md golden rule 3).
- **Atomicity contract**: relational row + vector + graph vertices + edges +
  provenance commit or abort **together**. This is the property no system in
  Table 1 has; it deserves a test, not a comment.
- **Accounting API**: bytes per scope (heap / index / graph / total), vertex and
  visible-edge counts, WAL delta per ingestion unit — the inputs §4.7 needs.

Deliverable: both the MCP server and both benchmark pipelines run on this core,
with the old tables migrated or dropped. This is the single highest-leverage
change in the whole program: it turns an existing, tested fused operator into an
agent-memory capability.

### 4.2 M2 — ingestion as a stream

- `ingest(event)` accepting dialogue turns, tool outputs, documents, and traces,
  with unit policy (turn / chunk / window / session) as configuration.
- An **arrival clock**: replay a trace at a fixed gap (the paper's 5 s) or a
  Poisson process, with synchronous and asynchronous construction scheduling and
  a write queue.
- **Staleness accounting** at query admission: number of prior sessions not yet
  committed, per the paper's definition. TriDB gets a sharper version than the
  paper can measure — commit visibility is a single WAL fact here, not an
  eventual-consistency guess across services.

Unlocks §4.6 / T7. Keep the batch path intact: benchmark comparability depends
on it.

### 4.3 M3 — the construction service (the big one)

Three tiers behind one interface, so a configuration is a choice, not a fork:

1. **Deterministic** — today's chunk + embed. Already done; move it behind the
   interface.
2. **LLM-mediated** — the tier that makes TriDB a Paradigm III system:
   - extractor with a **pinned prompt + versioned output schema** (facts,
     entities, typed relations);
   - **schema validation gate**: parse failure → reject the unit and roll back,
     never write a partial record. This gate is what makes §4.4's
     capability-floor experiment meaningful — a structural failure is a failed
     configuration, not a low accuracy score;
   - **entity resolution** to decide whether an extracted entity is a new
     vertex or an existing one;
   - **triple → typed edge writer** over the graph AM (`register_edge_type` +
     `gph_insert_edges`), never a relational edge table;
   - **consolidation decision** (ADD / UPDATE / DELETE) against existing
     memories — this is the III.b behavior, and in TriDB it is one transaction;
   - **batching policy** as an explicit knob: large offline batches (the III.a
     regime, GraphRAG ~2,300 sequences/call) vs per-record 1:1 (the III.b/IV
     regime). Recommendation 3 is only testable if both are reachable.
3. **Agentic** — LLM-controlled writes with a **hard iteration cap** (the
   paper's Recommendation 10: LLM-bounded phases need external caps).
   Lowest priority; do it only if we want a Paradigm IV row.

### 4.4 M4 — storage: two decisions, one of which is not mine to make

- **Typed memory partitions** (semantic / episodic / procedural / resource) as
  a `kind` taxonomy + optional per-type retrieval routing. Cheap; enables a
  MIRIX-shaped configuration later.
- **The lexical index — a golden-rule question.** CLAUDE.md rule 5 says "Three
  stores only... BM25 seam architected but closed for v1." The paper's BM25 is
  its accuracy winner (47.0% on LongMemEval, 55.8% macro) and its cheapest
  system, so we want it as a comparison point. **The resolution that respects
  the rule:** implement BM25 as an **external baseline system** (host-side
  `rank_bm25`, or Postgres `tsvector`/GIN in a *baseline-only* table that is not
  part of TriDB's store surface) — the paper treats BM25 as a separate *system*,
  not as a store inside another system. Nothing forces us to open TriDB's BM25
  seam to get the comparison. Only a SimpleMem-style hybrid *TriDB*
  configuration would require opening it, and that is optional. **Flagging
  rather than deciding: opening the v1 BM25 seam would be a spec change and is
  your call, not mine.**

### 4.5 M5 — the maintenance service

`dedup`, `consolidate`, `resolve_conflict`, `forget`, `compact`, `re_embed` —
each a separately measurable policy (Recommendation 8 requires per-policy
measurement), each transactional, each reporting bytes reclaimed, WAL, dead
tuples, latency, and **recall before vs after**. Plus failure injection:
every abort leaves vector + graph + relational state atomically old or
atomically new. `bench/wiki_consistency.py` already has the torn-state
methodology to reuse, and its known residual — the native graph leg is
commit-visible, not yet snapshot-isolated (DEV-1166) — must be restated in any
result, not quietly dropped.

### 4.6 M6 — the shared profiler

Lift the LongMemEval pipeline's recorder into a module every configuration
plugs into: the ten-phase vocabulary and four-artifact schema of
characterization §9, plus

- an **NVML power sampler** on the same monotonic clock, integrated
  trapezoidally per phase (characterization §10.3), extended toward the paper's
  SM/tensor-core/HBM counters where DCGM is available on our hardware;
- a **database telemetry collector**: `EXPLAIN (ANALYZE, BUFFERS, WAL, FORMAT
  JSON)`, `pg_stat_wal` and LSN deltas, relation/index bytes, and — on the same
  connection, immediately after the call — all six TJS probes plus
  `gph_visits()` / `gph_page_reads()` deltas and
  `graph_store.last_join_order()`. This is the axis the paper's ten systems
  cannot report at all.

## 5. Capability → experiment unlock matrix

| Build | §4.1 | §4.2 | §4.3 | §4.4 | §4.5 | §4.6 | §4.7 | §4.8 | Tri-modal thesis | Maintenance |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| *today* | ~ | — | — | — | — | — | — | ~ | — | — |
| +M6 profiler | ✓ | ~ | — | — | — | — | ~ | ✓ | — | — |
| +M1 unified core | ✓ | ~ | — | — | — | — | ✓ | ✓ | ~ | — |
| +M3 LLM construction | ✓ | ✓ | ✓ | ✓ | ~ | — | ✓ | ✓ | ✓ | — |
| +M2 stream ingest | ✓ | ✓ | ✓ | ✓ | ~ | ✓ | ✓ | ✓ | ✓ | — |
| +M5 maintenance | ✓ | ✓ | ✓ | ✓ | ~ | ✓ | ✓ | ✓ | ✓ | ✓ |

`~` = partial. §4.5 stays partial regardless of what we build, because it
macro-averages **four task families across ten systems**; it is gated on
dataset and comparison-system work (reproduction plan G3, G4), not on our
pipeline capabilities.

## 6. Build order

**M6 (profiler) → M1 (unified core) → M3 (LLM construction) → M2 (stream) →
M5 (maintenance)**, with M4's cheap half (typed partitions) folded into M1.

The reasoning:

1. **M6 first** because without it every subsequent measurement has to be
   redone, and because it makes the *existing* embedRAG configuration
   publishable — a real row with energy, tails, and DB telemetry.
2. **M1 second** because it is the highest leverage per line of code: the fused
   tri-modal operator already exists, tested; it is simply not reachable from
   the memory path. This is also the fix for the L2/cosine correctness trap.
3. **M3 third** because it is the gate on five of the eight experiments and on
   TriDB being anything other than embedRAG.
4. **M2 fourth** — freshness is a strong TriDB story (commit visibility is one
   WAL fact, not a cross-service guess), but it needs M3 to be interesting: a
   deterministic construction path is never stale.
5. **M5 last of the five**, and it is the one that makes this a *product* rather
   than a benchmark harness.

Per the reproduction plan, the comparison systems (LC, BM25) can proceed in
parallel with M1/M3 and are independent of all of this.

## 7. Scope limits and honesty gates

- **Hardware.** The paper is one H100 80 GB under SLURM isolation. We have the
  GX10 (ARM64+CUDA, 128 GB) and the Spark. Absolute numbers are not
  comparable, ever; the existing `comparability.paper_hardware_match: false`
  flag must survive into every report.
- **Nine of ten systems.** We are not reimplementing Letta, MIRIX, A-Mem,
  SimpleMem, Mem0, HippoRAG v2 and GraphRAG. Say so in the report body. The
  BM25 ordering check is what validates our harness instead.
- **Naming.** Until M1 + M3 land, every result is `TriDB-vector` / Paradigm II
  embedRAG. Calling it tri-modal before topology is constructed and traversed
  would be false.
- **Graph is native.** Extracted relations become graph-AM edges, never a
  relational join table (CLAUDE.md rule 3). `tests/test_tjs_no_bfs_in_operator.py`
  is the precedent for enforcing this in CI.
- **TR-1.** Every retrieval configuration honors Open/Next/Close and early
  termination. No blocking operator enters the memory path.
- **Censoring.** `graph_censored` / `termination_reason` / `budget_capped`
  (NULL = unknown) recorded per query. A censored run is a different operating
  point, not a faster exact run.
- **The v1 BM25 seam stays closed** unless you decide otherwise (§4.4).
- **Energy scope.** GPU-only unless whole-node measurement is added; label it.

## 8. Reference

Omri, Y., Gan, Z., Broveak, Z., Geens, R., He, Z., Pentland, A., Verhelst, M.,
Weissman, T., & Tambe, T. (2026). *Agent memory: Characterization and system
implications of stateful long-horizon workloads* (Version 1). arXiv:2606.06448.
https://doi.org/10.48550/arXiv.2606.06448

Related datasets and systems cited above: MemoryAgentBench (Hu, Wang & McAuley,
arXiv:2507.05257), MemoryArena (He et al., arXiv:2602.16313), LongMemEval
(Wu et al., arXiv:2410.10813).

## AI disclosure

Produced with AI-assisted paper reading (full PDF text extraction and
first-hand review), repository inspection, and drafting. The §1 verification
compared every reproduced figure against the PDF; the §3 gap analysis was
verified against the working tree at `7fb3e31` and cites files. No
measurements were produced or implied. The paper's own generative-AI statement
notes its codebase development was accelerated with Claude Code.
