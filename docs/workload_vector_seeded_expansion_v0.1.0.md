# Vector-seeded graph expansion — the workload, and what datasets exist to measure it

> **Version:** 0.1.0
> **Date:** 2026-08-17
> **Status:** Literature and dataset survey. **No measurements.**
> **Companion:** [`paper_plan_agent_memory_v0.1.0.md`](paper_plan_agent_memory_v0.1.0.md)
> (the submission plan this revises the emphasis of).

## 0. Source-access disclosure — read this before citing anything below

**The primary paper was NOT read first-hand.** `arxiv.org` is blocked by this
session's egress proxy, as are every mirror and metadata service tried:
`huggingface.co`, `api.semanticscholar.org`, `papers.cool`, `gangliao.me`,
`arxiviq.substack.com`, `starburst.io`. Only web *search* is reachable.

Everything in §1 is therefore reconstructed from search-result summaries and is
**second-hand**, exactly the condition
[`agent_memory_reproduction_plan_v0.1.0.md`](agent_memory_reproduction_plan_v0.1.0.md)
§0 flags as requiring re-verification before publication. The quoted
characterization of the workload in §1.2 is load-bearing for everything after
it and **must be checked against the PDF by someone with access** before it is
cited. Dataset statistics in §3 are likewise from secondary sources; each one
needs a check against the dataset's own paper or repo card before it enters a
report.

---

## 1. The paper (second-hand)

**Experience Graphs: The Data Foundation for Self-Improving Agents.**
Gang Liao et al., Meta Platforms + University of Maryland, College Park.
arXiv:2606.29823, submitted 2026-06-29.

### 1.1 The argument

Long-horizon agentic work — code generation, scientific discovery, hardware
design — explores by generating artifacts, executing tools, observing failures,
branching, and repairing over hundreds of steps. The residue is a structured
object the authors call an **experience graph**: executable artifacts, tool
outputs, rewards, sibling comparisons, and causal lineage. Agent frameworks
today treat it as disposable state — JSON checkpoints and session logs that
cannot be recovered after a crash, queried across users, or materialized into
training data. **Trellis** makes the experience graph first-class, governed,
queryable database state, which lets agent compute go stateless and serverless
with crash recovery and cross-session reuse for free.

Their four access patterns, restated as database operations:

| Agent behavior | Database operation |
|---|---|
| Frontier selection (what to try next) | a query |
| **Cross-session knowledge reuse** | **vector-seeded graph retrieval** |
| Training-data extraction | a materialized view |
| "What did the agent know at step *t*?" | a time-travel query |

Grounded in **KernelEvolve** (arXiv:2512.23236), a production accelerator-kernel
optimizer at Meta: cross-session reuse reaches a target speedup roughly **10×
faster at 52% lower token cost per valid solution**.

### 1.2 The sentence that matters to TriDB

> Vector-seeded graph expansion composes vector similarity, relational joins,
> and variable-length graph traversal in a **single logical request**. **No
> existing query planner has a cost model for this composition** — the vector
> index's selectivity determines the join fan-out, which determines the
> traversal cost.

That is a description of `tjs_open` and of the join-order problem in
`ADR-0011` / `docs/join_order_cost_model_v0.1.0.md`, written by someone at Meta
who does not know TriDB exists. It is the single most useful external
validation this project has received.

### 1.3 …and the trap inside it

The named open problem is the **cost model**, not the operator. And
`landscape_review_v0.1.0.md` F4 records that `src/planner/join_order.c`
**hardwires `avg_out_degree = 0.0` at three sites**, so the graph leg's
cardinality never enters the decision. A reviewer who reads Trellis and then
greps this repo finds the stub in about ninety seconds.

**Consequence:** "no planner has a cost model for this" is claimable as
*motivation*, and answerable only if the cost model is actually built and
measured across the selectivity × fan-out grid. Feeding real graph-leg
cardinality into the join-order decision moves from a backlog item (F4) to a
**P0 for any paper that cites this workload**.

### 1.4 A second, sharper prerequisite: lineage runs backwards

Two of Trellis's four patterns — causal lineage and "what did the agent know at
step *t*" — are **ancestor** queries. They traverse *incoming* edges of a DAG.

TriDB's adjacency list is out-edges only:
`gph_traverse_typed(src, type, 1, …)` with `direction=in` **RAISEs**
(`agent_memory_gem_interface_v0.1.0.md` §6.4). The GEM design worked around this
by orienting extension edges in the propagation direction — legitimate there,
because propagation only ever goes one way. **It does not work for experience
graphs**, where the same edge must be walked forward (what did this attempt
produce?) and backward (what produced this artifact?). ADR-0016 (reverse
adjacency, ~2× edge storage plus a second GenericXLog page per insert,
format-touching) therefore stops being an optimization and becomes a
prerequisite for this workload.

Two honest options: build ADR-0016, or materialize both edge directions at
ingest and pay the storage explicitly. Either way it must be a stated design
decision, not a silent omission.

---

## 2. Does a ready-made dataset exist? Short answer

**No single dataset covers the workload end-to-end.** Nothing public combines
(i) real vectors, (ii) a real high-fanout graph, (iii) real relational
predicates, (iv) queries whose ground truth requires all three, **and**
(v) experience-graph semantics — branching, rewards, sibling comparison,
causal lineage.

But the workload decomposes cleanly, and each component has a strong public
dataset. Tier A is directly usable now; Tier C is where the gap is real, and
where a generator is cheaper than a dataset hunt.

---

## 3. Tier A — usable now, closest semantics

### A1. STaRK (NeurIPS 2024 Datasets & Benchmarks) — **the best single fit**

`snap-stanford/stark`, arXiv:2404.13207. Semi-structured knowledge bases where
queries blend unstructured text properties with structured relational
constraints and multi-hop structure — *vector-seeded graph expansion with a
relational predicate*, near verbatim, with human-validated ground truth.

| SKB | Entities | Relations | Avg degree | Entity types | Relation types | Domain |
|---|---:|---:|---:|---:|---:|---|
| AMAZON | 1,035,542 | 9,443,802 | 18.2 | 4 | 5 | product search |
| MAG | 1,872,968 | 39,802,116 | 43.5 | 4 | 4 | academic search |
| PRIME | 129,375 | 8,100,498 | **125.2** | 10 | 18 | precision medicine |

Why it fits TriDB specifically:

- **PRIME's average degree of 125 is the regime the paper needs.**
  `benchmark_allpg_baseline_v0.1.0.md` shows the recursive-CTE baseline within
  16 µs at median 22 reached vertices; at degree 125 with 2–3 hops the CTE must
  materialize a reach set that early termination never touches. **If the fused
  operator does not separate on PRIME, it does not separate anywhere** — which
  makes this the cheapest decisive experiment in the whole program.
- Three datasets spanning 18.2 → 125.2 average degree gives a **fan-out ladder
  for free**, which is precisely the axis Trellis says no planner models.
- Ships as a pip package with data on HuggingFace; queries, ground truth, and
  baselines included.

Caveats: MAG/AMAZON entity counts are ~1M — comparable to the existing 1M
Wikidata slice, so no new scale story. Embeddings are the benchmark's own, so
pin the encoder and record it.

### A2. GRBench (Graph-CoT, ACL 2024) — the agentic read path

`PeterGriffinJin/Graph-CoT`, arXiv:2404.07103. 1,740 questions over **10 real
domain graphs** across academic, e-commerce, literature, healthcare and legal;
split 700 simple (single-hop) / 910 medium (multi-hop) / 130 hard (inductive).

Value: the questions are designed for an agent to explore *iteratively* —
seed, look, expand, look again. That is the multi-round retrieval TriDB does
not have (`agent_memory_system_design_v0.1.0.md` §3.4 marks it ❌) and it is the
read pattern Trellis's frontier-selection query implies. Smaller graphs, so
treat it as a *quality/coverage* workload, not a scale workload.

### A3. BigANN NeurIPS'23 filtered track (YFCC-10M) — calibrate the seed leg

`big-ann-benchmarks.com/neurips23.html`. 10M-image YFCC slice, CLIP embeddings
at **192 dimensions**, 200,386 unique labels, heavily skewed, ~10.8 labels per
point; **100,000 queries**, each one image embedding plus one or two tags that
must all be present; ranked on QPS subject to **recall@10 ≥ 0.90**.

This is the seed leg alone — vector + relational predicate, no traversal — and
that is exactly why it belongs in the paper. It is the workload on which
ACORN, Filtered-DiskANN, iRangeGraph, NaviX, pgvectorscale and Milvus/Weaviate
have published numbers. Running it answers "did you cherry-pick a weak vector
index?" before a reviewer asks, and it directly addresses the known
`benchmark_allpg_baseline_v0.1.0.md` finding that **plain pgvector currently
beats `tjs_open` on the seedless filtered-ANN leg**.

### A4. OGB-LSC MAG240M — the only public vector+graph dataset at real scale

`ogb.stanford.edu/docs/lsc/mag240m/`. **121M papers, 1.3B citation edges**,
node features = title+abstract through a RoBERTa sentence encoder at **768
dimensions**.

This is the answer to `docs/STATUS.md`'s "I/O-locality thesis dead at this
scale" finding: dim-384 float32 at 1M is RAM-resident on a 128 GB box, so
page-locality was structurally untestable. At 121M × 768-d float32 (~372 GB of
raw vectors alone) it is testable again, and the CSR-lite read-once seek win
(`csr_lite_gate_b_realio_v0.1.0.md`: ~2.9–3.6×, at ~33× on-disk footprint)
becomes a measurable claim rather than a micro-benchmark. No natural-language
queries — synthesize the query set, as `bench/wiki_h2h_queryset.py` already
does for enwiki.

---

## 4. Tier B — same shape, graph must be built

- **GraphRAG-Bench** (arXiv:2506.02404, `graphrag-bench.github.io`) — college-level
  domain questions across 16 disciplines / 20 textbooks, evaluating construction
  → retrieval → generation and reasoning coherence, not just final answers.
- **WildGraphBench** (arXiv:2602.02053) — wild-source corpora; stresses long
  context, noise robustness, multi-document aggregation.
- **HotpotQA / 2WikiMultihopQA / MuSiQue / MultiHop-RAG** — the standard
  multi-hop set. **Already wired here**: `bench/hotpot_stock_gate.py` and the
  committed +15.6 pt graph-bridge result (`benchmark_public_repro_v0.1.0.md`).
- **CRAG** (KDD Cup 2024, Meta) — mock KG plus web search; hybrid by
  construction.

These test *retrieval quality*, and their graphs are extracted rather than
given, so construction cost confounds the comparison. Use them as secondary
evidence, never as the systems headline.

---

## 5. Tier C — experience graphs proper: the real gap, and how to close it cheaply

Nothing public carries rewards **and** sibling comparisons **and** causal
lineage over executable artifacts. The closest substrates:

### C1. Public agent-trajectory corpora (raw material, wrong shape)

| Corpus | Scale |
|---|---|
| `nebius/SWE-rebench-openhands-trajectories` | 67k trajectories; 32,161 successful of 67,074 solution attempts |
| `nvidia/Open-SWE-Traces` (arXiv:2606.16038) | 207,489 trajectories, 9 languages |
| `nvidia/SWE-Zero-openhands-trajectories` | 318k trajectories |

Multi-attempt and multi-turn at real scale, with pass/fail rewards. **What they
lack is branching**: they are mostly independent linear rollouts, so sibling
comparison and causal lineage have to be *inferred* (same task, multiple
attempts → siblings; shared prefix → lineage). That inference is defensible and
cheap, and it should be stated as a construction, not presented as ground truth.

### C2. Evolutionary program databases — **the recommended generator**

OpenEvolve (open reimplementation of DeepMind's AlphaEvolve) keeps a **Program
Database storing every program with its code, fitness scores, generation, and
lineage**. That is an experience graph by definition: branching search,
per-node rewards, sibling comparison within a generation, explicit causal
lineage.

Why this is the right answer to "is there a dataset?": **run the generator and
mint one.** Point OpenEvolve at a cheap deterministic objective (a numeric
kernel, a packing problem, a bit-twiddling routine) and it produces a real
branching reward-bearing graph without spending an LLM budget on 300 SWE tasks.
Sweep generations × population × branching factor and the fan-out axis is a
knob rather than a property of a fixed corpus — which is exactly what a cost
model needs to be validated against. Pin the seed and the model, and it is
reproducible.

### C3. KernelEvolve (arXiv:2512.23236)

The paper's own grounding. **Whether an artifact or trace corpus was released
could not be checked** (arxiv unreachable). Worth checking first — if traces
are public, that is the highest-fidelity Tier C data available and it comes
with the workload's own provenance.

### C4. Tree-search RL rollouts

Tree-GRPO (ICLR 2026, `AMAP-ML/Tree-GRPO`) samples over a semantically defined
search tree of ReAct step-level nodes; Tree Training exploits shared prefixes
across branches. These produce genuine branching structures with rewards, at
the cost of running the RL loop.

---

## 6. Recommended evaluation — three workloads

| ID | Workload | Data | Answers |
|---|---|---|---|
| **W1** | Vector-seeded expansion across a fan-out ladder | STaRK AMAZON (18.2) → MAG (43.5) → PRIME (125.2) | Does the fused operator separate from the all-PG CTE, and **where**? Locate the crossover as a measured boundary. |
| **W2** | Seed-leg calibration | YFCC-10M filtered track | Are we competitive with filtered-ANN SOTA, or is the seed leg a liability? Publish it either way. |
| **W3** | Experience-graph patterns | OpenEvolve-generated graphs + one SWE trajectory corpus | Trellis's four patterns: frontier selection, vector-seeded reuse, materialized-view extraction, **time-travel**. |

Baselines for all three, pinned by version: all-PG SQL (pgvector + recursive
CTE), Kuzu/NaviX, Neo4j + Milvus app-side, FalkorDB, Amazon Neptune Analytics,
and pgvectorscale/ACORN on W2.

**The distinguishing experiment is W3's time-travel pattern.** "What did the
agent know at step *t*" is an as-of query; Postgres MVCC plus the
`valid_from` / `valid_to` / `superseded_by` columns already in
`bench/agent_memory/gem/schema.sql` answer it directly, and the GEM
`retrieve(route=TEMPORAL, as_of=…)` path already implements it
(`bench/agent_memory/gem/retrieve.py:191`). A Milvus+Neo4j stack has no
cross-store snapshot to be as-of *at*. That is a capability gap, not a
speed difference, and capability gaps survive review better than latency
ratios.

**The cost-model experiment is the contribution.** Sweep vector-leg selectivity
× graph fan-out × hop depth, and report plan choice, plan regret, and the
oracle-best plan. Trellis says no planner has this model; the deliverable is the
model plus the regret surface that shows what it buys. This subsumes F4 and
gives the fused operator a reason to exist that is independent of the 16 µs.

---

## 7. What this changes about the submission plan

`paper_plan_agent_memory_v0.1.0.md` proposed a correctness-first thesis
(trajectory-level invariants, anomaly rate, stale-answer rate). This paper
suggests a **second viable framing**: the cost model for vector-seeded
expansion, motivated by an independent Meta paper naming it as an open problem.

They compose rather than compete — governed state (correctness) and
vector-seeded expansion (cost model) are the write path and the read path of
the same system. But they have different critical paths:

| Framing | P0 prerequisites |
|---|---|
| Correctness-first | DEV-1166 snapshot isolation · real external memory baselines · MemDrift |
| **Cost-model-first** | **`avg_out_degree` stub (F4)** · **ADR-0016 reverse adjacency** · STaRK/YFCC ingest |

The cost-model framing is **cheaper**: STaRK and YFCC are downloadable today,
the fan-out ladder is a property of the data rather than something to build, and
it needs no LLM budget, no judge protocol, and no external agent-memory system
to be stood up. If the goal is a submission this cycle, **W1 on STaRK-PRIME is
the two-week experiment that tells you whether either paper exists.**

---

## 8. Immediate next steps

1. **Get the PDF read first-hand.** §1.2 is quoted from a search snippet and the
   entire framing rests on it. Also check whether KernelEvolve released traces.
2. **Download STaRK, load PRIME into TriDB** (129K entities / 8.1M edges — small
   enough to land in a day, dense enough to be decisive). Run the fused path
   against the all-PG CTE at 1/2/3 hops.
3. **Unstub `avg_out_degree`.** The metapage degree counters exist
   (`advisor-plans/006-graph-metapage-degree-stats.md`).
4. **Decide ADR-0016 vs dual-direction materialization** for lineage, and write
   it down as a decision either way.

---

## 9. Reference list — verification status

Every arXiv id below came from search results, **not** from a fetched page.
Verify id, title and venue before citing.

| Work | Identifier | Verified here |
|---|---|---|
| Experience Graphs / Trellis | arXiv:2606.29823 | title+authors from search only |
| KernelEvolve | arXiv:2512.23236 | search only |
| STaRK | arXiv:2404.13207, NeurIPS'24 D&B, `snap-stanford/stark` | search only |
| Graph-CoT / GRBench | arXiv:2404.07103, ACL 2024 | search only |
| NaviX | arXiv:2506.23397, PVLDB vol 18 | search only |
| GraphRAG-Bench | arXiv:2506.02404 | search only |
| WildGraphBench | arXiv:2602.02053 | search only |
| SeedER | arXiv:2605.23753 | search only |
| SAGE (structure-aware graph expansion) | arXiv:2602.16964 | search only |
| Query-aware spreading activation | arXiv:2606.30133 | search only |
| Filter-agnostic vector search on PostgreSQL (E&A) | arXiv:2603.23710 | search only |
| Open-SWE-Traces | arXiv:2606.16038 | search only |
| Tree-GRPO | ICLR 2026, `AMAP-ML/Tree-GRPO` | search only |
| BigANN NeurIPS'23 | `big-ann-benchmarks.com/neurips23.html` | search only |
| MAG240M | `ogb.stanford.edu/docs/lsc/mag240m/` | search only |
| UniBench / M2Bench | PVLDB 16(4) for M2Bench | search only; both confirmed to lack a vector leg |

---

## Addendum 2026-08-17 — superseded on source access

The PDF was subsequently supplied and read first-hand. §0 (source-access
disclosure) and §1 (the paper, second-hand) of this document are **superseded**
by [`baselines_vector_seeded_expansion_v0.1.0.md`](baselines_vector_seeded_expansion_v0.1.0.md)
§0, which records what the first-hand read changed — including that the venue is
CIDR'27, that the graph is a search tree rather than a hyperlink graph, that the
vector predicate is a threshold rather than a top-k, and that Trellis is itself a
federated design. The dataset survey (§2–§6) stands, with the two revisions in
that document's §6.
