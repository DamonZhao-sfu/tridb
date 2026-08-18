# Baselines for the vector-seeded graph expansion workload

> **Version:** 0.1.0
> **Date:** 2026-08-17
> **Status:** Baseline roster + corrections from a first-hand read. **No measurements.**
> **Supersedes** §0–§1 of [`workload_vector_seeded_expansion_v0.1.0.md`](workload_vector_seeded_expansion_v0.1.0.md),
> which was written from search results because arxiv was unreachable. The PDF
> has now been read directly. §1 below records what changed; the rest of that
> document's dataset survey stands, with the caveats in §6.

---

## 0. The paper, read first-hand

**Experience Graphs: The Data Foundation for Self-Improving Agents.**
Liao et al. (Meta Platforms) + Witten & **Daniel J. Abadi** (UMD).
arXiv:2606.29823v1, cs.DB, 2026-06-29. **Venue: CIDR'27**, January 19–22 2027,
Amsterdam.

That it is a **CIDR vision paper** is the single most useful fact. CIDR papers
state open problems and invite the community to solve them; §7 lists seven, and
**three of them are TriDB's existing engineering**:

| Paper's open problem (§7) | TriDB artifact |
|---|---|
| "Query planning for multi-modal composition… no system maintains **cross-modal statistics**" | `tjs_open` + ADR-0011 join order + the `avg_out_degree` stub (F4) |
| "**Consistency for concurrent tree search** — multi-hop read-then-write, tolerates eventual consistency on statistics, requires durability on node insertion… does not map cleanly to standard OLTP levels" | one txn / one WAL; `bench/wiki_consistency.py`; DEV-1166 |
| "**Physical design** — appends, path updates, ordered reads, multi-hop traversals, vector similarity, full scans. No existing physical design is optimized for this combination" | graph AM page layout, CSR-lite, `fillfactor=70` HOT result |

### 0.1 What the workload actually is

Four-table hierarchy: **tasks → sessions → nodes → prompt_histories**. Embeddings
sit on *task descriptions*. Large artifacts live in object storage by reference.
Parent–child edges are **virtual edges over the foreign key** — Trellis has no
edge table.

The vector-seeded graph expansion query, verbatim from §4:

```cypher
MATCH (t:tasks) WHERE t.embedding <~> $q > 0.8
MATCH (t)<-[:BELONGS_TO]-(s:sessions)
MATCH (s)<-[:IN_SESSION]-(n:nodes)
WHERE n.is_buggy = false AND policy_allows($user, n)
RETURN t.task_id, score(t), n.node_id, n.fitness_score
ORDER BY score(t) DESC, n.fitness_score DESC
LIMIT 10
```

followed by `[:HAS_CHILD*1..k]` expansion of each top-*k* candidate into its full
trajectory.

Access pattern (§3): writes are **appends (new nodes) mixed with localized path
updates** — MCTS backpropagation walks the ancestor chain rewriting
`visit_count` and cumulative reward. Reads span **four modalities**: ordered
scans (frontier selection), multi-hop traversal (ancestor + subtree), vector
similarity (cross-session reuse), and full scans (training extraction).
Operational SLO: **sub-50 ms frontier queries**.

Production measurements (§6, KernelEvolve, greedy search, 100-step budget,
~100-node sessions, 3 sessions per config): buggy-node rate **55% → 34%
(p=0.1) → 21% (p=0.5)**; valid nodes meeting baseline speedup **79.5% → 90.8%
→ 100%**; 1.2× reached at **~step 5 vs step 51** cold (the "10×"); **52%** token
cost per valid node. And an honest anti-result: at p=0.5 the search collapses to
**8 strategy combinations vs 20** cold, and the single best solution comes from
**no memory** (1.49× vs 1.36×) — the exploration–anchoring tradeoff.

### 0.2 Five corrections to the previous document

1. **The graph is a search tree/DAG, not a hyperlink graph.** Fan-out is the
   branching factor of MCTS/evolutionary search (single digits to low tens),
   not a degree-125 hub. My STaRK-PRIME recommendation stays useful as a
   *stress* case for the operator, but **it is not this workload's shape** and
   must not be presented as if it were.
2. **Both edge directions are required, in the same workload.** Ancestor chain
   (backprop, AS-OF trajectory reconstruction, context assembly) walks *up*;
   subtree expansion and sibling comparison (DPO pairs) walk *down*. TriDB's
   adjacency list is out-edges only and `direction=in` RAISEs
   (`agent_memory_gem_interface_v0.1.0.md` §6.4). The GEM workaround — orient
   edges in the one direction you need — **does not apply here**. ADR-0016 or
   explicit dual-direction materialization is a hard prerequisite.
3. **The vector predicate is a threshold, not a top-k.** `t.embedding <~> $q >
   0.8` selects a *set* of unknown cardinality, and that cardinality is exactly
   the selectivity that drives join fan-out in the open problem. `tjs_open`'s
   surface is k-oriented; a threshold-seeded variant needs thinking about.
4. **A governance predicate sits inside the query** (`policy_allows($user, n)`).
   That is a relational predicate that must push into the operator, which is
   what TriDB's filter pushdown already does — and it is a second, independent
   argument for a single engine, since a polyglot stack must apply policy at
   whichever tier can see the user.
5. **Trellis is itself a federated design.** Axiom (cost-based optimizer) over
   Velox plans SQL + Cypher + vector into one physical plan and **routes each
   fragment to a different backend** — operational store, vector index,
   columnar warehouse. TriDB's differentiator is therefore *not* "one logical
   model over many stores" (Trellis has that) but **one process, one heap, one
   WAL, one snapshot, one operator**. Sharpen the claim accordingly.

---

## 1. What you cannot compare against: Trellis

Axiom, the Trellis integration, KernelEvolve, and its experience-graph corpus
are all **Meta-internal**. Velox is open source; Axiom's availability is
unclear; there is no artifact link in the paper. There will be **no direct
Trellis comparison**, and any reviewer will accept that as long as you say it
plainly and build a labeled *Trellis-shaped* proxy (C3 below) instead of
comparing to a strawman and implying it is Trellis.

---

## 2. The baseline roster

Availability and version claims below are from prior repo work or from search
and are marked accordingly; **verify each before it enters a report.**

### A. Same-engine SQL — defends "why not just Postgres?"

| ID | Baseline | Status here | Notes |
|---|---|---|---|
| **A1** | **pgvector HNSW + FK joins + `WITH RECURSIVE`** | **harness exists**: `bench/wd_allpg_baseline.py` | The mandatory baseline. It came within **16 µs** of the fused operator on the anchored class (`benchmark_allpg_baseline_v0.1.0.md`), ~half of it CTE *planning*. Use prepared statements so you beat its strong form, not its weak one. |
| **A2** | **Apache AGE** (Cypher on Postgres) | build | The closest same-engine analogue to Trellis's *surface*: openCypher with `*1..k`, lowered to joins. Being faster than AGE at the same Cypher is a clean, legible result. |
| **A3** | **PG SQL/PGQ `GRAPH_TABLE`** (PG19) | verify availability | The standardization baseline. PG19 lowers GRAPH_TABLE to relational joins with no var-length paths — if that holds, it is a *positioning* result more than a perf one. |
| **A4** | **pgvectorscale** (StreamingDiskANN) / **VectorChord** | build | Stronger vector leg inside the same Postgres. Required if you want to claim anything about the seed leg. |

### B. Single-engine multi-model — the real head-to-head

| ID | Baseline | Why it matters |
|---|---|---|
| **B1** | **Kùzu + NaviX** | **The most important academic baseline.** VLDB'25, native disk-based vector index inside a graph DBMS, Cypher with variable-length paths, published predicate-agnostic numbers vs ACORN/iRangeGraph/pgvectorscale/VBase/Milvus/Weaviate. Also the backend [GEM]'s own MemState prototype uses — and which [GEM] §4.3 calls "a compatibility layer, not a native expression". Caveat: embedded, no network hop; note it, or run everything embedded-equivalent. |
| **B2** | **Neo4j 2026.x** | Native vector index + in-index filtering; the industry default for the graph half; the incumbent in your existing polyglot baseline. |
| **B3** | **FalkorDB** | Purpose-built for AI workloads: embeddings on nodes and edges, one Cypher query doing multi-hop traversal *and* vector kNN. This is literally the workload; it belongs in the table. |
| **B4** | **ArangoDB** | The multi-model incumbent used by UniBench/M2Bench; vector support needs verification. |
| **B5** | Memgraph · SurrealDB · Amazon Neptune Analytics · TigerGraph | Optional. Neptune Analytics integrates vector storage with graph traversal in one store but is cloud-only and metered. |

### C. Polyglot / federated — the status quo *and* the Trellis proxy

| ID | Baseline | Status here |
|---|---|---|
| **C1** | Milvus + Neo4j + Postgres, composed app-side | **exists**: `baseline/docker-compose.yml`, `baseline/harness.py`, `bench/wiki_fusion.py` |
| **C2** | Qdrant + Kùzu + Postgres | build — a lighter, more modern, *fairer* stack than C1 |
| **C3** | **"Trellis-shaped": C1/C2 behind one query API with a cost-based router** | build — **the fairness gate** |

C3 is not optional. Trellis's contribution is the *unified planner over
heterogeneous backends*; an app-side hard-coded pipeline is weaker than Trellis
by construction, so beating it proves nothing about Trellis's architecture. Give
the router at least: predicate pushdown to the relational tier, a
vector-first-vs-filter-first choice, and connection pooling. Then your claim
becomes the honest one — *one process beats a well-planned federation because of
round-trips, serialization, and the absence of a shared snapshot* — which is
exactly what `benchmark_wiki_fusion_v0.1.0.md` already measured (11.5×/3.26×
loopback, 16.7×/10.6× real network).

### D. Vector-leg SOTA — defends "your seed leg is weak"

ACORN (SIGMOD'24), Filtered-DiskANN (WWW'23), iRangeGraph, Milvus/Qdrant/
Weaviate filtered search, pgvectorscale. Run on the **BigANN NeurIPS'23 filtered
track (YFCC-10M, CLIP-192d, 100K queries, QPS at recall@10 ≥ 0.90)**. NaviX
already published against this set, so the setup is reusable and the numbers are
checkable. **You currently lose here** — `benchmark_allpg_baseline_v0.1.0.md`
records plain pgvector matching or beating `tjs_open` on the seedless filtered-ANN
leg at every matched-recall point. Publish it; it buys credibility for everything
else.

### E. Agent-memory application layer — defends "does it matter end to end?"

**Graphiti / Zep** first: the paper names it "**the closest prior work in
spirit**" — a temporal knowledge graph, though aimed at enterprise memory rather
than RSI experience graphs and "without CDC, materialized training views, or a
graph-native query layer". Then **Mem0** (named), **MemGPT/Letta** (named),
**HippoRAG v2**, **GraphRAG**. These are for the paper-plan's correctness/quality
experiments, not for the query-planning experiments.

### F. The status quo the paper attacks — your motivation figure

| ID | Baseline | Why |
|---|---|---|
| **F1** | **OpenEvolve `ProgramDatabase`** (in-memory + JSON checkpoints) | **The single most valuable baseline in this list.** It is real, open, cited by the paper (§8), and structurally identical to what KernelEvolve had before Trellis: process-local state, file checkpoints, no cross-session query, no crash recovery. It also *generates* your workload (§6 of the survey doc). One system that is simultaneously the data generator and the strawman is a gift. |
| **F2** | **SQLite + FTS5 + a vector extension** | The episodic-memory tier production agents actually ship. Cheap, honest, and surprisingly hard to beat at small scale — which is itself worth reporting. |
| **F3** | **MLflow / ModelDB** | §8 makes four falsifiable claims about experiment trackers: no sub-50 ms frontier query, no modeling of mutable search statistics, no AS-OF reconstruction, no graph-native traversal fused with vector retrieval. Testing them is a day of work and either corroborates the paper (good, cite it) or finds it overstated (also good, and publishable). |

---

## 3. Per-experiment baseline matrix

| Experiment | Claim it defends | Baselines |
|---|---|---|
| **W1** vector-seeded expansion: latency + recall vs selectivity × fan-out × hops | the composition is faster fused than routed | A1, A2, A4, B1, B2, B3, C1, C2, **C3** |
| **W2** seed-leg calibration on YFCC-10M | we did not pick a weak vector index | A4, **D** |
| **W3** the eight Trellis operations (Resume/Reuse/Repair/Train/Replay/Observe/Audit/Govern) | one store serves all eight | **F1**, F2, F3, A1, A2, B1, C3 |
| **W4** concurrent tree search: backprop under N workers | the open problem the paper names | C1, **C3**, B1, B2, **F1**; anomaly oracle from `bench/wiki_consistency.py` |
| **W5** AS-OF / time travel at step *t* | a capability gap, not a speed gap | A1 (PG MVCC + valid-time columns), B1/B2 (no cross-store snapshot), F1 (none), F3 (none) |

**W4 and W5 are where TriDB is structurally advantaged and everyone else is
structurally disadvantaged**, and both are named as open problems rather than
solved baselines — which means you are not competing against a tuned incumbent.
W1 is where you compete against genuinely strong systems (B1 especially) and
where the 16 µs result says the margin will be thin. Budget accordingly.

---

## 4. Fairness rules

- **Pin every version** and tune each baseline to its own documented best
  practice; `baseline/TUNING.md`'s "beat it" discipline already exists — extend
  it to B1/B2/B3.
- **Report at matched recall**, per-query paired, with an exact oracle.
- **Publish at least one loss.** You already know two (A1 at 16 µs, D on the
  seedless leg). Landscape review F7 lists this as a credibility requirement.
- **Kùzu is embedded** — no network hop. Either say so on every chart or run a
  matched embedded configuration.
- **Report the sub-50 ms SLO as pass/fail per query**, not as a mean. It is the
  paper's own operational bar and a violation rate is more legible than a
  percentile.
- **C3 must exist before any "one engine beats a federation" sentence is
  written.**

---

## 5. The realistic minimum set

For a first credible submission, five baselines:

1. **A1** all-PG SQL (harness exists — hours)
2. **B1** Kùzu + NaviX (the academic head-to-head — days)
3. **C1 → C3** your existing polyglot stack plus a router (days)
4. **F1** OpenEvolve JSON/in-memory (the motivation figure, and the generator — days)
5. **D** on YFCC-10M for the seed leg (a week, mostly ingest)

B2/B3 add industry legibility and can follow. E is a different paper (see
`paper_plan_agent_memory_v0.1.0.md`) and should not be pulled into this one.

---

## 6. What this does to the dataset survey

`workload_vector_seeded_expansion_v0.1.0.md` stands, with two revisions:

- **STaRK's role changes.** It is no longer "closest to the workload" — the
  workload is a search tree with a threshold-seeded vector predicate and a
  governance filter. STaRK is now the **operator stress test** (a real fan-out
  ladder at 18.2 → 43.5 → 125.2 average degree) and a fine W1 workload, but the
  faithful Trellis workload is **F1-generated experience graphs**.
- **OpenEvolve is promoted from "a generator option" to the primary data
  source and a baseline at once**, because the paper cites it (§8) as one of the
  systems whose traces live in "Python objects and JSON checkpoints".

The honest summary: **no public dataset for this workload exists, and no
comparable system is available — so the credible move is to generate the
workload with OpenEvolve, use OpenEvolve's own store as the status-quo baseline,
and compete against Kùzu/NaviX and a well-planned federation for the systems
claim.**
