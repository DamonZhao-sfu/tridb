# A SIGMOD/VLDB paper from the GEM branch — thesis, positioning, and experiment design

> **Version:** 0.1.0
> **Date:** 2026-08-17
> **Status:** Research plan. **Contains no measurements.** Every number cited is
> either a committed prior result (cited to its report) or explicitly marked
> NOT RUN.
> **Reads on:** [`agent_memory_gem_interface_v0.1.0.md`](agent_memory_gem_interface_v0.1.0.md)
> (the GEM operator design), [`agent_memory_reproduction_plan_v0.1.0.md`](agent_memory_reproduction_plan_v0.1.0.md)
> (gap register G0–G12), [`agent_memory_gem_wiki_demo_v0.1.0.md`](agent_memory_gem_wiki_demo_v0.1.0.md)
> (the one live GEM measurement that exists), [`benchmark_allpg_baseline_v0.1.0.md`](benchmark_allpg_baseline_v0.1.0.md)
> (the result that constrains the whole thesis), and
> [`landscape_review_v0.1.0.md`](landscape_review_v0.1.0.md) §2–§4.

---

## 0. The one-paragraph version

The naive framing — *"a multi-model DB is a better agent memory, look how fast
our fused operator is"* — **cannot be published**, and this repo already owns
the evidence that kills it: on the anchored query class, plain SQL in the same
Postgres is within **16 µs/query** of the fused operator
(`benchmark_allpg_baseline_v0.1.0.md`), and at the seedless filtered-ANN class
plain pgvector **wins**. A PC member will find that report. The publishable
thesis is one level up: **agent memory is a stateful, mutable, multi-model
workload whose correctness conditions are transactional, and today every
deployed agent-memory system implements those conditions as application
conventions over disjoint stores, where they are unenforceable.** The
contribution is (i) naming and measuring that failure class, (ii) showing the
conditions compile into ordinary engine mechanisms — constraints, commit
postconditions, one transaction — and (iii) showing enforcement is *cheaper*
than the polyglot stack's weaker guarantee. Retrieval speed is a supporting
result, not the claim.

---

## 1. Why the obvious paper fails, in detail

Four objections a SIGMOD/VLDB PC will raise against "multi-model DB as agent
memory", each already answerable from this repo — which is exactly why the
paper must be designed around them rather than into them.

| Objection | Repo evidence that it lands | Consequence for the paper |
|---|---|---|
| "Why not one Postgres, pgvector + a `links` CTE, no extension?" | `benchmark_allpg_baseline_v0.1.0.md`: fused beats all-PG SQL by **1.33× / ~16 µs**; ~half of that deficit is *CTE planning*, reclaimable with prepared statements. Seedless: **pgvector wins**. | The all-in-one-Postgres SQL stack must be a **first-class baseline**, not a footnote. The read-path win is not the thesis. |
| "Multi-model in one engine is a crowded category" | `landscape_review_v0.1.0.md` §3 lists CHASE, ARCADE, M2, SurrealDB, HelixDB and concludes tri-modality per se is **"not defensible"** as a category claim. | Novelty must be the *workload* (mutable agent memory) and the *mechanism*, never the architecture diagram. |
| "This is an integration/experience paper" | Fair unless a real mechanism is on the table. | Lead with mechanisms: correctness-as-commit-postcondition, retrieval-as-a-HOT-write, two-type native edge encoding for bounded entailment propagation, early-terminating fused Open/Next/Close. |
| "Your accuracy loses to BM25" | [AM] (arXiv:2606.06448) finds **BM25 is the aggregate accuracy winner** (47.0% LongMemEval, 55.8% macro) *and* the cheapest system. | **Do not evaluate the headline on static LongMemEval QA.** Evaluate where memory is *mutated*: conflict, supersession, drift, multi-session freshness. That is the regime BM25 and flat RAG structurally cannot serve. |

The naming trap, separately: in a DB venue **"multi-modal" reads as
vision-language**. Use **multi-model** (vector / graph / relational) throughout,
and say "tri-modal" only when quoting our own prior artifacts.

---

## 2. The thesis, stated as three falsifiable claims

> **T.** Long-horizon agent memory is a transactional multi-model workload. Its
> correctness conditions are trajectory-level (what the store returns *after* a
> sequence of updates), not record-level, and they are enforceable as engine
> mechanisms in a single-process multi-model DBMS while being structurally
> unenforceable across a polyglot memory stack.

| Claim | Statement | Falsified by |
|---|---|---|
| **C-I Correctness** | Under concurrent sessions and injected failure, polyglot agent-memory stacks exhibit a measurable, non-trivial rate of memory-state anomalies (duplicate current values, lost updates, dangling edges, orphaned vectors, torn cross-modal reads). A single-engine realization drives them to zero *without* application-level compensation. | Anomaly rate ≈ 0 in the polyglot baselines once a competent outbox/saga mitigation is applied. |
| **C-II Cost** | Enforcing the conditions costs less than the polyglot stack pays for a *weaker* guarantee — end-to-end, including the C6 retrieval-write and the maintenance path. | Governance overhead exceeds the polyglot's compensation overhead. |
| **C-III Quality** | On update-bearing workloads, governed multi-model memory answers with fewer stale/conflicted facts and higher multi-hop evidence recall than append-only flat or graph RAG at equal construction budget. | No gap outside noise once construction budget is matched, or the gap is attributable to the extractor rather than the store. |

C-I is the paper. C-II makes it a systems paper. C-III stops it being only a
correctness paper.

---

## 3. Positioning (what to cite, and how)

- **[AM] arXiv:2606.06448** (Omri et al.) — supplies the workload
  characterization and the four-paradigm taxonomy. Cite as *the motivation we
  extend*: it measures nine systems' cost but assumes the store is correct and
  never measures update anomalies. Our angle: their Insight "construction
  dominates" is a *cost* claim; we add the *correctness* claim they leave open.
- **[GEM] arXiv:2605.26252** (Orogat & Mansour, *Is Agent Memory a Database?*)
  — supplies `M_t = (D_t, S_t, P_t)`, the four state-level operators, and
  conditions C1–C6. Cite as *the abstraction we implement natively*: their §4.3
  says the Kuzu prototype is "a compatibility layer, not a native expression of
  GEM" and lists what a native engine must supply. **This paper is the
  empirical answer to their title question.** That framing alone is worth the
  submission — it is a named open problem with a named requirements list.
- **NaviX (VLDB'25)** — benchmarks VBASE by name and criticizes post-filter
  collapse and the missing recall knob; our lineage fixed both (`term_cond`,
  FR-6). Use their evaluation template (selectivity ladder × predicate
  correlation × recall/QPS curves) — reviewers will expect it.
- **ACORN / pgvector iterative scan / VectorChord** — filtered-ANN state of the
  art; the read-path baselines.
- **Mem0, Zep/Graphiti, HippoRAG v2, GraphRAG, A-Mem, Letta, MIRIX, Cognee** —
  the systems under test, not just related work.
- **PhaseGraph (arXiv:2603.28886)** — PPR and dense-similarity scores are
  distributionally incomparable without calibration; the principled reason the
  app-side fusion baseline is *ill-posed*, not merely slow. Load-bearing for
  C-III.
- **Rank-join theory (Ilyas 2008; Tziavelis SIGMOD'20)** — the route from
  `term_cond` as an empirical knob to a certified bound. Flagged in
  `landscape_review_v0.1.0.md` §2.5 as "the most publishable open item"; if it
  lands it upgrades the fused operator from engineering to a result.
- **PG19 native SQL/PGQ** lowers `GRAPH_TABLE` to relational joins with no
  var-length paths — cite as the standardization tailwind: this work is the
  *native executor* for the surface Postgres is standardizing.

---

## 4. Experiment design

Ten experiments. **E1, E2, E5 are the paper**; the rest support or defend.
Status column: `have` = machinery exists and has produced a committed result ·
`harness` = code exists, never run in a recorded run · `build` = does not exist.

### E1 — The anomaly study (the opening figure) · `harness`

*Claim C-I.* N concurrent agent sessions execute retrieve→decide→update cycles
against a **shared** memory scope, with a workload deliberately containing
conflicting updates to the same fact, entity merges, and edge rewires.

- **Stacks:** (a) Mem0 over its default store, (b) Zep/Graphiti over Neo4j +
  an embedding store, (c) HippoRAG v2, (d) a hand-built app-side
  vector+graph+relational stack **with** an outbox/saga mitigation, (e) the
  same stack **without** mitigation, (f) TriDB/GEM.
- **Anomaly oracle** (this is the contribution, not the plumbing): after each
  trajectory, replay the ground-truth operation log and check
  1. **duplicate current values** for one (unit, field),
  2. **lost update** (a committed supersession invisible later),
  3. **dangling edge** (edge to a vertex whose row is not visible),
  4. **orphaned vector** (embedding without its record, or vice versa),
  5. **torn cross-modal read** (legs disagree within one query),
  6. **provenance break** (a superseded value with no successor).
- **Metric:** anomalies per 1,000 memory operations, by class, vs concurrency
  level (1/2/4/8/16 sessions).
- **Reuse:** `bench/wiki_consistency.py` already implements exactly this
  methodology for generic writes — S1 atomicity, S2 crash, S3 torn reads — and
  produced TriDB **0** torn vs multi-store **42/42** injected torn; torn reads
  **1.0%** vs **76.7%** (`benchmark_wiki_consistency_v0.1.0.md`). E1 is that
  harness re-pointed at GEM operations instead of wiki writes.
- **Fairness rule (non-negotiable):** stack (d) must exist and must be a
  *competent* mitigation. `docs/STATUS.md` already concedes the multi-store
  tear is "mitigable app-side (2PC/sagas/outbox) at real complexity/latency
  cost". A paper that only beats the unmitigated stack is a strawman paper.
  The honest result is a **cost-of-mitigation** curve, not a zero-vs-nonzero
  bar.

### E2 — Failure injection and recovery · `harness`

*Claim C-I.* `kill -9` mid-consolidation (the operation that supersedes a
field, rewires k edges, and re-embeds). Measure post-recovery state: all-old,
all-new, or torn; time to a consistent store; whether any compensation is
needed. Protocol S1/S2/S3 from `benchmark_wiki_consistency_v0.1.0.md`.

**Known threat, must be closed first — see §6.1:** TriDB's own graph leg is
*commit-visible, not snapshot-isolated* (DEV-1166), which is why the prior run
reports a **1.0% residual tear on the graph leg**. A paper claiming isolation
with a 1.0% self-reported isolation defect will be desk-rejected on that line.

### E3 — The price of governance · `harness`

*Claim C-II.* The delta neither [AM] nor [GEM] reports. Ablate one switch at a
time from the same construction: `reinforce` off→on, `revise` off→on, `forget`
off→on, `mode` VECTOR→FUSED. Report per-operation latency (p50/p95/p99), WAL
bytes, dead tuples, index churn, and end-to-end serving cost.

The micro-result worth a subsection: **C6 makes every retrieval a write, and
that write is free** — the salience/access columns are unindexed and the tables
are `fillfactor=70`, measured at **100% HOT, +0 bytes** vs 84% HOT and relation
growth at the default fillfactor (`agent_memory_gem_interface_v0.1.0.md` §6.3,
`bench/agent_memory/gem/schema.sql:42-47`). Contrast with re-embedding: **0%
HOT, ~369 B/update**, an HNSW insert plus a dead tuple each time — which is why
`revise` batches embedding refreshes (`bench/agent_memory/gem/revise.py:439`).
"Governance is affordable *because* of a physical-design decision" is a genuine
DB result and it is already measured.

### E4 — Maintenance throughput · `build`

*Claim C-II.* Consolidation / conflict-resolution / merge / forget throughput
in ops/s at matched semantics: TriDB one transaction vs stack (d)'s
saga-compensated multi-step. Report tail latency and the compensation failure
rate under concurrency. Bytes reclaimed, recall before vs after.

### E5 — Retrieval quality under drift (the accuracy story that survives BM25) · `build`

*Claim C-III.* **Do not lead with LongMemEval.** Lead with workloads where the
memory is mutated:

| Workload | Source | What it isolates |
|---|---|---|
| FactConsolidation-SH / MH | MemoryAgentBench | selective forgetting / conflict resolution |
| MemoryArena (arXiv:2602.16313) | 20 multi-session physics tasks, retrieve-act-write cycles | freshness and write-visibility |
| **MemDrift (ours)** | synthetic generator, see §5 | controlled supersession rate with trajectory ground truth |
| LongMemEval / LoCoMo | — | **control only**, to show we are not worse on static QA |

**The headline metric is new and is ours to define: `stale-answer rate`** — the
fraction of answers grounded in a value that had been superseded at query time.
A C1-enforcing store has a structural floor of zero; append-only stores
(Paradigm II and III.a — GraphRAG, HippoRAG v2, embedRAG, BM25) have no
mechanism to select between two coexisting values and should show a measurable
rate rising with supersession density. **Plot stale-answer rate vs supersession
rate.** If that curve separates, the paper is accepted; if it does not, the
thesis is wrong and you learn it cheaply. Run this pilot *first* (§7).

### E6 — Modality ablation on memory workloads · `harness`

*Claim C-III.* vector only · graph only · relational+vector · vector+graph ·
fused `tjs_open` · **materialized three-leg composition in the same Postgres**
· **all-PG SQL (pgvector + recursive CTE)** · polyglot Milvus+Neo4j+PG.

The two bolded baselines are the ones that make this credible; per
`benchmark_allpg_baseline_v0.1.0.md` the all-PG SQL baseline is *within 16 µs*
on anchored queries, so **report that honestly and locate the regime where it
breaks**: high fanout, deeper hops, and bounded-work early termination, where
the CTE must materialize the reach set and the operator does not. State the
regime boundary as a measured crossover, not a claim.

Existing anchor: on the live GEM wiki slice, FUSED reached **joint evidence
recall@10 0.912** vs VECTOR **0.864** at 2.535 s vs 1.973 s aggregate, with
`term_cond` termination on 125/125 queries and **zero** budget-capping or
graph-censoring (`agent_memory_gem_wiki_demo_v0.1.0.md`). Small corpus, in-domain
pool — a pilot, not a result.

### E7 — Freshness and write visibility · `build` (G5)

*Claim C-I/C-II.* Replay session arrivals at the [AM] protocol's fixed 5 s gap
and Poisson means 0.1/1/5/30/60 s; synchronous vs asynchronous construction;
staleness = prior sessions not yet visible at query admission. TriDB's sharper
version: **commit visibility is one WAL fact**, not an eventual-consistency
guess across services — so we can report *exact* staleness where the polyglot
stacks can only estimate it. [AM] Figure 8b marks six systems accumulating
staleness under async scheduling; this is the axis where a single-engine store
is definitionally better and it has not been measured.

### E8 — Bounded footprint · `harness` (partly)

*Claim C-II.* 64K→1M history sweep, 1/10/1K users, with the C5 forgetting
ladder **on** — producing a *bounded* active footprint where all nine [AM]
systems grow unboundedly (their §4.7: 0.7 TB → 6.2 TB at 100K users, ~9×
spread). Report bytes/scope split heap/index/graph, plus recall before vs after
attenuation so that "bounded" is not bought with silent quality loss.
`bench/agent_memory/gem_bench/scaling.py` implements the sweep; note its
`physical_isolated: true` requirement.

### E9 — Tails and multi-client · `build`

`landscape_review_v0.1.0.md` F7 names median-only single-client latency as a
known credibility gap ("the standard VectorDBBench critique applies
verbatim"). p50/p95/p99 + QPS at multiple client counts, warm/cold labeled,
for every latency claim in the paper.

### E10 — Construction-form cost matrix ([AM] reproduction) · `harness`

Secondary section. `bench/agent_memory/gem_bench/` runs five arms
(II / III.a / III.b / IV / GEM-conformant) through one harness with
`paradigm_proxy: true` on every record. **Reviewers will not accept proxies as
a system comparison** — present this strictly as an internal cost-shape
ablation ("construction form held everything else fixed"), and use real
external systems (E1/E5) for anything comparative.

---

## 5. MemDrift — the benchmark contribution

[GEM]'s own research agenda asks for a trajectory benchmark with ground truth
at three levels: the current value of each unit over time (C2), the dependents
that change after each update (C3), and the active footprint at each
interaction count (C5). No such benchmark exists.
`landscape_review_v0.1.0.md` §2.3 independently reaches the same conclusion
from the DB side ("the cross-modal benchmark doesn't exist... category-defining
if the hygiene lands first").

A generator with: parameterized supersession rate, entailment-graph density and
depth, entity-merge rate, multi-tenant scope overlap, and a replayable arrival
clock — emitting, for every timestep, the ground-truth current value set, the
correct dependent set, and the correct active set. Plus the anomaly oracle from
E1 and the stale-answer metric from E5.

This is plausibly *half the paper's citations*. `gem_transition`
(`bench/agent_memory/gem/schema.sql:166`) already logs the trajectory — operator,
delta, policies evaluated, committed/aborted, `active_units`/`active_fields` —
in the same row as the [AM] phase telemetry, so the instrument exists; the
generator and the oracle do not.

---

## 6. What must be built or fixed before submission

### 6.1 P0 — blockers that a reviewer will find in five minutes

1. **DEV-1166: snapshot isolation on the native graph leg.**
   `gph_xmin_visible()` uses `TransactionIdDidCommit`, not `XidInMVCCSnapshot`
   (`agent_memory_gem_interface_v0.1.0.md` §6.5, reproduced live). A paper whose
   central claim is transactional correctness **cannot ship with a self-reported
   1.0% torn-read residual on its own graph leg.** Either implement
   `GraphTupleSatisfiesSnapshot` (already specified in
   `graph_store_layout_v0.1.0.md`) or scope every claim to *atomicity +
   constraint enforcement* and say "isolation of the graph leg is future work"
   in the abstract — which costs roughly half of C-I. **Recommendation: fix it.**
2. **Real external baselines on our hardware** (gap G3). Mem0, Zep/Graphiti,
   HippoRAG v2 at minimum, pinned versions. Paradigm proxies are not a
   comparison. This is the single largest cost item in the plan.
3. **Run the live GEM test suite green.** `tests/test_gem_live.py` and the live
   half of `tests/test_gem_conformance.py` are skip-gated on `TRIDB_GEM_DSN` and
   per `agent_memory_gem_implementation_status_v0.1.0.md` §2 **have never been
   executed**. The wiki demo covers C1–C6 on one slice; the suite must be green
   on demand.
4. **A judge protocol that is not us.** Gap G12: the local self-judge is a
   protocol variant and must never be reported as paper-equivalent. Budget for
   an independent judge plus a human-verified subset.

### 6.2 P1 — needed for the specific experiments

- MemDrift generator + anomaly oracle + stale-answer metric (§5) — E1, E5.
- FactConsolidation and MemoryArena fetchers (gap G4) — E5.
- Arrival/scheduling driver with staleness accounting (gap G5) — E7.
- Saga/outbox-mitigated polyglot baseline (E1 stack d) — the fairness gate.
- Multi-client tail harness (gap: F7) — E9.

### 6.3 Explicitly cut

- **The GX10 fork and every 128 GB claim.** The stock PG 16/17 x86_64 path is a
  major repro asset (`docs/INSTALL_stock_pg.md`, CI job `stock-pg`) — a
  reviewer can rebuild it. Put nothing on the fork's critical path.
- **Energy / joules.** [AM] owns that contribution and does it on an H100 we do
  not have. Keep `bench/agent_memory/energy.py` for internal cost work; report
  it, if at all, as a secondary table with `paper_hardware_match: false`.
- **Paradigm IV / agentic ingest.** Interesting, expensive, and not on any of
  the three claims.
- **The BM25 seam.** CLAUDE.md rule 5 keeps it closed; BM25 stays an *external
  baseline system*, which is also how [AM] treats it.

---

## 7. Sequencing — the cheapest path to knowing whether the paper exists

The thesis is falsifiable early and cheaply. Do that first.

| Step | Deliverable | Kills the paper if |
|---|---|---|
| **S1 — the drift pilot** (~2 weeks) | MemDrift v0 at three supersession rates; TriDB/GEM vs embedRAG vs one real append-only system; stale-answer rate curve | the curves do not separate |
| **S2 — the anomaly pilot** (~2 weeks) | E1 at 1/4/16 sessions, stacks (e) and (f) only | anomalies are ~0 in the unmitigated polyglot stack |
| **S3 — DEV-1166** | graph-leg snapshot visibility, re-run S3 torn-read protocol | residual tear does not go to 0 |
| **S4 — the fairness gate** | stack (d), saga/outbox mitigation | mitigation is free (it will not be — measure the cost) |
| **S5 — full E1/E2/E5 + E3/E6** | the paper's four figures | — |
| **S6 — E4/E7/E8/E9/E10** | supporting sections | — |

If S1 and S2 both come back positive, the paper is real and the rest is
execution. If either comes back flat, the fallback is an **experiments-and-
analysis** submission — "the first correctness characterization of agent-memory
stacks", which the anomaly oracle and MemDrift support on their own even if
TriDB is not the winner.

---

## 8. Venue

- **PVLDB (rolling)** — recommended. Systems-friendly, rolling deadlines fit an
  execution-bound plan, and the E&A track is a genuine fallback for the same
  work if the system contribution thins out.
- **SIGMOD** — viable if the rank-join / certified-`term_cond` result
  (`landscape_review_v0.1.0.md` §2.5) lands, which would give the fused operator
  a theoretical spine rather than an empirical knob.
- **CIDR** — the fastest way to plant the flag. "Is agent memory a database?
  We measured it" is a CIDR abstract almost verbatim, and a CIDR paper does not
  spend the external-baseline budget. Consider it as a *parallel* track, not a
  replacement.

Working title candidates (multi-**model**, never multi-modal):

- *Agent Memory Is a Transactional Workload*
- *Governed Memory: Trajectory-Level Correctness for LLM Agents in One Engine*
- *Answering "Is Agent Memory a Database?" — Empirically*

---

## 9. Threats to acceptance, and the prepared answer

| Threat | Answer |
|---|---|
| "Your win is an artifact of an unmitigated baseline" | E1 stack (d) exists and is measured; the result is a cost curve, not a bar. |
| "16 µs — the operator does not matter" | Conceded and reported (E6). The claim is the write path and the correctness path; the read-path regime boundary is measured, not asserted. |
| "Your extractor differs, so accuracy differs" | Construction budget matched; the GEM-conformant arm deliberately shares Paradigm II's deterministic ingest so the delta is governance alone (`gem_bench/README.md`). |
| "You claim isolation but your graph leg is commit-visible" | §6.1 item 1. Fix before submission. |
| "Median-only latency" | E9. |
| "Proxies, not systems" | E10 is labeled an internal ablation; all comparative claims use real systems. |
| "Single-node, single-tenant" | Multi-tenant scopes are in the schema and the C6 salience-leakage hazard is already filed as an explicit gate (`agent_memory_gem_interface_v0.1.0.md` §7c). Report it as a limitation with a measured leakage rate — reviewers reward that. |

---

## 10. Honest inventory

**Committed, measured, reusable:** the fusion speed win (3.3–16.7× at 200k;
23.68× on the 1M Wikidata slice at matched recall, stock PG 17), the
cross-modal consistency demo (0 torn vs 42/42; 1.0% vs 76.7% torn reads), the
PPR default (ADR-0021), the HotpotQA graph-bridge lift (+15.6 pt), the all-PG
honest tie, and the GEM wiki demo's live C1–C6 evidence plus its
VECTOR-vs-FUSED operating points.

**Authored but never run in a recorded run:** the LongMemEval and LoCoMo
end-to-end pipelines, `gem_bench` (all five arms), the live GEM test suite.

**Does not exist:** every experiment in §4 marked `build`, all external
baselines, MemDrift, the anomaly oracle, the stale-answer metric, the arrival
driver.

**No agent-memory number in this document has been measured.** The GEM demo
figures are the only live agent-memory results in the repository, and they are
one slice on stock PG 16.
