# GEM Interfaces on TriDB — Implementation Plan

> **Version:** 0.1.0
> **Date:** 2026-07-30
> **Status:** Interface specification + implementation plan. The type/protocol
> layer (`bench/agent_memory/gem/`) and the schema are landed and exercised;
> the operators are specified, not implemented. **No measurements.**
> **Sources (both read first-hand from PDF):**
> - **[GEM]** A. Orogat, E. Mansour, *Is Agent Memory a Database? Rethinking
>   Data Foundations for Long-Term AI Agent Memory*, arXiv:2605.26252v1
>   (2026-05-25), Concordia University. Code: `CoDS-GCS/MemState`.
> - **[AM]** Y. Omri et al., *Agent Memory: Characterization and System
>   Implications of Stateful Long-Horizon Workloads*, arXiv:2606.06448v1.
> **Companions:** [`agent_memory_system_design_v0.1.0.md`](agent_memory_system_design_v0.1.0.md)
> (seven-stage gap analysis), [`agent_memory_reproduction_plan_v0.1.0.md`](agent_memory_reproduction_plan_v0.1.0.md)
> (experiment gap register).

## 1. Why the two papers compose

[GEM] gives the **abstraction**: memory is a state `M_t = (D_t, S_t, P_t)`
evolving under four state-level operators, whose correctness is a property of
the trajectory `{M_t}` rather than of any record. [AM] gives the **workload**:
four construction forms, a phase-aware cost model, and eight experiments.

They meet at one seam. [GEM]'s `ingest` operator has a strategy slot; [AM]'s
central axis is *which construction form fills that slot*, because the choice
moves order-of-magnitude cost between the write and read paths. So:

> **One memory implementation with swappable strategies yields [AM]'s taxonomy
> as configurations, not as nine separate systems.**

That is the design thesis of this plan, and it is what makes reproducing [AM]'s
experiments tractable for us (§7).

There is a second, sharper reason to build on [GEM] specifically. Its §4.3 says
MemState's Kuzu backend is "a compatibility layer, not a native expression of
GEM", and lists what a **native engine** must supply:

| [GEM] native-engine requirement | TriDB status |
|---|---|
| (i) co-locate topics, field histories and embeddings on pages | partially — CSR-lite page work exists but is not applied to this layout. **Out of scope here.** |
| (ii) unified indexing: semantic + temporal + structural queries through one physical organisation **without duplicating data** | **this is `tjs_open`.** One operator, one heap, vector + native graph + relational predicate |
| (iii) **retrieval as a write** — one operator unifying search, traversal, temporal lookup and salience update | TriDB can commit the read and the salience write in **one transaction, one WAL**. MemState needs Kuzu's transaction for the same thing; TriDB additionally gets the vector and the graph edge into that same commit |
| policy postconditions compiled into the commit protocol | constraints + the atomic commit; §4.2 below shows one already enforcing a [GEM] correctness condition |

TriDB is closer to [GEM]'s "native engine" than the prototype the paper ships.
That is a claim worth *testing*, not asserting — §8 sets the gates.

## 2. Direct answers to the two questions asked

### 2.1 Is semantic revision supported today? **No — at any layer.**

Verified against the working tree at `e5810ff`:

| What revision needs ([GEM] §3.2, §4.2) | TriDB today |
|---|---|
| Semantic units with **field value histories** `H = <(v,t,π)>` | ❌ `agent_memory_units` stores one immutable `content` blob per row. There is no field, no history, no supersession |
| **Provenance** π retained on supersession (C4) | ❌ only an unstructured `metadata jsonb` |
| **Typed edges carrying propagation rights** (C3) | ❌ `register_edge_type()` gives a name→id map with **no propagation semantics**. `moved_to` and `part_of` are indistinguishable to the engine |
| **Propagation** on update, along entailment edges only | ❌ nothing propagates; nothing walks |
| **Policies** `P_t` checked as a commit postcondition (C2) | ❌ no policy concept |
| Conflict repair / merge / topic split | ❌ none |

So today TriDB exhibits [GEM]'s **Failure ②** exactly as described: an updated
fact would be appended beside the outdated one and both would compete at
retrieval on embedding similarity alone.

**But the substrate is unusually well suited to fixing it**, and one piece is
already proven — see §4.2.

### 2.2 Does retrieval reinforce facts and update memory structure? **No.**

`search()` and `search_fused()` are pure reads. Nothing increments a salience
signal, records an access, or touches structure. That is [GEM]'s **Failure ④**,
and per **Observation 1** it cannot be patched around the outside: "Caches,
materialized views, and post-retrieval triggers can record that an access
occurred, but they cannot lift the operator out of being a pure function,
because the state-modifying step is decoupled from the query." The salience
update has to be *inside* the retrieval operator's transaction. §5.3 specifies
it there.

## 3. The state model on TriDB

Landed: `bench/agent_memory/gem/types.py`, `schema.sql`.

```
D_t  gem_unit          semantic unit ("topic"): title, summary, routing vector,
                       state, salience.  id == graph vid (the tjs_open contract)
     gem_field_value   H = <(v, t, π)> per field: value, valid_from, valid_to,
                       superseded_by, provenance columns, PER-FIELD salience
S_t  native graph AM   topology — typed edges, never a relational join table
     gem_edge          relational metadata the AM cannot carry: propagation
                       right (extension|association), rel name, co_access_count
P_t  gem_policy        <event, condition, action>, living inside the state
     gem_transition    the trajectory {M_t} itself
```

Three details are load-bearing rather than cosmetic:

**Topic grain, not entity grain.** A unit holds every field of one concept.
[GEM] §4.1 is explicit that entity-grain designs (Zep) "scatter attributes
across nodes, requiring multiple accesses to reconstruct one concept" and that
topic grain is what keeps the C3 propagation frontier small enough to terminate
in a few hops.

**Two edge kinds, separately registered in the native AM.** `extension:` edges
carry entailment and are the *only* ones revision traverses; `association:`
edges expand retrieval context and never propagate. Registering them as
distinct native edge types means the operator can traverse one kind without
seeing the other, using the existing `gph_traverse_typed` type filter.

**Per-field salience.** [GEM] §3.2 requires sub-unit granularity — "part of a
unit may be attenuated while the rest stays current" — so the ladder applies to
`gem_field_value`, not only to `gem_unit`.

### 3.1 One log for both papers

`gem_transition` carries the [GEM] trajectory (operator, delta, policies
evaluated, committed/aborted, and `active_units`/`active_fields` = C5's
`|D_t^active|`) **and** the [AM] §3.3 phase telemetry (phase, seconds, LLM
calls, embed calls, embed sequences, prompt/completion/embed tokens, DB
statements, `gpu_joules`) in the same row. [GEM]'s research agenda asks for a
trajectory benchmark with ground truth at three levels — current value over
time (C2), dependents that change after each update (C3), active footprint at
each interaction count (C5). This log yields all three, and the same rows feed
[AM]'s Table 3. One instrument, two papers.

## 4. What is already proven on the live engine

Built from source in-container: **PG 16.14 + pgvector 0.8.1 + graph_store_am
0.2.0 + tjs_pg 0.2.0.**

### 4.1 Tri-modal retrieval with the relational predicate pushed in

Already landed in `TriDBMemoryBackend.search_fused()` (PR #1) and measured: a
query ≈"Alice" returns **Berlin** only in fused mode — Berlin's vector is
orthogonal to the query, it arrives across the `moved_to` edge — while the
scope predicate holds inside the operator (a `user_b` query returns no `user_a`
rows). Probes returned per call.

### 4.2 C1 + C4 are enforced by the schema, not by convention

[GEM] **Observation 3a**: *"Append-only storage without semantic units cannot
satisfy C2. Two appended values for the same fact coexist with equal status,
and a default query has no engine-level mechanism to select between them."*

The partial unique index in `schema.sql` **is** that engine-level mechanism:

```sql
CREATE UNIQUE INDEX gem_field_value_current_uq
    ON gem_field_value (unit_id, field)
    WHERE valid_to IS NULL AND state <> 'archived';
```

Run live against the paper's own Figure 1 scenario:

```
-- append a second current deadline (the paper's Failure ②)
ERROR:  duplicate key value violates unique constraint "gem_field_value_current_uq"
DETAIL:  Key (unit_id, field)=(0, deadline) already exists.

-- correct revision: supersede then append
C1  default query returns ONLY:   April 20
C4  history retained:             March 15 (valid_to=2026-03-08, superseded_by=3)
                                  April 20 (current)
```

**The failure mode [GEM] attributes to missing engine support is prevented at
commit by a Postgres index.** This is the single cheapest piece of evidence for
the "TriDB is closer to the native engine" claim, and it cost one index.

## 5. The four operator interfaces

Landed as protocols in `bench/agent_memory/gem/protocols.py`. Every operator is
**one transaction**: the proposed `M_{t+1}` is checked against `P_t` and either
commits atomically or aborts ([GEM] Algorithm 1, line 12). On TriDB that
transaction spans the relational row, the vector, **and** the native graph edges
together — one transaction manager, one WAL.

### 5.1 `ingest(events, *, strategy) -> IngestResult`

Integrates `I_t` into the existing state under `P_t`. The strategy is the
[AM] construction-form axis:

#### `DeterministicIngest` — no LLM ([AM] Paradigm II, embedRAG)
Chunk (4096 tokens, the MemoryAgentBench streaming protocol) → batch embed →
one unit per chunk with a single `content` field. No extraction, no
consolidation, no edges. **This is exactly TriDB's current behaviour**, moved
behind the interface so the embedRAG row reproduces unchanged and the
construction-cost comparison stays apples-to-apples.

#### `LLMMediatedIngest` — LLM as a fixed extractor ([AM] Paradigm III)
The LLM reads unit titles and summaries, picks a host unit ([GEM] §4.2's
recipe), and emits `(field, value, timestamp, provenance)` against a **pinned
prompt** and a **versioned output schema**. Two sub-modes, because [AM] §4.3
found embedding traffic is bimodal by paradigm and the regimes stress a serving
stack differently:

- **`batch`** (III.a, GraphRAG/HippoRAG-like) — large offline embedding
  batches, append-only. [AM] measured GraphRAG at ~2,300 sequences per
  embedding call, HippoRAG v2 at ~125.
- **`sequential`** (III.b, Mem0/SimpleMem-like) — embed each extracted fact
  *before* its similarity search resolves ADD/UPDATE/DELETE, giving the 1:1
  call-to-sequence ratio [AM] identifies as the latency-sensitive write-loop
  signature.

`validate()` is **not optional**. [AM] §4.4 shows that below an
algorithm-specific capability floor a weak construction model does not merely
lower accuracy — it *corrupts the store* (MIRIX fails outright at Qwen3-1.7B).
A unit failing schema or referential validation is rejected, rolled back, and
counted in `IngestResult.rejected`. **A structural failure is a failed
configuration, not an accuracy datapoint.**

#### `AgenticIngest` — LLM-controlled writes ([AM] Paradigm IV)
The model gets memory tools (`search_memory`, `read_unit`, `write_field`,
`link`, `split_topic`) and loops until it stops. `max_rounds` and
`max_tool_calls` are **mandatory fields**, not options: [AM] Recommendation 10
is that LLM-bounded phases need external iteration caps, with p95/p50 tails up
to 5.9× as the evidence.

### 5.2 `revise(scope_id, *, evidence, max_hops) -> RevisionResult`

Detects evidence in `M_t` — duplicate units, conflicting field values, schema
drift, dependency inconsistencies — and applies the matching repair:

- **conflict**: mark the superseded value with its provenance, never delete
  (C4). The partial unique index makes a violating result abort (C2).
- **merge**: fold duplicate units, union field histories.
- **propagate**: walk **extension edges only**, halting at any unit whose
  policy condition does not fire (C3). Bounded by `max_hops`; the frontier
  stays small because the grain is topics, not entities.
- **split**: promote a field subset into a standalone unit ([GEM] Figure 3 —
  "Alice splits out of Website Redesign").

Runs asynchronously as policy-triggered maintenance, like `forget`.

### 5.3 `forget(scope_id) -> ForgetResult`

Graded attenuation by **relevance**, never by age or capacity, at per-field
granularity. Three thresholds: below `θ_summary` compress the history, below
`θ_remove` hide from active retrieval, below `θ_archive` archive but keep
recoverable (C5's "archived content remains recoverable"). Never destructive
(C4). Hidden/archived state is excluded by the **relational predicate pushed
into `tjs_open`**, not by post-filtering — so attenuated content costs nothing
at retrieval instead of being fetched and discarded.

### 5.4 `retrieve(query) -> RetrievalResult` — the state-modifying one

Returns an output **and** commits a transition. Three axes:

- **`mode`**: `VECTOR` | `GRAPH` | `RELATIONAL` | `FUSED`. `FUSED` is the
  tri-modal path through `tjs_open` — vector leg, native-graph leg over
  extension+association edges, relational predicate (`scope_id`, `state =
  'active'`, session/time) pushed *into* the operator.
- **`route`**: `TOPIC` | `TEMPORAL` | `STRUCTURAL`, MemState's three routing
  modes. `TEMPORAL` + `as_of` is how C1's "prior values appear only when `q`
  explicitly requests historical context" is honoured.
- **`reinforce`** (C6): in the same transaction as the read — increment
  salience of every accessed unit *and field*, bump `access_count`/`last_access`,
  and **update structure**: co-retrieved units strengthen their association
  edge's `co_access_count`, and an association edge crossing a policy threshold
  becomes an extension candidate for `revise` to confirm. That is the concrete
  reading of *"Retrieval reinforces important facts and updates memory
  structure."*

`SaliencePolicy.reinforce` must be **strictly increasing** in hits, because C6
requires repeated retrieval to *strictly* reduce eligibility for attenuation —
it is the coupling between C6 and C5 that makes retention relevance-driven
rather than age-driven.

## 6. Three design tensions, flagged not buried

**(a) C6 changes what [AM] §4.8 measures.** With `reinforce=True`, retrieval
latency includes a write. TTFT is no longer a pure read path. That is not a bug
— it is the cost of governed memory — but it must be reported as a distinct
operating point. **`reinforce` is therefore an ablation switch**, and the
*delta* between the two is a measurement **neither paper reports**: the price
of C6. That is the most interesting new number this design can produce.

**(b) Forgetting breaks benchmark isolation.** C5 changes the corpus between
queries, while LongMemEval/LoCoMo assume a fixed corpus per question. So in
paradigm-faithful [AM] reproductions, `forget` is **off**, and its effect is
measured separately as a governed-memory experiment. Never silently on.

**(c) C6 is a multi-tenant leakage path.** [GEM] §5 raises it directly: tenant
A's query reinforces topic τ, and tenant B later surfaces τ *because its
salience is high*, even when content is access-controlled. Our fused path
already pushes `scope_id` into the operator (verified injection-safe), but
salience is a **derived** signal that the scope predicate does not cover. Any
shared-store configuration must either partition salience by scope or report
the leakage rate. Filed as an explicit gate, not solved here.

## 7. Configuration matrix — [AM]'s taxonomy as settings

| [AM] system / paradigm | `ingest` | `mode` | `revise` | `forget` | `reinforce` |
|---|---|---|---|---|---|
| I `long_context` | — (absent) | passthrough | off | off | off |
| II BM25 | deterministic (lexical) | RELATIONAL | off | off | off |
| **II embedRAG** ← today | **deterministic** | **VECTOR** | off | off | off |
| III.a GraphRAG-like | llm_mediated(`batch`) | FUSED | off | off | off |
| III.a HippoRAG-like | llm_mediated(`batch`) | FUSED + PPR | off | off | off |
| III.b Mem0-like | llm_mediated(`sequential`) | VECTOR (field-grain) | conflict only | off | off |
| IV agentic | agentic | FUSED | on | on | off |
| **GEM-conformant** | any | FUSED, 3 routes | **on** | **graded ladder** | **on** |

The last row is the one no system in [AM]'s Table 1 occupies and no paradigm in
[GEM]'s Table 1 covers. It is the contribution position, reachable only after
M1–M4 below.

## 8. Build order, and what each step unblocks

| Milestone | Deliverable | Unblocks |
|---|---|---|
| **M0** ✅ | types, protocols, schema; C1/C4 enforcement proven live | — |
| **M1** | `GovernedMemory` on TriDB: `ingest(deterministic)` + `retrieve(VECTOR, reinforce=False)` + `gem_transition` logging | reproduces today's embedRAG row through the new interface — **regression gate before anything else changes** |
| **M2** | NVML sampler → `gem_transition.gpu_joules` | **[AM] §4.2 completes** (Total kJ, J/correct — the 2 missing columns), and construction energy |
| **M3** | LC arm (`ingest` absent, passthrough retrieval) | **[AM] §4.1 completes** |
| **M4** | `LLMMediatedIngest` (both modes) + `validate()` | [AM] §4.3 (traffic shape), §4.4 (capability floor), and the first genuinely tri-modal TriDB row |
| **M5** | `retrieve(reinforce=True)` + `revise` | C3/C6; the **cost-of-C6** measurement (§6a) |
| **M6** | `forget` ladder + C5 accounting | [AM] §4.7 growth with a *bounded* store — which none of [AM]'s nine systems does |
| **M7** | `AgenticIngest` | Paradigm IV row |

M1–M3 are the [AM] reproduction critical path and need **no** GEM semantics.
M4 onward is where the two papers combine.

## 9. Honesty gates

- **Naming.** Until M4, every result is `TriDB-vector` / Paradigm II embedRAG.
  A GEM-conformant label requires C1–C6 all holding, and C2/C3/C5/C6 are not
  implemented yet.
- **Conformance is per-condition.** Report which of C1–C6 a configuration
  satisfies. Today: C1 and C4 hold *for the GEM schema* (§4.2); C2, C3, C5, C6
  do not hold anywhere.
- **`reinforce` and `forget` are recorded in every run manifest.** A run with
  either on is not comparable to a paper row that had neither.
- **Graph is native.** Extracted relations become graph-AM edges, never a
  relational join table (CLAUDE.md rule 3). `gem_edge` holds metadata *about*
  edges; topology lives in the AM.
- **TR-1.** Retrieval honours Open/Next/Close and early termination. C6's
  salience write happens after the operator closes, inside the same
  transaction — it must not turn retrieval into a blocking operator.
- **Censoring travels.** `graph_censored` / `termination_reason` /
  `budget_capped` recorded per retrieval, as today.
- **Hardware.** [AM] is one H100 80 GB; we are GX10/Spark. Absolute numbers are
  never comparable — `paper_hardware_match: false` propagates.

## 10. References

Orogat, A., & Mansour, E. (2026). *Is Agent Memory a Database? Rethinking Data
Foundations for Long-Term AI Agent Memory.* arXiv:2605.26252v1.

Omri, Y., Gan, Z., Broveak, Z., Geens, R., He, Z., Pentland, A., Verhelst, M.,
Weissman, T., & Tambe, T. (2026). *Agent Memory: Characterization and System
Implications of Stateful Long-Horizon Workloads.* arXiv:2606.06448v1.

## AI disclosure

Produced with AI-assisted paper reading (both PDFs read first-hand via full
text extraction), repository inspection, interface authoring, and drafting.
The §4.2 C1/C4 enforcement result was executed against a live PG 16.14 +
pgvector 0.8.1 + graph_store_am 0.2.0 + tjs_pg 0.2.0 stack built from source
in-container; the §4.1 fused-retrieval result is from PR #1. Everything in §5–§8
is specification, not measurement.
