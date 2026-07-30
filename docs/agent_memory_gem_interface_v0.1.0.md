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
That is a claim worth *testing*, not asserting — §10 sets the gates.

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

## 6. What the TriDB engine layer must change

The Python layer is not the whole story. Every claim below was executed against
the live stack, not inferred from source.

### 6.1 Retrieval — tri-modal today, with two constraints

**Yes, `tjs_open` already fuses all three modalities** (§4.1). Two constraints
shape how GEM must use it:

**(a) The fused vector-first path requires `hnsw.iterative_scan =
relaxed_order`.** The engine refuses `strict_order` outright. So fused and
vector-only retrieval are *different operating points* and must never be pooled
in one table.

**(b) `edge_type` is a single id, not a set.** The parameter is one `int32`,
with `0` = ANY. Measured on a fresh database:

```
extension-only  -> 1        association-only -> 2        ANY (0) -> 1,2
... after registering a third type and adding one edge of it:
                                                        ANY (0) -> 1,2,3
```

**This is fine for GEM, but only under a specific design choice**: register
exactly **two** native edge types, `extension` and `association`, and keep the
relation name (`moved_to`, `part_of`, …) in `gem_edge` relationally. Then
extension-only traversal (C3), association-only expansion, and both-kinds
(ANY) are all expressible with **zero engine change**. Registering per-`rel`
native types would break this — "all extension edges regardless of rel" is not
expressible against an equality filter. Recorded here because the choice is
easy to get wrong and expensive to migrate.

### 6.2 Writes — where the actual work is

| GEM write | Engine support today | Verdict |
|---|---|---|
| unit row + vector + graph vertex, one txn | `gph_upsert_vertex` in the caller's txn | ✅ proven (PR #1) |
| typed edge insert | `gph_insert_edge`, batch `gph_insert_edges` | ✅ |
| supersede a field value | relational; the partial unique index enforces C1 | ✅ proven (§4.2) |
| **salience write (C6)** | heap UPDATE of non-indexed columns | ✅ **but needs `fillfactor`** — see 6.3 |
| **re-embed on revision** | UPDATE of the hnsw-indexed column | ⚠️ **index churn** — see 6.3 |
| edge rewire | no edge UPDATE; tombstone + insert | ⚠️ two steps, acceptable |
| archive / reclaim | `gph_tombstone_vertex/_edge`, `gph_freeze` | ⚠️ workable |
| **propagate to dependents (C3)** | `direction=in`/`both` **RAISEs** (ADR-0016 deferred) | ❌ **design around it** — see 6.4 |
| **snapshot-consistent cross-modal read** | commit-visible, not snapshot-isolated | ❌ **real engine gap** — see 6.5 |

### 6.3 The two write costs, measured

C6 makes **every retrieval a write**, so the cost of that write decides whether
governed memory is affordable. Measured on 2,000 units, 200 updates each:

```
C6 salience write (embedding UNCHANGED):
  fillfactor=100 :  200 updates,  168 HOT (84.0%),  size +8192 B
  fillfactor= 70 :  200 updates,  200 HOT (100.0%), size    +0 B

revision re-embed (embedding CHANGED — the hnsw-indexed column):
  fillfactor= 70 :  200 updates,    0 HOT (0.0%),   size +73728 B
```

Two conclusions, both actionable:

1. **`gem_unit` and `gem_field_value` must be created `WITH (fillfactor=70)`.**
   At the default 100 there is no free space for heap-only tuples, so 16% of
   C6's writes become non-HOT and touch every index on the row. At 70, C6 is
   **100% HOT and costs zero index churn** — the salience write is genuinely
   free of the vector index. This is a one-line schema change, not an engine
   change, and it is the difference between C6 being cheap and C6 being a tax
   on every query.
2. **Re-embedding is the expensive write, not salience.** Refreshing a unit's
   vector is 0% HOT and grew the relation by ~369 B per update — every refresh
   is an HNSW insert plus a dead tuple. `revise` must therefore **batch
   embedding refreshes** rather than re-embed per field change, and the
   resulting recall drift is measurable with the repo's existing
   `bench/recall_decay.py`.

### 6.4 C3 propagation needs a direction the engine does not have

The adjacency list is **out-edges only**. `gph_traverse_typed(src, type, 1, …)`
(direction=in) raises:

```
ERROR: graph_store: only GRAPH_SCAN_OUTGOING is supported (got 1)
DETAIL: direction=in/both (getBacklinks) needs a reverse adjacency index —
        a follow-on, format-touching plan (docs/decisions/0016).
```

ADR-0016 costs reverse adjacency at roughly **2× edge storage plus a second
GenericXLog page per insert**, and calls it format-touching — it needs its own
ADR and a concurrency probe. That is far too large to pull into this work.

**The design answer is to orient extension edges in the propagation
direction**: write `A --extension--> B` to mean *"a change in A entails
re-evaluating B"*, so `revise` only ever walks out-edges. This is expressible
today and costs nothing. Its limitation must be stated honestly: a
bidirectional entailment needs **two** edges, and any query of the form "what
does B depend on?" is not answerable without reverse adjacency. If a workload
needs that, ADR-0016 becomes a prerequisite rather than an optimisation.

### 6.5 The one real engine gap: the graph leg is not snapshot-isolated

`gph_xmin_visible()` is commit visibility, not snapshot visibility:

```c
if (TransactionIdIsCurrentTransactionId(xmin)) return true;
return TransactionIdDidCommit(xmin);      /* NOT XidInMVCCSnapshot */
```

Reproduced live with two sessions and a `REPEATABLE READ` snapshot:

```
A opens a REPEATABLE READ snapshot
  graph neighbors: [7]   heap rows: 2

[C writes an edge, UNCOMMITTED]      A graph neighbors: [7]      <- no dirty read
[C rolls back]                       A graph neighbors: [7]      <- clean

[B COMMITS an edge, after A's snapshot was taken]
  A graph neighbors: [7, 8]
  A heap rows      : [6, 7]
  HEAP  respects the snapshot: YES
  GRAPH respects the snapshot: NO
  => TORN: graph returns dst 8 whose row this snapshot CANNOT see
```

What this means for GEM, precisely:

- **C2 (transition soundness) is SAFE.** Uncommitted state is invisible and a
  rollback leaves nothing behind, so "evaluate the proposed `M_{t+1}` and commit
  atomically or abort" ([GEM] Algorithm 1 line 12) holds today. This is the
  condition that matters most, and TriDB already satisfies it.
- **Read consistency across a multi-statement operator does NOT hold.** Any
  operator that reads the graph more than once, or reads graph and heap
  together, can observe a topology newer than its snapshot. Concretely:
  `revise`'s propagation walk can see edges appear mid-walk, and `retrieve`
  with C6 can reinforce a unit whose row its own snapshot cannot return.

This is DEV-1166, already known and already quantified by
`bench/wiki_consistency.py` (the 1.0% residual tear, heap legs 0.0%). It is
**the** engine change GEM wants: per-tuple `xmin`/`xmax` checked against the
caller's snapshot via `XidInMVCCSnapshot` rather than `TransactionIdDidCommit`.
The layout design (`docs/graph_store_layout_v0.1.0.md` §"MVCC visibility")
already specifies `GraphTupleSatisfiesSnapshot`; it is the implementation that
is deferred.

**Until it lands**, GEM operators must be written defensively: take the
topology snapshot once per operator (materialise the reach set at the start of
a propagation walk rather than re-traversing), and record in every run manifest
that graph reads are commit-visible. A GEM conformance claim that depends on
repeatable-read topology cannot be made on this engine today.

### 6.6 Summary — what is Python work and what is engine work

| Work | Layer | Blocking? |
|---|---|---|
| four operators, ingest strategies, salience policy, policy evaluation | **Python** | no |
| two-edge-kind registration, `fillfactor=70`, batched re-embedding, out-oriented extension edges | **Python/schema** | no — design choices, zero engine change |
| per-tuple snapshot visibility in the graph AM (DEV-1166) | **engine (C)** | not blocking C2; blocks a repeatable-read conformance claim |
| reverse adjacency (ADR-0016) | **engine (C), format-touching** | not blocking, if extension edges are oriented outward |
| edge attributes in the AM (weights on native edges) | **engine (C)** | not blocking — `gem_edge` carries them relationally |

**The honest headline: implementing ingest, revision and retrieval needs no C
changes.** One engine gap (snapshot isolation) bounds what can be *claimed*
about concurrent correctness, not what can be built.

## 7. Three design tensions, flagged not buried

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

## 8. Configuration matrix — [AM]'s taxonomy as settings

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

## 9. Build order, and what each step unblocks

| Milestone | Deliverable | Unblocks |
|---|---|---|
| **M0** ✅ | types, protocols, schema; C1/C4 enforcement proven live | — |
| **M1** | `GovernedMemory` on TriDB: `ingest(deterministic)` + `retrieve(VECTOR, reinforce=False)` + `gem_transition` logging | reproduces today's embedRAG row through the new interface — **regression gate before anything else changes** |
| **M2** | NVML sampler → `gem_transition.gpu_joules` | **[AM] §4.2 completes** (Total kJ, J/correct — the 2 missing columns), and construction energy |
| **M3** | LC arm (`ingest` absent, passthrough retrieval) | **[AM] §4.1 completes** |
| **M4** | `LLMMediatedIngest` (both modes) + `validate()` | [AM] §4.3 (traffic shape), §4.4 (capability floor), and the first genuinely tri-modal TriDB row |
| **M5** | `retrieve(reinforce=True)` + `revise` | C3/C6; the **cost-of-C6** measurement (§7a) |
| **M6** | `forget` ladder + C5 accounting | [AM] §4.7 growth with a *bounded* store — which none of [AM]'s nine systems does |
| **M7** | `AgenticIngest` | Paradigm IV row |

M1–M3 are the [AM] reproduction critical path and need **no** GEM semantics.
M4 onward is where the two papers combine.

## 10. Honesty gates

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

## 11. References

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
