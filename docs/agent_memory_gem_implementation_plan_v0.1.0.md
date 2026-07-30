# GEM Operators — Code Implementation Plan

> **Version:** 0.1.0
> **Date:** 2026-07-30
> **Status:** Build plan. Specifies the code for the four state-level operators,
> the three ingest strategies, the salience policy, and policy evaluation.
> **Superseded on implementation status** by
> [`agent_memory_gem_implementation_status_v0.1.0.md`](agent_memory_gem_implementation_status_v0.1.0.md):
> G1–G8 code has since landed, unverified against a live engine. The rest of
> this document — the design, the gates, the risks — still stands as written.
> **Design source:** [`agent_memory_gem_interface_v0.1.0.md`](agent_memory_gem_interface_v0.1.0.md)
> (interfaces, configuration matrix, measured engine constraints §6).
> **Papers:** [GEM] arXiv:2605.26252v1 · [AM] arXiv:2606.06448v1.

## 0. Scope

**In scope:** `ingest` / `revise` / `forget` / `retrieve`; `DeterministicIngest`
/ `LLMMediatedIngest` / `AgenticIngest`; `SaliencePolicy`; the policy engine;
the C1–C6 conformance suite.

**Out of scope:** the NVML energy sampler (separate milestone M2, unblocks
[AM] §4.2), the long-context arm (M3, unblocks [AM] §4.1), any engine (C)
change. Per interface doc §6.6, **none of this needs a C change.**

**Already landed** (PR #1, commits `2741df3`, `55f55d1`):
`bench/agent_memory/gem/{types,protocols}.py` (dataclasses + Protocols),
`schema.sql` (applies clean; `fillfactor=70` verified), and on
`TriDBMemoryBackend`: `add_units` / `link` / `neighbors` / `search_fused` /
`graph_stats` with the `id == vid` allocator.

## 1. Module layout

```
bench/agent_memory/gem/
  types.py          ✅ landed   state vocabulary
  protocols.py      ✅ landed   operator + strategy Protocols
  schema.sql        ✅ landed   D_t / S_t / P_t / trajectory
  store.py          ~250  connection, transition envelope, vid allocator, view
  salience.py       ~80   ExponentialSalience: reinforce / decay / thresholds
  policy.py         ~200  rule registry, condition eval, commit postconditions
  ingest.py         ~200  the ingest operator (strategy-agnostic write applier)
  retrieve.py       ~280  4 modes x 3 routes + the C6 write
  revise.py         ~320  4 repair kinds, propagation walk, batched re-embed
  forget.py         ~150  the graded ladder
  memory.py         ~150  TriDBGovernedMemory — wires the four operators
  strategies/
    deterministic.py ~120
    llm_mediated.py  ~350  prompt, schema, validation, batch|sequential
    agentic.py       ~250  tool loop with hard caps
  conformance.py    ~250  C1..C6 checks, emits a conformance report
tests/
  test_gem_unit.py       ~400  fake-connection tests (no PG) — CI
  test_gem_live.py       ~500  integration, skipped without TRIDB_GEM_DSN
  test_gem_conformance.py ~300 one test per correctness condition
```

≈2,400 LOC of implementation, ≈1,200 of tests.

## 2. Foundation — build this first (`store.py`)

Everything else depends on three primitives. Get them wrong and every operator
inherits the bug.

### 2.1 The transition envelope

Every operator is one transaction that either commits the proposed `M_{t+1}` or
aborts ([GEM] Algorithm 1 line 12).

```python
@contextmanager
def transition(self, operator: str, scope_id: str, phase: str) -> Iterator[Tx]:
    """One GEM state transition. Yields a Tx that accumulates the delta and
    the phase cost; evaluates P_t as a postcondition; commits or aborts."""
```

`Tx` exposes `execute()`, `delta` (a mutable `StateDelta`), `meter` (a
`PhaseCost` accumulator), and `transition_id`.

**The abort-logging problem.** A `gem_transition` row written inside the
transaction is rolled back with it, so failed transitions would vanish — but
[GEM] correctness is a property of the *trajectory*, and an abort is part of it
(C2 is precisely "a violating transition is rejected"). PostgreSQL has no
autonomous transactions.

> **Decision:** `store.py` holds a second, dedicated **audit connection** in
> autocommit. Committed transitions are logged on the main connection inside the
> transaction (so the log commits atomically with the state it describes);
> aborted ones are logged on the audit connection after rollback. Both paths set
> `committed` explicitly. This is 20 lines and removes a whole class of "we
> can't explain why the state looks like this" bugs.

### 2.2 The vid allocator and the writer lock

`_allocate_vertex()` moves from `backend.py` into `store.py`. Two hazards the
engine documents in `gph_upsert_vertex`:

- a lost allocation race returns the winner's vid, leaving an orphan vertex;
- **under REPEATABLE READ the re-SELECT can miss a concurrent winner and return
  NULL**, which today would raise `TypeError` inside `int(vid)`.

Both are excluded by the engine's **v1 single-writer contract**. GEM must
honour it explicitly rather than hope:

```python
# serialize writers on the allocator; readers are unaffected
tx.execute("SELECT pg_advisory_xact_lock(hashtext('gem_vid_alloc'))")
```

Taken at the top of every *writing* operator (`ingest`, `revise`, `forget`),
released automatically at commit. Add a `vid is None` guard raising a clear
"single-writer contract violated" error rather than a `TypeError`.

### 2.3 `TriDBMemoryView`

Read-only implementation of the `MemoryView` Protocol, so strategies can see
`M_t` without writing. Backed by the same connection, so a strategy running
inside an operator's transaction **sees that transaction's own uncommitted
writes** — which is exactly what `AgenticIngest` needs (§4.3).

### 2.4 Edge-kind bootstrap

Register exactly two native types once, per interface doc §6.1:

```python
EXTENSION_TYPE  = register_edge_type("extension")     # C3 propagates along these
ASSOCIATION_TYPE = register_edge_type("association")  # retrieval expansion only
```

`link()` writes the native edge with the kind's type id and the `rel` name into
`gem_edge`. **Enforce the orientation invariant in code**: `A --extension--> B`
means "a change in A entails re-evaluating B", because revision can only walk
out-edges (§6.4 of the interface doc). Docstring + test.

## 3. Operator: `ingest`

```python
def ingest(self, events, *, strategy) -> IngestResult
```

**Transaction shape** — one transaction for the whole call:

```
advisory lock
  → strategy.plan(events, view)          # may call LLM/embeddings; NO writes
  → validate plan (referential + schema gates)
  → apply writes:
      for each new unit:  allocate vid → INSERT gem_unit → gph_upsert_vertex
      for each fact:      supersede current (UPDATE valid_to) → INSERT new value
      for each edge:      register/resolve type → gph_insert_edge → INSERT gem_edge
      refresh embeddings for touched units   (ONE batched embed call)
  → flag extension-linked units for revision   (C3 hook, [GEM] Alg. 1 line 4)
  → evaluate P_t postconditions
  → log gem_transition → COMMIT
```

**Model calls happen in `plan()`, outside the write path but inside the
transaction.** That is deliberate: it keeps atomicity, and it makes the LLM
latency visible in the transaction duration, which is what [AM] §4.6's
freshness argument needs to measure. If long transactions become a problem
under concurrent load, the alternative — plan outside, apply inside — is a
one-line move, but it introduces a TOCTOU window against `M_t` that must then
be revalidated. **Start with plan-inside; revisit only if measured.**

**Supersession SQL** (the C1/C4 core):

```sql
UPDATE gem_field_value SET valid_to = $ts
 WHERE unit_id = $u AND field = $f AND valid_to IS NULL;
INSERT INTO gem_field_value (unit_id, field, value, valid_from, ...provenance...)
 VALUES (...) RETURNING id;
UPDATE gem_field_value SET superseded_by = $new WHERE id = $old;
```

The partial unique index makes a plan that would leave two current values abort
at commit — no application-level check needed (verified live, interface §4.2).

## 4. Ingest strategies

### 4.1 `DeterministicIngest` (~120 LOC)

Chunk → embed → one unit per chunk, single `content` field, **no edges, no
LLM**. Reuses the existing `TiktokenSentenceChunker`, which must first be
**factored out of `longmemeval_pipeline.py` into `bench/agent_memory/chunking.py`**
(a pure move, no behaviour change) so both the old pipeline and GEM share one
chunker and the comparison stays honest.

`gem_unit.embedding` here is the **chunk text** vector, not title+summary. That
differs from the LLM-mediated strategies, so the strategy records
`embedding_source` in `gem_unit.metadata`. Never compare retrieval quality
across two different `embedding_source` values without saying so.

**This strategy is the regression gate**: with `reinforce=False`, `revise` off
and `forget` off, a GEM run must reproduce the current embedRAG numbers.

### 4.2 `LLMMediatedIngest` (~350 LOC)

```
for each chunk:
  candidates = view.find_similar(scope, chunk_vec, k=8)      # host-topic slate
  extraction = llm(prompt_v1, candidates_titles_summaries, chunk)  # JSON
  ok, reason = validate(extraction)
  if not ok: record rejection, skip                          # [AM] §4.4 gate
  else: emit UpsertUnit / AppendFieldValue / LinkUnits ops
```

**Output contract** — pinned prompt, versioned schema, both recorded in every
`Provenance`:

```json
{"host": {"unit_id": 12} | {"new_title": "...", "summary": "..."},
 "facts": [{"field": "deadline", "value": "April 20",
            "valid_from": "2026-03-08", "confidence": 0.9}],
 "edges": [{"dst_title": "Milestones", "kind": "extension", "rel": "schedules"}]}
```

**`validate()` gates, in order** — any failure rejects the *unit*, not the run:

1. JSON parses;
2. conforms to the pinned schema (required keys, types, enum for `kind`);
3. referential validity — `host.unit_id` exists in scope; every `edges[].dst_title`
   resolves or is created in the same plan;
4. no dangling vertex — every referenced unit will have a vid after apply;
5. duplicate/conflict policy conformance — a fact that would create a second
   current value for the same `(unit, field)` must carry supersession intent.

`IngestResult.rejected` carries every failure with its reason. **A run whose
rejection rate exceeds a configured threshold is reported as a FAILED
CONFIGURATION, not as low accuracy** — that is the whole point of [AM] §4.4's
capability-floor finding.

**Two modes** (interface doc §5.1), differing only in call structure:

| | `batch` (III.a) | `sequential` (III.b) |
|---|---|---|
| embedding | one call per N units at the end | one call per extracted fact |
| conflict | append-only | similarity search → ADD/UPDATE/DELETE |
| target signature | large sequences-per-call | 1:1 call-to-sequence |

The mode is recorded in the manifest; `calls.items_by_kind /
calls.by_kind` already yields the sequences-per-call ratio [AM] §4.3 plots.

### 4.3 `AgenticIngest` (~250 LOC)

Tool loop: `search_memory`, `read_unit`, `write_field`, `link`, `split_topic`.

**Writes are applied inside the operator's transaction as the agent requests
them**, so the agent's later reads see its own earlier writes — PostgreSQL gives
this for free, and it is what makes an agentic loop coherent without per-round
commits (which would break atomicity).

`max_rounds` and `max_tool_calls` are **required constructor arguments** with no
defaults. On cap exhaustion the operator records `capped=True` and commits what
exists. [AM] Recommendation 10 exists because these tails reach p95/p50 = 5.9×.

## 5. Operator: `retrieve` (~280 LOC)

```
admit(q) → resolve query vector
  → route:
      TOPIC      tjs_open(gem_unit, k, …, filter, qvec[, src])   # 4 modes
      TEMPORAL   relational history predicate + optional vector
      STRUCTURAL anchored tjs_open (src=anchor) → filter_first path
  → fetch current field values for the returned unit ids   (C1)
  → build hits + prompt block
  → if q.reinforce:  the C6 write  (SAME transaction)
  → log gem_transition → COMMIT
```

**The relational predicate always includes the active-state filter**, so
attenuated content costs nothing instead of being fetched and discarded:

```python
predicate = sql.SQL("scope_id = {} AND state = 'active'").format(
    sql.Literal(scope_id))
```

`sql.Literal` quoting is already verified injection-safe. Set
`hnsw.iterative_scan = relaxed_order` for fused calls and **record it** — fused
and vector-only are different operating points (interface §6.1a).

**The C6 write:**

```sql
UPDATE gem_unit SET salience = $s_new, access_count = access_count + 1,
                    last_access = now()  WHERE id = ANY($ids);
UPDATE gem_field_value SET salience = ... WHERE unit_id = ANY($ids) AND field = ANY($f);
UPDATE gem_edge SET co_access_count = co_access_count + 1
 WHERE src = ANY($ids) AND dst = ANY($ids) AND kind = 'association';
```

None of these columns is indexed, and both tables are `fillfactor=70`, so these
are **100% HOT updates costing zero index churn** (measured, interface §6.3).

"Updates memory structure" has a second half: co-retrieved units with no edge
get an `association` edge created, and an association edge whose
`co_access_count` crosses a policy threshold is **flagged as an extension
candidate** — flagged only. Promotion to `extension` is a `revise` decision,
never a retrieval side effect, because an extension edge grants propagation
rights and retrieval must not silently widen the C3 frontier.

**TR-1**: materialise `tjs_open`'s ids first, then write. The operator closes
before any UPDATE runs; retrieval must not become blocking.

## 6. Operator: `revise` (~320 LOC)

Four repair kinds, run in this order (each is idempotent):

| Repair | Detection | Action |
|---|---|---|
| conflict | two candidate current values for one `(unit, field)` | supersede the older, chain `superseded_by` (C4). The unique index makes a wrong result abort |
| duplicate | two units, same scope, cosine ≥ θ_dup and title match | merge: union field histories, re-point edges, archive the loser |
| propagate | a field changed on `u_i` | walk **extension out-edges** from `u_i`, halt where the policy condition does not fire (C3) |
| split | a field subset accumulates ≥ N references | promote to a new unit, re-point edges ([GEM] Fig. 3, "Alice") |

**Propagation walk — materialise once.** The graph leg is commit-visible, not
snapshot-isolated (proven, interface §6.5), so a walk that re-traverses can see
edges appear mid-walk:

```python
# ONE traversal, then process the frozen set — never re-traverse
reach = tx.execute(
    "SELECT graph_store.gph_traverse_bfs(%s, %s, %s)",
    (unit_id, max_hops, EXTENSION_TYPE)).fetchall()
```

**Batch the re-embedding.** Re-embedding is 0% HOT and grew the relation ~369 B
per update (measured). So `revise` collects every unit whose fields changed into
a set and issues **one** embedding call plus one `UPDATE ... FROM (VALUES …)` at
the end of the operator. Never re-embed per field change.

## 7. Operator: `forget` + the salience policy

### 7.1 `salience.py` (~80 LOC)

```python
class ExponentialSalience:
    def reinforce(self, s, *, rank, k):
        return s + self.gain * (1.0 - rank / k) + self.floor   # floor > 0
    def decay(self, s, *, seconds_idle):
        return s * math.exp(-self.lam * seconds_idle)
```

**C6 requires `reinforce` to be strictly increasing** — "repeated retrieval
strictly reduces eligibility for attenuation". The `floor > 0` term guarantees
it even for the last-ranked hit. This is a property test, not a comment:
`assert reinforce(s, rank=k-1, k=k) > s` for all `s` in a sampled range.

**Decay is lazy.** Applying decay on read would touch every row on every query.
Instead `forget` computes it from `last_access` at tick time. Consequence to
state in the manifest: salience is only current as of the last `forget` tick.

### 7.2 `forget.py` (~150 LOC)

The graded ladder, per field and per unit:

```
s < theta_summary  → compress history: keep first + current + N most salient,
                     archive the middle (state='compressed')
s < theta_remove   → state='hidden'    (excluded by the retrieval predicate)
s < theta_archive  → state='archived'; tombstone the unit's edges via
                     gph_tombstone_edge; the row REMAINS (C5 recoverable)
```

Never `DELETE`. C4 requires the provenance chain of anything still reachable to
survive, and C5 requires archived content to remain recoverable.

Records `active_units` / `active_fields` into `gem_transition` after every
transition — that is C5's `|D_t^active|` sampled at each interaction count, and
the [GEM] research agenda's third ground-truth level.

## 8. Policy evaluation (`policy.py`, ~200 LOC)

[GEM] Definition 3 policies are `<event, condition, action>` rules living in
`M_t`. A general policy *language* is the paper's own research direction —
**out of scope**. Instead:

```python
CONDITIONS: dict[str, Callable[[Tx, dict], bool]] = {}
ACTIONS:    dict[str, Callable[[Tx, dict], None]] = {}

@condition("no_duplicate_current")   # C1/C2 postcondition
@condition("dependents_flagged")     # C3
@condition("active_below_bound")     # C5: |D_t^active| <= beta(n)
@condition("salience_monotone")      # C6
@action("flag_for_revision")
@action("promote_edge_candidate")
@action("attenuate")
```

A named, closed registry with JSONB parameters. Adding a policy = adding a
registry entry, which is a code change but a 10-line one; the *rules* stay data.

**Two evaluation points:**

1. **Postcondition, before commit** (this is what lifts C2 to a data-model
   guarantee): for every enabled policy matching the operator's event, evaluate
   `condition` against the *proposed* `M_{t+1}` — which, inside the transaction,
   is simply what this transaction can see. Any failure raises → rollback →
   audit-connection log with `aborted_reason`.
2. **Action triggering, inside the operator**: e.g. `propagate-on-change` flags
   dependent units after a field update.

Ship three seed policies matching [GEM] Listing 1 and the correctness
conditions: `propagate-on-change`, `bound-active-state`, `reinforce-on-read`.

## 9. Test plan

**Layer 1 — `test_gem_unit.py` (no PostgreSQL, runs in CI).** Extends the
existing fake-connection pattern in `tests/test_agent_memory_adapters.py`.
Covers: strategy planning (given a view, assert the emitted plan), salience
monotonicity property test, policy registry dispatch, validate() gate matrix
(one case per gate), edge-orientation invariant, ladder threshold arithmetic.

**Layer 2 — `test_gem_live.py` (needs PostgreSQL + the three extensions).**
`pytest.mark.skipif(not os.environ.get("TRIDB_GEM_DSN"))`, matching how the rest
of the repo gates engine work. Covers: the ingest→retrieve round trip, the C6
write actually landing, propagation across extension edges, the forget ladder,
abort paths leaving no partial state, and **an assertion that the C6 updates are
HOT** (`n_tup_hot_upd` delta == `n_tup_upd` delta) so a future schema change
that drops `fillfactor` fails loudly.

**Layer 3 — `test_gem_conformance.py`.** One test per condition, emitting a
machine-readable report:

| | Check |
|---|---|
| C1 | after supersession the default query returns only the new value; `as_of` returns the old one |
| C2 | a transition violating a policy leaves `M_t` byte-identical and is logged aborted |
| C3 | updating `u_i` flags every extension-reachable `u_j` and **no** association-only neighbour |
| C4 | after revision *and* after forgetting, the provenance chain of a reachable unit is intact |
| C5 | `active_units` stays ≤ β(n) across a scripted trajectory; archived content is still retrievable by explicit lookup |
| C6 | salience after retrieval > before, strictly, for every hit; a unit retrieved k times outranks an equally-similar unit retrieved once for attenuation |

The report is the deliverable that lets us say **which conditions hold** rather
than claiming "GEM-conformant" wholesale.

## 10. Milestones

| # | Contents | LOC | Gate |
|---|---|---|---|
| **G1** | `store.py`, `salience.py`, edge bootstrap, chunker extraction | ~400 | transition envelope commits/aborts; abort is logged; advisory lock serialises writers |
| **G2** | `ingest` + `DeterministicIngest` + `retrieve` (VECTOR, `reinforce=False`) + `memory.py` | ~750 | **regression gate — reproduces the current embedRAG retrieval results through the new interface** |
| **G3** | `retrieve` FUSED + all 3 routes + the C6 write | ~250 | C6 conformance test passes; HOT assertion passes |
| **G4** | `policy.py` + `revise` (conflict, propagate) | ~450 | C1–C4 conformance |
| **G5** | `forget` + ladder + C5 accounting; `revise` (duplicate, split) | ~300 | C5 conformance; nothing is ever DELETEd |
| **G6** | `LLMMediatedIngest`, both modes + gates | ~350 | rejection accounting works; first tri-modal TriDB row |
| **G7** | `AgenticIngest` | ~250 | caps enforced and recorded |
| **G8** | `conformance.py` + report | ~250 | full C1–C6 report published |

**Order rationale.** G1–G2 land no GEM semantics at all — they are pure
refactoring behind a new interface, and G2's job is to prove the numbers did
not move. Only then do C6 (G3) and revision (G4) change behaviour. G6 is where
[AM] §4.3/§4.4 become measurable.

**G2 is the one that must not be skipped.** If the embedRAG numbers move when
nothing semantic changed, everything measured afterwards is uninterpretable.

## 11. Decisions needed before G1

1. **Does GEM replace `agent_memory_units`, or live beside it?** Recommend
   *beside* through G2 (so the regression gate can diff old vs new on the same
   database), then migrate the pipelines and retire the old table. Cost: two
   schemas coexisting for one milestone.
2. **Which embedding model for unit routing vectors?** Must match whatever the
   comparison runs use; changing it invalidates cross-run comparison.
3. **`reinforce` default in the benchmark pipelines** — recommend **off**, so a
   paradigm-faithful [AM] reproduction stays faithful, with C6 as an explicit
   opt-in experiment.
4. **Where does the construction LLM come from for G6?** The same vLLM endpoint
   as generation (co-located, which is what [AM] §4.3's interference finding is
   about), or a separate one. Recommend co-located to keep the interference
   measurable, with the separation as a later experiment.

## 12. Risks

| Risk | Mitigation |
|---|---|
| Long transactions under LLM-mediated ingest (model latency inside the txn) | measure first; the plan-outside/apply-inside variant is a small move but needs `M_t` revalidation |
| Advisory lock serialises all writers → ingest throughput ceiling | it is the engine's documented v1 contract, not our choice; measure and report, and note DIRECTION-04 as the unblock |
| Re-embedding churn degrades HNSW recall over a long trajectory | batch it (§6); `bench/recall_decay.py` already measures the drift — run it as part of G5 |
| Propagation frontier explodes on a dense extension graph | `max_hops` cap + policy halting; topic grain keeps it small by construction; record the frontier size per revision |
| C6 salience becomes a cross-tenant signal | interface doc §7c — partition salience by scope, or report the leakage rate; do not ship a shared store without one of the two |
| Scope creep into a general policy language | closed registry (§8); a language is [GEM]'s own research direction, not ours |

## 13. Reference

Orogat, A., & Mansour, E. (2026). *Is Agent Memory a Database?* arXiv:2605.26252v1.
Omri, Y., et al. (2026). *Agent Memory: Characterization and System Implications
of Stateful Long-Horizon Workloads.* arXiv:2606.06448v1.

## AI disclosure

Produced with AI-assisted repository inspection and drafting. The engine
constraints this plan is built on (`fillfactor` HOT behaviour, the single edge
type filter, commit-visible graph reads, the `gph_upsert_vertex` race and its
NULL return) were measured or read from source during this session and are
documented in `agent_memory_gem_interface_v0.1.0.md` §6. LOC figures are
estimates. No measurements are claimed here.
