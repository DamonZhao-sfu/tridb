# GEM Operators — Implementation Status

> **Version:** 0.1.0
> **Date:** 2026-07-30
> **Status:** Code landed for G1–G8. **Unverified against a live engine.**
> **Plan:** [`agent_memory_gem_implementation_plan_v0.1.0.md`](agent_memory_gem_implementation_plan_v0.1.0.md)
> **Interfaces:** [`agent_memory_gem_interface_v0.1.0.md`](agent_memory_gem_interface_v0.1.0.md)

This is an addendum to the build plan, not a rewrite of it. The plan's §0 said
"nothing here is implemented yet beyond the type layer"; that sentence is what
this document supersedes.

## 1. What landed

| Plan module | File | Status |
|---|---|---|
| `store.py` | `bench/agent_memory/gem/store.py` | transition envelope, audit connection, writer lock, vid allocator, `TriDBMemoryView`, edge-kind bootstrap |
| `salience.py` | `.../salience.py` | `ExponentialSalience`, ladder, lazy decay |
| `policy.py` | `.../policy.py` | closed condition/action registry, commit postcondition, seed policies |
| `ingest.py` | `.../ingest.py` | gate matrix, plan apply, supersession, batched embed, C3 flag hook |
| `retrieve.py` | `.../retrieve.py` | 4 modes × 3 routes, pushed predicate, the C6 write |
| `revise.py` | `.../revise.py` | conflict / duplicate / propagate / split, one-shot walk, batched re-embed |
| `forget.py` | `.../forget.py` | graded ladder, edge tombstoning, C5 accounting |
| `memory.py` | `.../memory.py` | `TriDBGovernedMemory`, the run manifest |
| `strategies/` | `.../strategies/{deterministic,llm_mediated,agentic}.py` | Paradigms II, III.a/III.b, IV |
| `conformance.py` | `.../conformance.py` | C1–C6 report, per-condition |
| — (new) | `.../plan.py` | the write-plan vocabulary shared by strategies and operators |
| — (new) | `bench/agent_memory/chunking.py` | `TiktokenSentenceChunker`, moved verbatim out of `longmemeval_pipeline.py` |

Tests: `tests/test_gem_unit.py`, `tests/test_gem_strategies.py`,
`tests/test_gem_roundtrip.py`, plus `tests/test_gem_conformance.py` and
`tests/test_gem_live.py` (live halves skip-gated). Suite: **764 passed,
26 skipped** (was 625 passed, 2 skipped).

`test_gem_roundtrip.py` is worth calling out: it runs the real operators against
a fake that keeps rows and **enforces the C1 partial unique index**, so the
paper's own Figure 1 scenario is exercised end to end — append March 15, watch
an unsuperseded April 20 abort, supersede it properly, and confirm one current
value with the prior one chained. Statement-level tests cannot catch a wrong
*sequence* of individually-correct statements; this one can.

## 2. What is verified, and what is not

**Verified here** (x86_64 workstation, no engine): the Python layer. Strategy
planning, the validate gate matrix, salience monotonicity as a swept property,
the policy registry, the transition envelope's commit/abort/audit paths, the
edge-orientation invariant, and the SQL each operator emits.

**NOT verified here.** Every claim that depends on the engine. The partial
unique index actually aborting a second current value, HOT update behaviour at
`fillfactor=70`, `tjs_open` fusion, `gph_traverse_bfs` type filtering, and
graph visibility are engine properties. `tests/test_gem_live.py` and the live
half of `tests/test_gem_conformance.py` encode them and are **skip-gated on
`TRIDB_GEM_DSN`** — they have never been executed. Treat the SQL as unverified
against the engine until a green run exists.

Nothing here is GX10-gated: per interface doc §6.6, implementing these four
operators needs **no C change**. The stack these tests want is stock
PG 16/17 + pgvector + graph_store_am + tjs_pg.

## 3. Decisions taken (the plan's §11, resolved)

1. **GEM lives BESIDE `agent_memory_units`** through G2, so the regression gate
   can diff old vs new on the same database. `backend.py` keeps its own
   `_allocate_vertex`; `store.py` has GEM's. They are intentionally separate
   until the old table is retired.
2. **Embedding model pinned to `BAAI/bge-small-en-v1.5`, dim 384** — unchanged
   from today. Changing it would guarantee the G2 numbers move, which is
   exactly what G2 exists to detect.
3. **`reinforce` defaults OFF in the benchmark pipelines.** `Query.reinforce`
   still defaults True (that is the GEM-conformant default); the pipelines pass
   False so a paradigm-faithful [AM] reproduction stays faithful, with C6 as an
   explicit opt-in ablation.
4. **Construction LLM co-located on the answer endpoint** (`VLLM_BASE_URL`), so
   [AM] §4.3's construction/generation interference stays measurable. Point
   `OpenAIChatExtractor.base_url` elsewhere to run the separation as an
   experiment.
5. **Plan-inside transaction shape**, per plan §3 — LLM latency is visible in
   the transaction duration, which is what [AM] §4.6's freshness argument
   needs. Revisit only if measured.

## 4. Open gates

- **G2 has not been run.** The regression gate — "reproduces the current
  embedRAG retrieval results through the new interface" — needs the engine
  stack and the corpus. Until it passes, no measurement taken downstream of it
  is interpretable, and that is the plan's own emphasis.
- **No conformance claim is made.** `conformance.run()` produces the report;
  no report has been produced against real data. The label stays
  `TriDB-vector` (Paradigm II embedRAG).
- **M2 (NVML sampler) remains out of scope**, so `gem_transition.gpu_joules`
  is still NULL and [AM] §4.2's two energy columns are still missing.
- **Pipeline integration is not done.** `longmemeval_pipeline.py` and
  `locomo_pipeline.py` still write through `TriDBMemoryBackend`; routing them
  through `TriDBGovernedMemory` is the G2 work item.

## 5. Defects found and fixed during review

An adversarial pass over the diff, plus a sweep that replays every operator's
emitted SQL and type-checks each parameter against its placeholder's context,
found eight real defects. All are fixed, and each carries a regression test
**verified to fail without the fix**.

| # | Defect | Consequence |
|---|---|---|
| 1 | `retrieve._temporal` bound its ranking vector to the wrong placeholder — the vector sits in the SELECT list and is the FIRST parameter, but was appended last | every argument shifted one position; the temporal route could not run. A placeholder *count* check cannot see this |
| 2 | All four operators captured their result **inside** the `with` block, but `transition()` samples C5, evaluates `P_t`, and logs during `__exit__` | `transition_id` always `None`, `policies_evaluated` always empty, `active_units`/`active_fields` always `None` on every result — while the `gem_transition` row itself was correct, so the log and the API disagreed |
| 3 | `policy._dependents_flagged` read `delta.propagated_units` (the units it just *flagged*) instead of the units that *changed* | demanded the flagged units' own dependents also be flagged — one hop further than anyone flags. The default `propagate-on-change` policy aborted **every** ingest on any `A→B→C` extension chain |
| 4 | `gem_field_value.transition_id` was never populated | C4's provenance column, documented as "the transition that committed it", was always NULL. Fixed by reserving the id at the top of the envelope via `nextval` |
| 5 | `SPLIT_TOPIC` ops had no apply branch | the agentic `split_topic` tool returned `{"ok": true}` to the model and did nothing — a no-op counted as a write |
| 6 | `validate_plan` built `defined_refs` from *all* upserts, including rejected ones | a dependent of a rejected upsert passed the dangling-vertex gate, then failed to resolve at apply and rolled the whole batch back — breaking the per-op rejection contract the function documents |
| 7 | `revise._split_topics` re-detected its own output | the split-out unit inherits the field history that triggered it, so every `revise()` forked another unit, unboundedly |
| 8 | duplicate-merge re-pointed `gem_edge` rows onto the winner without guarding the `(src, dst, edge_type)` primary key | a near-duplicate pair sharing a target aborted the transaction and lost every repair in the batch |

Two of these (#2, #3) would have been invisible until a live run and then
misleading rather than loud — #2 makes the API disagree with its own log, #3
turns the default policy into a blanket ingest failure. Worth noting for the
G2 gate: neither would have shown up as a wrong *number*.

## 6. Deviations from the plan worth flagging

- **`plan.py` is new.** The plan listed `UpsertUnit` / `AppendFieldValue` /
  `LinkUnits` / `SplitTopic` as Protocols in `protocols.py`, but the
  `IngestStrategy` Protocol returns `Sequence[Mapping[str, Any]]`. A small
  constructor module keeps the key names to one spelling without inventing a
  class hierarchy the interface does not use.
- **Plan-local `ref` handles.** Not in the plan, but a plan routinely creates a
  unit and attaches fields and edges to it before any vid exists. Refs make the
  "no dangling vertex" gate checkable statically rather than at apply time.
- **`IngestResult.capped`** added to `types.py` (additive). The plan requires
  the operator to record cap exhaustion; there was nowhere to put it.
- **Split detection is crude on purpose** (a field history reaching N entries).
  [GEM] Figure 3 specifies the split *mechanism*, not a trigger heuristic;
  inventing one and presenting it as theirs would be a misattribution.

## AI disclosure

Produced with AI-assisted implementation against the two design documents named
above. No measurements are claimed. The engine constraints the code is built on
were read from `agent_memory_gem_interface_v0.1.0.md` §6, which records them as
measured in an earlier session; they were **not** re-measured here.
