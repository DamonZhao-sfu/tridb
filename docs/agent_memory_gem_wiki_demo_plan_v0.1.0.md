# GEM Wikipedia Demo — Plan

> **Version:** 0.1.0
> **Date:** 2026-07-30
> **Status:** **G0–G3 green 2026-07-30** (all four operators end to end,
> deterministic construction). G4 (the three-way construction-form cost
> comparison) is now unblocked but remains deferred.
> **Depends on:** [`agent_memory_gem_interface_v0.1.0.md`](agent_memory_gem_interface_v0.1.0.md)
> (interfaces, configuration matrix, measured engine constraints §6) and the
> landed operator code in `bench/agent_memory/gem/`.
> **Papers:** [GEM] arXiv:2605.26252v1 · [AM] arXiv:2606.06448v1.

## 0. What this demo is for

The four state-level operators exist in code but have never run against a live
engine and have never been shown on data anyone recognises. This demo does both
at once: it drives `ingest` / `retrieve` / `revise` / `forget` over a real
Wikipedia slice, and each of the four is tied to a correctness condition that
can be *falsified* on that data rather than asserted in prose.

The claim the demo is built to support:

> These four state transitions happen inside one Postgres process, one
> transaction manager, one WAL — across the relational row, the vector, and the
> native graph edges together. A Milvus + Neo4j + Postgres stack cannot make
> the revision and forgetting steps atomic at all.

## 1. Environment (measured 2026-07-30, this workstation)

| Component | State |
| --- | --- |
| Cluster | `.tridb-pgdata/` running on `127.0.0.1:55432`, superuser `hza214` — **no sudo required** |
| Extensions | `vector 0.8.0`, `graph_store_am 0.2.0`, `tjs_pg 0.2.0`, already created |
| DSN | `postgresql://hza214@127.0.0.1:55432/gem_demo` (demo gets its own database) |
| Embedder | fastembed `BAAI/bge-small-en-v1.5`, `vector(384)`, cosine — same model as the LongMemEval/LoCoMo adapters, so numbers stay comparable |
| LLM (G4 only) | local vLLM `Qwen/Qwen3-VL-30B-A3B-Instruct` on `127.0.0.1:8000` |
| Network | `en.wikipedia.org` and `www.wikidata.org` APIs reachable (HTTP 200) |

This is the **stock-PG path** of `CLAUDE.md` — hardware-independent, not the
GX10 fork. No claim in the demo output may imply the ARM64 fork build or the
128 GB live benchmark.

## 2. Data: A + B, with the split doing real work

### A. A live Wikipedia slice (`wiki_source.py`)

BFS from ~20 seed articles to 300–800 articles. For each: REST summary +
extract (unit `title`, `summary`, body chunks), page links, the Wikidata QID,
and the article's real revision history.

### B. HotpotQA fullwiki dev (`hotpot_link.py`)

Reuses `tools/fetch_hotpot.py`. Supplies **real multi-hop questions with gold
supporting titles**, so retrieval is graded (joint evidence recall@k) instead
of eyeballed.

### The bridge

HotpotQA gold titles are resolved into slice unit ids with the same
normalisation `tools/wiki_hotpot_link.py` uses. A question counts only when
**all** its gold titles resolve (`fully_resolved`); coverage is reported, never
hidden, and metrics are defined only on that subset.

### Edge typing — the technical core

| Wikipedia relation | GEM edge kind | Why |
| --- | --- | --- |
| article → article hyperlink | `association` | relatedness; expands retrieval context, **no propagation rights** |
| Wikidata `P31` / `P279` (instance of / subclass of) | `extension` | entailment; this is the only kind `revise` may traverse (C3) |

This is what makes C3 falsifiable on real data: change a label on an entity and
only the subclass chain may be flagged — a co-occurrence neighbour that is not
on an extension path must remain untouched.

### Provenance and reproducibility

`wiki_source.py` pins every revision id, caches raw API responses, and emits a
`manifest.json`. `scenario.py` never calls the network — it reads the manifest.
Wikipedia changes; a demo that re-fetches at run time is not a demo, it is a
different experiment each morning.

## 3. The five acts

| Act | Operator | Real action | What it proves |
| --- | --- | --- | --- |
| ① build | `ingest` | stream article bodies as `InteractionEvent`s (deterministic strategy) | construction cost recorded as `PhaseCost` on the [AM] phase timeline |
| ② ask | `retrieve` | HotpotQA multi-hop questions, `VECTOR` vs `FUSED` (`tjs_open`) | tri-modal in one plan; retrieval **writes salience in the same transaction** (C6) |
| ③ change | `revise` | replay real revisions: a label/claim moved | default query returns only the new value, `as_of` returns the old (C1); provenance chain intact (C4); propagation follows extension edges only (C3) |
| ④ forget | `forget` | one tick | never-retrieved units walk compressed → hidden → archived; retrieved units survive (C5 × C6); archived still recoverable by explicit lookup, and there is no `DELETE` anywhere in the operator |
| ⑤ report | `conformance` | C1–C6 + `manifest()` | per-condition verdicts, never a wholesale conformance claim |

## 4. Deliverables

```
bench/agent_memory/demo/
  wiki_source.py   A: fetcher -> JSONL + pinned manifest + response cache
  hotpot_link.py   B: question set, gold -> unit id resolution, coverage
  adapter.py       Article -> InteractionEvent; Revision -> revise evidence; Edge(kind)
  scenario.py      the five acts;每 act -> TransitionResult + PhaseCost
  report.py        bench/out/gem_wiki_demo/{report.md,results.json,manifest.json}
  __main__.py      CLI: --phase all|ingest|retrieve|revise|forget --dsn --scope
docs/agent_memory_gem_wiki_demo_v0.1.0.md   the demo script + honest labels
tests/test_gem_demo_adapter.py              offline fixtures, no database
tests/test_gem_demo_live.py                 gated on TRIDB_GEM_DSN
Makefile: gem-demo-fetch / gem-demo
```

Separation of concerns that must hold: the fetcher only writes files; the
scenario only reads them; every run writes `TriDBGovernedMemory.manifest(...)`.

## 5. Gates

| Gate | Work | Passes when |
| --- | --- | --- |
| **G0** ✅ | `createdb gem_demo`, `init_schema()`, run `tests/test_gem_live.py` — **the first execution of GEM's SQL against a real engine** — and fix what it exposes | **passed 2026-07-30**, see §5.1 |
| **G1** ✅ | `wiki_source` + `hotpot_link` + `adapter` + offline tests | **passed 2026-07-30**, see §5.2 |
| **G2** ✅ | act ① deterministic ingest; act ② `VECTOR` vs `FUSED` | **passed 2026-07-30**, see §5.4 |
| **G3** ✅ | act ③ `revise` over real revisions; act ④ `forget`; the C1–C6 report | **passed 2026-07-30**, see §5.5 |

G4 (`llm_mediated` batch/sequential + `agentic` on the local vLLM, giving [AM]'s
construction-cost table) starts only after G3.

### 5.1 G0 result — the first live evidence for GEM's SQL (2026-07-30)

Ran against `postgresql://hza214@127.0.0.1:55432/gem_demo`, stock PG 16 +
`vector 0.8.0` + `graph_store_am 0.2.0` + `tjs_pg 0.2.0`.

* `tests/test_gem_live.py` — **17 passed**. The five `gem_*` tables were created
  by `init_schema()` and the trajectory log filled, so this is real engine
  execution, not a skipped suite.
* Offline suites (`test_gem_unit` / `_strategies` / `_roundtrip` /
  `_conformance`) — 141 passed, 7 skipped.
* **No defect was found.** The G0 rework budget is unspent.

Two things the live suite does *not* cover, probed separately because act ② of
the demo depends on them:

**All four retrieval modes commit on the engine.** The suite only ever asks for
`RetrievalMode.VECTOR`; `FUSED`, `GRAPH` and `RELATIONAL` were unverified. A
direct probe over four linked units:

| mode | hits | `via` | probes worth noting |
| --- | --- | --- | --- |
| `VECTOR` | 3 | vector | `hnsw_iterative_scan=strict_order` |
| `FUSED` | 1 | vector | `graph_examined=1`, `termination_reason=filter_first`, `relaxed_order`, `anchor_id` honoured |
| `GRAPH` | 1 | graph | `edge_type=0` (ANY) |
| `RELATIONAL` | 3 | vector | no HNSW scan |

The graph leg genuinely runs. Note `FUSED` returned fewer hits than `VECTOR` on
this four-unit toy; on a corpus this size that is uninformative, and **G2 must
establish whether it is the filter-first plan behaving correctly or an early
termination worth reporting** before any fused-vs-vector number is quoted.

**Re-ingesting the same `external_id` aborts; it is not idempotent.** The
second `ingest` of an identical event returns `committed=False` with
`aborted_reason = UniqueViolation on gem_field_value_current_uq (unit_id,
field)`. The partial unique index is doing exactly its C1 job — at most one
current value per (unit, field) — and the abort leaves `M_t` unchanged and
logged, which is C2 behaving correctly. But it has two consequences the demo
must handle at G1:

1. Every demo run uses a **fresh scope**, or `--reset` drops the scope first.
   Re-running `--phase ingest` against a populated scope will abort by design.
2. Open design question, deliberately *not* decided here: [GEM] §3.2 says
   ingestion must "integrate, do not append blindly", which argues that
   `DeterministicIngestStrategy` should recognise an already-ingested
   `external_id` and skip it rather than collide at the index. Changing that is
   an operator-semantics decision, not a demo fix, so it is filed here and left
   alone.

### 5.2 G1 result — the slice (2026-07-30)

`make gem-demo-fetch` produces, from a cold cache in ~15 minutes:

| | |
| --- | --- |
| articles | 700 (all with a QID) |
| hyperlinks in slice | 13 657 → association edges |
| Wikidata classes | 1 445 → class units |
| extension edges | 4 343 (membership + subsumption) |
| Wikidata revisions | 2 400 over 60 article entities + 20 extension classes |
| parsed as field assignments | 1 437 (59.875%) |
| **observed supersessions** | **168 fields, across 72 entities** |
| HotpotQA questions fully resolved | **125** of the first 150 (2 gold titles each) |

The resulting plan is 2 144 units, 2 087 field values and 17 998 edges — one
transaction.

Three things this gate settled, none of them predictable from the plan:

**Corpus construction runs question-first, not topic-first.** The original
design seeded from 20 computing articles and expected HotpotQA to overlap. It
resolved **0 of 1500** dev questions — HotpotQA is mostly films, bands and
athletes. Seeding instead from the gold titles of the first 150 questions
resolves 125 of them (the other 25 lost a gold title to 2017→2026 title drift).
The cost is stated wherever the metric appears: **the pool is built to contain
the gold**, so act 2 measures ranking within 700 in-domain articles, never
retrieve-from-all-of-Wikipedia. The topic seeds stay, but for graph depth only.

**Two live-API defects, both found by running it.** `pllimit` bounds links for
the whole query rather than per page, so an unpaginated fetch truncated the link
graph to a quarter of its size *and* biased the survivors alphabetically —
exactly the bias the co-citation rule exists to prevent. And the expansion walk
filtered its frontier on "already collected", so a title that redirects onto a
held article, or returns no lead extract, was re-proposed every round: the fetch
span at 387/400 forever. Both are fixed; the second has a regression test that
fails on the old logic.

**Edit-comment parsing recovers 60%, and the rest are not failures.**
`wbeditentity-update` batch edits carry no field-level assignment at all. The
demo reports the parse rate rather than implying the stream was fully consumed.

### 5.3 What G2 inherited (resolved)

* `WikiSliceIngestStrategy` and `WikidataRevisionStrategy` now run against the
  live store in the one-command scenario.
* The `FUSED`-returns-fewer-hits question from §5.1 is resolved by §5.4.
* `scenario.py`, `report.py`, `__main__.py`, `tests/test_gem_demo_live.py`,
  the Make targets, and the demo runbook now exist.

### 5.4 G2 result — deterministic construction and retrieval (2026-07-30)

`make gem-demo` created exactly 2 144 units (zero title-collision updates),
2 087 field values and 17 998 typed edges in one transaction. Construction
took 48.529 s, one batched embedding call over 2 144 sequences, and 46 667 DB
statements.

On all 125 fully resolved HotpotQA questions at k=10:

| operating point | joint evidence recall@10 | mean evidence recall@10 | returned | short | probe outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| VECTOR / `strict_order` | 0.864 | 0.932 | 10.0 | 0 | no tjs probe |
| FUSED / `relaxed_order` | 0.912 | 0.952 | 10.0 | 0 | `term_cond` 125/125; 0 capped; 0 censored |

The modes are separate tables in the rendered report and separate objects in
`results.json`; every per-query probe is copied verbatim. The G0 four-unit
short result does not reproduce on the live slice: filter-first returns all ten
requested units for all 125 queries.

### 5.5 G3 result — real revisions, typed propagation and forgetting (2026-07-30)

The first attempt exposed a data-selection defect: all 60 revision sources were
articles, but extension edges are class → member, so no changed source had a
positive extension frontier. The fetcher now additionally pins 20
high-frontier class histories. A second live failure exposed a real operator
defect: split repair created a new extension child after the propagation walk
without flagging it, so the C3 policy rejected the transaction. The split now
flags its new child in the same transaction.

The final run replays 1 437 parsed edits, touches 81 units, and supersedes 257
stored values. Evidence:

* **C1:** real `Q183/P209` defaults to `Q56025`; `as_of` the preceding interval
  returns `Q702424`.
* **C2:** a stored impossible-bound policy aborts a proposed write with
  byte-identical pre/post state and a logged reason.
* **C3:** 896 extension-reachable units propagate; 2 124 observed
  association-only neighbours from changed units do not.
* **C4:** 3 540 values, zero broken supersession chains.
* **C5:** all 1 348 never-retrieved units archive, all 806 retrieved units avoid
  archival, row count stays 2 154, and explicit archived lookup succeeds.
* **C6:** 806 retrieved units have positive salience and outrank unretrieved
  units.

The generated condition report has C1–C6 satisfied individually, with no
violated or unchecked condition. It deliberately makes no wholesale label.

## 6. Labels that travel with every number

1. Graph reads on this engine are **commit-visible, not snapshot-isolated**
   (interface §6.5). No conformance claim may depend on repeatable-read
   topology.
2. `reinforce`, `revise` and `forget` go in every run manifest. A run with any
   of them on is a different operating point from an [AM] paper row.
3. Conformance is reported **per condition**. "GEM-conformant" is not a phrase
   this demo is permitted to print.
4. HotpotQA metrics are defined only on the `fully_resolved` subset, and the
   coverage fraction is printed next to them.
5. Stock PG 16, x86_64. Not the GX10 fork, not the 128 GB benchmark.

## 7. Risks

| Risk | Handling |
| --- | --- |
| GEM's SQL has never touched an engine | that *is* G0; expect rework and budget for it before any demo work |
| Wikipedia drifts under the demo | pin revision ids, cache raw responses, never fetch inside `scenario.py` |
| Gold-title resolution coverage may be low on a 300–800 article slice | report coverage; widen the BFS seeds if the resolved subset is too small to be meaningful |
| `forget` changes the corpus between queries | act ④ runs after act ②, and the retrieval numbers quoted are the pre-forget ones; a post-forget re-run is reported as a separate row |
