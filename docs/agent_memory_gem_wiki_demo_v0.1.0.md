# GEM Wikipedia demo

> Version 0.1.0 · measured 2026-07-30 · stock PostgreSQL 16, x86_64

## Run it

From the TriDB repository root:

```bash
make gem-demo
```

This runs all five acts against
`postgresql://hza214@127.0.0.1:55432/gem_demo`, uses the stable
`gem-wiki-demo` scope, and resets only that scope before rebuilding it. The
database must already have `vector`, `graph_store_am`, and `tjs_pg` available.

Outputs:

- `bench/out/gem_wiki_demo/report.md`
- `bench/out/gem_wiki_demo/results.json`
- `bench/out/gem_wiki_demo/manifest.json`

To fetch or reproduce the pinned source files first:

```bash
make gem-demo-fetch
make gem-demo
```

The scenario itself never accesses the network. `gem-demo-fetch` is the only
networked step, pins API responses under `data/wiki_demo/cache`, and writes the
source manifest consumed by the scenario.

The equivalent direct command is:

```bash
.venv/bin/python -m bench.agent_memory.demo \
  --phase all \
  --dsn postgresql://hza214@127.0.0.1:55432/gem_demo \
  --slice data/wiki_demo \
  --scope gem-wiki-demo \
  --output bench/out/gem_wiki_demo \
  --reset
```

Individual phases are available through `--phase
ingest|retrieve|revise|forget`, but they expect the preceding state to exist.
A run without `--reset` should use a fresh `--scope`; re-ingesting a populated
scope is deliberately not idempotent.

## Measured G2 result

The pinned plan contains 2,144 units, 2,087 initial field values, and 17,998
typed edges. Deterministic ingest committed them in one transaction with one
batched embedding call:

| Metric | Result |
| --- | ---: |
| units created / updated | 2,144 / 0 |
| field values | 2,087 |
| edges | 17,998 |
| ingest time | 48.529 s |
| DB statements | 46,667 |

HotpotQA metrics cover only the 125 fully resolved questions out of 1,500
considered. The corpus was built to contain the gold titles, so this is ranking
inside a 700-article in-domain pool, not retrieval from all of Wikipedia.

VECTOR and FUSED are separate operating points:

| VECTOR (`strict_order`) | Result |
| --- | ---: |
| joint evidence recall@10 | 0.864 |
| mean evidence recall@10 | 0.932 |
| mean returned units | 10.0 |
| short / failed queries | 0 / 0 |
| aggregate retrieval time | 1.973 s |

| FUSED (`relaxed_order`) | Result |
| --- | ---: |
| joint evidence recall@10 | 0.912 |
| mean evidence recall@10 | 0.952 |
| mean returned units | 10.0 |
| short / failed queries | 0 / 0 |
| termination reason | `term_cond` for 125/125 |
| budget-capped / graph-censored | 0 / 0 |
| aggregate retrieval time | 2.535 s |

This resolves the G0 toy-corpus question: on the live slice the filter-first
FUSED path returns the full requested result count. Its probes show bounded
`term_cond` termination without budget capping or graph censoring. Every raw
per-query probe is reproduced under
`acts.retrieve.operating_points.<mode>.queries[].probes` in `results.json`.

## Measured G3 result

The pinned source now contains 2,400 real revisions: 60 high-link-degree
article entities plus 20 high-extension-frontier classes. Of those revisions,
1,437 (59.875%) carry a parseable field assignment, with 168 observed repeated
fields. Class histories are necessary for a positive C3 test because extension
edges are oriented class → member.

The revision replay touched 81 semantic units (one QID legitimately has both
article and class representations), superseded 257 stored values, and then ran
the state-level `revise` operator.

- C1: for real field `Q183/P209`, the default value is `Q56025`; `as_of`
  2026-07-24T07:02:25-07:00 returns the prior `Q702424`.
- C2: a deliberately impossible stored policy rejected a proposed field
  update; the pre/post state SHA-256 fingerprints are identical and the
  aborted transition retains its reason.
- C3: 896 extension-reachable units were propagated. Across 9,180 association
  edges from changed units, 2,124 association-only neighbours remained outside
  the propagation set.
- C4: 3,540 values remain, with zero broken supersession links and zero closed
  non-archived values missing a successor.
- C5: the forgetting tick keeps all 2,154 rows. All 1,348 never-retrieved units
  archive; all 806 retrieved units avoid archival; an archived article is
  recovered by explicit lookup.
- C6: every retrieved unit has positive salience, and retrieved units outrank
  never-retrieved units after reinforcement.

The report names C1–C6 individually. It does not print a wholesale conformance
label.

## Labels and limits

- Stock PostgreSQL 16 on x86_64; this is not the GX10 fork or the 128 GB
  benchmark.
- Graph reads are commit-visible, not snapshot-isolated. No result depends on
  repeatable-read topology.
- `reinforce`, `revise`, and `forget` are enabled and recorded in both
  retrieval manifests. This operating point is not pooled with an Agent Memory
  paper row that disables them.
- Salience decay is lazy and is current only as of the forgetting tick.
- Construction-form comparison with LLM-mediated and agentic strategies is G4
  and is not part of this run.
