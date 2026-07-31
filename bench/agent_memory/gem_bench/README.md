# `gem_bench` — reproducing [AM] §4.1, §4.2 and §4.8 on TriDB/GEM

`[AM]` is **arXiv:2606.06448**, *Agent Memory: Characterization and System
Implications of Stateful Long-Horizon Workloads*. It profiles nine memory
systems across three phases; this package reproduces the three sections a
single-system repository can reproduce honestly.

| Section | Figures | Status |
|---|---|---|
| §4.1 serving latency vs accuracy | Fig. 2 | reproduced, **without** the long-context arm |
| §4.2 construction dominates | Fig. 3, Fig. 4, Table 3 | reproduced; energy columns need `nvidia-ml-py` |
| §4.8 serving latency structure | Fig. 10, Fig. 11 | reproduced |
| §4.7 per-user footprint growth | Fig. 9 | **not reproduced** — out of scope |

## What is and is not being compared

The paper's headline is a two-orders-of-magnitude **spread across systems**. We
have one system. So each arm here is a GEM *setting* chosen to reproduce a
paradigm's cost shape — `memory.py`'s configuration matrix made executable:

| Arm | Paradigm | ingest | mode | revise | forget | reinforce |
|---|---|---|---|---|---|---|
| `II_embedrag` | II | deterministic | VECTOR | off | off | off |
| `IIIa_graphrag_like` | III.a | LLM, batched | FUSED | off | off | off |
| `IIIb_mem0_like` | III.b | LLM, sequential | VECTOR | on | off | off |
| `IV_agentic` | IV | agentic, capped | FUSED | on | on | off |
| `gem_conformant` | GEM | deterministic | FUSED | **on** | **on** | **on** |

Every emitted record carries `paradigm_proxy: true`. **A row labelled
`IIIa_graphrag_like` is not GraphRAG.** These numbers belong beside the paper's
paradigm claims (Insight 1, 2, 8), never beside its per-system bars.

`gem_conformant` shares Paradigm II's ingest strategy on purpose: its delta
against `II_embedrag` is the cost of governance with construction held
identical — the comparison [AM]'s Table 1 has no row for.

## Running it

Prerequisites:

1. **A separate database.** This runs at Qwen3-Embedding-0.6B's dimension
   (1024) and `gem_unit.embedding` is fixed at whatever dimension created it —
   `gem_demo` is `vector(384)`. `GemStore.init_schema` refuses the mismatch
   loudly rather than corrupting the store.
   ```sh
   createdb gem_bench     # needs vector, graph_store_am, tjs_pg
   ```
2. **Both vLLM endpoints**: `scripts/serve_longmemeval_vllm.sh`
   (answer `:8000`, embedding `:8001`). Each endpoint must advertise exactly
   one model; the runner checks this before the first history rather than after
   five hours of ingest.
3. `pip install -r requirements-agent-memory.txt` — includes the optional
   `nvidia-ml-py`. Without it every `gpu_joules` is `NULL` and Table 3 has
   three of its five columns. That is a reported state, not a failure.

```sh
make gem-bench-smoke     # 1 history, 1 question, no judge, no energy
make gem-bench           # 5 x 60 = 300 queries per arm
```

Cost warning: the agentic arm issues LLM calls per chunk *and* per tool round
over ~1.8 M tokens of history. [AM] measured comparable systems at 4–14 hours on
one H100. Start with `--points II_embedrag` and add arms deliberately.

## Output

```
<output-dir>/
  run_manifest.json         models, caps, energy sampler, git state
  paper_sections.json       the three sections, one row per arm
  report.md                 the same, as Markdown tables
  <arm>/
    construction.jsonl      per history: seconds, units, cost, energy
    predictions.jsonl       per query: timing, hits, usage, energy, probes
    maintenance.jsonl       forget ticks (arms with forget on)
    judge_results.jsonl
    call_ledger.jsonl       every model call, phase-tagged
    summary.json            the arm's three sections plus raw aggregates
```

## Honesty properties worth knowing

- **An unmeasured joule is `None`, never `0`.** `_energy_total` reports
  `windows_missing` and `windows_partial_coverage`; a lifecycle total is `None`
  if any phase went unsampled.
- **The judge is local by default**, so `judge_protocol` reads
  `protocol_variant`, not `memoryagentbench_gpt4o`. Accuracy here is comparable
  across these arms and not against the paper's published numbers.
- **Judge calls are excluded** from every serving metric and from
  `paper_model_calls`.
- **Two independent call counters** are reported side by side: GEM's own
  `PhaseCost` meter and the HTTP `CallLedger`. They are not reconciled, so a
  disagreement is visible rather than averaged away.
- **A forget tick is neither construction nor serving.** It is timed as
  `maintenance` and kept out of both totals, because Table 3 prices
  "Construct + 300 QA".
- **Caps are recorded, never silent.** An agentic run that exhausts
  `--agentic-max-rounds` sets `capped: true` ([AM] Recommendation 10), and a
  construction rejection rate over `--max-rejection-rate` aborts the history as
  a FAILED CONFIGURATION rather than reporting a corrupted store's accuracy
  ([AM] §4.4).
- **Absolute wallclock and joules are this box's**, not the paper's H100. Only
  the spread across arms measured here is comparable.

## Shared code

Everything that is not a memory system — workload loading, streaming
generation, effective-TTFT timing, the MemoryAgentBench judge prompts,
percentile summaries — lives in `bench/agent_memory/serving.py`, which
`tridbBackend/longmemeval_pipeline.py` imports too. That sharing is the point:
Fig. 2 compares memory systems, and it can only do so if both arms generate,
time and grade through identical code. `tests/test_gem_bench.py` asserts the
pipeline still resolves those names through the shared module.
