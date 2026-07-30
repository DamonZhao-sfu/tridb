# TriDB adapters for LongMemEval and LoCoMo

These adapters use TriDB as a scoped PostgreSQL/pgvector memory backend while
preserving each benchmark's native ids and output contracts. This first
version is intentionally vector-only: it establishes a reproducible storage
and retrieval baseline before adding graph construction or LLM-derived memory
units.

For the phase-aware construction, retrieval, generation, energy, freshness,
growth, and tail-latency methodology, see
[`docs/agent_memory_workload_characterization_v0.1.0.md`](../../docs/agent_memory_workload_characterization_v0.1.0.md).

## What is implemented

| Component | LongMemEval | LoCoMo |
|---|---|---|
| Scope | one `question_id` in the retrieval adapter; one shared history in the end-to-end pipeline | one `sample_id` |
| Ingestion | session or turn | dialogue turn |
| Reuse | retrieval adapter rebuilds per question; end-to-end MemoryAgentBench pipeline builds each of five histories once for its 60 questions | conversation ingested once for all its QA items |
| Output | official-compatible retrieval JSONL | original JSON plus ranked ids, text, and prompts |
| Metrics | recall-any/all and NDCG at 1/3/5/10/30/50 | evidence ids are returned for the official evaluator |
| Ground-truth isolation | `has_answer` is used only after retrieval to grade ids | `evidence` is never read by the index builder |

The default embedder is `BAAI/bge-small-en-v1.5` through fastembed
(`vector(384)`, cosine distance). Use the same model and dimension for every
run being compared.

## Start TriDB

From the TriDB repository:

```bash
docker build -f scripts/pg17/Dockerfile.release \
  -t tridb/postgres-trimodal:pg17 .
docker run -d --name tridb-memory \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=tridb \
  tridb/postgres-trimodal:pg17

python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.lock
export TRIDB_DSN=postgresql://postgres:tridb@localhost:5432/postgres
```

The adapters bootstrap the `vector` extension, benchmark table, relational
scope indexes, and cosine HNSW index. They do not modify the MCP `memories`
table.

## LongMemEval

Session-granularity retrieval matching the official user-only flat-index
input:

```bash
python -m bench.agent_memory.longmemeval_adapter \
  --input /path/to/longmemeval_s.json \
  --output bench/out/longmemeval_tridb_session.jsonl \
  --granularity session \
  --retrieve-k 50
```

Turn-granularity retrieval:

```bash
python -m bench.agent_memory.longmemeval_adapter \
  --input /path/to/longmemeval_s.json \
  --output bench/out/longmemeval_tridb_turn.jsonl \
  --granularity turn \
  --retrieve-k 50
```

The JSONL can be passed directly to LongMemEval's
`src/generation/run_generation.py` with `--retriever_type flat-session` or
`flat-turn`. It contains the original question data plus:

```text
retrieval_results.ranked_items
retrieval_results.metrics
retrieval_results.backend
```

By default, only user turns are indexed to match the official flat baseline.
`--include-assistant` indexes both roles and records that choice in the
output. Report it as a separate role-preserving variant; do not compare it to
the user-only baseline without noting the corpus change.

### End-to-end MemoryAgentBench LongMemEval pipeline

The paper-reproduction path is

```text
5 histories × (construct TriDB once + answer 60 questions) = 300 questions
```

It uses `Qwen/Qwen3-32B` on the local port 8000 for streamed answer generation
and `Qwen/Qwen3-Embedding-0.6B` on port 8001 for construction/query embeddings.
The answer endpoint is deliberately strict: the run fails before construction
unless port 8000 advertises exactly `Qwen/Qwen3-32B`.

Prepare the optional dependencies and official five-history dataset:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements-agent-memory.txt
.venv/bin/python -m nltk.downloader punkt punkt_tab

.venv/bin/python tools/fetch_memoryagentbench_longmemeval.py \
  --output data/longmemeval/memoryagentbench_longmemeval_sstar.json
```

Start the two vLLM components in separate terminals. The launcher refuses to
replace a service when the selected port is already occupied:

```bash
export TRIDB_LME_VLLM_BIN=/path/to/vllm
scripts/serve_longmemeval_vllm.sh answer

# Second terminal; defaults to GPU 1 and port 8001.
export TRIDB_LME_VLLM_BIN=/path/to/vllm
scripts/serve_longmemeval_vllm.sh embedding
```

The answer launcher loads the official `Qwen/Qwen3-32B-FP8` artifact used for
the local FP8 regime but advertises the stable experiment model name
`Qwen/Qwen3-32B` on port 8000. Override `TRIDB_LME_ANSWER_ARTIFACT` only when
recording a separate precision/quantization variant.

Verify the advertised models:

```bash
curl -sS http://127.0.0.1:8000/v1/models
curl -sS http://127.0.0.1:8001/v1/models
```

Run a one-question smoke experiment without judging:

```bash
export TRIDB_DSN="postgresql://$(id -un)@127.0.0.1:55432/postgres"

.venv/bin/python -m bench.agent_memory.longmemeval_pipeline \
  --input data/longmemeval/memoryagentbench_longmemeval_sstar.json \
  --output-dir bench/out/longmemeval_tridb_qwen32b_smoke \
  --limit-samples 1 \
  --limit-questions 1 \
  --skip-judge
```

Run all 300 questions and use the official MemoryAgentBench GPT-4o judging
protocol:

```bash
export OPENAI_API_KEY=...

.venv/bin/python -m bench.agent_memory.longmemeval_pipeline \
  --input data/longmemeval/memoryagentbench_longmemeval_sstar.json \
  --output-dir bench/out/longmemeval_tridb_qwen32b \
  --top-k 10
```

The default embedding batch size is 64. With roughly 80–100 chunks per
history, this normally produces two construction-embedding HTTP calls per
history, making the full call ledger comparable to the paper's 610-call
embedRAG row: about 10 construction embedding calls, 300 query embedding
calls, and 300 answer-generation calls. The ledger records both HTTP calls and
the number of embedded items; do not hide a different batching policy.

TriDB still retrieves and records `top_k=10`, but the local Qwen3-32B prompt
uses at most the first five 4,096-token chunks. This follows the paper's local
context-overflow rule while preserving retrieval@10. Override
`--max-prompt-memories` only as a separately reported prompt-budget variant.

When GPT-4o is unavailable, a local self-judge can produce a protocol-variant
accuracy. It must not be reported as the paper-equivalent accuracy:

```bash
.venv/bin/python -m bench.agent_memory.longmemeval_pipeline \
  --input data/longmemeval/memoryagentbench_longmemeval_sstar.json \
  --output-dir bench/out/longmemeval_tridb_qwen32b_local_judge \
  --top-k 10 \
  --judge-base-url http://127.0.0.1:8000/v1 \
  --judge-api-key EMPTY \
  --judge-model Qwen/Qwen3-32B
```

The output directory contains:

```text
run_manifest.json       pinned input/model/software configuration
events.jsonl            construction and per-query timing events
predictions.jsonl       answers, retrieved chunks, usage, and timing
judge_results.jsonl     offline correctness decisions, when enabled
call_ledger.jsonl       embedding, answer, judge, insert, and retrieval calls
summary.json            accuracy, walltime, TTFT, total-time, and tail metrics
```

`summary.json` reports effective TTFT from query admission through the first
streamed answer token, total response time through the final token, p50/p95/p99,
phase breakdowns, construction + QA walltime, and exact call accounting. Judge
time and judge calls are explicitly excluded from the paper-style serving
walltime and call total.

## LoCoMo

Retrieval only:

```bash
python -m bench.agent_memory.locomo_adapter \
  --input /path/to/locomo/data/locomo10.json \
  --output bench/out/locomo_tridb.json \
  --top-k 10
```

For every QA item, the adapter adds:

```text
tridb_prediction_context    ranked LoCoMo dia_id values
tridb_prediction_retrieval  ids, text, timestamps, and scores
tridb_prediction_prompt     context plus question for an answer model
```

Run an answer model on each `tridb_prediction_prompt` and save its response as
`tridb_prediction`. LoCoMo's existing `eval_question_answering` function can
then use `eval_key="tridb_prediction"`; it will also read
`tridb_prediction_context` to calculate evidence recall.

Change `--prediction-key` when comparing multiple answer models or retrieval
configurations in the same file.

### End-to-end LoCoMo pipeline

The resumable pipeline runs TriDB retrieval, answer generation through an
OpenAI-compatible endpoint, binary LLM judging, and metric aggregation:

```bash
export TRIDB_DSN="postgresql://$(id -un)@127.0.0.1:55432/postgres"
export VLLM_BASE_URL="http://127.0.0.1:8000/v1"
export VLLM_API_KEY="EMPTY"

python -m bench.agent_memory.locomo_pipeline \
  --input /path/to/locomo/data/locomo10.json \
  --retrieval-output bench/out/locomo_tridb_top20.json \
  --output bench/out/locomo_tridb_qwen.json \
  --metrics-output bench/out/locomo_tridb_qwen_metrics.json \
  --top-k 20 \
  --workers 4
```

When `--model` is omitted, the endpoint must advertise exactly one model.
Generation and judging default to that model. Use `--judge-base-url`,
`--judge-api-key`, and `--judge-model` for a separate judge.

The output JSON is checkpointed and can be resumed with the same command. The
metrics report contains evidence recall/hit/MRR/NDCG, diagnostic lexical F1,
token and latency summaries, and LLM-judged accuracy for Single, Multi,
Temporal, Open, and Overall. Category 5 is excluded by default to match the
four-category Mandol paper table. A local-model judge produces the same metric
shape but is not protocol-identical to Mandol's GPT-backed paper result.

## Scope modes

The default is benchmark isolation:

- LongMemEval truncates its dedicated table before each question corpus.
- LoCoMo truncates its dedicated table before each conversation.

This matches the official evaluation semantics and avoids carrying HNSW
tombstones from one independent corpus into the next. `--shared-index`
retains all scopes and pushes `scope_id` into the ANN query. Use that mode to
measure persistent multi-user filtering rather than to reproduce the official
flat baselines.

## Smoke runs

Use `--limit 2` for LongMemEval or `--limit-samples 1` for LoCoMo before a
full run. The fast Python test layer covers parsing, ids, role handling,
metric shapes, ground-truth isolation, output contracts, and SQL scope
pushdown:

```bash
pytest tests/test_agent_memory_adapters.py -q
ruff check bench/agent_memory tests/test_agent_memory_adapters.py
ruff format --check bench/agent_memory tests/test_agent_memory_adapters.py
```

The adapters report retrieval quality only. End-to-end QA scores still depend
on the selected answer model and the benchmark's official evaluation script.
