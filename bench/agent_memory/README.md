# TriDB adapters for LongMemEval and LoCoMo

These adapters use TriDB as a scoped PostgreSQL/pgvector memory backend while
preserving each benchmark's native ids and output contracts. This first
version is intentionally vector-only: it establishes a reproducible storage
and retrieval baseline before adding graph construction or LLM-derived memory
units.

## What is implemented

| Component | LongMemEval | LoCoMo |
|---|---|---|
| Scope | one `question_id` | one `sample_id` |
| Ingestion | session or turn | dialogue turn |
| Reuse | corpus rebuilt per question, matching the official flat retriever | conversation ingested once for all its QA items |
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

## LoCoMo

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
