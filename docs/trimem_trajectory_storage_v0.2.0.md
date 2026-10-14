# TriMEM trajectory storage v0.2.0: EvoTrace in a tri-modal store

Status: design, not built. Supersedes the implicit v0.1 layout in `trimem/data/evotrace.py`, which
loads only `attempt`, lineage edges, a self-built kNN `similar` graph, and `task`.
Every number below comes from a read-only scan of `dataset/evotrace/normalized/` and
`dataset/evotrace/raw/` (revision `349117b0`) on 2026-10-13. No database was written.

## 1. What the dataset actually contains

| file | rows | grain and key | link to a node | loaded in v0.1? |
|---|---:|---|---|---|
| nodes | 10,672 | `node_uid` = `rev:run_rel#program_id` | itself | yes, as `attempt` |
| vectors.npz | 10,690 | `uids` | 10,672 `kind=node` (track `node_artifact`) + 18 `kind=task` (track `task_spec`, uid = `task_uid`) | node vectors only |
| lineage_edges | 10,479 | (src, dst) = parent to child, `has_child` | both ends are nodes | yes |
| context_edges | 30,086 | (src, dst) = context program to the node it was shown to, `context_for` | both ends are nodes | **no** |
| prompts | 6,536 | `prompt_uid`; `node_uid` 1:1 | `node_uid` (exact) | **no** |
| llm_calls | 18,400 | (`session_uid`, `seq`); `seq == row_offset` in raw `logs/llm_calls.jsonl` | **none in the file**; recoverable (see 1.3) | **no** |
| state_events | 488,658 | (`session_uid`, `iteration`, `entity`, `role`, `slot_key`) | `node_uid` on membership rows | **no** |
| decision_points | 18,291 | `dp_id` | `outcome_uid`, `seed_uid` | **no** (and should not be, see 1.4) |
| artifacts | 9,733 | `artifact_uid` = sha256 of code | via `nodes.artifact_uid` (10,672 refs, 21 shared across sessions) | code only, inlined |
| sessions | 121 | `session_uid` | via `nodes.session_uid` | as a text column only |
| tasks | 18 | `task_uid` | via `nodes.task_uid` | as `task(task, specification)` |

### 1.1 vectors.npz

`uids`, `kinds`, `sources`, `vectors` (10,690 × 1024 float32, L2-normalised, no zero rows, no
duplicate uids). The 18 non-node rows are exactly the 18 tasks, `kind=task`, `source=task_spec`.
**Every node has a vector**, and no vector lacks a node. The two tracks are separate experiments
(`embedding_manifest.json`). The node track embeds the first 2,000 characters of code. It was
"reconstructed from loaded rows without re-embedding", so its timing is unknown. The task track is
not loaded anywhere today.

### 1.2 context_edges

Each edge comes from the raw `programs.jsonl:other_context_ids` list. It says "program `src` was in
the prompt as an inspiration, sibling or top-program when the LLM produced `dst`". The direction is
context to consumer. The fields are `src_node_uid`, `dst_node_uid`, `session_uid`, `relation`
(always `context_for`), `provenance` and `row_offset` (the dst row).

- Node to node only, 100% resolvable. **0 cross-session edges**, 0 self-loops, 0 duplicate pairs
  (13 duplicate ids were dropped in normalisation). **0 overlap with the parent edge**: the parent
  is never also listed as context. 30,029 edges point backward in iteration and 57 come from the
  same iteration.
- The in-degree is capped at 4 in three backends. That is the frameworks' inspiration count.

| backend | edges | nodes with ≥1 context | in-degree histogram |
|---|---:|---:|---|
| evox | 9,039 | 2,312 / 2,396 | 4:2,191, 3:62, 0:84 |
| gepa_native | 6,938 | 1,985 / 2,137 | 4:1,522, 1:213, 0:152 (includes rejected leaves) |
| openevolve_native | 11,120 | 3,286 / 4,267 | 4:2,586, 1:624, 0:981 |
| shinkaevolve | 2,989 | 1,443 / 1,872 | 0:429 … 6:4 (from `archive_inspiration_ids`; `top_k_inspiration_ids` is always empty) |

- Labels: the raw `context_info` label is `''` everywhere except 92 gepa edges labelled
  `Merge Parent B`. Those come from gepa's merge proposals, where `parent_info` is `Merge Parent A`.
  For those 92 nodes the context edge is a **second parent**, not an inspiration.
- On **math:heilbronn_triangle** there are 19 sessions (oe 9, evox 5, gepa 4, shinka 1), 1,667
  nodes, 4,755 context edges and 1,644 lineage edges. The in-degree is again mostly 4. Mean
  out-degree is 2.5–3.8 with hubs up to 76 (evox) or 45 (gepa): a few elites are shown again and
  again. 24% of context sources are ancestors of the consumer and 76% are off-lineage.
  37% of context sources have higher fitness than the parent.

**Retrieval value.** Context edges add a second kind of influence path inside a session. A
"which programs shaped this one" or "how was this elite reused" query becomes a 1-hop typed
traversal instead of a population replay. Out-degree also gives a free in-run "elite reuse"
signal. They **do not** help cross-session retrieval (0 cross-session edges), so they do not
replace `similar`.

### 1.3 llm_calls and prompts

`llm_calls.jsonl` keeps only `session_uid, seq, model, api_base, temperature, finish_reason,
prompt_tokens, completion_tokens, reasoning_tokens, error, provenance, row_offset`. **It drops the
response text.** The raw log at `raw/<run_rel>/logs/llm_calls.jsonl` row `seq` still has `content`,
`reasoning_content` and, for shinka only, `system_msg` and `user_msg`.

There is no program id on either side, and `integrity_report` says the join "would be a
fabrication". **That is too pessimistic.** Two exact keys exist:

- **prompts → node:** `prompts.node_uid` (exact). The blob `raw/<run>/<blob_rel>` is
  `{template: {system, user, responses:[text]}}`. The templates are `diff_user_message`,
  `full_rewrite_user_message`, or `merge` (gepa).
- **llm_call → node:** `sha1(strip(blob.responses[0]))` equals `sha1(strip(raw_llm.content))`
  within the session. For shinka, `programs.metadata.llm_result.content` plays the role of the
  blob. **Every node that has response text matches an llm_call: 8,318 / 8,318.** There are 30
  duplicate contents within sessions, so 8,303 distinct calls are linked. A call can map to 2
  nodes, e.g. gepa 400 nodes vs 386 calls.

Coverage:

| backend | nodes | with prompt blob | with LLM response text | linked llm_calls | unlinked calls (error / no content / content) |
|---|---:|---:|---:|---:|---|
| evox | 2,396 | 2,366 | 2,366 (98.7%) | 2,365 | 724 / 1 / 1,610 (search-algorithm co-evolution calls) |
| gepa_native | 2,137 | 400 | 400 (18.7%) | 386 | 402 / 0 / 3,406 (proposals whose program was not retained) |
| openevolve_native | 4,267 | 3,770 | 3,770 (88.4%) | 3,770 | 401 / 3 / 890 |
| shinkaevolve | 1,872 | 0 (prompt in llm log) | 1,782 (95.2%) | 1,782 | 0 / 135 / 2,525 (per-program summaries, meta-recommendations, novelty) |
| total | 10,672 | 6,536 | 8,318 (77.9%) | 8,303 | 1,527 / 139 / 8,431 |

The roots (one per session) never have a call. Gepa is the weak backend: 1,708 of its 2,137
nodes are rejected proposals rescued from a 20-slot sliding window, and they have no prompt
blob. A heuristic test for whether all `>>>>>>> REPLACE` blocks of an unlinked call are
substrings of a node's code links 943 more gepa calls to exactly one node, plus 42 ambiguous ones.
This is **heuristic**. It should be loaded only with `link_method='replace_block'` and kept apart
from exact links.

**Ground-truth change intent exists.** The response begins with the LLM's own reasoning before the
first `<<<<<<< SEARCH`. Shinka has `<NAME>`/`<DESCRIPTION>`, and also
`metadata.patch_name`/`patch_description` on all 1,872 nodes. The node field `changes` (the
`changes_raw` column) is only the framework's mechanical diff summary. Two heilbronn_triangle
examples:

- openevolve `a1fd3e6b`, iteration 5. `changes`: *"Change 1: Modified function
  `heilbronn_triangle11` (403→272 lines)"*. The response says: *"Very slow execution (228 seconds)
  due to exhaustive triangle area calculations … The score is stuck at 0.3683 … we should use a
  direct construction approach based on known optimal or near-optimal configurations."*
- shinka `2dc015b2`, iteration 5. `changes` is null. `patch_description` says: *"fix_collinear_points:
  The original program placed multiple points on the edges … leading to three collinear points …
  The fix perturbs points slightly off the edges and hypotenuse to break collinearity."*

Of 16.6k content-bearing calls, 8,714 have non-empty prose: about 7.7 MB in total, median 766
characters, p90 1,376.

**Tokens** (non-error calls): evox medians are prompt 10.2k, completion 2.8k, reasoning 1.2k. Gepa
is 15.1k / 4.2k / 2.2k, oe 12.2k / 4.0k / 0.9k, shinka 4.4k / 2.6k / 1.0k. The totals are 274.7M
prompt and 80.3M completion tokens, matching the paper. The raw text is 136 MB of content, 170 MB
of reasoning_content and 113 MB of shinka messages. The prompt blobs are 582 MB, because the user
prompts embed parent and inspiration code. The code blobs are 181 MB.

### 1.4 decision_points: derived workload, not trajectory data

`query_id ∈ {W1.a 10,479, W1.b 6,525, W1.b-fail 668, W1.d 619}`. These are the **benchmark query
instances** from the earlier GEM/EvoTrace work (`gem_evotrace_archive/.../EvoTraceDoc.md` §B).
The normaliser derived them from lineage and fitness:

- W1.a is task-seeded reuse (`seed_uid` = task).
- W1.b is the stuck-state case (`meta.parent_uid`, `delta`).
- W1.b-fail is failure repair (`meta.error_signature`, `repairs` = node uids).
- W1.d is improvement-path reconstruction (`meta.ancestors`).

`meta.parent_uid` agrees with lineage in 17,004 / 17,004 cases. `leakage_suspect` is true on
2,686 rows. They carry **no information that is not already in nodes + lineage**. There is no
record of the candidates the framework considered. Store them, if at all, as a benchmark table,
never as graph edges.

### 1.5 state_events: reconstructable search state

There are 448k `membership` rows from `iterations.jsonl` and 40k `scalar` rows from
`iter_scalars.jsonl`. Membership rows are full snapshots per checkpoint, which makes them highly
redundant.

| backend | roles (rows) |
|---|---|
| evox | population_member 109,583 |
| gepa | elite_pool 3,830, metric_front 5,937, metric_best 1,762, rejection_history 4,600 |
| openevolve | island_member 112,800 (slot = island), archive 107,636, feature_map 83,743 (slot = MAP-Elites cell `"5-0"`), island_best 18,925 |
| shinka | **no membership**; only 3 scalars per run (best, last_iteration, beam_search_parent_id) |

Scalars: `best_program_id` (with `node_uid`), `last_iteration`, `current_island`,
`island_generation[i]`, `last_migration_generation`, `initial_program_id`.

Membership links to nodes: 447,230 rows resolve to a node. 1,586 openevolve `island_best` rows have
a null id. 166 rows list a node before its own iteration, an artefact of checkpoint timing.
Run-length compression to `(session, role, slot, node, from_iter, to_iter)` intervals gives
**103,001 rows** (4.7× smaller).

**Candidates "considered but not selected".** The data does not record them. They can be
*reconstructed*: the pool at the previous checkpoint contains the chosen parent in 2,337/2,337
evox, 3,683/3,811 oe and 1,909/1,985 gepa cases. It contains the context nodes in 99.9%,
99.9% and 95% of cases. The pool size has median 45 for evox, 34 for oe and 27 for gepa.
So "pool(t−1) minus {parent, context}" is a defensible negative set. It cannot be built for
shinka, and the selection policy stays unknown (stochastic and per-island).

### 1.6 Integrity and replay caveats

The caveats come from `integrity_report.json` and `replay_manifest.json`:

- All wall-clock timestamps are null. Order runs only by `iteration`.
- There are 18 tasks, while the paper says 16. Heilbronn and autocorr have variants.
- Nodes are attempts and are never deduplicated. Artifacts are deduplicated by sha256.
- 13 duplicate context ids were rejected.
- 21 runs (all shinka plus 3 others) have `prompt_complete_ratio = 0`, i.e. no prompt history.
- Only 18 runs carry an evaluator. 79 lack `run_config`.
- `artifact_replay_ready` and `same_prompt_replay_ready` are false for all 121 runs.

## 2. Proposed schema

Principles:

- Topology goes in the graph AM. Properties, time-varying state and text are relational.
- `attempt` stays the **single vector-bearing vertex table** that `tjs_open` scans
  (`id = vid` alignment unchanged).
- Graph edges carry only a type id (the edge slot has no payload). Edge properties go in a mirror
  table that is never traversed.

### 2.1 Relational tables

```sql
task       (task text PK, domain text, family text, specification text,
            emb vector(1024))                               -- 18 task_spec vectors, now loaded
session    (session text PK, task text FK, backend text, model text, mode text,
            temperature real, search_algorithm text, run_rel text,
            prompt_complete_ratio real, has_evaluator bool, best_attempt int)
attempt    (id int PK, uid text UNIQUE, vid bigint UNIQUE, task, session FK, backend,  -- new: backend
            program_id text, iteration int, generation int, status, is_valid, fitness,
            judge, exec_time, memory_mib, language, error, artifact_sha text,
            changes_raw text,        -- framework diff summary (nodes.changes)
            changes text,            -- existing LLM-derived summary, unchanged
            intent text,             -- NEW: LLM's own pre-diff rationale / shinka patch_description
            intent_source text,      -- 'response_prose' | 'shinka_patch_description' | NULL
            code text, cell text, parent_id int, tokens int,
            emb vector(1024),        -- node_artifact (existing)
            intent_emb vector(1024)) -- NEW, optional (§2.3)
prompt     (attempt_id int PK FK, template text, system_sha text, user_text text,
            response_text text, prompts_sha256 text, source text)  -- 6,536 blobs + 1,782 shinka from llm log
prompt_system (system_sha text PK, text text)               -- system prompts deduplicated (one per run, roughly)
llm_call   (session text, seq int, PRIMARY KEY (session, seq),
            attempt_id int NULL FK,  -- relational, not a graph edge
            link_method text,        -- 'response_sha1' | 'replace_block' | NULL
            purpose text,            -- 'mutation' | 'search_algo' | 'summary' | 'other' | 'error'
            model, temperature, finish_reason, prompt_tokens, completion_tokens,
            reasoning_tokens, error text, content text, reasoning text, content_sha1 text)
search_membership (session text, role text, slot_key text, attempt_id int NULL,
            from_iter int, to_iter int, value jsonb)        -- 103,001 run-length intervals
search_scalar (session text, iteration int, key text, value jsonb, attempt_id int NULL)  -- ~40k
edge_prop  (src int, dst int, edge_type int, label text, ordinal int, provenance text,
            PRIMARY KEY (src, dst, edge_type))  -- audit/property mirror, NEVER traversed
bench_decision_point (dp_id PK, query_id, ...)  -- optional, only if the W1 harness needs it
```

Indexes:

- `attempt(task, judge)`, `attempt(session, iteration)`, HNSW on `emb`, and HNSW on `intent_emb`
  if loaded.
- `llm_call(attempt_id)`.
- `search_membership(session, from_iter, to_iter)`. An AS-OF lookup is
  `WHERE session=$1 AND $t BETWEEN from_iter AND to_iter`.

### 2.2 Native graph edges versus relational foreign keys

| relation | where | why |
|---|---|---|
| parent → child | graph `lineage` + `lineage_up` (existing) | traversal (ancestry, subtree) |
| context → consumer | graph **`context`** + **`context_up`** (new, 30,086 each) | traversal ("what shaped X", "where was elite E reused"). Kept separate from lineage: the two never overlap |
| gepa merge second parent (92) | graph `context`, plus `edge_prop.label='Merge Parent B'` | alternative: a `lineage2` type. See open question 2 |
| cross-session kNN | graph `similar` + `similar_up` (existing, **derived**) | the only cross-session bridge. Context edges give 0 |
| llm_call → attempt | relational `llm_call.attempt_id` | a 1:n attribute lookup, no path semantics |
| prompt → attempt | relational 1:1 PK | text payload |
| attempt → session → task | relational columns | pushdown filters for `tjs_open` (no hierarchy vertices needed) |
| pool membership over time | relational intervals | time-varying set membership, not node-to-node topology |
| "considered but not selected" | **not materialised**. A view joins the `search_membership` AS-OF previous checkpoint | it is a reconstruction. Materialising it as an edge type would add about 350k edges of inferred data |

### 2.3 Vectors

- Keep `attempt.emb` (node_artifact, 10,672).
- Load `task.emb` (18, already in the npz).
- **New, optional:** `attempt.intent_emb`, made by the same Qwen3-Embedding-0.6B model from
  `intent`. About 8.3k non-empty rows; a re-embed takes minutes on CPU.
  Retrieving on "what the agent meant to do" is the natural seed for "find past attempts that
  tried this idea". Prompts and llm_calls get no vectors: they are long, and the code inside them
  is already covered by `emb`.

### 2.4 Size estimate

Raw sizes, before TOAST (pglz/lz4 typically compresses 2.5–4× on code and prose):

| object | rows | raw |
|---|---:|---:|
| attempt (excluding code) + emb | 10,672 | about 45 MB + HNSW about 60 MB |
| attempt.code | 10,672 | about 180 MB |
| attempt.intent + intent_emb | 10,672 | 8 MB + 35 MB + HNSW |
| prompt.user_text + response_text | 8,318 | about 580 MB (oe/evox/gepa blobs) + about 110 MB (shinka) |
| llm_call content + reasoning | 18,400 | about 306 MB |
| search_membership / search_scalar | 103k / 40k | about 15 MB |
| graph edges (lineage, context, similar with k=16, each with an `_up` copy) | about 423k slots | about 14 MB at 32 B/slot |

The total on disk is roughly 0.5–0.8 GB with text, or about 120 MB without the `prompt` and
`llm_call` text. Add a `--no-text` loader flag that loads llm_call/prompt metadata and hashes only.

## 3. Loader changes (`trimem/data/evotrace.py`)

1. **Target a new cluster.** Refuse ports 55440 and 55441, as `tests/test_trimem_optimizer_native.py`
   already does. Suggested target: a fresh PG16 `initdb` at `.tridb-pgdata-traj`, port 55450,
   database `trimem_traj`. 55442–55446 are taken by the exg runs. (Correction, see Addendum A:
   `data/evotrace` is a symlink to `dataset/evotrace`, so the old `SRC` was not stale. The new
   loader anchors its default paths at the repository root so it works from any cwd.)
2. Add `task` and `session` loads, with `task.emb` from the npz rows where `kind=='task'`.
3. Add the new `attempt` columns: `backend`, `program_id`, `generation`, `artifact_sha`, and
   `changes_raw` (from `nodes.changes`), plus `intent` and `intent_source`.
4. **Response resolver.** For each node, read the prompt blob from the raw run dir (`sessions.run_rel`
   + `prompts.blob_rel`), or for shinka read `programs.jsonl` `metadata.llm_result` and
   `patch_description`. Extract `intent` from the text before the first `<<<<<<< SEARCH`, from
   `<DESCRIPTION>`, or from the text before the first code fence.
5. **llm_call load.** Stream the raw logs. Compute `content_sha1` and join it to the node response
   hashes per session to set `attempt_id` and `link_method='response_sha1'`. Optionally run the
   `replace_block` pass for gepa, flagged as such. Classify `purpose` with simple rules: error,
   SEARCH/REPLACE or `<NAME>` means mutation, evox non-diff means search_algo, a shinka system-prompt
   prefix means summary.
6. **Context edges.** Add `register_edge_type('context')` and `('context_up')`, using the same
   COPY → temp table → `gph_insert_edge` pattern as lineage. Fill `edge_prop` for the gepa merge
   label and the ordinal. Also emit `lineage_up` and `similar_up` here so the loader is the single
   source of truth.
7. **State.** Stream `state_events.jsonl` line by line. Run-length encode membership per
   `(session, role, slot_key, node)` into `search_membership`, and copy scalars into `search_scalar`.
8. Make the load idempotent: use `edge_prop` as the dedup guard before `gph_insert_edge`, then run
   `ANALYZE` and `trimem.analyze_graph()`. Write a load receipt with per-file counts that match §1.

## 4. Open questions for the user

1. Should the heuristic `replace_block` links (943 gepa calls) be loaded, or should only the 8,303
   exact links be loaded?
2. Should gepa merge "Parent B" (92 edges) count as `context`, or as a second `lineage` type? This
   changes ancestor queries.
3. Should prompt and response full text go in the DB (about 0.7 GB raw), or stay as blob pointers
   with only `intent` in the DB?
4. Is `intent_emb` wanted now? It needs the embedding server on 8011 or a CPU re-embed.
5. Should the reconstructed "not selected" pool become a materialised view, or remain an ad-hoc
   query?
6. Should the derived `changes` (the current LLM summary) be kept alongside the new ground-truth
   `intent`, or retired?

## Addendum A (2026-10-13): decisions taken and implementation

The user approved the plan with the decisions below. The implementation lives in:

- `trimem/data/evotrace.py`: schema, relational core, llm_call, state, receipt, CLI.
- `trimem/data/evotrace_text.py`: intent extraction, prompt resolver, sha1 pairing, membership
  run-length encoding.
- `trimem/data/evotrace_graph.py`: native edges.
- `trimem/data/embed_intent.py`: the intent vectors.
- `scripts/traj_cluster.sh`: the new cluster.
- `tests/test_evotrace_loader.py`: tests.

| question (§4) | decision |
|---|---|
| 1. heuristic `replace_block` links | **not loaded**. Only exact `response_sha1` links |
| 2. gepa Merge Parent B | its own native type **`merge_parent` / `merge_parent_up`** |
| 3. full prompt and response text in DB | **yes**: `prompt`, `prompt_system`, `llm_call.content/reasoning` |
| 4. `intent_emb` | **yes**, from the 8011 Qwen3-Embedding-0.6B server |
| 5. "not selected" | **per query**: `not_selected(attempt_id)` SQL function, not materialised |
| 6. LLM-derived `changes` | **dropped**. The dataset's own `nodes.changes` is kept as `framework_changes` |

Details that the plan left open:

- **Edge counts.** Merge Parent B edges are a subset of `context_edges.jsonl`, so the loader
  *partitions* that file. The result is `context` = 30,086 − merge and `merge_parent` = merge
  (expected 92). `lineage` stays 10,479: Parent A is the ordinary lineage parent. The receipt
  checks `context + merge_parent == 30,086`.
- **Edge direction and adjacency order.** Every forward type has an `_up` reverse.
  - lineage, merge_parent and context: forward is source → child or consumer, and children are
    ranked by (iteration, id).
  - `context_up`: consumer → its inspirations, in the raw `other_context_ids` order.
  - `similar`: kNN rank, descending cosine. `similar_up` is ranked by descending cosine as well.
  - Edges are inserted in (src, type, ordinal) order, so `trimem.expand(v, t, fanout)` returns
    the top-ranked neighbours first.
- **Idempotency.** `edge_prop` (src, dst, edge_type) is the guard. A native edge is inserted
  only when its mirror row is new, in the same transaction. Vertices are reused through
  `gph_vid_map`. Relational tables are truncated and reloaded on every run. `load_meta` stores
  the sha256 of `nodes.jsonl`, and a database that already holds edges from a different dataset
  is refused.
- **Duplicate responses.** When several calls and/or nodes in a session share the same response
  text, they are paired in order (k-th call ↔ k-th node by iteration) with
  `link_method='response_sha1_dup'`. Surplus calls stay unlinked; this is still exact, not
  heuristic.
- **Intent.**
  - For oe, evox and gepa: prose before the first SEARCH block, else the `<DESCRIPTION>`
    body, else prose before the first code fence.
  - For shinka: `metadata.patch_description`. This is NULL for the `init` root and its island
    copies, whose text is "Initial program setup".
  - The shinka prompt text comes from `metadata.llm_result`. The prompt's
    `response_text` is the full LLM reply.
- **`llm_call.purpose`** is rule-based:
  - `error` and `empty`;
  - `mutation`: linked to a node;
  - `mutation_unretained`: a diff or `<NAME>` reply whose program was not kept, e.g. gepa
    proposals outside the 20-slot window;
  - `search_algo`: evox;
  - `summary` and `recommendation`: shinka, by system prompt;
  - `other`.
- **Membership intervals** are contiguous on the session's *checkpoint* sequence, not on
  iteration+1. A value change, such as a gepa `metric_best` score, starts a new interval.
- **Port guard.** The loader refuses 5432, 55440 and 55441, whether from the DSN or from
  `PGPORT`. After connecting it checks the server's own `port` and `data_directory`, and refuses
  `.tridb-pgdata-trimem`.
- **Not carried over.** `cell` exists but is NULL: the replay isolation column is set by replay
  runs, not by the corpus. `tokens` is filled with the Qwen3.8 tokenizer unless `--no-tokens`
  is given. The replay code in `trimem/replay/` reads `attempt.changes`, so it keeps using the
  production database and is not pointed at `trimem_traj`.

## Addendum B — loaded three-edge corpus and query migration (2026-10-13)

This addendum supersedes the earlier separate `merge_parent` choice for this new load.
User-selected logical types are `lineage`, `context`, and `similar`. Merge Parent B is a
context edge with its raw label retained. All three have native `_up` inverse copies.
`--merge-as-context` selects this policy, and `load_meta.graph_policy` prevents reuse of a
nonempty native graph under another policy. Existing loader callers retain the older default.

The loaded database is `host=/tmp/trimem-traj port=55450 dbname=trimem_traj`. It contains 18
tasks, 121 sessions, 10,672 attempts, 8,318 prompt rows and 18,400 LLM calls. Forward adjacency
counts are lineage 10,479, context 30,086, similar 170,752. Similar edges are same-task,
cross-session code cosine top-16 with no fixed threshold. All 10,672 code vectors and 18 task
vectors are retained. All 5,989 extracted intents have an intent vector.

The existing HTTP embedder was unavailable during import. Completion used the exact local
Qwen3-Embedding-0.6B checkpoint `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, last-token pooling
and L2 normalization in bf16. Eleven intents exceeded the 6000-character embedding input cap.
Full text is retained, and the embedding receipt records this truncation and backend.

Repeated identical responses cannot prove which call generated which attempt. Such rows keep
`attempt_id=NULL`, `link_method=response_sha1_ambiguous`, and `link_candidates` with matching
attempt IDs. Only unique call/response matches are asserted as exact links.

`trimem/trajectory_query.py` is the new-schema query entrypoint. Semantic scoring evidence
uses `intent + intent_source + framework_changes + child/parent fitness` plus backend/language.
Intent fallback is explicit. Code is available for relational edit checks and downstream
memory, but not automatically sent to the judge. `SemJoin/lowered` is bounded relational
pairing followed by SemFilter, not an implemented native SemJoin operator. SemScore remains
pointwise LLM scoring rather than global ordering. Dedicated reranker models are not connected.

The frozen optimizer path is unchanged. Switching experiments requires an explicit snapshot
and compiler/formatter version update. Full-corpus ingestion does not establish online memory
availability or eliminate leakage. Cross-session iteration values are not comparable timestamps.

Artifacts: `triMEM_optimizer/trajectory_storage_v020_20261013/`. The English it20 report embeds
the new explorer while retaining its historical experiment results. It has tabs for plans,
three-edge graphs, every application relation, and recorded/new semantic prompt examples.
The source generation prompts are real historical inputs. New judge prompts are previews.

## Addendum C — current-agent-aware semantic inputs (2026-10-13)

New-schema SemFilter/SemScore input v2 adds the actual problem, current objective, current
program state and observed failure to candidate evidence. `CurrentAgentContext` validates the
required fields and `bind_context` binds a context-specific digest and predicate instructions.
All semantic placements use the same current context for controlled comparisons. Context is
supplied by the caller, not inferred from candidate intent or outcomes. The frozen old optimizer
has not been switched to this entrypoint.

The native semantic queue defaults to 32 KiB for predicate plus input. Full ALE user prompts
can exceed that bound, so the context view is capped at 12,000 UTF-8 bytes with explicit head/tail
omission markers, source lengths and hashes. This local formatting uses no LLM. Full originals
remain in the database and context object. Token limits must additionally be checked by the live
runner. Historical HTML examples reconstruct the pre-attempt parent state and label unknown
failure diagnoses explicitly. They are prompt previews, not executed semantic judgments.


## Addendum D — Full-code embeddings and live OpenEvolve state (2026-10-13)

The new `qwen3-full-code-v1` contract embeds raw complete code with pinned Qwen3-Embedding-0.6B
revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, last-token pooling and L2 normalization.
Both the offline corpus and live query use `trimem/code_embedding.py`. No 2000-character slice
is applied in this path. Token overflow raises an error. This changes both prefix coverage and
the old node-artifact metadata wrapper, so it is a new representation version, not an isolated
truncation ablation. Legacy experiment paths remain available for reproducibility.

`trimem.data.rebuild_full_code` stages per-record vectors, validates source hashes, backs up old
vectors/similarity properties, and transactionally updates vectors and native similarity adjacency.
Lineage/context stay intact. Similarity is top-16 cosine within task across sessions, with reverse
adjacency, and has no fixed score threshold. Maintenance may materialize offline edge batches.
Online query execution remains on native graph adjacency with cursor early termination.

`trajectory-context-v3-full` supersedes v2's excerpted current state. It preserves the entire problem,
parent code, actual feedback, fitness, objective and configured remaining generation budget.
Inputs over the explicit context limit fail rather than silently dropping code. The isolated
cluster mailbox limit is raised via `scripts/traj_cluster.sh`; no legacy cluster restart is needed.

OpenEvolve opts into this adapter with `--trajectory-store --arm trimem --task-source trimem
--trimem-dsn 'host=/tmp/trimem-traj port=55450 dbname=trimem_traj' --inject-as changes`.
`--trajectory-plan` selects VGR/P1/P2/SemRankOnly/P1+Rank/P2+Rank.
`--trajectory-memory-tokens` defaults to 8192. `--trajectory-total-token-budget` is optional and
accounts for generation usage only. `--trajectory-objective` can override the feedback-based goal.

The new schema has no universal summary column. Query rows combine candidate attempt,
historical parent fitness and session metadata. Intent may be historical LLM prose and
framework_changes is recorded framework text. `evidence_card()` deterministically renders those
fields plus the pointwise semantic score. It does not call a summarizer. Optional historical code
is included whole only when the memory budget permits. The current parent remains complete.
OpenEvolve registration updates cards on repeat retrieval so a new semantic score is not lost.

`tools/trajectory_prompt_example.py` runs retrieval and the real OpenEvolve renderer over a recorded
parent state and exports messages, evidence cards and receipts. It does not execute agent generation
or establish a quality improvement. Semantic operator calls are distinct from summary generation.

### D.1 — Summary-only inspiration rendering (2026-10-13)

This supersedes the metadata-rich card rendering described above. Agent-facing inspirations now
contain only stored intent prose, falling back to framework_changes when intent is absent.
Records with neither description are omitted. IDs, status, historical scores and semantic scores
stay in sidecar provenance rather than the inspiration text. No summarization model is called.
Historical intent describes the author's stated change and is not a newly verified code summary.
Optional complete historical code remains controlled by `--trajectory-include-code`.

The updated iteration-6 example reuses the same three retrieved records and current parent.
Its inspiration section contains 370 generation-tokenizer tokens, and its system/user message
bodies contain 2854 tokens excluding the chat template. The previous rendering and retrieval
receipt are preserved under `updated_prompt/metadata_card_previous/`.

### D.2 — Task-native systems, own-run history and complete inspirations (2026-10-13)

This supersedes D.1's summary-only rendering. Each retrieved inspiration now includes its complete
program code and stored intent (framework_changes fallback). Missing intent is explicitly marked.
The default target is four inspirations within the existing 8192-token section budget. No code is
truncated. Oversize records are skipped whole and counted. Provenance remains outside the prompt.

The trajectory-only worker wrapper uses the original task system message verbatim. It displays
the latest three completed, retained own-run attempts, with changes, metrics and combined_score
outcome against the recorded parent. It selects one highest-scoring alternative own-run program,
excluding external memories, future records, explicit failures and the current parent's ID/code.
The top-program section has a separate 8192-token budget and logs whole-record omissions.
Current-run history can span islands. Empty history or missing alternatives are not fabricated.

`tools/trajectory_prompt_catalog.py` exports all 18 original-system task templates, historical
prompt examples and Math/ALE context-size distributions. The reusable USER TEMPLATE reflects
the requested design, while `original_example.txt` preserves the original backend's message.

### D.3 — Evaluation feedback and candidate-code semantic evidence (2026-10-13)

Current parent metrics, recorded diagnostics and artifacts now render under Current Evaluation
Feedback. Each Previous Attempt has its own Evaluation Feedback subsection. The trajectory
snapshot includes selected-parent and recent-attempt artifacts even past the default 100-program
artifact snapshot cap. Rendering adds no diagnostic truncation or summarization call. Missing
feedback stays explicit, and already-shortened evaluator output cannot be recovered.

Semantic evidence version trajectory-context-v4-candidate-code includes complete candidate code
for SemFilter and shared SemScore input. The predicate asks whether the attempt contains useful
information for the current objective/failure, based on the implementation and recorded evidence.
Intent and framework-change excerpts retain their existing 2400/1200-character limits. Native
request-size enforcement rejects oversize requests rather than truncating code. Scoring remains
pointwise and does not reorder final records.

### D.4 — Complete prompt budget and Codex Math experiment (2026-10-13)

This supersedes D.2's 8192-token inspiration limit. Retrieved inspirations are capped at four
complete programs and 16384 tokens in total. The complete generation input is capped at 32768
tokens, including the local Qwen chat template and assistant generation prefix. Generation output
has a separate cap. Top-program context retains its independent 8192-token budget.

Whole reference records are omitted on overflow, first inspirations, then top programs, then the
oldest previous attempts. Mandatory current code, original task system and current feedback are
preserved. If these alone cannot fit, assembly fails explicitly. Receipts distinguish original and
final complete input counts, section budgets, omitted records and rejected prompts. No retained
program or diagnostic is shortened by this step.

The authorized execution plan is `triMEM_optimizer/handoffs/math_trajectory_v4_codex_20261013.md`.
It compares three Math tasks and four current plans over 20 rounds, reporting checkpoints at 10
and 20. The corpus is fixed across cells while each cell maintains independent OpenEvolve history.
SemRankOnly remains score-only, with no score-based reordering. Runtime results and background
status must be read from the new experiment outputs rather than inferred from this plan.
