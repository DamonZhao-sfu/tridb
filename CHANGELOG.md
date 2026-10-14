# Changelog

## [0.13.8] - 2026-10-13
### Features
- Added circle packing to the four-task comparison, a fixed historical-best dashed reference,
  method legends with visibility toggles, and observed-only curves for checkpoints 10/20/30/40/50.
- Added a staged coordinator: finish all sixteen task-plan cells at 20, then resume each saved
  checkpoint into a separate directory for iterations 21–50, saving checkpoints every ten rounds.
- Designated experiment-trajectory-retrieval-v4 as the default code branch in tracked experiment
  defaults, with branch/commit/source hashes recorded in launch manifests.
### Design Rationale
- Preserve completed first-stage measurements and compare continuation quality without rerunning
  iterations 1–20. Historical corpus scores provide a stable reference rather than a moving target.
### Notes & Caveats
- Pending checkpoints remain unavailable until observed. Historical best is not an upper bound.
- Runtime data, local database directories and model/corpus files remain external experiment inputs.

## [0.13.7] - 2026-10-13
### Features
- Added task-priority parallel scheduling for the Math trajectory matrix, with live concurrency
  control, per-cell status, completed-cell reuse, and CPU/GPU embedding placement.
- Dedicated semantic worker concurrency is configurable. Parallel semantic usage with ambiguous
  context ownership is marked unavailable instead of charged repeatedly to multiple cells.
### Design Rationale
- Fill both existing generation-service GPU replicas while prioritizing within-task plan comparisons.
  Experiment semantics, prompt budgets, seeds and iteration counts remain fixed.
### Notes & Caveats
- The first convex13/VGR cell ran serially. Later parallel latency measurements include contention.
- The user stopped the optimizer experiment and retained its archive before this resource change.

## [0.13.6] - 2026-10-13
### Features
- Trajectory generation input budget is 32768 tokens including the local Qwen chat template.
  Retrieved inspiration context has a 16384-token section budget and a hard maximum of four programs.
- Budget accounting records original/final input tokens and whole-record removals. Mandatory current
  code and feedback are never clipped. An oversized mandatory prompt is rejected explicitly.
- Added the Codex execution plan for three Math tasks, four retrieval plans and 20 iterations,
  with checkpoints at 10/20 and isolated new-store experiment outputs.
### Design Rationale
- Complete source programs and evaluation feedback need more room than summary-only context.
  Separate input, reference-section and generation-output limits keep accounting unambiguous.
### Notes & Caveats
- The top-program section retains its independent 8192-token limit. The 32768 input cap excludes output.
- SemRankOnly still annotates scores without reranking. Experiment reports must state that limitation.
- Existing legacy experiments and their budgets remain unchanged.

## [0.13.5] - 2026-10-13
### Features
- Added Current Evaluation Feedback and per-attempt Evaluation Feedback to trajectory prompts.
  These render recorded metrics, diagnostics and artifacts without additional clipping or LLM summarization.
- The trajectory snapshot explicitly loads selected-parent and recent-attempt artifacts beyond the
  default OpenEvolve snapshot cap. Missing feedback remains explicit.
- SemFilter now judges contextual usefulness using complete candidate code. Shared SemScore evidence
  also includes complete code. The semantic prompt version is trajectory-context-v4-candidate-code.
### Design Rationale
- The agent needs its own observed failures in generation context, and semantic selection needs the
  candidate implementation rather than relying only on historical intent claims.
### Notes & Caveats
- Source evaluators may already have shortened diagnostics. This change cannot reconstruct lost text.
- Native semantic request limits reject oversize input instead of clipping candidate code.
- Semantic scoring remains pointwise annotation; no ranking/sorting behavior changes in this release.

## [0.13.4] - 2026-10-13
### Features
- Trajectory-store prompts preserve each task's original system message and show complete
  retrieved code plus stored intent. Default target: four inspirations and one own-run top program.
- Added actual recent-attempt history, with Changes/Metrics/Outcome based on combined_score
  versus the recorded parent. History excludes external memories and future iterations.
- Exported original-system templates for all 18 Math/ALE tasks, raw prompt examples, provenance,
  system variants and measured historical reference-program counts.
### Design Rationale
- Recent attempts, own-run best programs and retrieved experiences serve different purposes.
  Keep all three explicit. Template assembly makes no summarization model call.
### Notes & Caveats
- Opt-in trajectory-store worker wrapper only. Installed OpenEvolve and legacy runs are unchanged.
- Inspiration and top-program sections each have an 8192-token limit. Oversize programs are skipped
  whole and counted, never clipped. Thus the returned count can be below the configured target.
- Top programs are selected from retained, evaluated current-run records across islands, excluding
  explicitly invalid records and the current parent's ID/code. A missing alternative is shown honestly.

## [0.13.3] - 2026-10-13
### Features
- Added a versioned full-code Qwen3 embedding contract and resumable isolated-corpus rebuild.
  Corpus/query inputs share the same renderer, checkpoint and pooling. Oversize code is rejected,
  never silently cut. Similarity adjacency is rebuilt as same-task, cross-session cosine top-16.
- Added an opt-in OpenEvolve trajectory-store adapter with full task/parent code, actual evaluator
  feedback, objective and remaining generation budget on every retrieval. Full-context semantic
  requests replace v2 excerpts. Context files and hashes are recorded per round.
- Added deterministic inspiration evidence cards with provenance, intent, framework changes,
  fitness gain, validity/error and optional complete historical code. No post-retrieval summary
  LLM is called. Whole cards exceeding the 8192-token memory section budget are skipped and counted.
### Design Rationale
- Code suffixes and current failures must affect retrieval. A 2000-character prefix omitted code
  in 95.5% of stored Math/ALE attempts. Code vectors and derived graph edges must migrate together.
- Existing summaries/intent are evidence from history, not instructions or newly generated summaries.
### Notes & Caveats
- New-schema path is explicitly enabled with --trajectory-store and the rebuilt DB on port 55450.
  Active legacy ALE and frozen BO experiment paths retain their original embedding contracts.
- Semantic scores annotate a streaming candidate sequence. This is not a global sort by score.
- Remaining-token accounting currently covers generation calls reported in that run's telemetry,
  not semantic retrieval calls. It remains null when no total budget is configured.
- Native semantic mailbox capacity is 524288 bytes in the isolated cluster. Full current state
  is capped at 393216 bytes with explicit rejection on overflow, not truncation.

## [0.13.2] - 2026-10-13
### Features
- New-schema semantic retrieval now binds a `CurrentAgentContext` containing the actual
  problem, immediate objective, program state, failure evidence, constraints and provenance.
  Seed filters, endpoint filters, pointwise scores and lowered semantic joins receive it.
- Added `trajectory-context-v2` input version and context digest. Bounded UTF-8 context views
  report field omissions and source hashes. Unknown historical failures remain explicit.
- Updated English prompt previews and added measured graph-degree statistics, traversal
  directions and proposed seed/fanout/hop profiles to the explorer.
### Design Rationale
- Task identity alone cannot establish usefulness for the current agent state. Current-state
  evidence is supplied independently of the retrieved candidate to avoid conflating the two.
- Contexts are capped at 12,000 bytes, with request-size validation against the native
  semantic input limit. This is separate from the final memory token budget.
### Notes & Caveats
- Historical contexts are reconstructed for inspection, not claimed to be recorded semantic
  calls or leakage-free retrieval examples. Live callers must supply current state per request.
- Active frozen BO runs and their old memory formatter remain unchanged. The new query path
  does not automatically migrate their runtime. Proposed graph budgets are not quality optima.

## [0.13.1] - 2026-10-13
### Features
- Loaded the complete EvoTrace corpus into the isolated `trimem_traj` database on port 55450.
- Added optional `--merge-as-context` graph policy: lineage, context and similar, each with
  reverse adjacency. Merge Parent B remains a context edge with its original label.
- Preserved ambiguous response-to-attempt links as `llm_call.link_candidates` without asserting
  an exact attempt ID. Added graph-policy reload guards.
- Added `trimem/trajectory_query.py` for the new schema. Semantic score/filter evidence now
  includes intent provenance, framework changes and parent/child fitness instead of the old
  `changes` field. Native SemJoin is not claimed: a bounded join plus SemFilter expresses it.
- Added offline full-table, native-graph and prompt explorers in
  `triMEM_optimizer/trajectory_storage_v020_20261013/`, embedded in the English it20 report.
### Design Rationale
- Preserve active optimizer experiment hashes and snapshots. New-schema query compilation is
  a separate entrypoint, with explicit SQL/EXPLAIN artifacts and pointwise LLM score semantics.
- Native topology is exported directly and checked against the property mirror. All 11
  application tables retain every row and field in paginated files and compressed JSONL.
### Notes & Caveats
- This is a full offline corpus, not a leakage-free online evaluation snapshot.
- Intent embeddings use the pinned Qwen3-Embedding-0.6B checkpoint. Eleven long inputs use
  the documented 6000-character limit. Full intent text is retained in the relation.
- Existing BO runs retain their old scorer input. No new downstream quality claim is made.

## [0.13.0] - 2026-10-13
### Features
- TriMEM trajectory store v0.2 (`docs/trimem_trajectory_storage_v0.2.0.md`): the EvoTrace loader
  now loads every relation in the dataset, into a NEW cluster (`scripts/traj_cluster.sh`:
  `.tridb-pgdata-traj`, port 55450, database `trimem_traj`). Relational tables:
  - `task`, with 18 task_spec vectors;
  - `session`;
  - `attempt`, gaining `framework_changes`, `intent`, `intent_source`, `intent_emb`,
    `backend`, `program_id`, `generation` and `artifact_sha`;
  - `prompt` + `prompt_system`, holding full prompt and response text;
  - `llm_call`, with full content and reasoning, `attempt_id` exact links and a `purpose`;
  - `search_membership` (run-length intervals), `search_scalar` and `search_checkpoint`;
  - `edge_prop`, a property mirror and idempotency guard.
- New native edge types `context` / `context_up` (inspirations shown to the LLM) and
  `merge_parent` / `merge_parent_up` (gepa Merge Parent B). `lineage`, `similar` and their
  reverses are kept.
- `not_selected(attempt_id)` reconstructs the pool at the previous checkpoint, minus the parent
  and the context sources, which it reads from the native graph.
- `trimem/data/embed_intent.py` embeds the LLM's stated change intent through the 8011 server.
### Design Rationale
- llm_call → attempt is an exact sha1 match of the response text within a session, not a
  heuristic. Rows that need a heuristic link are left unlinked. Topology is native; properties,
  text and time are relational.
- The LLM-derived `changes` is dropped in this store in favour of the agent's own intent.
### Notes & Caveats
- The loader refuses ports 5432, 55440 and 55441 and the `.tridb-pgdata-trimem` data
  directory. Production `trimem_evo` is untouched.
- The replay code still reads `attempt.changes`, so it stays on 55441.

## [0.12.0] - 2026-10-12
### Features
- TriMEM SemExtract (`trimem_pg` 0.3.0, ADR-0028): `trimem.sem_extract(question, input) -> text`
  answers a question about a text with a short lower-case label, planned as
  `Custom Scan (TriMEM SemExtract)` like SemScore. Labels feed ordinary relational operators:
  WHERE label = ... (lifted into the node's own qual), GROUP BY / HAVING, per-label caps with
  window functions, comparison in an outer query.
- Placement enumeration covers SemFilter × SemScore × SemExtract (BASE on the extract relation vs
  TOP above the join, `trimem.force_placement`), cache-coverage-aware cost, label-qual
  selectivity from the cached label distribution; EXPLAIN shows Placement, Placement Decision,
  Inputs Extracted, Cache Hits, Worker Requests, Unlabeled, LLM Wait.
- Worker / LLM: `OP_EXTRACT`, `EXTRACT_PROMPT` + strict `parse_label`, `extract_many` on
  `OpenAIJudge` and `MockJudge` (deterministic `extract_rule`).
- Upgrade script `trimem_pg--0.2.0--0.3.0.sql`: `sem_cache.extracted text`, `sem_extract`,
  `sem_extract_support`, `complete_extracts(slots, seqs)` (REVOKEd from PUBLIC).
### Design Rationale
- The mailbox stays numeric: the worker commits the label to `sem_cache` and only signals
  completion; the node re-reads its keys with the latest snapshot (the worker commits after the
  query's snapshot). One LLM call per distinct input replaces one `sem_filter` per category.
### Notes & Caveats
- `sem_extract` inside a join condition is rejected (v1); compare labels of two relations in an
  outer query over a MATERIALIZED CTE. Label quals are never placed above an outer join.
- Unparsable replies return NULL and are not cached, so they are asked again on the next query.
- Production needs the new `.so` installed and a restart (shared_preload_libraries), then
  `ALTER EXTENSION trimem_pg UPDATE TO '0.3.0'` in each database.

## [0.11.0] - 2026-10-12
### Features
- Objective O4 (v3 space): ĝ/ĝ_ref − 0.1·L̂/L_ref with L̂ = expected LLM inference seconds per valid
  child, from per-candidate output-token and validity models and a fitted latency law
  (`trimem/objective/costs.py` joins OpenEvolve telemetry to iterations by prompt hash).
- Mixed traversal path (similar → lineage, two chained TriMEM Expand nodes) in the graph space
  (27 configurations, 325 candidates); online models scaled to Var(Δ_best).
- Workbook `triMEM_optimizer/trimem_experiments.xlsx` holds every plan (fixed, random, A1–A6,
  B0–B3, C0–C3); the summary page hides the random and v1 arms by default (`--hide`).
### Design Rationale
- Input tokens cost almost no time (0.00018 s vs 0.029 s per output token, R² 0.85); penalising
  them made the optimizers inject too little (docs/trimem_downstream_objective_plan_v0.4.0.md).
### Notes & Caveats
- Output-token prediction is weak (LOTO Spearman 0.04–0.59); latency still dominated by the
  model's output length, which context barely changes.

## [0.10.0] - 2026-10-12
### Features
- Downstream-aware plan selection for TriMEM in OpenEvolve (`trimem/objective/`,
  `bench/agent_memory/gem_oe/trimem_adaptive.py`, `--arm trimem-adaptive`): per-iteration choice of
  plan family × k (v1) and, in v2, × graph traversal configuration (edge lineage/similar,
  direction down/up via native reverse edge types, 1/2 chained Expand hops, fanout 4/8/16; 289
  candidates). Every decision and its child outcome is logged in `trimem_outcome` with propensity.
- Estimators: hurdle model (calibrated logistic × gradient boosting) as a leave-one-task-out offline
  prior; Bayesian linear online model (cold / warm start, Thompson sampling); v2 trains a separate
  "inject nothing" estimator. Objectives O1 (tokens per expected gain), O2 (utility − λ·tokens),
  O3 (cheapest candidate predicted at least as good as rank:5).
- OpenEvolve cells can be resumed from a checkpoint into a new directory (`--resume-from`,
  `oe_matrix --resume-dir`), restoring the adaptive policy state (`policy.pkl`).
- Summary page `triMEM_optimizer/it20_summary/` (fixed plans to 50 iterations, optimizer section).
### Design Rationale
- v2 targets improvement over the run's best (the reported quality) instead of over the sampled
  parent; v1 arms saved 25–38% tokens but chose "inject nothing" 21–40% of the time and lost
  1–7% quality (docs/trimem_downstream_objective_plan_v0.3.0.md).
### Notes & Caveats
- OpenEvolve's evolution trace drops ~3.5% of children (parent evicted before the child returns);
  the analysis now merges the checkpoint database. The v1 offline priors were trained before this
  fix. Single seed (42) so far.

## [0.9.0] - 2026-10-11
### Features
- TriMEM (`src/trimem_pg`, `trimem/`): LLM semantic predicates as a planner CustomScan
  (`TriMEM SemFilter`) and a costed native graph expansion node (`TriMEM Expand`) in one PG plan;
  cost-based SemFilter placement over joins (every base/top assignment is a full join search);
  shared-memory mailbox to an out-of-process LLM worker; cache-coverage-aware cost model.
- trimem_pg 0.2.0 soft semantic operators: `sem_filter(predicate, input, unknown_as)` (keep or
  drop rows whose input is NULL) and `sem_score(predicate, input)` in [0, 1], planned as
  `TriMEM SemScore` (batched, cached) so an LLM relevance score can weight an ORDER BY instead of
  filtering. Replay arms A5 (keep unknown) and A6 (score-weighted rank).
- EvoTrace loader (attempt table, HNSW, lineage + cross-session similarity graphs) and the
  plan-choice and end-to-end agent replay experiments (`trimem/experiments`, `trimem/replay`).
### Design Rationale
- ADR-0027: the LLM stays out of process, but the semantic step is a real plan node, so the
  optimizer can cost and place it against vector, graph and relational operators.
### Notes & Caveats
- Requires `shared_preload_libraries = 'tjs_pg,trimem_pg'` (order matters). Stock PG16 only; the
  PG13.4 fork / GX10 build is unbuilt here. The objective does not yet include downstream agent
  cost (P5); the replay shows that term is material.

## [0.8.0] - 2026-10-10

### Features
- **Figure 1: semantic operators under one plan** (branch `feat/figure1-semantic-ops`, ADR-0026).
  `bench/agent_memory/semantic/`: LLM result cache and pending set as PG heap tables,
  sem_extract/filter/match/conflict/rank/topk/summarize with batching and offline replay,
  predicate placement via the new `tridb_place_sem` (`src/planner/join_order_sem.c`,
  join_order 0.2.0, bit-identical Python spec), streaming θsem over a `tjs_vsgt` portal.
- GEM: deferred ingest (`gem/ingest_plan.py`), strong/stale reads, ACL predicate, token
  budget, cache-backed strategy client, D10 graph-AM merge fix; P5 downstream estimator.
- EvoTrace ingest/retrieve replay experiment (`bench/agent_memory/fig1/`), write-up in
  `docs/figure1_evotrace_experiment_v0.1.0.md`.

### Fixes
- **VSGT use-after-free on updated rows** (`src/tjs_pg/vsgt_exec.c`): index AM calls ran in
  the per-row scratch context, so btree's lazily allocated `killedItems` was freed before
  `index_endscan` (error or SIGSEGV). Regression test `test/vsgt_dead_tuple_test.sql`.

### Design Rationale
- LLM stays out of process (no HTTP in the backend; TR-1 and byte-parity gates), but its
  verdicts and the pending set live in the same database and WAL; the planner sees the
  semantic leg through an additive decision function, never by changing the frozen core.
- Eager ≡ deferred requires the cache: with a real LLM two identical runs disagree on half
  the extracted lessons; with a shared cache every plan materializes the same state.

### Notes & Caveats
- Planner C built and tested on stock PG16 only; PG13.4 fork / GX10 unbuilt here.
- P−1: `make test` (2039 passed, 0 failed) and `make lint` (clean) now pass. Result trees,
  archives and vendored code are excluded from lint; graph_store_am bumped to 0.3.0
  (no SQL change) to match tjs_pg for release coherence; `candidate_policy_hash` format
  cannot reproduce hashes stored in old `results/gem_eg_e1_e2_v0.1.0` rows.
- `test_gem_live.py::TestC6Write::test_the_c6_write_is_HOT` is flaky (pre-existing).

## [0.7.1] - 2026-09-02

### Fixes
- **Selecting "No memory" on a math task blanked every panel.** The iter20 runs contain
  no control cell for the math tasks, so all four panels 409'd at once — which reads as
  a broken demo rather than as missing data. A task now declares which arms it actually
  recorded (`arms_selectable`), and the selector disables and labels the ones it does
  not have instead of letting the request fail.
- **ALE and math were showing one unlabelled number.** They do not score the same way.

### Features
- `bench/agent_memory/gem_viz/scoring.py`: the score, in each domain's own terms.
  - **math** — one evaluator, one score, no split; the panel says so.
  - **ALE** — search optimises a **public** score (the cell's `ahcNNN_n50.py`, 50 public
    cases) and the reportable number is the **private** one: ALE-Bench Lite's official
    held-out split, re-scored afterwards by `tools/evotrace/ale_private_rescore.py` and
    never fed back into search. Both are shown, separately labelled, the private one set
    off in its own column.

### Design Rationale
- **Private scores exist only on the public-best-so-far chain.** That is what the
  protocol requires: the held-out split is re-scored after the fact, so it is computed
  at the seed and at the iterations where the public best advanced, and nowhere else.
  Most iterations therefore have a public score and no private one. The panel says which
  iterations are on the chain and shows the cell's endpoint instead; it never carries a
  private number onto an off-chain iteration, and never relabels the public number as
  private. Doing either would claim exactly the held-out result the split exists to
  protect.

### Notes & Caveats
- Measured on `ale:ahc011` (`gem`, graph_expand, p=1): the chain is iterations
  [0, 2, 3, 4]; at iteration 2 the private performance is 1660 (Δseed +8, Δvs-nocontext
  +10, 300 cases); at iteration 1 there is no private score and the cell endpoint
  (1652 → 1660 @it4) is shown instead.
- The private chain files are discovered under the replay roots
  (`private_chain_scores.csv`), not hardcoded: which run holds the rescore varies.

## [0.7.0] - 2026-09-02

### Features
- **A memory selector: GEM vs No memory.** The `arm` axis is threaded through every
  panel and every route. In replay it selects a DIFFERENT recorded cell — the corpus's
  `true_nocontext` control, whose receipt has all four slot counts at 0. In live it
  skips retrieval entirely and generates with no inspirations.
- **The evaluator's verdict is the headline of panel 5.** parent fitness → child fitness
  → delta, with the judge result and the number of injected memories, above the timing
  bar instead of buried in a JSON blob.

### Design Rationale
- **The control is a different run, not a filtered view.** Hiding the injected rows from
  the GEM cell would fake an experiment nobody performed; switching arms switches which
  recorded run the panels describe. `true_nocontext` — no external inspirations AND no
  own top/diverse programs — is the strict control, because the weaker one still shows
  the run its own output and so answers "which SOURCE of references is better" rather
  than "do references help at all".
- **The no-memory arm issues no query.** Retrieving and then discarding would spend the
  latency and make "0 injected" mean something different from what the recorded control
  means by it. The three legs say "not run" rather than showing empty results.
- **The score is what the chain exists to produce.** memory → prompt → code → SCORE, and
  the score was the hardest thing on the panel to find. A zero is now labelled as a real
  outcome rather than looking like a missing one.

### Notes & Caveats
- Measured on `ale:ahc011` iteration 1 (replay): GEM injects 4 and scores 429,289.48
  (Δ −4,910.06); the control injects 0 and scores 430,640.76 (Δ −1,640.90). Two
  different runs with two different parents — this is one iteration of one task, not a
  result.
- Live generation already ran the sha256-verified evaluator; this change makes its
  verdict visible rather than adding it.

## [0.6.2] - 2026-09-02

### Fixes
- **The second live generation always failed.** `llm_telemetry.install()` patches
  `OpenAILLM` once and refuses a second, different path — it is a process-wide
  singleton, not a per-call hook. `generate.py` handed it a fresh `TemporaryDirectory`
  per iteration, so the first generation succeeded and every one after it raised
  `RuntimeError: LLM telemetry already targets /tmp/.../llm_telemetry.jsonl, not
  /tmp/.../llm_telemetry.jsonl`. The first directory was also deleted on the way out,
  leaving the hook pointed at a path that no longer existed.

### Design Rationale
- **One stable file, sliced by byte offset.** The telemetry file now lives for the
  process; each call records the file size before the LLM call and reads only what was
  appended after it. That is the shape the hook actually supports, and it makes the
  slice — not the file — the unit that belongs to an iteration.
- **If the hook is already owned elsewhere, report no tokens.** Reusing whatever file
  another owner points at would silently attribute someone else's counts to this
  iteration, which is worse than reporting none; the panel says why instead.

## [0.6.1] - 2026-09-02

### Features
- **Work in flight is visible.** Every panel shows a spinner and a ticking elapsed
  counter while its request is running; the stream emits a `stage_start` event BEFORE
  each stage, so a 30-second stage reads as running rather than as a hung server. The
  Run button disables itself and, in live mode, says what it costs
  ("~60s: incl. one real iteration"). The live-generation button counts up too.
- **The Experience Graph opens one level at a time.** It starts at Task -> Sessions —
  which is the first hop of the W1.a rollup — and expands on demand: clicking a SESSION
  draws its attempts with the lineage edges among them; clicking a NODE draws the whole
  lineage TREE that attempt belongs to, root and all. Vertices with more behind them
  have a dashed outline and say what a click will open; a "collapse all" link resets.

### Design Rationale
- **A texture is not a picture.** A task has up to 1,769 attempts and the corpus has
  10,672. Drawn at once, nothing is legible and every vertex looks the same — so the
  disclosure follows the query's own shape rather than a scroll bar.
- **Clicking a node opens its TREE, not its neighbours.** "Which tree is this attempt
  in" is the question a node click asks, so the server walks up to the root and back
  down through every descendant (a recursive CTE over `parent_node_uid`) and returns
  the connected component.
- **Expansion state lives in the client and travels on every request.** The server holds
  no per-viewer state, so a reload cannot disagree with what is on screen.
- **The stream does not light up a stage card.** Its stage names and the three legs'
  names overlap only by accident ("graph" is both), so a card highlight would point at
  the wrong thing half the time.

### Notes & Caveats
- `expand=["*"]` means "everything": the vertex inspector and the stage merge must
  resolve any vertex the panels can name, not only the ones opened on the canvas.
- Measured on `ale:ahc011` (live): level 0 is 5 vertices; opening one session draws 116
  attempts; opening one attempt draws its whole tree.

## [0.6.0] - 2026-09-02

### Features
- **Live generation.** `bench/agent_memory/gem_viz/generate.py` runs one real iteration:
  live retrieval -> `PromptSampler` -> a real LLM call -> `apply_diff` -> the cell's own
  evaluator. Panel 5 in live mode now shows a prompt, a diff and a fitness that were
  produced just now, with measured tokens and latency.
- **`seed_layer="node"` (W1.b) is the default.** The vector leg finally has real work:
  it returns several similar historical attempts with their cosine distances, and the
  graph leg walks `eg_lineage` for 4 (ALE) / 6 (math) hops instead of the 2-hop `hier`
  rollup.
- **Endpoint discovery.** The LLM and embedding endpoints are probed across known ports
  rather than taken from the receipts; `/api/health` reports which answered.
- **Replay's vector leg highlights the query anchor** — the parent program whose code
  becomes the query vector — and separately names the ANN entries where the plan
  recorded them.

### Design Rationale
- **Nothing about the experiment is restated.** Sampling parameters come from
  `run_arm.FROZEN`, the prompt from openevolve's own `PromptSampler`, the memory
  rendering from `GemMemoryDatabase`, the token counts from `gem_oe.llm_telemetry`, and
  the score from the cell's `wrapped_evaluator.py`. If this module and `run_arm` ever
  disagreed about what an iteration IS, the demo would have stopped describing the
  system.
- **The evaluator is sha256-verified before anything is scored.** A swapped evaluator
  does not error, it just returns different numbers — and a fitness that merely LOOKS
  comparable is worse than no fitness at all.
- **The anchor is the cell's own recorded seed.** Anchoring on an arbitrary high-reward
  corpus node (the first attempt) put the query above almost everything the predicate
  could admit and returned nothing: a correct answer to the wrong question.
- **"seed" means two different things and they are now separated.** The QUERY ANCHOR
  (the parent, always recorded) is not the ANN ENTRIES (what the scan returned, recorded
  only by the host plans). Replay highlights the first and names the second only where
  it exists.
- **Degradation is loud.** With `seed_layer="node"` and the embedding service down, the
  adapter still runs — as task-seeded — but `health()` reports `degraded: true` and says
  why. The old `auto` default made that substitution silently, so the same demo computed
  a different query on a machine whose embedding service happened to be down.
- **One generation at a time.** Two concurrent runs share one GPU; both would be slower
  than the recorded runs and neither latency number would be comparable, so the second
  request is refused rather than queued.

### Fixes found while verifying this feature
- **Live generation is never automatic.** Panel 5 used to load on every task switch;
  with generation wired up that fired a real ~50s GPU job nobody asked for, and
  concurrent ones collided on the single-flight lock. It is now behind an explicit
  button that says what it costs.
- **DB access is serialised.** `EgStore` holds one psycopg connection and FastAPI serves
  from a thread pool; interleaving a generation with a retrieval on the same connection
  produced sporadic 500s that looked like query bugs.
- **`retrieve()` reported the wrong query family.** It printed "W1.a" regardless of the
  seed layer; with node seeding the default that label was wrong on every live call.
- **A leg that names zero vertices now says why.** With `ale:ahc008`'s anchor at fitness
  3,669,080.6, all 10 lineage-reached candidates fail `fitness > anchor` — the reuse
  query correctly declining to inject anything. A bare "0" read as a broken panel.

### Notes & Caveats
- One iteration, never a run: a full 20-iteration cell is ~27 minutes.
- Live generation needs a cell — the evaluator, its sha256 and the frozen config live in
  the cell directory and nowhere else.
- The evaluator executes LLM-generated code. It runs through the cell's existing wrapper
  (own process group, hard deadline), with a shorter outer timeout (default 300s).
- Endpoints found on this box: LLM `127.0.0.1:8003` (`qwen3.8-262k`), embeddings
  `127.0.0.1:8021` (`Qwen3-Embedding-0.6B`, dim 1024) — the receipts record 8005/8011,
  which is where they were, not where they are.

## [0.5.3] - 2026-09-02

### Fixes
- Live reads no longer leave the Postgres backend `idle in transaction`. Every public
  read on `EvoTraceLiveAdapter` is wrapped by `_releases_snapshot`, which rolls back on
  the way out however the call exits.

### Design Rationale
- `EgStore.connect` uses `autocommit=False`, so a bare SELECT opens a transaction that
  nothing closes. A long-lived demo server then pins the oldest xmin, stalls vacuum, and
  will eventually block DDL on the tables it is reading. The failure is silent for hours,
  which is exactly why it is a decorator on the methods and a test on the class rather
  than a `finally` someone can forget to add to the next method.

### Notes & Caveats
- Confirmed read-only: a full live run leaves `gem_eg_vertex` / `gem_eg_edge` /
  `gem_eg_plan_stat` row counts unchanged.

## [0.5.2] - 2026-09-02

### Features
- The execution graph now draws **both** relations: `hier` (task → session → node) and
  `lineage` (parent attempt → child attempt). The relation the query traversed is solid;
  the other is dashed and labelled "present in the graph, not walked by this query".
- A layout switch: **traversal hops** (BFS distance from the ANN entries) or **lineage
  tree** (depth in the session's own lineage). Within a column, vertices are ordered by
  parent so siblings sit together.
- Each vertex carries `lineage_depth_in_subgraph`, `parent_vid`, `children_in_subgraph`
  and `siblings_in_subgraph`; the graph leg reports edges per relation, lineage roots and
  max lineage depth.

### Design Rationale
- **Drawing one relation was wrong, not merely incomplete.** W1.a traverses `hier`, which
  is two hops and renders as a two-level star — so the picture implied a session's
  attempts were unrelated siblings. The corpus says otherwise: 406 of `ale:ahc011`'s 414
  nodes have a parent, one parent has 38 children, and the busiest session is 9 levels
  deep. The tree was in the data and absent from the screen.
- **The untraversed relation stays visible.** Hiding it would trade one wrong picture for
  another; dashing it says "this structure exists, this query did not use it", which is
  the actual state of affairs.
- **Lineage depth is scoped and labelled.** It is computed over the induced subgraph, so
  a vertex whose parent was not reached looks like a root; the docs say so rather than
  letting "root" be read as a corpus fact.

### Notes & Caveats
- Hop bounds differ by relation: `hier` is fixed at 2; `lineage` is 4 (ALE) / 6 (math),
  pinned against `gem_oe.retrievers.W1B_HOPS` by a test.
- `island_id` is NULL for every node in this corpus, so island membership is not shown
  anywhere.
- Replay's execution graph still carries only the run's own lineage (from
  `evolution_trace`), not the corpus lineage — that needs the database.

## [0.5.1] - 2026-09-02

### Features
- Replay now **names** the legs the trace actually recorded, instead of reporting all
  three as counted-only: `retrieval_telemetry.entries` gives the ANN entry uids on the
  host plans, and every returned vertex is a recorded sample of the relational leg's
  passing side. Only the graph leg stays counted-not-named.
- The result leg carries **the retrieved content**: the injected fingerprint in replay,
  the full `gem_eg_artifact.payload` in live. Rendered inline under "what came back",
  and on any row you click.
- Selecting a leg highlights it in the corpus layout too, so replay's graph responds to
  a click rather than silently relaying out. A leg with no named vertices now says so on
  the graph and shows a dashed card instead of appearing to do nothing.
- Static assets are served `Cache-Control: no-cache`.

### Design Rationale
- **A named vertex must be drawn.** Highlighting a set whose members are absent from the
  subgraph dims the whole picture and highlights nothing, which reads as a broken panel
  rather than a correct one. Vertices a leg names but the trace-derived subgraph lacks
  are merged in, tagged with the leg that named them.
- **Hop columns only where hops were measured.** Only the live probe measures BFS depth;
  laying replay's subgraph out in hop columns would invent a depth we never measured and
  collapse every vertex into one meaningless column. Replay keeps the corpus layout.
- **"Seeded but not recorded" ≠ "no vector leg".** The native fused plans seed inside the
  operator and never write the uids out; the panel now distinguishes that from a plan
  with no ANN stage, and from a leg the trace simply does not cover.
- **The passing sample is labelled as a sample.** Every returned vertex passed the
  filter, but they are not the whole passing set and nothing that failed is recorded.
  The note states both limits rather than letting the row count imply completeness.
- **`no-cache` on static.** Without it a browser keeps a stale `app.js` after a redeploy
  and the demo silently runs old code — indistinguishable from the new code being broken.

### Notes & Caveats
- Replay's payload is a ~100-character fingerprint, not the program; the note now says
  how many characters it is and points at live mode for the full artifact.

## [0.5.0] - 2026-09-02

### Features
- New panel: **the three legs**. Vector search, graph traverse and relational scan are
  executed separately and each leg's vertex set is highlighted on the Experience Graph,
  with the operator's own fused answer shown alongside as a fourth card.
- The execution graph lays vertices out by **BFS hop** — ANN entries, hop 1, hop 2 … —
  computed by calling `gph_traverse_bounded` once per depth and differencing the reaches.
- `GET /api/stages`, `ExecutionTrace` / `ExecutionStage` / `StageNode` in the adapter
  protocol, implemented by the replay, live and stub adapters.
- The live adapter now defaults to the 8005 LLM endpoint and an 8011 embedding endpoint,
  reports both in `/api/health`, and picks node- vs task-seeded retrieval from whether
  the embedding service is actually reachable.

### Design Rationale
- **The legs are probed, not intercepted.** `tjs_open` answers the whole query in one
  call and reports counters, so the only way to name the vertices each leg touched is to
  run the separable two-stage plan and query the graph AM directly. Every payload says
  this is a second execution, and stage 4 states how many returned vertices fell outside
  the probed reach rather than hiding the discrepancy.
- **Hops are measured, not inferred.** Placing a vertex at "hop 2" because it looked far
  away would be a drawing decision dressed as a fact; differencing `reach(d)` against
  `reach(d-1)` makes the column a real BFS level.
- **Replay refuses to highlight.** A recorded run's telemetry names only `result_uids`,
  so the other three legs render as "counted, not named". Highlighting a plausible set
  would be the most convincing lie this demo could tell.
- **Schema-qualified AM calls.** `graph_store_am` installs into its own schema, which is
  not on the default `search_path`; the schema is looked up at runtime rather than
  hardcoded so a differently-installed instance still works.

### Notes & Caveats
- Under the `same_task` split with a task-level entry kind there is exactly one eligible
  task vertex, so the ANN leg is degenerate — it confirms the entry rather than choosing
  among candidates. The panel says so. Node-seeded (W1.b) runs give that leg real work,
  and live mode uses them whenever the embedding service is up.
- The stage tables cap at 200 rows; highlighting always covers the full vertex set.

## [0.4.1] - 2026-09-02

### Features
- `gem_viz` now reads `results/e2/` and shows only the cells that ran **20 iterations** —
  the recent sweeps. `--replay-root` is repeatable, `--iterations N` moves the filter and
  `--any-iterations` drops it; `bench/out/oe` still works through the same flag.
- Curated tasks are now `ale:ahc008`, `ale:ahc011`, `ale:ahc025` (requested) plus
  `math:circle_packing` and `math:heilbronn_triangle` for cross-domain contrast.
- Cells are labelled by **variant** (physical plan · seed layer · top-m · injection
  frequency), and panel 5 states the axis it is actually diffing.
- Panel 4 reports **measured** LLM tokens and API latency from `llm_telemetry.jsonl`,
  and the timing bar splits into retrieval / LLM api / evaluator+harness.
- The UI opens on each cell's first iteration whose injection gate opened, and reports a
  shut gate as sampling policy rather than as an empty result.

### Design Rationale
- **Cell discovery is layout-agnostic.** `results/e2/<run>/cells/<cell>/` and
  `bench/out/oe/<run>/<cell>/` differ by a directory level, so a cell is defined by
  holding a `run_receipt.json` rather than by a fixed glob depth.
- **The iteration filter is on by default.** `results/e2` mixes 20- and 100-iteration
  sweeps; a browser that silently offered both would invite comparisons between cells
  that never ran the same experiment.
- **`arm` was no longer the axis.** Six `gem` cells per ALE task differ by plan and
  injection frequency, so collapsing them into "gem" would have hidden what the
  experiment sweeps and made the diff compare a cell with itself.
- **Tokens are joined, not assumed.** `llm_telemetry.jsonl` has no iteration field, so
  calls are attributed to the iteration whose `[start, next start)` window on the same
  pid contains them. If any call falls outside every window the per-iteration view is
  withheld and the panel falls back to a labelled estimate — a mis-attributed token
  count is worse than none.
- **A shut gate and an empty diff are outcomes, not bugs.** Both now render as explicit
  statements; a blank panel would have read as a rendering failure and invited the wrong
  conclusion about the retrieval system.

### Notes & Caveats
- These runs contain no polyglot or cognee cells, so panel 5 is a **plan** diff, not a
  memory-system diff, and says so.
- `predicate_sql` reconstruction now handles node-seeded (W1.b) runs: the reward bound is
  `fitness_gt` (strictly beat the parent) rather than `min_fitness`, plus the finite-
  fitness clause. It still cannot recover the target session's exclusion clause or the
  parent's duplicate-code exclusions; the panel names that gap.

## [0.4.0] - 2026-09-02

### Features
- `bench/agent_memory/gem_viz/`: an interactive four-panel walkthrough of GEM on
  EvoTrace — task browser, Experience Graph with a vertex inspector, retrieval trace,
  generate-and-evaluate, plus a three-arm diff. FastAPI backend, dependency-free single
  page front end, SSE run stream.
- Two adapters behind one `DatasetAdapter` protocol: `replay` (reads the cell
  directories `gem_oe.run_arm` already wrote — no Postgres, no LLM, no network) and
  `live` (real `tjs_open` execution against TriDB, real `gem_eg_vertex` rows).
- `stub_adapter.StubAdapter`: a deliberately EvoTrace-unlike second implementation, so
  "the UI is not coupled to EvoTrace" is a test rather than a claim.
- Five curated tasks (`curated.py`), each with the reason it was picked rendered in the
  UI: 4 five-arm math tasks plus `ale:ahc015` as the cross-domain check.

### Design Rationale
- **Provenance is mandatory, not decorative.** Every payload carries `{mode, source,
  note}` and every panel renders it. A number that cannot name its file line or its SQL
  does not get shown. The mode badge means a replayed panel can never be mistaken for a
  live one.
- **Live retrieval reuses the real arm.** `EvoTraceLiveAdapter` constructs the actual
  `GemRetriever`, asks it for its own `QuerySpec`, and prints the predicate string
  `tjs_open` was handed. A parallel re-implementation would drift and the demo would
  quietly stop describing the system.
- **Rejection attribution is a labelled diagnostic.** The operator reports counters, not
  the identity of what it discarded, so live mode re-evaluates the same pushdown clauses
  over the task's node set and attributes each miss to the first failing clause. The
  clause list is rebuilt and then checked verbatim against `Predicate.to_sql()`; on any
  mismatch the panel reports attribution as unavailable instead of guessing.
- **The default cell is chosen by richness.** A curated task whose default cell injects
  nothing at iteration 1 looks broken; `injection_frequency` is part of the ranking for
  exactly that reason, and a test pins it.
- **Panel 4 shows retrieval to scale against the LLM.** On the recorded runs retrieval is
  ~0.1-0.5% of an iteration's wall clock. Since the end-to-end bottleneck is the LLM, the
  UI has to say so rather than let a retrieval-latency panel imply otherwise.

### Notes & Caveats
- Live mode needs `.venv-e0` (it reuses `gem_oe`'s retriever, which imports openevolve).
  Replay mode runs in `.venv` with no database.
- Live **generation** needs the OpenEvolve LLM endpoint; when it is down panel 4 reports
  it and points at replay. Live **three-arm diff** needs the polyglot and cognee stacks,
  so it stays a replay panel.
- Replay cannot attribute per-candidate rejection, holds no problem statement, records no
  real token counts (`est_*` is `chars/4`, basis printed), and reconstructs the predicate
  SQL from the recorded bound — it cannot recover the target session's exclusion clause.
  All four limits are stated in the UI, not just in this file.
