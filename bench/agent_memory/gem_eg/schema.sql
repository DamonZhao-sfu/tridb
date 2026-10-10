-- The Experience Graph, as first-class TriDB state.
--
-- arXiv:2606.29823 (Trellis) argues that long-horizon agentic search produces "a
-- structured object we call an experience graph: executable artifacts, tool outputs,
-- rewards, sibling comparisons, and causal lineage", and that treating it as database
-- state makes four access patterns fall out:
--
--     "Frontier selection is a query, cross-session reuse is vector-seeded graph
--      retrieval, training-data extraction is a materialized view, and reconstructing
--      what an agent knew at any past step is a time-travel query."
--
-- This schema is that object on stock PostgreSQL + pgvector + graph_store_am + tjs_pg.
--
-- WHY NOT gem_unit
-- ---------------
-- GEM's `gem_unit` is a knowledge-oriented semantic unit: one node type, one
-- UNIQUE(scope_id,title), and `embedding NOT NULL`. The paper's Task / Session / Node /
-- Prompt objects have different lifecycles, different predicates and different index
-- needs, and only Task genuinely needs a description vector. Flattening them into
-- gem_unit would force invented embeddings and invented titles. So the Experience Graph
-- is a SIBLING sub-model, and distilled knowledge derived from it still lands in
-- gem_unit with a DERIVED_FROM provenance link.
--
-- Placeholder: :dim (embedding dimension), substituted by EgStore.init_schema().

-- ===========================================================================
-- The graph-vertex table
-- ===========================================================================
--
-- ONE physical table, because tjs_open takes a single `tbl regclass` and resolves both
-- its ANN leg and its per-candidate fetch against it
-- (`SELECT <vec> FROM <tbl> WHERE <id_col> = $1`). A Task/Session/Node split across
-- three tables would make the fused operator unable to see the whole path. The typed
-- detail lives in the 1:1 tables below; this is the routing surface.
--
-- `id` MUST equal the native graph vid: tjs_open passes raw vids to that fetch, so a
-- row is reachable through the graph only when the two agree. vids come from
-- graph_store.gph_allocated_vids().
--
-- *** embedding IS NULLABLE, AND THAT IS THE DESIGN ***
-- Measured in tjs_pg.c: both the filter-first leg (~:1318) and the PPR bridge leg
-- (~:1744) do `if (vnull) continue;`. A vertex with no vector is therefore traversable
-- but never rankable — which is exactly right for Sessions and Prompts, which are hops
-- on the path to an answer and not answers themselves. The consequence is load-bearing
-- and must not be forgotten: ANY vertex kind that a query must RETURN needs a vector.
CREATE TABLE IF NOT EXISTS gem_eg_vertex (
    id            bigint PRIMARY KEY,
    scope_id      text NOT NULL,
    kind          text NOT NULL
                  CHECK (kind IN ('task','session','node','prompt','artifact')),
    uid           text NOT NULL,              -- the normalizer's stable identity
    label         text NOT NULL DEFAULT '',
    embedding     vector(:dim),               -- NULL = "a hop, not an answer"
    -- Which index track produced the vector. Task-ANN and Node-ANN are different
    -- experiments and must never be pooled in one results table, so the source is
    -- stored beside the vector rather than inferred from `kind`.
    embedding_source text,
    embedding_model  text,

    -- ---- pushdown columns -------------------------------------------------
    -- tjs_open's `filter` is raw SQL evaluated per candidate against THIS table, so
    -- every predicate the paper's queries need is a real column here rather than a
    -- jsonb lookup: a filter is executed once per graph-reached vertex.
    task_uid      text,
    session_uid   text,
    backend       text,
    domain        text,
    language      text,
    model         text,
    status        text,
    is_valid      boolean,
    fitness       double precision,
    iteration     integer,
    generation    integer,

    -- ---- governance -------------------------------------------------------
    -- The paper's Collectivity property is "shared across sessions and users under
    -- policy". `scope_id` is the corpus/experiment boundary; `visibility_group` is the
    -- split label a leakage audit filters on (source vs held-out target).
    visibility_group text,
    provenance    jsonb NOT NULL DEFAULT '{}'::jsonb,
    metadata      jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at    timestamptz NOT NULL DEFAULT now(),
    UNIQUE (scope_id, uid)
);

CREATE INDEX IF NOT EXISTS gem_eg_vertex_kind_idx ON gem_eg_vertex (scope_id, kind);
CREATE INDEX IF NOT EXISTS gem_eg_vertex_session_idx ON gem_eg_vertex (session_uid);
CREATE INDEX IF NOT EXISTS gem_eg_vertex_task_idx ON gem_eg_vertex (task_uid);
-- The reward predicate the reuse query pushes down.
CREATE INDEX IF NOT EXISTS gem_eg_vertex_fitness_idx
    ON gem_eg_vertex (task_uid, fitness DESC)
    WHERE kind = 'node' AND is_valid;
-- Cosine, matching gem_unit and the rest of bench/agent_memory. NULL embeddings are
-- simply not indexed, which is why a vectorless hop costs nothing here.
CREATE INDEX IF NOT EXISTS gem_eg_vertex_hnsw
    ON gem_eg_vertex USING hnsw (embedding vector_cosine_ops);

-- ===========================================================================
-- Typed detail — one row per vertex of that kind
-- ===========================================================================

-- Task: "problem specification, target environment, success metric". The ONLY object
-- the paper requires a description embedding for; its vector is the strict entry point
-- of cross-session reuse.
CREATE TABLE IF NOT EXISTS gem_eg_task (
    id               bigint PRIMARY KEY REFERENCES gem_eg_vertex(id) ON DELETE CASCADE,
    task_uid         text NOT NULL UNIQUE,
    domain           text NOT NULL,
    task_key         text NOT NULL,
    task_family      text,
    specification    text NOT NULL DEFAULT '',
    -- Whether `specification` is a real problem statement or a stand-in derived from
    -- the run name. A false here forbids calling the ANN leg a task-description ANN.
    specification_complete boolean NOT NULL DEFAULT false,
    specification_source   text,
    success_metric   text,
    evaluator_ref    text,
    evaluator_sha256 text,
    target_environment jsonb NOT NULL DEFAULT '{}'::jsonb
);

-- Session: "who is searching, with what algorithm and configuration, and how far it got".
CREATE TABLE IF NOT EXISTS gem_eg_session (
    id            bigint PRIMARY KEY REFERENCES gem_eg_vertex(id) ON DELETE CASCADE,
    session_uid   text NOT NULL UNIQUE,
    task_uid      text NOT NULL,
    run_rel       text NOT NULL,
    backend       text NOT NULL,
    search_algorithm text NOT NULL,
    experiment_group text,                    -- ablation / empty_seed / strong_seed / ...
    model         text,
    edit_mode     text,                       -- diff vs nodiff: what an edit IS
    temperature   double precision,
    configured_iterations integer,
    node_count    integer NOT NULL DEFAULT 0,
    lineage_edge_count integer NOT NULL DEFAULT 0,
    best_node_uid text,
    best_fitness  double precision,
    status_counts jsonb NOT NULL DEFAULT '{}'::jsonb,
    prompt_complete_ratio double precision NOT NULL DEFAULT 0,
    -- EvoTrace anonymisation nulled every wall-clock field. Recording the ABSENCE
    -- stops a null `started_at` from reading as "not captured yet", and forbids any
    -- chronological cutoff claim.
    wall_clock_available boolean NOT NULL DEFAULT false,
    files_present jsonb NOT NULL DEFAULT '{}'::jsonb
);

-- Node: one ATTEMPT. Rejected and failed attempts are first-class rows, not filtered
-- out: they carry the failure evidence the Repair pattern and DPO pairs are built from.
CREATE TABLE IF NOT EXISTS gem_eg_node (
    id             bigint PRIMARY KEY REFERENCES gem_eg_vertex(id) ON DELETE CASCADE,
    node_uid       text NOT NULL UNIQUE,
    session_uid    text NOT NULL,
    task_uid       text NOT NULL,
    program_id     text NOT NULL,
    parent_node_uid text,
    iteration      integer,
    generation     integer,
    island_id      integer,
    status         text NOT NULL,
    is_valid       boolean NOT NULL,
    language       text,
    fitness        double precision,
    metrics        jsonb NOT NULL DEFAULT '{}'::jsonb,
    error_signature text,
    changes        text,                      -- the backend's own edit description
    artifact_uid   text,
    prompts_sha256 text,
    source_file    text,
    source_row     integer
);
CREATE INDEX IF NOT EXISTS gem_eg_node_session_idx ON gem_eg_node (session_uid, iteration);
CREATE INDEX IF NOT EXISTS gem_eg_node_parent_idx ON gem_eg_node (parent_node_uid);
CREATE INDEX IF NOT EXISTS gem_eg_node_error_idx ON gem_eg_node (error_signature)
    WHERE error_signature IS NOT NULL;

-- Prompt history: the exact messages the model saw and produced. A REFERENCE plus a
-- presence flag — `blob_present=false` means the trace named a payload we do not hold,
-- which is a different fact from "there was no prompt" and must never collapse into it.
CREATE TABLE IF NOT EXISTS gem_eg_prompt (
    id             bigint PRIMARY KEY REFERENCES gem_eg_vertex(id) ON DELETE CASCADE,
    prompt_uid     text NOT NULL UNIQUE,
    node_uid       text NOT NULL,
    session_uid    text NOT NULL,
    prompts_sha256 text NOT NULL,
    blob_rel       text,
    blob_present   boolean NOT NULL DEFAULT false,
    payload        jsonb                      -- inlined when small enough to hold
);

-- Content-addressed artifacts. Deduplicated by sha256 ACROSS the whole corpus, which is
-- also how the leakage audit finds source code that reappears in a held-out session.
CREATE TABLE IF NOT EXISTS gem_eg_artifact (
    artifact_uid   text PRIMARY KEY,
    sha256         text NOT NULL,
    kind           text NOT NULL DEFAULT 'solution',
    language       text,
    blob_rel       text,
    byte_len       bigint,
    payload        text,                      -- TOASTed; stays inside the one WAL
    reference_count integer NOT NULL DEFAULT 0,
    session_count  integer NOT NULL DEFAULT 0
);

-- ===========================================================================
-- The logical-step change log — the time-travel substrate
-- ===========================================================================
--
-- The paper's time-travel is "reconstructing what an agent knew at any past step".
-- STEP, not timestamp. EvoTrace has no wall clock at all, and even with one the search
-- state that matters (island membership, archive contents, elite pool, best-so-far) is
-- indexed by iteration. So the key is (session_uid, iteration) and there is deliberately
-- no timestamptz column to be tempted by.
CREATE TABLE IF NOT EXISTS gem_eg_state_event (
    id            bigint GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    scope_id      text NOT NULL,
    session_uid   text NOT NULL,
    iteration     integer NOT NULL,
    entity        text NOT NULL CHECK (entity IN ('membership','scalar')),
    role          text,                       -- population_member / archive / island / ...
    slot_key      text,
    node_uid      text,
    value         jsonb,
    provenance    text NOT NULL,
    source_row    integer
);
CREATE INDEX IF NOT EXISTS gem_eg_state_event_asof_idx
    ON gem_eg_state_event (session_uid, iteration, entity);
CREATE INDEX IF NOT EXISTS gem_eg_state_event_role_idx
    ON gem_eg_state_event (session_uid, role, iteration);

-- Per-session LLM usage. Session grain ON PURPOSE: EvoTrace's llm_calls.jsonl carries
-- no program id, so a per-node join would be fabricated.
CREATE TABLE IF NOT EXISTS gem_eg_llm_call (
    id            bigint GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    scope_id      text NOT NULL,
    session_uid   text NOT NULL,
    seq           integer NOT NULL,
    model         text,
    api_base      text,
    temperature   double precision,
    prompt_tokens bigint,
    completion_tokens bigint,
    reasoning_tokens  bigint,
    error         text
);
CREATE INDEX IF NOT EXISTS gem_eg_llm_call_session_idx ON gem_eg_llm_call (session_uid);

-- ===========================================================================
-- Edge metadata — topology itself lives in the AM
-- ===========================================================================
--
-- CLAUDE.md rule 3: topology is an adjacency-list access method, never a relational
-- join table. This table carries only what the AM cannot: the relation name, the
-- provenance receipt, and whether the edge is a derived inverse. Path execution reads
-- the graph AM; this table is the audit trail.
--
-- `derived_inverse` matters because graph_store supports OUTGOING traversal only
-- (direction=in raises). Ancestors are reachable solely through inverse edges we
-- materialize ourselves, so they are counted separately and never inflate a headline
-- edge count.
CREATE TABLE IF NOT EXISTS gem_eg_edge (
    src             bigint NOT NULL REFERENCES gem_eg_vertex(id) ON DELETE CASCADE,
    dst             bigint NOT NULL REFERENCES gem_eg_vertex(id) ON DELETE CASCADE,
    edge_type       integer NOT NULL,          -- graph_store.edge_type.id
    relation        text NOT NULL,
    derived_inverse boolean NOT NULL DEFAULT false,
    -- Also written under the eg_hier / eg_hier_inv rollup type so one bounded BFS can
    -- walk Task -> Session -> Node. The rollup copy is not a second logical edge.
    rollup_of       text,
    provenance      text NOT NULL DEFAULT '',
    source_row      integer,
    dataset_revision text,
    PRIMARY KEY (src, dst, edge_type)
);
CREATE INDEX IF NOT EXISTS gem_eg_edge_relation_idx ON gem_eg_edge (relation);

-- ===========================================================================
-- Load receipts
-- ===========================================================================
--
-- Idempotence is a claim, so it is measured: a second load of the same corpus must add
-- zero vertices and zero edges, and this table is where that is proven.
CREATE TABLE IF NOT EXISTS gem_eg_load (
    id             bigint GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    scope_id       text NOT NULL,
    dataset_revision text NOT NULL,
    normalizer_version text NOT NULL,
    counts         jsonb NOT NULL DEFAULT '{}'::jsonb,
    rejects        jsonb NOT NULL DEFAULT '{}'::jsonb,
    xid            xid8 NOT NULL DEFAULT pg_current_xact_id(),
    at             timestamptz NOT NULL DEFAULT now()
);
