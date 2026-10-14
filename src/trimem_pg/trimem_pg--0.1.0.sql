-- TriMEM semantic-predicate CustomScan (ADR-0027). The LLM stays out of process: verdicts come
-- from trimem.sem_cache or from the worker through the shared-memory mailbox.
\echo Use "CREATE EXTENSION trimem_pg" to load this file. \quit

CREATE TABLE trimem.sem_cache (
    key               text PRIMARY KEY,          -- trimem.key(op, model, prompt_version, predicate, input)
    op                text NOT NULL,
    model             text NOT NULL,
    prompt_version    text NOT NULL,
    predicate         text NOT NULL,
    verdict           boolean NOT NULL,
    prompt_tokens     integer NOT NULL DEFAULT 0,
    completion_tokens integer NOT NULL DEFAULT 0,
    seconds           double precision NOT NULL DEFAULT 0,
    created_at        timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE trimem.stats (
    model          text NOT NULL,
    prompt_version text NOT NULL,
    predicate      text NOT NULL,
    n_eval         bigint NOT NULL DEFAULT 0,   -- unique inputs judged
    n_true         bigint NOT NULL DEFAULT 0,
    n_hit          bigint NOT NULL DEFAULT 0,   -- of n_eval, answered from sem_cache
    updated_at     timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (model, prompt_version, predicate)
);
CREATE INDEX sem_cache_predicate_idx ON trimem.sem_cache (model, prompt_version, predicate);
SELECT pg_catalog.pg_extension_config_dump('trimem.sem_cache', '');
SELECT pg_catalog.pg_extension_config_dump('trimem.stats', '');

CREATE FUNCTION trimem.key(op text, model text, prompt_version text, predicate text, input text)
RETURNS text AS 'MODULE_PATHNAME', 'trimem_key' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION trimem.sem_filter_support(internal) RETURNS internal
AS 'MODULE_PATHNAME', 'trimem_sem_filter_support' LANGUAGE C STRICT;

-- Marker: planned as "Custom Scan (TriMEM SemFilter)"; direct evaluation fails closed.
CREATE FUNCTION trimem.sem_filter(predicate text, input text) RETURNS boolean
AS 'MODULE_PATHNAME', 'trimem_sem_filter' LANGUAGE C STABLE PARALLEL UNSAFE
SUPPORT trimem.sem_filter_support;

-- Worker protocol.
CREATE FUNCTION trimem.take(max_n integer DEFAULT 64, timeout_ms integer DEFAULT 1000)
RETURNS TABLE (slot integer, seq bigint, op text, model text, prompt_version text,
               predicate text, input text)
AS 'MODULE_PATHNAME', 'trimem_take' LANGUAGE C VOLATILE STRICT PARALLEL UNSAFE;
CREATE FUNCTION trimem.complete(slots integer[], seqs bigint[], verdicts boolean[])
RETURNS integer AS 'MODULE_PATHNAME', 'trimem_complete' LANGUAGE C VOLATILE PARALLEL UNSAFE;
CREATE FUNCTION trimem.flush_stats() RETURNS integer
AS 'MODULE_PATHNAME', 'trimem_flush_stats' LANGUAGE C VOLATILE PARALLEL UNSAFE;
CREATE FUNCTION trimem.reset_stats() RETURNS void
AS 'MODULE_PATHNAME', 'trimem_reset_stats' LANGUAGE C VOLATILE PARALLEL UNSAFE;
CREATE FUNCTION trimem.status() RETURNS TABLE (state text, slots integer)
AS 'MODULE_PATHNAME', 'trimem_status' LANGUAGE C VOLATILE PARALLEL UNSAFE;

REVOKE ALL ON FUNCTION trimem.take(integer, integer), trimem.complete(integer[], bigint[], boolean[]),
    trimem.flush_stats(), trimem.reset_stats() FROM PUBLIC;

-- ---------------------------------------------------------------- graph Expand (Fig. 1(b))
CREATE TABLE trimem.graph_stats (
    type_id     integer PRIMARY KEY,
    edge_type   text NOT NULL,
    n_vertices  bigint NOT NULL,   -- all vertices at analyze time
    n_src       bigint NOT NULL,   -- vertices with >= 1 out-edge of this type
    n_edges     bigint NOT NULL,
    max_deg     integer NOT NULL,
    analyzed_at timestamptz NOT NULL DEFAULT now()
);

-- Per-edge-type degree statistics for the Expand cost model (like ANALYZE for the graph AM).
CREATE FUNCTION trimem.analyze_graph() RETURNS integer LANGUAGE sql VOLATILE AS $$
    WITH d AS (
        SELECT type_id, src, count(*) AS deg FROM graph_store.gph_dump_edges() GROUP BY 1, 2
    ), agg AS (
        SELECT d.type_id, t.name, count(*) AS n_src, sum(d.deg) AS n_edges, max(d.deg) AS max_deg
        FROM d JOIN graph_store.edge_type t ON t.id = d.type_id GROUP BY 1, 2
    ), up AS (
        INSERT INTO trimem.graph_stats AS g (type_id, edge_type, n_vertices, n_src, n_edges, max_deg)
        SELECT type_id, name, graph_store.gph_vertex_count(), n_src, n_edges, max_deg FROM agg
        ON CONFLICT (type_id) DO UPDATE SET edge_type = EXCLUDED.edge_type,
            n_vertices = EXCLUDED.n_vertices, n_src = EXCLUDED.n_src, n_edges = EXCLUDED.n_edges,
            max_deg = EXCLUDED.max_deg, analyzed_at = now()
        RETURNING 1
    )
    SELECT count(*)::int FROM up
$$;

CREATE FUNCTION trimem.expand_support(internal) RETURNS internal
AS 'MODULE_PATHNAME', 'trimem_expand_support' LANGUAGE C STRICT;

-- Planned as "Custom Scan (TriMEM Expand)"; direct (FunctionScan) evaluation fails closed.
CREATE FUNCTION trimem.expand(seed bigint, edge_type text, fanout integer,
                              OUT src bigint, OUT dst bigint) RETURNS SETOF record
AS 'MODULE_PATHNAME', 'trimem_expand' LANGUAGE C STABLE STRICT PARALLEL UNSAFE
SUPPORT trimem.expand_support;
