-- E0 benchmark-only native execution surface.
--
-- The loader replaces @GRAPH_LIB@ / @TJS_LIB@ with absolute paths to the locally built
-- shared objects.  These declarations intentionally do not belong to graph_query() and do
-- not widen the single public v1 query template.

-- CREATE EXTENSION installs the released entry points but does not dlopen the image until a
-- C function is first called.  Rebind every entry point used by the E0 loader/validator before
-- that first call: PostgreSQL must not load both the installed and freshly-built images in one
-- backend because both legitimately register the same graph_store.* GUCs in _PG_init().
CREATE OR REPLACE FUNCTION graph_store.gph_insert_vertex() RETURNS bigint
AS '@GRAPH_LIB@', 'gph_insert_vertex'
LANGUAGE C VOLATILE;

CREATE OR REPLACE FUNCTION graph_store.gph_insert_edge(bigint, bigint) RETURNS void
AS '@GRAPH_LIB@', 'gph_insert_edge'
LANGUAGE C VOLATILE STRICT;

CREATE OR REPLACE FUNCTION graph_store.gph_insert_edge(bigint, bigint, integer) RETURNS void
AS '@GRAPH_LIB@', 'gph_insert_edge'
LANGUAGE C VOLATILE STRICT;

CREATE OR REPLACE FUNCTION graph_store.gph_insert_edges(bigint, bigint[]) RETURNS bigint
AS '@GRAPH_LIB@', 'gph_insert_edges'
LANGUAGE C VOLATILE STRICT;

CREATE OR REPLACE FUNCTION graph_store.gph_insert_edges(bigint, bigint[], integer) RETURNS bigint
AS '@GRAPH_LIB@', 'gph_insert_edges'
LANGUAGE C VOLATILE STRICT;

CREATE OR REPLACE FUNCTION graph_store.gph_vertex_count() RETURNS bigint
AS '@GRAPH_LIB@', 'gph_vertex_count'
LANGUAGE C VOLATILE;

CREATE OR REPLACE FUNCTION graph_store.gph_edge_count() RETURNS bigint
AS '@GRAPH_LIB@', 'gph_edge_count'
LANGUAGE C VOLATILE;

CREATE OR REPLACE FUNCTION graph_store.gph_visible_edge_count() RETURNS bigint
AS '@GRAPH_LIB@', 'gph_visible_edge_count'
LANGUAGE C VOLATILE;

CREATE OR REPLACE FUNCTION graph_store.gph_page_reads() RETURNS bigint
AS '@GRAPH_LIB@', 'gph_page_reads'
LANGUAGE C VOLATILE;

CREATE OR REPLACE FUNCTION graph_store.gph_traverse_bounded_multi(
    bigint[], integer, integer[], boolean, bigint
) RETURNS SETOF bigint
AS '@GRAPH_LIB@', 'gph_traverse_bounded_multi'
LANGUAGE C VOLATILE STRICT;

-- Keep the traversal counters in the same loaded graph_store_am image as the new iterator.
CREATE OR REPLACE FUNCTION graph_store.gph_visits() RETURNS bigint
AS '@GRAPH_LIB@', 'gph_visits'
LANGUAGE C VOLATILE;

CREATE OR REPLACE FUNCTION graph_store.gph_traverse_bounded_censored() RETURNS boolean
AS '@GRAPH_LIB@', 'gph_traverse_bounded_censored'
LANGUAGE C VOLATILE;

CREATE OR REPLACE FUNCTION public.tjs_e0_open(
    regclass, integer, integer, integer, text, text, vector,
    bigint[], integer[], boolean, text, text
) RETURNS SETOF bigint
AS '@TJS_LIB@', 'tjs_e0_open_pg'
LANGUAGE C VOLATILE;

CREATE OR REPLACE FUNCTION public.tjs_e0_vector_us() RETURNS bigint
AS '@TJS_LIB@', 'tjs_e0_vector_us_pg'
LANGUAGE C VOLATILE;

CREATE OR REPLACE FUNCTION public.tjs_e0_graph_us() RETURNS bigint
AS '@TJS_LIB@', 'tjs_e0_graph_us_pg'
LANGUAGE C VOLATILE;

CREATE OR REPLACE FUNCTION public.tjs_e0_filter_us() RETURNS bigint
AS '@TJS_LIB@', 'tjs_e0_filter_us_pg'
LANGUAGE C VOLATILE;

CREATE OR REPLACE FUNCTION public.tjs_e0_termination() RETURNS text
AS '@TJS_LIB@', 'tjs_e0_termination_pg'
LANGUAGE C VOLATILE;

-- Existing honesty probes must resolve to the same tjs_pg image as tjs_e0_open.
CREATE OR REPLACE FUNCTION public.tjs_open_candidates_examined() RETURNS bigint
AS '@TJS_LIB@', 'tjs_open_candidates_examined_pg'
LANGUAGE C VOLATILE;

CREATE OR REPLACE FUNCTION public.tjs_open_graph_examined() RETURNS bigint
AS '@TJS_LIB@', 'tjs_open_graph_examined_pg'
LANGUAGE C VOLATILE;

CREATE OR REPLACE FUNCTION public.tjs_open_graph_censored() RETURNS boolean
AS '@TJS_LIB@', 'tjs_open_graph_censored_pg'
LANGUAGE C VOLATILE;
