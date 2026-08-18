-- Benchmark-internal E0 native path: multi-anchor/type graph iterator plus three
-- bounded tri-modal plan shapes.  These declarations intentionally are not extension
-- members and do not widen graph_query()'s public v1 surface.

CREATE EXTENSION vector;
CREATE EXTENSION graph_store_am;
CREATE EXTENSION tjs_pg;

CREATE FUNCTION graph_store.gph_traverse_bounded_multi(
  bigint[], integer, integer[], boolean, bigint
) RETURNS SETOF bigint
AS '$libdir/graph_store_am', 'gph_traverse_bounded_multi'
LANGUAGE C VOLATILE STRICT;

CREATE FUNCTION tjs_e0_open(
  regclass, integer, integer, integer, text, text, vector,
  bigint[], integer[], boolean, text, text
) RETURNS SETOF bigint
AS '$libdir/tjs_pg', 'tjs_e0_open_pg'
LANGUAGE C VOLATILE STRICT;

CREATE FUNCTION tjs_e0_termination() RETURNS text
AS '$libdir/tjs_pg', 'tjs_e0_termination_pg'
LANGUAGE C VOLATILE;

SELECT set_config('e0.a', graph_store.register_edge_type('A')::text, false);
SELECT set_config('e0.b', graph_store.register_edge_type('B')::text, false);

SELECT count(graph_store.gph_insert_vertex()) FROM generate_series(1, 6);
SELECT graph_store.gph_set_identity_mode(true);
SELECT graph_store.gph_insert_edge(0, 1, current_setting('e0.a')::integer);
SELECT graph_store.gph_insert_edge(0, 2, current_setting('e0.b')::integer);
SELECT graph_store.gph_insert_edge(1, 3, current_setting('e0.b')::integer);
SELECT graph_store.gph_insert_edge(4, 3, current_setting('e0.a')::integer);

CREATE TABLE e0_node (
  id bigint PRIMARY KEY,
  entity_type text NOT NULL,
  generation integer,
  parent_vid bigint,
  embedding vector(2) NOT NULL
);
INSERT INTO e0_node VALUES
  (0, 'seed', 0, NULL, '[1,0]'),
  (1, 'x', 1, 0, '[0.99,0.01]'),
  (2, 'y', 1, 0, '[0.9,0.1]'),
  (3, 'x', 2, 1, '[0.8,0.2]'),
  (4, 'seed', 0, NULL, '[0,1]'),
  (5, 'x', 1, 4, '[0.1,0.9]');
CREATE INDEX e0_node_hnsw ON e0_node USING hnsw (embedding vector_cosine_ops);
ANALYZE e0_node;
SET enable_seqscan = off;
SET hnsw.iterative_scan = relaxed_order;
SET graph_store.assume_dense_open = on;

DO $$
DECLARE got bigint[];
BEGIN
  SELECT array_agg(v ORDER BY v) INTO got
  FROM graph_store.gph_traverse_bounded_multi(
    ARRAY[0]::bigint[], 2,
    ARRAY[current_setting('e0.a')::integer,current_setting('e0.b')::integer],
    false, 100) v;
  IF got <> ARRAY[1,2,3]::bigint[] THEN
    RAISE EXCEPTION 'multi-type union got %', got;
  END IF;

  SELECT array_agg(v ORDER BY v) INTO got
  FROM graph_store.gph_traverse_bounded_multi(
    ARRAY[0,4]::bigint[], 2,
    ARRAY[current_setting('e0.a')::integer,current_setting('e0.b')::integer],
    true, 100) v;
  IF got <> ARRAY[3]::bigint[] THEN
    RAISE EXCEPTION 'multi-anchor intersection got %', got;
  END IF;
END $$;

DO $$
DECLARE shape text; placement text; got bigint[];
BEGIN
  FOR shape, placement IN VALUES
    ('vector_first','post'), ('filter_first','pre'), ('traverse_first','during')
  LOOP
    SELECT array_agg(v ORDER BY v) INTO got FROM tjs_e0_open(
      'e0_node', 4, 3, 2, 'id', 'entity_type IN (''x'',''y'')', '[1,0]'::vector,
      ARRAY[0]::bigint[],
      ARRAY[current_setting('e0.a')::integer,current_setting('e0.b')::integer],
      false, shape, placement) v;
    IF got <> ARRAY[1,2,3]::bigint[] THEN
      RAISE EXCEPTION '%/% got %', shape, placement, got;
    END IF;
    IF tjs_e0_termination() <> 'graph_exhausted' THEN
      RAISE EXCEPTION '%/% termination %', shape, placement, tjs_e0_termination();
    END IF;
  END LOOP;
  RAISE NOTICE 'PASS: E0 native multi traversal and all three shapes';
END $$;
