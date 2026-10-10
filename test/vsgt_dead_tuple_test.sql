\ir vsgt_fixture.sql
-- Regression: payload / vector updates leave dead heap tuples behind the pk btree. When the
-- request probe meets one, btgettuple lazily pallocs so->killedItems in the CURRENT memory
-- context. VSGT used to probe while the per-row scratch context was current, so the next
-- scratch reset freed killedItems and index_endscan() later failed with
-- "could not find block containing chunk" (seen in the Figure 1 ingest/retrieve workload).
-- Non-HOT updates (an indexed column changes) guarantee dead index entries; autovacuum is
-- off for the table so the dead entries are still there when the probe runs. Before the
-- fix this file crashed the backend (SIGSEGV) on stock PG16.
ALTER TABLE vsgt_nodes SET (autovacuum_enabled = false);
-- (A byte-identical vector would still allow HOT, so change it and change it back.)
UPDATE vsgt_nodes SET embedding = embedding + '[0,0,1]'::vector,
                      metadata = metadata || jsonb_build_object('lesson','v2');
UPDATE vsgt_nodes SET embedding = embedding - '[0,0,1]'::vector,
                      metadata = metadata || jsonb_build_object('lesson','v3');
SET hnsw.iterative_scan = relaxed_order;
-- Several requests in one backend, each fully consumed and closed (index_endscan runs).
SELECT vsgt_assert((SELECT count(*) FROM vsgt_test_collect(vsgt_req('dead1','graph_only_f')))>0,
                   'graph_only_f survives dead tuples');
SELECT vsgt_assert((SELECT count(*) FROM vsgt_test_collect(vsgt_req('dead2','graph_only_b')))>=0,
                   'graph_only_b survives dead tuples');
SELECT vsgt_assert((SELECT count(*) FROM vsgt_test_collect(vsgt_req('dead3','graph_only_f')))>0,
                   'repeat request survives dead tuples');
-- Streaming form with early close: the probe scan is ended by the consumer close path.
BEGIN;
DECLARE dead_cursor NO SCROLL CURSOR FOR
  SELECT result FROM tjs_vsgt(vsgt_req('dead4','graph_only_f'), NULL::vector);
FETCH 1 FROM dead_cursor;
CLOSE dead_cursor;
COMMIT;
SELECT vsgt_assert((SELECT array_agg(result->>'payload' IS NOT NULL)
                      FROM vsgt_test_collect(vsgt_req('dead5','graph_only_f')))
                   = ARRAY[true,true], 'payload of the live version is returned');
SELECT vsgt_assert((SELECT bool_and(result->'payload'->>'lesson' = 'v3')
                      FROM vsgt_test_collect(vsgt_req('dead6','graph_only_f'))),
                   'probe returns the newest tuple version');
SELECT 'vsgt_dead_tuple_test passed' AS result;
