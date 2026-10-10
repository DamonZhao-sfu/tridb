-- join_order 0.1.0 -> 0.2.0: semantic-operator placement (Figure-1 plan P4, ADR-0026).
-- ADDITIVE: the 0.1.0 functions (frozen core + cost path) are unchanged.
\echo Use "ALTER EXTENSION join_order UPDATE TO '0.2.0'" to load this file. \quit

-- Expensive-predicate (Hellerstein) order of predicate legs. 0-based indices, descending
-- rank (1-sel)/c_eff with c_eff = cost*(1-hit) + cache_cost*hit; c_eff = 0 ranks first;
-- ties keep input order. Bit-identical to src/planner/join_order_sem_ref.py:place_sem.
CREATE FUNCTION tridb_place_sem(sels float8[], costs float8[], hits float8[],
                                cache_cost float8 DEFAULT 0.0)
	RETURNS int[]
	AS 'MODULE_PATHNAME', 'tridb_place_sem_sql'
	LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

-- Expected cost of an order: n_in * sum_pos c_eff[ord[pos]] * prod_{q<pos} sels[ord[q]].
CREATE FUNCTION tridb_sem_plan_cost(sels float8[], costs float8[], hits float8[], ord int[],
                                    n_in float8, cache_cost float8 DEFAULT 0.0)
	RETURNS float8
	AS 'MODULE_PATHNAME', 'tridb_sem_plan_cost_sql'
	LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;

-- FETCH size for a streaming semantic filter over a tjs_vsgt portal:
-- clamp(ceil(k_remaining / max(sel_hat, 1e-6)), min_batch, max_batch); k_remaining <= 0 -> 0.
CREATE FUNCTION tridb_sem_batch(k_remaining int, sel_hat float8, min_batch int, max_batch int)
	RETURNS int
	AS 'MODULE_PATHNAME', 'tridb_sem_batch_sql'
	LANGUAGE C STRICT IMMUTABLE PARALLEL SAFE;
