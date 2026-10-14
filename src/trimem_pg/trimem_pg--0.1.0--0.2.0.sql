-- 0.2.0: soft semantic operators. sem_filter(predicate, input, unknown_as) keeps or drops rows
-- whose input is NULL; sem_score(predicate, input) -> [0, 1] is planned as TriMEM SemScore and
-- used to weight a ranking instead of filtering.
\echo Use "ALTER EXTENSION trimem_pg UPDATE TO '0.2.0'" to load this file. \quit

ALTER TABLE trimem.sem_cache ALTER COLUMN verdict DROP NOT NULL;
ALTER TABLE trimem.sem_cache ADD COLUMN score double precision;

CREATE FUNCTION trimem.sem_filter(predicate text, input text, unknown_as boolean) RETURNS boolean
AS 'MODULE_PATHNAME', 'trimem_sem_filter' LANGUAGE C STABLE PARALLEL UNSAFE
SUPPORT trimem.sem_filter_support;

CREATE FUNCTION trimem.sem_score_support(internal) RETURNS internal
AS 'MODULE_PATHNAME', 'trimem_sem_score_support' LANGUAGE C STRICT;

CREATE FUNCTION trimem.sem_score(predicate text, input text) RETURNS double precision
AS 'MODULE_PATHNAME', 'trimem_sem_score' LANGUAGE C STABLE STRICT PARALLEL UNSAFE
SUPPORT trimem.sem_score_support;

CREATE FUNCTION trimem.complete_scores(slots integer[], seqs bigint[], scores double precision[])
RETURNS integer AS 'MODULE_PATHNAME', 'trimem_complete_scores' LANGUAGE C VOLATILE PARALLEL UNSAFE;
REVOKE ALL ON FUNCTION trimem.complete_scores(integer[], bigint[], double precision[]) FROM PUBLIC;
