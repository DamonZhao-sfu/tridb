-- 0.3.0: SemExtract. sem_extract(question, input) -> short lower-case label, planned as
-- "Custom Scan (TriMEM SemExtract)" (ADR-0028). The worker commits the label into
-- trimem.sem_cache.extracted and wakes the query through complete_extracts (completion only).
\echo Use "ALTER EXTENSION trimem_pg UPDATE TO '0.3.0'" to load this file. \quit

ALTER TABLE trimem.sem_cache ADD COLUMN extracted text;

CREATE FUNCTION trimem.sem_extract_support(internal) RETURNS internal
AS 'MODULE_PATHNAME', 'trimem_sem_extract_support' LANGUAGE C STRICT;

-- Marker: planned as "Custom Scan (TriMEM SemExtract)"; direct evaluation fails closed.
CREATE FUNCTION trimem.sem_extract(question text, input text) RETURNS text
AS 'MODULE_PATHNAME', 'trimem_sem_extract' LANGUAGE C STABLE STRICT PARALLEL UNSAFE
SUPPORT trimem.sem_extract_support;

CREATE FUNCTION trimem.complete_extracts(slots integer[], seqs bigint[])
RETURNS integer AS 'MODULE_PATHNAME', 'trimem_complete_extracts' LANGUAGE C VOLATILE PARALLEL UNSAFE;
REVOKE ALL ON FUNCTION trimem.complete_extracts(integer[], bigint[]) FROM PUBLIC;
