-- VSGT native stream-prefix interface; legacy tjs_open is unchanged.
CREATE TABLE public.tjs_vsgt_anchors (
    scope_id text NOT NULL,
    task_uid text NOT NULL,
    anchor_set_id text NOT NULL,
    ordinal bigint NOT NULL CHECK (ordinal >= 0),
    node_id bigint NOT NULL,
    PRIMARY KEY (scope_id, task_uid, anchor_set_id, ordinal)
);
COMMENT ON TABLE public.tjs_vsgt_anchors IS
    'Frozen non-vector anchor manifests, not graph topology. Query-time selection is native.';
SELECT pg_catalog.pg_extension_config_dump('public.tjs_vsgt_anchors','');
GRANT SELECT ON public.tjs_vsgt_anchors TO PUBLIC;
CREATE FUNCTION public.tjs_vsgt_support(internal) RETURNS internal
AS 'MODULE_PATHNAME','tjs_vsgt_support' LANGUAGE C STRICT;
CREATE FUNCTION public.tjs_vsgt(request jsonb, query vector DEFAULT NULL)
RETURNS TABLE (result jsonb)
AS 'MODULE_PATHNAME','tjs_vsgt' LANGUAGE C VOLATILE PARALLEL UNSAFE
SUPPORT public.tjs_vsgt_support;
CREATE FUNCTION public.tjs_vsgt_stats(request_id text) RETURNS jsonb
AS 'MODULE_PATHNAME','tjs_vsgt_stats' LANGUAGE C VOLATILE STRICT PARALLEL UNSAFE;
