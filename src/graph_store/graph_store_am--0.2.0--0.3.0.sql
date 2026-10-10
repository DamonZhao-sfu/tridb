/* graph_store_am 0.2.0 -> 0.3.0 upgrade.
 *
 *   ALTER EXTENSION graph_store_am UPDATE TO '0.3.0';
 *
 * Lockstep version bump only: the release carries ONE version across both extension
 * .control files (tests/test_release_coherence.py), and tjs_pg moved to 0.3.0 (the VSGT
 * stream-prefix interface, tjs_pg--0.2.0--0.3.0.sql). graph_store_am has no SQL object or
 * shared-library change between 0.2.0 and 0.3.0, hence the empty DDL body below.
 */
\echo Use "ALTER EXTENSION graph_store_am UPDATE TO '0.3.0'" to load this file. \quit
