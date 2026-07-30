  cd /localhome/hza214/tridb
  make gem-demo

  To inspect the PostgreSQL tables and their schemas:

  psql postgresql://hza214@127.0.0.1:55432/gem_demo \
    -c '\dt gem_*' \
    -c '\d+ gem_unit' \
    -c '\d+ gem_field_value' \
    -c '\d+ gem_edge' \
    -c '\d+ gem_transition'

  To view sample demo rows:

  psql postgresql://hza214@127.0.0.1:55432/gem_demo -x \
    -c "SELECT id, title, state, salience, access_count, metadata
        FROM gem_unit
        WHERE scope_id = 'gem-wiki-demo'
        ORDER BY id
        LIMIT 10;"

  For an interactive PostgreSQL session:

  psql postgresql://hza214@127.0.0.1:55432/gem_demo

  Then use \dt gem_*, \d+ gem_unit, and \q to exit.