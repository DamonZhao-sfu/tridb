# PGXS build for the TriMEM semantic-predicate CustomScan (ADR-0027). Stock PG 16/17.
# Must be listed in shared_preload_libraries AFTER tjs_pg (mailbox shmem + hook order).
MODULE_big = trimem_pg
OBJS = trimem_pg.o trimem_path.o trimem_exec.o trimem_graph.o trimem_score.o trimem_extract.o
EXTENSION = trimem_pg
DATA = trimem_pg--0.1.0.sql trimem_pg--0.1.0--0.2.0.sql trimem_pg--0.2.0.sql \
       trimem_pg--0.2.0--0.3.0.sql trimem_pg--0.3.0.sql
PGFILEDESC = "trimem_pg - TriMEM SemFilter CustomScan"
PG_CFLAGS = -Wall -Wmissing-prototypes
PG_CPPFLAGS = -I$(CURDIR)/../graph_store

PG_CONFIG ?= pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
