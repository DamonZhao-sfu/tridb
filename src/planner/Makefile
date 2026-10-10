# join_order — PGXS build. Build against the MSVBASE fork's pg_config:
#   make PG_CONFIG=/u01/app/postgres/product/13.4/bin/pg_config
#   make PG_CONFIG=... install
EXTENSION   = join_order
DATA        = join_order--0.1.0.sql join_order--0.1.0--0.2.0.sql join_order--0.2.0.sql
MODULE_big  = join_order
OBJS        = join_order.o join_order_legstats.o join_order_cost.o join_order_sem.o
PG_CONFIG  ?= pg_config
# legstats.c includes src/graph_store/gph_page.h to read GphMeta:
PG_CPPFLAGS += -I$(srcdir)/../graph_store
PGXS       := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

# Defense-in-depth: -ffast-math allows FP reassociation, which could change the IEEE division
# result and break the FROZEN boundary case (selectivity == threshold -> filter_first). The
# bit-identical-with-the-Python-reference contract requires strict IEEE; strip it if a future
# toolchain/PGXS ever injects it. (Linus review, DEV-1170.)
CFLAGS := $(filter-out -ffast-math,$(CFLAGS))


# join_order_sem.c (Figure-1 P4) is bit-identical to join_order_sem_ref.py only if no a*b+c is
# contracted into an FMA (GCC defaults to -ffp-contract=fast on ARM64/GX10). Scoped to that one
# object (and its JIT bitcode) so the frozen core's flags are untouched.
join_order_sem.o: CFLAGS += -ffp-contract=off
join_order_sem.bc: BITCODE_CFLAGS += -ffp-contract=off
