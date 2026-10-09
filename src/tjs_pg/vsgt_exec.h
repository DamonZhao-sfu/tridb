#ifndef TRIDB_VSGT_EXEC_H
#define TRIDB_VSGT_EXEC_H
#include "postgres.h"
#include "utils/jsonb.h"
#include "utils/snapshot.h"
typedef struct VsgtState VsgtState;
extern VsgtState *vsgt_begin(Jsonb *request, Datum vector, bool vector_null, Oid vector_type,
                            Snapshot snapshot, MemoryContext owner);
extern Jsonb *vsgt_next(VsgtState *state);
extern void vsgt_end(VsgtState *state);
extern Jsonb *vsgt_stats(const char *request_id);
extern void vsgt_init(void);
#endif
