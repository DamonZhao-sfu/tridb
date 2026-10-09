/* Versioned in-backend ABI; no shared-object static link or second store. */
#ifndef TRIDB_GPH_PULL_API_H
#define TRIDB_GPH_PULL_API_H
#include "postgres.h"
#include "utils/snapshot.h"
#define GPH_PULL_API_NAME "tridb.graph.pull.v1"
typedef struct GphPullApi
{
    uint32 version;
    void *(*open)(int64 src, uint32 type, Snapshot snapshot);
    /* 1 visible matching edge, -1 inspected nonmatching slot, 0 exhausted. */
    int (*next)(void *scan, int64 *dst);
    void (*close)(void *scan);
} GphPullApi;
#endif
