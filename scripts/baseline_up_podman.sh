#!/usr/bin/env bash
#
# baseline_up_podman.sh — bring up the DEV-1171 multi-store baseline (Milvus + Neo4j +
# pgvector) under ROOTLESS PODMAN, for boxes where Docker is not installable.
#
# WHY THIS EXISTS ALONGSIDE baseline/docker-compose.yml
# -----------------------------------------------------
# The compose file stays the source of truth for what the baseline IS (images, versions,
# credentials, volume layout). This script exists because two things about it cannot be
# honoured on a no-root workstation:
#
#   1. NO DOCKER DAEMON. Installing dockerd needs root. Rootless podman does not, given
#      subordinate id ranges in /etc/subuid + /etc/subgid (see PREREQUISITES). NOTE that
#      `usermod --add-subuids` only edits LOCAL users -- on an LDAP/AD-joined box the
#      range must be appended to those two files directly, as root, once.
#
#   2. HOST NETWORKING IS REQUIRED FOR AN HONEST LATENCY NUMBER. Rootless podman forwards
#      published ports through a userspace proxy (pasta/slirp4netns). That proxy sits in
#      the request path and adds latency to EVERY baseline round-trip -- and the baseline's
#      per-query cost is precisely what bench/wiki_fusion.py measures against TriDB's single
#      in-process call. Publishing ports would inflate the multi-store side and manufacture
#      a TriDB win. So every service runs with --network=host and a distinct host port,
#      exactly as loopback-co-located processes would. The compose file's `ports:` mappings
#      are therefore intentionally not reproduced here.
#
# The port defaults below match bench/wiki_fusion.py's CLI flags rather than the compose
# file's, EXCEPT Milvus: it stays on its own default 19530 (remapping the standalone proxy
# port needs a config-file edit, and the harness takes --milvus-port anyway).
#
# Usage: scripts/baseline_up_podman.sh {up|down|reset|status|logs <svc>}
#
# `reset` exists because the volume files are owned by MAPPED subordinate uids (200000+),
# not by you -- a plain `rm -rf baseline/volumes` fails with EPERM on every file. Deleting
# them requires re-entering the user namespace (`podman unshare`), where those uids are
# yours. This is the single most common operational surprise of the rootless path.
#
# PREREQUISITES (one-time, the only part needing root):
#   sudo apt-get install -y uidmap
#   echo "$(id -un):200000:65536" | sudo tee -a /etc/subuid /etc/subgid
#   podman system migrate
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOL="${BASELINE_VOLUMES:-$ROOT/baseline/volumes}"
PODMAN="${PODMAN:-podman}"

# Images are pinned to the same tags as baseline/docker-compose.yml. Keep them in sync:
# a version drift between the two paths is a silent comparability break.
IMG_NEO4J="docker.io/library/neo4j:5.20"
IMG_ETCD="quay.io/coreos/etcd:v3.5.5"
IMG_MINIO="docker.io/minio/minio:RELEASE.2023-03-20T20-16-18Z"
IMG_MILVUS="docker.io/milvusdb/milvus:v2.4.5"
# pgvector/pgvector:pg16 == postgres:16 + the pgvector files; see the comment on the
# postgres service in baseline/docker-compose.yml for why plain postgres:16 is not enough.
IMG_PG="docker.io/pgvector/pgvector:pg16"

NEO4J_BOLT_PORT="${NEO4J_BOLT_PORT:-7688}"
NEO4J_HTTP_PORT="${NEO4J_HTTP_PORT:-7475}"
ETCD_PORT="${ETCD_PORT:-2379}"
MINIO_PORT="${MINIO_PORT:-9000}"
MINIO_CONSOLE_PORT="${MINIO_CONSOLE_PORT:-9001}"
MILVUS_PORT="${MILVUS_PORT:-19530}"
MILVUS_HEALTH_PORT="${MILVUS_HEALTH_PORT:-9091}"
PG_PORT="${PG_PORT:-5434}"
PG_DB="${PG_DB:-tridb_wiki}"

# Dev-only credentials, identical to the compose file's. Loopback-bound by construction
# (host networking on a workstation); replace before any shared-host use.
NEO4J_PASSWORD="${NEO4J_PASSWORD:-testpassword}"

log() { printf '\033[1;34m[baseline]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[baseline] FATAL:\033[0m %s\n' "$*" >&2; exit 1; }

check_prereqs() {
  command -v "$PODMAN" >/dev/null 2>&1 || die "podman not on PATH (set PODMAN=/path/to/podman)"
  grep -q "^$(id -un):" /etc/subuid 2>/dev/null || die \
    "no /etc/subuid range for $(id -un) -- rootless podman falls back to single-uid mapping and
   images that run as a non-root uid (neo4j 7474, postgres 999) fail at chown. See PREREQUISITES."
  command -v newuidmap >/dev/null 2>&1 || die "newuidmap missing -- apt-get install uidmap"
}

wait_for() {  # wait_for <label> <seconds> <command...>
  local label="$1" limit="$2"; shift 2
  local waited=0
  until "$@" >/dev/null 2>&1; do
    waited=$((waited + 5))
    [ "$waited" -ge "$limit" ] && { log "$label: NOT healthy after ${limit}s"; return 1; }
    sleep 5
  done
  log "$label: healthy (${waited}s)"
}

# Milvus readiness, the honest version. /healthz answers OK as soon as the metrics HTTP
# server binds -- BEFORE the proxy can serve a gRPC request -- so gating on it hands the
# harness a collection-not-loadable error at the first real call. Probe the actual gRPC
# surface the harness uses (pymilvus list_collections) whenever a python with pymilvus is
# available, and fall back to /healthz only when it is not (announced, not silent).
PY_PROBE="${PY_PROBE:-$ROOT/.venv/bin/python}"

milvus_grpc_ready() {
  "$PY_PROBE" -W ignore -c "
import sys
from pymilvus import connections, utility
connections.connect(alias='probe', host='localhost', port='$MILVUS_PORT')
utility.list_collections(using='probe')
" >/dev/null 2>&1
}

wait_for_milvus() {
  if [ -x "$PY_PROBE" ] && "$PY_PROBE" -c "import pymilvus" >/dev/null 2>&1; then
    wait_for milvus 300 milvus_grpc_ready
  else
    log "milvus: no pymilvus at $PY_PROBE -- falling back to /healthz, which is READY-EARLY."
    log "milvus: the first harness call may still race the proxy. Set PY_PROBE to fix."
    wait_for milvus 300 curl -sf "http://127.0.0.1:$MILVUS_HEALTH_PORT/healthz"
  fi
}

up() {
  check_prereqs
  mkdir -p "$VOL"/neo4j/{data,logs} "$VOL"/milvus/{etcd,minio,data} "$VOL"/postgres/data

  log "neo4j  -> bolt :$NEO4J_BOLT_PORT  http :$NEO4J_HTTP_PORT"
  $PODMAN run -d --name tridb-baseline-neo4j --network=host \
    -e NEO4J_AUTH="neo4j/$NEO4J_PASSWORD" \
    -e NEO4J_server_bolt_listen__address=":$NEO4J_BOLT_PORT" \
    -e NEO4J_server_http_listen__address=":$NEO4J_HTTP_PORT" \
    -v "$VOL/neo4j/data:/data" -v "$VOL/neo4j/logs:/logs" \
    "$IMG_NEO4J" >/dev/null

  log "etcd   -> :$ETCD_PORT"
  $PODMAN run -d --name tridb-baseline-etcd --network=host \
    -e ETCD_AUTO_COMPACTION_MODE=revision -e ETCD_AUTO_COMPACTION_RETENTION=1000 \
    -e ETCD_QUOTA_BACKEND_BYTES=4294967296 -e ETCD_SNAPSHOT_COUNT=50000 \
    -v "$VOL/milvus/etcd:/etcd" \
    "$IMG_ETCD" etcd "-advertise-client-urls=http://127.0.0.1:$ETCD_PORT" \
    "-listen-client-urls=http://0.0.0.0:$ETCD_PORT" --data-dir=/etcd >/dev/null

  log "minio  -> :$MINIO_PORT  console :$MINIO_CONSOLE_PORT"
  $PODMAN run -d --name tridb-baseline-minio --network=host \
    -e MINIO_ACCESS_KEY=minioadmin -e MINIO_SECRET_KEY=minioadmin \
    -v "$VOL/milvus/minio:/minio_data" \
    "$IMG_MINIO" server /minio_data --console-address ":$MINIO_CONSOLE_PORT" >/dev/null

  log "pgvector -> :$PG_PORT  db $PG_DB"
  $PODMAN run -d --name tridb-baseline-postgres --network=host \
    -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB="$PG_DB" \
    -e PGDATA=/var/lib/postgresql/data/pgdata \
    -v "$VOL/postgres/data:/var/lib/postgresql/data" \
    "$IMG_PG" -c port="$PG_PORT" >/dev/null

  wait_for etcd  120 curl -sf "http://127.0.0.1:$ETCD_PORT/health"
  wait_for minio 120 curl -sf "http://127.0.0.1:$MINIO_PORT/minio/health/live"

  # Milvus only after its two deps are healthy (the compose file's depends_on condition).
  log "milvus -> :$MILVUS_PORT  health :$MILVUS_HEALTH_PORT"
  $PODMAN run -d --name tridb-baseline-milvus --network=host \
    -e ETCD_ENDPOINTS="localhost:$ETCD_PORT" -e MINIO_ADDRESS="localhost:$MINIO_PORT" \
    -v "$VOL/milvus/data:/var/lib/milvus" \
    "$IMG_MILVUS" milvus run standalone >/dev/null

  wait_for neo4j    180 bash -c "</dev/tcp/127.0.0.1/$NEO4J_BOLT_PORT"
  wait_for pgvector 120 pg_isready -h 127.0.0.1 -p "$PG_PORT" -U postgres
  wait_for_milvus

  log "up. harness flags:  --neo4j-uri bolt://localhost:$NEO4J_BOLT_PORT" \
      "--milvus-port $MILVUS_PORT --pg-port $PG_PORT"
  log "NOTE: 'CREATE EXTENSION vector' in $PG_DB is the loader's job, not the image's."
}

down() {
  for c in tridb-baseline-milvus tridb-baseline-minio tridb-baseline-etcd \
           tridb-baseline-neo4j tridb-baseline-postgres; do
    $PODMAN rm -f "$c" >/dev/null 2>&1 || true
  done
  log "down (volumes under $VOL kept -- delete them by hand to reload from scratch)"
}

reset() {
  down
  # The volume files belong to mapped subuids; only inside the user namespace are they
  # ours to delete. Without `podman unshare` every unlink is EPERM.
  log "deleting $VOL inside the user namespace"
  $PODMAN unshare rm -rf "$VOL"
  log "reset: volumes gone -- the next 'up' loads from scratch"
}

status() {
  $PODMAN ps -a --filter name=tridb-baseline --format "{{.Names}}\t{{.Status}}"
}

case "${1:-}" in
  up)     up ;;
  down)   down ;;
  reset)  reset ;;
  status) status ;;
  logs)   $PODMAN logs "${2:?usage: $0 logs <container>}" ;;
  *) die "usage: $0 {up|down|reset|status|logs <container>}" ;;
esac
