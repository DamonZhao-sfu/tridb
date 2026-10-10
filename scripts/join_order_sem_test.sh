#!/usr/bin/env bash
#
# join_order_sem_test.sh — build + test semantic-operator placement (Figure-1 P4, ADR-0026)
# against a RUNNING stock PostgreSQL (16/17; no Docker, no fork image).
#
#   1. PGXS-build + install src/planner (the join_order extension, now 0.2.0).
#   2. Create a throwaway database, run test/join_order_sem_test.sql (pinned + hand cases,
#      validation errors).
#   3. Parity: generate a seeded random grid in Python, evaluate it through
#      src/planner/join_order_sem_ref.py AND through SQL, require EXACT equality (float8 compared
#      bit-for-bit after a round-trip through PG's shortest-exact text output).
#   4. Drop the database.
#
# Env: PG_CONFIG (default: pg_config on PATH), PGHOST (127.0.0.1), PGPORT (5432), PGUSER,
#      DESTDIR (optional; e.g. a relocatable private prefix), PYTHON (python3),
#      TEST_DB (jo_sem_test), PARITY_CASES (2000).
#
# Usage: PG_CONFIG=/usr/bin/pg_config PGPORT=55440 DESTDIR=$PWD/.pgprefix16 \
#          scripts/join_order_sem_test.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PG_CONFIG="${PG_CONFIG:-pg_config}"
export PGHOST="${PGHOST:-127.0.0.1}" PGPORT="${PGPORT:-5432}"
PYTHON="${PYTHON:-python3}"
TEST_DB="${TEST_DB:-jo_sem_test}"
PARITY_CASES="${PARITY_CASES:-2000}"
BIN="$("$PG_CONFIG" --bindir)"
# Prefer the client next to the server if DESTDIR relocates the install; fall back to pg_config's.
PSQL="$BIN/psql"
[ -n "${DESTDIR:-}" ] && [ -x "${DESTDIR}${BIN}/psql" ] && PSQL="${DESTDIR}${BIN}/psql"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "=== make + install (join_order, $("$PG_CONFIG" --version)) ==="
make -C "$ROOT/src/planner" PG_CONFIG="$PG_CONFIG" >"$WORK/make.log" 2>&1 ||
  { tail -30 "$WORK/make.log"; echo "BUILD FAILED"; exit 1; }
if grep -E "join_order_sem\.c:[0-9]+:[0-9]+: warning" "$WORK/make.log"; then
  echo "BUILD WARNINGS in join_order_sem.c"; exit 1
fi
make -C "$ROOT/src/planner" PG_CONFIG="$PG_CONFIG" install ${DESTDIR:+DESTDIR="$DESTDIR"} \
  >"$WORK/install.log" 2>&1 || { tail -20 "$WORK/install.log"; echo "INSTALL FAILED"; exit 1; }

q() { "$PSQL" -X -v ON_ERROR_STOP=1 "$@"; }
q -d postgres -qc "DROP DATABASE IF EXISTS $TEST_DB" -qc "CREATE DATABASE $TEST_DB"
trap 'q -d postgres -qc "DROP DATABASE IF EXISTS $TEST_DB" >/dev/null 2>&1 || true; rm -rf "$WORK"' EXIT

echo "=== test/join_order_sem_test.sql ==="
q -d "$TEST_DB" -f "$ROOT/test/join_order_sem_test.sql"

echo "=== C <-> join_order_sem_ref.py parity ($PARITY_CASES random cases) ==="
"$PYTHON" - "$ROOT" "$WORK" "$PARITY_CASES" gen <<'PY'
import random, sys
root, work, n_cases = sys.argv[1], sys.argv[2], int(sys.argv[3])
sys.path.insert(0, f"{root}/src")
from planner.join_order_sem_ref import place_sem, sem_plan_cost, sem_batch  # noqa: E402

def fa(xs):
    return "'{" + ",".join(repr(float(x)) for x in xs) + "}'::float8[]"

def pick(rng, specials, lo, hi):
    return rng.choice(specials) if rng.random() < 0.3 else rng.uniform(lo, hi)

rng = random.Random(20261010)
lines, expect = [], []
for i in range(n_cases):
    n = rng.randint(0, 12)
    sels = [pick(rng, [0.0, 0.1, 0.5, 1.0], 0.0, 1.0) for _ in range(n)]
    costs = [pick(rng, [0.0, 1.0, 3.0], 0.0, 1e4) for _ in range(n)]
    hits = [pick(rng, [0.0, 0.25, 1.0], 0.0, 1.0) for _ in range(n)]
    cc = pick(rng, [0.0, 0.5], 0.0, 10.0)
    n_in = pick(rng, [0.0, 1.0, 1e6], 0.0, 1e9)
    perm = list(range(n))
    rng.shuffle(perm)
    k = rng.randint(-2, 10**6)
    sh = pick(rng, [0.0, 1e-7, 1.0], 0.0, 1.0)
    lo = rng.randint(1, 100)
    hi = rng.randint(lo, 10**7)
    o = place_sem(sels, costs, hits, cc)
    expect.append((o, sem_plan_cost(sels, costs, hits, perm, n_in, cc), sem_batch(k, sh, lo, hi)))
    pa = "'{" + ",".join(map(str, perm)) + "}'::int[]"
    lines.append(
        f"SELECT {i}, tridb_place_sem({fa(sels)},{fa(costs)},{fa(hits)},{cc!r}), "
        f"tridb_sem_plan_cost({fa(sels)},{fa(costs)},{fa(hits)},{pa},{n_in!r},{cc!r}), "
        f"tridb_sem_batch({k},{sh!r},{lo},{hi});"
    )
open(f"{work}/parity.sql", "w").write("SET extra_float_digits = 3;\n" + "\n".join(lines) + "\n")
import json
json.dump(expect, open(f"{work}/expect.json", "w"))
PY
q -d "$TEST_DB" -qAt -F '|' -f "$WORK/parity.sql" >"$WORK/parity.out"
"$PYTHON" - "$WORK" <<'PY'
import json, struct, sys
work = sys.argv[1]
expect = json.load(open(f"{work}/expect.json"))
rows = [r.split("|") for r in open(f"{work}/parity.out").read().splitlines() if r]
assert len(rows) == len(expect), f"row count {len(rows)} != {len(expect)}"
bad = 0
for (i, ord_s, cost_s, batch_s) in rows:
    o, cost, batch = expect[int(i)]
    got_ord = [int(x) for x in ord_s.strip("{}").split(",") if x]
    # extra_float_digits=3 -> PG prints the shortest exact round-trip form; compare bit patterns.
    same_cost = struct.pack("<d", float(cost_s)) == struct.pack("<d", float(cost))
    if got_ord != o or not same_cost or int(batch_s) != batch:
        bad += 1
        if bad <= 5:
            print(f"MISMATCH case {i}: C=({ord_s},{cost_s},{batch_s}) ref=({o},{cost!r},{batch})")
if bad:
    sys.exit(f"PARITY FAILED: {bad}/{len(rows)} cases differ")
print(f"PASS parity: {len(rows)} cases bit-identical (place_sem, sem_plan_cost, sem_batch)")
PY
echo "[join_order_sem_test] PASS — semantic placement C == join_order_sem_ref.py (Figure-1 P4)."
