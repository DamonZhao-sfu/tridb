"""Plan or create an isolated same-server ScienceWorld memory database clone.

The source is never modified. A real copy is only attempted with --commit,
an idle source and a limited cgroup. Profiling has a separate execution gate.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path

import psycopg
from psycopg import sql

from tools.scienceworld_ablation_resource_preflight import collect
from tools.scienceworld_ablation_runtime import verify_current_cgroup
from trimem.experiments import scienceworld_trace_ablation as ab

SOURCE_DATABASE = "trimem_agentenv"
ADMIN_DSN = "host=/tmp/trimem-traj port=55450 dbname=postgres"


def validate_clone_admission(
    target,
    *,
    source_bytes,
    free_bytes,
    max_source_bytes,
    min_free_bytes,
    active_connections,
    target_exists,
):
    if not re.fullmatch(r"trace_ablation_sci_[a-f0-9]{8,32}", target):
        raise ValueError(
            "clone target must be a uniquely named isolated ScienceWorld database"
        )
    if active_connections != 0:
        raise ValueError("source memory database has active or paused connections")
    if target_exists:
        raise ValueError("clone target already exists; inspect it before any retry")
    if (
        any(
            type(x) is not int or x < 0
            for x in (source_bytes, free_bytes, max_source_bytes, min_free_bytes)
        )
        or source_bytes == 0
        or source_bytes > max_source_bytes
        or free_bytes < min_free_bytes + 2 * source_bytes
    ):
        raise ValueError("source size or free disk does not satisfy clone limits")
    return True


def verify_clone_database_state(state, receipt):
    """Reject a changed, writable-by-default or publicly reachable clone."""
    if (
        not re.fullmatch(
            r"trace_ablation_sci_[a-f0-9]{8,32}", str(state.get("name", ""))
        )
        or state.get("name") != receipt.get("target_database")
        or type(state.get("oid")) is not int
        or state["oid"] != receipt.get("target_oid")
        or state.get("allow_connections") is not True
        or state.get("read_only_default") is not True
        or state.get("public_connect") is not False
        or receipt.get("read_only_default") is not True
        or receipt.get("public_connect_revoked") is not True
    ):
        raise ValueError("isolated memory clone identity or access policy differs")
    return True


def inspect_clone_database_state(connection, target):
    row = connection.execute(
        "SELECT d.datname, d.oid::int, d.datallowconn, "
        "EXISTS (SELECT 1 FROM pg_db_role_setting s WHERE s.setdatabase=d.oid AND s.setrole=0 "
        "AND 'default_transaction_read_only=on'=ANY(s.setconfig)), "
        "EXISTS (SELECT 1 FROM aclexplode(COALESCE(d.datacl,acldefault('d',d.datdba))) a "
        "WHERE a.grantee=0 AND a.privilege_type='CONNECT') "
        "FROM pg_database d WHERE d.datname=%s",
        (target,),
    ).fetchone()
    if not row:
        raise ValueError("isolated memory clone is missing")
    return dict(
        zip(
            ("name", "oid", "allow_connections", "read_only_default", "public_connect"),
            row,
        )
    )


def verify_campaign_clone(root, manifest, snapshot, env):
    """Bind the live isolated database to the finalized identity and clone receipt."""
    root = Path(root).resolve()
    if not root.name.startswith("scienceworld_trace_ablation_"):
        raise ValueError("clone requires an isolated campaign root")
    from tools.sci_embedding_guard import verify_embedding_service

    verify_embedding_service(root)
    name = manifest.get("memory_database")
    ab.validate_isolated_memory_database(name, env)
    ab.verify_isolated_memory_evidence(snapshot, manifest["identity_evidence"], name)
    evidence = json.loads(
        Path(manifest["identity_evidence"]["target"]["path"]).read_text()
    )
    raw = (root / "memory_clone_receipt.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != evidence["_verification"].get(
        "clone_receipt_sha256"
    ):
        raise ValueError("memory clone receipt differs from frozen identity")
    receipt = json.loads(raw)
    if receipt.get("target_database") != name:
        raise ValueError("memory clone receipt names a different database")
    with psycopg.connect(
        ADMIN_DSN + " options='-c default_transaction_read_only=on'"
    ) as connection:
        verify_clone_database_state(
            inspect_clone_database_state(connection, name), receipt
        )
    return receipt


def inspect_clone_admission(connection, target, *, max_source_bytes, min_free_bytes):
    db = connection.execute(
        "SELECT oid, pg_database_size(oid) FROM pg_database WHERE datname=%s",
        (SOURCE_DATABASE,),
    ).fetchone()
    if not db:
        raise ValueError("source memory database is missing")
    active = connection.execute(
        "SELECT count(*) FROM pg_stat_activity WHERE datname=%s", (SOURCE_DATABASE,)
    ).fetchone()[0]
    exists = connection.execute(
        "SELECT EXISTS(SELECT 1 FROM pg_database WHERE datname=%s)", (target,)
    ).fetchone()[0]
    data_dir = connection.execute("SHOW data_directory").fetchone()[0]
    free = shutil.disk_usage(data_dir).free
    validate_clone_admission(
        target,
        source_bytes=db[1],
        free_bytes=free,
        max_source_bytes=max_source_bytes,
        min_free_bytes=min_free_bytes,
        active_connections=active,
        target_exists=exists,
    )
    return {
        "source_database": SOURCE_DATABASE,
        "source_oid": db[0],
        "source_bytes": db[1],
        "target_database": target,
        "free_bytes": free,
        "active_connections": active,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--system-limits", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--max-source-bytes", type=int, default=16 * 1024**3)
    parser.add_argument("--min-free-bytes", type=int, default=32 * 1024**3)
    parser.add_argument("--commit", action="store_true")
    args = parser.parse_args()
    ab.check_global_quota()
    ab.validate_ablation_environment(os.environ)
    root = args.campaign_root.resolve()
    if not root.name.startswith("scienceworld_trace_ablation_"):
        raise ValueError("clone evidence requires an isolated ablation campaign")
    limits = json.loads(args.system_limits.read_text())
    ab.validate_system_limits(limits)
    with psycopg.connect(ADMIN_DSN, autocommit=True) as conn:
        plan = inspect_clone_admission(
            conn,
            args.target,
            max_source_bytes=args.max_source_bytes,
            min_free_bytes=args.min_free_bytes,
        )
        if not args.commit:
            print(
                json.dumps(
                    {"status": "validated_read_only", **plan, "database_created": False}
                )
            )
            return 0
        cgroup = verify_current_cgroup(limits)["cgroup"]
        ab.validate_resource_preflight(collect(cgroup, ["0", "1"]), limits)
        monitor = ab.RuntimeResourceMonitor(limits)
        monitor.check()
        backend_pid = conn.execute("SELECT pg_backend_pid()").fetchone()[0]
        if Path(f"/proc/{backend_pid}").stat().st_uid != os.getuid():
            raise ValueError("cannot limit the dedicated clone backend safely")
        os.setpriority(os.PRIO_PROCESS, backend_pid, 10)
        subprocess.run(
            ["ionice", "-c", "2", "-n", "7", "-p", str(backend_pid)],
            check=True,
            timeout=5,
        )
        conn.execute("SET lock_timeout = '1s'")
        inspect_clone_admission(
            conn,
            args.target,
            max_source_bytes=args.max_source_bytes,
            min_free_bytes=args.min_free_bytes,
        )
        attempt = {**plan, "backend_pid": backend_pid, "started_wall_s": time.time()}
        raw = json.dumps(attempt, sort_keys=True, indent=2).encode() + b"\n"
        ab._install_verified_bytes(
            raw, root / "memory_clone_attempt.json", hashlib.sha256(raw).hexdigest()
        )
        statement = sql.SQL(
            "CREATE DATABASE {} TEMPLATE {} STRATEGY WAL_LOG ALLOW_CONNECTIONS false"
        ).format(sql.Identifier(args.target), sql.Identifier(SOURCE_DATABASE))
        last_resource_check = time.monotonic()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(conn.execute, statement)
            while not future.done():
                try:
                    future.result(timeout=5)
                except TimeoutError:
                    try:
                        ab.check_global_quota()
                        if time.monotonic() - last_resource_check >= 30:
                            monitor.check()
                            last_resource_check = time.monotonic()
                    except SystemExit:
                        conn.cancel_safe(timeout=5)
                        raise
            future.result()
        ab.check_global_quota()
        conn.execute(
            sql.SQL("REVOKE CONNECT ON DATABASE {} FROM PUBLIC").format(
                sql.Identifier(args.target)
            )
        )
        conn.execute(
            sql.SQL("ALTER DATABASE {} SET default_transaction_read_only = on").format(
                sql.Identifier(args.target)
            )
        )
        conn.execute(
            sql.SQL("ALTER DATABASE {} ALLOW_CONNECTIONS true").format(
                sql.Identifier(args.target)
            )
        )
        target_state = inspect_clone_database_state(conn, args.target)
        receipt = {
            **attempt,
            "completed_wall_s": time.time(),
            "read_only_default": True,
            "public_connect_revoked": True,
            "target_oid": target_state["oid"],
            "memory_fingerprint": None,
            "note": "Physical clone in same PostgreSQL instance; fingerprint and full runtime attestation still required.",
        }
        verify_clone_database_state(target_state, receipt)
        raw = json.dumps(receipt, sort_keys=True, indent=2).encode() + b"\n"
        ab._install_verified_bytes(
            raw, root / "memory_clone_receipt.json", hashlib.sha256(raw).hexdigest()
        )
    print(
        json.dumps(
            {
                "status": "cloned_unverified",
                "target_database": args.target,
                "memory_fingerprint": None,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
