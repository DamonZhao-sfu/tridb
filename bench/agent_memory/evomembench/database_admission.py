"""Read-only PostgreSQL concurrency admission for formal EvoMemBench runs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from typing import Any, Iterable

from psycopg import connect


def inspect_server(
    *, label: str, dsn: str, allowed_applications: Iterable[str] = ()
) -> dict[str, Any]:
    """Report active external database work without exposing query text."""
    allowed = frozenset(str(value) for value in allowed_applications)
    with connect(dsn, application_name="evomembench_db_monitor") as connection:
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT pid, coalesce(datname,''), coalesce(usename,''),"
                " coalesce(application_name,''), coalesce(state,''), backend_type,"
                " extract(epoch FROM (clock_timestamp()-query_start)),"
                " extract(epoch FROM (clock_timestamp()-xact_start)),"
                " coalesce(wait_event_type,''), coalesce(wait_event,'')"
                " FROM pg_stat_activity"
                " WHERE pid <> pg_backend_pid()"
                " AND (backend_type='autovacuum worker' OR"
                "      (backend_type='client backend' AND"
                "       (state IS DISTINCT FROM 'idle' OR xact_start IS NOT NULL)))"
                " ORDER BY pid"
            )
            observed = cursor.fetchall()
    external: list[dict[str, Any]] = []
    allowed_active: list[dict[str, Any]] = []
    observed_background: list[dict[str, Any]] = []
    for row in observed:
        payload = {
            "pid": int(row[0]),
            "database": str(row[1]),
            "user": str(row[2]),
            "application_name": str(row[3]),
            "state": str(row[4]),
            "backend_type": str(row[5]),
            "query_age_seconds": None if row[6] is None else float(row[6]),
            "transaction_age_seconds": None if row[7] is None else float(row[7]),
            "wait_event_type": str(row[8]),
            "wait_event": str(row[9]),
        }
        if payload["backend_type"] == "autovacuum worker":
            observed_background.append(payload)
        elif (
            payload["backend_type"] == "client backend"
            and payload["application_name"] in allowed
        ):
            allowed_active.append(payload)
        else:
            external.append(payload)
    return {
        "label": label,
        "passed": not external,
        "allowed_applications": sorted(allowed),
        "external_active_backends": external,
        "allowed_active_backends": allowed_active,
        "observed_background_backends": observed_background,
    }


def inspect_servers(
    servers: Iterable[tuple[str, str]], *, allowed_applications: Iterable[str] = ()
) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for label, dsn in servers:
        try:
            reports.append(
                inspect_server(
                    label=label,
                    dsn=dsn,
                    allowed_applications=allowed_applications,
                )
            )
        except Exception as exc:
            errors.append({"label": label, "error": f"{type(exc).__name__}: {exc}"})
    return {
        "schema_version": "evomembench_database_admission_v0.1.0",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "passed": not errors and all(report["passed"] for report in reports),
        "servers": reports,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server",
        action="append",
        required=True,
        metavar="LABEL=DSN",
        help="PostgreSQL server to inspect; repeat for native and baseline",
    )
    parser.add_argument("--allowed-application", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    servers: list[tuple[str, str]] = []
    for value in args.server:
        if "=" not in value:
            raise SystemExit("--server must be LABEL=DSN")
        label, dsn = value.split("=", 1)
        if not label or not dsn:
            raise SystemExit("--server must be LABEL=DSN")
        servers.append((label, dsn))
    report = inspect_servers(servers, allowed_applications=args.allowed_application)
    print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))
    if not report["passed"]:
        raise SystemExit(69)


if __name__ == "__main__":
    main()
