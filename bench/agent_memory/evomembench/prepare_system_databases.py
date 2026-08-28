"""Prepare fresh native databases for the frozen EvoMemBench systems run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from psycopg import connect, sql

from bench.agent_memory.gem.memory import TriDBGovernedMemory


TJS_FUNCTIONS = (
    ("tjs_open", "tjs_open_pg", True),
    ("tjs_open_candidates_examined", "tjs_open_candidates_examined_pg", False),
    ("tjs_open_relational_examined", "tjs_open_relational_examined_pg", False),
    ("tjs_open_relational_passed", "tjs_open_relational_passed_pg", False),
    ("tjs_open_graph_examined", "tjs_open_graph_examined_pg", False),
    ("tjs_open_graph_reached", "tjs_open_graph_reached_pg", False),
    ("tjs_open_graph_censored", "tjs_open_graph_censored_pg", False),
    ("tjs_open_termination_reason", "tjs_open_termination_reason_pg", False),
    ("tjs_open_budget_capped", "tjs_open_budget_capped_pg", False),
    ("tjs_open_bridges_injected", "tjs_open_bridges_injected_pg", False),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rebind_tjs(connection: Any, library: Path) -> None:
    library_literal = sql.Literal(str(library.with_suffix("")))
    scalar_signatures = {
        "tjs_open_candidates_examined": "() RETURNS bigint",
        "tjs_open_relational_examined": "() RETURNS bigint",
        "tjs_open_relational_passed": "() RETURNS bigint",
        "tjs_open_graph_examined": "() RETURNS bigint",
        "tjs_open_graph_reached": "() RETURNS bigint",
        "tjs_open_graph_censored": "() RETURNS boolean",
        "tjs_open_termination_reason": "() RETURNS text",
        "tjs_open_budget_capped": "() RETURNS boolean",
        "tjs_open_bridges_injected": "() RETURNS bigint",
    }
    with connection.cursor() as cursor:
        cursor.execute(
            sql.SQL(
                "CREATE OR REPLACE FUNCTION tjs_open("
                "regclass,integer,integer,integer,integer,text,text,vector,bigint,integer"
                ") RETURNS SETOF bigint AS {}, 'tjs_open_pg' LANGUAGE C VOLATILE"
            ).format(library_literal)
        )
        for name, symbol, is_set in TJS_FUNCTIONS:
            if is_set:
                continue
            cursor.execute(
                sql.SQL(
                    "CREATE OR REPLACE FUNCTION {} {} AS {}, {} LANGUAGE C VOLATILE"
                ).format(
                    sql.Identifier(name),
                    sql.SQL(scalar_signatures[name]),
                    library_literal,
                    sql.Literal(symbol),
                )
            )


def prepare_database(*, label: str, dsn: str, library: Path | None) -> dict[str, Any]:
    memory = TriDBGovernedMemory.connect(dsn, dim=1024)
    try:
        initialized = memory.init_schema()
    finally:
        memory.close()
    with connect(dsn) as connection:
        if library is not None:
            _rebind_tjs(connection, library)
        with connection.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM gem_unit")
            units = int(cursor.fetchone()[0])
            cursor.execute(
                "SELECT proname, probin FROM pg_proc WHERE proname = ANY(%s)"
                " ORDER BY proname",
                ([name for name, _symbol, _is_set in TJS_FUNCTIONS],),
            )
            functions = {str(name): str(probin) for name, probin in cursor.fetchall()}
        if units != 0:
            raise RuntimeError(f"{label} database is not fresh: {units} GEM units")
        if len(functions) != len(TJS_FUNCTIONS):
            missing = sorted({row[0] for row in TJS_FUNCTIONS} - set(functions))
            raise RuntimeError(f"{label} database lacks TJS functions: {missing}")
        if library is not None and any(
            Path(value).resolve() != library.with_suffix("").resolve()
            for value in functions.values()
        ):
            raise RuntimeError(f"{label} TJS functions do not share the frozen library")
    return {
        "label": label,
        "gem_units": units,
        "schema": initialized,
        "tjs_functions": functions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-dsn", required=True)
    parser.add_argument("--scale-dsn", required=True)
    parser.add_argument("--tjs-library", type=Path)
    args = parser.parse_args()
    library = args.tjs_library.resolve() if args.tjs_library else None
    if library is not None and not library.is_file():
        raise FileNotFoundError(library)
    report = {
        "schema_version": "evomembench_database_preparation_v0.1.0",
        "tjs_library": (
            None
            if library is None
            else {"path": str(library), "sha256": _sha256(library)}
        ),
        "databases": [
            prepare_database(label="native", dsn=args.native_dsn, library=library),
            prepare_database(label="scale", dsn=args.scale_dsn, library=library),
        ],
        "status": "complete",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
