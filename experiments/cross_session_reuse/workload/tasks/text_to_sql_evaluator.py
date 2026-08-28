"""Execute generated SQL on hidden SQLite data and compare result sets."""

from __future__ import annotations

import importlib.util
import sqlite3
import time
from pathlib import Path

from experiments.cross_session_reuse.calibration.ledger import append_attempt


CASES = (
    ("How many orders are there?", ((4,),)),
    ("What is the total revenue?", ((55.0,),)),
    ("Which customer has the largest total spend?", (("Ada", 30.0),)),
    ("List product names with at least two orders, alphabetically.", (("book",),)),
)


def _database() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        """
        CREATE TABLE customers(id INTEGER PRIMARY KEY, name TEXT NOT NULL);
        CREATE TABLE orders(
            id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            product TEXT NOT NULL,
            amount REAL NOT NULL
        );
        INSERT INTO customers VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
        INSERT INTO orders VALUES
            (1, 1, 'book', 10.0),
            (2, 1, 'gpu', 20.0),
            (3, 2, 'book', 10.0),
            (4, 3, 'pen', 15.0);
        """
    )
    return connection


def evaluate(program_path: str) -> dict[str, float | str]:
    started = time.monotonic()
    status = "valid"
    error_message = ""
    passed = 0
    try:
        spec = importlib.util.spec_from_file_location("csr_sql_candidate", program_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load candidate: {program_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        connection = _database()
        for question, expected in CASES:
            sql = module.generate_sql(question)
            if not isinstance(sql, str) or not sql.lstrip().lower().startswith("select"):
                continue
            try:
                actual = tuple(connection.execute(sql).fetchall())
            except sqlite3.Error:
                continue
            passed += int(actual == expected)
        connection.close()
        if passed != len(CASES):
            status = "invalid"
            error_message = f"{len(CASES) - passed} execution cases failed"
    except Exception as exc:
        status = "error"
        error_message = f"{type(exc).__name__}: {exc}"
    pass_rate = passed / len(CASES)
    output: dict[str, float | str] = {
        "combined_score": pass_rate,
        "execution_accuracy": pass_rate,
        "validity": float(passed == len(CASES)),
        "is_buggy": float(passed != len(CASES)),
        "eval_seconds": time.monotonic() - started,
        "status": status,
        "error_message": error_message,
    }
    append_attempt(
        program_path,
        metrics={key: value for key, value in output.items() if key != "error_message"},
        status=status,
        error_message=error_message,
    )
    return output


if __name__ == "__main__":
    print(evaluate(str(Path(__file__).with_name("text_to_sql_initial.py"))))

