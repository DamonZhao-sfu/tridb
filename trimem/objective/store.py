"""D0: token counts per attempt and the per-iteration outcome log."""

from __future__ import annotations

import glob
from functools import lru_cache

import psycopg

OUTCOME_DDL = """
CREATE TABLE IF NOT EXISTS trimem_outcome (
    id          bigserial PRIMARY KEY,
    session     text NOT NULL,
    task        text NOT NULL,
    iteration   integer NOT NULL,
    policy      text NOT NULL,
    candidate   text NOT NULL,          -- family:k chosen
    propensity  double precision,       -- probability the policy chose it
    features    jsonb NOT NULL,         -- feature vector of the chosen candidate
    candidates  jsonb,                  -- every candidate's features + scores (receipt)
    parent_fitness double precision,
    child_fitness  double precision,    -- filled by observe()
    delta       double precision,       -- max(0, child - parent)
    valid       boolean,
    tok_injected integer, tok_child integer, ret_tokens integer, ret_ms double precision,
    decided_at  timestamptz NOT NULL DEFAULT now(),
    observed_at timestamptz,
    UNIQUE (session, iteration)
);
ALTER TABLE trimem_outcome ADD COLUMN IF NOT EXISTS best_fitness double precision;  -- run best at decision
ALTER TABLE trimem_outcome ADD COLUMN IF NOT EXISTS delta_best double precision;    -- max(0, child - best)
ALTER TABLE trimem_outcome ADD COLUMN IF NOT EXISTS space text;                     -- v1 | v2
"""


@lru_cache(maxsize=1)
def tokenizer():
    from tokenizers import Tokenizer

    path = glob.glob(
        "/localhome/hza214/hf-cache/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*/tokenizer.json"
    )[0]
    return Tokenizer.from_file(path)


def count_tokens(text: str | None) -> int:
    return len(tokenizer().encode(text or "").ids)


def ensure(dsn: str) -> None:
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(OUTCOME_DDL)
        conn.execute("ALTER TABLE attempt ADD COLUMN IF NOT EXISTS tokens integer")


def fill_tokens(dsn: str, batch: int = 500) -> int:
    """Token count of every attempt's code (Qwen3.8 tokenizer), once per row."""
    n = 0
    with psycopg.connect(dsn, autocommit=True) as conn:
        while True:
            rows = conn.execute(
                "SELECT id, code FROM attempt WHERE tokens IS NULL AND code IS NOT NULL "
                "LIMIT %s",
                (batch,),
            ).fetchall()
            if not rows:
                return n
            with conn.transaction():
                for i, code in rows:
                    conn.execute(
                        "UPDATE attempt SET tokens = %s WHERE id = %s",
                        (count_tokens(code), i),
                    )
            n += len(rows)
