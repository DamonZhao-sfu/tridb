"""Dense ``config x query`` outcome matrix of a session-BO experiment (training panel only).

Every configuration the BO measured ran the full frozen panel, so the profiling store already holds
a dense matrix of per-query outcomes: quality (ALFWorld: won 0/1; ScienceWorld: score/100), paired
gain, total tokens and end-to-end latency. Only rows with ``split = 'train'`` of the given BO
experiment are accepted; a configuration enters the matrix only with exactly one evaluated row per
panel query. NoMemory (the paired baseline, official evaluator) can be added as a column from its
per-item result files and call log.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trimem.optimizer import session_space_agentenv as space

ROW_SQL = (
    "SELECT experiment_id, dataset, split, query_id, status, config, protocol, quality, paired_gain, "
    "total_tokens, e2e_ms, query_context FROM opt.session_retrieval "
    "WHERE experiment_id = %s AND split = 'train' AND status = 'evaluated'"
)


@dataclass(frozen=True)
class OutcomeMatrix:
    query_ids: tuple[str, ...]
    task_keys: tuple[str, ...]
    configs: tuple[dict, ...]
    quality: np.ndarray  # (Q, C)
    tokens: np.ndarray  # (Q, C)
    latency_ms: np.ndarray  # (Q, C)
    baseline: np.ndarray  # (Q,) NoMemory quality of each query

    @property
    def gain(self) -> np.ndarray:
        return self.quality - self.baseline[:, None]

    def subset(self, rows: np.ndarray) -> OutcomeMatrix:
        rows = np.asarray(rows)
        return OutcomeMatrix(
            tuple(self.query_ids[i] for i in rows),
            tuple(self.task_keys[i] for i in rows),
            self.configs,
            self.quality[rows],
            self.tokens[rows],
            self.latency_ms[rows],
            self.baseline[rows],
        )


def fetch_rows(dsn: str, experiment: str) -> list[dict]:
    import psycopg
    from psycopg.rows import dict_row

    with psycopg.connect(dsn, row_factory=dict_row) as c:
        return c.execute(ROW_SQL, (experiment,)).fetchall()


def build(rows: list[dict], *, panel_ids: list[str], experiment: str) -> OutcomeMatrix:
    """Rows of one BO experiment -> dense matrix over the configurations with a complete panel."""
    expected = [str(q) for q in panel_ids]
    if len(set(expected)) != len(expected):
        raise ValueError("panel ids must be distinct")
    groups: dict[str, dict[str, dict]] = defaultdict(dict)
    keys: dict[str, str] = {}
    base: dict[str, float] = {}
    for r in rows:
        if r["split"] != "train" or r["experiment_id"] != experiment:
            raise ValueError(
                "router training accepts only training rows of the BO experiment"
            )
        if r["status"] != "evaluated":
            continue
        if r["protocol"].get("space_version") != space.VERSION:
            raise ValueError("row profiled under a different space version")
        q = str(r["query_id"])
        if q not in set(expected):
            raise ValueError(f"query {q} outside the frozen panel")
        k = space.key(r["protocol"]["space_config"])
        if q in groups[k]:
            raise ValueError(f"duplicate evaluated row for {q} / {k}")
        for f in ("quality", "paired_gain", "total_tokens", "e2e_ms"):
            if r[f] is None or not math.isfinite(float(r[f])):
                raise ValueError(f"missing {f} for {q}")
        if float(r["total_tokens"]) <= 0 or float(r["e2e_ms"]) <= 0:
            raise ValueError(
                f"non-positive cost for {q} (log-cost models need tokens > 0 and latency > 0)"
            )
        groups[k][q] = r
        keys[q] = (r.get("query_context") or {}).get("task_key", q)
        b = float(r["quality"]) - float(r["paired_gain"])
        if q in base and abs(base[q] - b) > 1e-9:
            raise ValueError(f"inconsistent NoMemory baseline for {q}")
        base[q] = b
    complete = sorted(k for k, g in groups.items() if set(g) == set(expected))
    if not complete:
        raise ValueError("no configuration covers the full panel")
    cfgs = tuple(json.loads(k) for k in complete)

    def mat(field: str) -> np.ndarray:
        return np.array(
            [[float(groups[k][q][field]) for k in complete] for q in expected]
        )

    return OutcomeMatrix(
        tuple(expected),
        tuple(keys.get(q, q) for q in expected),
        cfgs,
        mat("quality"),
        mat("total_tokens"),
        mat("e2e_ms"),
        np.array([base[q] for q in expected]),
    )


def nomemory_column(
    official_dir: Path, query_ids: tuple[str, ...], dataset: str
) -> dict[str, np.ndarray]:
    """NoMemory per-query quality, tokens and latency from the official evaluator's files.

    Latency is the span of the item's LLM calls (first call start to last call end), i.e. the
    episode wall time without the environment reset; tokens are prompt + completion tokens.
    """
    calls: dict[int, list[dict]] = defaultdict(list)
    for line in open(official_dir / "llm_calls.jsonl"):
        c = json.loads(line)
        calls[int(c["item"])].append(c)
    q, t, lat = [], [], []
    for qid in query_ids:
        d = json.load(open(official_dir / f"{qid}.json"))
        r = float(d["reward"])
        q.append(max(r, 0) / 100 if dataset == "sciworld" else r)
        cs = calls.get(int(qid.split("_")[-1]), [])
        if not cs or any(c.get("prompt_tokens") is None for c in cs):
            raise ValueError(f"no complete LLM call log for {qid}")
        t.append(sum(c["prompt_tokens"] + (c["completion_tokens"] or 0) for c in cs))
        lat.append(
            1000 * (max(c["t"] + c["seconds"] for c in cs) - min(c["t"] for c in cs))
        )
    return {
        "quality": np.array(q),
        "tokens": np.array(t, dtype=float),
        "latency_ms": np.array(lat),
    }


def fingerprint(m: OutcomeMatrix) -> str:
    """Identity of the exact matrix (queries, configurations, outcomes) a report / artifact used."""
    h = hashlib.sha256()
    h.update(
        json.dumps(
            [
                list(m.query_ids),
                [
                    space.key(c) if c["plan"] != "NoMemory" else "NoMemory"
                    for c in m.configs
                ],
            ]
        ).encode()
    )
    for a in (m.quality, m.tokens, m.latency_ms, m.baseline):
        h.update(np.round(a, 6).tobytes())
    return h.hexdigest()[:16]


def with_nomemory(m: OutcomeMatrix, col: dict[str, np.ndarray]) -> OutcomeMatrix:
    if np.max(np.abs(col["quality"] - m.baseline)) > 1e-9:
        raise ValueError("NoMemory files disagree with the paired baseline in opt")
    return OutcomeMatrix(
        m.query_ids,
        m.task_keys,
        (dict(space.NOMEM), *m.configs),
        np.hstack([col["quality"][:, None], m.quality]),
        np.hstack([col["tokens"][:, None], m.tokens]),
        np.hstack([col["latency_ms"][:, None], m.latency_ms]),
        m.baseline,
    )
