"""Offline measurement for SemFilter v2c (TRACE v0.7.1): judge-card tokens raw vs compact, and the live judge load.

1. Cards: a deterministic sample of real candidate sessions of the frozen read-only memory clone (agent-env
   sessions with a known outcome, non-held-out tasks = what an ALFWorld / ScienceWorld / WebShop retrieval can
   judge), hydrated like ``AgentEnvTaskAdapter.hydrate`` (``trimem.expand`` has_step + ``step``), labelled as under
   memory policy success+failure-v1, rendered as the stock raw judge card (``semantic_rubric.env_card``) and as
   ``judge-card-compact-v1``, and tokenized with the Qwen tokenizer of the agent / judge model (chars / 4 when the
   tokenizer is unavailable). Read-only connection; SELECT only (no ANALYZE).
2. Optional ``--receipts-run R``: read-only scan of ``R/episodes/*/*/memory/*.json`` plus a read-only replay of the
   same retrievals' receipts (``opt.session_retrieval``): candidates judged per episode under stock / TOPN 16 /
   TOPN 16 + episode reuse, and the periodic hit rate.

    .venv-optimizer/bin/python -m tools.semfilter_v071_measure --dataset alfworld \\
        --db trace_ablation_sci_48d26f8d18eba20f --n 400 --receipts-run results/trace_v070_alfworld_init_20261105 \\
        --out results/trace_v070_e2e_20261105/semfilter_v071_card_tokens_alfworld.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from collections import Counter
from pathlib import Path

MODEL = "Qwen/Qwen3.8-27B-FP8"
SESSION_SQL = (
    "SELECT s.vid, s.success, s.final_score, s.n_steps, t.env, t.description FROM session s "
    "JOIN task t ON t.id = s.task_id WHERE t.env = %s AND s.success IS NOT NULL AND NOT t.held_out "
    "AND s.n_steps > 0 ORDER BY md5(s.key || %s) LIMIT %s"
)


def tokenizer():
    os.environ.setdefault("HF_HOME", "/localhome/hza214/hf-cache")
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        return (lambda s: len(tok.encode(s, add_special_tokens=False))), MODEL
    except Exception as exc:  # noqa: BLE001 - fall back to the documented approximation
        return (lambda s: len(s) / 4), f"chars/4 (tokenizer unavailable: {exc!r})"


JUDGED_SQL = ("SELECT s.vid, s.id, s.success, s.final_score, s.n_steps, t.env, t.description FROM session s "
              "JOIN task t ON t.id = s.task_id WHERE s.id = ANY(%s)")


def cards(dataset: str, db: str, n: int, seed: str, ids: list | None = None) -> list[dict]:
    import psycopg
    from psycopg.rows import dict_row

    from trimem.experiments import memory_policy as mp
    from trimem.experiments.agentenv_task_adapter import env_action

    dsn = f"host=/tmp/trimem-traj port=55450 dbname={db} options='-c default_transaction_read_only=on'"
    out = []
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        rows = (conn.execute(JUDGED_SQL, (ids,)).fetchall() if ids is not None
                else conn.execute(SESSION_SQL, (dataset, seed, n)).fetchall())
        for s in rows:
            nodes = conn.execute("SELECT dst FROM trimem.expand(%s, 'has_step', %s)",
                                 (s["vid"], s["n_steps"] + 1)).fetchall()
            steps = conn.execute("SELECT ord, action, observation FROM step WHERE vid = ANY(%s) ORDER BY ord",
                                 ([r["dst"] for r in nodes],)).fetchall()
            out.append({**s, **mp.card_stamp(mp.SUCCESS_FAILURE),
                        "steps": [{**x, "action": env_action(dataset, x["action"])} for x in steps]})
    return out


def sr_text(card: dict) -> str:
    from trimem.experiments import semantic_rubric as sr

    return sr.env_card(card)


def compact_text(card: dict) -> str:
    from trimem.experiments.semfilter_v071 import compact_card

    return compact_card(card)


def _stats(xs) -> dict:
    xs = sorted(xs)
    return {"mean": statistics.mean(xs), "p50": xs[len(xs) // 2], "p90": xs[int(0.9 * (len(xs) - 1))],
            "max": xs[-1]}


def measure_cards(rows, count) -> dict:
    from trimem.experiments import semantic_rubric as sr
    from trimem.experiments.semfilter_v071 import CARD_CAP, compact_card

    raw = [sr.env_card(c) for c in rows]
    new = [compact_card(c) for c in rows]
    rt, nt = [count(t) for t in raw], [count(t) for t in new]
    keep = all(("SUCCESSFUL" in b or "FAILED" in b) == ("SUCCESSFUL" in a or "FAILED" in a)
               and "Goal:" in b for a, b in zip(raw, new))
    return {"cards": len(rows), "raw_tokens": _stats(rt), "compact_tokens": _stats(nt),
            "raw_chars": _stats([len(t) for t in raw]), "compact_chars": _stats([len(t) for t in new]),
            "token_reduction": 1 - sum(nt) / sum(rt), "compact_at_cap": sum(len(t) >= CARD_CAP - 1 for t in new),
            "label_and_goal_kept": keep, "example_raw": raw[0][:800], "example_compact": new[0][:800]}


def _judged_ids(rec: dict) -> list:
    """Candidates a stock retrieval judged, in query-cosine order (the prefilter's order)."""
    pre = rec.get("semfilter_prefilter") or {}
    return list(pre.get("judged_ids") or [d["id"] for d in rec.get("semantic_decisions") or []])


def replay_episode(receipts: list[dict], top_n: int) -> dict:
    """Candidates judged by one episode's retrievals (in order) under stock / TOPN / TOPN + episode reuse."""
    stock = top = new = periodic_seen = periodic_hits = 0
    seen: set = set()
    for i, rec in enumerate(receipts):
        ids = _judged_ids(rec)
        stock += len(ids)
        cut = ids[:top_n]
        top += len(cut)
        fresh = [c for c in cut if c not in seen]
        new += len(fresh)
        if i:
            periodic_seen += len(cut)
            periodic_hits += len(cut) - len(fresh)
        seen |= set(cut)
    return {"stock": stock, "top": top, "reuse": new, "periodic_seen": periodic_seen, "periodic_hits": periodic_hits}


def measure_receipts(run: str, top_n: int = 16, store_dsn: str | None = None) -> dict:
    """Read-only: memory files + (optional) ``opt.session_retrieval`` receipts of the same retrieval ids."""
    episodes = []
    for f in glob.glob(str(Path(run) / "episodes/*/*/memory/*.json")):
        m = json.loads(Path(f).read_text())
        sf = (m.get("v2") or {}).get("semfilter") if isinstance(m.get("v2"), dict) else None
        if sf:
            episodes.append((m, sf))
    if not episodes:
        return {"episodes_with_semfilter": 0}
    tot = sum(float(m.get("semantic_tokens") or 0) for m, _ in episodes)
    per = sum(float(r.get("semantic_tokens") or 0) for m, _ in episodes for r in m.get("periodic") or [])
    out = {"episodes_with_semfilter": len(episodes),
           "retrievals_per_episode": statistics.mean(1 + len(m.get("periodic") or []) for m, _ in episodes),
           "episodes_with_periodic": sum(bool(m.get("periodic")) for m, _ in episodes),
           "judged_initial_retrieval": statistics.mean(sf["judged"] for _, sf in episodes),
           "judge_prompt_tokens_initial_retrieval": statistics.mean(sf["judge_prompt_tokens"] for _, sf in episodes),
           "semantic_tokens_per_episode": tot / len(episodes), "periodic_share_of_semantic_tokens": per / tot if tot else 0}
    if store_dsn:
        out["replay"] = replay_store(episodes, store_dsn, top_n)
    return out


def replay_store(episodes, dsn: str, top_n: int) -> dict:
    import psycopg

    totals = {"stock": 0, "top": 0, "reuse": 0, "periodic_seen": 0, "periodic_hits": 0}
    missing = 0
    judged: list = []
    with psycopg.connect(dsn + " options='-c default_transaction_read_only=on'") as conn:
        for m, _ in episodes:
            rids = [m["rid"]] + [r["rid"] for r in m.get("periodic") or []]
            rows = dict(conn.execute("SELECT retrieval_id::text, receipt FROM opt.session_retrieval "
                                     "WHERE retrieval_id::text = ANY(%s)", (rids,)).fetchall())
            if any(r not in rows for r in rids):
                missing += 1
                continue
            one = replay_episode([rows[r] for r in rids], top_n)
            judged += [c for r in rids for c in _judged_ids(rows[r])]
            totals = {k: totals[k] + one[k] for k in totals}
    s = totals["stock"] or 1
    return {**totals, "judged_ids": judged, "episodes_missing_receipts": missing, "top_n": top_n,
            "judged_vs_stock_top": totals["top"] / s,
            "judged_vs_stock_top_reuse": totals["reuse"] / s,
            "periodic_hit_rate": totals["periodic_hits"] / totals["periodic_seen"] if totals["periodic_seen"] else None}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, choices=("alfworld", "sciworld", "webshop"))
    ap.add_argument("--db", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", default="semfilter-v071")
    ap.add_argument("--receipts-run", default=None)
    ap.add_argument("--store-dsn", default="host=/tmp/trimem-traj port=55450 dbname=trimem_sweqa")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    count, how = tokenizer()
    body = {"dataset": a.dataset, "db": a.db, "tokenizer": how, "sample": {"n": a.n, "seed": a.seed},
            **measure_cards(cards(a.dataset, a.db, a.n, a.seed), count)}
    if a.receipts_run:
        body["live_receipts"] = {"run": a.receipts_run, **measure_receipts(a.receipts_run, store_dsn=a.store_dsn)}
        rep = body["live_receipts"].get("replay") or {}
        judged = rep.pop("judged_ids", [])
        if judged:  # the candidates the live run judged, weighted by how often they were judged
            weights = Counter(judged)
            rows = {c["id"]: c for c in cards(a.dataset, a.db, a.n, a.seed, ids=list(weights))}
            count_of = {k: (count(sr_text(c)), count(compact_text(c))) for k, c in rows.items()}
            raw = sum(weights[k] * v[0] for k, v in count_of.items())
            new = sum(weights[k] * v[1] for k, v in count_of.items())
            total = sum(weights[k] for k in count_of)
            body["judged_cards"] = {"unique_cards": len(rows), "judgements": total, "raw_tokens_mean": raw / total,
                                    "compact_tokens_mean": new / total, "token_reduction": 1 - new / raw}
    a.out.write_text(json.dumps(body, indent=1, default=str) + "\n")
    print(json.dumps({k: v for k, v in body.items() if not k.startswith("example")}, default=str))


if __name__ == "__main__":
    main()
