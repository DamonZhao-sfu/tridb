"""Native-DB integration checks with a deterministic fake model; never uses a GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from trimem.exg.run import cell_path, run_job
from trimem.exg.runtime import ARMS, digest, read, restrict_resources, save


class FixtureCalls:
    def __init__(self, root, gate, seed, thinking=False):
        self.root = Path(root)
        self.context, self.receipts = {}, {}

    def embed(self, key, question, phase="retrieval"):
        return json.dumps([1.0] + [0.0] * 1023)

    def tokens(self, text=None, messages=None):
        return len(text if text is not None else json.dumps(messages)) // 4

    def chat(self, key, stage, messages, max_tokens=8192):
        if stage == "generation":
            obj = dict(
                answer="wrong"
                if self.context["task_id"].endswith("0")
                and self.context["attempt"] == 1
                else "ok"
            )
        elif stage == "retrieval_semantic":
            obj = dict(verdict=True)
        else:
            obj = dict(lesson="Check the supplied passages and use evaluator feedback.")
        receipt = dict(
            request=dict(stage=stage),
            context=dict(self.context),
            response={
                "choices": [
                    {"message": {"content": json.dumps(obj)}, "finish_reason": "stop"}
                ]
            },
            seconds=0.001,
            admission_seconds=0,
            usage=dict(prompt_tokens=20, completion_tokens=10),
        )
        self.receipts.setdefault(key, receipt)
        save(self.root / "fixture_calls" / f"{digest(key)}.json", receipt)
        return receipt


def run(root, dsn):
    root = Path(root)
    if root.exists():
        raise RuntimeError("Use a fresh fixture directory")
    protocol = dict(
        validation_subset=True,
        thinking=False,
        max_hint_tokens=8192,
        context_tokens=32768,
        generation_tokens=8192,
    )
    save(root / "protocol.json", protocol)
    for dataset in ("musique", "hotpotqa"):
        save(
            root / "data" / dataset / "manifest.json",
            dict(
                online=["c0", "c1", "t0", "t1"],
                offline_collection=["c0", "c1"],
                offline_test=["t0", "t1"],
                pilot=["c0", "c1", "t0", "t1"],
            ),
        )
        save(
            root / "data" / dataset / "inputs.json",
            {
                uid: dict(id=uid, question=f"Question {uid}", passages=[])
                for uid in ("c0", "c1", "t0", "t1")
            },
        )
        save(
            root / "data" / dataset / "evaluator_labels.json",
            {uid: dict(answer="ok", aliases=[]) for uid in ("c0", "c1", "t0", "t1")},
        )
    checked = []
    with patch("trimem.exg.run.Calls", FixtureCalls):
        for dataset in ("musique", "hotpotqa"):
            for mode in ("collection", "online", "offline"):
                for arm in ["T"] if mode == "collection" else ARMS:
                    job = dict(dataset=dataset, mode=mode, arm=arm, rep=0)
                    result = run_job(root, job, dsn)
                    assert result["tasks"] == 2
                    folder = cell_path(root, job)
                    episodes = [
                        read(p) for p in sorted((folder / "episodes").glob("*.json"))
                    ]
                    assert sum(e["pass1"] for e in episodes) == 1
                    assert all(e["pass2"] for e in episodes)
                    assert sum(e["retries"] for e in episodes) == 1
                    if mode == "offline":
                        assert all(not e["published"] for e in episodes)
                    if arm != "N":
                        assert result["graph"]["cases"] == 3
                    # Resume must not generate again or append duplicate graph cases.
                    assert run_job(root, job, dsn) == result
                    checked.append(job)
    save(
        root / "verified.json",
        dict(
            cells=len(checked),
            episodes=2 * len(checked),
            model="deterministic fixture, no real inference",
            checks=[
                "native graph publication",
                "all six arms",
                "two-attempt budget",
                "offline frozen fingerprint",
                "completed cell resume",
                "nonblocking plan checks",
            ],
            jobs=checked,
        ),
    )


def main():
    restrict_resources()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--dsn", default="host=127.0.0.1 port=55443 user=hza214 dbname=postgres"
    )
    args = ap.parse_args()
    run(args.out, args.dsn)


if __name__ == "__main__":
    main()
