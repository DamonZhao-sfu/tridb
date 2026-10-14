"""Regenerate detailed change summaries for a frozen experience-graph snapshot (no DB access).

Every attempt whose parent is in the snapshot and whose parent/child code are both present gets a new
`changes` summary from the LLM (DERIVE_SYSTEM_DETAILED over the parent->child unified diff). Results go
to a resumable JSONL cache keyed by (attempt id, prompt sha, diff sha); a rerun only calls the LLM for
missing keys. With --write-snapshot (and zero failures) a new snapshot is written that is identical to
the input except attempts[*].changes and an added provenance block.

    taskset -c 16-19 .venv/bin/python -m trimem.data.regen_change_summaries \
        --snapshot .../math_heilbronn_triangle_frozen_v2.json --out-dir .../summaries_v3 --write-snapshot
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from trimem.replay.memory import DERIVE_SYSTEM_DETAILED, Chat, _diff

# One thread for numpy & co. (the snapshot verifier imports it lazily); CPU pinning is done by taskset.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

MAX_CHARS = 1024
MAX_CONCURRENCY = 4  # the vLLM server is shared with a live run
USER_PREFIX = "PARENT/CHILD unified diff:\n"
SENTENCE_END = re.compile(r"[.!?](?=\s|$)")
LIST_MARKER = re.compile(
    r"(^|\s)\(?\d{1,2}[.)]$"
)  # "... 3." is a list marker, not a sentence end


def sha256(text: str | bytes) -> str:
    return hashlib.sha256(text.encode() if isinstance(text, str) else text).hexdigest()


def clamp(text: str) -> tuple[str, bool, bool]:
    """Drop markdown bold markers, whitespace-normalise, cut to the last sentence end <= MAX_CHARS (else
    hard-cut). Returns (text, truncated, had_markdown)."""
    had_markdown = "**" in text
    text = " ".join(text.replace("**", "").split())
    if len(text) <= MAX_CHARS:
        return text, False, had_markdown
    head = text[:MAX_CHARS]
    ends = [
        m.end()
        for m in SENTENCE_END.finditer(head)
        if not LIST_MARKER.search(head[: m.end()])
    ]
    return (head[: ends[-1]].rstrip() if ends else head), True, had_markdown


def load_cache(path: Path) -> dict[tuple, dict]:
    rows: dict[tuple, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[(r["id"], r["prompt_sha256"], r["diff_sha256"])] = r
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True, type=Path)
    ap.add_argument(
        "--sha256", help="expected input sha (default: the sibling .sha256 file)"
    )
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--out-name", default="math_heilbronn_triangle_frozen_v3.json")
    ap.add_argument(
        "--ids", help="comma-separated attempt ids (smoke test); default all eligible"
    )
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--diff-limit", type=int, default=12000)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument(
        "--retries",
        type=int,
        default=5,
        help="outer retries per attempt (Chat retries 4x inside)",
    )
    ap.add_argument("--write-snapshot", action="store_true")
    args = ap.parse_args(argv)
    if not 1 <= args.concurrency <= MAX_CONCURRENCY:
        ap.error(f"--concurrency must be in 1..{MAX_CONCURRENCY}")

    raw = args.snapshot.read_bytes()
    in_sha = sha256(raw)
    expected = (
        args.sha256 or Path(str(args.snapshot) + ".sha256").read_text().split()[0]
    )
    if in_sha != expected:
        raise SystemExit(f"input snapshot sha mismatch: {in_sha} != {expected}")
    doc = json.loads(raw)
    by_id = {a["id"]: a for a in doc["attempts"]}

    prompt_sha = sha256(DERIVE_SYSTEM_DETAILED)
    eligible, parentless, no_code = [], [], []
    for a in doc["attempts"]:
        p = by_id.get(a["parent_id"]) if a["parent_id"] is not None else None
        if p is None:
            parentless.append(a["id"])
        elif not a.get("code") or not p.get("code"):
            no_code.append(a["id"])
        else:
            eligible.append(a["id"])
    targets = [int(x) for x in args.ids.split(",")] if args.ids else eligible
    if bad := sorted(set(targets) - set(eligible)):
        raise SystemExit(f"not eligible (no parent or no code): {bad}")

    def job(aid: int) -> dict:
        a, p = by_id[aid], by_id[by_id[aid]["parent_id"]]
        full = _diff(p["code"], a["code"], 10**9)
        diff = full[: args.diff_limit]
        return {
            "id": aid,
            "parent_id": p["id"],
            "prompt_sha256": prompt_sha,
            "diff_sha256": sha256(diff),
            "diff_chars": len(diff),
            "diff_full_chars": len(full),
            "diff": diff,
        }

    jobs = [job(i) for i in targets]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.out_dir / "summaries_v3.jsonl"
    fail_path = args.out_dir / "failures_v3.jsonl"
    cache = load_cache(cache_path)
    todo = [
        j for j in jobs if (j["id"], j["prompt_sha256"], j["diff_sha256"]) not in cache
    ]
    print(
        f"targets={len(jobs)} cached={len(jobs) - len(todo)} todo={len(todo)}",
        flush=True,
    )

    local = threading.local()
    lock = threading.Lock()
    failures: list[dict] = []

    def call(j: dict) -> dict | None:
        if not hasattr(local, "chat"):
            local.chat = Chat()
        chat = local.chat
        err = None
        for attempt in range(args.retries):
            try:
                text, pt, ct, ms = chat.complete(
                    DERIVE_SYSTEM_DETAILED,
                    USER_PREFIX + j["diff"],
                    temperature=0.0,
                    seed=0,
                    max_tokens=args.max_tokens,
                )
                if not text.strip():
                    raise ValueError(
                        f"empty completion (finish_reason={chat.last_finish_reason})"
                    )
                new, truncated, had_markdown = clamp(text)
                row = {k: v for k, v in j.items() if k != "diff"} | {
                    "model": chat.model,
                    "max_tokens": args.max_tokens,
                    "diff_limit": args.diff_limit,
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "ms": ms,
                    "finish_reason": chat.last_finish_reason,
                    "raw_chars": len(" ".join(text.split())),
                    "truncated": truncated,
                    "markdown_stripped": had_markdown,
                    "old": by_id[j["id"]].get("changes"),
                    "new": new,
                    "raw": text,
                }
                with lock, cache_path.open("a") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                return row
            except Exception as e:  # noqa: BLE001 -- record any failure, never fall back silently
                err = f"{type(e).__name__}: {e}"
                time.sleep(min(120, 5 * 2**attempt))
        fail = {
            "id": j["id"],
            "error": err,
            "at_utc": datetime.now(timezone.utc).isoformat(),
        }
        with lock:
            failures.append(fail)
            with fail_path.open("a") as f:
                f.write(json.dumps(fail) + "\n")
        return None

    t0 = time.perf_counter()
    done = 0
    with ThreadPoolExecutor(args.concurrency) as pool:
        for fut in as_completed([pool.submit(call, j) for j in todo]):
            fut.result()
            done += 1
            if done % 50 == 0:
                print(
                    f"{done}/{len(todo)} {time.perf_counter() - t0:.0f}s failures={len(failures)}",
                    flush=True,
                )
    wall = time.perf_counter() - t0

    cache = load_cache(cache_path)
    results = {
        j["id"]: cache[k]
        for j in jobs
        if (k := (j["id"], j["prompt_sha256"], j["diff_sha256"])) in cache
    }
    for r in results.values():  # post-processing is recomputed from the raw completion
        r["new"], r["truncated"], r["markdown_stripped"] = clamp(r["raw"])
    missing = sorted(j["id"] for j in jobs if j["id"] not in results)
    rows = list(results.values())

    def dist(xs: list[int]) -> dict:
        return (
            {"min": min(xs), "median": statistics.median(xs), "max": max(xs)}
            if xs
            else {}
        )

    receipt = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).read_bytes()),
        "input_snapshot": str(args.snapshot.resolve()),
        "input_snapshot_sha256": in_sha,
        "model": Chat().model,
        "temperature": 0.0,
        "seed": 0,
        "max_tokens": args.max_tokens,
        "diff_limit": args.diff_limit,
        "diff_context_lines": 1,
        "max_summary_chars": MAX_CHARS,
        "concurrency": args.concurrency,
        "system_prompt": DERIVE_SYSTEM_DETAILED,
        "system_prompt_sha256": prompt_sha,
        "user_message_template": USER_PREFIX
        + "<_diff(parent.code, child.code, diff_limit)>",
        "attempts_total": len(doc["attempts"]),
        "eligible": len(eligible),
        "targets": len(jobs),
        "parentless_kept_old": len(parentless),
        "parent_without_code_kept_old": len(no_code),
        "summarised": len(rows),
        "llm_calls_this_run": len(todo),
        "reused_from_cache": len(jobs) - len(todo),
        "failures": len(missing),
        "failed_ids": missing,
        "failures_this_run": failures,
        "truncations": sum(r["truncated"] for r in rows),
        "finish_reason_counts": {
            k: sum(r["finish_reason"] == k for r in rows)
            for k in sorted({str(r["finish_reason"]) for r in rows})
        },
        "diff_input_truncated": sum(
            r["diff_full_chars"] > r["diff_chars"] for r in rows
        ),
        "markdown_stripped": sum(r.get("markdown_stripped", False) for r in rows),
        "diff_empty": sum(r["diff_full_chars"] == 0 for r in rows),
        "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
        "completion_tokens": sum(r["completion_tokens"] for r in rows),
        "llm_ms_sum": sum(r["ms"] for r in rows),
        "wall_s_this_run": wall,
        "chars_old": dist([len(r["old"] or "") for r in rows]),
        "chars_new": dist([len(r["new"]) for r in rows]),
        "cache": str(cache_path.resolve()),
    }
    receipt_path = args.out_dir / (
        "summary_build_receipt.json" if not args.ids else "smoke_receipt.json"
    )

    if args.write_snapshot:
        if args.ids or missing:
            raise SystemExit(
                f"refusing to write snapshot: ids={bool(args.ids)} missing={len(missing)}"
            )
        out = json.loads(raw)
        for a in out["attempts"]:
            if a["id"] in results:
                a["changes"] = results[a["id"]]["new"]
        out["provenance"] = out["provenance"] | {
            "v3_created_at_utc": receipt["created_at_utc"],
            "v3_change": "only attempts[*].changes replaced by regenerated detailed LLM summaries for every attempt "
            "with an in-snapshot parent and both codes; parentless attempts keep the v2 text; code, "
            "fitness, validity, embeddings, edges, queries byte-for-byte equal to v2",
            "v3_derived_from": str(args.snapshot.resolve()),
            "v3_derived_from_sha256": in_sha,
            "v3_builder": receipt["script"],
            "v3_builder_sha256": receipt["script_sha256"],
            "v3_model": receipt["model"],
            "v3_system_prompt_sha256": prompt_sha,
            "v3_diff_limit": args.diff_limit,
            "v3_max_tokens": args.max_tokens,
            "v3_summaries_replaced": len(results),
            "v3_receipt": str(receipt_path.resolve()),
            "v3_summaries": str(cache_path.resolve()),
        }
        out_path = args.out_dir / args.out_name
        blob = (json.dumps(out, ensure_ascii=False, allow_nan=False) + "\n").encode()
        out_path.write_bytes(blob)
        out_sha = sha256(blob)
        Path(str(out_path) + ".sha256").write_text(f"{out_sha}  {out_path.name}\n")
        verify(doc, out_path, out_sha, results)
        receipt |= {
            "output_snapshot": str(out_path.resolve()),
            "output_snapshot_sha256": out_sha,
            "output_verified": "FrozenSnapshot.read(path, sha) ok; non-changes content equal to input",
        }
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in receipt.items()
                if k not in ("system_prompt", "failures_this_run")
            },
            indent=2,
        )
    )
    return 1 if missing else 0


def verify(src: dict, path: Path, sha: str, results: dict) -> None:
    from trimem.optimizer.snapshot import FrozenSnapshot

    snap = FrozenSnapshot.read(path, sha)
    new = snap.document
    for key in src:
        if key not in ("attempts", "provenance") and src[key] != new[key]:
            raise AssertionError(f"{key} differs from input")
    if any(new["provenance"].get(k) != v for k, v in src["provenance"].items()):
        raise AssertionError("input provenance not preserved")
    for a, b in zip(src["attempts"], new["attempts"], strict=True):
        want = results[a["id"]]["new"] if a["id"] in results else a.get("changes")
        if {**a, "changes": None} != {**b, "changes": None} or b.get("changes") != want:
            raise AssertionError(f"attempt {a['id']} differs beyond changes")


if __name__ == "__main__":
    sys.exit(main())
