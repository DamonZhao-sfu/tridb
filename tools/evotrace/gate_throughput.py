"""Phase 0 gate 1 -- measure serving throughput at the corpus's REAL prompt lengths.

The plan's wall-clock estimate was arithmetic over vendor-shaped assumptions. This
replaces it with a measurement on the actual distribution: `llm_calls.jsonl` records
prompt_tokens and completion_tokens for 16,873 real generations, so the probe samples
that distribution rather than a round number someone liked.

Prompts are synthetic filler at the sampled length -- this measures the SERVER, not the
agent, and filler avoids paying for a corpus read that would not change the timing.
Prefix caching is defeated by giving each request a distinct prefix, because a cache hit
would report a throughput the real workload never sees.

    python3 -m tools.evotrace.gate_throughput --endpoint http://127.0.0.1:8001/v1 -n 50
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_CALLS = Path("data/evotrace/normalized/llm_calls.jsonl")


def sample_lengths(path: Path, n: int, seed: int) -> list[tuple[int, int]]:
    import random

    pairs = []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            p, c = row.get("prompt_tokens"), row.get("completion_tokens")
            if p and c:
                pairs.append((int(p), int(c)))
    rng = random.Random(seed)
    return rng.sample(pairs, min(n, len(pairs)))


def one(endpoint: str, model: str, idx: int, prompt_tok: int, out_tok: int,
        cap: int) -> dict[str, object]:
    # ~4 chars/token for filler; the leading index defeats prefix caching.
    body = json.dumps({
        "model": model,
        "prompt": f"req{idx} " + ("word " * max(1, prompt_tok // 2)),
        "max_tokens": min(out_tok, cap),
        "temperature": 0.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{endpoint}/completions", data=body, headers={"Content-Type": "application/json"}
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=1800) as response:
            payload = json.loads(response.read())
    except Exception as exc:  # noqa: BLE001 - any failure is a datum, not a crash
        return {"ok": False, "error": type(exc).__name__, "detail": str(exc)[:120]}
    usage = payload.get("usage", {})
    return {
        "ok": True,
        "seconds": time.perf_counter() - start,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--model", default="qwen3.8")
    ap.add_argument("--calls", type=Path, default=DEFAULT_CALLS)
    ap.add_argument("-n", type=int, default=50)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-out", type=int, default=2048,
                    help="cap on completion tokens; the corpus p99 is 16,384 and "
                         "replaying that in full costs more than the measurement is worth")
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    lengths = sample_lengths(args.calls, args.n, args.seed)
    ptoks = sorted(p for p, _ in lengths)
    print(f"sampled {len(lengths)} real (prompt, completion) pairs")
    print(f"  prompt_tokens  p50={ptoks[len(ptoks)//2]:,}  max={ptoks[-1]:,}")
    print(f"  concurrency={args.concurrency}  max_out={args.max_out}")

    wall = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(
            lambda item: one(args.endpoint, args.model, item[0], item[1][0], item[1][1],
                             args.max_out),
            enumerate(lengths),
        ))
    wall = time.perf_counter() - wall

    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    if not ok:
        print(f"\nall {len(results)} requests failed; first: {results[0]}")
        return 1

    prompt_sum = sum(int(r["prompt_tokens"] or 0) for r in ok)
    out_sum = sum(int(r["completion_tokens"] or 0) for r in ok)
    lat = sorted(float(r["seconds"]) for r in ok)
    summary = {
        "requests_ok": len(ok),
        "requests_failed": len(failed),
        "wall_seconds": round(wall, 2),
        "prompt_tokens_total": prompt_sum,
        "completion_tokens_total": out_sum,
        "prompt_tokens_per_s": round(prompt_sum / wall, 1),
        "completion_tokens_per_s": round(out_sum / wall, 1),
        "latency_s_p50": round(statistics.median(lat), 2),
        "latency_s_p95": round(lat[int(len(lat) * 0.95) - 1], 2),
        "concurrency": args.concurrency,
        "max_out_cap": args.max_out,
    }
    print()
    for key, value in summary.items():
        print(f"  {key:<26} {value}")
    if failed:
        print(f"\n  first failure: {failed[0]}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
        print(f"\nreceipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
