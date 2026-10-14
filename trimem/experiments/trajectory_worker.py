"""Dedicated worker with per-HTTP-attempt usage and context attribution."""

import argparse
import hashlib
import json
import threading
import time
from pathlib import Path

from trimem.llm import SCORE_PROMPT, SYSTEM_PROMPT, OpenAIJudge
from trimem.worker import Worker


class LoggedJudge(OpenAIJudge):
    def __init__(self, base, model, logfile, concurrency=2):
        super().__init__(base, model, concurrency=concurrency)
        self.logfile = Path(logfile)
        self.local = threading.local()
        self.loglock = threading.Lock()
        original = self._http.post

        def post(*args, **kwargs):
            started = time.perf_counter()
            row = dict(getattr(self.local, "context", {}), started_unix=time.time())
            try:
                response = original(*args, **kwargs)
                row["http_status"] = response.status_code
                try:
                    row["usage"] = response.json().get("usage")
                except (ValueError, AttributeError):
                    row["usage"] = None
                return response
            except Exception as e:
                row.update(error=f"{type(e).__name__}: {e}", usage=None)
                raise
            finally:
                row["api_ms"] = (time.perf_counter() - started) * 1000
                with self.loglock, self.logfile.open("a") as f:
                    f.write(json.dumps(row) + "\n")

        self._http.post = post

    def _one(self, pair, system=SYSTEM_PROMPT, parse=None, **kwargs):
        text = pair[1]
        context = text.removeprefix("CURRENT_AGENT_CONTEXT=").split(
            "\nCANDIDATE_EVIDENCE=", 1
        )[0]
        self.local.context = {
            "op": "SemScore" if system == SCORE_PROMPT else "SemFilter",
            "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
            "input_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
        return super()._one(pair, system, parse, **kwargs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dsn", required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8")
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--max-take", type=int, default=2)
    a = p.parse_args()
    if min(a.concurrency, a.max_take) < 1:
        p.error("concurrency and max-take must be positive")
    Path(a.log).parent.mkdir(parents=True, exist_ok=True)
    Worker(
        a.dsn,
        LoggedJudge(a.base_url, a.model, a.log, a.concurrency),
        max_take=a.max_take,
    ).run()


if __name__ == "__main__":
    main()
