"""Durable API receipts and resource admission for long QA experiments."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

import httpx
import psutil

from trimem.experiments.exg_pilot import EMBED, MODEL
from trimem.llm import SYSTEM_PROMPT, Verdict, parse_verdict, user_prompt

CPUS = [16, 17, 18, 19, 36, 37, 38, 39]
ARMS = ["N", "V", "T", "S0", "S1", "S2"]


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temp, path)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def restrict_resources():
    available = set(os.sched_getaffinity(0))
    selected = set(CPUS) & available
    if selected:
        os.sched_setaffinity(0, selected)
    process = psutil.Process()
    process.nice(max(10, process.nice()))
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Calls:
    """A stable name identifies one logical call; completed responses are never resent."""

    def __init__(self, root, gate, seed, thinking=False):
        self.root, self.gate, self.seed = Path(root), Path(gate), seed
        self.gate.mkdir(parents=True, exist_ok=True)
        self.http = httpx.Client(timeout=900)
        self.context = {}
        self.receipts = {}
        self.thinking = thinking

    @contextlib.contextmanager
    def admission(self):
        start = time.perf_counter()
        while True:
            cpu = psutil.cpu_percent(interval=0.2)
            mem = psutil.virtual_memory()
            if cpu < 65 and mem.available > 12 * 1024**3:
                for slot in range(4):
                    f = (self.gate / f"llm_slot_{slot}.lock").open("a")
                    try:
                        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        f.close()
                        continue
                    try:
                        yield time.perf_counter() - start
                    finally:
                        fcntl.flock(f, fcntl.LOCK_UN)
                        f.close()
                    return
            time.sleep(2)

    def post(self, key, stage, port, route, body):
        path = self.root / "calls" / f"{key}.json"
        request = dict(stage=stage, port=port, route=route, body=body)
        sha = digest(request)
        if path.exists():
            old = read(path)
            if old["request_hash"] != sha:
                raise RuntimeError(f"Changed request on resume: {path}")
            if old["status"] != "complete":
                raise RuntimeError(
                    f"Unresolved prior API request; cannot resend silently: {path}"
                )
            self.receipts[str(path)] = old
            return old
        with self.admission() as admission_seconds:
            receipt = dict(
                status="inflight",
                request_hash=sha,
                request=request,
                context=dict(self.context),
                start_epoch=time.time(),
                admission_seconds=admission_seconds,
            )
            save(path, receipt)
            start = time.perf_counter()
            try:
                response = self.http.post(f"http://127.0.0.1:{port}{route}", json=body)
                response.raise_for_status()
                data = response.json()
            except Exception as exc:
                save(
                    path,
                    receipt
                    | dict(
                        status="error",
                        error=repr(exc),
                        seconds=time.perf_counter() - start,
                    ),
                )
                raise
            receipt.update(
                status="complete",
                seconds=time.perf_counter() - start,
                response=data,
                usage=data.get("usage", {}),
            )
            save(path, receipt)
            self.receipts[str(path)] = receipt
            return receipt

    def chat(self, key, stage, messages, max_tokens=8192):
        reasoning = stage == "generation"
        return self.post(
            key,
            stage,
            8000,
            "/v1/chat/completions",
            dict(
                model=MODEL,
                messages=messages,
                temperature=0.7 if reasoning else 0,
                top_p=0.95 if reasoning else 1,
                seed=self.seed if reasoning else 0,
                max_tokens=max_tokens,
                chat_template_kwargs={"enable_thinking": reasoning and self.thinking},
                response_format={"type": "json_object"},
            ),
        )

    def embed(self, key, question, phase="retrieval"):
        result = self.post(
            key,
            f"{phase}_embedding",
            8011,
            "/v1/embeddings",
            dict(model=EMBED, input=question),
        )
        vector = result["response"]["data"][0]["embedding"]
        assert len(vector) == 1024
        return json.dumps(vector)

    def tokens(self, text=None, messages=None):
        body = dict(model=MODEL)
        if messages is None:
            body["prompt"] = text
        else:
            body.update(
                messages=messages,
                add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": self.thinking},
            )
        response = self.http.post("http://127.0.0.1:8000/tokenize", json=body)
        response.raise_for_status()
        return response.json()["count"]


class LoggedJudge:
    model = MODEL

    def __init__(self, calls):
        self.calls = calls

    def judge_many(self, pairs):
        out = []
        for pred, text in pairs:
            # Exact predicate+input+model define a semantic-cache entry within this job.
            key = "semantic/" + digest([MODEL, pred, text])
            receipt = self.calls.chat(
                key,
                "retrieval_semantic",
                [
                    dict(role="system", content=SYSTEM_PROMPT),
                    dict(role="user", content=user_prompt(pred, text)),
                ],
                128,
            )
            data = receipt["response"]
            value = parse_verdict(data["choices"][0]["message"].get("content") or "")
            if value is None:
                raise RuntimeError(f"Invalid semantic verdict: {key}")
            usage = receipt["usage"]
            out.append(
                Verdict(
                    value,
                    usage["prompt_tokens"],
                    usage["completion_tokens"],
                    receipt["seconds"],
                )
            )
        return out
