"""Local-model evaluator for prompt-based structured extraction."""

from __future__ import annotations

import importlib.util
import json
import time
import urllib.request
from pathlib import Path

from experiments.cross_session_reuse.calibration.ledger import append_attempt

CASES = (
    (
        "Ada Lovelace works as a mathematician and lives in London.",
        {"name": "Ada Lovelace", "role": "mathematician", "city": "London"},
    ),
    (
        "In Helsinki, Linus Torvalds is a software engineer.",
        {"name": "Linus Torvalds", "role": "software engineer", "city": "Helsinki"},
    ),
)


def _complete(prompt: str) -> str:
    body = json.dumps(
        {
            "model": "Qwen/Qwen3-32B",
            "messages": [{"role": "user", "content": "/no_think " + prompt}],
            "temperature": 0,
            "max_tokens": 128,
        }
    ).encode()
    request = urllib.request.Request(
        "http://127.0.0.1:8000/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    return payload["choices"][0]["message"]["content"].strip()


def _json_object(text: str) -> dict[str, str]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return {}
    value = json.loads(text[start : end + 1])
    return value if isinstance(value, dict) else {}


def evaluate(program_path: str) -> dict[str, float | str]:
    started = time.monotonic()
    status = "valid"
    error_message = ""
    passed = 0
    prompt_chars = 0
    try:
        spec = importlib.util.spec_from_file_location("csr_prompt_candidate", program_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load candidate: {program_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for text, expected in CASES:
            prompt = module.build_prompt(text)
            if not isinstance(prompt, str):
                continue
            prompt_chars += len(prompt)
            try:
                actual = _json_object(_complete(prompt))
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
            passed += int(actual == expected)
        if passed != len(CASES):
            status = "invalid"
            error_message = f"{len(CASES) - passed} held-out prompt cases failed"
    except Exception as exc:
        status = "error"
        error_message = f"{type(exc).__name__}: {exc}"
    accuracy = passed / len(CASES)
    concision = 1.0 / (1.0 + prompt_chars / max(len(CASES), 1) / 1000.0)
    output: dict[str, float | str] = {
        "combined_score": accuracy * 0.95 + concision * 0.05,
        "task_accuracy": accuracy,
        "prompt_concision": concision,
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
    print(evaluate(str(Path(__file__).with_name("prompt_optimization_initial.py"))))

