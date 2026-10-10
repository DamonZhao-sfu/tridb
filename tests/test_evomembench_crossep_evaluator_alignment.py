from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from bench.agent_memory.evomembench import evaluate_know


REPO = Path(__file__).resolve().parents[1]
OFFICIAL_EVAL = REPO / "EvoMemBench/Cross-Episode-Knowledge/CROSSEP-KNOW/eval.py"
OFFICIAL_SHA256 = "f5766616907e02ce523226d0ae8ad1e28686c5c830ede30c345d5b8f303696f6"


class _Completions:
    def __init__(self, content: str):
        self.content = content
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _Client:
    def __init__(self, content: str):
        self.chat = SimpleNamespace(completions=_Completions(content))


def _item(task_id: str = "task-0") -> dict:
    return {
        "idx": 7,
        "metadata": {"task_id": task_id},
        "model_output": "student answer",
        "rubrics": [
            {"rubric_criteria": "must mention alpha"},
            {"rubric_criteria": "must mention beta"},
        ],
    }


def _args(tmp_path: Path, *, max_retries: int = 5) -> argparse.Namespace:
    input_path = tmp_path / "predictions.jsonl"
    input_path.write_text(json.dumps(_item()) + "\n")
    return argparse.Namespace(
        input=str(input_path),
        output=str(tmp_path / "graded.jsonl"),
        receipt=str(tmp_path / "receipt.json"),
        official_eval=str(OFFICIAL_EVAL),
        judge_model="fake-judge",
        base_url="http://fake.invalid/v1",
        api_key="EMPTY",
        workers=1,
        max_retries=max_retries,
        resume=False,
    )


def _install_fake_transport(monkeypatch, official, client: _Client) -> None:
    monkeypatch.setattr(
        official,
        "get_api_credentials",
        lambda *_args: ("EMPTY", "http://fake.invalid/v1", "openai"),
    )
    monkeypatch.setattr(official, "OpenAI", lambda **_kwargs: client)
    monkeypatch.setattr(evaluate_know, "_load_official", lambda _path: official)


def test_pinned_crossep_evaluator_source_hash() -> None:
    assert hashlib.sha256(OFFICIAL_EVAL.read_bytes()).hexdigest() == OFFICIAL_SHA256


def test_lossless_adapter_is_differentially_identical_on_success(
    tmp_path: Path, monkeypatch
) -> None:
    official = evaluate_know._load_official(OFFICIAL_EVAL)
    payload = json.dumps(
        {
            "Grading Rationale": "alpha and beta are both present",
            "List of Requirement Satisfaction Status": ["yes", "yes"],
            "Overall Score": 1,
        }
    )
    direct_client = _Client(payload)
    _, direct, direct_error = official.process_single_item(
        (_item(), direct_client, "fake-judge", 5)
    )
    assert direct_error is None

    adapter_client = _Client(payload)
    _install_fake_transport(monkeypatch, official, adapter_client)
    receipt = evaluate_know.evaluate(_args(tmp_path))
    adapted = json.loads((tmp_path / "graded.jsonl").read_text())

    assert adapted == direct
    assert receipt["predictions"] == receipt["graded"] == 1
    assert receipt["judge_failures_counted_as_score_zero"] == 0
    assert (
        direct_client.chat.completions.calls[0]["messages"]
        == (adapter_client.chat.completions.calls[0]["messages"])
    )
    assert direct_client.chat.completions.calls[0]["response_format"] == {
        "type": "json_object"
    }


def test_adapter_preserves_official_failure_as_score_zero_denominator(
    tmp_path: Path, monkeypatch
) -> None:
    official = evaluate_know._load_official(OFFICIAL_EVAL)
    client = _Client("not-json")
    _install_fake_transport(monkeypatch, official, client)

    receipt = evaluate_know.evaluate(_args(tmp_path, max_retries=1))
    adapted = json.loads((tmp_path / "graded.jsonl").read_text())

    assert adapted["score"] == 0
    assert "JSON parse failed" in adapted["grading_rationale"]
    assert "JSON parse failed" in adapted["evaluator_error"]
    assert receipt["predictions"] == receipt["graded"] == 1
    assert receipt["denominator_preserved"] is True
    assert receipt["judge_failures_counted_as_score_zero"] == 1


def test_adapter_cli_defaults_match_official_driver(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_know",
            "--input",
            "in.jsonl",
            "--output",
            "out.jsonl",
            "--receipt",
            "receipt.json",
            "--official-eval",
            str(OFFICIAL_EVAL),
        ],
    )
    args = evaluate_know.parse_args()
    assert args.max_retries == 5
    assert args.workers == 1
    assert args.judge_model == "gpt-5.1"
    assert args.base_url is None
    assert args.api_key is None
