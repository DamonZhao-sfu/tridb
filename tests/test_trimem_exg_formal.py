"""Scientific protocol and crash-recovery invariants for the QA experiment."""

import ast
import collections
import json
import re
import string
from pathlib import Path

import pytest

from trimem.exg.runtime import Calls, digest, read, save
from trimem.exg.scoring import score

REPO = Path(__file__).resolve().parents[1]


def official(dataset):
    filename = "musique_answer.py" if dataset == "musique" else "hotpot_evaluate_v1.py"
    tree = ast.parse((REPO / "exg/vendor/qa_metrics" / filename).read_text())
    wanted = {
        "normalize_answer",
        "get_tokens",
        "compute_exact",
        "compute_f1",
        "f1_score",
        "exact_match_score",
    }
    module = ast.Module(
        body=[
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted
        ],
        type_ignores=[],
    )
    namespace = dict(
        re=re, string=string, collections=collections, Counter=collections.Counter
    )
    exec(compile(module, filename, "exec"), namespace)
    return namespace


@pytest.mark.parametrize("dataset", ["musique", "hotpotqa"])
def test_official_score_differential(dataset):
    funcs = official(dataset)
    examples = [
        "",
        "the",
        "A red apple",
        "apple",
        "YES",
        "yes man",
        "no",
        "noanswer",
        "New-York",
        "New York",
        "1999",
        "O'Neil",
        "a cat, a cat",
    ]
    for answer in examples:
        for target in examples:
            actual = score(answer, dict(answer=target), dataset)
            if dataset == "musique":
                em, f1 = (
                    funcs["compute_exact"](target, answer),
                    funcs["compute_f1"](target, answer),
                )
            else:
                em, f1 = (
                    funcs["exact_match_score"](answer, target),
                    funcs["f1_score"](answer, target)[0],
                )
            assert actual["em"] == em
            assert actual["f1"] == pytest.approx(f1)


def test_aliases_and_invalid_output():
    gold = dict(answer="United States", aliases=["USA"])
    assert score("USA", gold, "musique")["passed"]
    assert not score("USA", gold, "musique", valid=False)["passed"]


def test_durable_calls_never_repeat_completed_request(tmp_path):
    calls = Calls(tmp_path, tmp_path / "gate", 42)
    body = dict(model="fixture", input="public input")
    request = dict(stage="embedding", port=8011, route="/v1/embeddings", body=body)
    receipt = dict(
        status="complete", request_hash=digest(request), response={"answer": 1}
    )
    save(tmp_path / "calls/example.json", receipt)
    calls.http.close()  # A network request would now fail.
    assert calls.post("example", "embedding", 8011, "/v1/embeddings", body) == receipt
    with pytest.raises(RuntimeError, match="Changed request"):
        calls.post(
            "example", "embedding", 8011, "/v1/embeddings", dict(input="changed")
        )


@pytest.mark.parametrize("status", ["inflight", "error"])
def test_ambiguous_transport_not_silently_retried(tmp_path, status):
    calls = Calls(tmp_path, tmp_path / "gate", 42)
    request = dict(stage="generation", port=8000, route="/v1/chat/completions", body={})
    save(
        tmp_path / "calls/uncertain.json",
        dict(status=status, request_hash=digest(request)),
    )
    with pytest.raises(RuntimeError, match="Unresolved prior API"):
        calls.post("uncertain", "generation", 8000, "/v1/chat/completions", {})


def test_dataset_manifest_has_no_pilot_or_offline_leakage():
    root = REPO / "results/trimem_exg_qwen38_v0.1.0/pilot_20261011_01"
    for dataset in ("musique", "hotpotqa"):
        manifest = read(root / dataset / "manifest.json")
        assert len(set(manifest["online"])) == 500
        assert len(manifest["offline_collection"]) == 350
        assert len(manifest["offline_test"]) == 150
        assert set(manifest["offline_collection"]).isdisjoint(manifest["offline_test"])
        assert set(manifest["online"]).isdisjoint(manifest["pilot"])
        inputs = json.dumps(read(root / dataset / "inputs.json"))
        assert '"aliases"' not in inputs
        assert '"answer"' not in inputs
        assert '"supporting_facts"' not in inputs
        assert '"question_decomposition"' not in inputs
