"""End-to-end durable orchestration contracts, using no DB or LLM."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from trimem.optimizer.adapters import NativeRetrievalAdapter
from trimem.optimizer.archive import TrialArchive
from trimem.optimizer.cli import main
from trimem.optimizer.controller import JointOptimizer
from trimem.optimizer.demo import demo_optimizer
from trimem.optimizer.evaluator import CohortObservation
from trimem.optimizer.ledger import MissingUsageError, TokenBudgetExhausted, TokenLedger
from trimem.optimizer.schema import CandidateSpec


class CharacterTokenizer:
    def encode(self, text, *, add_special_tokens=False):
        return list(text.encode())

    def decode(self, tokens):
        return bytes(tokens).decode()


def test_canonical_inactive_knobs_and_memory_budget_match_executor():
    a = CandidateSpec("no_memory_v1")
    b = replace(a, seed_count=-12, memory_token_cap=99999)
    assert a.action_hash == b.action_hash
    assert CandidateSpec.from_dict(b.to_dict()).to_dict() == a.to_dict()
    with pytest.raises(ValueError, match="8192"):
        CandidateSpec("vgr_v1", memory_token_cap=8193)
    with pytest.raises(ValueError):
        CandidateSpec("vgr_v1", schema_version=True)


def test_trial_pending_resume_out_of_order_and_idempotent_tell(tmp_path):
    optimizer = demo_optimizer(tmp_path)
    first, second = optimizer.ask(), optimizer.ask()
    assert first["action_hash"] != second["action_hash"]
    observed = optimizer.evaluate(second["trial_id"])
    optimizer.tell(
        second["trial_id"],
        CohortObservation.from_dict(json.loads(json.dumps(observed.to_dict()))),
    )
    resumed = JointOptimizer.restore(tmp_path)
    assert resumed.archive.get(first["trial_id"])["status"] == "pending"
    assert len(resumed.training_data()) == 1
    assert resumed.ask()["action_hash"] not in {
        first["action_hash"],
        second["action_hash"],
    }
    assert len(resumed.pareto()) == 1


def test_restore_rejects_changed_manifest_objective_reference(tmp_path):
    optimizer = demo_optimizer(tmp_path)
    optimizer.step()
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["search_settings"]["reference_tokens"] = 1234
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="integrity"):
        JointOptimizer.restore(tmp_path)


def test_restore_accepts_pre_v5_manifest_without_new_search_fields(tmp_path):
    from trimem.optimizer.schema import stable_hash

    demo_optimizer(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    for field in ("initial_strategy", "initial_random"):
        del manifest["search_settings"][field]  # as written by v4 code
    path.write_text(json.dumps(manifest))
    (tmp_path / "manifest.sha256").write_text(stable_hash(manifest) + "\n")
    resumed = JointOptimizer.restore(tmp_path)
    assert resumed.settings.initial_strategy == "per_plan"
    assert resumed.settings.initial_random == 0
    trial = resumed.ask()
    again = JointOptimizer.restore(tmp_path)
    assert again.archive.get(trial["trial_id"])["status"] == "pending"
    assert "initial_strategy" not in again.archive.manifest["search_settings"]
    manifest["search_settings"]["initial_strategy"] = "sobol"
    with pytest.raises(ValueError, match="search_settings"):
        JointOptimizer(
            resumed.space, resumed.settings, SimpleNamespace(manifest=manifest)
        )


def test_create_resumes_pre_v5_manifest_without_new_search_fields(tmp_path):
    """`cli run` resumes through create(); the stored manifest stays byte-identical."""
    from trimem.optimizer.schema import stable_hash

    first = demo_optimizer(tmp_path)
    first.step()
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    for field in ("initial_strategy", "initial_random"):
        del manifest["search_settings"][field]  # as written by v4 code
    path.write_text(json.dumps(manifest))
    (tmp_path / "manifest.sha256").write_text(stable_hash(manifest) + "\n")
    # Re-sign the existing trial for the old-format manifest identity.
    (trial_path,) = (tmp_path / "trials").glob("*.json")
    trial = json.loads(trial_path.read_text())
    trial["manifest_hash"] = stable_hash(manifest)
    trial_path.write_text(json.dumps(trial))
    before = path.read_bytes()
    resumed = demo_optimizer(tmp_path)  # JointOptimizer.create on the same dir
    assert path.read_bytes() == before
    assert resumed.archive.manifest_hash == stable_hash(manifest)
    assert len(resumed.training_data()) == 1
    resumed.step()
    assert len(JointOptimizer.restore(tmp_path).training_data()) == 2
    # A genuinely different setting is still refused.
    manifest["search_settings"]["initial_random"] = 3
    path.write_text(json.dumps(manifest))
    (tmp_path / "manifest.sha256").write_text(stable_hash(manifest) + "\n")
    with pytest.raises(ValueError, match="differs"):
        demo_optimizer(tmp_path)


def test_bo_failure_without_fallback_reserves_nothing(tmp_path, monkeypatch):
    """fallback_on_error=False: a surrogate failure raises before reserve()."""

    def broken(*args):
        raise RuntimeError("synthetic GP fit failure")

    monkeypatch.setattr("trimem.optimizer.search._qlognehvi", broken)
    optimizer = demo_optimizer(tmp_path)
    optimizer.settings = replace(optimizer.settings, fallback_on_error=False)
    with pytest.raises(RuntimeError, match="synthetic GP fit failure"):
        for _ in range(len(optimizer.space) + 1):
            optimizer.step()
    records = optimizer.archive.records()
    assert records and all(r["status"] == "observed" for r in records)
    assert all(r["proposal"]["strategy"] != "qlognehvi" for r in records)
    assert len(optimizer.training_data()) >= 2  # BO was due, so the fit ran


def test_tell_rejects_token_or_quality_aggregate_forgery(tmp_path):
    optimizer = demo_optimizer(tmp_path)
    trial = optimizer.ask()
    observation = optimizer.evaluator.evaluate(CandidateSpec.from_dict(trial["action"]))
    for forged in (
        replace(observation, total_tokens_mean=0),
        replace(observation, quality_mean=100),
    ):
        with pytest.raises(ValueError):
            optimizer.tell(trial["trial_id"], forged)
    assert optimizer.archive.get(trial["trial_id"])["status"] == "pending"
    optimizer.tell(trial["trial_id"], observation)
    assert optimizer.training_data()[0].total_tokens_mean > 0


def test_trial_action_and_observation_disk_corruption_detected(tmp_path):
    optimizer = demo_optimizer(tmp_path)
    trial = optimizer.step()
    path = tmp_path / "trials" / (trial["trial_id"] + ".json")
    altered = json.loads(path.read_text())
    altered["observation"]["total_tokens_mean"] = 0
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="integrity"):
        optimizer.training_data()


def test_atomic_reserve_does_not_repeat_finished_action_without_permission(tmp_path):
    optimizer = demo_optimizer(tmp_path)
    trial = optimizer.step()
    other_writer = TrialArchive(tmp_path, optimizer.archive.manifest)
    with pytest.raises(ValueError, match="reserved"):
        other_writer.reserve(CandidateSpec.from_dict(trial["action"]))


def test_missing_valid_checkpoint_is_not_silently_dropped_from_search(tmp_path):
    optimizer = demo_optimizer(tmp_path)
    optimizer.evaluator.units = tuple(
        replace(u, initial_fitness=None) for u in optimizer.evaluator.units
    )
    with pytest.raises(ValueError, match="initial checkpoint"):
        JointOptimizer(
            optimizer.space, optimizer.settings, optimizer.archive, optimizer.evaluator
        )


def test_no_memory_never_connects_embeds_or_uses_inactive_limits():
    def forbidden(*args):
        raise AssertionError("must not access a backend")

    adapter = NativeRetrievalAdapter(
        executor=forbidden,
        query_context=forbidden,
        tokenizer=CharacterTokenizer(),
        tokenizer_id="fixture",
        render_prompt=lambda context, state, memory: "base" + memory,
        model_context_window=100,
        max_output_tokens=10,
    )
    context = SimpleNamespace(
        protocol=SimpleNamespace(tokenizer_version="fixture"), ledger=TokenLedger(100)
    )
    result = adapter(
        CandidateSpec("no_memory_v1", memory_token_cap=99999), context, None
    )
    assert result.text == "" and result.packed.memory_tokens == 0
    assert result.receipt.actual_executions == 0 and result.receipt.closed
    assert context.ledger.total_tokens == 0
    context.ledger.record("spent", "generation", 99, 1)
    with pytest.raises(TokenBudgetExhausted):
        adapter(CandidateSpec("no_memory_v1"), context, None)


def test_malformed_usage_receipt_never_silently_becomes_zero_cost(monkeypatch):
    from trimem.optimizer.compiler import QueryContext
    import trimem.optimizer.adapters as module

    receipt = SimpleNamespace(
        execution_id="receipt",
        usage_complete=True,
        usage_events=[{"request_id": "bad", "input_tokens": 20, "output_tokens": 3}],
    )

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    cursor = Cursor()
    cursor.receipt = receipt
    backend = SimpleNamespace(open_selected=lambda plan: cursor)
    monkeypatch.setattr(module, "pack_stream", lambda *a, **kw: None)
    adapter = NativeRetrievalAdapter(
        executor=lambda ctx: backend,
        query_context=lambda ctx, state: QueryContext("math:test", (1.0, 0.0)),
        tokenizer=CharacterTokenizer(),
        tokenizer_id="fixture",
        render_prompt=lambda *args: "base",
        model_context_window=100,
        max_output_tokens=10,
    )
    context = SimpleNamespace(
        protocol=SimpleNamespace(tokenizer_version="fixture"), ledger=TokenLedger(100)
    )
    with pytest.raises(KeyError):
        adapter(CandidateSpec("vgr_score_v1"), context, None)
    with pytest.raises(MissingUsageError):
        _ = context.ledger.total_tokens


def test_cli_synthetic_run_and_fixed_plan_resumption(tmp_path, capsys):
    assert (
        main(
            [
                "demo",
                "--archive",
                str(tmp_path),
                "--fixed-plan",
                "vgr_v1",
                "--steps",
                "1",
            ]
        )
        == 0
    )
    optimizer = JointOptimizer.restore(tmp_path)
    assert optimizer.archive.manifest["provenance"]["synthetic"] is True
    assert optimizer.space.plan_ids == ("vgr_v1",)
    assert main(["status", "--archive", str(tmp_path)]) == 0
    assert "observed" in capsys.readouterr().out
