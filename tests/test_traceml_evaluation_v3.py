"""Nine-target RANZCR evaluation contract with agent-reported validation (spec Addendum D.8).

Artifact-contract v3 tests (receipts, node-local outputs, Landlock, process-group timeout,
strict pred_<target> OOF format) and the in-run OOF audit tests were reverted per D.8.2/D.8.3 and
moved to .superpowers/sdd/2026-09-12-traceml-vsgt-ablation/quarantine_codex_v3/.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bench.agent_memory.traceml.vsgt_mle import runtime_v2
from bench.agent_memory.traceml.vsgt_mle.evaluation_contract_v3 import (
    RANZCR_EXCLUDED,
    RANZCR_TARGETS,
    VERSION,
    instruction,
    prepare_ranzcr_view,
)

ROOT = Path(__file__).resolve().parents[1]
ALL_11 = [*RANZCR_TARGETS, *RANZCR_EXCLUDED]
RANZCR_DIR = (
    ROOT
    / "third_party/mle-bench/mlebench/competitions/ranzcr-clip-catheter-line-classification"
)


def test_version_names_the_agent_reported_protocol():
    assert VERSION == "public_oof5_v2_ranzcr9"


def test_nine_targets_match_the_mlebench_grader():
    if not (RANZCR_DIR / "classes.py").is_file():
        pytest.skip("MLE-bench checkout not present")
    namespace: dict = {}
    exec((RANZCR_DIR / "classes.py").read_text(), namespace)
    assert tuple(namespace["CLASSES"]) == RANZCR_TARGETS
    assert set(RANZCR_EXCLUDED) == {"CVC - Normal", "Swan Ganz Catheter Present"}


def test_public_nine_label_view_preserves_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    train = pd.DataFrame({"StudyInstanceUID": ["x"], **{c: [0] for c in ALL_11}})
    train.to_csv(source / "train.csv", index=False)
    train.to_csv(source / "sample_submission.csv", index=False)
    (source / "description.md").write_text(
        "11 labels\n- `CVC - Normal`\n- `Swan Ganz Catheter Present`\n"
    )
    (source / "images").mkdir()
    original = (source / "train.csv").read_bytes()
    view = tmp_path / "view"
    prepare_ranzcr_view(source, view)
    assert set(pd.read_csv(view / "train.csv")) == {"StudyInstanceUID", *RANZCR_TARGETS}
    assert list(pd.read_csv(view / "sample_submission.csv")) == [
        "StudyInstanceUID",
        *RANZCR_TARGETS,
    ]
    assert (source / "train.csv").read_bytes() == original
    assert (view / "images").is_symlink()
    description = (view / "description.md").read_text()
    assert "9 labels" in description and "`CVC - Normal`" not in description


def test_instruction_names_nine_targets_without_oof_artifacts():
    text = instruction()
    assert VERSION in text and all(t in text for t in RANZCR_TARGETS)
    for forbidden in ("pred_", "ONLY", "OOF", "validation_oof", "audit"):
        assert forbidden not in text


def test_validation_prompt_appends_contract_for_ranzcr_only(tmp_path):
    from bench.agent_memory.traceml.vsgt_mle.validation_v2 import (
        METRICS,
        instruction as prompt,
    )

    assert "9" in METRICS["ranzcr"]
    assert instruction() in prompt("ranzcr", tmp_path / "folds.csv")
    for task in ("aes2", "gquest", "uspatent", "ventilator"):
        text = prompt(task, tmp_path / "folds.csv")
        assert VERSION not in text and "pred_" not in text
    for task in ("ranzcr", "aes2", "gquest", "uspatent", "ventilator"):
        text = prompt(task, tmp_path / "folds.csv")
        assert "validation_oof" not in text and "audit" not in text.lower()
        assert "Report Final Validation Score" in text


def _ranzcr_frames():
    rng = np.random.default_rng(0)
    truth = pd.DataFrame({c: rng.integers(0, 2, 40) for c in ALL_11})
    truth.insert(0, "StudyInstanceUID", [f"1.2.{i}" for i in range(40)])
    pred = truth[["StudyInstanceUID"]].assign(**{c: rng.random(40) for c in ALL_11})
    return truth, pred


def test_offline_grade_matches_direct_per_class_auc_and_accepts_extra_columns():
    # scikit-learn ships in the TraceML evaluator env (.traceml_envs/), not the
    # buildable-anywhere dev venv; grade() itself needs it, so skip when absent.
    pytest.importorskip("sklearn")
    from sklearn.metrics import roc_auc_score

    from bench.agent_memory.traceml.vsgt_mle.ranzcr_grade_v2 import grade

    truth, pred = _ranzcr_frames()
    shuffled = pred.sample(frac=1, random_state=1).assign(extra=0.5)
    result = grade(shuffled, truth)
    direct = np.mean([roc_auc_score(truth[c], pred[c]) for c in RANZCR_TARGETS])
    assert result["primary_9"] == pytest.approx(direct)
    assert result["contract"] == VERSION
    with pytest.raises(ValueError, match="missing"):
        grade(pred.drop(columns=[RANZCR_TARGETS[-1]]), truth)


def test_offline_grade_agrees_with_mlebench_grader():
    import sys

    sys.path.insert(0, str(ROOT / "third_party/mle-bench"))
    try:
        from mlebench.competitions.utils import prepare_for_auroc_metric  # noqa: F401
        import importlib

        mle = importlib.import_module(
            "mlebench.competitions.ranzcr-clip-catheter-line-classification.grade"
        )
    except Exception as exc:  # optional dependency chain of the MLE-bench package
        pytest.skip(f"MLE-bench grader not importable here: {exc}")
    finally:
        sys.path.pop(0)
    from bench.agent_memory.traceml.vsgt_mle.ranzcr_grade_v2 import grade

    truth, pred = _ranzcr_frames()
    assert grade(pred, truth)["primary_9"] == pytest.approx(mle.grade(pred, truth))


# ---- launch plan (prepare_run only; nothing is launched) -------------------------------------

PROTOCOL = ROOT / "results/traceml_gem_v0.2.0/validation/public_oof5_v2/manifest.json"


def _plan(tmp_path, task, seed, gpu):
    job = dict(
        next(j for j in runtime_v2.matrix("evalv3") if j["task"] == task), seed=seed
    )
    if not (
        ROOT
        / "dataset/mle-bench/cache"
        / job["slug"]
        / "prepared/public/description.md"
    ).is_file():
        pytest.skip("public MLE-bench data not present")
    if seed == 41 and not PROTOCOL.is_file():
        pytest.skip("frozen public five-fold protocol not present")
    snapshot = tmp_path / "snapshot"
    if not snapshot.exists():
        skill = snapshot / "bench/agent_memory/traceml/vsgt_mle/skill_block_60min.md"
        skill.parent.mkdir(parents=True)
        shutil.copy2(
            ROOT / "bench/agent_memory/traceml/vsgt_mle/skill_block_60min.md", skill
        )
        runtime_v2.atomic_json(
            snapshot / "source_manifest.json",
            {"files": {str(skill.relative_to(snapshot)): runtime_v2.sha(skill)}},
        )
        database = tmp_path / "corpus/db/database.json"
        database.parent.mkdir(parents=True)
        database.write_text("{}")
        (tmp_path / "corpus/manifest.json").write_text(
            json.dumps({"purpose": "full release corpus preparation"})
        )
    out = tmp_path / f"out_{task}_{seed}"
    command, env = runtime_v2.prepare_run(
        snapshot=snapshot,
        out=out,
        job=job,
        gpu=gpu,
        database=tmp_path / "corpus/db/database.json",
    )
    config = json.loads((out / "run_config.json").read_text())
    skill_text = Path(env["SKILL_BLOCK"]).read_text()
    data_dir = Path(
        next(o for o in command if o.startswith("data_dir="))[len("data_dir=") :]
    )
    return config, env, skill_text, data_dir


@pytest.mark.parametrize("task", ["ranzcr", "aes2"])
def test_launch_plan_is_agent_reported_without_oof_artifacts(tmp_path, task):
    config, env, skill_text, data_dir = _plan(tmp_path, task, 41, 0)
    assert "require_oof" not in config and "VSGT_REQUIRE_OOF" not in env
    assert "VSGT_ARTIFACT_CONTRACT" not in env
    assert config["validation_score_source"] == "agent_reported"
    assert config["evaluation_contract"] == VERSION
    assert config["validation"]["contract"] == VERSION
    assert "validation_oof" not in skill_text and "audit" not in skill_text.lower()
    assert (VERSION in skill_text) == (task == "ranzcr")
    assert (data_dir.name == "public_ranzcr9") == (task == "ranzcr")


def test_ranzcr_view_applies_for_any_seed(tmp_path):
    config, env, skill_text, data_dir = _plan(tmp_path, "ranzcr", 7, 1)
    assert config["validation"] is None
    assert data_dir.name == "public_ranzcr9"
    assert list(pd.read_csv(data_dir / "sample_submission.csv", nrows=1)) == [
        "StudyInstanceUID",
        *RANZCR_TARGETS,
    ]
    assert VERSION in skill_text and "validation_oof" not in skill_text
    assert config["evaluation_contract"] == VERSION


@pytest.mark.parametrize("seed", [41, 7])
def test_ranzcr_sandbox_ro_keeps_the_original_public_dir_readable_for_every_seed(
    tmp_path, seed
):
    # Regression (R56): prepare_ranzcr_view symlinks test/, train/, train_annotations.csv into the
    # original public dir's resolved real paths for every RANZCR run, not only seed 41. Landlock's
    # read-only allowlist must therefore include the original public dir for every RANZCR seed.
    original_public = (
        ROOT
        / "dataset/mle-bench/cache"
        / runtime_v2.SLUGS["ranzcr"]
        / "prepared/public"
    ).resolve()
    if not (original_public / "description.md").is_file():
        pytest.skip("public MLE-bench data not present")
    config, env, skill_text, data_dir = _plan(tmp_path, "ranzcr", seed, 0)
    ro = config["sandbox_ro"]
    assert str(original_public) in ro
    assert str(data_dir.resolve()) in ro


def test_aes2_sandbox_ro_does_not_contain_the_ranzcr_original_public_dir(tmp_path):
    ranzcr_public = (
        ROOT
        / "dataset/mle-bench/cache"
        / runtime_v2.SLUGS["ranzcr"]
        / "prepared/public"
    ).resolve()
    config, env, skill_text, data_dir = _plan(tmp_path, "aes2", 41, 0)
    assert str(ranzcr_public) not in config["sandbox_ro"]
