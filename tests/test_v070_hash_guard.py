"""T11: TRACE v0.7.0 never changes a file hashed by a live / paused / queued run (plan 2026-11-05 section 1).

The rules live in ``tools.trace_v070_hash_guard`` (also run by the smoke and by the queue script)."""

from __future__ import annotations

import pytest

from tools import trace_v070_hash_guard as hg


@pytest.mark.parametrize("run", hg.present(hg.GUARDED))
def test_guarded_manifest_hashes_match(run):  # GUARDED absent from disk is itself a report() violation
    assert hg.mismatches(run) == []


@pytest.mark.parametrize("run", hg.present(hg.OLDER))
def test_older_runs_have_no_new_drift(run):
    assert hg.new_drift(run) == []


def test_no_hashed_python_imports_v070():
    assert hg.v070_importers(hg.present(hg.protected())) == []


def test_report_ok():
    assert hg.report()["ok"]


def test_discovery_covers_the_floor_and_skips_v070_runs():
    found = set(hg.discover())
    assert set(hg.present(hg.GUARDED)) <= found and len(found) >= len(hg.present(hg.GUARDED + hg.OLDER))
    assert not any("trace_v070" in r for r in found)


def test_fails_closed_without_snapshot_baseline_or_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr(hg, "SNAPSHOT", tmp_path / "missing_before.json")
    r = hg.report()
    assert not r["ok"] and any("pre-v0.7.0 snapshot" in v for v in r["violations"])
    monkeypatch.undo()
    monkeypatch.setattr(hg, "BASELINE", tmp_path / "missing_baseline.json")
    assert any("discovered-run baseline" in v for v in hg.report()["violations"])
    monkeypatch.undo()
    monkeypatch.setattr(hg, "GUARDED", (*hg.GUARDED, "results/no_such_guarded_run"))
    r = hg.report()
    assert not r["ok"] and "missing protected manifest: results/no_such_guarded_run" in r["violations"]


def test_baseline_drift_rules(monkeypatch, tmp_path):
    f = tmp_path / "artifact.py"
    f.write_text("v1")
    recorded = hg.sha(f)
    monkeypatch.setattr(hg, "hashed", lambda run: {str(f): recorded})
    assert hg.baseline_drift("fake", {}) == []  # matches its manifest
    f.write_text("v2")  # drift not frozen in the baseline -> violation
    assert hg.baseline_drift("fake", {}) == [str(f)]
    assert hg.baseline_drift("fake", {str(f.resolve()): hg.sha(f)}) == []  # pre-existing, frozen
    f.write_text("v3")  # changed again after freezing -> violation
    assert hg.baseline_drift("fake", {str(f.resolve()): "old"}) == [str(f)]


def test_guarded_mismatch_is_not_hidden_by_the_discovered_baseline(monkeypatch, tmp_path):
    """Fix round 2: a guarded file that differs from its manifest fails the guard even when it equals the frozen
    discovered baseline (checks are aggregated independently)."""
    import json

    f = tmp_path / "artifact.txt"
    f.write_text("changed")
    run = "results/fake_guarded_run"
    manifest = {"live": {"artifacts": {str(f): "0" * 64}}}  # manifest hash != file
    snap, base = tmp_path / "before.json", tmp_path / "baseline.json"
    snap.write_text(json.dumps({"files": {}}))
    base.write_text(json.dumps({"files": {str(f.resolve()): hg.sha(f)}}))  # equals the file
    monkeypatch.setattr(hg, "GUARDED", (run,))
    monkeypatch.setattr(hg, "OLDER", ())
    monkeypatch.setattr(hg, "discover", lambda: [run])
    monkeypatch.setattr(hg, "_manifest", lambda r: manifest if r == run else None)
    monkeypatch.setattr(hg, "SNAPSHOT", snap)
    monkeypatch.setattr(hg, "BASELINE", base)
    r = hg.report()
    assert r["discovered_new_drift"] == {} and r["guarded"] == {run: [str(f)]}
    assert not r["ok"] and r["violations"] == [f"guarded {run}: {f}"]
