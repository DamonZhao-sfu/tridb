"""Hardware-free tests for GEM paper figures and Figure 9 input preparation."""

from __future__ import annotations

import ast
import copy
import csv
import json

from bench.agent_memory.gem_bench import comparison_export, export, figures, scaling


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sections():
    row = {
        "operating_point": "gem_conformant",
        "paradigm": "GEM",
        "label": "GEM-conformant",
    }
    return {
        "section_4_1": {
            "rows": [
                {
                    **row,
                    "accuracy": 0.5,
                    "mean_qa_wallclock_per_query_seconds": 2.0,
                    "mean_retrieval_per_query_seconds": 0.1,
                    "mean_generation_per_query_seconds": 1.9,
                    "queries": 10,
                }
            ]
        },
        "section_4_2": {
            "rows": [
                {
                    **row,
                    "construction_wallclock_seconds": 20.0,
                    "retrieval_per_query_seconds": 0.1,
                    "generation_per_query_seconds": 1.9,
                    "lifecycle_wallclock_seconds": 40.0,
                    "construction_calls": 4,
                    "qa_calls": 10,
                    "construction_tokens": 1000,
                    "qa_tokens": 2000,
                    "total_kilojoules": None,
                    "joules_per_correct": None,
                }
            ]
        },
        "section_4_8": {
            "rows": [
                {
                    **row,
                    "ttft_p50_seconds": 1.5,
                    "total_p50_seconds": 2.0,
                    "post_first_token_streaming_p50_seconds": 0.5,
                    "qa_p50_seconds": 2.0,
                    "qa_p95_seconds": 3.0,
                    "qa_p95_over_p50": 1.5,
                    "ttft_p95_over_p50": 1.2,
                }
            ]
        },
    }


def _scale_results(point="gem_conformant"):
    points = []
    for budget, construction, footprint in (
        (64 * 1024, 1.0, 1024 * 1024),
        (128 * 1024, 2.0, 2 * 1024 * 1024),
    ):
        points.append(
            {
                "requested_tokens": budget,
                "actual_input_tokens": budget - 100,
                "operating_point": point,
                "construction": {"seconds": construction},
                "tokens": {
                    "construction_embed_tokens": budget,
                    "construction_prompt_tokens": 0,
                    "construction_completion_tokens": 0,
                },
                "footprint": {
                    "logical_total_bytes": footprint,
                    "physical_isolated": False,
                },
                "retrieval": {"end_to_end_seconds": {"p50": 0.01, "p95": 0.02}},
            }
        )
    return {"points": points, "repeats": 1, "operating_point": point}


def test_render_figures_writes_png_pdf_csv_and_manifest(tmp_path):
    input_dir = tmp_path / "run"
    output_dir = input_dir / "figures"
    _write_json(input_dir / "paper_sections.json", _sections())
    _write_json(
        input_dir / "gem_conformant" / "summary.json",
        {"accuracy": {"wilson_95": [0.3, 0.7]}},
    )
    scale_path = tmp_path / "scale_results.json"
    _write_json(scale_path, _scale_results())

    manifest = figures.render_figures(
        input_dir=input_dir,
        output_dir=output_dir,
        points=["gem_conformant"],
        scale_results=scale_path,
    )

    for number, stem in (
        (2, "latency_accuracy"),
        (3, "phase_breakdown"),
        (9, "scaling"),
        (10, "effective_ttft"),
        (11, "tail_latency"),
    ):
        for suffix in ("png", "pdf"):
            path = output_dir / f"figure{number}_{stem}.{suffix}"
            assert path.stat().st_size > 0
    with (output_dir / "figure_data.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["figure"] for row in rows} == {"2", "3", "9", "10", "11"}
    assert manifest["selected_points"] == ["gem_conformant"]
    assert (output_dir / "plot_manifest.json").exists()


def test_scale_only_render_does_not_need_core_sections(tmp_path):
    scale_path = tmp_path / "scale_results.json"
    output_dir = tmp_path / "figures"
    _write_json(scale_path, _scale_results())

    manifest = figures.render_scale_figure(
        scale_results=scale_path, output_dir=output_dir
    )

    assert (output_dir / "figure9_scaling.png").stat().st_size > 0
    assert (output_dir / "figure9_scaling.pdf").stat().st_size > 0
    assert (output_dir / "figure9_data.csv").exists()
    assert manifest["selected_points"] == "Figure 9 scaling only"


def test_render_figures_overlays_multiple_scale_results(tmp_path):
    input_dir = tmp_path / "run"
    output_dir = input_dir / "figures"
    _write_json(input_dir / "paper_sections.json", _sections())
    _write_json(
        input_dir / "gem_conformant" / "summary.json",
        {"accuracy": {"wilson_95": [0.3, 0.7]}},
    )
    gem_scale = tmp_path / "gem_scale.json"
    embedrag_scale = tmp_path / "embedrag_scale.json"
    _write_json(gem_scale, _scale_results("gem_conformant"))
    _write_json(embedrag_scale, _scale_results("II_embedrag"))

    manifest = figures.render_figures(
        input_dir=input_dir,
        output_dir=output_dir,
        points=["gem_conformant"],
        scale_comparison_results=[embedrag_scale, gem_scale],
    )

    assert (output_dir / "figure9_scaling_comparison.png").stat().st_size > 0
    assert (output_dir / "figure9_scaling_comparison.pdf").stat().st_size > 0
    assert manifest["figure9"]["operating_points"] == [
        "II_embedrag",
        "gem_conformant",
    ]
    with (output_dir / "figure_data.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert any(row["operating_point"].startswith("II_embedrag:") for row in rows)


class _CharacterEncoding:
    def encode(self, text, **kwargs):
        return list(text)


def test_scaling_prefixes_are_nested_and_stop_at_complete_sessions():
    sessions = [f"session-{index}-" + "x" * 20 for index in range(20)]
    prefixes = scaling.build_prefixes(sessions, [100, 180], _CharacterEncoding())

    first = ast.literal_eval(prefixes[0]["context"])
    second = ast.literal_eval(prefixes[1]["context"])
    assert second[: len(first)] == first
    assert prefixes[0]["actual_tokens"] <= 100
    assert prefixes[1]["actual_tokens"] <= 180
    assert prefixes[0]["sessions"] < prefixes[1]["sessions"]


def test_physical_delta_is_relation_wise_and_totalled():
    before = {
        "public.gem_unit": {
            "heap_bytes": 100,
            "index_bytes": 50,
            "total_bytes": 150,
        }
    }
    after = {
        "public.gem_unit": {
            "heap_bytes": 140,
            "index_bytes": 60,
            "total_bytes": 200,
        },
        "public.gem_transition": {
            "heap_bytes": 20,
            "index_bytes": 10,
            "total_bytes": 30,
        },
    }
    delta = scaling._physical_delta(before, after)
    assert delta["physical_delta_bytes"] == 80
    assert delta["relations"]["public.gem_unit"]["total_bytes"] == 50
    assert delta["relations"]["public.gem_transition"]["total_bytes"] == 30


def test_export_bundle_writes_commit_ready_csv_figures_and_manifest(tmp_path):
    core_dir = tmp_path / "run"
    figures_dir = core_dir / "figures"
    _write_json(core_dir / "paper_sections.json", _sections())
    _write_json(
        core_dir / "gem_conformant" / "summary.json",
        {"accuracy": {"wilson_95": [0.3, 0.7]}},
    )
    _write_json(core_dir / "run_manifest.json", {"schema_version": "test"})
    _write_json(figures_dir / "plot_manifest.json", {"schema_version": "test"})
    for stem in export.FIGURE_STEMS:
        for suffix in ("png", "pdf"):
            path = figures_dir / f"{stem}.{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"figure")

    scale_path = tmp_path / "scale_results.json"
    scale_input_manifest = tmp_path / "scale_input_manifest.json"
    _write_json(scale_path, _scale_results())
    _write_json(scale_input_manifest, {"schema_version": "test-input"})
    output_dir = tmp_path / "results"

    manifest = export.export_bundle(
        core_dir=core_dir,
        scale_results=scale_path,
        scale_input_manifest=scale_input_manifest,
        figures_dir=figures_dir,
        output_dir=output_dir,
        point="gem_conformant",
    )

    with (output_dir / "metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["figure"] for row in rows} == {"2", "3", "9", "10", "11"}
    assert (output_dir / "figures" / "figure9_scaling.png").read_bytes() == b"figure"
    assert (output_dir / "raw" / "scale_input_manifest.json").exists()
    assert (output_dir / "README.md").exists()
    assert manifest["source_boundaries"]["large_dataset_included"] is False
    assert (output_dir / "MANIFEST.json").exists()


def test_export_comparison_bundle_includes_all_core_and_scale_points(tmp_path):
    core_dir = tmp_path / "run"
    figures_dir = core_dir / "figures"
    sections = _sections()
    for section in ("section_4_1", "section_4_2", "section_4_8"):
        embedrag = copy.deepcopy(sections[section]["rows"][0])
        embedrag.update(operating_point="II_embedrag", paradigm="II", label="embedRAG")
        sections[section]["rows"].insert(0, embedrag)
    _write_json(core_dir / "paper_sections.json", sections)
    _write_json(core_dir / "run_manifest.json", {"serial_execution": True})
    for point in ("II_embedrag", "gem_conformant"):
        _write_json(
            core_dir / point / "summary.json",
            {"accuracy": {"wilson_95": [0.3, 0.7]}},
        )
    _write_json(figures_dir / "plot_manifest.json", {"schema_version": "test"})
    for stem in comparison_export.FIGURE_STEMS:
        for suffix in ("png", "pdf"):
            path = figures_dir / f"{stem}.{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"figure")
    embedrag_scale = tmp_path / "embedrag_scale.json"
    gem_scale = tmp_path / "gem_scale.json"
    _write_json(embedrag_scale, _scale_results("II_embedrag"))
    _write_json(gem_scale, _scale_results("gem_conformant"))
    output_dir = tmp_path / "comparison"

    manifest = comparison_export.export_comparison_bundle(
        core_dir=core_dir,
        scale_results=[embedrag_scale, gem_scale],
        figures_dir=figures_dir,
        output_dir=output_dir,
        points=["II_embedrag", "gem_conformant"],
    )

    with (output_dir / "metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["operating_point"] for row in rows} == {
        "II_embedrag",
        "gem_conformant",
    }
    assert (output_dir / "figures" / "figure9_scaling_comparison.png").exists()
    assert (output_dir / "raw" / "II_embedrag_scale_results.json").exists()
    assert manifest["comparability"]["official_external_systems"] is False
