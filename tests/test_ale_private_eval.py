import json

import pytest

from tools.evotrace.ale_private_rescore import (
    canonical_code,
    code_sha256,
    generalization_label,
    load_cell,
)
from tools.evotrace.ale_private_tables import (
    aggregate_groups,
    load_rows,
    paired_rows,
    render,
)


def _write_cell(tmp_path):
    cell = tmp_path / "ale_ahc008__gem__r100__p010"
    (cell / "openevolve/best").mkdir(parents=True)
    checkpoint = cell / "openevolve/checkpoints/checkpoint_40/programs"
    checkpoint.mkdir(parents=True)
    seed = "# EVOLVE-BLOCK-START\nint seed;\n# EVOLVE-BLOCK-END\n"
    final = "int final;\n"
    receipt = {
        "status": "complete",
        "task_uid": "ale:ahc008",
        "arm": "gem",
        "physical_plan": "vfwd",
        "injection_frequency": 0.1,
        "seed_fitness": 10.0,
    }
    info = {"id": "winner", "metrics": {"combined_score": 15.0}}
    (cell / "run_receipt.json").write_text(json.dumps(receipt))
    (cell / "initial_program.py").write_text(seed)
    (cell / "openevolve/best/best_program.py").write_text(final)
    (cell / "openevolve/best/best_program_info.json").write_text(json.dumps(info))
    (checkpoint / "winner.json").write_text(
        json.dumps({"code": final, "metrics": {"combined_score": 15.0}})
    )
    (cell / "evolution_trace.jsonl").write_text("")
    return cell


def test_canonical_code_removes_only_harness_markers():
    code = "# EVOLVE-BLOCK-START\nint x;\n# EVOLVE-BLOCK-END\n"
    assert canonical_code(code) == "int x;\n"
    assert code_sha256(code) == code_sha256("int x;\n")


def test_checkpoint_is_authoritative_when_winner_absent_from_trace(tmp_path):
    artifact = load_cell(_write_cell(tmp_path))
    assert artifact.final_program_id == "winner"
    assert artifact.physical_plan == "vfwd"
    assert artifact.final_in_trace is False
    assert artifact.public_delta == 5.0


def test_paper_overfit_boundary_is_strictly_past_200():
    assert generalization_label(1, 1) == "aligned"
    assert generalization_label(1, -1) == "overfit(mild)"
    assert generalization_label(1, -200) == "overfit(mild)"
    assert generalization_label(1, -201) == "overfit(severe)"
    assert generalization_label(0, -500) == "no movement"


def test_group_aggregate_and_paired_control_delta(tmp_path):
    rows = [
        {
            "cell": "ale_ahc008__nocontext",
            "problem": "ahc008",
            "arm": "nocontext",
            "public_seed_fitness": 10,
            "public_final_fitness": 11,
            "public_delta": 1,
            "final_private_performance": 1000,
            "seed_private_performance": 900,
            "private_performance_delta_from_seed": 100,
            "private_performance_delta_vs_nocontext": None,
            "generalization_label": "aligned",
        },
        {
            "cell": "ale_ahc008__gem__r100__p010",
            "problem": "ahc008",
            "arm": "gem",
            "public_seed_fitness": 10,
            "public_final_fitness": 12,
            "public_delta": 2,
            "final_private_performance": 1200,
            "seed_private_performance": 900,
            "private_performance_delta_from_seed": 300,
            "private_performance_delta_vs_nocontext": 200,
            "generalization_label": "aligned",
        },
    ]
    source = tmp_path / "rows.json"
    source.write_text(json.dumps(rows))
    loaded = load_rows(source)
    summary = aggregate_groups(loaded)
    gem = next(row for row in summary if row["arm"] == "gem")
    assert gem["mean_private_performance"] == 1200
    assert gem["mean_private_performance_delta_vs_nocontext"] == 200
    assert paired_rows(loaded)[0]["private_performance_delta_vs_nocontext"] == 200
    with pytest.raises(ValueError):
        render(private_json=source, output_dir=tmp_path / "incomplete")

    full_rows = []
    for problem in (
        "ahc008",
        "ahc011",
        "ahc015",
        "ahc016",
        "ahc024",
        "ahc025",
        "ahc026",
        "ahc027",
        "ahc039",
        "ahc046",
    ):
        for template in rows:
            row = dict(template)
            row["problem"] = problem
            row["cell"] = row["cell"].replace("ahc008", problem)
            row["seed_sha256"] = f"seed-{problem}"
            row["final_sha256"] = f"final-{problem}-{row['arm']}"
            row["private_case_count"] = 150
            row["seed_private_rank"] = 2
            row["final_private_rank"] = 1
            full_rows.append(row)
    render_source = tmp_path / "full_rows.json"
    render_source.write_text(json.dumps(full_rows))
    output = tmp_path / "report"
    manifest = render(private_json=render_source, output_dir=output)
    assert manifest["cells_scored"] == 20
    assert manifest["completeness_gate"] == "passed"
    assert (output / "table11_performance_delta.csv").is_file()
    assert (output / "group_summary.csv").is_file()
    assert (output / "paired_vs_nocontext.csv").is_file()
    assert (output / "table11_private_delta.png").is_file()
    assert (output / "table12_verdict_counts.png").is_file()
    assert (output / "figure12_verdict_counts.png").is_file()
    assert (output / "figure5_public_vs_private.png").is_file()
