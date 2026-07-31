"""Render the three reproduced sections as Markdown.

Deliberately plain: one table per figure, every operating point a row, and the
caveats printed with the numbers rather than filed in a doc nobody opens beside
the table. A number whose measurement did not happen prints as ``n/a``, never as
zero.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from bench.agent_memory.serving import _atomic_write_json


def _num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number and (abs(number) >= 100_000 or abs(number) < 10**-digits):
        return f"{number:.3g}"
    return f"{number:,.{digits}f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    alignment = ["---"] + ["---:"] * (len(headers) - 1)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(alignment) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _spread_line(name: str, spread: Mapping[str, Any]) -> str:
    if spread.get("ratio") is None:
        return f"- {name}: not computable ({spread.get('arms', 0)} arm(s) measured)"
    return (
        f"- {name}: **{_num(spread['ratio'], 1)}x** "
        f"({_num(spread['min'])} … {_num(spread['max'])}, "
        f"{spread['arms']} arms)"
    )


def render(sections: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    models = manifest.get("models", {})
    energy = manifest.get("energy", {})
    parts: list[str] = [
        "# LongMemEval_S* on TriDB/GEM — [AM] §4.1, §4.2, §4.8",
        "",
        f"> {sections['scope']}",
        "",
        "## Operating labels",
        "",
        f"- answer / construction model: `{models.get('answer')}` "
        f"@ `{models.get('answer_base_url')}`",
        f"- embedding model: `{models.get('embedding')}` "
        f"(dim {models.get('embedding_dim')}) @ `{models.get('embedding_base_url')}`",
        f"- judge: `{models.get('judge')}` @ `{models.get('judge_base_url')}` "
        f"— protocol `{manifest.get('evaluation', {}).get('judge_protocol')}`",
        f"- GPU energy: {'sampled' if energy.get('available') else 'NOT sampled'}"
        + (f" via `{energy.get('method')}`" if energy.get("available") else "")
        + (f" — {energy.get('init_error')}" if not energy.get("available") else ""),
        "- absolute wallclock and joules are this box's, not the paper's H100;",
        "  only the spread across arms measured here is comparable.",
        "",
    ]

    rows41 = [
        [
            f"`{row['operating_point']}`",
            row["paradigm"],
            _num(row.get("accuracy")),
            _num(row.get("mean_qa_wallclock_per_query_seconds")),
            _num(row.get("mean_retrieval_per_query_seconds")),
            _num(row.get("mean_generation_per_query_seconds")),
            str(row.get("queries")),
        ]
        for row in sections["section_4_1"]["rows"]
    ]
    parts += [
        "## §4.1 — Serving latency vs accuracy (construction excluded)",
        "",
        _table(
            [
                "Operating point",
                "Paradigm",
                "Accuracy",
                "QA s/query",
                "Retrieval s/query",
                "Generation s/query",
                "Queries",
            ],
            rows41,
        ),
        "",
        _spread_line(
            "per-query serving latency spread",
            sections["section_4_1"]["serving_latency_spread"],
        ),
        f"- long-context baseline: {sections['section_4_1']['long_context_baseline']}",
        "",
    ]

    rows42 = [
        [
            f"`{row['operating_point']}`",
            _num(row.get("construction_wallclock_seconds"), 1),
            _num(row.get("retrieval_per_query_seconds")),
            _num(row.get("generation_per_query_seconds")),
            f"{row.get('model_calls')}",
            f"{row.get('construction_tokens'):,}"
            if row.get("construction_tokens") is not None
            else "n/a",
            _num(row.get("total_kilojoules"), 1),
            _num(row.get("joules_per_correct"), 1),
        ]
        for row in sections["section_4_2"]["rows"]
    ]
    parts += [
        "## §4.2 — Construction dominates the lifecycle",
        "",
        _table(
            [
                "Operating point",
                "Construct (s)",
                "Retrieval s/q",
                "Generation s/q",
                "Model calls",
                "Construct tokens",
                "Total kJ",
                "J / correct",
            ],
            rows42,
        ),
        "",
        _spread_line(
            "construction wallclock spread",
            sections["section_4_2"]["construction_wallclock_spread"],
        ),
        _spread_line(
            "lifecycle energy spread",
            sections["section_4_2"]["lifecycle_energy_spread"],
        ),
        _spread_line(
            "energy per correct answer spread",
            sections["section_4_2"]["energy_per_correct_spread"],
        ),
        "",
    ]

    rows48 = [
        [
            f"`{row['operating_point']}`",
            row.get("bound_regime", ""),
            _num(row.get("ttft_p50_seconds")),
            _num(row.get("total_p50_seconds")),
            _num(row.get("qa_p95_seconds")),
            _num(row.get("qa_p95_over_p50"), 2),
            _num(row.get("ttft_p95_over_p50"), 2),
        ]
        for row in sections["section_4_8"]["rows"]
    ]
    parts += [
        "## §4.8 — Serving latency structure",
        "",
        _table(
            [
                "Operating point",
                "Regime",
                "TTFT p50 (s)",
                "Total p50 (s)",
                "Total p95 (s)",
                "p95/p50",
                "TTFT p95/p50",
            ],
            rows48,
        ),
        "",
        _spread_line("effective TTFT spread", sections["section_4_8"]["ttft_spread"]),
        "",
        "## Not reproduced",
        "",
        f"- §4.7: {sections['section_4_7']}",
        "",
    ]
    return "\n".join(parts) + "\n"


def write(
    sections: Mapping[str, Any],
    manifest: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.md"
    sections_path = output_dir / "paper_sections.json"
    report_path.write_text(render(sections, manifest), encoding="utf-8")
    _atomic_write_json(sections_path, dict(sections))
    return {"report": report_path, "sections": sections_path}
