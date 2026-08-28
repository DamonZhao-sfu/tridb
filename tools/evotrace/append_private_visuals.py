"""Embed the four ALE private-evaluation figures and their exact CSV data.

The operation is idempotent: it replaces only ``Visualizations`` and the four
private-plot support sheets, preserves every other workbook sheet, validates the
saved workbook, then atomically replaces the requested XLSX.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import zipfile
from pathlib import Path
from typing import Any, Sequence

from openpyxl import load_workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

FIGURES = [
    {
        "file": "table11_private_delta.png",
        "title": "Table 11 analogue: private performance delta",
        "description": "Seed to final public-best change on held-out private cases; bold red marks public-up/private-down overfitting.",
        "data_sheet": "Table11 Figure Data",
        "data_file": "table11_performance_delta.csv",
    },
    {
        "file": "table12_verdict_counts.png",
        "title": "Table 12 analogue: generalization verdict counts",
        "description": "Per-configuration counts of aligned, mild overfit, severe overfit, and no movement.",
        "data_sheet": "Table12 Figure Data",
        "data_file": "table12_verdict_counts.csv",
    },
    {
        "file": "figure12_verdict_counts.png",
        "title": "Figure 12: stacked generalization verdict counts",
        "description": "Graphical rendering of the same verdict counts used by the Table 12 analogue.",
        "data_sheet": "Table12 Figure Data",
        "data_file": "table12_verdict_counts.csv",
    },
    {
        "file": "figure5_public_vs_private.png",
        "title": "Figure 5: public improvement versus private improvement",
        "description": "Each point is a cell; the public-up/private-down quadrant indicates held-out overfitting.",
        "data_sheet": "Public-Private Plot Data",
        "data_file": "private_figure_data.csv",
    },
]

MANAGED_SHEETS = [
    "Private Plot Index",
    "Table11 Figure Data",
    "Table12 Figure Data",
    "Public-Private Plot Data",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def convert(value: str) -> Any:
    if value == "":
        return None
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def style_data_sheet(ws: Any) -> None:
    fill = PatternFill("solid", fgColor="D9EAF7")
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = fill
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for index in range(1, ws.max_column + 1):
        values = [
            str(ws.cell(row, index).value or "")
            for row in range(1, min(ws.max_row, 201) + 1)
        ]
        ws.column_dimensions[get_column_letter(index)].width = min(
            max(len(value) for value in values) + 2, 42
        )


def import_csv_sheet(
    wb: Any, report_dir: Path, sheet_name: str, file_name: str
) -> None:
    if sheet_name in wb.sheetnames:
        del wb[sheet_name]
    ws = wb.create_sheet(sheet_name)
    rows = read_csv(report_dir / file_name)
    for row_index, row in enumerate(rows, start=1):
        for column_index, value in enumerate(row, start=1):
            ws.cell(row_index, column_index, convert(value))
    style_data_sheet(ws)


def write_index(wb: Any, report_dir: Path) -> None:
    if "Private Plot Index" in wb.sheetnames:
        del wb["Private Plot Index"]
    ws = wb.create_sheet("Private Plot Index")
    headers = [
        "figure",
        "title",
        "interpretation",
        "data_sheet",
        "data_file",
        "figure_sha256",
        "data_sha256",
    ]
    for column, header in enumerate(headers, start=1):
        cell = ws.cell(1, column, header)
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="B4C7E7")
    for row, figure in enumerate(FIGURES, start=2):
        figure_path = report_dir / figure["file"]
        data_path = report_dir / figure["data_file"]
        values = [
            figure["file"],
            figure["title"],
            figure["description"],
            figure["data_sheet"],
            figure["data_file"],
            sha256(figure_path),
            sha256(data_path),
        ]
        for column, value in enumerate(values, start=1):
            ws.cell(row, column, value)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    widths = [34, 52, 90, 28, 38, 67, 67]
    for column, width in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(column)].width = width
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")


def write_visualizations(wb: Any, report_dir: Path) -> None:
    if "Visualizations" in wb.sheetnames:
        del wb["Visualizations"]
    ws = wb.create_sheet("Visualizations")
    ws.sheet_view.showGridLines = False
    ws.column_dimensions["A"].width = 18
    ws.column_dimensions["B"].width = 85
    row = 1
    for number, figure in enumerate(FIGURES, start=1):
        ws.cell(row, 1, f"Private Figure {number}")
        ws.cell(row, 1).font = Font(bold=True, size=12)
        ws.cell(row, 2, figure["title"])
        ws.cell(row, 2).font = Font(bold=True, size=12)
        ws.cell(row + 1, 2, figure["description"])
        ws.cell(row + 1, 2).alignment = Alignment(wrap_text=True)
        ws.cell(row + 2, 2, f"Data: {figure['data_sheet']} ({figure['data_file']})")

        image = XLImage(report_dir / figure["file"])
        scale = min(1050 / image.width, 600 / image.height)
        image.width = int(image.width * scale)
        image.height = int(image.height * scale)
        ws.add_image(image, f"A{row + 4}")
        # 20 px is a practical approximation of the default Excel row height.
        row += max(34, int(image.height / 20) + 8)


def update_manifest_sheet(wb: Any, report_dir: Path) -> None:
    if "Manifest & Sources" not in wb.sheetnames:
        return
    ws = wb["Manifest & Sources"]
    rows_to_delete = [
        row
        for row in range(2, ws.max_row + 1)
        if str(ws.cell(row, 1).value or "").startswith("private_visuals.")
    ]
    for row in reversed(rows_to_delete):
        ws.delete_rows(row)
    entries = [
        ("private_visuals.figure_count", 4),
        ("private_visuals.data_sheet_count", 3),
        ("private_visuals.report_dir", str(report_dir.resolve())),
    ]
    for figure in FIGURES:
        entries.append(
            (
                f"private_visuals.{Path(figure['file']).stem}.sha256",
                sha256(report_dir / figure["file"]),
            )
        )
    for key, value in entries:
        ws.append((key, value))


def reorder(wb: Any) -> None:
    preferred = ["Summary", "Visualizations", "Private Plot Index", *MANAGED_SHEETS[1:]]
    sheets = [wb[name] for name in preferred if name in wb.sheetnames]
    sheets.extend(sheet for sheet in wb.worksheets if sheet not in sheets)
    wb._sheets = sheets


def render(workbook: Path, report_dir: Path) -> None:
    for figure in FIGURES:
        for key in ("file", "data_file"):
            path = report_dir / figure[key]
            if not path.is_file():
                raise FileNotFoundError(path)

    wb = load_workbook(workbook)
    imported: set[tuple[str, str]] = set()
    for figure in FIGURES:
        key = (figure["data_sheet"], figure["data_file"])
        if key not in imported:
            import_csv_sheet(wb, report_dir, *key)
            imported.add(key)
    write_index(wb, report_dir)
    write_visualizations(wb, report_dir)
    update_manifest_sheet(wb, report_dir)
    reorder(wb)
    wb.calculation.fullCalcOnLoad = True
    wb.calculation.forceFullCalc = True

    temporary = workbook.with_suffix(".private-visuals.tmp.xlsx")
    wb.save(temporary)
    with zipfile.ZipFile(temporary) as archive:
        broken = archive.testzip()
        if broken:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"corrupt XLSX member: {broken}")
    check = load_workbook(temporary, read_only=False, data_only=False)
    if len(check["Visualizations"]._images) != 4:
        temporary.unlink(missing_ok=True)
        raise ValueError("expected four embedded private figures")
    expected_rows = {
        "Table11 Figure Data": len(
            read_csv(report_dir / "table11_performance_delta.csv")
        ),
        "Table12 Figure Data": len(read_csv(report_dir / "table12_verdict_counts.csv")),
        "Public-Private Plot Data": len(
            read_csv(report_dir / "private_figure_data.csv")
        ),
    }
    for name, rows in expected_rows.items():
        if check[name].max_row != rows:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"{name}: expected {rows} rows, got {check[name].max_row}")
    check.close()
    os.replace(temporary, workbook)
    digest = sha256(workbook)
    workbook.with_suffix(workbook.suffix + ".sha256").write_text(
        f"{digest}  {workbook.resolve()}\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    render(args.workbook, args.report_dir)
    print(args.workbook.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
