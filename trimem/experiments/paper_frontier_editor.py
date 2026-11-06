"""Embed an offline paper-curve editor; preserve every other atlas section."""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import re
from datetime import UTC, datetime
from html import escape as html_escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "triMEM_optimizer/it20_summary/qa-trajectory.html"
START, END = "<!-- paper-frontier-editor:start -->", "<!-- paper-frontier-editor:end -->"
KEYS = ("alfworld", "sciworld", "webshop")
REVISION = ROOT / "paper/sigmod/revision_20261024_curve_editor"
E2_DATA = REVISION / "e2_test_points.json"  # ALFWorld (unchanged); other datasets: e2_test_points_<key>.json
HYDRA_DATA = REVISION / "hydra_local_test_points.json"
AUTORAG_DATA = REVISION / "agentic_autorag_test_points.json"
AUTORAG_OBSERVED = REVISION / "agentic_autorag_observed_points.json"
UNIFIED_TEST_DATA = REVISION / "unified_search_test_points.json"
E2_TITLES = {"alfworld": "ALFWorld exploration: MO-TPE vs TRACE",
             "sciworld": "ScienceWorld exploration: search methods",
             "webshop": "WebShop exploration: search methods"}


def e2_path(dataset: str) -> Path:
    """One E2 document per dataset; ALFWorld keeps its original file."""
    if dataset not in KEYS:
        raise ValueError(f"Unsupported E2 dataset {dataset!r}")
    return E2_DATA if dataset == "alfworld" else E2_DATA.parent / f"e2_test_points_{dataset}.json"


def e2_documents() -> dict[str, dict]:
    """Existing per-dataset E2 documents, validated, in KEYS order."""
    docs = {}
    for key in KEYS:
        path = e2_path(key)
        if path.exists():
            doc = json.loads(path.read_text())
            validate_e2(doc)
            if doc["dataset"] != key:
                raise ValueError(f"{path}: E2 document for {doc['dataset']}")
            docs[key] = doc
    return docs


def validate_e2(document: dict) -> None:
    dataset = document.get("dataset")
    if dataset not in KEYS or document.get("schema") != f"{dataset}-validation-pareto-test-v1":
        raise ValueError("Unsupported E2 test points")
    points = document.get("points", [])
    if not points or len({p["id"] for p in points}) != len(points):
        raise ValueError("Empty or duplicate E2 points")
    for p in points:
        if p.get("split") != "test" or p.get("n") != 200 or p.get("kind") != "e2":
            raise ValueError("E2 editor points require complete 200-task TEST coverage")
        if not all(isinstance(p.get(k), (int, float)) and math.isfinite(p[k]) for k in ("q", "tok", "lat")):
            raise ValueError("Nonfinite E2 point")


def validate_autorag(document: dict) -> None:
    if document.get("schema") != "autorag-source-test-points-v1":
        raise ValueError("Unsupported Agentic AutoRAG points")
    points = document.get("points", [])
    if not points or len({p.get("id") for p in points}) != len(points):
        raise ValueError("Empty or duplicate Agentic AutoRAG points")
    for p in points:
        if (p.get("dataset") not in KEYS or p.get("split") != "test" or p.get("n") != 200
                or p.get("kind") != "e2" or p.get("method") != "Agentic AutoRAG"):
            raise ValueError("Agentic AutoRAG editor points require 200-task TEST results")
        if not all(isinstance(p.get(k), (int, float)) and math.isfinite(p[k]) for k in ("q", "tok", "lat")):
            raise ValueError("Nonfinite Agentic AutoRAG point")


def validate_autorag_observed(document: dict) -> None:
    if document.get('schema') != 'autorag-observed-points-v1':
        raise ValueError('Unsupported Agentic AutoRAG observations')
    points = document.get('points', [])
    if not points or len({p.get('id') for p in points}) != len(points):
        raise ValueError('Empty or duplicate Agentic AutoRAG observations')
    for p in points:
        split = p.get('split')
        if (p.get('dataset') not in KEYS or split not in ('train', 'dev') or p.get('n') != 100
                or p.get('kind') != 'e2' or p.get('method') != f'Agentic AutoRAG ({split.upper()})'):
            raise ValueError('Agentic AutoRAG observations require labeled 100-task TRAIN or DEV data')
        if not all(isinstance(p.get(k), (int, float)) and math.isfinite(p[k]) for k in ('q', 'tok', 'lat')):
            raise ValueError('Nonfinite Agentic AutoRAG observation')


def publish_e2(document: dict, svg: str) -> None:
    validate_e2(document)
    if "<svg" not in svg or "</svg>" not in svg:
        raise ValueError("Missing E2 plot")
    with PAGE.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # One atomic document binds plotted metrics and SVG to the same generation.
        document = {**document, "plot_svg": svg[svg.index("<svg"):svg.index("</svg>") + 6]}
        path = e2_path(document["dataset"])
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, json.dumps(document, indent=2) + "\n")
    refresh()


def frontier(points: list[dict], cost: str) -> list[str]:
    selected = [p for p in points if not any(
        o["q"] >= p["q"] and o[cost] <= p[cost]
        and (o["q"] > p["q"] or o[cost] < p[cost]) for o in points
    )]
    seen, ids = set(), []
    for p in sorted(selected, key=lambda p: (p[cost], p["q"], p["id"])):
        coordinate = (p[cost], p["q"])
        if coordinate not in seen:
            seen.add(coordinate)
            ids.append(p["id"])
    return ids


def collect(html: str) -> dict:
    from trimem.experiments.perquery_sweep_page import point

    raw = re.search(r'<script[^>]*id="fd-data"[^>]*>(.*?)</script>', html, re.S).group(1)
    source = json.loads(raw)
    result = {"schema": "trimem-paper-curve-data-v1", "generated": datetime.now(UTC).isoformat(),
              "datasets": [], "defaults": {}}
    for key in KEYS:
        d = next(d for d in source["datasets"] if d["key"] == key)
        sweep = d["pqs"]
        points = [dict(p, id=p["arm"], kind="perquery", source="original sweep") for p in sweep["points"]]
        supplemental = {"status": "not run"}
        if key == "webshop":
            extra = ROOT / sweep["dir"] / "../pq_sweep_qualityfirst/report_pq_sweep.json"
            if extra.exists():
                report = json.loads(extra.read_text())
                original = json.loads((ROOT / sweep["dir"] / "report_pq_sweep.json").read_text())
                if report.get("protocol") != "perquery-weight-sweep-test-v1":
                    raise ValueError("Unexpected quality-first report protocol")
                for field in ("dataset", "runtime_profile", "concurrency", "quality_reuse"):
                    if report.get(field) != original.get(field):
                        raise ValueError(f"Incompatible quality-first report: {field}")
                if report.get("weight_set") != "quality-first-v1":
                    raise ValueError("Unexpected supplemental weight set")
                if not report["artifact"]["sha"].startswith(sweep["artifact"]):
                    raise ValueError("Quality-first and original sweeps use different routers")
                supplemental = {"status": "complete" if report["complete"] else "partial",
                                "complete_weights": report["complete_weights"], "weights": report["weights"],
                                "report": str(extra.resolve())}
                ids = {p["id"] for p in points}
                for r in report["points"].values():
                    if r["complete"] and r["arm"] not in ids:
                        assert r["covered"] == r["n"] == 200
                        p = point(r)
                        points.append(dict(p, id=p["arm"], kind="perquery", source="quality-first sweep"))
                        ids.add(p["arm"])
        for p in sweep["over"]:
            if p["kind"] == "perquery" or (key != "sciworld" and p["kind"] == "plan"):
                continue
            points.append(dict(p, source="reference"))
        for i, p in enumerate(points):
            p["short"] = f"W{i+1:02d}" if p["kind"] == "perquery" else p["id"].replace("Opt-OneConfig-", "")
            if p["id"] == "NoMemory":
                p["short"] = "No memory"
            if p["id"] == "VGR":
                p["short"] = "VGR"
        entry = {"key": key, "name": d["name"], "metric": d["metric"], "points": points,
                 "artifact": sweep["artifact"], "n": d["n"], "supplemental": supplemental}
        result["datasets"].append(entry)
        pool = [p for p in points if p["kind"] == "perquery" or
                (key != "sciworld" and p["id"] != "NoMemory")]
        defaults = {cost: frontier(pool, cost) for cost in ("tok", "lat")}
        extras = ([(.888889, 0., .111111), (.8, 0., .2)] if key == "alfworld" else
                  [(1., 0., 0.), (.8, 0., .2)] if key == "sciworld" else [])
        for cost in defaults:
            for weights in extras:
                p = next(p for p in points if p.get("w") == list(weights))
                if p["id"] not in defaults[cost]:
                    defaults[cost].append(p["id"])
        result["defaults"][key] = defaults
    for key, extra in e2_documents().items():
        entry = next(d for d in result["datasets"] if d["key"] == key)
        if {p["id"] for p in entry["points"]} & {p["id"] for p in extra["points"]}:
            raise ValueError("E2 point identity collides with existing editor data")
        entry["points"].extend(extra["points"])
        # Existing defaults and all saved selections retain their point identities.
        entry["e2_source"] = extra["test_plan"]
    if HYDRA_DATA.exists():
        extra = json.loads(HYDRA_DATA.read_text())
        if extra.get("schema") != "hydra-local-test-points-v1" or extra.get("dataset") != "alfworld":
            raise ValueError("Unsupported Hydra Local TEST points")
        entry = next(d for d in result["datasets"] if d["key"] == "alfworld")
        additions = extra["points"]
        existing = {p["id"] for p in entry["points"]}
        if any(p["id"] in existing or p.get("kind") != "hydra" or p.get("n") != 200
               or p.get("split") != "test" or not all(
                   isinstance(p.get(k), (int, float)) and math.isfinite(p[k]) for k in ("q", "tok", "lat")
               ) for p in additions) or len({p["id"] for p in additions}) != len(additions):
            raise ValueError("Invalid or duplicate Hydra Local TEST points")
        entry["points"].extend(additions)
        for ax in ("tok", "lat"):
            result["defaults"]["alfworld"][ax].extend(extra["default_ids"])
    if AUTORAG_DATA.exists():
        extra = json.loads(AUTORAG_DATA.read_text())
        validate_autorag(extra)
        for point in extra["points"]:
            entry = next(d for d in result["datasets"] if d["key"] == point["dataset"])
            if point["id"] in {p["id"] for p in entry["points"]}:
                raise ValueError("Agentic AutoRAG point identity collides")
            entry["points"].append(point)
            for axis in ("tok", "lat"):
                if point.get(f"default_{axis}", True):
                    result["defaults"][point["dataset"]][axis].append(point["id"])
    if AUTORAG_OBSERVED.exists():
        extra = json.loads(AUTORAG_OBSERVED.read_text())
        validate_autorag_observed(extra)
        for point in extra['points']:
            entry = next(d for d in result['datasets'] if d['key'] == point['dataset'])
            if point['id'] in {p['id'] for p in entry['points']}:
                raise ValueError('Agentic AutoRAG observation identity collides')
            entry['points'].append(point)
    if UNIFIED_TEST_DATA.exists():
        extra = json.loads(UNIFIED_TEST_DATA.read_text())
        if extra.get('schema') != 'unified-search-test-points-v1':
            raise ValueError('Unsupported unified search TEST points')
        for key, points in extra['datasets'].items():
            if key not in ('sciworld', 'webshop'):
                raise ValueError('Unexpected unified search dataset')
            entry = next(d for d in result['datasets'] if d['key'] == key)
            existing = {p['id'] for p in entry['points']}
            for point in points:
                if (point['id'] in existing or point.get('kind') != 'e2' or point.get('split') != 'test'
                        or point.get('n') != 200 or not all(isinstance(point.get(k), (int, float))
                        and math.isfinite(point[k]) for k in ('q', 'tok', 'lat'))):
                    raise ValueError('Invalid or duplicate unified search TEST point')
                existing.add(point['id']); entry['points'].append(point)
    result["fingerprint"] = hashlib.sha256(json.dumps(result["datasets"], sort_keys=True).encode()).hexdigest()
    return result


def e2_block(key: str, extra: dict) -> str:
    """One dataset's comparison caption + SVG (ALFWorld's block is unchanged)."""
    note = html_escape(extra["selection_note"]) if extra.get("selection_note") else (
        'All validation-selected TEST configurations are shown. '
        'Lines connect each method\'s observed TEST Pareto subset for display only; '
        'TEST outcomes do not select deployment policies. TRACE reuses a larger historical search catalog.'
    )
    return (
        f'<h3>{html_escape(extra.get("title", E2_TITLES[key]))}</h3>'
        f'<p class="small">{note}</p>'
        + extra["plot_svg"]
    )


def selection_document(data: dict) -> dict:
    return {"schema": "trimem-paper-curve-selection-v1", "saved_at": datetime.now(UTC).isoformat(),
            "data_fingerprint": data["fingerprint"], "selections": data["defaults"],
            "curve_kind": "author-selected display curve; may include dominated observations",
            "datasets": data["datasets"]}


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".pfe.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def refresh() -> dict:
    with PAGE.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        html = PAGE.read_text()
        data = collect(html)
        template = Path(__file__).with_suffix(".html").read_text().rstrip()
        comparison = "".join(e2_block(key, extra) for key, extra in e2_documents().items())
        template = template.replace("__PFE_E2_COMPARISON__", comparison)
        section = template.replace("__PFE_DATA__", json.dumps(data).replace("</", "<\\/"))
        if START in html:
            html = html[:html.index(START)] + section + html[html.index(END) + len(END):]
        else:
            anchor = "<!-- four-dataset-results:start -->"
            assert anchor in html
            html = html.replace(anchor, section + "\n" + anchor, 1)
        # Readers of the sidecars share this lock, so a refresh publishes one generation.
        REVISION.mkdir(parents=True, exist_ok=True)
        atomic_write(REVISION / "editor_data.json", json.dumps(data, indent=2) + "\n")
        atomic_write(REVISION / "suggested_selection.json", json.dumps(selection_document(data), indent=2) + "\n")
        atomic_write(PAGE, html)
    return data


if __name__ == "__main__":
    d = refresh()
    print(json.dumps({x["key"]: len(x["points"]) for x in d["datasets"]}))
