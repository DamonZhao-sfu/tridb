"""Shared presentation filter for the local atlas, applied before publication.

The adjacent presentation JSON is reversible display configuration. Source
datasets and experiment results are not deleted or recomputed by this module.
"""

import fcntl
import json
import re
import subprocess
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "triMEM_optimizer/it20_summary/qa-trajectory.html"
CONFIG = PAGE.with_name("qa-trajectory.presentation.json")
TRANSLATIONS = Path(__file__).with_name("atlas_english.json")
SCRIPT = re.compile(r"(<script\b[^>]*>)(.*?)(</script\s*>)", re.S | re.I)


def script_attributes(tag):
    attributes = {}
    parser = HTMLParser()
    parser.handle_starttag = lambda _tag, attrs: attributes.update(attrs)
    parser.feed(tag)
    return attributes


@lru_cache(maxsize=32)
def validate_javascript(script):
    """Check generated JavaScript without executing it."""
    subprocess.run(
        ["node", "--check", "-"],
        input=script,
        text=True,
        capture_output=True,
        check=True,
        timeout=15,
    )


class VisibleText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hidden = False
        self.text = []
        self.iframes = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden = True
        if tag == "iframe":
            self.iframes += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden = False

    def handle_data(self, data):
        if not self.hidden:
            self.text.append(data)


def english_renderer():
    translations = json.loads(TRANSLATIONS.read_text())
    pattern = re.compile(
        "|".join(re.escape(k) for k in sorted(translations, key=len, reverse=True))
    )
    punctuation = str.maketrans(
        {
            "（": " (",
            "）": ")",
            "，": ", ",
            "。": ". ",
            "；": "; ",
            "：": ": ",
            "、": ", ",
            "“": "‘",
            "”": "’",
        }
    )

    def translate(value):
        return pattern.sub(lambda m: translations[m[0]], value).translate(punctuation)

    return translate


def remove_block(text, name):
    return re.sub(
        r"<!--\s*"
        + re.escape(name)
        + r":start\s*-->.*?<!--\s*"
        + re.escape(name)
        + r":end\s*-->",
        "",
        text,
        flags=re.S | re.I,
    )


def with_dataset_tasks(text, keep=None):
    """Place the English task overview immediately below the page header."""
    text = re.sub(
        r"\n?<!-- dataset-tasks:start -->.*?<!-- dataset-tasks:end -->",
        "",
        text,
        flags=re.S,
    )
    if "</header>" not in text:
        raise ValueError("Page header missing: cannot place the dataset overview")
    section = (
        Path(__file__).with_name("dataset_tasks_section_en.html").read_text().strip()
    )
    if keep is not None:
        section = re.sub(
            r'<tr data-dataset="([^"]+)">.*?</tr>',
            lambda m: m[0] if m[1] in keep else "",
            section, flags=re.S,
        )
    return text.replace("</header>", "</header>\n" + section, 1)


def remove_cost_saving_hv_panel(text):
    """Retain the TEST cost-saving panel while removing its historical TRAIN HV panel."""
    pattern = r"<!-- cost-saving-hv:start -->.*?<!-- cost-saving-hv:end -->"

    def clean(match):
        block = match[0]
        if 'id="cs-b"' not in block:
            if "// ---------------- B\n" in block or "<h3>B. BO" in block:
                raise ValueError("Legacy HV panel still present with an unexpected layout")
            return block
        block, n_html = re.subn(
            r'<h3>B\. BO.*?<div id="cs-b"></div>', "", block, flags=re.DOTALL
        )
        block, n_js = re.subn(
            r"// ---------------- B\n.*?(?=\$\('cs-hv'\)\.addEventListener)",
            "",
            block,
            flags=re.DOTALL,
        )
        if (n_html, n_js) != (1, 1):
            raise ValueError("Legacy cost-saving HV layout changed; refusing partial removal")
        block = block.replace("if(c.dataset.k.startsWith('r:'))bShow();else aShow()", "aShow()")
        block = re.sub(
            r"<h2>.*?</h2>",
            "<h2>Quality gain at matched cost and cost savings at matched quality (test)</h2>",
            block,
            count=1,
            flags=re.DOTALL,
        )
        return block

    return re.sub(pattern, clean, text, flags=re.DOTALL)


def prepare_html(text):
    if not CONFIG.exists():
        return text
    config = json.loads(CONFIG.read_text())
    keep = config["datasets"]
    translate = (
        english_renderer() if config["language"] == "en" else lambda value: value
    )
    serialized = []
    seen = set()
    tau_snapshot = None
    display_fields = {
        "label",
        "name",
        "metric",
        "notes",
        "pending",
        "panel",
        "why",
        "store",
        "graph_note",
        "example",
        "purpose",
        "scope",
    }

    def display_text(value):
        if isinstance(value, str):
            result = translate(value) if re.search(r"[\u4e00-\u9fff]", value) else value
            if config["language"] == "en" and re.search(r"[\u4e00-\u9fff]", result):
                raise ValueError(f"Untranslated display metadata: {value[:100]}")
            return result
        if isinstance(value, list):
            return [display_text(item) for item in value]
        return value

    def localize(data):
        # Translate explicit display fields only. Recorded content is immutable.
        if isinstance(data, list):
            return [localize(value) for value in data]
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                if key == "sample":
                    result[key] = [
                        {
                            k: re.sub(
                                r"^\[(\d+) 维向量\]$", r"[\1 dimensional vector]", v
                            )
                            if isinstance(v, str)
                            else v
                            for k, v in row.items()
                        }
                        for row in value
                    ]
                elif key in display_fields:
                    result[key] = display_text(value)
                elif key in {"tasks", "sessions", "steps", "trace", "native"}:
                    result[key] = value
                else:
                    result[key] = localize(value)
            return result
        return data

    if not config["archive"]:
        text = remove_block(text, "MULTISESSION-ARCHIVE").replace(
            "__MULTISESSION_EMBED__", ""
        )
    text = remove_block(text, "swe-expansion-research")
    for name in config.get("hidden_sections", []):
        text = remove_block(text, name)
    if config.get("hide_cost_saving_hv_panel", False):
        text = remove_cost_saving_hv_panel(text)
    text = with_dataset_tasks(text, keep)

    def payload(match):
        nonlocal tau_snapshot
        before, body, after = match.groups()
        attributes = script_attributes(before)
        if attributes.get("type", "").lower() != "application/json":
            return match[0]
        identifier = attributes.get("id")
        marker = f"__ATLAS_JSON_PAYLOAD_{len(serialized)}__"
        if marker in text:
            raise ValueError("Reserved presentation placeholder already present")
        known = {
            "atlas-data",
            "fd-data",
            "rt-data",
            "cs-data",
            "pp-data",
            "pp-pool-data",
            "or-data",
        }
        if identifier not in known:
            serialized.append(body)
            return before + marker + after
        if identifier in seen:
            raise ValueError(f"Duplicate atlas payload: {identifier}")
        seen.add(identifier)
        data = json.loads(body)
        if identifier in ("atlas-data", "fd-data", "rt-data"):
            data["datasets"] = [d for d in data["datasets"] if d["key"] in keep]
            if identifier in ("atlas-data", "rt-data") and "tau2" in keep:
                if tau_snapshot is None:
                    path = PAGE.parent / "experience_atlas/tau2_atlas_snapshot.json"
                    tau_snapshot = json.loads(path.read_text())
                part = "graph" if identifier == "atlas-data" else "tables"
                data["datasets"] = [
                    d for d in data["datasets"] if d["key"] != "tau2"
                ] + [tau_snapshot[part]]
            data["datasets"].sort(key=lambda d: keep.index(d["key"]))
            if (
                identifier == "atlas-data"
                and [d["key"] for d in data["datasets"]] != keep
            ):
                raise ValueError(f"Required dataset missing from {identifier}")
        elif identifier == "cs-data":
            data["A"] = [d for d in data["A"] if d["ds"] in keep]
            data["B"] = [d for d in data["B"] if d["ds"] in keep]
            data["B_skipped"] = [
                d for d in data["B_skipped"] if d["experiment"].split(":")[0] in keep
            ]
        elif identifier == "pp-data":
            for part in ("train", "test"):
                data[part] = [d for d in data[part] if d["ds"] in keep]
        elif identifier == "pp-pool-data":
            data = [d for d in data if d["ds"] in keep]
        elif identifier == "or-data":
            data["pools"] = [
                d for d in data["pools"] if d["experiment"].split(":")[0] in keep
            ]
            data["skipped"] = [
                d for d in data["skipped"] if d["name"].split("_")[0] in keep
            ]
        serialized.append(
            json.dumps(
                data if identifier == "atlas-data" else localize(data),
                ensure_ascii=False,
            ).replace("<", "\\u003c")
        )
        return before + marker + after

    text = SCRIPT.sub(payload, text)
    if "atlas-data" not in seen:
        raise ValueError(
            "Atlas payload missing: refusing to publish an unfiltered page"
        )
    # Dataset-split rows and their source notes are static HTML rather than JSON.
    start, end = "<!-- dataset-splits:start -->", "<!-- dataset-splits:end -->"
    if re.search(r"""id\s*=\s*["']ds-splits["']""", text) and (
        start not in text or end not in text
    ):
        raise ValueError(
            "Dataset split markers missing: refusing to publish unfiltered rows"
        )
    if start in text:
        a, b = text.index(start), text.index(end) + len(end)
        section = text[a:b]
        labels = {
            "alfworld": "ALFWorld",
            "sciworld": "ScienceWorld",
            "tau2": "τ²-bench",
            "appworld": "AppWorld",
            "webshop": "WebShop",
        }
        row_pattern = r'<tr\b[^>]*>\s*<th\b[^>]*\bscope\s*=\s*["\']?row["\']?[^>]*>(.*?)</th>.*?</tr>'
        rows = re.findall(row_pattern, section, flags=re.S | re.I)
        if not rows or any(labels[k] not in rows for k in keep):
            raise ValueError(
                "Dataset split rows changed: update the presentation filter"
            )
        section = re.sub(
            row_pattern,
            lambda m: m[0] if m[1] in {labels[k] for k in keep} else "",
            section,
            flags=re.S | re.I,
        )
        used = set(re.findall(r'href="#ds-src-(\d+)"', section))
        section = re.sub(
            r'<li id="ds-src-(\d+)">.*?</li>',
            lambda m: m[0] if m[1] in used else "",
            section,
            flags=re.S,
        )
        section = re.sub(r"出处（\d+ 条）", f"Sources ({len(used)})", section)
        text = text[:a] + section + text[b:]
    dataset_heading = "Five datasets" if len(keep) == 5 else "Three datasets"
    text = text.replace("五个数据集", dataset_heading).replace(
        "关系型表 · 6 个数据集", "Relational tables"
    )
    text = text.replace("Three datasets · Quality", dataset_heading + " · Quality")
    text = text.replace("across the three datasets", "across the displayed datasets")
    if {"appworld", "webshop"}.issubset(keep):
        text = text.replace(
            "Held-out test panels: ALFWorld 200 tasks, ScienceWorld 200 tasks, and τ² 100 official test tasks × 1 trial (pool -v5 test, trial 0).",
            "Held-out test panels: ALFWorld 200 tasks, ScienceWorld 200 tasks, and τ² 100 official test tasks × 1 trial (pool -v5 test, trial 0). Archived panels: AppWorld test_normal 168 tasks and WebShop AgentEval 200 tasks; historical protocols are retained.",
        ) if "Archived panels: AppWorld" not in text else text
    text = re.sub(r"关系型表 · \d+ 个数据集", "Relational tables", text)
    if "appworld" not in keep:
        text = text.replace("、AppWorld test_normal 168", "").replace("AppWorld / τ²", "τ²")
    if "webshop" not in keep:
        text = text.replace("、WebShop test 200", "")
        text = text.replace("{alfworld,sciworld,webshop}", "{alfworld,sciworld}")
    # Remove dataset-specific conclusions outside the currently selected scope.
    for key, label in (("appworld", "AppWorld"), ("webshop", "WebShop")):
        if key not in keep:
            text = re.sub(r"<li>" + label + r"[：:].*?</li>", "", text, flags=re.S)
    text = re.sub(
        r"<footer>.*?</footer>",
        '<footer><span id="built"></span><p>Offline graph and experiment snapshot. '
        "Graph examples illustrate storage and retrieval, independently of benchmark results.</p></footer>",
        text,
        flags=re.S,
    )
    dataset_names = {"alfworld": "ALFWorld", "sciworld": "ScienceWorld", "tau2": "τ²-bench",
                     "appworld": "AppWorld", "webshop": "WebShop"}
    title_names = ", ".join(dataset_names[k] for k in keep)
    text = re.sub(
        r'<h1>Experience graph atlas</h1><p class="muted">.*?</p>',
        '<h1>Experience graph atlas</h1><p class="muted">' + title_names + '. '
        "Explore tasks, sessions and steps, then follow a recorded memory retrieval.</p>",
        text,
        flags=re.S,
    )
    if config["language"] == "en":
        text = text.replace('lang="zh-CN"', 'lang="en"')
        text = translate(text)
        # JSON is masked here. Stop before publication on untranslated new UI.
        if re.search(r"[\u4e00-\u9fff]", text):
            raise ValueError("Untranslated atlas UI: update atlas_english.json")
    public = VisibleText()
    public.feed(text)
    excluded_names = ["SWE-QA", "SWE-rebench", "SWE-Gym", "CodeActInstruct", "ToRA-MATH"]
    excluded_names.extend(label for key, label in (("appworld", "AppWorld"), ("webshop", "WebShop")) if key not in keep)
    if re.search(
        r"\b(?:" + "|".join(re.escape(name) for name in excluded_names) + r")\b",
        " ".join(public.text),
        re.I,
    ):
        raise ValueError("Excluded dataset remains in static atlas content")
    if not config["archive"] and public.iframes:
        raise ValueError("Archive iframe remains in the atlas")
    for script in SCRIPT.finditer(text):
        if (
            script_attributes(script[1]).get("type", "").lower() != "application/json"
            and script[2].strip()
        ):
            validate_javascript(script[2])
    for i, body in enumerate(serialized):
        text = text.replace(f"__ATLAS_JSON_PAYLOAD_{i}__", body)
    return text


def main():
    with PAGE.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before = PAGE.read_text()
        backup = PAGE.with_name("qa-trajectory.before-english-focus.html")
        if not backup.exists():
            backup.write_text(before)
        after = prepare_html(before)
        temp = PAGE.with_suffix(".presentation.tmp.html")
        temp.write_text(after)
        temp.replace(PAGE)
    print(f"Updated {PAGE}: {len(before):,} -> {len(after):,} characters")


if __name__ == "__main__":
    main()
