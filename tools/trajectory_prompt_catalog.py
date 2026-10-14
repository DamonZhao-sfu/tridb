"""Export all task-native systems/templates and audit ALE reference programs."""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import psycopg
import yaml
from psycopg.rows import dict_row

from bench.agent_memory.gem_oe.trajectory_prompt import TEMPLATE_DIR


def context_section(user):
    # Locate Markdown headings outside fenced code, so source comments are safe.
    lines = user.splitlines(keepends=True)
    inside = False
    start = None
    end = len(lines)
    for i, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            inside = not inside
        if inside:
            continue
        if line.rstrip() == "## Other Context Solutions":
            start = i
        elif start is not None and line.startswith("# "):
            end = i
            break
    if start is None:
        return None, []
    section = "".join(lines[start:end])
    codes = re.findall(
        r"^```[^\n]*\n(.*?)^```[ \t]*$", section, re.MULTILINE | re.DOTALL
    )
    return section, codes


def program_count(section):
    inside = False
    count = 0
    for line in (section or "").splitlines():
        if line.lstrip().startswith("```"):
            inside = not inside
        elif not inside and re.match(r"^### Program \d+\b", line):
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dsn", default="host=/tmp/trimem-traj port=55450 dbname=trimem_traj"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "triMEM_optimizer/trajectory_storage_v020_20261013/task_prompt_templates"
        ),
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    conn = psycopg.connect(args.dsn, row_factory=dict_row)
    tasks = conn.execute(
        "SELECT task, specification FROM task ORDER BY task"
    ).fetchall()
    raw = Path("dataset/evotrace/raw")
    # All source hashes, grouped by session, prove whether displayed code is a
    # stored source program rather than a generated prose summary.
    hashes = defaultdict(set)
    for r in conn.execute(
        "SELECT session, md5(btrim(code, E' \\t\\r\\n')) AS h FROM attempt WHERE code IS NOT NULL"
    ):
        hashes[r["session"]].add(r["h"])
    manifest, systems = [], defaultdict(dict)
    for task in tasks:
        uid, system = task["task"], task["specification"]
        folder = args.out / uid.replace(":", "_")
        folder.mkdir(exist_ok=True)
        sources = conn.execute(
            """SELECT a.id,a.backend,s.run_rel,p.template,p.prompts_sha256,
            p.user_text,ps.text AS system_text FROM attempt a JOIN session s ON s.session=a.session
            JOIN prompt p ON p.attempt_id=a.id JOIN prompt_system ps ON ps.system_sha=p.system_sha
            WHERE a.task=%s AND ps.text=%s
            ORDER BY (a.backend='openevolve_native') DESC, a.iteration,a.id LIMIT 1""",
            (uid, system),
        ).fetchall()
        if not sources:
            raise ValueError(f"No exact original system match for {uid}")
        source = sources[0]
        sha = source["prompts_sha256"]
        config_path = Path("data/evotrace/raw") / source["run_rel"] / "run_config.yaml"
        config_matches = None
        if config_path.exists():
            config_system = (
                (yaml.safe_load(config_path.read_text()) or {})
                .get("prompt", {})
                .get("system_message")
            )
            config_matches = config_system == system if config_system else None
        blob = (
            raw / source["run_rel"] / "blobs" / sha[:2] / (sha + ".json")
            if sha
            else None
        )
        verified = False
        if blob:
            payload = json.loads(blob.read_text())[source["template"]]
            assert payload["system"] == system
            assert payload["user"] == source["user_text"]
            verified = True
        language = "cpp" if uid.startswith("ale:") else "python"
        # A parameterized template: placeholders are deliberate, not example data.
        user = f"""# Current Program Information
- Fitness: {{{{current_fitness}}}}
- Evaluation feedback: {{{{current_evaluation_feedback}}}}
- Current improvement objective: {{{{current_objective}}}}
- Remaining budget: {{{{remaining_budget}}}}

## Current Evaluation Feedback
{{{{current_parent_recorded_metrics_errors_and_artifacts}}}}

# Program Evolution History
## Previous Attempts
### Attempt {{{{recent_attempt_1_iteration}}}}
- Changes: {{{{recent_attempt_1_changes_or_Unknown_changes}}}}
- Metrics: {{{{recent_attempt_1_metrics}}}}
- Outcome: {{{{recent_attempt_1_combined_score_vs_parent}}}}

#### Evaluation Feedback
{{{{recent_attempt_1_recorded_errors_and_artifacts_or_not_recorded}}}}

### Attempt {{{{recent_attempt_2_iteration}}}}
- Changes: {{{{recent_attempt_2_changes_or_Unknown_changes}}}}
- Metrics: {{{{recent_attempt_2_metrics}}}}
- Outcome: {{{{recent_attempt_2_combined_score_vs_parent}}}}

#### Evaluation Feedback
{{{{recent_attempt_2_recorded_errors_and_artifacts_or_not_recorded}}}}

### Attempt {{{{recent_attempt_3_iteration}}}}
- Changes: {{{{recent_attempt_3_changes_or_Unknown_changes}}}}
- Metrics: {{{{recent_attempt_3_metrics}}}}
- Outcome: {{{{recent_attempt_3_combined_score_vs_parent}}}}

#### Evaluation Feedback
{{{{recent_attempt_3_recorded_errors_and_artifacts_or_not_recorded}}}}

## Top Performing Programs
### Program 1 (combined_score: {{{{top_1_fitness}}}})
```{language}
{{{{top_1_complete_code}}}}
```

## Inspiration Programs
{{{{four_retrieved_program_blocks_code_plus_intent}}}}

# Current Program
```{language}
{{{{current_parent_complete_code}}}}
```

# Task
Suggest improvements to the current program under the task requirements.
Return exact SEARCH/REPLACE edits:
<<<<<<< SEARCH
{{{{original_code_to_replace}}}}
=======
{{{{replacement_code}}}}
>>>>>>> REPLACE
"""
        inspiration_blocks = "\n\n".join(
            f"### Inspiration {i}\nIntent: {{{{inspiration_{i}_intent_or_Not_recorded}}}}\n\n"
            f"```{language}\n{{{{inspiration_{i}_complete_code}}}}\n```"
            for i in range(1, 5)
        )
        user = user.replace(
            "{{four_retrieved_program_blocks_code_plus_intent}}", inspiration_blocks
        )
        (folder / "system.txt").write_text(system)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        (folder / "template.messages.json").write_text(
            json.dumps(messages, ensure_ascii=False, indent=2)
        )
        (folder / "template.txt").write_text(
            "[SYSTEM MESSAGE]\n" + system + "\n\n[USER TEMPLATE]\n" + user
        )
        (folder / "original_example.txt").write_text(
            "[SYSTEM MESSAGE]\n" + system + "\n\n[USER MESSAGE]\n" + source["user_text"]
        )
        entry = {
            "task": uid,
            "folder": folder.name,
            "system_sha256": hashlib.sha256(system.encode()).hexdigest(),
            "source_attempt": source["id"],
            "source_run": source["run_rel"],
            "raw_blob": str(blob),
            "raw_blob_verified": verified,
            "config_path": str(config_path) if config_path.exists() else None,
            "config_system_matches": config_matches,
        }
        manifest.append(entry)

    counts, by_task, code_checks, samples = (
        defaultdict(Counter),
        defaultdict(Counter),
        defaultdict(Counter),
        {},
    )
    prompt_rows, math_rows = [], []
    math_counts, math_by_task = defaultdict(Counter), defaultdict(Counter)
    with conn.cursor(name="catalog_prompts") as cur:
        cur.itersize = 16
        cur.execute("""SELECT a.id,a.task,a.session,a.backend,a.iteration,s.run_rel,
            p.user_text,p.template,p.prompts_sha256,ps.text AS system_text
            FROM prompt p JOIN attempt a ON a.id=p.attempt_id JOIN session s ON s.session=a.session
            LEFT JOIN prompt_system ps ON ps.system_sha=p.system_sha ORDER BY a.id""")
        for row in cur:
            system = row["system_text"]
            if system:
                sh = hashlib.sha256(system.encode()).hexdigest()
                systems[row["task"]].setdefault(
                    sh, {"text": system, "count": 0, "source_attempt": row["id"]}
                )["count"] += 1
            section, codes = context_section(row["user_text"] or "")
            n = program_count(section)
            if row["task"].startswith("math:"):
                math_counts[row["backend"]][n] += 1
                math_by_task[row["task"]][n] += 1
                math_rows.append(
                    {
                        "attempt_id": row["id"],
                        "task": row["task"],
                        "backend": row["backend"],
                        "iteration": row["iteration"],
                        "other_context_programs": n,
                    }
                )
                continue
            counts[row["backend"]][n] += 1
            by_task[row["task"]][n] += 1
            matches = sum(
                hashlib.md5(c.strip().encode(), usedforsecurity=False).hexdigest()
                in hashes[row["session"]]
                for c in codes
            )
            code_checks[row["backend"]].update(
                {
                    "displayed_code_blocks": len(codes),
                    "exact_session_source_matches": matches,
                    "sections_present": int(section is not None),
                    "explicit_truncation_markers": sum(
                        "... (truncated)" in c for c in codes
                    ),
                    "search_replace_blocks": sum("<<<<<<< SEARCH" in c for c in codes),
                }
            )
            prompt_rows.append(
                {
                    "attempt_id": row["id"],
                    "task": row["task"],
                    "backend": row["backend"],
                    "iteration": row["iteration"],
                    "other_context_programs": n,
                    "source_matches": matches,
                }
            )
            if n >= 3 and row["backend"] not in samples:
                samples[row["backend"]] = row | {"section": section}
    for task, variants in systems.items():
        folder = args.out / task.replace(":", "_") / "system_variants"
        folder.mkdir(exist_ok=True)
        index = []
        for sh, info in variants.items():
            (folder / (sh + ".txt")).write_text(info["text"])
            index.append(
                {k: v for k, v in info.items() if k != "text"} | {"sha256": sh}
            )
        (folder / "index.json").write_text(json.dumps(index, indent=2))
    sample_folder = args.out / "ale_original_examples"
    sample_folder.mkdir(exist_ok=True)
    for backend, row in samples.items():
        (sample_folder / f"{backend}_{row['id']}.txt").write_text(
            "[SYSTEM MESSAGE]\n"
            + (row["system_text"] or "")
            + "\n\n[USER MESSAGE]\n"
            + row["user_text"]
        )
        (sample_folder / f"{backend}_{row['id']}_context.txt").write_text(
            row["section"]
        )
    stats = {
        "unit": "stored prompt rows joined to attempts; missing context counted as zero; not all original LLM calls are necessarily retained",
        "by_backend": dict(counts),
        "by_task": dict(by_task),
        "source_code_checks": dict(code_checks),
        "examples": {
            b: {
                k: r[k]
                for k in (
                    "id",
                    "task",
                    "iteration",
                    "run_rel",
                    "prompts_sha256",
                    "template",
                )
            }
            for b, r in samples.items()
        },
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    (args.out / "ale_context_statistics.json").write_text(json.dumps(stats, indent=2))
    (args.out / "ale_prompt_counts.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in prompt_rows)
    )
    (args.out / "math_context_statistics.json").write_text(
        json.dumps(
            {
                "unit": stats["unit"],
                "by_backend": dict(math_counts),
                "by_task": dict(math_by_task),
            },
            indent=2,
        )
    )
    (args.out / "math_prompt_counts.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in math_rows)
    )
    index = [
        "# Math/ALE 任务 prompt 模板",
        "",
        "覆盖库内全部 18 个任务。SYSTEM MESSAGE 与对应原始 prompt 完全相同，来源与哈希见 manifest.json。USER TEMPLATE 是按当前要求设计的模板，不冒充原始历史 prompt。每个任务同时保留 original_example.txt。",
        "",
        "Previous Attempts：最近三次已完成尝试的 Changes / Metrics / Outcome。Top Performing Programs：本次运行有效程序中按 combined_score 选最高分 1 个，排除当前 parent。Inspiration：目标 4 个外部检索的完整程序 + 历史 intent。空历史不编造记录。",
        "",
        "当前运行历史覆盖自身所有 islands 中仍保留的已完成记录，不包含外部检索记录或未来迭代。Top 1 上限 8192 tokens，Inspiration 上限 16384 tokens，完整输入含 chat template 上限 32768 tokens。代码不截断，超限整条跳过并记录，因此实际数量可能少于目标值。模板中不存在的历史/参考槽位应省略。原始 SYSTEM MESSAGE 不额外包装通用指令。",
        "",
        "模板变量以 {{...}} 标记。全部观测到的原始 system 变体另存 system_variants/，主模板选用库中该任务 specification 精确匹配的原始 system。",
        "",
        "| Task | 模板 | 原始示例 |",
        "|---|---|---|",
    ]
    for m in manifest:
        index.append(
            f"| {m['task']} | [template]({m['folder']}/template.txt) | [original]({m['folder']}/original_example.txt) |"
        )
    index += [
        "",
        "## ALE Other Context Solutions 实测数量",
        "",
        "按已保存的 prompt/attempt 记录统计，缺少该 section 计 0。Shinka 使用不同 section，0 不代表没有任何参考程序。",
        "",
        "| Backend | 记录数 | 程序数分布（个数:记录数） | 源码完全匹配 |",
        "|---|---:|---|---:|",
    ]
    for b, c in counts.items():
        check = code_checks[b]
        index.append(
            f"| {b} | {sum(c.values())} | {dict(sorted(c.items()))} | {check['exact_session_source_matches']}/{check['displayed_code_blocks']} |"
        )
    index += [
        "",
        "ALE 主要放程序源码，但原始记录存在少量带 `... (truncated)` 的代码、diff 或 response 内容，不能声称所有历史参考均为完整源码。详细计数见 ale_context_statistics.json。",
        "",
        "## Math Other Context Solutions 实测数量",
        "",
        "| Backend | 记录数 | 程序数分布（个数:记录数） |",
        "|---|---:|---|",
    ]
    for b, c in math_counts.items():
        index.append(f"| {b} | {sum(c.values())} | {dict(sorted(c.items()))} |")
    (args.out / "README.md").write_text("\n".join(index) + "\n")
    (args.out / "inspiration_program.template.txt").write_text(
        (TEMPLATE_DIR / "inspiration_program.txt").read_text()
    )
    print(
        json.dumps(
            {
                "tasks": len(manifest),
                "raw_verified": sum(m["raw_blob_verified"] for m in manifest),
                "ALE": stats,
                "Math": dict(math_counts),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
