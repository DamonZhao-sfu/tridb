"""Quality guardrails for the locked CrossEp-Know systems experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from bench.agent_memory.evomembench.metrics import clustered_paired_effect


ARMS = ("memory_off", "long_context", "full_gem")


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _key(row: dict[str, Any]) -> tuple[str, str]:
    metadata = row["metadata"]
    return str(metadata["context_id"]), str(metadata["task_id"])


def summarize(root: Path, *, bootstrap_repetitions: int = 10_000) -> dict[str, Any]:
    receipt = json.loads((root / "run_receipt.json").read_text())
    if receipt.get("status") != "complete":
        raise ValueError("cannot summarize an incomplete CrossEp-Know run")
    if set(receipt.get("arms", [])) != set(ARMS):
        raise ValueError("quality summary requires memory_off, long_context, full_gem")
    graded = {arm: _rows(root / "graded" / f"{arm}.jsonl") for arm in ARMS}
    by_arm = {arm: {_key(row): row for row in rows} for arm, rows in graded.items()}
    key_sets = {frozenset(rows) for rows in by_arm.values()}
    if len(key_sets) != 1:
        raise ValueError("graded CrossEp-Know arms have different target sets")
    keys = sorted(next(iter(key_sets)))
    if receipt.get("formal") and len(keys) != 884:
        raise ValueError(f"formal CrossEp-Know expected 884 episodes, got {len(keys)}")
    cross_keys = [
        key for key in keys if int(by_arm["memory_off"][key]["metadata"]["ordinal"]) > 0
    ]
    if receipt.get("formal") and len(cross_keys) != 764:
        raise ValueError(
            f"formal CrossEp-Know expected 764 reuse decisions, got {len(cross_keys)}"
        )

    def score(arm: str, key: tuple[str, str]) -> int:
        return int(by_arm[arm][key]["score"])

    arm_metrics: dict[str, Any] = {}
    for arm in ARMS:
        context_scores: dict[str, list[int]] = {}
        ordinal_scores: dict[int, list[int]] = {}
        category_scores: dict[str, list[int]] = {}
        for key in cross_keys:
            row = by_arm[arm][key]
            metadata = row["metadata"]
            value = int(row["score"])
            context_scores.setdefault(str(metadata["context_id"]), []).append(value)
            ordinal_scores.setdefault(int(metadata["ordinal"]), []).append(value)
            category_scores.setdefault(
                str(metadata.get("context_category", "unknown")), []
            ).append(value)
        arm_metrics[arm] = {
            "episodes": len(keys),
            "reuse_decisions": len(cross_keys),
            "strict_rubric_accuracy": mean(score(arm, key) for key in keys),
            "cross_episode_accuracy": mean(score(arm, key) for key in cross_keys),
            "macro_context_accuracy": mean(
                mean(values) for values in context_scores.values()
            ),
            "accuracy_by_ordinal": {
                str(ordinal): {"n": len(values), "accuracy": mean(values)}
                for ordinal, values in sorted(ordinal_scores.items())
            },
            "accuracy_by_category": {
                category: {"n": len(values), "accuracy": mean(values)}
                for category, values in sorted(category_scores.items())
            },
        }

    def effect(treatment: str) -> dict[str, Any]:
        result = clustered_paired_effect(
            (
                (
                    key[0],
                    score("memory_off", key),
                    score(treatment, key),
                )
                for key in cross_keys
            ),
            repetitions=bootstrap_repetitions,
        )
        result["beneficial_flips"] = sum(
            score("memory_off", key) == 0 and score(treatment, key) == 1
            for key in cross_keys
        )
        result["harmful_flips"] = sum(
            score("memory_off", key) == 1 and score(treatment, key) == 0
            for key in cross_keys
        )
        result["beneficial_flip_rate"] = result["beneficial_flips"] / len(cross_keys)
        result["harmful_flip_rate"] = result["harmful_flips"] / len(cross_keys)
        return result

    reuse = effect("full_gem")

    def grouped_reuse(field: str) -> dict[str, Any]:
        grouped: dict[str, list[tuple[int, int]]] = {}
        for key in cross_keys:
            metadata = by_arm["memory_off"][key]["metadata"]
            group = str(metadata.get(field, "unknown"))
            grouped.setdefault(group, []).append(
                (score("memory_off", key), score("full_gem", key))
            )
        return {
            group: {
                "n": len(values),
                "no_memory_accuracy": mean(value[0] for value in values),
                "full_gem_accuracy": mean(value[1] for value in values),
                "reuse_gain": mean(value[1] - value[0] for value in values),
                "beneficial_flip_rate": sum(
                    baseline == 0 and treatment == 1 for baseline, treatment in values
                )
                / len(values),
                "harmful_flip_rate": sum(
                    baseline == 1 and treatment == 0 for baseline, treatment in values
                )
                / len(values),
            }
            for group, values in sorted(grouped.items())
        }

    full_vs_long = clustered_paired_effect(
        (
            (key[0], score("long_context", key), score("full_gem", key))
            for key in cross_keys
        ),
        repetitions=bootstrap_repetitions,
    )
    lower = float(reuse["clustered_bootstrap_95_ci"][0])
    return {
        "schema_version": "evomembench_know_quality_v0.1.0",
        "run_id": receipt["run_id"],
        "formal": bool(receipt.get("formal")),
        "arms": arm_metrics,
        "full_gem_vs_no_memory": reuse,
        "reuse_gain_by_session_position": grouped_reuse("ordinal"),
        "reuse_gain_by_context_category": grouped_reuse("context_category"),
        "full_gem_vs_long_context": full_vs_long,
        "claim_gates": {
            "memory_utility_pass": (
                float(reuse["paired_mean_delta"]) >= 0.03 and lower > 0
            ),
            "memory_utility_threshold": (
                "paired gain >= 0.03 and context-cluster bootstrap CI lower > 0"
            ),
            "long_context_noninferiority_pass": (
                float(full_vs_long["paired_mean_delta"]) >= -0.02
            ),
            "noninferiority_margin": -0.02,
        },
        "unavailable_retrieval_metrics": {
            name: "N/A: EvoMemBench has no independent source-experience qrels"
            for name in ("recall_at_k", "ndcg_at_k", "mrr", "graph_path_recall")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    args = parser.parse_args()
    output = args.run_dir / "quality_summary.json"
    if output.exists():
        raise FileExistsError(f"refusing existing quality summary: {output}")
    payload = summarize(args.run_dir, bootstrap_repetitions=args.bootstrap_repetitions)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
