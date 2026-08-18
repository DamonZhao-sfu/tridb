"""Deterministic evaluator adapted from OpenEvolve v0.3.2's official example."""

from __future__ import annotations

import importlib.util
import math

import numpy as np


def _load_program(program_path: str):
    spec = importlib.util.spec_from_file_location("e0_candidate", program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(program_path: str) -> dict[str, float]:
    target_x, target_y, target_value = -1.704, 0.678, -1.519
    values: list[float] = []
    distances: list[float] = []
    successes = 0
    try:
        program = _load_program(program_path)
        if not hasattr(program, "run_search"):
            return {
                "value_score": 0.0,
                "distance_score": 0.0,
                "reliability_score": 0.0,
                "combined_score": 0.0,
            }
        for seed in range(8):
            np.random.seed(seed)
            result = program.run_search()
            if not isinstance(result, tuple) or len(result) not in {2, 3}:
                continue
            x, y = float(result[0]), float(result[1])
            value = (
                float(result[2])
                if len(result) == 3
                else float(program.evaluate_function(x, y))
            )
            if not all(math.isfinite(item) for item in (x, y, value)):
                continue
            if not (-5.0 <= x <= 5.0 and -5.0 <= y <= 5.0):
                continue
            values.append(value)
            distances.append(math.hypot(x - target_x, y - target_y))
            successes += 1
    except Exception:
        return {
            "value_score": 0.0,
            "distance_score": 0.0,
            "reliability_score": 0.0,
            "combined_score": 0.0,
        }

    if not successes:
        return {
            "value_score": 0.0,
            "distance_score": 0.0,
            "reliability_score": 0.0,
            "combined_score": 0.0,
        }
    value_score = 1.0 / (1.0 + abs(float(np.mean(values)) - target_value))
    distance_score = 1.0 / (1.0 + float(np.mean(distances)))
    reliability_score = successes / 8.0
    combined_score = (
        0.55 * value_score + 0.3 * distance_score + 0.15 * reliability_score
    )
    return {
        "value_score": float(value_score),
        "distance_score": float(distance_score),
        "reliability_score": float(reliability_score),
        "combined_score": float(combined_score),
    }
