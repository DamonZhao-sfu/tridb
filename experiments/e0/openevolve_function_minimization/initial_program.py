# Adapted from OpenEvolve v0.3.2 examples/function_minimization (Apache-2.0).
# EVOLVE-BLOCK-START
"""Small deterministic function-minimization workload for E0 trace generation."""

import numpy as np


def search_algorithm(iterations=500, bounds=(-5.0, 5.0)):
    best_x = np.random.uniform(*bounds)
    best_y = np.random.uniform(*bounds)
    best_value = evaluate_function(best_x, best_y)
    for _ in range(iterations):
        x = np.random.uniform(*bounds)
        y = np.random.uniform(*bounds)
        value = evaluate_function(x, y)
        if value < best_value:
            best_x, best_y, best_value = x, y, value
    return best_x, best_y, best_value


# EVOLVE-BLOCK-END


def evaluate_function(x, y):
    return np.sin(x) * np.cos(y) + np.sin(x * y) + (x**2 + y**2) / 20


def run_search():
    return search_algorithm()
