# EVOLVE-BLOCK-START
"""Construct a grid packing for the requested number of circles."""

import math
import os

import numpy as np


def run_packing():
    n = int(os.environ.get("CSR_CIRCLE_N", "18"))
    columns = math.ceil(math.sqrt(n))
    rows = math.ceil(n / columns)
    radius = 0.45 * min(1.0 / columns, 1.0 / rows)
    centers = []
    for row in range(rows):
        for column in range(columns):
            if len(centers) == n:
                break
            centers.append(((column + 0.5) / columns, (row + 0.5) / rows))
    radii = np.full(n, radius, dtype=float)
    return np.asarray(centers, dtype=float), radii, float(np.sum(radii))


# EVOLVE-BLOCK-END

