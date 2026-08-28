# Adapted from OpenEvolve's Apache-2.0 circle-packing example.
# EVOLVE-BLOCK-START
"""Construct a valid packing of 18 circles in a unit square."""

import numpy as np


def construct_packing():
    n = 18
    columns = 5
    rows = 4
    centers = []
    for row in range(rows):
        for column in range(columns):
            if len(centers) == n:
                break
            centers.append(((column + 0.5) / columns, (row + 0.5) / rows))
    centers = np.asarray(centers, dtype=float)
    radii = np.full(n, 0.09, dtype=float)
    return centers, radii, float(np.sum(radii))


def run_packing():
    return construct_packing()


# EVOLVE-BLOCK-END
