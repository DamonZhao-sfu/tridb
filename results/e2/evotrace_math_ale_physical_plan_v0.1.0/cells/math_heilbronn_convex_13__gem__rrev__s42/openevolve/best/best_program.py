# EVOLVE-BLOCK-START
import numpy as np


def _min_area(pts):
    """Compute minimum triangle area among all C(n,3) triples."""
    n = len(pts)
    best = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = 0.5 * abs((pts[j, 0] - pts[i, 0]) * (pts[k, 1] - pts[i, 1])
                              - (pts[k, 0] - pts[i, 0]) * (pts[j, 1] - pts[i, 1]))
                if a < best:
                    best = a
    return best


def _local_search(pts, n_iter=100, seed=42):
    """Coordinate-wise local search to maximize min triangle area."""
    n = len(pts)
    pts = pts.copy()
    rng = np.random.default_rng(seed)
    current = _min_area(pts)

    for outer in range(n_iter):
        step = 0.06 * (1.0 - outer / n_iter) + 0.002
        improved = False
        # Cycle through points in a shuffled order
        order = rng.permutation(n)
        for idx in order:
            # Try moving x and y in 4 directions each
            for dx in [-step, -step * 0.5, step * 0.5, step]:
                np_ = pts.copy()
                np_[idx, 0] = np.clip(pts[idx, 0] + dx, 0.0, 1.0)
                val = _min_area(np_)
                if val > current:
                    pts = np_
                    current = val
                    improved = True
            for dy in [-step, -step * 0.5, step * 0.5, step]:
                np_ = pts.copy()
                np_[idx, 1] = np.clip(pts[idx, 1] + dy, 0.0, 1.0)
                val = _min_area(np_)
                if val > current:
                    pts = np_
                    current = val
                    improved = True
        if not improved:
            break

    return pts, current


def _scale_to_unit_square(pts):
    """Scale and center points to fill the unit square."""
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    center = (mn + mx) / 2.0
    extent = mx - mn
    max_ext = np.max(extent)
    if max_ext > 1e-10:
        scale = 0.95 / max_ext
        pts = (pts - center) * scale + 0.5
    return pts


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region to maximize
    the area of the smallest triangle formed by these points.

    Uses multi-restart local search with structured initializations.

    Returns:
        points: np.ndarray of shape (13, 2) with the x,y coordinates of the points.
    """
    best_pts = None
    best_score = -1.0

    # Multiple structured initializations to escape local optima
    inits = []

    # Init 1: Two concentric hexagons + center
    ai = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    ao = np.linspace(np.pi / 6, 2 * np.pi + np.pi / 6, 6, endpoint=False)
    inits.append(np.vstack([
        np.column_stack([0.5 + 0.30 * np.cos(ai), 0.5 + 0.30 * np.sin(ai)]),
        np.column_stack([0.5 + 0.45 * np.cos(ao), 0.5 + 0.45 * np.sin(ao)]),
        [[0.5, 0.5]]
    ]))

    # Init 2: Two concentric hexagons with different radii
    inits.append(np.vstack([
        np.column_stack([0.5 + 0.25 * np.cos(ai), 0.5 + 0.25 * np.sin(ai)]),
        np.column_stack([0.5 + 0.48 * np.cos(ao), 0.5 + 0.48 * np.sin(ao)]),
        [[0.5, 0.5]]
    ]))

    # Init 3: Regular 13-gon
    angles13 = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    inits.append(np.column_stack([
        0.5 + 0.45 * np.cos(angles13),
        0.5 + 0.45 * np.sin(angles13)
    ]))

    # Init 4: Hexagonal grid 4+5+4
    pts4 = []
    for y, cnt in zip([0.15, 0.5, 0.85], [4, 5, 4]):
        xs = np.linspace(0.1, 0.9, cnt)
        for x in xs:
            pts4.append([x, y])
    inits.append(np.array(pts4))

    # Run local search from each initialization
    for i, init_pts in enumerate(inits):
        optimized, score = _local_search(init_pts, n_iter=80, seed=42 + i)
        if score > best_score:
            best_score = score
            best_pts = optimized

    # Scale the best result to fill unit square
    best_pts = _scale_to_unit_square(best_pts)

    return best_pts


# EVOLVE-BLOCK-END
