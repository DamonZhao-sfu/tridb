# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 11.

    Returns:
        points: np.ndarray of shape (11,2) with the x,y coordinates of the points.
    """
    # Equilateral triangle: vertices at (0,0), (1,0), (0.5, sqrt(3)/2)
    h = np.sqrt(3) / 2

    def in_triangle(p):
        x, y = p
        return y >= -1e-9 and y <= h + 1e-9 and x >= y / h - 1e-9 and x <= 1 - y / h + 1e-9

    def min_area(pts):
        m = float('inf')
        for i in range(11):
            for j in range(i+1, 11):
                for k in range(j+1, 11):
                    a, b, c = pts[i], pts[j], pts[k]
                    ar = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))
                    if ar < m:
                        m = ar
        return m

    # Initial: 11 points spread in equilateral triangle using barycentric-like grid
    # Rows: 3 points at bottom, 4 in middle, 4 at top (adjusted for triangle shape)
    pts = np.array([
        [0.1, 0.05], [0.5, 0.05], [0.9, 0.05],
        [0.2, 0.25], [0.5, 0.25], [0.8, 0.25], [0.35, 0.42],
        [0.25, 0.55], [0.5, 0.55], [0.75, 0.55],
        [0.5, 0.75],
    ])
    # Ensure all inside
    for i in range(11):
        if not in_triangle(pts[i]):
            pts[i] = [0.5, 0.3]

    best = pts.copy()
    best_ma = min_area(best)
    rng = np.random.RandomState(42)
    T = 0.01
    for _ in range(3000):
        idx = rng.randint(11)
        d = rng.randn(2) * 0.03
        np_new = best[idx] + d
        if in_triangle(np_new):
            trial = best.copy()
            trial[idx] = np_new
            ma = min_area(trial)
            if ma > best_ma or rng.rand() < np.exp((ma - best_ma) / T):
                best, best_ma = trial, ma
        T *= 0.999
    return best


# EVOLVE-BLOCK-END
