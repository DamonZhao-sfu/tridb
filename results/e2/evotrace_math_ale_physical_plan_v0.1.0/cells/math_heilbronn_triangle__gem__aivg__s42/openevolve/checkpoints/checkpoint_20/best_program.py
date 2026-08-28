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

    # Known good 11-point config in equilateral triangle (side=1)
    # Using barycentric coordinates converted to Cartesian
    # Triangle vertices: A=(0,0), B=(1,0), C=(0.5,h)
    # Barycentric (a,b,c) -> x = b + c*0.5, y = c*h
    bc = np.array([
        [0.1, 0.4, 0.5],  # bottom row
        [0.3, 0.4, 0.3],
        [0.5, 0.4, 0.1],
        [0.1, 0.25, 0.65],  # middle row
        [0.35, 0.35, 0.3],
        [0.55, 0.25, 0.2],
        [0.1, 0.1, 0.8],   # upper
        [0.3, 0.3, 0.4],
        [0.5, 0.2, 0.3],
        [0.2, 0.2, 0.6],   # top
        [0.4, 0.4, 0.2],
    ])
    pts = np.column_stack([bc[:,1] + bc[:,2]*0.5, bc[:,2]*h])
    # Clamp to ensure inside
    for i in range(11):
        if not in_triangle(pts[i]):
            pts[i] = [0.5, 0.3]

    best = pts.copy()
    best_ma = min_area(best)
    rng = np.random.RandomState(42)
    T = 0.05
    step = 0.1
    for it in range(5000):
        idx = rng.randint(11)
        d = rng.randn(2) * step
        np_new = best[idx] + d
        if in_triangle(np_new):
            trial = best.copy()
            trial[idx] = np_new
            ma = min_area(trial)
            if ma > best_ma or rng.rand() < np.exp((ma - best_ma) / T):
                best, best_ma = trial, ma
        T *= 0.9995
        step *= 0.9997
    return best


# EVOLVE-BLOCK-END
