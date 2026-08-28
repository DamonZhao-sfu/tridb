# EVOLVE-BLOCK-START
import numpy as np

def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 11.

    Returns:
        points: np.ndarray of shape (11,2) with the x,y coordinates of the points.
    """
    h = np.sqrt(3) / 2

    def in_tri(p):
        x, y = p
        return y >= -1e-9 and y <= h + 1e-9 and x >= y / h - 1e-9 and x <= 1 - y / h + 1e-9

    def min_area(pts):
        m = float('inf')
        for i in range(11):
            for j in range(i+1, 11):
                for k in range(j+1, 11):
                    ar = 0.5 * abs((pts[j,0]-pts[i,0])*(pts[k,1]-pts[i,1]) - (pts[k,0]-pts[i,0])*(pts[j,1]-pts[i,1]))
                    if ar < m:
                        m = ar
        return m

    # Better initial config: use a structured arrangement closer to known optimal
    pts = np.array([
        [0.05, 0.02], [0.5, 0.02], [0.95, 0.02],
        [0.2, 0.2], [0.5, 0.2], [0.8, 0.2],
        [0.35, 0.4], [0.65, 0.4],
        [0.5, 0.55], [0.25, 0.55], [0.75, 0.55],
    ])
    for i in range(11):
        if not in_tri(pts[i]):
            pts[i] = [0.5, 0.3]

    best = pts.copy()
    bma = min_area(best)
    rng = np.random.RandomState(42)
    T = 0.005
    for it in range(5000):
        idx = rng.randint(11)
        step = 0.02 * (1.0 - it/5000) + 0.005
        d = rng.randn(2) * step
        np_new = best[idx] + d
        if in_tri(np_new):
            trial = best.copy()
            trial[idx] = np_new
            ma = min_area(trial)
            if ma > bma or rng.rand() < np.exp((ma - bma) / T):
                best, bma = trial, ma
        T *= 0.9995
    return best

# EVOLVE-BLOCK-END
