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
        # Vectorized: compute all triangle areas at once
        n = len(pts)
        idx = np.array([[i,j,k] for i in range(n) for j in range(i+1,n) for k in range(j+1,n)])
        a, b, c = pts[idx[:,0]], pts[idx[:,1]], pts[idx[:,2]]
        areas = 0.5 * np.abs((b[:,0]-a[:,0])*(c[:,1]-a[:,1]) - (c[:,0]-a[:,0])*(b[:,1]-a[:,1]))
        return float(np.min(areas))

    # Use barycentric coordinates for well-spread initial config in equilateral triangle
    # Vertices: A=(0,0), B=(1,0), C=(0.5,h)
    # Point = u*A + v*B + w*C where u+v+w=1, u,v,w>=0
    def bary_to_xy(u, v, w):
        return np.array([v + w*0.5, w*h])

    # 11 points with good spread using barycentric coords
    bary = np.array([
        [0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.25, 0.25, 0.5],
        [0.7, 0.15, 0.15], [0.15, 0.7, 0.15], [0.15, 0.15, 0.7],
        [0.4, 0.4, 0.2], [0.4, 0.2, 0.4], [0.2, 0.4, 0.4],
        [0.5, 0.5, 0.0], [0.0, 0.5, 0.5],
    ])
    pts = np.array([bary_to_xy(*b) for b in bary])

    best = pts.copy()
    best_ma = min_area(best)
    rng = np.random.RandomState(42)
    T = 0.005
    for it in range(5000):
        idx = rng.randint(11)
        step = 0.02 * (1.0 - it/6000.0) + 0.005
        d = rng.randn(2) * step
        np_new = best[idx] + d
        if in_triangle(np_new):
            trial = best.copy()
            trial[idx] = np_new
            ma = min_area(trial)
            if ma > best_ma or rng.rand() < np.exp((ma - best_ma) / T):
                best, best_ma = trial, ma
        T *= 0.9995
    return best


# EVOLVE-BLOCK-END
