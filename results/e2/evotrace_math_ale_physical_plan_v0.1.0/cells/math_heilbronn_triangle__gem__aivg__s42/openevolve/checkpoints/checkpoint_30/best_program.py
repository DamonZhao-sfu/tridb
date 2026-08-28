# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """
    Arrange 11 points in a unit equilateral triangle to maximize
    the area of the smallest triangle formed by any 3 points.
    """
    h = np.sqrt(3) / 2
    rng = np.random.default_rng(42)
    # Triangular lattice rows 4,3,2,1 = 10 + 1 interior = 11
    pts = np.array([
        [0.05, 0.02], [0.35, 0.02], [0.65, 0.02], [0.95, 0.02],
        [0.2, h/3], [0.5, h/3], [0.8, h/3],
        [0.33, 2*h/3], [0.67, 2*h/3],
        [0.5, 0.9*h],
        [0.5, 0.45*h],
    ])
    # Small perturbation to break collinearity
    eps = 0.005
    pts += rng.uniform(-eps, eps, size=pts.shape)
    # Ensure inside triangle: for point (x,y), need y <= h*(1-|2x-1|)
    for i in range(len(pts)):
        x, y = pts[i]
        y_max = h * (1 - abs(2*x - 1))
        if y > y_max:
            y = y_max * 0.99
        if y < 0:
            y = 0.01
        if x < 0:
            x = 0.01
        if x > 1:
            x = 0.99
        pts[i] = [x, y]
    return pts
# EVOLVE-BLOCK-END
