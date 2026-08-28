# EVOLVE-BLOCK-START
import numpy as np


def _min_triangle_area(points: np.ndarray) -> float:
    """Compute the minimum triangle area among all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]))
                if area < min_area:
                    min_area = area
    return min_area


def _move_point_toward_bigger_min_triangle(points: np.ndarray, idx: int, step: float, rng: np.random.Generator) -> np.ndarray:
    """Try to move point idx to improve the minimum triangle area."""
    n = len(points)
    best_new_point = points[idx].copy()
    best_min_area = _min_triangle_area(points)
    
    # Try several random perturbations
    for _ in range(20):
        direction = rng.normal(size=2)
        direction /= (np.linalg.norm(direction) + 1e-10)
        new_point = points[idx] + step * direction
        # Keep within unit square [0,1]^2
        new_point = np.clip(new_point, 0.0, 1.0)
        
        new_points = points.copy()
        new_points[idx] = new_point
        new_min_area = _min_triangle_area(new_points)
        
        if new_min_area > best_min_area:
            best_min_area = new_min_area
            best_new_point = new_point
    
    return best_new_point


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Uses a structured initial layout followed by iterative local optimization
    to maximize the minimum triangle area.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    rng = np.random.default_rng(seed=42)
    
    # Initialize with a structured layout: points on a regular 13-gon inscribed in unit square
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # Scale to fit in unit square centered at (0.5, 0.5)
    radius = 0.45
    points = np.column_stack([
        0.5 + radius * np.cos(angles),
        0.5 + radius * np.sin(angles)
    ])
    
    # Also add a center point and adjust: use 12 on circle + 1 center
    # Actually let's use a better init: 12 points on two concentric rings + 1 center
    # Ring 1: 6 points at radius 0.3
    # Ring 2: 6 points at radius 0.45 (offset by 30 degrees)
    # Center: 1 point
    
    angles_inner = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    angles_outer = np.linspace(np.pi/6, 2 * np.pi + np.pi/6, 6, endpoint=False)
    
    points = np.vstack([
        np.column_stack([0.5 + 0.3 * np.cos(angles_inner), 0.5 + 0.3 * np.sin(angles_inner)]),
        np.column_stack([0.5 + 0.45 * np.cos(angles_outer), 0.5 + 0.45 * np.sin(angles_outer)]),
        np.array([[0.5, 0.5]])
    ])
    
    # Iterative local optimization with simulated annealing style
    current_min_area = _min_triangle_area(points)
    step_size = 0.1
    n_iter = 500
    
    for iteration in range(n_iter):
        # Gradually decrease step size
        step_size = 0.1 * (1 - iteration / n_iter) + 0.005
        
        # Pick a random point to move
        idx = rng.integers(0, n)
        
        # Try to improve by moving this point
        new_point = _move_point_toward_bigger_min_triangle(points, idx, step_size, rng)
        
        if np.any(new_point != points[idx]):
            points[idx] = new_point
            new_min_area = _min_triangle_area(points)
            if new_min_area > current_min_area:
                current_min_area = new_min_area
    
    # Final normalization: scale points to fill the unit square better
    # Find bounding box and scale
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    center = (min_coords + max_coords) / 2
    extent = (max_coords - min_coords)
    max_extent = np.max(extent)
    if max_extent > 0:
        scale = 0.9 / max_extent
        points = (points - center) * scale + 0.5
    
    return points


# EVOLVE-BLOCK-END
