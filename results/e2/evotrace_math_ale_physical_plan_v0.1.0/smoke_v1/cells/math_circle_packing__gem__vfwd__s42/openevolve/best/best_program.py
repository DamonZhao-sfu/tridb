# EVOLVE-BLOCK-START
"""Improved circle packing with local search optimization"""
import numpy as np

def construct_packing():
    """
    Construct a specific arrangement of 26 circles in a unit square
    using a jittered grid initialization followed by local search optimization.
    """
    n = 26
    
    # 1. Initialization: Jittered Grid
    # A grid of roughly 5x5 = 25 points, plus one extra, provides a good starting distribution.
    # We use a 5x5 grid and add one random point, or just use 26 points in a jittered grid.
    # Let's create a grid of 5x6=30 points and pick 26, or just 5x5 + 1.
    # Better: Use a 5x5 grid (25 points) and add 1 point in the center or random.
    # Actually, a simple jittered grid of size ceil(sqrt(26)) ~ 5 is good.
    
    rows = 5
    cols = 5
    # Generate base grid points
    x_coords = np.linspace(0.1, 0.9, cols)
    y_coords = np.linspace(0.1, 0.9, rows)
    
    centers = np.array([[x, y] for y in y_coords for x in x_coords])
    
    # We have 25 points. We need 26. Add one more point, e.g., in the center or random.
    # To keep it deterministic and structured, let's add a point in the center of one of the cells?
    # Or just use a 5x5 grid and jitter it, then add a 26th point randomly?
    # For reproducibility and simplicity, let's just use the 25 grid points and add one at (0.5, 0.5) if not already close?
    # Actually, 26 is close to 25. Let's just use 25 grid points and add one random point?
    # No, let's use a 5x6 grid and drop 4? 
    # Simplest: Use 25 grid points. Add 1 point at (0.5, 0.5).
    extra_point = np.array([[0.5, 0.5]])
    centers = np.vstack([centers, extra_point])
    
    # Apply small deterministic jitter to break symmetry (using a fixed seed for reproducibility in evolution)
    rng = np.random.default_rng(42)
    centers += rng.uniform(-0.02, 0.02, size=centers.shape)
    
    # Clip to ensure inside unit square with a small margin
    centers = np.clip(centers, 0.05, 0.95)
    
    # 2. Local Search Optimization
    # Objective: Maximize the minimum distance between any two circles (or sum of radii).
    # We will iteratively move circles to increase the minimum distance.
    
    current_score = calculate_score(centers)
    
    for iteration in range(1000):
        # Pick a random circle to move
        idx = rng.integers(0, n)
        
        # Try to move it in a random direction
        angle = rng.uniform(0, 2 * np.pi)
        step_size = 0.01
        dx = step_size * np.cos(angle)
        dy = step_size * np.sin(angle)
        
        new_center = centers[idx].copy()
        new_center[0] += dx
        new_center[1] += dy
        
        # Keep inside bounds
        if new_center[0] < 0.05 or new_center[0] > 0.95 or new_center[1] < 0.05 or new_center[1] > 0.95:
            continue
            
        # Calculate new score
        old_center = centers[idx].copy()
        centers[idx] = new_center
        new_score = calculate_score(centers)
        
        # Accept if better (or with some probability for exploration, but greedy is safer for now)
        if new_score > current_score:
            current_score = new_score
        else:
            # Revert
            centers[idx] = old_center
            
    # 3. Compute Radii
    radii = compute_max_radii(centers)
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

def calculate_score(centers):
    """
    Calculate a score for the packing. 
    We want to maximize the sum of the minimum distances to neighbors and walls.
    A simple proxy is the minimum distance between any two circles, 
    but sum of min-distances is better for total radii.
    """
    n = centers.shape[0]
    min_dists = np.zeros(n)
    
    for i in range(n):
        # Distance to walls
        wall_dist = min(centers[i][0], 1 - centers[i][0], centers[i][1], 1 - centers[i][1])
        min_dists[i] = wall_dist
        
        # Distance to other circles
        for j in range(n):
            if i == j:
                continue
            dist = np.linalg.norm(centers[i] - centers[j])
            # The constraint is r_i + r_j <= dist.
            # To maximize sum of radii, we want to maximize the minimum of (dist/2) roughly?
            # Actually, the max radius for i given j is dist - r_j.
            # This is complex. A good heuristic score is the sum of min(dist_to_neighbor, dist_to_wall).
            # Or simply: For each i, r_i is limited by min(dist_to_wall, min_j(dist_ij - r_j)).
            # Let's just use the minimum distance to any other circle or wall as a proxy for potential radius.
            min_dists[i] = min(min_dists[i], dist)
            
    # The score is the sum of these minimum distances.
    # Note: This is an upper bound on the sum of radii if we could perfectly distribute them,
    # but it's a good optimization target.
    return np.sum(min_dists)

def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.
    Uses a simple iterative relaxation.
    """
    n = centers.shape[0]
    radii = np.ones(n)
    
    # Initialize with wall distances
    for i in range(n):
        radii[i] = min(centers[i][0], 1 - centers[i][0], centers[i][1], 1 - centers[i][1])
        
    # Iteratively adjust for overlaps
    for _ in range(50):
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                if radii[i] + radii[j] > dist:
                    # Scale down both to fit
                    scale = dist / (radii[i] + radii[j])
                    radii[i] *= scale
                    radii[j] *= scale
                    # Re-apply wall constraints
                    radii[i] = min(radii[i], min(centers[i][0], 1 - centers[i][0], centers[i][1], 1 - centers[i][1]))
                    radii[j] = min(radii[j], min(centers[j][0], 1 - centers[j][0], centers[j][1], 1 - centers[j][1]))
                    
    return radii


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


def visualize(centers, radii):
    """
    Visualize the circle packing

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw unit square
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    # Draw circles
    for i, (center, radius) in enumerate(zip(centers, radii)):
        circle = Circle(center, radius, alpha=0.5)
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center")

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
    # AlphaEvolve improved this to 2.635

    # Uncomment to visualize:
    visualize(centers, radii)
