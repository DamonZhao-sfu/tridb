# EVOLVE-BLOCK-START
"""Optimized circle packing for n=26 circles using grid + corner layout"""
import numpy as np


def construct_packing():
    """
    Construct arrangement of 26 circles in unit square maximizing sum of radii.
    Uses a 5x5 grid (25 circles) plus 1 extra circle in a gap.
    """
    # 5x5 grid pattern with equal spacing
    # This gives 25 evenly-spaced circles, then add 1 more
    n = 26
    centers = np.zeros((n, 2))

    # 5x5 grid: positions at (0.1, 0.3, 0.5, 0.7, 0.9)
    grid_pos = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    idx = 0
    for x in grid_pos:
        for y in grid_pos:
            centers[idx] = [x, y]
            idx += 1

    # 26th circle: place in a gap between grid points
    # The center gap at (0.5, 0.5) is already taken, so use offset
    # Place at (0.4, 0.4) area - between grid points
    centers[25] = [0.4, 0.4]

    # Compute optimal radii
    radii = compute_radii(centers)

    return centers, radii, np.sum(radii)


def compute_radii(centers):
    """
    Compute max radii ensuring no overlap and circles stay in unit square.
    Uses iterative relaxation for better results.
    """
    n = centers.shape[0]
    # Initial radius: distance to nearest border
    radii = np.array([min(c[0], c[1], 1-c[0], 1-c[1]) for c in centers])

    # Iteratively resolve overlaps
    for _ in range(50):
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if radii[i] + radii[j] > d and d > 1e-10:
                    # Scale down proportionally
                    total = radii[i] + radii[j]
                    scale = d / total
                    radii[i] *= scale
                    radii[j] *= scale
                    changed = True
        if not changed:
            break

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
