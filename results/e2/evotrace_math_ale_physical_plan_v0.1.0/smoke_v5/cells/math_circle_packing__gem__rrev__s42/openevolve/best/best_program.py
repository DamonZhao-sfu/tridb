# EVOLVE-BLOCK-START
import numpy as np

def construct_packing():
    n = 26
    # Use a 5x5 grid (25 circles) + 1 extra at center offset
    # Grid with spacing ~0.18 gives good radius utilization
    xs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    ys = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    centers = np.array([[x, y] for y in ys for x in xs])  # 25 circles
    # Add 26th circle at a position that fits - use center of a cell
    centers = np.vstack([centers, [0.5, 0.5]])
    # Remove duplicate if 0.5,0.5 already exists (it does in 5x5 grid)
    # Instead, shift: use 5x5 minus center, plus two center-adjacent
    centers = np.array([[x, y] for y in ys for x in xs if not (x == 0.5 and y == 0.5)])  # 24
    centers = np.vstack([centers, [0.4, 0.5], [0.6, 0.5]])  # 26
    # Compute radii: min distance to border and to other circles / 2
    radii = np.zeros(n)
    for i in range(n):
        r = min(centers[i][0], centers[i][1], 1 - centers[i][0], 1 - centers[i][1])
        for j in range(n):
            if i != j:
                d = np.linalg.norm(centers[i] - centers[j])
                r = min(r, d / 2)
        radii[i] = r
    return centers, radii, np.sum(radii)
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
