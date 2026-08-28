# EVOLVE-BLOCK-START
import numpy as np

def construct_packing():
    # 5x5 grid of circles with radius 0.1, plus one small circle in a gap
    r = 0.1
    centers = np.array([[c, rr] for rr in [0.1, 0.3, 0.5, 0.7, 0.9] for c in [0.1, 0.3, 0.5, 0.7, 0.9]])
    # 26th circle in the gap between 4 circles at (0.2, 0.2)
    centers = np.vstack([centers, [[0.2, 0.2]]])
    radii = np.full(25, r)
    # Small circle radius: distance to nearest big circle minus r
    # Distance from (0.2,0.2) to (0.1,0.1) = sqrt(0.01+0.01)=0.1414
    radii = np.append(radii, np.sqrt(2)*0.1 - r)
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
