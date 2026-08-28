# EVOLVE-BLOCK-START
import numpy as np

def construct_packing():
    # 5x5 grid of r=0.1 circles + 1 small circle in central gap
    # Grid positions: (0.1+0.2*i, 0.1+0.2*j) for i,j in 0..4
    # Gap center at (0.4, 0.4): dist to nearest = sqrt(0.1^2+0.1^2)=0.1414
    # Small circle radius = 0.1414 - 0.1 = 0.0414
    s = 0.2
    r = 0.1
    centers = []
    for i in range(5):
        for j in range(5):
            centers.append([r + i*s, r + j*s])
    # 26th circle in the central gap (0.4, 0.4)
    centers.append([0.4, 0.4])
    centers = np.array(centers)
    # Compute radii: large circles get 0.1, small circle gets gap radius
    radii = np.full(26, r)
    radii[25] = np.sqrt(2)*0.1 - r  # = 0.0414
    return centers, radii, radii.sum()
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
