# EVOLVE-BLOCK-START
import numpy as np

def construct_packing():
    n = 26
    # Use a 5x5 grid pattern with 1 extra, or better: structured rows
    # 6 rows: 5,4,5,4,4,4 = 26 circles in staggered hex-like pattern
    rows = [5, 4, 5, 4, 4, 4]
    centers = []
    y_positions = np.linspace(0.1, 0.9, len(rows))
    for row_idx, count in enumerate(rows):
        y = y_positions[row_idx]
        if count % 2 == 1:
            x_start = (1.0 - (count - 1) * 0.16) / 2.0
            x_positions = np.linspace(x_start, x_start + (count-1)*0.16, count)
        else:
            x_start = (1.0 - (count - 1) * 0.16) / 2.0 + 0.08
            x_positions = np.linspace(x_start, x_start + (count-1)*0.16, count)
        for x in x_positions:
            centers.append([x, y])
    centers = np.array(centers)
    radii = compute_max_radii(centers)
    return centers, radii, np.sum(radii)

def compute_max_radii(centers):
    n = centers.shape[0]
    radii = np.array([min(x, y, 1-x, 1-y) for x, y in centers])
    # Iterative relaxation: repeatedly resolve overlaps
    for _ in range(50):
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if radii[i] + radii[j] > d and d > 1e-10:
                    # Split the gap equally
                    target_r = d / 2.0
                    new_ri = min(radii[i], target_r)
                    new_rj = min(radii[j], target_r)
                    if new_ri < radii[i] or new_rj < radii[j]:
                        radii[i] = new_ri
                        radii[j] = new_rj
                        changed = True
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
