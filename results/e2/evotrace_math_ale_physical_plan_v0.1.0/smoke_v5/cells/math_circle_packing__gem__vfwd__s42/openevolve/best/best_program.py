# EVOLVE-BLOCK-START
import numpy as np

def construct_packing():
    """Construct 26 circles in unit square maximizing sum of radii."""
    # Hexagonal grid pattern: rows of 5,5,5,5,4,2 = 26 circles
    # Use offset rows for hexagonal packing
    rows = [5, 5, 5, 5, 4, 2]
    centers = []
    
    # Vertical spacing and horizontal spacing for hex grid
    # For n circles in a row with hexagonal offset:
    # spacing_x = 1.0 / (n_max + 0.5) approximately
    # Let's use a more direct approach: place in grid with proper spacing
    
    n_rows = len(rows)
    # Even rows (0,2,4) aligned, odd rows (1,3,5) offset
    # 5-column rows: x positions at 0.1, 0.3, 0.5, 0.7, 0.9
    # 4-column rows: x positions at 0.2, 0.4, 0.6, 0.8
    # 2-column rows: x positions at 0.4, 0.6
    
    # y positions: spread 6 rows across [0.1, 0.9]
    y_positions = np.linspace(0.1, 0.9, n_rows)
    
    for r_idx, n_cols in enumerate(rows):
        y = y_positions[r_idx]
        if n_cols == 5:
            xs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        elif n_cols == 4:
            xs = np.array([0.2, 0.4, 0.6, 0.8])
        elif n_cols == 2:
            xs = np.array([0.4, 0.6])
        else:
            xs = np.linspace(0.1, 0.9, n_cols)
        for x in xs:
            centers.append([x, y])
    
    centers = np.array(centers)
    
    # Compute radii: for each circle, radius = min(distance to walls, min distance to neighbors / 2)
    n = len(centers)
    radii = np.zeros(n)
    for i in range(n):
        # Distance to walls
        wall_dist = min(centers[i][0], centers[i][1], 1 - centers[i][0], 1 - centers[i][1])
        # Distance to nearest neighbor / 2
        min_neighbor = np.inf
        for j in range(n):
            if i != j:
                d = np.sqrt((centers[i][0] - centers[j][0])**2 + (centers[i][1] - centers[j][1])**2)
                min_neighbor = min(min_neighbor, d)
        radii[i] = min(wall_dist, min_neighbor / 2)
    
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii

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
