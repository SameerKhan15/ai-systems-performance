import numpy as np
import matplotlib.pyplot as plt

def rotation_matrix(theta: float) -> np.ndarray:
    """
    2D rotation matrix.

    Rotates a vector counterclockwise by angle theta radians.
    """
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])


def rotate_vector(v: np.ndarray, theta: float) -> np.ndarray:
    """
    Rotate vector v by theta radians.
    """
    return rotation_matrix(theta) @ v


def plot_rotations():
    """
    Lab 1.1:
    Plot v = [1, 0] rotated at positions 0, 1, 2, 3.

    For simplicity, position m uses angle:
        angle = m * theta_base

    This mirrors the RoPE idea:
        position -> rotation angle
    """
    v = np.array([1.0, 0.0])

    theta_base = np.pi / 6  # 30 degrees per position
    positions = [0, 1, 2, 3]

    plt.figure(figsize=(7, 7))

    # Draw unit circle
    circle = plt.Circle((0, 0), 1.0, fill=False, linestyle="--")
    ax = plt.gca()
    ax.add_patch(circle)

    for pos in positions:
        theta = pos * theta_base
        rotated_v = rotate_vector(v, theta)

        plt.arrow(
            0,
            0,
            rotated_v[0],
            rotated_v[1],
            head_width=0.04,
            length_includes_head=True,
        )

        plt.text(
            rotated_v[0] * 1.1,
            rotated_v[1] * 1.1,
            f"pos={pos}\nθ={np.degrees(theta):.0f}°",
            ha="center",
            va="center",
        )

        print(
            f"position={pos}, "
            f"angle={theta:.4f} radians, "
            f"degrees={np.degrees(theta):.1f}, "
            f"rotated_vector={rotated_v}"
        )

    plt.axhline(0, linewidth=0.8)
    plt.axvline(0, linewidth=0.8)
    plt.xlim(-1.3, 1.3)
    plt.ylim(-1.3, 1.3)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("RoPE Lab 1.1: Position Encoded as Vector Rotation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # same original vector [1,0], but each position rotates it farther around the unit circle.
    # This is the basic RoPE idea: position becomes rotation angle
    plot_rotations()