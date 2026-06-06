import numpy as np


def rotation_matrix(theta: float) -> np.ndarray:
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])


def rope_frequencies(head_dim: int, base: int = 10_000) -> np.ndarray:
    """
    For head_dim=4, returns [1.0, 0.01].
    One frequency per 2D pair.
    """
    assert head_dim % 2 == 0
    pair_indices = np.arange(0, head_dim // 2)
    return 1.0 / (base ** (2 * pair_indices / head_dim))


def apply_rope_to_vector(x: np.ndarray, position: int, freqs: np.ndarray) -> np.ndarray:
    """
    Apply RoPE to one token's Q or K vector.
    x shape: [head_dim]
    """
    rotated = x.copy()

    for pair_idx, theta_i in enumerate(freqs):
        dim0 = 2 * pair_idx
        dim1 = 2 * pair_idx + 1

        pair = x[dim0:dim1 + 1]
        angle = position * theta_i

        rotated_pair = rotation_matrix(angle) @ pair
        rotated[dim0:dim1 + 1] = rotated_pair

    return rotated


def apply_rope_to_matrix(X: np.ndarray) -> np.ndarray:
    """
    Apply RoPE to a full Q or K matrix.

    X shape:
        [seq_len, head_dim]

    Each row is one token vector.
    Each row gets a different position-based rotation.
    """
    seq_len, head_dim = X.shape
    freqs = rope_frequencies(head_dim)

    rotated = np.zeros_like(X)

    for position in range(seq_len):
        rotated[position] = apply_rope_to_vector(
            X[position],
            position,
            freqs
        )

    return rotated


def main():
    tokens = ["The", "cat", "sat", "down"]

    # Toy Q matrix: 4 tokens, each with head_dim=4.
    # Each row is one token's query vector.
    Q = np.array([
        [1.0, 0.0, 1.0, 0.0],   # The, position 0
        [1.0, 0.0, 1.0, 0.0],   # cat, position 1
        [1.0, 0.0, 1.0, 0.0],   # sat, position 2
        [1.0, 0.0, 1.0, 0.0],   # down, position 3
    ])

    print("Original Q matrix:")
    print(Q)

    freqs = rope_frequencies(head_dim=4)
    print("\nRoPE frequencies per 2D pair:")
    print(freqs)

    Q_rope = apply_rope_to_matrix(Q)

    print("\nRoPE-applied Q matrix:")
    for token, row in zip(tokens, Q_rope):
        print(f"{token:>4}: {row}")

    print("\nAttention-style similarity matrix: Q_rope @ Q_rope.T")
    similarity = Q_rope @ Q_rope.T
    print(np.round(similarity, 4))


if __name__ == "__main__":
    main()