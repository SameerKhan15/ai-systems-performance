import math
import time
from dataclasses import dataclass

import torch
import matplotlib.pyplot as plt


@dataclass
class AttentionProfile:
    seq_len: int
    embed_dim: int
    q_proj_ms: float
    k_proj_ms: float
    v_proj_ms: float
    scores_ms: float
    softmax_ms: float
    output_ms: float
    total_ms: float
    attention_matrix_mb: float


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync_if_needed(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def timed_step(device: torch.device, fn):
    sync_if_needed(device)
    start = time.perf_counter()
    result = fn()
    sync_if_needed(device)
    end = time.perf_counter()
    return result, (end - start) * 1000


def self_attention_profile(seq_len: int, embed_dim: int, device: torch.device) -> AttentionProfile:
    """
    Implements single-head self-attention from scratch:

        X -> Q, K, V
        scores = QK^T / sqrt(d)
        weights = softmax(scores)
        output = weights V

    Shapes:
        X       : [seq_len, embed_dim]
        Wq/Wk/Wv: [embed_dim, embed_dim]
        Q/K/V   : [seq_len, embed_dim]
        scores  : [seq_len, seq_len]
        output  : [seq_len, embed_dim]
    """

    torch.manual_seed(42)

    X = torch.randn(seq_len, embed_dim, device=device)

    Wq = torch.randn(embed_dim, embed_dim, device=device)
    Wk = torch.randn(embed_dim, embed_dim, device=device)
    Wv = torch.randn(embed_dim, embed_dim, device=device)

    Q, q_proj_ms = timed_step(device, lambda: X @ Wq)
    K, k_proj_ms = timed_step(device, lambda: X @ Wk)
    V, v_proj_ms = timed_step(device, lambda: X @ Wv)

    scores, scores_ms = timed_step(
        device,
        lambda: (Q @ K.T) / math.sqrt(embed_dim)
    )

    weights, softmax_ms = timed_step(
        device,
        lambda: torch.softmax(scores, dim=-1)
    )

    output, output_ms = timed_step(
        device,
        lambda: weights @ V
    )

    total_ms = (
        q_proj_ms
        + k_proj_ms
        + v_proj_ms
        + scores_ms
        + softmax_ms
        + output_ms
    )

    attention_matrix_mb = (
        seq_len * seq_len * scores.element_size()
    ) / (1024 ** 2)

    # Keep output alive so computation is not optimized away.
    _ = output.sum().item()

    return AttentionProfile(
        seq_len=seq_len,
        embed_dim=embed_dim,
        q_proj_ms=q_proj_ms,
        k_proj_ms=k_proj_ms,
        v_proj_ms=v_proj_ms,
        scores_ms=scores_ms,
        softmax_ms=softmax_ms,
        output_ms=output_ms,
        total_ms=total_ms,
        attention_matrix_mb=attention_matrix_mb,
    )


def print_profile(profile: AttentionProfile):
    print("=" * 60)
    print(f"seq_len              : {profile.seq_len}")
    print(f"embed_dim            : {profile.embed_dim}")
    print(f"Q projection         : {profile.q_proj_ms:.3f} ms")
    print(f"K projection         : {profile.k_proj_ms:.3f} ms")
    print(f"V projection         : {profile.v_proj_ms:.3f} ms")
    print(f"QK^T scores          : {profile.scores_ms:.3f} ms")
    print(f"softmax              : {profile.softmax_ms:.3f} ms")
    print(f"weights @ V          : {profile.output_ms:.3f} ms")
    print(f"total                : {profile.total_ms:.3f} ms")
    print(f"attention matrix     : {profile.attention_matrix_mb:.2f} MB")
    print("=" * 60)


def plot_results(profiles):
    seq_lens = [p.seq_len for p in profiles]

    q_proj_ms = [p.q_proj_ms for p in profiles]
    k_proj_ms = [p.k_proj_ms for p in profiles]
    v_proj_ms = [p.v_proj_ms for p in profiles]
    scores_ms = [p.scores_ms for p in profiles]
    softmax_ms = [p.softmax_ms for p in profiles]
    output_ms = [p.output_ms for p in profiles]
    total_ms = [p.total_ms for p in profiles]
    attn_mb = [p.attention_matrix_mb for p in profiles]

    # 1. Combined runtime chart
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lens, q_proj_ms, marker="o", label="Q projection")
    plt.plot(seq_lens, k_proj_ms, marker="o", label="K projection")
    plt.plot(seq_lens, v_proj_ms, marker="o", label="V projection")
    plt.plot(seq_lens, scores_ms, marker="o", label="QK^T scores")
    plt.plot(seq_lens, softmax_ms, marker="o", label="Softmax")
    plt.plot(seq_lens, output_ms, marker="o", label="weights @ V")
    plt.plot(seq_lens, total_ms, marker="o", label="Total")

    plt.xlabel("Sequence Length")
    plt.ylabel("Runtime (ms)")
    plt.title("Self-Attention Runtime Breakdown by Sequence Length")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2. Separate chart for the big bottleneck pieces
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lens, scores_ms, marker="o", label="QK^T scores")
    plt.plot(seq_lens, softmax_ms, marker="o", label="Softmax")
    plt.plot(seq_lens, output_ms, marker="o", label="weights @ V")
    plt.plot(seq_lens, total_ms, marker="o", label="Total")

    plt.xlabel("Sequence Length")
    plt.ylabel("Runtime (ms)")
    plt.title("Dominant Attention Costs by Sequence Length")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 3. Attention matrix memory chart
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lens, attn_mb, marker="o")

    plt.xlabel("Sequence Length")
    plt.ylabel("Attention Matrix Size (MB)")
    plt.title("Attention Matrix Memory Scaling")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    device = pick_device()
    print(f"Using device: {device}")

    embed_dim = 128

    seq_lens = [
        64,
        128,
        256,
        512,
        1024,
        2048,
    ]

    profiles = []

    # Warmup
    _ = self_attention_profile(64, embed_dim, device)

    for seq_len in seq_lens:
        profile = self_attention_profile(seq_len, embed_dim, device)
        print_profile(profile)
        profiles.append(profile)

    plot_results(profiles)


if __name__ == "__main__":
    main()