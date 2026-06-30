import math
import time
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from contextlib import redirect_stdout

import torch
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


NUM_RUNS = 30
WINDOW_SIZE = 2


@dataclass
class AttentionProfile:
    seq_len: int
    embed_dim: int
    window_size: int
    max_attended_tokens: int
    q_proj_ms: float
    k_proj_ms: float
    v_proj_ms: float
    scores_ms: float
    softmax_ms: float
    output_ms: float
    total_ms: float
    attention_matrix_mb: float
    gpu_peak_mb: float


@dataclass
class AttentionStats:
    seq_len: int
    embed_dim: int
    window_size: int
    max_attended_tokens: int
    runs: int

    q_proj_median_ms: float
    q_proj_p95_ms: float

    k_proj_median_ms: float
    k_proj_p95_ms: float

    v_proj_median_ms: float
    v_proj_p95_ms: float

    scores_median_ms: float
    scores_p95_ms: float

    softmax_median_ms: float
    softmax_p95_ms: float

    output_median_ms: float
    output_p95_ms: float

    total_median_ms: float
    total_p95_ms: float

    attention_matrix_mb: float
    gpu_peak_median_mb: float
    gpu_peak_p95_mb: float


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def percentile(values, pct: float) -> float:
    values = sorted(values)
    if not values:
        raise ValueError("Cannot compute percentile of empty values")

    k = (len(values) - 1) * (pct / 100.0)
    lower = math.floor(k)
    upper = math.ceil(k)

    if lower == upper:
        return values[int(k)]

    weight = k - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def median(values) -> float:
    return percentile(values, 50)


def p95(values) -> float:
    return percentile(values, 95)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync_if_needed(device: torch.device) -> None:
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


def build_sliding_window_indices(
    seq_len: int,
    window_size: int,
    device: torch.device,
):
    """
    For each token i, build indices [i-window_size, ..., i, ..., i+window_size].

    Invalid edge positions are clamped for safe indexing, then masked out before
    softmax. For window_size=2 and seq_len=4, the effective neighborhoods are:

      token 0 -> [0, 1, 2]
      token 1 -> [0, 1, 2, 3]
      token 2 -> [0, 1, 2, 3]
      token 3 -> [1, 2, 3]

    For the user's sentence example with window_size=1:
      The  -> [The, cat]
      cat  -> [The, cat, sat]
      sat  -> [cat, sat, down]
      down -> [sat, down]
    """
    positions = torch.arange(seq_len, device=device).unsqueeze(1)
    offsets = torch.arange(-window_size, window_size + 1, device=device).unsqueeze(0)

    raw_indices = positions + offsets
    valid_mask = (raw_indices >= 0) & (raw_indices < seq_len)
    safe_indices = raw_indices.clamp(0, seq_len - 1)

    return safe_indices, valid_mask


@torch.no_grad()
def sliding_window_attention_profile(
    seq_len: int,
    embed_dim: int,
    window_size: int,
    device: torch.device,
) -> AttentionProfile:
    torch.manual_seed(42)

    X = torch.randn(seq_len, embed_dim, device=device)

    Wq = torch.randn(embed_dim, embed_dim, device=device)
    Wk = torch.randn(embed_dim, embed_dim, device=device)
    Wv = torch.randn(embed_dim, embed_dim, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    Q, q_proj_ms = timed_step(device, lambda: X @ Wq)
    K, k_proj_ms = timed_step(device, lambda: X @ Wk)
    V, v_proj_ms = timed_step(device, lambda: X @ Wv)

    safe_indices, valid_mask = build_sliding_window_indices(
        seq_len=seq_len,
        window_size=window_size,
        device=device,
    )

    def compute_window_scores():
        # K_windows shape: [seq_len, 2 * window_size + 1, embed_dim]
        # Q[:, None, :] shape: [seq_len, 1, embed_dim]
        # scores shape: [seq_len, 2 * window_size + 1]
        K_windows = K[safe_indices]
        scores = (Q.unsqueeze(1) * K_windows).sum(dim=-1) / math.sqrt(embed_dim)

        # Invalid edge positions must not receive probability mass.
        return scores.masked_fill(~valid_mask, float("-inf"))

    scores, scores_ms = timed_step(device, compute_window_scores)

    weights, softmax_ms = timed_step(
        device,
        lambda: torch.softmax(scores, dim=-1),
    )

    def compute_window_output():
        # V_windows shape: [seq_len, 2 * window_size + 1, embed_dim]
        # weights[:, :, None] shape: [seq_len, 2 * window_size + 1, 1]
        # output shape: [seq_len, embed_dim]
        V_windows = V[safe_indices]
        return (weights.unsqueeze(-1) * V_windows).sum(dim=1)

    output, output_ms = timed_step(device, compute_window_output)

    sync_if_needed(device)
    _ = output.sum().item()

    total_ms = (
        q_proj_ms
        + k_proj_ms
        + v_proj_ms
        + scores_ms
        + softmax_ms
        + output_ms
    )

    max_attended_tokens = 2 * window_size + 1
    attention_matrix_mb = (
        seq_len * max_attended_tokens * scores.element_size()
    ) / (1024 ** 2)

    if device.type == "cuda":
        gpu_peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        gpu_peak_mb = 0.0

    return AttentionProfile(
        seq_len=seq_len,
        embed_dim=embed_dim,
        window_size=window_size,
        max_attended_tokens=max_attended_tokens,
        q_proj_ms=q_proj_ms,
        k_proj_ms=k_proj_ms,
        v_proj_ms=v_proj_ms,
        scores_ms=scores_ms,
        softmax_ms=softmax_ms,
        output_ms=output_ms,
        total_ms=total_ms,
        attention_matrix_mb=attention_matrix_mb,
        gpu_peak_mb=gpu_peak_mb,
    )


def summarize_profiles(profiles) -> AttentionStats:
    first = profiles[0]

    return AttentionStats(
        seq_len=first.seq_len,
        embed_dim=first.embed_dim,
        window_size=first.window_size,
        max_attended_tokens=first.max_attended_tokens,
        runs=len(profiles),

        q_proj_median_ms=median([p.q_proj_ms for p in profiles]),
        q_proj_p95_ms=p95([p.q_proj_ms for p in profiles]),

        k_proj_median_ms=median([p.k_proj_ms for p in profiles]),
        k_proj_p95_ms=p95([p.k_proj_ms for p in profiles]),

        v_proj_median_ms=median([p.v_proj_ms for p in profiles]),
        v_proj_p95_ms=p95([p.v_proj_ms for p in profiles]),

        scores_median_ms=median([p.scores_ms for p in profiles]),
        scores_p95_ms=p95([p.scores_ms for p in profiles]),

        softmax_median_ms=median([p.softmax_ms for p in profiles]),
        softmax_p95_ms=p95([p.softmax_ms for p in profiles]),

        output_median_ms=median([p.output_ms for p in profiles]),
        output_p95_ms=p95([p.output_ms for p in profiles]),

        total_median_ms=median([p.total_ms for p in profiles]),
        total_p95_ms=p95([p.total_ms for p in profiles]),

        attention_matrix_mb=first.attention_matrix_mb,
        gpu_peak_median_mb=median([p.gpu_peak_mb for p in profiles]),
        gpu_peak_p95_mb=p95([p.gpu_peak_mb for p in profiles]),
    )


def print_profile(profile: AttentionProfile, run_idx: int) -> None:
    print(f"Run {run_idx}")
    print(f"window size          : {profile.window_size}")
    print(f"max attended tokens  : {profile.max_attended_tokens}")
    print(f"Q projection         : {profile.q_proj_ms:.3f} ms")
    print(f"K projection         : {profile.k_proj_ms:.3f} ms")
    print(f"V projection         : {profile.v_proj_ms:.3f} ms")
    print(f"windowed QK scores   : {profile.scores_ms:.3f} ms")
    print(f"windowed softmax     : {profile.softmax_ms:.3f} ms")
    print(f"windowed weights @ V : {profile.output_ms:.3f} ms")
    print(f"total                : {profile.total_ms:.3f} ms")
    print(f"attention scores     : {profile.attention_matrix_mb:.2f} MB")
    print(f"peak GPU memory      : {profile.gpu_peak_mb:.2f} MB")
    print("-" * 60)


def print_stats(stats: AttentionStats) -> None:
    print("=" * 80)
    print(f"SUMMARY seq_len      : {stats.seq_len}")
    print(f"embed_dim            : {stats.embed_dim}")
    print(f"window size          : {stats.window_size}")
    print(f"max attended tokens  : {stats.max_attended_tokens}")
    print(f"runs                 : {stats.runs}")
    print()
    print("Median")
    print(f"Q projection         : {stats.q_proj_median_ms:.3f} ms")
    print(f"K projection         : {stats.k_proj_median_ms:.3f} ms")
    print(f"V projection         : {stats.v_proj_median_ms:.3f} ms")
    print(f"windowed QK scores   : {stats.scores_median_ms:.3f} ms")
    print(f"windowed softmax     : {stats.softmax_median_ms:.3f} ms")
    print(f"windowed weights @ V : {stats.output_median_ms:.3f} ms")
    print(f"total                : {stats.total_median_ms:.3f} ms")
    print()
    print("P95")
    print(f"Q projection         : {stats.q_proj_p95_ms:.3f} ms")
    print(f"K projection         : {stats.k_proj_p95_ms:.3f} ms")
    print(f"V projection         : {stats.v_proj_p95_ms:.3f} ms")
    print(f"windowed QK scores   : {stats.scores_p95_ms:.3f} ms")
    print(f"windowed softmax     : {stats.softmax_p95_ms:.3f} ms")
    print(f"windowed weights @ V : {stats.output_p95_ms:.3f} ms")
    print(f"total                : {stats.total_p95_ms:.3f} ms")
    print()
    print(f"attention scores     : {stats.attention_matrix_mb:.2f} MB")
    print(f"peak GPU memory med  : {stats.gpu_peak_median_mb:.2f} MB")
    print(f"peak GPU memory p95  : {stats.gpu_peak_p95_mb:.2f} MB")
    print("=" * 80)


def save_csv(stats_list, output_dir: str = "plots") -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_file = output_path / "metrics.csv"

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "seq_len",
            "embed_dim",
            "window_size",
            "max_attended_tokens",
            "runs",

            "q_proj_median_ms",
            "q_proj_p95_ms",

            "k_proj_median_ms",
            "k_proj_p95_ms",

            "v_proj_median_ms",
            "v_proj_p95_ms",

            "scores_median_ms",
            "scores_p95_ms",

            "softmax_median_ms",
            "softmax_p95_ms",

            "output_median_ms",
            "output_p95_ms",

            "total_median_ms",
            "total_p95_ms",

            "attention_matrix_mb",
            "gpu_peak_median_mb",
            "gpu_peak_p95_mb",
        ])

        for s in stats_list:
            writer.writerow([
                s.seq_len,
                s.embed_dim,
                s.window_size,
                s.max_attended_tokens,
                s.runs,

                s.q_proj_median_ms,
                s.q_proj_p95_ms,

                s.k_proj_median_ms,
                s.k_proj_p95_ms,

                s.v_proj_median_ms,
                s.v_proj_p95_ms,

                s.scores_median_ms,
                s.scores_p95_ms,

                s.softmax_median_ms,
                s.softmax_p95_ms,

                s.output_median_ms,
                s.output_p95_ms,

                s.total_median_ms,
                s.total_p95_ms,

                s.attention_matrix_mb,
                s.gpu_peak_median_mb,
                s.gpu_peak_p95_mb,
            ])

    print(f"Saved metrics to: {csv_file}")


def plot_results(stats_list, output_dir: str = "plots") -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    seq_lens = [s.seq_len for s in stats_list]
    window_size = stats_list[0].window_size

    plt.figure(figsize=(10, 6))
    plt.plot(seq_lens, [s.q_proj_median_ms for s in stats_list], marker="o", label="Q projection median")
    plt.plot(seq_lens, [s.q_proj_p95_ms for s in stats_list], marker="o", label="Q projection p95")
    plt.plot(seq_lens, [s.k_proj_median_ms for s in stats_list], marker="o", label="K projection median")
    plt.plot(seq_lens, [s.k_proj_p95_ms for s in stats_list], marker="o", label="K projection p95")
    plt.plot(seq_lens, [s.v_proj_median_ms for s in stats_list], marker="o", label="V projection median")
    plt.plot(seq_lens, [s.v_proj_p95_ms for s in stats_list], marker="o", label="V projection p95")
    plt.plot(seq_lens, [s.scores_median_ms for s in stats_list], marker="o", label="Windowed QK median")
    plt.plot(seq_lens, [s.scores_p95_ms for s in stats_list], marker="o", label="Windowed QK p95")
    plt.plot(seq_lens, [s.softmax_median_ms for s in stats_list], marker="o", label="Windowed softmax median")
    plt.plot(seq_lens, [s.softmax_p95_ms for s in stats_list], marker="o", label="Windowed softmax p95")
    plt.plot(seq_lens, [s.output_median_ms for s in stats_list], marker="o", label="Windowed weights @ V median")
    plt.plot(seq_lens, [s.output_p95_ms for s in stats_list], marker="o", label="Windowed weights @ V p95")
    plt.plot(seq_lens, [s.total_median_ms for s in stats_list], marker="o", label="Total median")
    plt.plot(seq_lens, [s.total_p95_ms for s in stats_list], marker="o", label="Total p95")
    plt.xlabel("Sequence Length")
    plt.ylabel("Runtime (ms)")
    plt.title(f"Sliding-Window Attention Runtime Breakdown: Median and P95, window={window_size}")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path / "sliding_window_attention_runtime_breakdown.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(seq_lens, [s.scores_median_ms for s in stats_list], marker="o", label="Windowed QK median")
    plt.plot(seq_lens, [s.scores_p95_ms for s in stats_list], marker="o", label="Windowed QK p95")
    plt.plot(seq_lens, [s.softmax_median_ms for s in stats_list], marker="o", label="Windowed softmax median")
    plt.plot(seq_lens, [s.softmax_p95_ms for s in stats_list], marker="o", label="Windowed softmax p95")
    plt.plot(seq_lens, [s.output_median_ms for s in stats_list], marker="o", label="Windowed weights @ V median")
    plt.plot(seq_lens, [s.output_p95_ms for s in stats_list], marker="o", label="Windowed weights @ V p95")
    plt.plot(seq_lens, [s.total_median_ms for s in stats_list], marker="o", label="Total median")
    plt.plot(seq_lens, [s.total_p95_ms for s in stats_list], marker="o", label="Total p95")
    plt.xlabel("Sequence Length")
    plt.ylabel("Runtime (ms)")
    plt.title(f"Dominant Sliding-Window Attention Costs: Median and P95, window={window_size}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path / "sliding_window_attention_dominant_costs.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(seq_lens, [s.attention_matrix_mb for s in stats_list], marker="o", label="Attention score storage")
    plt.plot(seq_lens, [s.gpu_peak_median_mb for s in stats_list], marker="o", label="Peak GPU memory median")
    plt.plot(seq_lens, [s.gpu_peak_p95_mb for s in stats_list], marker="o", label="Peak GPU memory p95")
    plt.xlabel("Sequence Length")
    plt.ylabel("Memory (MB)")
    plt.title(f"Sliding-Window Attention Memory Scaling: Median and P95, window={window_size}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path / "sliding_window_attention_memory_scaling.png", dpi=150)
    plt.close()

    print(f"Saved plots to: {output_path.resolve()}")


def run_experiment(output_dir: str) -> None:
    device = pick_device()
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU Name        : {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version    : {torch.version.cuda}")
        print(f"PyTorch Version : {torch.__version__}")

    embed_dim = 128
    window_size = WINDOW_SIZE

    seq_lens = [
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
    ]

    stats_list = []

    print()
    print("Warmup...")
    for _ in range(5):
        _ = sliding_window_attention_profile(512, embed_dim, window_size, device)

    for seq_len in seq_lens:
        print()
        print("#" * 80)
        print(
            f"Running seq_len={seq_len}, embed_dim={embed_dim}, "
            f"window_size={window_size}, runs={NUM_RUNS}"
        )
        print("#" * 80)

        profiles = []

        for run_idx in range(1, NUM_RUNS + 1):
            profile = sliding_window_attention_profile(
                seq_len=seq_len,
                embed_dim=embed_dim,
                window_size=window_size,
                device=device,
            )
            print_profile(profile, run_idx)
            profiles.append(profile)

        stats = summarize_profiles(profiles)
        print_stats(stats)
        stats_list.append(stats)

    plot_results(stats_list, output_dir)
    save_csv(stats_list, output_dir)


def main() -> None:
    output_dir = "results/sliding_window_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    log_file = output_path / "run.log"

    with open(log_file, "w") as f:
        tee = Tee(sys.stdout, f)
        with redirect_stdout(tee):
            print(f"Output directory: {output_path.resolve()}")
            print(f"Sliding window size: {WINDOW_SIZE}")
            run_experiment(output_dir)

    print(f"Saved run log to: {log_file}")


if __name__ == "__main__":
    main()