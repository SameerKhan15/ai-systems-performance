#!/usr/bin/env python3
"""
compare_full_vs_sliding_attention.py

Lab#3 comparison script:
  Full quadratic self-attention vs sliding-window self-attention.

What this measures:
  - median runtime
  - p95 runtime
  - attention score entries
  - score tensor memory
  - peak GPU memory
  - plots
  - CSV
  - run.log

Nsight Systems support:
  The script adds NVTX ranges, so when you run it under `nsys profile`,
  the timeline will show labeled ranges such as:

    full/seq_len=4096/scores
    sliding_r2/seq_len=4096/scores
    full/seq_len=4096/softmax
    sliding_r2/seq_len=4096/softmax

Important naming:
  --window-radius 2 means:
      2 tokens left + current token + 2 tokens right = max 5 attended tokens

  For the original sentence example:
      The  -> [The, cat]
      cat  -> [The, cat, sat]
      sat  -> [cat, sat, down]
      down -> [sat, down]
  use:
      --window-radius 1
"""

import argparse
import csv
import math
import sys
import time
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, List

import torch
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


DEFAULT_NUM_RUNS = 30
DEFAULT_EMBED_DIM = 128
DEFAULT_SEQ_LENS = [64, 128, 256, 512, 1024, 2048, 4096, 8192]


@dataclass
class AttentionProfile:
    method: str
    seq_len: int
    embed_dim: int
    window_radius: int

    q_proj_ms: float
    k_proj_ms: float
    v_proj_ms: float
    scores_ms: float
    softmax_ms: float
    output_ms: float
    attention_total_ms: float
    total_ms: float

    score_entries: int
    valid_score_entries: int
    score_tensor_mb: float
    valid_score_math_mb: float
    qk_dot_scalar_work: int
    gpu_peak_mb: float


@dataclass
class AttentionStats:
    method: str
    seq_len: int
    embed_dim: int
    window_radius: int
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

    attention_total_median_ms: float
    attention_total_p95_ms: float

    total_median_ms: float
    total_p95_ms: float

    score_entries: int
    valid_score_entries: int
    score_tensor_mb: float
    valid_score_math_mb: float
    qk_dot_scalar_work: int

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


def parse_seq_lens(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def percentile(values: Iterable[float], pct: float) -> float:
    sorted_values = sorted(values)
    if not sorted_values:
        raise ValueError("Cannot compute percentile of empty values")

    k = (len(sorted_values) - 1) * (pct / 100.0)
    lower = math.floor(k)
    upper = math.ceil(k)

    if lower == upper:
        return sorted_values[int(k)]

    weight = k - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def median(values: Iterable[float]) -> float:
    return percentile(values, 50)


def p95(values: Iterable[float]) -> float:
    return percentile(values, 95)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@contextmanager
def nvtx_range(device: torch.device, name: str):
    if device.type == "cuda":
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if device.type == "cuda":
            torch.cuda.nvtx.range_pop()


def timed_step(device: torch.device, label: str, fn: Callable):
    sync_if_needed(device)
    with nvtx_range(device, label):
        start = time.perf_counter()
        result = fn()
        sync_if_needed(device)
        end = time.perf_counter()
    return result, (end - start) * 1000


def make_inputs(seq_len: int, embed_dim: int, device: torch.device):
    # The seed is reset per run to reduce data-dependent variation.
    # This is a learning benchmark, not a randomness benchmark.
    torch.manual_seed(42)

    X = torch.randn(seq_len, embed_dim, device=device)
    Wq = torch.randn(embed_dim, embed_dim, device=device)
    Wk = torch.randn(embed_dim, embed_dim, device=device)
    Wv = torch.randn(embed_dim, embed_dim, device=device)

    return X, Wq, Wk, Wv


def build_sliding_indices(seq_len: int, window_radius: int, device: torch.device):
    positions = torch.arange(seq_len, device=device).unsqueeze(1)
    offsets = torch.arange(-window_radius, window_radius + 1, device=device).unsqueeze(0)

    raw_indices = positions + offsets
    valid_mask = (raw_indices >= 0) & (raw_indices < seq_len)
    safe_indices = raw_indices.clamp(0, seq_len - 1)

    return safe_indices, valid_mask


def compute_sliding_scores(
    Q: torch.Tensor,
    K: torch.Tensor,
    safe_indices: torch.Tensor,
    valid_mask: torch.Tensor,
    embed_dim: int,
):
    # K_windows shape:
    #   [N, 2 * window_radius + 1, D]
    K_windows = K[safe_indices]

    # scores shape:
    #   [N, 2 * window_radius + 1]
    scores = (Q.unsqueeze(1) * K_windows).sum(dim=-1) / math.sqrt(embed_dim)

    # Invalid edge positions become -inf so softmax assigns them zero probability.
    scores = scores.masked_fill(~valid_mask, float("-inf"))
    return scores


def compute_sliding_output(
    weights: torch.Tensor,
    V: torch.Tensor,
    safe_indices: torch.Tensor,
):
    # V_windows shape:
    #   [N, 2 * window_radius + 1, D]
    V_windows = V[safe_indices]

    # output shape:
    #   [N, D]
    output = (weights.unsqueeze(-1) * V_windows).sum(dim=1)
    return output


@torch.no_grad()
def full_attention_profile(
    seq_len: int,
    embed_dim: int,
    device: torch.device,
) -> AttentionProfile:
    method = "full"

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    X, Wq, Wk, Wv = make_inputs(seq_len, embed_dim, device)

    Q, q_proj_ms = timed_step(device, f"{method}/seq_len={seq_len}/q_proj", lambda: X @ Wq)
    K, k_proj_ms = timed_step(device, f"{method}/seq_len={seq_len}/k_proj", lambda: X @ Wk)
    V, v_proj_ms = timed_step(device, f"{method}/seq_len={seq_len}/v_proj", lambda: X @ Wv)

    scores, scores_ms = timed_step(
        device,
        f"{method}/seq_len={seq_len}/scores_NxN",
        lambda: (Q @ K.T) / math.sqrt(embed_dim),
    )

    weights, softmax_ms = timed_step(
        device,
        f"{method}/seq_len={seq_len}/softmax_NxN",
        lambda: torch.softmax(scores, dim=-1),
    )

    output, output_ms = timed_step(
        device,
        f"{method}/seq_len={seq_len}/weights_at_V_NxN",
        lambda: weights @ V,
    )

    sync_if_needed(device)
    _ = output.sum().item()

    attention_total_ms = scores_ms + softmax_ms + output_ms
    total_ms = q_proj_ms + k_proj_ms + v_proj_ms + attention_total_ms

    score_entries = seq_len * seq_len
    valid_score_entries = score_entries
    score_tensor_mb = (score_entries * scores.element_size()) / (1024 ** 2)
    valid_score_math_mb = score_tensor_mb
    qk_dot_scalar_work = valid_score_entries * embed_dim

    if device.type == "cuda":
        gpu_peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        gpu_peak_mb = 0.0

    return AttentionProfile(
        method=method,
        seq_len=seq_len,
        embed_dim=embed_dim,
        window_radius=-1,
        q_proj_ms=q_proj_ms,
        k_proj_ms=k_proj_ms,
        v_proj_ms=v_proj_ms,
        scores_ms=scores_ms,
        softmax_ms=softmax_ms,
        output_ms=output_ms,
        attention_total_ms=attention_total_ms,
        total_ms=total_ms,
        score_entries=score_entries,
        valid_score_entries=valid_score_entries,
        score_tensor_mb=score_tensor_mb,
        valid_score_math_mb=valid_score_math_mb,
        qk_dot_scalar_work=qk_dot_scalar_work,
        gpu_peak_mb=gpu_peak_mb,
    )


@torch.no_grad()
def sliding_attention_profile(
    seq_len: int,
    embed_dim: int,
    window_radius: int,
    device: torch.device,
) -> AttentionProfile:
    method = f"sliding_r{window_radius}"

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    X, Wq, Wk, Wv = make_inputs(seq_len, embed_dim, device)

    Q, q_proj_ms = timed_step(device, f"{method}/seq_len={seq_len}/q_proj", lambda: X @ Wq)
    K, k_proj_ms = timed_step(device, f"{method}/seq_len={seq_len}/k_proj", lambda: X @ Wk)
    V, v_proj_ms = timed_step(device, f"{method}/seq_len={seq_len}/v_proj", lambda: X @ Wv)

    safe_indices, valid_mask = build_sliding_indices(seq_len, window_radius, device)
    valid_score_entries = int(valid_mask.sum().item())

    scores, scores_ms = timed_step(
        device,
        f"{method}/seq_len={seq_len}/scores_Nx{2 * window_radius + 1}",
        lambda: compute_sliding_scores(Q, K, safe_indices, valid_mask, embed_dim),
    )

    weights, softmax_ms = timed_step(
        device,
        f"{method}/seq_len={seq_len}/softmax_Nx{2 * window_radius + 1}",
        lambda: torch.softmax(scores, dim=-1),
    )

    output, output_ms = timed_step(
        device,
        f"{method}/seq_len={seq_len}/weights_at_V_Nx{2 * window_radius + 1}",
        lambda: compute_sliding_output(weights, V, safe_indices),
    )

    sync_if_needed(device)
    _ = output.sum().item()

    attention_total_ms = scores_ms + softmax_ms + output_ms
    total_ms = q_proj_ms + k_proj_ms + v_proj_ms + attention_total_ms

    score_entries = seq_len * (2 * window_radius + 1)
    score_tensor_mb = (score_entries * scores.element_size()) / (1024 ** 2)
    valid_score_math_mb = (valid_score_entries * scores.element_size()) / (1024 ** 2)
    qk_dot_scalar_work = valid_score_entries * embed_dim

    if device.type == "cuda":
        gpu_peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        gpu_peak_mb = 0.0

    return AttentionProfile(
        method=method,
        seq_len=seq_len,
        embed_dim=embed_dim,
        window_radius=window_radius,
        q_proj_ms=q_proj_ms,
        k_proj_ms=k_proj_ms,
        v_proj_ms=v_proj_ms,
        scores_ms=scores_ms,
        softmax_ms=softmax_ms,
        output_ms=output_ms,
        attention_total_ms=attention_total_ms,
        total_ms=total_ms,
        score_entries=score_entries,
        valid_score_entries=valid_score_entries,
        score_tensor_mb=score_tensor_mb,
        valid_score_math_mb=valid_score_math_mb,
        qk_dot_scalar_work=qk_dot_scalar_work,
        gpu_peak_mb=gpu_peak_mb,
    )


def summarize_profiles(profiles: List[AttentionProfile]) -> AttentionStats:
    first = profiles[0]

    return AttentionStats(
        method=first.method,
        seq_len=first.seq_len,
        embed_dim=first.embed_dim,
        window_radius=first.window_radius,
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

        attention_total_median_ms=median([p.attention_total_ms for p in profiles]),
        attention_total_p95_ms=p95([p.attention_total_ms for p in profiles]),

        total_median_ms=median([p.total_ms for p in profiles]),
        total_p95_ms=p95([p.total_ms for p in profiles]),

        score_entries=first.score_entries,
        valid_score_entries=first.valid_score_entries,
        score_tensor_mb=first.score_tensor_mb,
        valid_score_math_mb=first.valid_score_math_mb,
        qk_dot_scalar_work=first.qk_dot_scalar_work,

        gpu_peak_median_mb=median([p.gpu_peak_mb for p in profiles]),
        gpu_peak_p95_mb=p95([p.gpu_peak_mb for p in profiles]),
    )


def print_profile(profile: AttentionProfile, run_idx: int) -> None:
    print(f"Run {run_idx} | method={profile.method} | seq_len={profile.seq_len}")
    print(f"Q projection              : {profile.q_proj_ms:.3f} ms")
    print(f"K projection              : {profile.k_proj_ms:.3f} ms")
    print(f"V projection              : {profile.v_proj_ms:.3f} ms")
    print(f"scores                    : {profile.scores_ms:.3f} ms")
    print(f"softmax                   : {profile.softmax_ms:.3f} ms")
    print(f"weights @ V               : {profile.output_ms:.3f} ms")
    print(f"attention total           : {profile.attention_total_ms:.3f} ms")
    print(f"total                     : {profile.total_ms:.3f} ms")
    print(f"score tensor entries      : {profile.score_entries:,}")
    print(f"valid score entries       : {profile.valid_score_entries:,}")
    print(f"QK scalar work approx     : {profile.qk_dot_scalar_work:,}")
    print(f"score tensor memory       : {profile.score_tensor_mb:.4f} MB")
    print(f"peak GPU memory           : {profile.gpu_peak_mb:.2f} MB")
    print("-" * 80)


def print_stats(stats: AttentionStats) -> None:
    print("=" * 100)
    print(f"SUMMARY method            : {stats.method}")
    print(f"seq_len                   : {stats.seq_len}")
    print(f"embed_dim                 : {stats.embed_dim}")
    print(f"window_radius             : {stats.window_radius}")
    print(f"runs                      : {stats.runs}")
    print()
    print("Median")
    print(f"Q projection              : {stats.q_proj_median_ms:.3f} ms")
    print(f"K projection              : {stats.k_proj_median_ms:.3f} ms")
    print(f"V projection              : {stats.v_proj_median_ms:.3f} ms")
    print(f"scores                    : {stats.scores_median_ms:.3f} ms")
    print(f"softmax                   : {stats.softmax_median_ms:.3f} ms")
    print(f"weights @ V               : {stats.output_median_ms:.3f} ms")
    print(f"attention total           : {stats.attention_total_median_ms:.3f} ms")
    print(f"total                     : {stats.total_median_ms:.3f} ms")
    print()
    print("P95")
    print(f"scores                    : {stats.scores_p95_ms:.3f} ms")
    print(f"softmax                   : {stats.softmax_p95_ms:.3f} ms")
    print(f"weights @ V               : {stats.output_p95_ms:.3f} ms")
    print(f"attention total           : {stats.attention_total_p95_ms:.3f} ms")
    print(f"total                     : {stats.total_p95_ms:.3f} ms")
    print()
    print(f"score tensor entries      : {stats.score_entries:,}")
    print(f"valid score entries       : {stats.valid_score_entries:,}")
    print(f"QK scalar work approx     : {stats.qk_dot_scalar_work:,}")
    print(f"score tensor memory       : {stats.score_tensor_mb:.4f} MB")
    print(f"valid score math memory   : {stats.valid_score_math_mb:.4f} MB")
    print(f"peak GPU memory median    : {stats.gpu_peak_median_mb:.2f} MB")
    print(f"peak GPU memory p95       : {stats.gpu_peak_p95_mb:.2f} MB")
    print("=" * 100)


def save_csv(stats_list: List[AttentionStats], output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_file = output_path / "comparison_metrics.csv"

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "method",
            "seq_len",
            "embed_dim",
            "window_radius",
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

            "attention_total_median_ms",
            "attention_total_p95_ms",

            "total_median_ms",
            "total_p95_ms",

            "score_entries",
            "valid_score_entries",
            "score_tensor_mb",
            "valid_score_math_mb",
            "qk_dot_scalar_work",

            "gpu_peak_median_mb",
            "gpu_peak_p95_mb",
        ])

        for s in stats_list:
            writer.writerow([
                s.method,
                s.seq_len,
                s.embed_dim,
                s.window_radius,
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

                s.attention_total_median_ms,
                s.attention_total_p95_ms,

                s.total_median_ms,
                s.total_p95_ms,

                s.score_entries,
                s.valid_score_entries,
                s.score_tensor_mb,
                s.valid_score_math_mb,
                s.qk_dot_scalar_work,

                s.gpu_peak_median_mb,
                s.gpu_peak_p95_mb,
            ])

    print(f"Saved comparison metrics to: {csv_file}")


def split_stats(stats_list: List[AttentionStats]):
    full = sorted([s for s in stats_list if s.method == "full"], key=lambda s: s.seq_len)
    sliding = sorted([s for s in stats_list if s.method.startswith("sliding")], key=lambda s: s.seq_len)
    return full, sliding


def plot_results(stats_list: List[AttentionStats], output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    full, sliding = split_stats(stats_list)

    if not full or not sliding:
        print("Skipping plots because both full and sliding results are required.")
        return

    seq_lens = [s.seq_len for s in full]

    plt.figure(figsize=(10, 6))
    plt.plot(seq_lens, [s.total_median_ms for s in full], marker="o", label="Full total median")
    plt.plot(seq_lens, [s.total_p95_ms for s in full], marker="o", label="Full total p95")
    plt.plot(seq_lens, [s.total_median_ms for s in sliding], marker="o", label="Sliding total median")
    plt.plot(seq_lens, [s.total_p95_ms for s in sliding], marker="o", label="Sliding total p95")
    plt.xlabel("Sequence length")
    plt.ylabel("Runtime (ms)")
    plt.title("Full vs Sliding-Window Attention: Total Runtime")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path / "comparison_total_runtime.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(seq_lens, [s.attention_total_median_ms for s in full], marker="o", label="Full attention total median")
    plt.plot(seq_lens, [s.attention_total_p95_ms for s in full], marker="o", label="Full attention total p95")
    plt.plot(seq_lens, [s.attention_total_median_ms for s in sliding], marker="o", label="Sliding attention total median")
    plt.plot(seq_lens, [s.attention_total_p95_ms for s in sliding], marker="o", label="Sliding attention total p95")
    plt.xlabel("Sequence length")
    plt.ylabel("Runtime (ms)")
    plt.title("Full vs Sliding-Window Attention: Attention-Only Runtime")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path / "comparison_attention_only_runtime.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(seq_lens, [s.scores_median_ms for s in full], marker="o", label="Full scores median")
    plt.plot(seq_lens, [s.output_median_ms for s in full], marker="o", label="Full weights @ V median")
    plt.plot(seq_lens, [s.scores_median_ms for s in sliding], marker="o", label="Sliding scores median")
    plt.plot(seq_lens, [s.output_median_ms for s in sliding], marker="o", label="Sliding weights @ V median")
    plt.xlabel("Sequence length")
    plt.ylabel("Runtime (ms)")
    plt.title("Full vs Sliding-Window Attention: Dominant Attention Costs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path / "comparison_dominant_attention_costs.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(seq_lens, [s.score_tensor_mb for s in full], marker="o", label="Full score tensor MB")
    plt.plot(seq_lens, [s.score_tensor_mb for s in sliding], marker="o", label="Sliding score tensor MB")
    plt.plot(seq_lens, [s.gpu_peak_median_mb for s in full], marker="o", label="Full peak GPU MB median")
    plt.plot(seq_lens, [s.gpu_peak_median_mb for s in sliding], marker="o", label="Sliding peak GPU MB median")
    plt.xlabel("Sequence length")
    plt.ylabel("Memory (MB)")
    plt.title("Full vs Sliding-Window Attention: Memory Scaling")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path / "comparison_memory_scaling_log.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(seq_lens, [s.valid_score_entries for s in full], marker="o", label="Full valid score entries")
    plt.plot(seq_lens, [s.valid_score_entries for s in sliding], marker="o", label="Sliding valid score entries")
    plt.xlabel("Sequence length")
    plt.ylabel("Valid attention score entries")
    plt.title("Full vs Sliding-Window Attention: N² vs N×Window Score Entries")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path / "comparison_score_entries_log.png", dpi=150)
    plt.close()

    print(f"Saved comparison plots to: {output_path.resolve()}")


def print_comparison_table(stats_list: List[AttentionStats]) -> None:
    full, sliding = split_stats(stats_list)
    by_full = {s.seq_len: s for s in full}
    by_sliding = {s.seq_len: s for s in sliding}

    print()
    print("#" * 100)
    print("FULL VS SLIDING COMPARISON")
    print("#" * 100)
    print(
        f"{'N':>8} | "
        f"{'full attn ms':>13} | "
        f"{'slide attn ms':>14} | "
        f"{'attn speedup':>13} | "
        f"{'full entries':>14} | "
        f"{'slide entries':>14} | "
        f"{'entry ratio':>11} | "
        f"{'full score MB':>13} | "
        f"{'slide score MB':>14}"
    )
    print("-" * 130)

    for seq_len in sorted(set(by_full) & set(by_sliding)):
        f = by_full[seq_len]
        s = by_sliding[seq_len]

        attn_speedup = (
            f.attention_total_median_ms / s.attention_total_median_ms
            if s.attention_total_median_ms > 0
            else float("inf")
        )

        entry_ratio = (
            f.valid_score_entries / s.valid_score_entries
            if s.valid_score_entries > 0
            else float("inf")
        )

        print(
            f"{seq_len:8d} | "
            f"{f.attention_total_median_ms:13.3f} | "
            f"{s.attention_total_median_ms:14.3f} | "
            f"{attn_speedup:13.2f} | "
            f"{f.valid_score_entries:14,d} | "
            f"{s.valid_score_entries:14,d} | "
            f"{entry_ratio:11.2f} | "
            f"{f.score_tensor_mb:13.2f} | "
            f"{s.score_tensor_mb:14.4f}"
        )

    print("#" * 100)


def run_experiment(args) -> None:
    device = pick_device()

    print(f"Using device              : {device}")
    print(f"embed_dim                 : {args.embed_dim}")
    print(f"num_runs                  : {args.num_runs}")
    print(f"seq_lens                  : {args.seq_lens}")
    print(f"sliding window radius     : {args.window_radius}")
    print(f"sliding max attended toks : {2 * args.window_radius + 1}")
    print()

    if device.type == "cuda":
        print(f"GPU Name                  : {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version              : {torch.version.cuda}")
        print(f"PyTorch Version           : {torch.__version__}")

    stats_list: List[AttentionStats] = []

    print()
    print("Warmup...")
    for _ in range(args.warmup_runs):
        _ = full_attention_profile(512, args.embed_dim, device)
        _ = sliding_attention_profile(512, args.embed_dim, args.window_radius, device)

    for seq_len in args.seq_lens:
        print()
        print("#" * 100)
        print(f"Running FULL attention | seq_len={seq_len}, embed_dim={args.embed_dim}, runs={args.num_runs}")
        print("#" * 100)

        full_profiles = []
        for run_idx in range(1, args.num_runs + 1):
            profile = full_attention_profile(seq_len, args.embed_dim, device)
            print_profile(profile, run_idx)
            full_profiles.append(profile)

        full_stats = summarize_profiles(full_profiles)
        print_stats(full_stats)
        stats_list.append(full_stats)

        print()
        print("#" * 100)
        print(
            f"Running SLIDING attention | seq_len={seq_len}, "
            f"embed_dim={args.embed_dim}, window_radius={args.window_radius}, "
            f"runs={args.num_runs}"
        )
        print("#" * 100)

        sliding_profiles = []
        for run_idx in range(1, args.num_runs + 1):
            profile = sliding_attention_profile(seq_len, args.embed_dim, args.window_radius, device)
            print_profile(profile, run_idx)
            sliding_profiles.append(profile)

        sliding_stats = summarize_profiles(sliding_profiles)
        print_stats(sliding_stats)
        stats_list.append(sliding_stats)

    print_comparison_table(stats_list)
    plot_results(stats_list, args.output_dir)
    save_csv(stats_list, args.output_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare quadratic full attention against sliding-window attention."
    )

    parser.add_argument(
        "--seq-lens",
        type=parse_seq_lens,
        default=DEFAULT_SEQ_LENS,
        help="Comma-separated sequence lengths. Default: 64,128,256,512,1024,2048,4096,8192",
    )

    parser.add_argument(
        "--embed-dim",
        type=int,
        default=DEFAULT_EMBED_DIM,
        help=f"Embedding dimension. Default: {DEFAULT_EMBED_DIM}",
    )

    parser.add_argument(
        "--window-radius",
        type=int,
        default=2,
        help=(
            "Sliding attention radius. "
            "2 means two tokens left + self + two tokens right = max 5 tokens. "
            "Use 1 for the original sentence example."
        ),
    )

    parser.add_argument(
        "--num-runs",
        type=int,
        default=DEFAULT_NUM_RUNS,
        help=f"Runs per method per sequence length. Default: {DEFAULT_NUM_RUNS}",
    )

    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=5,
        help="Warmup runs before benchmark. Default: 5",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output directory. Default: results/compare_full_vs_sliding_<timestamp>",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.window_radius < 0:
        raise ValueError("--window-radius must be >= 0")

    if not args.output_dir:
        args.output_dir = "results/compare_full_vs_sliding_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    log_file = output_path / "run.log"

    with open(log_file, "w") as f:
        tee = Tee(sys.stdout, f)
        with redirect_stdout(tee):
            print(f"Output directory: {output_path.resolve()}")
            run_experiment(args)

    print(f"Saved run log to: {log_file}")


if __name__ == "__main__":
    main()