#!/usr/bin/env python3
"""
Prefill Latency vs. Prompt Length
=================================

This lab measures how the latency of one attention-only transformer block
changes as the input prompt becomes longer.

The model intentionally contains only:
  - token embedding
  - Q, K, V projections
  - causal self-attention
  - output projection

It intentionally omits:
  - feed-forward network
  - layer normalization
  - residual connections
  - positional encoding
  - logits/language-model head

The purpose is to isolate prefill attention and connect measured latency to:

    projection MACs = 4 * B * T * D^2
    attention MACs  = 2 * B * T^2 * D

where:
    B = batch size
    T = prompt length
    D = embedding dimension

The script saves aggregated PNG plots only. It prints median and p95 results,
but does not save individual timing samples.

Example — small CPU run:
    python prefill_latency_lab.py \
        --device cpu \
        --embed-dim 128 \
        --num-heads 4 \
        --prompt-lengths 32,64,128,256 \
        --runs 30

Example — A100-style run:
    python prefill_latency_lab.py \
        --device cuda \
        --dtype float16 \
        --embed-dim 512 \
        --num-heads 8 \
        --prompt-lengths 128,256,512,1024,2048 \
        --runs 30
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Configuration and theoretical accounting
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class LabConfig:
    embed_dim: int
    num_heads: int
    batch_size: int
    vocab_size: int
    dtype: torch.dtype
    device: str
    attention_backend: str

    @property
    def head_dim(self) -> int:
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        return self.embed_dim // self.num_heads

    @property
    def dtype_bytes(self) -> int:
        return torch.empty((), dtype=self.dtype).element_size()


@dataclass(frozen=True)
class PrefillAccounting:
    prompt_length: int
    projection_macs: int
    attention_score_macs: int
    attention_value_macs: int
    total_macs: int
    logical_score_elements: int
    logical_score_bytes: int
    qkv_bytes: int


@dataclass(frozen=True)
class BenchmarkResult:
    prompt_length: int
    median_ms: float
    p95_ms: float
    median_prompt_tokens_per_second: float
    accounting: PrefillAccounting


def calculate_prefill_accounting(
    *,
    batch_size: int,
    prompt_length: int,
    embed_dim: int,
    num_heads: int,
    dtype_bytes: int,
) -> PrefillAccounting:
    """Calculate theoretical MAC and logical tensor-size counts for one layer.

    Projection assumptions:
        Wq, Wk, Wv, Wo each have shape [D, D].

    Attention assumptions:
        num_heads * head_dim = D.

    Note that ``logical_score_bytes`` describes the full [B, H, T, T] score
    matrix. The manual backend materializes it. An optimized SDPA backend may
    avoid materializing the entire matrix even though the logical attention
    relationship remains the same.
    """
    if embed_dim % num_heads != 0:
        raise ValueError("embed_dim must be divisible by num_heads")

    head_dim = embed_dim // num_heads

    # Four square projections: Q, K, V, and O.
    projection_macs = 4 * batch_size * prompt_length * embed_dim * embed_dim

    # Q @ K^T and attention_weights @ V have the same MAC count.
    attention_score_macs = (
        batch_size * num_heads * prompt_length * prompt_length * head_dim
    )
    attention_value_macs = attention_score_macs
    total_macs = projection_macs + attention_score_macs + attention_value_macs

    logical_score_elements = batch_size * num_heads * prompt_length * prompt_length
    logical_score_bytes = logical_score_elements * dtype_bytes
    qkv_bytes = 3 * batch_size * prompt_length * embed_dim * dtype_bytes

    return PrefillAccounting(
        prompt_length=prompt_length,
        projection_macs=projection_macs,
        attention_score_macs=attention_score_macs,
        attention_value_macs=attention_value_macs,
        total_macs=total_macs,
        logical_score_elements=logical_score_elements,
        logical_score_bytes=logical_score_bytes,
        qkv_bytes=qkv_bytes,
    )


# -----------------------------------------------------------------------------
# Minimal attention-only prefill model
# -----------------------------------------------------------------------------


class PrefillAttentionBlock(nn.Module):
    """One attention-only transformer block used for prefill measurement."""

    def __init__(self, cfg: LabConfig, seed: int = 7) -> None:
        super().__init__()
        self.cfg = cfg

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.wq = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=False)
        self.wk = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=False)
        self.wv = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=False)
        self.wo = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=False)

        for parameter in self.parameters():
            nn.init.normal_(parameter, mean=0.0, std=0.02)

        self.to(device=cfg.device, dtype=cfg.dtype)
        self.eval()

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, D] -> [B, H, T, Hd]
        batch_size, prompt_length, embed_dim = x.shape
        return x.view(
            batch_size,
            prompt_length,
            self.cfg.num_heads,
            self.cfg.head_dim,
        ).transpose(1, 2)

    @staticmethod
    def merge_heads(x: torch.Tensor) -> torch.Tensor:
        # [B, H, T, Hd] -> [B, T, D]
        batch_size, num_heads, prompt_length, head_dim = x.shape
        return (
            x.transpose(1, 2)
            .contiguous()
            .view(batch_size, prompt_length, num_heads * head_dim)
        )

    def manual_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.cfg.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = scores.masked_fill(causal_mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, v)

    def forward(
        self,
        token_ids: torch.Tensor,
        causal_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # The entire prompt is supplied in one operation. That is prefill.
        x = self.embedding(token_ids)  # [B, T, D]

        q = self.split_heads(self.wq(x))
        k = self.split_heads(self.wk(x))
        v = self.split_heads(self.wv(x))

        if self.cfg.attention_backend == "manual":
            if causal_mask is None:
                raise ValueError("manual attention requires a causal mask")
            attention_output = self.manual_attention(q, k, v, causal_mask)
        elif self.cfg.attention_backend == "sdpa":
            # PyTorch can dispatch this to an optimized kernel. Such a kernel
            # may avoid materializing the complete score matrix.
            attention_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
            )
        else:
            raise ValueError(
                f"Unknown attention backend: {self.cfg.attention_backend}"
            )

        merged = self.merge_heads(attention_output)
        return self.wo(merged)


# -----------------------------------------------------------------------------
# Benchmark helpers
# -----------------------------------------------------------------------------


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def nearest_rank_percentile(values: Sequence[float], percentile: float) -> float:
    """Return the nearest-rank percentile for a non-empty sequence."""
    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 < percentile <= 100.0:
        raise ValueError("percentile must be in (0, 100]")

    ordered = sorted(values)
    rank = math.ceil((percentile / 100.0) * len(ordered))
    return ordered[rank - 1]


def make_token_ids(
    *,
    batch_size: int,
    prompt_length: int,
    vocab_size: int,
    device: str,
    seed: int,
) -> torch.Tensor:
    generator_device = device if device == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed + prompt_length)

    token_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, prompt_length),
        dtype=torch.long,
        generator=generator,
        device=generator_device,
    )

    # MPS does not support a device-specific torch.Generator in all versions,
    # so create on CPU and then transfer when needed.
    if token_ids.device.type != device:
        token_ids = token_ids.to(device)
    return token_ids


def make_causal_mask(prompt_length: int, device: str) -> torch.Tensor:
    # Shape broadcasts across batch and heads: [1, 1, T, T].
    return torch.ones(
        (1, 1, prompt_length, prompt_length),
        dtype=torch.bool,
        device=device,
    ).triu(diagonal=1)


@torch.inference_mode()
def benchmark_prompt_length(
    *,
    model: PrefillAttentionBlock,
    token_ids: torch.Tensor,
    causal_mask: torch.Tensor | None,
    warmups: int,
    runs: int,
) -> Tuple[float, float]:
    """Return median and p95 prefill latency in milliseconds."""
    if warmups < 0:
        raise ValueError("warmups must be non-negative")
    if runs <= 0:
        raise ValueError("runs must be positive")

    for _ in range(warmups):
        output = model(token_ids, causal_mask)
        # Force a cheap use of the output so the call is observably consumed.
        _ = output[:, -1, :]
    synchronize(model.cfg.device)

    samples_ms: List[float] = []

    if model.cfg.device == "cuda":
        for _ in range(runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            output = model(token_ids, causal_mask)
            _ = output[:, -1, :]
            end_event.record()
            end_event.synchronize()

            samples_ms.append(float(start_event.elapsed_time(end_event)))
    else:
        for _ in range(runs):
            synchronize(model.cfg.device)
            start = time.perf_counter_ns()
            output = model(token_ids, causal_mask)
            _ = output[:, -1, :]
            synchronize(model.cfg.device)
            elapsed_ns = time.perf_counter_ns() - start
            samples_ms.append(elapsed_ns / 1_000_000.0)

    return statistics.median(samples_ms), nearest_rank_percentile(samples_ms, 95.0)


def fit_power_law(
    prompt_lengths: Sequence[int],
    median_latencies_ms: Sequence[float],
) -> Tuple[float, float, float]:
    """Fit latency ~= coefficient * prompt_length^alpha in log space.

    Returns:
        alpha, coefficient, r_squared
    """
    if len(prompt_lengths) != len(median_latencies_ms):
        raise ValueError("x and y must have the same length")
    if len(prompt_lengths) < 2:
        raise ValueError("at least two observations are required")
    if any(x <= 0 for x in prompt_lengths):
        raise ValueError("prompt lengths must be positive")
    if any(y <= 0 for y in median_latencies_ms):
        raise ValueError("latencies must be positive")

    x = [math.log(float(value)) for value in prompt_lengths]
    y = [math.log(float(value)) for value in median_latencies_ms]

    x_mean = statistics.fmean(x)
    y_mean = statistics.fmean(y)
    covariance = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    variance_x = sum((xi - x_mean) ** 2 for xi in x)

    if variance_x == 0:
        raise ValueError("prompt lengths must not all be equal")

    alpha = covariance / variance_x
    intercept = y_mean - alpha * x_mean
    coefficient = math.exp(intercept)

    predicted = [intercept + alpha * xi for xi in x]
    total_sum_squares = sum((yi - y_mean) ** 2 for yi in y)
    residual_sum_squares = sum((yi - pi) ** 2 for yi, pi in zip(y, predicted))
    r_squared = (
        1.0 - residual_sum_squares / total_sum_squares
        if total_sum_squares > 0
        else 1.0
    )

    return alpha, coefficient, r_squared


# -----------------------------------------------------------------------------
# Reporting and plotting
# -----------------------------------------------------------------------------


def bytes_to_human(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def macs_to_human(macs: int) -> str:
    if macs < 1_000:
        return f"{macs}"
    if macs < 1_000_000:
        return f"{macs / 1_000:.2f} K"
    if macs < 1_000_000_000:
        return f"{macs / 1_000_000:.2f} M"
    if macs < 1_000_000_000_000:
        return f"{macs / 1_000_000_000:.2f} B"
    return f"{macs / 1_000_000_000_000:.2f} T"


def print_configuration(
    cfg: LabConfig,
    prompt_lengths: Sequence[int],
    warmups: int,
    runs: int,
) -> None:
    print("\n================ Prefill Lab Configuration ================")
    print(f"device             : {cfg.device}")
    print(f"dtype              : {str(cfg.dtype).replace('torch.', '')}")
    print(f"attention backend  : {cfg.attention_backend}")
    print(f"batch size         : {cfg.batch_size}")
    print(f"embed_dim          : {cfg.embed_dim}")
    print(f"num_heads          : {cfg.num_heads}")
    print(f"head_dim           : {cfg.head_dim}")
    print(f"prompt lengths     : {list(prompt_lengths)}")
    print(f"warmups per length : {warmups}")
    print(f"timed runs/length  : {runs}")
    print("===========================================================\n")


def print_results(results: Sequence[BenchmarkResult]) -> None:
    print("====================== Aggregated Results ======================")
    header = (
        f"{'T':>7} "
        f"{'median ms':>12} "
        f"{'p95 ms':>12} "
        f"{'prompt tok/s':>15} "
        f"{'total MACs':>14} "
        f"{'score bytes':>14}"
    )
    print(header)
    print("-" * len(header))

    for result in results:
        print(
            f"{result.prompt_length:7,d} "
            f"{result.median_ms:12.4f} "
            f"{result.p95_ms:12.4f} "
            f"{result.median_prompt_tokens_per_second:15,.1f} "
            f"{macs_to_human(result.accounting.total_macs):>14} "
            f"{bytes_to_human(result.accounting.logical_score_bytes):>14}"
        )

    print("================================================================\n")


def plot_latency(results: Sequence[BenchmarkResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_lengths = [result.prompt_length for result in results]
    medians = [result.median_ms for result in results]
    p95s = [result.p95_ms for result in results]

    plt.figure(figsize=(9, 5.5))
    plt.plot(prompt_lengths, medians, marker="o", label="Median prefill latency")
    plt.plot(prompt_lengths, p95s, marker="o", label="P95 prefill latency")
    plt.xlabel("Prompt length, tokens")
    plt.ylabel("Latency, milliseconds")
    plt.title("Prefill Latency vs. Prompt Length")
    plt.xticks(prompt_lengths)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_throughput(results: Sequence[BenchmarkResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_lengths = [result.prompt_length for result in results]
    throughputs = [result.median_prompt_tokens_per_second for result in results]

    plt.figure(figsize=(9, 5.5))
    plt.plot(prompt_lengths, throughputs, marker="o")
    plt.xlabel("Prompt length, tokens")
    plt.ylabel("Prompt tokens processed per second")
    plt.title("Prefill Throughput vs. Prompt Length")
    plt.xticks(prompt_lengths)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_log_log_fit(
    results: Sequence[BenchmarkResult],
    alpha: float,
    coefficient: float,
    r_squared: float,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_lengths = [result.prompt_length for result in results]
    medians = [result.median_ms for result in results]
    fitted = [coefficient * (length**alpha) for length in prompt_lengths]

    plt.figure(figsize=(9, 5.5))
    plt.loglog(prompt_lengths, medians, marker="o", label="Measured median")
    plt.loglog(
        prompt_lengths,
        fitted,
        linestyle="--",
        label=f"Fit: latency ∝ T^{alpha:.2f}, R²={r_squared:.3f}",
    )
    plt.xlabel("Prompt length, tokens — logarithmic scale")
    plt.ylabel("Median latency, ms — logarithmic scale")
    plt.title("Empirical Prefill Scaling")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


# -----------------------------------------------------------------------------
# CLI parsing and main experiment
# -----------------------------------------------------------------------------


def parse_prompt_lengths(raw_value: str) -> List[int]:
    try:
        values = [int(item.strip()) for item in raw_value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "prompt lengths must be comma-separated integers"
        ) from exc

    if len(values) < 2:
        raise argparse.ArgumentTypeError("provide at least two prompt lengths")
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("prompt lengths must be positive")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("prompt lengths must not contain duplicates")
    if values != sorted(values):
        raise argparse.ArgumentTypeError("prompt lengths must be increasing")
    return values


def resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but torch.backends.mps.is_available() is False")
    return requested


def resolve_dtype(requested: str, device: str) -> torch.dtype:
    if requested == "auto":
        return torch.float16 if device == "cuda" else torch.float32

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    dtype = mapping[requested]

    if device == "cpu" and dtype == torch.float16:
        raise ValueError(
            "float16 CPU matrix multiplication is often unsupported or very slow; "
            "use float32/bfloat16 or run on CUDA"
        )
    return dtype


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure prefill latency as prompt length increases."
    )
    parser.add_argument(
        "--prompt-lengths",
        type=parse_prompt_lengths,
        default=parse_prompt_lengths("64,128,256,512,1024"),
        help="increasing comma-separated prompt lengths",
    )
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32", "float64"],
        default="auto",
    )
    parser.add_argument(
        "--attention-backend",
        choices=["manual", "sdpa"],
        default="manual",
        help="manual materializes scores; sdpa may use optimized fused kernels",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="CPU-only PyTorch thread count",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs"),
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.embed_dim <= 0:
        raise ValueError("embed_dim must be positive")
    if args.num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if args.embed_dim % args.num_heads != 0:
        raise ValueError("embed_dim must be divisible by num_heads")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.vocab_size <= 1:
        raise ValueError("vocab_size must be greater than 1")
    if args.warmups < 0:
        raise ValueError("warmups must be non-negative")
    if args.runs <= 0:
        raise ValueError("runs must be positive")
    if args.torch_threads <= 0:
        raise ValueError("torch_threads must be positive")


def run_experiment(args: argparse.Namespace) -> List[BenchmarkResult]:
    validate_args(args)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    if device == "cpu":
        torch.set_num_threads(args.torch_threads)

    cfg = LabConfig(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        dtype=dtype,
        device=device,
        attention_backend=args.attention_backend,
    )

    model = PrefillAttentionBlock(cfg, seed=args.seed)
    print_configuration(cfg, args.prompt_lengths, args.warmups, args.runs)

    results: List[BenchmarkResult] = []

    for prompt_length in args.prompt_lengths:
        token_ids = make_token_ids(
            batch_size=cfg.batch_size,
            prompt_length=prompt_length,
            vocab_size=cfg.vocab_size,
            device=cfg.device,
            seed=args.seed,
        )
        causal_mask = (
            make_causal_mask(prompt_length, cfg.device)
            if cfg.attention_backend == "manual"
            else None
        )

        try:
            median_ms, p95_ms = benchmark_prompt_length(
                model=model,
                token_ids=token_ids,
                causal_mask=causal_mask,
                warmups=args.warmups,
                runs=args.runs,
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise RuntimeError(
                    f"Out of memory at prompt length {prompt_length}. "
                    "Reduce --prompt-lengths, --embed-dim, or --batch-size; "
                    "or use --attention-backend sdpa."
                ) from exc
            raise

        accounting = calculate_prefill_accounting(
            batch_size=cfg.batch_size,
            prompt_length=prompt_length,
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            dtype_bytes=cfg.dtype_bytes,
        )

        throughput = (
            cfg.batch_size * prompt_length / (median_ms / 1_000.0)
            if median_ms > 0
            else float("inf")
        )

        results.append(
            BenchmarkResult(
                prompt_length=prompt_length,
                median_ms=median_ms,
                p95_ms=p95_ms,
                median_prompt_tokens_per_second=throughput,
                accounting=accounting,
            )
        )

        print(
            f"Completed T={prompt_length:,}: "
            f"median={median_ms:.4f} ms, p95={p95_ms:.4f} ms"
        )

    return results


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    results = run_experiment(args)
    print_results(results)

    prompt_lengths = [result.prompt_length for result in results]
    median_latencies = [result.median_ms for result in results]
    alpha, coefficient, r_squared = fit_power_law(
        prompt_lengths,
        median_latencies,
    )

    print("=================== Empirical Scaling Fit ===================")
    print(f"median latency ≈ {coefficient:.6g} × T^{alpha:.3f}")
    print(f"R²             = {r_squared:.4f}")
    print("Interpretation: an exponent near 1 is linear; near 2 is quadratic.")
    print("The measured exponent can differ because kernels, launch overhead,")
    print("hardware utilization, and memory behavior also affect latency.")
    print("=============================================================\n")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    latency_path = args.out_dir / "prefill_latency_vs_prompt_length.png"
    throughput_path = args.out_dir / "prefill_throughput_vs_prompt_length.png"
    log_log_path = args.out_dir / "prefill_latency_loglog_fit.png"

    plot_latency(results, latency_path)
    plot_throughput(results, throughput_path)
    plot_log_log_fit(
        results,
        alpha,
        coefficient,
        r_squared,
        log_log_path,
    )

    print("Saved aggregated plots:")
    print(f"  {latency_path}")
    print(f"  {throughput_path}")
    print(f"  {log_log_path}")
    print("No individual timing samples were saved.")


if __name__ == "__main__":
    main()