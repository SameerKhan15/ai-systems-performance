#!/usr/bin/env python3
"""
Lab 1: KV Cache from First Principles

Goal:
- Build the smallest possible attention-only transformer block.
- Simulate autoregressive generation one token at a time.
- Compare naive no-cache generation against KV-cache generation.
- Count projection work, attention matmul work, memory, and timing.

Default dimensions match the lab brief:
    embed_dim = 16
    num_heads = 2
    head_dim  = 8
    seq_len   = 8

No feed-forward network. No layer normalization.

Use the small sequence first for conceptual clarity:
python lab1_kv_cache.py --seq-len 6 --verbose

Then use larger sequence lengths to make the deltas obvious:
python lab1_kv_cache.py --seq-len 256 --runs 10
python lab1_kv_cache.py --seq-len 512 --runs 10
python lab1_kv_cache.py --seq-len 1024 --runs 10
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Configuration and accounting
# -----------------------------

# ---------------------------------------------------------------------------------------------------------------------
# Configuration object for the lab, defined as Python dataclass.
# A dataclass is a clean way to define a small container of related values without writing a full constructor manually.
# frozen=True means once you create a LabConfig, you cannot accidentally mutate it.
# ---------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LabConfig:
    # each token is represented as a vector of length 16
    # So if the input sentence has 6 tokens: "The cat sat on the mat", then before attention, we represent it as something shaped like:
    # [6 tokens, 16 features]
    # [seq_len, embed_dim] = [6, 16]
    embed_dim: int = 16
    # This means the attention layer has 2 attention heads.
    # So instead of doing one attention operation over a 16-dimensional vector, we split the vector across 2 heads.
    # token vector length 16
    # head 1 sees 8 dimensions
    # head 2 sees 8 dimensions
    num_heads: int = 2
    # default sequence length for the lab
    # the code lets us override this from the command line
    # python lab1_kv_cache.py --seq-len 6 --verbose
    seq_len: int = 8
    # This controls the size of the fake vocabulary.
    vocab_size: int = 16384
    # we will use float32 by default
    # So every floating-point number in Q, K, V, attention scores, and output uses 4 bytes.
    # This matters for the memory accounting part of the lab.
    # For example, one KV cache element in float32 uses: 4 bytes
    dtype: torch.dtype = torch.float32
    device: str = "cuda"

    # This computes the dimension of each attention head.
    @property
    def head_dim(self) -> int:
        # This checks that embed_dim can be evenly split across heads.
        assert self.embed_dim % self.num_heads == 0
        return self.embed_dim // self.num_heads
    # This computes how many bytes each number consumes for the configured dtype.
    # For our default, we get 4 bytes
    # Using the defaults and Wk, Wq being [16,16]: each token's full K tensor has 16 numbers
    # Each token's V tensor also has 16 numbers
    # Therefore, for one generated token, the KV cache stores:
    # K: 16 numbers
    # V: 16 numbers
    # total: 32 numbers
    # In float32:
    # 32 numbers * 4 bytes = 128 bytes per token
    # So after 6 tokens, the cache contains 6 * 128 = 768 bytes
    @property
    def dtype_bytes(self) -> int:
        if self.dtype == torch.float16 or self.dtype == torch.bfloat16:
            return 2
        if self.dtype == torch.float64:
            return 8
        return 4

@dataclass
class Counters:
    # This defines a mutable dataclass. Unlike LabConfig, this one is not frozen, because the counters need to increase during the run.
    # The docstring is important: MACs = multiply-accumulate operations.
    # A matrix multiplication is made of repeated multiply-and-add operations.
    # [1, 2, 3] · [4, 5, 6] = 1×4 + 2×5 + 3×6
    # That is 3 MACs.
    # Some people count one MAC as 2 FLOPs because it has one multiply and one add. This lab keeps things simpler and counts MACs.
    """Operation accounting in MACs: multiply-accumulate units, not FLOPs."""
    # This section is the instrumentation core of the lab.
    # It does not perform attention. It just counts how much work the attention code is doing.
    # This class records how many token projections and attention MACs happened.

    # These count how many token vectors were projected through each linear layer.
    # For example, if at one step we process 5 tokens without cache:
    # K projection tokens += 5
    # V projection tokens += 5

    # If we process only the newest token with KV cache:
    # K projection tokens += 1
    # V projection tokens += 1
    # This is what makes the recomputation waste visible
    q_projection_tokens: int = 0
    k_projection_tokens: int = 0
    v_projection_tokens: int = 0
    o_projection_tokens: int = 0

    # These count the actual matrix multiplication work for each projection.
    # In this lab, each projection is assumed to be:
    # [embed_dim] → [embed_dim]
    # So for one token:
    # projection MACs = embed_dim × embed_dim
    # With:
    # embed_dim = 16
    # one projection costs: 16 × 16 = 256 MACs
    # So for one token:
    # Q projection = 256 MACs
    # K projection = 256 MACs
    # V projection = 256 MACs
    # O projection = 256 MACs
    #
    # For 5 tokens:
    # 5 × 16 × 16 = 1280 MACs per projection
    # Important caveat: this assumes Wq, Wk, Wv, and Wo are all square matrices of shape:
    # [embed_dim, embed_dim]
    q_projection_macs: int = 0
    k_projection_macs: int = 0
    v_projection_macs: int = 0
    o_projection_macs: int = 0

    # These count attention work.
    # There are two major attention matrix multiplications:
    # 1. Q @ K^T // This produces attention scores.
    # 2. attention_weights @ V // This produces the weighted value output.
    # So attention has two main matmul costs:
    # score computation
    # value aggregation
    # The third field: attention_score_elements does not count compute. It counts how many scalar attention scores were materialized.
    # For example, for one head with:
    # query_len = 5, key_len = 5
    # the attention score matrix has shape: [5,5] = 25 score elements
    # With 2 heads: 2 × 5 × 5 = 50 score elements
    attention_score_macs: int = 0      # Q @ K^T
    attention_value_macs: int = 0      # softmax(scores) @ V
    attention_score_elements: int = 0  # number of score scalars materialized

    # projection accounting function
    def add_projection(self, name: str, token_count: int, embed_dim: int) -> None:
        # Each token costs embed_dim × embed_dim MACs.
        # e.g. token_count = 5, embed_dim = 16: macs = 5 × 16 × 16 = 1280
        # Then the method assigns those MACs to the right projection counter.
        macs = token_count * embed_dim * embed_dim
        # If the projection is q, increment Q token count and Q MAC count. Same pattern for others
        if name == "q":
            self.q_projection_tokens += token_count
            self.q_projection_macs += macs
        elif name == "k":
            self.k_projection_tokens += token_count
            self.k_projection_macs += macs
        elif name == "v":
            self.v_projection_tokens += token_count
            self.v_projection_macs += macs
        # The O projection is the final output projection after attention.
        elif name == "o":
            self.o_projection_tokens += token_count
            self.o_projection_macs += macs
        else:
            raise ValueError(f"unknown projection {name}")
    # attention accounting function
    def add_attention(self, query_len: int, key_len: int, num_heads: int, head_dim: int) -> None:
        # For every head, QK^T has query_len * key_len dot-products of length head_dim.
        score_macs = num_heads * query_len * key_len * head_dim
        # For every head, attention_weights @ V has query_len * head_dim outputs,
        # each combining key_len values.
        value_macs = num_heads * query_len * key_len * head_dim
        self.attention_score_macs += score_macs
        self.attention_value_macs += value_macs
        self.attention_score_elements += num_heads * query_len * key_len

    @property
    def projection_macs(self) -> int:
        return self.q_projection_macs + self.k_projection_macs + self.v_projection_macs + self.o_projection_macs

    @property
    def attention_macs(self) -> int:
        return self.attention_score_macs + self.attention_value_macs

    @property
    def total_macs(self) -> int:
        return self.projection_macs + self.attention_macs


@dataclass
class StepRecord:
    mode: str
    step: int
    token: str
    prefix_len: int
    q_projected_tokens_this_step: int
    k_projected_tokens_this_step: int
    v_projected_tokens_this_step: int
    attention_query_len_this_step: int
    attention_key_len_this_step: int
    persistent_kv_cache_bytes: int
    temporary_qkv_bytes_this_step: int
    attention_score_bytes_this_step: int
    total_macs_after_step: int


# -----------------------------
# Tiny attention-only transformer
# -----------------------------


class TinyAttentionOnlyTransformer(nn.Module):
    """
    This is intentionally tiny and incomplete.

    It has:
      token embedding lookup
      W_Q, W_K, W_V projections
      scaled dot-product attention
      W_O output projection

    It does NOT have:
      feed-forward network
      layer norm
      residual connections
      positional encodings
      logits head

    The point is to isolate attention and KV caching.
    """

    def __init__(self, cfg: LabConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.wq = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=False)
        self.wk = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=False)
        self.wv = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=False)
        self.wo = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=False)

        # Deterministic initialization keeps the lab reproducible.
        torch.manual_seed(7)
        for p in self.parameters():
            nn.init.normal_(p, mean=0.0, std=0.02)

        self.to(device=cfg.device, dtype=cfg.dtype)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, D] -> [B, H, T, Hd]
        bsz, seq_len, embed_dim = x.shape
        return x.view(bsz, seq_len, self.cfg.num_heads, self.cfg.head_dim).transpose(1, 2)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, H, T, Hd] -> [B, T, D]
        bsz, num_heads, seq_len, head_dim = x.shape
        return x.transpose(1, 2).contiguous().view(bsz, seq_len, num_heads * head_dim)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
        # q: [B, H, Tq, Hd]
        # k: [B, H, Tk, Hd]
        # v: [B, H, Tk, Hd]
        scale = 1.0 / math.sqrt(self.cfg.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, Tq, Tk]

        if causal:
            # Used only in no-cache full-prefix mode. For token i, it cannot see future token j > i.
            t_q, t_k = scores.shape[-2], scores.shape[-1]
            mask = torch.ones((t_q, t_k), dtype=torch.bool, device=scores.device).triu(diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, v)  # [B, H, Tq, Hd]


# -----------------------------
# Input construction
# -----------------------------


def make_token_ids(tokens: List[str], device: str) -> torch.Tensor:
    """Map each distinct token to a stable toy id."""
    vocab: Dict[str, int] = {}
    ids: List[int] = []
    for tok in tokens:
        if tok not in vocab:
            vocab[tok] = len(vocab) + 1
        ids.append(vocab[tok])
    return torch.tensor(ids, dtype=torch.long, device=device)


# -----------------------------
# Version A: no KV cache
# -----------------------------


@torch.no_grad()
def generate_without_cache(
    model: TinyAttentionOnlyTransformer,
    token_ids: torch.Tensor,
    token_text: List[str],
    verbose: bool = False,
) -> Tuple[torch.Tensor, Counters, List[StepRecord]]:
    """
    Naive autoregressive generation.

    At generation step t, feed the entire prefix [1..t] back into the model.
    This recomputes Q, K, and V for all previous tokens every step.
    """
    cfg = model.cfg
    counters = Counters()
    records: List[StepRecord] = []
    last_outputs: List[torch.Tensor] = []

    for step in range(1, len(token_ids) + 1):
        prefix_ids = token_ids[:step].unsqueeze(0)  # [1, t]
        x = model.embedding(prefix_ids)             # [1, t, D]

        # Full-prefix projections: recompute all old tokens plus the new one.
        q = model.wq(x)
        k = model.wk(x)
        v = model.wv(x)
        counters.add_projection("q", step, cfg.embed_dim)
        counters.add_projection("k", step, cfg.embed_dim)
        counters.add_projection("v", step, cfg.embed_dim)

        qh = model.split_heads(q)
        kh = model.split_heads(k)
        vh = model.split_heads(v)

        # Full causal attention over the prefix.
        counters.add_attention(query_len=step, key_len=step, num_heads=cfg.num_heads, head_dim=cfg.head_dim)
        attn_out = model.attention(qh, kh, vh, causal=True)
        merged = model.merge_heads(attn_out)
        out = model.wo(merged)
        counters.add_projection("o", step, cfg.embed_dim)

        last_outputs.append(out[:, -1, :])

        # Persistent KV cache is zero because this mode discards K and V after every step.
        persistent_kv_cache_bytes = 0
        temporary_qkv_bytes = 3 * step * cfg.embed_dim * cfg.dtype_bytes
        attention_score_bytes = cfg.num_heads * step * step * cfg.dtype_bytes

        records.append(StepRecord(
            mode="without_cache",
            step=step,
            token=token_text[step - 1],
            prefix_len=step,
            q_projected_tokens_this_step=step,
            k_projected_tokens_this_step=step,
            v_projected_tokens_this_step=step,
            attention_query_len_this_step=step,
            attention_key_len_this_step=step,
            persistent_kv_cache_bytes=persistent_kv_cache_bytes,
            temporary_qkv_bytes_this_step=temporary_qkv_bytes,
            attention_score_bytes_this_step=attention_score_bytes,
            total_macs_after_step=counters.total_macs,
        ))

        if verbose:
            print(f"[No cache] step {step}: prefix={token_text[:step]}")
            print(f"  recomputed Q/K/V for tokens 1..{step}")
            print(f"  attention shape per head: [{step} queries x {step} keys]")

    return torch.cat(last_outputs, dim=0), counters, records


# -----------------------------
# Version B: with KV cache
# -----------------------------


@torch.no_grad()
def generate_with_kv_cache(
    model: TinyAttentionOnlyTransformer,
    token_ids: torch.Tensor,
    token_text: List[str],
    verbose: bool = False,
) -> Tuple[torch.Tensor, Counters, List[StepRecord]]:
    """
    KV-cache autoregressive generation.

    At generation step t:
      - compute Q_t, K_t, V_t only for the new token
      - append K_t and V_t to cache
      - compute Q_t @ K_cache^T
      - compute attention_weights @ V_cache

    We do NOT cache Q because old Q vectors are not needed to generate the next token.
    """
    cfg = model.cfg
    counters = Counters()
    records: List[StepRecord] = []
    last_outputs: List[torch.Tensor] = []

    # A real inference engine usually allocates cache storage and writes each
    # new token into the next slot. This avoids repeatedly copying the whole cache.
    max_steps = len(token_ids)
    k_cache = torch.empty((1, cfg.num_heads, max_steps, cfg.head_dim), device=cfg.device, dtype=cfg.dtype)
    v_cache = torch.empty((1, cfg.num_heads, max_steps, cfg.head_dim), device=cfg.device, dtype=cfg.dtype)

    for step in range(1, len(token_ids) + 1):
        current_id = token_ids[step - 1].view(1, 1)  # [1, 1]
        x = model.embedding(current_id)              # [1, 1, D]

        # Only current token projections.
        q = model.wq(x)
        k = model.wk(x)
        v = model.wv(x)
        counters.add_projection("q", 1, cfg.embed_dim)
        counters.add_projection("k", 1, cfg.embed_dim)
        counters.add_projection("v", 1, cfg.embed_dim)

        qh = model.split_heads(q)  # [1, H, 1, Hd]
        kh = model.split_heads(k)  # [1, H, 1, Hd]
        vh = model.split_heads(v)  # [1, H, 1, Hd]

        # Append K_t and V_t into the next cache slot.
        k_cache[:, :, step - 1:step, :] = kh
        v_cache[:, :, step - 1:step, :] = vh
        active_k_cache = k_cache[:, :, :step, :]
        active_v_cache = v_cache[:, :, :step, :]

        # KEY MOMENT: one current query attends to all cached keys.
        counters.add_attention(query_len=1, key_len=step, num_heads=cfg.num_heads, head_dim=cfg.head_dim)
        attn_out = model.attention(qh, active_k_cache, active_v_cache, causal=False)
        merged = model.merge_heads(attn_out)
        out = model.wo(merged)
        counters.add_projection("o", 1, cfg.embed_dim)

        last_outputs.append(out[:, -1, :])

        persistent_kv_cache_bytes = 2 * step * cfg.embed_dim * cfg.dtype_bytes
        temporary_qkv_bytes = 3 * 1 * cfg.embed_dim * cfg.dtype_bytes
        attention_score_bytes = cfg.num_heads * 1 * step * cfg.dtype_bytes

        records.append(StepRecord(
            mode="with_kv_cache",
            step=step,
            token=token_text[step - 1],
            prefix_len=step,
            q_projected_tokens_this_step=1,
            k_projected_tokens_this_step=1,
            v_projected_tokens_this_step=1,
            attention_query_len_this_step=1,
            attention_key_len_this_step=step,
            persistent_kv_cache_bytes=persistent_kv_cache_bytes,
            temporary_qkv_bytes_this_step=temporary_qkv_bytes,
            attention_score_bytes_this_step=attention_score_bytes,
            total_macs_after_step=counters.total_macs,
        ))

        if verbose:
            print(f"[KV cache] step {step}: token={token_text[step - 1]}")
            print("  computed Q/K/V only for current token")
            print(f"  active K_cache shape: {list(active_k_cache.shape)}")
            print(f"  active V_cache shape: {list(active_v_cache.shape)}")
            print(f"  attention shape per head: [1 query x {step} cached keys]")

    return torch.cat(last_outputs, dim=0), counters, records


# -----------------------------
# Reporting and plotting
# -----------------------------


def bytes_to_human(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 ** 2:
        return f"{num_bytes / 1024:.2f} KB"
    if num_bytes < 1024 ** 3:
        return f"{num_bytes / 1024 ** 2:.2f} MB"
    return f"{num_bytes / 1024 ** 3:.2f} GB"


def save_step_records(records: List[StepRecord], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))


def plot_cache_growth(no_cache_records: List[StepRecord], cache_records: List[StepRecord], out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    x_no = [r.step for r in no_cache_records]
    y_no = [r.persistent_kv_cache_bytes for r in no_cache_records]
    x_cache = [r.step for r in cache_records]
    y_cache = [r.persistent_kv_cache_bytes for r in cache_records]

    plt.figure(figsize=(8, 5))
    plt.plot(x_no, y_no, marker="o", label="Without cache: persistent K/V cache")
    plt.plot(x_cache, y_cache, marker="o", label="With cache: persistent K/V cache")
    plt.xlabel("Generation steps elapsed")
    plt.ylabel("Persistent KV cache size, bytes")
    plt.title("KV Cache Growth")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_attention_score_memory(no_cache_records: List[StepRecord], cache_records: List[StepRecord], out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    x_no = [r.step for r in no_cache_records]
    y_no = [r.attention_score_bytes_this_step for r in no_cache_records]
    x_cache = [r.step for r in cache_records]
    y_cache = [r.attention_score_bytes_this_step for r in cache_records]

    plt.figure(figsize=(8, 5))
    plt.plot(x_no, y_no, marker="o", label="Without cache: score matrix bytes this step")
    plt.plot(x_cache, y_cache, marker="o", label="With cache: score vector bytes this step")
    plt.xlabel("Generation steps elapsed")
    plt.ylabel("Attention score memory, bytes")
    plt.title("Temporary Attention Score Memory Per Step")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_total_macs(no_cache: Counters, cache: Counters, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    labels = ["Projections", "Attention", "Total"]
    no_values = [no_cache.projection_macs, no_cache.attention_macs, no_cache.total_macs]
    cache_values = [cache.projection_macs, cache.attention_macs, cache.total_macs]

    x = range(len(labels))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar([i - width / 2 for i in x], no_values, width, label="Without cache")
    plt.bar([i + width / 2 for i in x], cache_values, width, label="With cache")
    plt.xticks(list(x), labels)
    plt.ylabel("MACs")
    plt.title("Compute Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def print_counter_summary(no_cache: Counters, cache: Counters, final_step_cache_bytes: int) -> None:
    print("\n==================== Counter Summary ====================")
    print(f"{'Metric':36s} {'Without cache':>18s} {'With KV cache':>18s} {'Reduction':>12s}")
    print("-" * 88)

    rows = [
        ("Q projected token-count", no_cache.q_projection_tokens, cache.q_projection_tokens),
        ("K projected token-count", no_cache.k_projection_tokens, cache.k_projection_tokens),
        ("V projected token-count", no_cache.v_projection_tokens, cache.v_projection_tokens),
        ("O projected token-count", no_cache.o_projection_tokens, cache.o_projection_tokens),
        ("Projection MACs", no_cache.projection_macs, cache.projection_macs),
        ("Attention score MACs", no_cache.attention_score_macs, cache.attention_score_macs),
        ("Attention value MACs", no_cache.attention_value_macs, cache.attention_value_macs),
        ("Total MACs", no_cache.total_macs, cache.total_macs),
        ("Attention score elements", no_cache.attention_score_elements, cache.attention_score_elements),
    ]

    for label, a, b in rows:
        if a == 0:
            reduction = "n/a"
        else:
            reduction = f"{(1 - b / a) * 100:8.1f}%"
        print(f"{label:36s} {a:18,d} {b:18,d} {reduction:>12s}")

    print("-" * 88)
    print(f"Final persistent KV cache size: {final_step_cache_bytes:,} bytes ({bytes_to_human(final_step_cache_bytes)})")
    print("Formula for this toy, one layer, batch=1:")
    print("  KV bytes = 2 * seq_len * embed_dim * dtype_bytes")
    print("           = K cache + V cache")
    print("=========================================================\n")


def run_timing(model: TinyAttentionOnlyTransformer, token_ids: torch.Tensor, token_text: List[str], runs: int) -> Tuple[float, float]:
    # Warmup
    generate_without_cache(model, token_ids, token_text, verbose=False)
    generate_with_kv_cache(model, token_ids, token_text, verbose=False)

    if model.cfg.device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(runs):
        generate_without_cache(model, token_ids, token_text, verbose=False)
    if model.cfg.device == "cuda":
        torch.cuda.synchronize()
    no_cache_s = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(runs):
        generate_with_kv_cache(model, token_ids, token_text, verbose=False)
    if model.cfg.device == "cuda":
        torch.cuda.synchronize()
    cache_s = time.perf_counter() - start

    return no_cache_s / runs, cache_s / runs


def build_token_sequence(seq_len: int) -> List[str]:
    base = ["The", "cat", "sat", "on", "the", "mat"]
    if seq_len <= len(base):
        return base[:seq_len]
    return base + [f"tok{i}" for i in range(len(base) + 1, seq_len + 1)]


def maybe_parse_dtype(dtype_name: str) -> torch.dtype:
    normalized = dtype_name.lower()
    if normalized == "float32":
        return torch.float32
    if normalized == "float16":
        return torch.float16
    if normalized == "bfloat16":
        return torch.bfloat16
    if normalized == "float64":
        return torch.float64
    raise ValueError("dtype must be float32, float16, bfloat16, or float64")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lab 1: KV Cache from First Principles")
    parser.add_argument("--seq-len", type=int, default=8, help="number of generation steps")
    parser.add_argument("--embed-dim", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=2048, help="toy embedding vocabulary size")
    parser.add_argument("--runs", type=int, default=50, help="timing repetitions")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16", "float64"])
    parser.add_argument("--out-dir", type=str, default="lab1_outputs")
    parser.add_argument("--torch-threads", type=int, default=1, help="CPU threads for PyTorch; 1 is best for tiny matrices")
    parser.add_argument("--verbose", action="store_true", help="print step-by-step generation details")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    if args.device == "cpu":
        torch.set_num_threads(args.torch_threads)

    dtype = maybe_parse_dtype(args.dtype)
    cfg = LabConfig(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        vocab_size=max(args.vocab_size, args.seq_len + 2),
        dtype=dtype,
        device=args.device,
    )

    tokens = build_token_sequence(cfg.seq_len)
    token_ids = make_token_ids(tokens, cfg.device)
    model = TinyAttentionOnlyTransformer(cfg)

    print("\n==================== Lab 1 Configuration ====================")
    print(f"tokens      : {tokens}")
    print(f"embed_dim   : {cfg.embed_dim}")
    print(f"num_heads   : {cfg.num_heads}")
    print(f"head_dim    : {cfg.head_dim}")
    print(f"seq_len     : {cfg.seq_len}")
    print(f"dtype       : {args.dtype} ({cfg.dtype_bytes} bytes/scalar)")
    print(f"device      : {cfg.device}")
    print("=============================================================\n")

    no_cache_out, no_cache_counters, no_cache_records = generate_without_cache(
        model, token_ids, tokens, verbose=args.verbose
    )
    cache_out, cache_counters, cache_records = generate_with_kv_cache(
        model, token_ids, tokens, verbose=args.verbose
    )

    # The last-token outputs should match closely because the two modes implement the same math.
    max_abs_diff = (no_cache_out - cache_out).abs().max().item()
    print(f"Max abs difference between no-cache and cache outputs: {max_abs_diff:.6g}")

    final_cache_bytes = cache_records[-1].persistent_kv_cache_bytes
    print_counter_summary(no_cache_counters, cache_counters, final_cache_bytes)

    avg_no_cache_s, avg_cache_s = run_timing(model, token_ids, tokens, runs=args.runs)
    print("==================== Timing ====================")
    print(f"Average time without cache: {avg_no_cache_s * 1000:.4f} ms/run over {args.runs} runs")
    print(f"Average time with KV cache: {avg_cache_s * 1000:.4f} ms/run over {args.runs} runs")
    if avg_cache_s > 0:
        print(f"Speedup: {avg_no_cache_s / avg_cache_s:.2f}x")
    print("Note: for seq_len=8, Python overhead can dominate. Increase --seq-len to see the asymptotic effect.")
    print("================================================\n")

    out_dir = Path(args.out_dir)
    save_step_records(no_cache_records + cache_records, out_dir / "step_records.csv")
    plot_cache_growth(no_cache_records, cache_records, out_dir / "cache_growth_bytes.png")
    plot_attention_score_memory(no_cache_records, cache_records, out_dir / "attention_score_memory_bytes.png")
    plot_total_macs(no_cache_counters, cache_counters, out_dir / "compute_comparison_macs.png")

    print("Saved outputs:")
    print(f"  {out_dir / 'step_records.csv'}")
    print(f"  {out_dir / 'cache_growth_bytes.png'}")
    print(f"  {out_dir / 'attention_score_memory_bytes.png'}")
    print(f"  {out_dir / 'compute_comparison_macs.png'}")


if __name__ == "__main__":
    main()
