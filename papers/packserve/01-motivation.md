# PackServe — Motivation

**Paper:** [PackServe (arXiv:2609.33224)](https://arxiv.org/pdf/2609.33224)

## Problem Statement

Agentic LLM workloads introduce competing requirements for inference serving systems:

1. **KV-cache locality:** Reuse previously computed KV-cache prefixes to avoid redundant prefill computation.
2. **Latency SLO compliance:** Protect request latency, particularly Time to First Token (TTFT) and Time per Output Token (TPOT).
3. **GPU efficiency:** Consolidate requests to improve batching efficiency and reduce the active GPU footprint.

These objectives can conflict.

Routing a request to an instance with a matching cached prefix reduces prefill work but may increase contention. Routing to a less-loaded instance can reduce contention but sacrifice cache locality. Packing more requests onto an instance may improve throughput while increasing TPOT.

## Central Scheduling Challenge

Determine where to place incoming requests such that:

- Existing and incoming requests satisfy latency SLOs.
- KV-cache reuse is exploited when beneficial.
- GPU resources are utilized efficiently.

## Key Engineering Insight

**The least-loaded GPU or the GPU with the largest cached-prefix match is not necessarily the optimal placement.**

Scheduling must consider the marginal performance impact of admitting a request, including prefill cost, decode interference, batching efficiency, and remaining SLO headroom.

## Conceptual Optimization Objective

Minimize GPU resource cost, subject to latency and resource constraints.

This is our conceptual framing of the problem, not a transcription of PackServe's formal optimization formulation.

