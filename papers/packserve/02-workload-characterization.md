# PackServe — Workload Characterization

## Production Workload

The paper analyzes approximately 10.1 million inference requests spanning 14 models over 14 days.

### Key Observations

| Metric | Observation |
|---|---|
| Input/output token volume ratio | Approximately 192:1 |
| Cached input token fraction | 77.7% |
| Aggregate request duration spent in decode | 78.4% |
| Median instance concurrency | 0.76 requests |
| Median TTFT improvement in highest versus lowest cache-reuse group | 66.2% lower |

## Interpretation

### 1. Agentic workloads exhibit substantial prefix reuse

Agentic interactions frequently carry forward system instructions, tool context, and conversation history.

This creates opportunities to reuse existing KV-cache entries and reduce redundant prefill computation.

### 2. Input-token-heavy does not imply prefill-time-heavy

Although input tokens vastly outnumber output tokens, decode can dominate elapsed request duration.

Contributing factors include:

- Parallel execution of prefill operations.
- KV-cache reuse reducing new prefill work.
- Sequential autoregressive token generation during decode.

### 3. Low concurrency suggests opportunities for consolidation

Low observed instance concurrency suggests that requests may be spread across more serving capacity than necessary.

Consolidating compatible workloads can improve GPU efficiency, provided latency and memory constraints are respected.

## Worked Example

Assume:

- Input tokens: 192,000
- Output tokens: 1,000
- Cached input fraction: 77.7%

New input tokens requiring prefill computation:

\[
192{,}000(1-0.777)=42{,}816
\]

Thus, 149,184 input tokens can be reused, assuming their cached KV entries remain available.

**Important:** A 77.7% reduction in newly processed input tokens does not imply an equivalent reduction in prefill latency.

## Engineering Takeaway

Workload characterization should measure token distributions, cache reuse, phase-specific execution times, concurrency, and latency SLOs rather than relying solely on request counts.  

