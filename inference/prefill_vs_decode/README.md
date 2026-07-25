# Two phases of LLM inference  
## Prefill  
For an input prompt such as 2,000 tokens:  
* All prompt tokens are processed together  
* Large matrix multiplications operate over many tokens  
* The KV cache is initially populated  
* GPU parallelism is relatively high  
* Performance is often more compute-bound  

**Primary metric:**  
`TTFT=Time to First Token`  

## Decode  
After prefill, generation happens one token at a time:  
* Only one new token is projected  
* Its query reads the increasingly large KV cache  
* Matrix operations are much smaller  
* GPU utilization can be poor at batch size 1  
* Performance increasingly becomes memory-bandwidth-bound  

**Primary metric:**  
`TPOT=Time Per Output Token`  

This distinction is central to understanding real inference systems. KV cache is not simply “a compute optimization.” It trades recomputation for persistent memory and repeated memory reads.  

# Build a Prefill vs. Decode Performance Lab using your existing transformer  
**Experiment, What to vary**  
Prefill latency, Prompt length: 128, 256, 512, 1024, 2048  
* Decode latency, Current KV-cache length  
* Batch behavior, Batch sizes: 1, 2, 4, 8  
* Cache memory, Sequence length and dtype  
* Arithmetic intensity, MACs divided by bytes moved  
* User-facing latency, TTFT and average TPOT  

Produce these plots:  
1. Prefill latency versus prompt length  
2. Decode time per token versus KV-cache length  
3. KV-cache bytes versus sequence length  
4. Arithmetic intensity: prefill versus decode  
5. Tokens per second versus batch size  

The conceptual result we want to demonstrate is:  
Prefill:  
many tokens × large matrix operations  
→ substantial parallelism  
→ usually compute-heavy  

Decode:  
one token × growing KV cache  
→ small matrix operations plus large memory reads  
→ usually memory-bandwidth-heavy  

# Lab: Prefill Latency vs. Prompt Length  
## Central question  

**How does the latency of transformer prefill change as the input prompt becomes longer, and how does that behavior relate to linear projection work and quadratic attention work?**  
## Why this follows the KV-cache lab  
The KV-cache lab focused on autoregressive **decode**, where the model processes one new token at a time and reuses stored keys and values.  
This lab isolates **prefill**:  
```text
Entire input prompt → one forward pass → initial KV state / first-token readiness
```

Understanding prefill gives you the basis for reasoning about **time to first token (TTFT)**.  

## Learning objectives

After completing this lab, we should be able to explain:  
1. Why Q/K/V/O projection work is linear in prompt length.  
2. Why full self-attention work is quadratic in prompt length.  
3. Why doubling prompt length does not necessarily just double prefill latency.  
4. Why attention-score memory grows quadratically for a naïve implementation.  
5. Why measured latency can differ from theoretical MAC growth.  
6. How to estimate an empirical scaling exponent from a log-log plot.  

## Scope  
The model contains one attention-only transformer block:  

- token embedding  
- Q, K, and V projections  
- causal self-attention  
- output projection  

It deliberately excludes feed-forward layers, layer normalization, residual connections, positional encodings, and a language-model head. This keeps the experiment focused on attention prefill.  

## Hypothesis  
For batch size $B$, prompt length $T$, and embedding dimension $D$:  

$$
\text{Projection MACs} = 4BTD^2
$$

$$
\text{Attention MACs} = 2BT^2D
$$

Therefore:  
- projection work grows as $O(T)$;  
- attention work grows as $O(T^2)$;  
- measured latency should become increasingly nonlinear as prompts grow and attention becomes a larger fraction of total work.  

## Files

```text
02_prefill_latency_vs_prompt_length/
├── README.md
├── hand_calculations.md
├── prefill_latency_lab.py
├── test_prefill_latency_lab.py
└── outputs/
    └── .gitkeep
```

## Run the tests first

From this directory:

```bash
python -m unittest -v
```

The tests verify:

- closed-form MAC accounting;
- quadratic attention growth when prompt length doubles;
- output tensor shape;
- causal behavior;
- percentile calculation;
- power-law fitting.

## Recommended A100 run

```bash
python prefill_latency_lab.py \
  --device cuda \
  --dtype float16 \
  --embed-dim 512 \
  --num-heads 8 \
  --batch-size 1 \
  --prompt-lengths 128,256,512,1024,2048 \
  --warmups 5 \
  --runs 30 \
  --attention-backend manual
```

The `manual` backend explicitly materializes the attention-score matrix. This makes the first-principles tensor shapes visible, but its memory usage grows as $O(T^2)$.

For longer prompts, compare the optimized PyTorch backend:

```bash
python prefill_latency_lab.py \
  --device cuda \
  --dtype float16 \
  --embed-dim 512 \
  --num-heads 8 \
  --batch-size 1 \
  --prompt-lengths 128,256,512,1024,2048,4096 \
  --warmups 5 \
  --runs 30 \
  --attention-backend sdpa \
  --out-dir outputs_sdpa
```

Do not treat this backend comparison as a complete FlashAttention lab. It is only a way to observe that an optimized implementation can have the same logical attention operation while using different kernels and memory behavior.

## Outputs

The script prints aggregated results and saves only PNG plots:

```text
outputs/
├── prefill_latency_vs_prompt_length.png
├── prefill_throughput_vs_prompt_length.png
└── prefill_latency_loglog_fit.png
```

It does **not** save individual timing samples.

### Plot 1: Prefill latency vs. prompt length

This is the primary experiment. Compare median and p95 latency as prompt length increases.

Questions to answer:

- Does latency initially appear close to linear?
- Where does the curve begin bending upward?
- Is p95 close to the median, or is there substantial timing variability?

### Plot 2: Prefill throughput vs. prompt length

This reports prompt tokens processed per second:

$$
\text{Prompt throughput} = \frac{B \times T}{\text{median latency in seconds}}
$$

Throughput may rise initially as larger matrices use the hardware more efficiently. It may later flatten or decline as quadratic attention becomes increasingly expensive.

### Plot 3: Log-log scaling fit

The script fits:

$$
\text{Latency} \approx cT^\alpha
$$

Interpret the fitted exponent cautiously:

```text
alpha near 1 → approximately linear over the measured range
alpha near 2 → approximately quadratic over the measured range
between 1 and 2 → mixed projection/attention behavior plus hardware effects
```

A fitted exponent is a summary of the selected measurement range, not a universal constant for the model.

## Expected conceptual result

When prompt length doubles:

```text
Projection MACs: 2×
Attention MACs:  4×
Score memory:    4×
```

For the simplified block, projection and attention MACs are equal when:

$$
T = 2D
$$

With `embed_dim = 512`, this theoretical crossover occurs around `T = 1024` tokens.

## Experiment worksheet

After running the lab, record your answers:

### Observation 1

At what prompt length does measured latency begin increasing noticeably faster than linearly?

```text
Answer:
```

### Observation 2

What empirical exponent $\alpha$ did the script report?

```text
Answer:
```

### Observation 3

At `T = 1024` and `D = 512`, are projection and attention MACs equal in the script's accounting?

```text
Answer:
```

### Observation 4

How does the manual backend compare with SDPA at longer prompt lengths?

```text
Answer:
```

### Final systems takeaway

Complete this statement in your own words:

> Prefill latency grows with prompt length because...

```text
Answer:
```