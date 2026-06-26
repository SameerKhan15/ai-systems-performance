# Attention Runtime Breakdown  
## Main Read  
The benchmark is showing the expected transition from fixed/linear-ish overhead at small sequence lengths to quadratic attention cost at larger sequence lengths.  
The script fixes `embed_dim = 128`, runs `sequence lengths` from 64 through 8192, and times the attention steps separately: `Q/K/V projections`, `QK^T`, `softmax`, and `weights @ V`.  

### What stands out  
At small sequence lengths — say 64 to 1024 — everything is very compressed near the bottom. That usually means the benchmark is dominated by fixed costs: kernel launch overhead, synchronization, PyTorch dispatch, memory allocation effects, and small-matrix inefficiency. The GPU is probably not fully utilized yet.  

Around 2048 and especially 4096, the real attention work starts becoming visible.  

By 8192, the dominant contributors are clearly:  
* QKᵀ scores  
* weights @ V  
* softmax  
* Q/K/V projections are comparatively tiny  

That is exactly what we expect for naïve self-attention.  

The core reason is:  
* `Q/K/V projections: O(N * D^2)`  
* `QKᵀ scores: O(N^2 * D)`
* `softmax: O(N^2)`  
* `weights @ V: O(N^2 * D)`  

Here `D = 128` is fixed, so as N = sequence length grows, the N² terms eventually dominate.  

### Important visual caveat  
The legend is hard to read because Matplotlib reused colors. The high green/red lines near the top are Total median and Total p95, not K projection. K projection itself is one of the tiny curves near the bottom.  

So mentally, read the chart as:  
* Top rising curves: total runtime  
* Middle rising curves: QKᵀ, weights @ V, softmax  
* Bottom flat curves: Q/K/V projections  

### P95 vs median  
The p95 curves are very close to the median curves, especially at larger sequence lengths. That is a good sign.  
It means the benchmark is fairly stable and not showing huge run-to-run variance.  

At `8192`, visually:  
* Total median ≈ 3.0 ms  
* Total p95    ≈ 3.1 ms  

### The big takeaway  
This chart is an empirical confirmation of the mental model:  
**Once sequence length gets large enough, self-attention cost is no longer about producing Q, K, and V. It is about materializing and consuming the N × N attention matrix.**  

That is the key reason long-context attention becomes expensive. The attention score matrix grows as:  
* seq_len = 1024 → ~1M scores  
* seq_len = 4096 → ~16M scores  
* seq_len = 8192 → ~67M scores  

The math is:  
X shape  = [N, D]  

Q shape  = [N, D]  
K shape  = [N, D]  

K.T shape = [D, N]  

Q @ K.T  = [N, D] @ [D, N]  
         = [N, N]  

So the attention score matrix has:  
N × N entries  
where N = sequence length  

Even with small embed_dim = 128, the N² work takes over.  

Concrete examples:  
* For `seq_len = 1024`: `1024 × 1024 = 1,048,576 scores`  
* For `seq_len = 4096`: `4096 × 4096 = 16,777,216 scores`
* For `seq_len = 8192`: `8192 × 8192 = 67,108,864 scores`  

**One benchmark caveat**  
This is a good educational benchmark for naïve/materialized attention. It is not representative of fused kernels like FlashAttention,  
because this benchmark explicitly computes and keeps scores, then softmax weights, then weights @ V. That is perfect for learning the cost model,  
but real optimized attention tries hard to avoid materializing the full attention matrix in memory.  

**That is the quadratic explosion.**  

And each score is not free:  
Each individual score is a dot product between one query vector and one key vector.  
Since `embed_dim = 128`, each score requires roughly 128 multiply-adds  

So the QK^T compute is roughly: `N² × D`  

### Memory math  
Assuming float32, each score is 4 bytes.
So the raw score matrix alone is:  
`1024² × 4 bytes  = 4 MB`  
`4096² × 4 bytes  = 64 MB`  
`8192² × 4 bytes  = 256 MB`  

## Key intuition  
Doubling sequence length does not double attention score work. It quadruples it.  
`1024 → 2048 means scores go from 1M to 4M`  
`2048 → 4096 means scores go from 4M to 16M`  
`4096 → 8192 means scores go from 16M to 67M`  

# Attention Dominant Costs  
This chart is the cleaner version of the first one. It removes the tiny Q/K/V projection lines and focuses on the expensive parts of attention:  
* `QKᵀ scores`  
* `softmax`  
* `weights @ V`  
* `total`  

The script explicitly times those stages separately for each sequence length, with `embed_dim = 128` and 30 runs per size.  

## Main interpretation  
The chart says:  
* The dominant cost is not creating Q, K, V.  
* The dominant cost is the N × N attention interaction.  

At larger sequence lengths, the runtime is mostly:  
`total ≈ QKᵀ + softmax + weights @ V`  

So the expensive parts are the two matrix multiplications involving the attention matrix:  
* `Q @ K.T`  
* `weights @ V`  

## Why QKᵀ and weights @ V dominate  
Let:  
* `N = sequence length`  
* `D = embed dimension = 128`  

For QKᵀ:  
* `Q shape   = [N, D]`  
* `K.T shape = [D, N]`  
* `Q @ K.T   = [N, N]`  

That produces the attention score matrix. The compute is approximately:  
`N × N × D`  

Then weights @ V has similar shape:  
* `weights shape = [N, N]`  
* `V shape       = [N, D]`
* `weights @ V   = [N, D]`  

Its compute is also:  
`N × N × D`  

So `QKᵀ` and `weights @ V` are both quadratic in sequence length and linear in embedding dimension.  

## Why softmax is lower but still grows  
Softmax works over the N × N score matrix:  
`softmax(scores, dim=-1)`  

So it touches:  
`N² values`  

But it does not do a `128-wide` dot product for each pair like `QKᵀ` does. That is why `softmax` grows with sequence length but remains below `QKᵀ` and `weights @ V`.  

## Best takeaway from this chart  
For long sequences, self-attention runtime is governed by two large quadratic matmuls plus a quadratic softmax.  

`Attention runtime ≈ O(N²D) + O(N²) + O(N²D) ≈ O(N²D)`  

# Attention Dominant Costs  
**Attention matrix memory is exactly quadratic**  

The attention score matrix has shape:  
scores = Q @ K.T

`Q      = [N, D]  
K.T    = [D, N]  
scores = [N, N]`  

So the number of score values is: `N × N`  

Assuming float32, each value is 4 bytes.  
So memory is:  
`attention matrix memory = N × N × 4 bytes`  

For our sequence lengths:  
| Sequence length |     Scores | Matrix memory |  
| --------------: | ---------: | ------------: |  
|            1024 |  1,048,576 |          4 MB |  
|            2048 |  4,194,304 |         16 MB |  
|            4096 | 16,777,216 |         64 MB |  
|            8192 | 67,108,864 |        256 MB |  

That blue line is exactly this formula.  

## Why peak GPU memory is ~2x attention matrix  
At 8192, the attention matrix itself is around:  
`8192 × 8192 × 4 bytes = 256 MB`  

But our peak GPU memory is around 540 MB.  
That makes sense because our benchmark materializes both:  
`scores  = [N, N]  
weights = [N, N]`  

The softmax creates another full-size matrix:  
`weights = torch.softmax(scores, dim=-1)`  

So at 8192:  
`scores  ≈ 256 MB`  
`weights ≈ 256 MB`  
`-------------------  
subtotal ≈ 512 MB`  

Then add smaller tensors:  
`X      = [N, D]  
Q      = [N, D]  
K      = [N, D]  
V      = [N, D]  
output = [N, D]`  

For N = 8192, D = 128:  
`8192 × 128 × 4 bytes ≈ 4 MB per tensor`  

Five such tensors are about:  
`5 × 4 MB = 20 MB`  

So total expected peak is roughly:  
`scores + weights + X/Q/K/V/output  
≈ 256 MB + 256 MB + 20 MB  
≈ 532 MB`  

Our chart shows roughly 540 MB, which is very consistent.  

## Why median and p95 overlap  
The orange and green lines appear almost identical. That is expected.
Memory here is mostly determined by tensor shapes, not runtime variability. Every run allocates the same major tensors:  
`[N, N] scores  
[N, N] weights  
[N, D] Q/K/V/output`  

## Core takeaway  
This chart proves the key long-context attention issue:  
* Runtime grows because the model computes all token-to-token interactions.  
* Memory grows because it stores those interactions as full N × N matrices.  

For naïve attention:  
`compute ≈ O(N²D)  
memory  ≈ O(N²)`  

And because our implementation materializes both scores and weights, actual peak memory is roughly:  
`peak memory ≈ 2 × N² × bytes_per_value + smaller N×D tensors`  

**This is exactly why FlashAttention-style kernels matter: they try to avoid keeping the full scores and weights matrices in GPU memory at once.**  
