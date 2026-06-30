# Test Setup  
window = 2 as radius = 2  
`2 tokens to the left + current token + 2 tokens to the right`  

So the maximum attended tokens are:  
`2 + 1 + 2 = 5`  

# Attention Runtime Breakdown  
## Main readout  
**Sliding-window attention runtime is nearly flat as sequence length grows.**  
That is exactly the behavior we wanted.  

In full attention, the expensive parts are:  
`QK^T scores     ~ O(N²D)`  
`softmax         ~ O(N²)`  
`weights @ V     ~ O(N²D)`  

But with window=2, each token attends to only about 5 positions:  
`left 2 + self + right 2 = 5`  

So the attention part becomes closer to:  
`O(N × 5 × D)`  

For fixed window size, that is effectively:  
`O(ND)`  

instead of:  
`O(N²D)`  

Note: The chart is not showing per-token cost explicitly. The script times the whole tensor operation for all tokens.  
So the measured runtime is the wall-clock runtime for all tokens, not runtime / token. But the theoretical work is absolutely not constant. It grows linearly with sequence length.  

For radius window size `r = 2`, each token attends to at most:  
`2r + 1 = 5 tokens`  

So total score dot-products are approximately:  
`N × 5`  

Each score is a dot product of dimension D, so total QK work is:  
`O(N × window × D)`  

With fixed window size:  
`O(ND)`  
not constant  

The reason the plotted runtime looks almost flat is because this is GPU wall-clock time, **not total mathematical work**.  
On an A100, for these small local windows, the GPU can process many token windows in parallel.  
So as N increases from 64 to 8192, the GPU is probably still not saturated by this tiny operation.  
Kernel launch overhead, memory access overhead, and vectorized parallel execution can dominate, making the elapsed time look nearly flat.  

**A more precise interpretation is:**  
The measured wall-clock time for the windowed QK step is nearly flat over this tested range, even though the total mathematical work grows linearly with sequence length.  
Sliding-window attention removes the quadratic N² term. 

`The total attention work grows roughly linearly with N, but on this A100 benchmark the wall-clock time remains nearly flat because the operation is small and highly parallel.`  

## Nore on what the chart shows  
The windowed QK median line is almost flat, around roughly:  
`~0.14 ms`  
from small sequence lengths all the way to 8192.  

The windowed softmax median is also tiny and mostly flat, around:  
`~0.02–0.03 ms`  

The windowed weights @ V cost rises somewhat, but still stays small.  

### Total runtime behavior  
The total median stays roughly in this band:  
`~0.35 ms to ~0.45 ms`  

This means the benchmark is now dominated less by attention math and more by fixed overheads and projection costs:  
`Q = X @ Wq`  
`K = X @ Wk`  
`V = X @ Wv`  

