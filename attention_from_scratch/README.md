# Transformer Self-Attention Runtime Benchmarking  
![](png/1.png "This is a sample image.")  

## Description of Key, Query, Value Vectors  
Query: Represents the current token in a sentence the model focuses on, or tries to understand.  
The query is used to probe the other parts of the input sentence to determine how much attention to pay to them.
Key: Each token has an associated key. These keys are used to match the query.
Value: Encodes actual content or representation of a token. Once the model determines which keys are most relevant 
to the query, it retrieves the corresponding values.  

E.g. sequence: "The cat sat down"  
d=4, where d is the token embedding dimension  

Let X be the input embedding matrix:  
Token, d0, d1, d2, d3  
=======================  
The, 1.0, 0.2, 0.1, 0.5  
cat, 0.3, 1.2, 0.7, 0.1  
sat, 0.5, 0.4, 1.1, 0.9  
down, 0.8, 0.6, 0.2, 1.0  

X ∈ R^4×4  
4 tokens x 4 dimensions  

### What "Q Projection" means  
Q = X @ Wq  => Q = X Wq  
Wq ∈ R^4×4  

Note: Wq is a learned Query projection matrix.  
So, (4x4)(4x4) = 4x4  

The output Q has one query vector per token.  
e.g. 
Wq = [0.1 0.2 0.0 0.3  
      0.0 0.1 0.4 0.2  
      0.3 0.0 0.2 0.1  
      0.2 0.3 0.1 0.0]  

Compute query for "The":  
X_The = [1.0, 0.2, 0.1, 0.5]  
Q_The = X_The * Wq  
(1x4) (4x4) = (1x4)  

Q_The = [0.23, 0.37, 0.15, 0.35]  

Q projection = token_embeddings -> query_projection_matrix -> query_vectors
Same idea applies to K and V projections.  

K = X @ Wk  
V = X @ Wv  

Note: Wq is the same across ALL tokens. This is a fundamental property of Transformers.  The same Wq matrix is applied  
to every token in the sequence.  

### Analogy  
Imagine "The", "cat", "sat", "down" are 4 photographs.  And Wq is a camera filter.  You don't apply a new filter to every picture. You apply 
the same filter to every picture.  The input differs so the output differs.  

So mathematically, we compute:  
`The x Wq`, `cat x Wq`, `sat x Wq`, `down x Wq` using one shared learned matrix.  Weight sharing is one of the key ideas behind transformers.  

## The Big Picture  
For one transformer layer, there are three learned projection matrices:  
* Wq -> creates Queries  
* Wk -> creates Keys  
* Wv -> creates Values  

They are shared across every token in the sequence. This is why PyTorch code can compute:  
Q = X @ Wq in one matrix multiplication  
It is simultaneously applying the same Wq to every row of X, producing one query vector per token.  

This is exactly why matrix multiplication is so efficient on GPUs. 
Instead of looping:  
`for token in tokens: q = token @ Wq`  
PyTorch performs one large batched operation `Q = X @ Wq`  
GPUs can execute this extremely efficiently using highly optimized linear algebra kernels.  

## Profiling Stack  
`profile_lab3_full_vs_sliding.sh`  
Put it next to:  
`compare_full_vs_sliding_attention.py`  

Then run:  
`chmod +x profile_lab3_full_vs_sliding.sh`  
`./profile_lab3_full_vs_sliding.sh`  

### What this profiling package does  
#### Pass 1 — Script counters and benchmark plots  
This runs the comparison script normally and collects:  
`full attention score entries      = N²`  
`sliding attention score entries   ≈ N × local_window_width`  
`full score tensor memory          = O(N²)`  
`sliding score tensor memory       = O(N × window)`  
`median runtime`  
`p95 runtime`   
`peak GPU memory`  
**This is where to prove the math shape.**  

#### Pass 2 — Nsight Systems  
This runs:  
`nsys profile --trace=cuda,nvtx,osrt ...`  

The comparison script already has NVTX labels, so the Nsight Systems timeline should show readable ranges like:  
`full/seq_len=8192/scores_NxN`  
`full/seq_len=8192/softmax_NxN`  
`full/seq_len=8192/weights_at_V_NxN`  

`sliding_r2/seq_len=8192/scores_Nx5`  
`sliding_r2/seq_len=8192/softmax_Nx5`  
`sliding_r2/seq_len=8192/weights_at_V_Nx5`  

Nsight Systems is the right tool for timeline-level questions: wall-clock growth, tiny kernels, launch overhead, CPU/GPU gaps, and overlap. 
NVIDIA’s docs describe NVTX ranges as visible in the timeline and projected onto GPU activity, which is exactly why we added those labels.  

#### Pass 3 — Nsight Compute  
This runs one representative sequence length under ncu, defaulting to:  
NCU_SEQ_LEN=8192  
NCU_SET=full  

Nsight Compute is the right tool for kernel-level questions: SM throughput, occupancy, memory throughput, and whether kernels are compute-bound or memory-bound.  

Best analysis order:  
* comparison_score_entries_log.png  
* comparison_attention_only_runtime.png  
* comparison_memory_scaling_log.png  
* Nsight Systems screenshot around seq_len=8192  
* Nsight Compute console summary for seq_len=8192  

What we are trying to prove (for Lab#3 specifically):  
Full attention grows in score entries and score memory as N².  
Sliding-window attention grows as N × window. Runtime may not perfectly follow operation count on an A100 because small local-window kernels   
can be highly parallel and launch/overhead dominated, so we use script counters for algorithmic complexity, Nsight Systems for timeline behavior,  
and Nsight Compute for kernel saturation.  

### Before Profiling execute the following    
`which nsys`  
`nsys --version`  

`which ncu`  
`ncu --version`  

`python3 - << 'EOF'    
import torch    
print("cuda_available:", torch.cuda.is_available())  
print("gpu:", torch.cuda.get_device_name(0))  
print("torch:", torch.__version__)  
print("cuda:", torch.version.cuda)  
EOF`  
