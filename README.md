# Repository on learnings and experiments on AI Systems Performance Engineering
## Lab#3: Sliding Window Attention  
We already have:  
`scores = Q @ K.T`  
where every token attends to every other token.  

Now change it so each token only attends to neighbors.  
Example:  
Window = 2

The   → attends to [The, cat]
cat   → attends to [The, cat, sat]
sat   → attends to [cat, sat, down]
down  → attends to [sat, down]

Then rerun exactly the same benchmark.  
Collect:  
* median runtime  
* p95 runtime  
* memory  
* plots  

Compare directly against full attention.  
Answer:  
* How much faster?  
* How much memory saved?  
* Does runtime become linear?  
* What information is lost?  

## Lab#4: Vary the Window Size  
Keep the implementation identical.  
Only change:  
`window = 16`
`window = 32`
`window = 64`
`window = 128`
`window = 256`

Now plot:  
window size
      ↓
runtime

and  
window size
      ↓
memory

See the tradeoff between context and cost. 

## Lab#5: Multi-Head Attention  
Right now we have 1 head. Change to:  
`2 heads`
`4 heads`
`8 heads`
`16 heads`

Measure:  
* runtime  
* GPU memory  
* projection costs  

Understand why multiple heads exist.

## Lab#6: Grouped Query Attention (GQA)  
Instead of:  
`8 Query Heads`
`8 Key Heads`
`8 Value Heads`

experiment with  
`8 Query Heads`
`2 Key Heads`
`2 Value Heads`

Measure the difference.  
This leads directly into modern LLM inference.  

## Lab#7: KV Cache  
The lab should answer: "Why does inference become so much faster?"  
Measure  
`No KV Cache`
`vs`
`KV Cache` 

## Lab#8: FlashAttention  
Appreciate
* tiling
* SRAM
* memory traffic
* avoiding materializing QKᵀ

## Lab#9: Profilers (and compare multiple algorithms) 
* `torch.profiler`
* `nsys`
* `Roofline`
* `ncu`

## Repository Structure  
`01_rope_geometry/`
`02_full_attention_profile/`
`03_sliding_window_attention/`
`04_window_size_scaling/`
`05_multi_head_attention/`
`06_grouped_query_attention/`
`07_kv_cache/`
`08_flashattention/`
`09_torch_profiler/`
`10_nsys/`
`11_roofline/`
`12_ncu/`