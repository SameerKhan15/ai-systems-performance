# Goal of Lab 1  
At the end of the lab, we should be able to answer these questions without hesitation:  
* Why is KV caching needed?  
* Why do we cache K and V but not Q?  
* What exactly is stored?  
* How much computation is avoided?  
* How much memory is consumed?  
* Why does inference become memory-bound?  
* What happens at every generation step?  

# Part 1 – Build the Smallest Possible Transformer  
`embed_dim = 16`  
`num_heads = 2`  
`head_dim = 8`  
`seq_len = 8`  
No feed-forward network. No layer normalization.  

Just  
   `Input`  
      ↓  
  `Linear`  
      ↓  
   `Q K V`  
      ↓  
  `Attention`  
      ↓  
   `Output`  
The purpose is to isolate attention.  

# Part 2 – Simulate Autoregressive Generation  
Suppose the sentence is  
`The`  
`cat`  
`sat`  
`on`  
`the`  
`mat`  
Instead of processing all six tokens simultaneously, simulate generation one token at a time. 

Iteration 1: The  
Iteration 2: cat  
...
Exactly how a real LLM works.  

# Part 3 – Version A (No Cache)  
Every iteration recomputes everything.  

Step 5  
`The cat sat on the`  
computes  
`Q1 K1 V1`  
`Q2 K2 V2`  
`Q3 K3 V3`  
`Q4 K4 V4`  
`Q5 K5 V5`  

Step 6  
`The cat sat on the mat`  
again computes  
`Q1 K1 V1`  
`Q2 K2 V2`  
`Q3 K3 V3`  
`Q4 K4 V4`  
`Q5 K5 V5`  
`Q6 K6 V6`  
This immediately makes the waste visible.  

# Part 4 – Version B (KV Cache)  
Now keep  
`K_cache`  
`V_cache`  

Iteration 1  
Compute  
`K1`  
`V1`  
Append  

Iteration 2  
Only compute  
`K2`  
`V2`
Append  

Eventually  
K_cache  
`K1`  
`K2`  
`K3`  
`K4`  
`K5`  

Then  
`Q6 × K_cache`  -- THIS IS THE KEY MOMENT  

# Instrument Everything  
## Maintain Counters  
Without cache  
Linear projections  
`Q projections`  
  
`K projections`  
  
`V projections`  

`Attention matmuls`  

With cache  
Count again and compare.  

## Visualize Cache Growth  
After every N generation steps, plot K,V cache size against the 'number of generation steps elapsed'  
Do this for WITHOUT and WITH cache and correlate it with Memory Accounting.  

## Timing  
Measure  
`Time without cache`  
`Time with cache`  










