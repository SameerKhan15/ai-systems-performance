**KV Cache is probably the single most important inference optimization to understand before studying modern LLM architectures.**  

Learning Order:  
1. Why KV cache exists  
2. How autoregressive generation works  
3. Memory cost of KV cache  
4. Why KV cache becomes a bottleneck  
5. How Multiheaded-Latent-Attention (MLA) solves it  

# Why do we need KV cache?  
Imagine we've already processed `The cat sat on the` and now we want to predict the next token.  

The transformer has already computed  
`K1 V1`  
`K2 V2`  
`K3 V3`  
`K4 V4`  
`K5 V5`  
for those five tokens.  

Now suppose the next token becomes `mat`. Without KV cache, we'd run the transformer on:  
`The`  
`The cat`  
`The cat sat`  
`The cat sat on`  
`The cat sat on the`  
`The cat sat on the mat`  
over and over. Every previous Key and Value would be recomputed every generation step. That is enormously wasteful.  

Instead we do:  
* Step1: Compute `K1 V1` and store them  
* Step2: Compute `K2 V2` and store them  
* Step3: Compute `K3 V3` and store them  
* Step3: Compute `K4 V4` and store them  
* Step3: Compute `K5 V5` and store them  
* Step3: Compute `K6 V6` and store them  

Eventually we have:  
KV Cache
`K1 V1`  
`K2 V2`  
`K3 V3`  
`K4 V4`  
`K5 V5`  
`K6 V6`  

When token 7 arrives, we compute K6, V6 and Q6, and simply attend against:  
`K1,V1`  
`K2,V2`  
`K3,V3`  
`K4,V4`  
`K5,V5`  
No previous Keys or Values are recomputed. **That is the whole idea of KV caching.**  

## Why don't we cache Q?  
Because Queries are only used once. At generation step for token 6, we compute Q6, perform attention and then throw Q6 away. Next step we compute Q7 instead.  
Queries never need to be reused. Keys and Values do.  

# What gets stored?  
Suppose:   
`Heads = 32`
`Head dimension = 128`  
Every generated token stores  
For each head:  
`Key  = 128 numbers`  
`Value =128 numbers`  
So 256 numbers/head/token. Across 32 heads 8192 numbers per token. Now imagine 32,000 tokens. This would be 32,000 * 8192 = 262M numbers. The cache becomes enormous.  

# Why KV cache dominates inference  
During training, most memory goes toward:  
* activations  
* gradients  

During inference:  
* There are no gradients  
* There are almost no activations  

The dominant memory consumer becomes KV Cache. For long-context models (32k–1M tokens), the KV cache can consume tens or even hundreds of GB depending on the model size and precision.  
That is why almost every modern inference optimization is about reducing KV cache size.  

# Traditional Multi-Head Attention  
Each head independently creates `Q_i, K_i, V_i` for head i.  

So we end up storing  
**Head 1**  
`K1`  
`V1`
**Head 2**
`K2`  
`V2`
...

**Head 32**
`K32`  
`V32`  
Every token stores all of these.  

# MLA intuition  
The key insight is: Do we really need to store every full-sized Key and Value?  
DeepSeek's answer was NO.  
Instead of storing Full K, Full V, store a compressed representation. Think of it like image compression. Instead of storing 4000 numbers, store 256 numbers that preserve most of the important information.  
This compressed vector is called the latent KV. During attention, the model reconstructs what it needs from this latent representation. So instead of caching K,V, it caches Latent KV, which is much smaller.  
"Latent" means a compressed internal representation. It's similar to an autoencoder:  
`Large vector`
     ↓
 `Encoder`
     ↓
`Small latent vector`
     ↓
 `Decoder`  
     ↓
`Approximate original vector`  
MLA follows this general idea, although the exact mathematical implementation differs from a classic autoencoder.  

## Why is RoPE tricky?  
This is the subtle part. Normally  
`RoPE`  
 ↓
`K`
`Q`  

If we applied RoPE directly to the cached keys, the cached representation would become tied to positional transformations in a way that defeats the purpose of storing a reusable compressed latent.  
Instead, DeepSeek separates the information into `Content information + Position information`.  The latent cache stores mostly content. Position is handled by a much smaller, separate component that can be recomputed efficiently.  
This lets the large cached latent remain reusable while still allowing attention to incorporate positional information.  

## Visual picture  
Standard Multi-Head Attention  
`Input`  
   ↓
  `Q`  
  `K`  
  `V`  
   ↓
 `Store`
  `K`  
  `V`  

MLA  
`Input`  
   ↓  
`Compress`  
   ↓
`Latent KV`  
   ↓  
`Store`
`Latent KV`  

Later  
`Latent KV`
    ↓
`Expand`
    ↓
`Approximate K`  
`Approximate V`
    ↓  
`Attention`  
Much less memory is stored.  

## Why is MLA such a big deal?  
Modern large-context models are increasingly limited by memory bandwidth rather than raw compute during inference.   
Reducing the amount of KV data that must be stored and read can significantly improve serving efficiency, especially for long prompts and many concurrent users.  

# Labs Structure
* Lab 1: Build KV caching from scratch for a tiny transformer  
* Lab 2: Measure KV cache memory growth as sequence length increases  
* Lab 3: Compare standard Multi-Head Attention with Multi-Query Attention and Grouped-Query Attention to see how sharing keys/values reduces cache size  
* Lab 4: Implement a simplified version of MLA using low-rank projections to understand the compression idea  
Profile memory usage and runtime on a GPU to quantify the tradeoffs  




 














