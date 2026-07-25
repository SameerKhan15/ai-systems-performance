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
