**When one can look at a new AI workload and predict, "this should become memory-bandwidth-bound here; this tensor will scale quadratically; this kernel is probably suffering from poor occupancy; this distributed configuration will hit communication overhead"**  
**—and then experimentally verify it—you've moved well beyond knowing AI concepts into AI systems mastery.**  

For GPU AI workloads, the best stack is usually **Nsight Systems + Nsight Compute**, with each answering a different question.  

Use **Nsight Systems** first to find where the time is going. It tells you whether execution is dominated by GPU kernels, memory copies, synchronization, CPU gaps, launch overhead, communication, etc. It is your timeline-level profiler.  

Then use Nsight Compute on the suspicious kernels to answer why. For identifying a memory-bound kernel, the most useful signals are:  
* DRAM throughput / % of peak  
* L2 throughput / cache hit rate  
* SM utilization  
* Tensor Core / FP32 utilization  
* memory pipe utilization  
* achieved occupancy  
* warp stall reasons, especially stalls related to memory dependencies  
* bytes moved vs FLOPs executed  

The core diagnosis is basically a **Roofline** question:  
$
\text{Arithmetic Intensity}
=
\frac{\text{FLOPs}}{\text{Bytes transferred}}
$
If arithmetic intensity is low and the kernel is already consuming a large fraction of available memory bandwidth, then adding more compute units will not help much—the workload is memory-bandwidth bound.  

A very typical pattern looks like this:  
* DRAM bandwidth = 80–95% of achievable peak  
* SM/Tensor utilization = relatively modest  
* lots of memory-related warp stalls  
* low arithmetic intensity  
* runtime barely improves when compute capability increases  

Overall Observability stack:  
1. **Nsight Systems** — system-level bottleneck discovery  
2. **Nsight Compute** — kernel-level diagnosis  
3. **Nsight Compute Roofline analysis** — compute-bound vs memory-bound classification  
4. **PyTorch Profiler** — model/operator attribution  
5. **DCGM / nvidia-smi metrics** — fleet-level GPU telemetry  
6. **CUPTI** — when you eventually want to build custom observability  

PyTorch Profiler tells you which operator is expensive. Nsight Systems maps it into kernel execution. Nsight Compute tells you whether that kernel is constrained by compute, memory bandwidth, latency, occupancy, or instruction dependencies.  

# Lab1: Vector Addition  
## Step 1 — What we're going to measure  
Our kernel is:  
$C_i = A_i + B_i$
We want to derive three quantities:  
`Arithmetic Intensity, Achieved Bandwidth, Achieved Performance`  

From those, we'll determine whether the A100 is memory-bound.  
For each FP32 element:  
`FLOPs=1`  
and approximately:  
`Bytes = 4 + 4 + 4 = 12`  

Therefore:  
$$
AI = \frac{1}{12} = 0.0833\ \text{FLOP/byte}
$$
That's our theoretical arithmetic intensity.  

## Step 2 — vector_add.py  

