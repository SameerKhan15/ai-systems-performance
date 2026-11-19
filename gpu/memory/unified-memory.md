# CUDA Unified Memory — First Principles and Performance Engineering

## 1. Central Idea

**CUDA Unified Memory provides a unified address space, not a guarantee that data is physically present in every processor's local memory.**

With a managed allocation, CPU and GPU code can use the same pointer. CUDA manages accessibility, page mappings, and—when needed—movement of pages between physical memories.

```cpp
float* data = nullptr;
cudaMallocManaged(&data, bytes);

// CPU initializes data.
for (size_t i = 0; i < count; ++i) data[i] = 1.0f;

// GPU uses the same pointer.
kernel<<<grid, block>>>(data);
cudaDeviceSynchronize();

cudaFree(data);
```

This simplifies programming compared with separate host/device allocations and explicit transfers, but **does not remove the physical cost of data movement**.

### Traditional device memory versus managed memory

| Traditional explicit allocation/copy | CUDA managed memory |
|---|---|
| Separate host and device allocations | A managed allocation with a pointer usable by supported processors |
| Programmer orchestrates `cudaMemcpy` transfers | Runtime can migrate or remotely map memory as supported |
| Transfer cost is explicit | Transfer and fault-handling costs can be less obvious |
| Programmer has direct placement control | Runtime makes placement decisions, influenced by hints and prefetch |

## 2. Virtual Address versus Physical Residency

Consider a managed allocation comprising pages A, B, C, and D:

```text
             One managed virtual address range
                          |
                 A   B   C   D
                          |
         +----------------+----------------+
         |                                 |
     CPU memory                        GPU HBM
     A  B  C  D                        (none yet)
```

If the CPU initializes all pages and the GPU first accesses an element in page B, a migration-capable system may need to:

1. Detect that the GPU cannot currently access the page locally.
2. Service a GPU page fault and arrange access.
3. Migrate the page to GPU memory (if migration is the selected mechanism).
4. Update the mappings and resume affected GPU execution.

**Hardware nuance:** Managed pages do not invariably migrate on first access. Some platforms support direct access to remote memory and hardware-coherent sharing. Their behavior depends on GPU, host, interconnect, CUDA version, and allocation/access policy. Grace Hopper/Blackwell systems differ significantly from conventional discrete PCIe GPUs.

## 3. Why On-Demand Migration Can Be Expensive

A GPU kernel can be interrupted while the runtime resolves page faults and transfers data. A useful conceptual performance model is:

$$
T_{\text{observed}} \approx T_{\text{compute}} + T_{\text{nonoverlapped fault/migration overhead}}
$$

The overhead may include fault processing, mapping changes, interconnect transfer, and execution stalls. **It is not necessarily a simple sum of every individual fault's latency**, since faults and migrations can be batched or overlap.

### Transfer amplification

Suppose a kernel reads one 4-byte float from each of 100 different pages, and each page is 4 KiB. If the runtime migrates each complete page:

$$
\text{Useful bytes}=100\times4=400\text{ bytes}
$$

$$
\text{Transferred bytes}=100\times4096=409{,}600\text{ bytes}=400\text{ KiB}
$$

$$
\text{Transfer amplification}=\frac{409{,}600}{400}=1024\times
$$

This is an illustrative calculation. Actual page granularity, batching, prefetch heuristics, and remote-memory access can change transfer volumes. Sparse/irregular accesses make such amplification especially relevant.

## 4. Prefetching: Move or Prepare Data Before First Use

`cudaMemPrefetchAsync()` requests proactive placement/access preparation for a managed-memory range. The exact API signature and supported targets vary by CUDA version; the following uses the familiar device-ID form:

```cpp
cudaMemPrefetchAsync(data, bytes, gpuId, stream);
kernel<<<grid, block, 0, stream>>>(data);
```

Operations in **the same stream are ordered**, so the kernel starts after the preceding prefetch operation in that stream has completed. Prefetch reduces first-touch surprises and may enable overlapping data movement with unrelated work.

### Prefetch does not eliminate transfer time

If a transfer takes 10 ms, computation takes 20 ms, and there is no overlap:

$$
T_{\text{end-to-end}}\approx10+20=30\text{ ms}
$$

The kernel itself can appear faster after prefetching because the migration was moved outside its timed interval. **Measure end-to-end latency as well as kernel duration.**

### Overlap with useful computation

If processing buffer A takes 15 ms while an independent prefetch of buffer B takes 25 ms, and they overlap perfectly:

$$
T_{\text{combined}}\approx\max(15,25)=25\text{ ms}
$$

Without overlap, it would be 40 ms. In practice, bandwidth contention, memory capacity, GPU concurrency capabilities, and dependencies can limit overlap.

```text
Stream A: [ compute on buffer A ---------------- ]
Stream B: [ prefetch next buffer B ---------------------- ]
          ^ concurrent start
```

Double buffering is a natural pattern: process batch N while preparing/prefetching batch N+1, then wait for the prefetch before consuming the next buffer.

## 5. Memory Advice: Three Different Questions

`cudaMemAdvise()` conveys expected access patterns. **Advice is not a hard residency guarantee.** The examples below show the conceptual API forms used in the lesson; check the installed CUDA version for supported signatures and device/location identifiers.

### A. Preferred location: Where should this data usually live?

```cpp
cudaMemAdvise(data, bytes,
              cudaMemAdviseSetPreferredLocation, gpuId);
```

This communicates a placement preference; **it does not immediately migrate all pages**. If the GPU accesses nonresident pages, faults or remote accesses may still occur. The runtime can override the preference due to memory pressure and other constraints.

### B. Read mostly: Would multiple read copies help?

```cpp
cudaMemAdvise(data, bytes,
              cudaMemAdviseSetReadMostly, gpuId);
```

On supported systems, largely read-only pages can be duplicated so multiple processors read local copies. Writes require the runtime to preserve correct visibility through coherence/invalidation and applicable synchronization rules. Duplication is an optimization opportunity, not a guarantee.

Potential use: weights or lookup tables read repeatedly by more than one GPU and rarely modified.

### C. Accessed by: Which processor needs to reach the allocation?

```cpp
cudaMemAdvise(data, bytes,
              cudaMemAdviseSetAccessedBy, otherGpuId);
```

This can prepare mappings so another GPU can access the data without incurring an access-time mapping fault. **It does not mean the allocation is copied to that GPU.** On a peer-access-capable system, access may occur remotely over the interconnect.

Potential use: GPU 1 occasionally reads a few elements of a frequently updated allocation primarily resident on GPU 0. Remote reads may be preferable to repeatedly migrating a large buffer.

### Comparison

| Mechanism | Main concern | What it does **not** guarantee |
|---|---|---|
| `PreferredLocation` | Preferred physical residency | Immediate migration or permanent pinning |
| `ReadMostly` | Read duplication opportunities | A replica on each processor |
| `AccessedBy` | Processor mappings/accessibility | Local memory placement or duplication |
| `cudaMemPrefetchAsync` | Proactive placement/access preparation | Free transfers or automatic overlap |

## 6. Stream Attachment: Accessibility and Synchronization

```cpp
cudaStreamAttachMemAsync(streamA, data, 0, cudaMemAttachSingle);
```

`cudaMemAttachSingle` associates a managed allocation with a particular stream under the stream-specific managed-memory access model. It is **not a data transfer**, and **streams do not own separate physical memory**.

Why it exists: On older GPUs without full concurrent managed-memory access, the CPU might otherwise need to wait for unrelated GPU activity before safely accessing managed memory. A single-stream association can reduce the scope of the access restriction to the associated stream.

Example:

- Stream A uses buffer A for 10 ms.
- Stream B uses an independent buffer B for 30 ms.
- Buffer A is associated with Stream A.
- With appropriate synchronization and supported hardware, the CPU may access buffer A once Stream A completes, without waiting for unrelated Stream B work.

**Caveats:** The attachment is stream-ordered, the programmer must obey the access restrictions, and modern GPUs with `concurrentManagedAccess` have more flexible behavior. Stream attachment alone does not prefetch pages or guarantee asynchronous migration overlap.

## 7. A Performance Engineer's Decision Framework

Ask these questions, in order:

1. **Placement:** Where are the pages now, and where will they be used most often?
2. **Access pattern:** Sequential, dense, sparse, irregular, read-mostly, or frequently written?
3. **Movement:** Will accesses migrate pages, use remote mappings, or hit local HBM?
4. **Timing:** Can migration be moved before the critical kernel and overlapped with useful work?
5. **Concurrency:** Are there ordering or managed-memory access restrictions between CPU, GPUs, and streams?
6. **Measurement:** Does optimization improve *end-to-end* performance, not merely kernel time?

### Applying this to an inference pipeline

For an inference system with 8 GB of mostly immutable weights and 2 GB of input tensors per batch:

- Prefer to keep frequently reused weights in GPU HBM, assuming sufficient capacity.
- Prefetch or explicitly copy upcoming inputs before they are consumed.
- Double-buffer independent batches and overlap transfer with compute where supported.
- Consider `ReadMostly` if multiple processors repeatedly read the same data.
- Benchmark managed-memory prefetch **against** `cudaMemcpyAsync()` with pinned host memory. Managed memory is not inherently faster.

A rough idealized steady-state pipeline interval is:

$$
T_{\text{batch interval}}\approx\max(T_{\text{transfer}},T_{\text{compute}})
$$

provided that input preparation, transfers, and computation can be pipelined without a bottleneck elsewhere.

## 8. Proposed Hands-On Lab — A100 on RunPod

**Platform:** NVIDIA A100 80 GB on RunPod, ideally with an available CUDA compiler and Nsight Systems. A discrete A100 is suitable for observing Unified Memory behavior, but is not a Grace Hopper NVLink-C2C configuration.

**We do not need Nsight Compute (`ncu`) or access to restricted hardware performance counters for the initial experiment.** CUDA timing alone can demonstrate the essential effects.

### Environment checks

```bash
nvidia-smi
nvcc --version
nsys --version
nsys status -e
```

Some RunPod images may require installing `nvcc` or `nsys`; availability of Unified Memory tracing depends on host/driver/container permissions.

### Experimental treatments

Use the same managed allocation size (start with **1 GiB**) and the same memory-reading kernel in each case:

| Case | Procedure | Primary question |
|---|---|---|
| A — On-demand | Initialize on CPU, launch GPU kernel without prefetch | What is first-touch cost? |
| B — Prefetched | Initialize on CPU, prefetch to GPU, then launch kernel | How does cost split between prefetch and kernel? |
| C — Warm | Run again when pages are already GPU-resident | What is steady-state execution cost? |

Optional later case: alternate CPU and GPU accesses to observe ping-pong behavior.

### Record separately

- CPU wall-clock time for full operation, including synchronization.
- Prefetch duration (if applicable).
- Kernel duration using CUDA events, with correct stream/event ordering.
- First-run versus warmed-up timing.
- When supported, Nsight Systems Unified Memory fault and migration events.

**Measurement hygiene:** Warm up the CUDA context, repeat trials, control allocation size and access pattern, synchronize correctly, and ensure the compiler cannot optimize away the reads. Because the managed-memory residency state changes between trials, reinitialize or deliberately reset it so comparisons are fair. Avoid accidentally including setup in one treatment but not another.

### Expected learning outcomes

1. On-demand migrations can lengthen the first kernel execution.
2. Prefetch can reduce kernel-visible stalls while leaving transfer cost in the full request path.
3. Warm execution can be faster because pages are already accessible locally.
4. Sparse reads may cause disproportionate page traffic.
5. Whether managed-memory prefetch is competitive with explicit copies is a **measurement question**, not an assumption.

## 9. Core Takeaways

- **Unified address space ≠ unified physical residency.**
- **Page faults can hide expensive data movement inside GPU execution.**
- **Prefetch moves the work earlier; overlap is what can hide its latency.**
- **`PreferredLocation`, `ReadMostly`, and `AccessedBy` solve distinct problems.**
- **Stream attachment controls access/synchronization relationships, not physical placement.**
- **Optimize and measure end-to-end latency, transfer volume, and GPU stalls together.**

## Further Reading

- NVIDIA CUDA C++ Programming Guide — Unified Memory: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- NVIDIA CUDA Runtime API — Memory Management: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html
- NVIDIA Nsight Systems Documentation: https://docs.nvidia.com/nsight-systems/
