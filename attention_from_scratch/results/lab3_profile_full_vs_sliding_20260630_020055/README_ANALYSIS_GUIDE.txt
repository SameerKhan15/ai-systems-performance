Lab#3 full vs sliding-window profiling guide

Output root:
  results/lab3_profile_full_vs_sliding_20260630_020055

Pass 1: Normal benchmark + counters
  Directory:
    results/lab3_profile_full_vs_sliding_20260630_020055/benchmark

  Key files:
    comparison_metrics.csv
    comparison_total_runtime.png
    comparison_attention_only_runtime.png
    comparison_dominant_attention_costs.png
    comparison_memory_scaling_log.png
    comparison_score_entries_log.png
    run.log

  What to inspect:
    - full valid_score_entries grows as N^2
    - sliding valid_score_entries grows approximately as N * (2 * radius + 1)
    - full score_tensor_mb grows as N^2
    - sliding score_tensor_mb grows approximately linearly
    - runtime may not perfectly match operation count because GPU parallelism and kernel launch overhead matter

Pass 2: Nsight Systems
  Directory:
    results/lab3_profile_full_vs_sliding_20260630_020055/nsys

  Key files:
    compare_full_vs_sliding_nsys.nsys-rep
    nsys_stats.txt

  What to inspect:
    - wall-clock runtime by NVTX range
    - kernel launch overhead and gaps
    - tiny kernels vs long kernels
    - whether full attention kernels grow with sequence length
    - whether sliding attention kernels stay short
    - whether GPU work is serialized or overlapping

Pass 3: Nsight Compute
  Directory:
    results/lab3_profile_full_vs_sliding_20260630_020055/ncu

  Key files:
    compare_full_vs_sliding_seq8192_full.ncu-rep
    ncu_console_full.txt
    ncu_list_sets.txt

  What to inspect:
    - SM utilization / throughput
    - achieved occupancy
    - memory throughput
    - whether kernels are compute-bound, memory-bound, or launch/overhead dominated
    - compare full NxN kernels to sliding NxW kernels

Suggested first interpretation:
  Full attention:
    score entries = N^2
    score tensor memory = O(N^2)
    QK and weights@V should eventually dominate

  Sliding-window attention:
    score entries ≈ N * (2 * radius + 1)
    score tensor memory = O(N * window)
    wall-clock may look almost flat on A100 for small windows because the GPU can process the local windows in parallel

