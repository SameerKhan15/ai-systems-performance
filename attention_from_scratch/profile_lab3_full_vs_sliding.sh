#!/usr/bin/env bash
set -euo pipefail

# profile_lab3_full_vs_sliding.sh
#
# End-to-end profiling harness for Lab#3:
#   1. Normal benchmark + script counters
#   2. Nsight Systems timeline profile
#   3. Nsight Compute kernel-level profile
#
# Assumes this file is in the same directory as:
#   compare_full_vs_sliding_attention.py
#
# Default:
#   WINDOW_RADIUS=2
#   NCU_SEQ_LEN=8192
#
# For the original sentence example behavior:
#   WINDOW_RADIUS=1 ./profile_lab3_full_vs_sliding.sh
#
# For a quicker test:
#   BENCH_RUNS=5 NSYS_RUNS=3 NCU_SET=basic ./profile_lab3_full_vs_sliding.sh

WINDOW_RADIUS="${WINDOW_RADIUS:-2}"
EMBED_DIM="${EMBED_DIM:-128}"
SEQ_LENS="${SEQ_LENS:-64,128,256,512,1024,2048,4096,8192}"

BENCH_RUNS="${BENCH_RUNS:-30}"
NSYS_RUNS="${NSYS_RUNS:-5}"
NCU_RUNS="${NCU_RUNS:-1}"
NCU_SEQ_LEN="${NCU_SEQ_LEN:-8192}"
NCU_SET="${NCU_SET:-full}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PY_SCRIPT="${PY_SCRIPT:-compare_full_vs_sliding_attention.py}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-results/lab3_profile_full_vs_sliding_${STAMP}}"

mkdir -p "${OUT_ROOT}"

log() {
  echo
  echo "================================================================================"
  echo "$1"
  echo "================================================================================"
}

find_tool() {
  local tool="$1"

  if command -v "${tool}" >/dev/null 2>&1; then
    command -v "${tool}"
    return 0
  fi

  find /opt /usr/local -name "${tool}" -type f 2>/dev/null | sort -V | tail -n 1
}

ensure_symlink_if_found() {
  local tool="$1"
  local found

  found="$(find_tool "${tool}" || true)"
  if [[ -n "${found}" ]]; then
    if ! command -v "${tool}" >/dev/null 2>&1; then
      ln -sf "${found}" "/usr/local/bin/${tool}" 2>/dev/null || true
    fi
    command -v "${tool}" || echo "${found}"
    return 0
  fi

  return 1
}

check_inputs() {
  log "Checking inputs"

  if [[ ! -f "${PY_SCRIPT}" ]]; then
    echo "ERROR: ${PY_SCRIPT} not found in current directory: $(pwd)"
    echo "Put this profiling script next to compare_full_vs_sliding_attention.py."
    exit 1
  fi

  echo "Python script: ${PY_SCRIPT}"
  echo "Output root  : ${OUT_ROOT}"
  echo "Window radius: ${WINDOW_RADIUS}"
  echo "Embed dim    : ${EMBED_DIM}"
  echo "Seq lens     : ${SEQ_LENS}"
  echo "NCU seq len  : ${NCU_SEQ_LEN}"
}

run_benchmark_with_counters() {
  log "Pass 1: Normal benchmark with script counters"

  mkdir -p "${OUT_ROOT}/benchmark"

  "${PYTHON_BIN}" "${PY_SCRIPT}" \
    --seq-lens "${SEQ_LENS}" \
    --embed-dim "${EMBED_DIM}" \
    --window-radius "${WINDOW_RADIUS}" \
    --num-runs "${BENCH_RUNS}" \
    --warmup-runs 5 \
    --output-dir "${OUT_ROOT}/benchmark"

  echo
  echo "Benchmark outputs:"
  find "${OUT_ROOT}/benchmark" -maxdepth 1 -type f | sort
}

run_nsys_profile() {
  log "Pass 2: Nsight Systems timeline profile"

  if ! ensure_symlink_if_found nsys >/dev/null 2>&1; then
    echo "WARNING: nsys not found. Skipping Nsight Systems pass."
    return 0
  fi

  mkdir -p "${OUT_ROOT}/nsys"

  echo "nsys path:"
  command -v nsys
  echo
  nsys --version || true

  nsys profile \
    --trace=cuda,nvtx,osrt \
    --stats=true \
    --force-overwrite=true \
    -o "${OUT_ROOT}/nsys/compare_full_vs_sliding_nsys" \
    "${PYTHON_BIN}" "${PY_SCRIPT}" \
      --seq-lens "${SEQ_LENS}" \
      --embed-dim "${EMBED_DIM}" \
      --window-radius "${WINDOW_RADIUS}" \
      --num-runs "${NSYS_RUNS}" \
      --warmup-runs 2 \
      --output-dir "${OUT_ROOT}/nsys/script_output"

  if [[ -f "${OUT_ROOT}/nsys/compare_full_vs_sliding_nsys.nsys-rep" ]]; then
    nsys stats "${OUT_ROOT}/nsys/compare_full_vs_sliding_nsys.nsys-rep" \
      > "${OUT_ROOT}/nsys/nsys_stats.txt" || true
  fi

  echo
  echo "Nsight Systems outputs:"
  find "${OUT_ROOT}/nsys" -maxdepth 2 -type f | sort
}

run_ncu_profile() {
  log "Pass 3: Nsight Compute kernel-level profile"

  if ! ensure_symlink_if_found ncu >/dev/null 2>&1; then
    echo "WARNING: ncu not found. Skipping Nsight Compute pass."
    echo "Nsight Compute is optional but useful for SM utilization, occupancy, memory bandwidth, and achieved throughput."
    return 0
  fi

  mkdir -p "${OUT_ROOT}/ncu"

  echo "ncu path:"
  command -v ncu
  echo
  ncu --version || true

  echo
  echo "Available Nsight Compute sets:"
  ncu --list-sets > "${OUT_ROOT}/ncu/ncu_list_sets.txt" || true
  head -80 "${OUT_ROOT}/ncu/ncu_list_sets.txt" || true

  echo
  echo "Profiling one representative sequence length under Nsight Compute."
  echo "This can be much slower than the normal benchmark because kernels may be replayed."

  # One seq length, one measured run keeps NCU manageable.
  # NCU_SET=full is comprehensive but slower.
  # NCU_SET=basic is faster for first-pass sanity.
  ncu \
    --target-processes all \
    --set "${NCU_SET}" \
    --force-overwrite \
    -o "${OUT_ROOT}/ncu/compare_full_vs_sliding_seq${NCU_SEQ_LEN}_${NCU_SET}" \
    "${PYTHON_BIN}" "${PY_SCRIPT}" \
      --seq-lens "${NCU_SEQ_LEN}" \
      --embed-dim "${EMBED_DIM}" \
      --window-radius "${WINDOW_RADIUS}" \
      --num-runs "${NCU_RUNS}" \
      --warmup-runs 0 \
      --output-dir "${OUT_ROOT}/ncu/script_output" \
    > "${OUT_ROOT}/ncu/ncu_console_${NCU_SET}.txt" 2>&1 || {
      echo "WARNING: ncu profiling failed. See ${OUT_ROOT}/ncu/ncu_console_${NCU_SET}.txt"
      return 0
    }

  echo
  echo "Nsight Compute outputs:"
  find "${OUT_ROOT}/ncu" -maxdepth 2 -type f | sort
}

write_readme() {
  cat > "${OUT_ROOT}/README_ANALYSIS_GUIDE.txt" <<EOF
Lab#3 full vs sliding-window profiling guide

Output root:
  ${OUT_ROOT}

Pass 1: Normal benchmark + counters
  Directory:
    ${OUT_ROOT}/benchmark

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
    ${OUT_ROOT}/nsys

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
    ${OUT_ROOT}/ncu

  Key files:
    compare_full_vs_sliding_seq${NCU_SEQ_LEN}_${NCU_SET}.ncu-rep
    ncu_console_${NCU_SET}.txt
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

EOF
}

main() {
  check_inputs
  run_benchmark_with_counters
  run_nsys_profile
  run_ncu_profile
  write_readme

  log "All requested profiling passes completed"
  echo "Output root:"
  echo "  ${OUT_ROOT}"
  echo
  echo "Start with:"
  echo "  ${OUT_ROOT}/benchmark/comparison_score_entries_log.png"
  echo "  ${OUT_ROOT}/benchmark/comparison_attention_only_runtime.png"
  echo "  ${OUT_ROOT}/nsys/compare_full_vs_sliding_nsys.nsys-rep"
  echo "  ${OUT_ROOT}/ncu/ncu_console_${NCU_SET}.txt"
}

main "$@"