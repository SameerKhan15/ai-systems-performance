#!/usr/bin/env bash
set -euo pipefail

# setup_lab3_sliding_window_attention.sh
#
# Purpose:
#   One-shot setup for Lab#3: Sliding Window Attention benchmark on RunPod.
#
# Target machine:
#   A100 PCIe, 80GB VRAM, Ubuntu 22.04-style RunPod image
#
# What this installs/checks:
#   - Nsight Systems CLI: nsys
#   - Python/pip
#   - PyTorch
#   - Matplotlib
#   - nano
#   - CUDA visibility sanity check from PyTorch
#
# Optional:
#   INSTALL_EXTRA_NLP_STACK=1 ./setup_lab3_sliding_window_attention.sh
#   This also installs transformers, datasets, and accelerate.
#
# Usage:
#   chmod +x setup_lab3_sliding_window_attention.sh
#   ./setup_lab3_sliding_window_attention.sh

export DEBIAN_FRONTEND=noninteractive

log() {
  echo
  echo "================================================================================"
  echo "$1"
  echo "================================================================================"
}

warn() {
  echo "WARNING: $1" >&2
}

if [[ "$(id -u)" -eq 0 ]]; then
  SUDO=""
else
  SUDO="sudo"
  if ! command -v sudo >/dev/null 2>&1; then
    echo "This script is not running as root and sudo is not installed."
    echo "Run as root, or install sudo first."
    exit 1
  fi
fi

find_nsys() {
  if command -v nsys >/dev/null 2>&1; then
    command -v nsys
    return 0
  fi

  local found
  found="$(
    {
      find /opt -name nsys -type f 2>/dev/null || true
      find /usr/local -name nsys -type f 2>/dev/null || true
    } | sort -V | tail -n 1
  )"

  if [[ -n "${found}" ]]; then
    echo "${found}"
    return 0
  fi

  return 1
}

install_nsys_if_needed() {
  log "Checking Nsight Systems / nsys"

  local nsys_path
  if nsys_path="$(find_nsys)"; then
    echo "nsys already found at: ${nsys_path}"
  else
    echo "nsys not found. Installing Nsight Systems..."

    log "Installing apt prerequisites"
    ${SUDO} apt-get update
    ${SUDO} apt-get install -y wget gnupg2 ca-certificates

    log "Installing NVIDIA CUDA apt keyring"
    cd /tmp
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    ${SUDO} dpkg -i cuda-keyring_1.1-1_all.deb

    log "Installing Nsight Systems"
    ${SUDO} apt-get update

    if ${SUDO} apt-get install -y nsight-systems-2025.5.2; then
      echo "Installed nsight-systems-2025.5.2"
    else
      warn "nsight-systems-2025.5.2 failed. Trying nsight-systems-2025.3.2..."
      ${SUDO} apt-get install -y nsight-systems-2025.3.2
    fi
  fi

  log "Locating nsys"
  echo "which nsys:"
  which nsys || true

  echo
  echo "find /opt -name nsys:"
  find /opt -name nsys -type f 2>/dev/null | head || true

  echo
  echo "find /usr/local -name nsys:"
  find /usr/local -name nsys -type f 2>/dev/null | head || true

  nsys_path="$(find_nsys || true)"
  if [[ -z "${nsys_path}" ]]; then
    echo "ERROR: nsys still not found after installation attempt."
    exit 1
  fi

  log "Making nsys available on PATH"
  ${SUDO} ln -sf "${nsys_path}" /usr/local/bin/nsys

  echo "nsys path:"
  which nsys

  echo
  echo "nsys version:"
  nsys --version
}

install_python_stack() {
  log "Installing minimal Python stack for Lab#3"

  ${SUDO} apt-get update
  ${SUDO} apt-get install -y python3 python3-pip python3-venv nano

  python3 -m pip install --upgrade pip

  # Minimal dependencies for sliding_window_attention_profile.py:
  #   torch      -> tensor ops + CUDA execution
  #   matplotlib -> plot generation
  python3 -m pip install torch matplotlib

  if [[ "${INSTALL_EXTRA_NLP_STACK:-0}" == "1" ]]; then
    log "Installing optional NLP stack"
    python3 -m pip install torchvision torchaudio transformers datasets accelerate
  else
    echo
    echo "Skipping optional NLP stack."
    echo "To install it too, run:"
    echo "  INSTALL_EXTRA_NLP_STACK=1 ./setup_lab3_sliding_window_attention.sh"
  fi
}

sanity_check_gpu() {
  log "GPU sanity checks"

  echo "nvidia-smi:"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
  else
    warn "nvidia-smi not found on PATH."
  fi

  echo
  echo "PyTorch CUDA sanity check:"
  python3 - << 'EOF'
import sys
import torch

print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("ERROR: PyTorch cannot see CUDA. On the RunPod A100 instance this should be True.")

print("gpu:", torch.cuda.get_device_name(0))
print("cuda_device_count:", torch.cuda.device_count())
print("cuda_runtime_version_from_torch:", torch.version.cuda)
EOF
}

print_next_steps() {
  log "Setup complete"

  cat << 'EOF'
You can now run the Lab#3 benchmark:

  python3 sliding_window_attention_profile.py

Or profile it with Nsight Systems:

  nsys profile \
    --trace=cuda,nvtx,osrt \
    --stats=true \
    --force-overwrite=true \
    -o lab3_sliding_window_attention \
    python3 sliding_window_attention_profile.py

Expected outputs from the benchmark script:

  results/sliding_window_<timestamp>/run.log
  results/sliding_window_<timestamp>/metrics.csv
  results/sliding_window_<timestamp>/sliding_window_attention_runtime_breakdown.png
  results/sliding_window_<timestamp>/sliding_window_attention_dominant_costs.png
  results/sliding_window_<timestamp>/sliding_window_attention_memory_scaling.png

To edit files on the box:

  nano sliding_window_attention_profile.py
EOF
}

main() {
  log "Lab#3 Sliding Window Attention setup starting"

  install_nsys_if_needed
  install_python_stack
  sanity_check_gpu
  print_next_steps
}

main "$@"