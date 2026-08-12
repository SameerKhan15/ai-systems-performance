#!/usr/bin/env bash
set -Eeuo pipefail

# Prefill Latency Lab setup for a RunPod A100 instance.
#
# Usage:
#   chmod +x setup_prefill_latency_lab.sh
#   ./setup_prefill_latency_lab.sh
#
# Optional:
#   LAB_DIR=/workspace/prefill-latency-lab ./setup_prefill_latency_lab.sh

LAB_DIR="${LAB_DIR:-/workspace/prefill-latency-lab}"
VENV_DIR="${VENV_DIR:-${LAB_DIR}/.venv}"

log() {
    printf '\n\033[1;34m==>\033[0m %s\n' "$*"
}

warn() {
    printf '\n\033[1;33mWARNING:\033[0m %s\n' "$*" >&2
}

die() {
    printf '\n\033[1;31mERROR:\033[0m %s\n' "$*" >&2
    exit 1
}

if [[ "${EUID}" -ne 0 ]]; then
    die "Run this script as root, which is the normal user on most RunPod containers."
fi

export DEBIAN_FRONTEND=noninteractive

log "Checking operating system"
if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    echo "OS: ${PRETTY_NAME:-unknown}"
else
    die "/etc/os-release was not found."
fi

case "${VERSION_ID:-}" in
    "22.04")
        NVIDIA_REPO="ubuntu2204"
        ;;
    "24.04")
        NVIDIA_REPO="ubuntu2404"
        ;;
    "20.04")
        NVIDIA_REPO="ubuntu2004"
        ;;
    *)
        warn "Ubuntu ${VERSION_ID:-unknown} is not explicitly covered. Falling back to the Ubuntu 22.04 NVIDIA repository."
        NVIDIA_REPO="ubuntu2204"
        ;;
esac

log "Installing base operating-system packages"
apt-get update
apt-get install -y \
    ca-certificates \
    wget \
    gnupg2 \
    python3 \
    python3-pip \
    python3-venv \
    nano

find_nsys() {
    local candidate

    if command -v nsys >/dev/null 2>&1; then
        command -v nsys
        return 0
    fi

    for candidate in \
        /usr/local/cuda/bin/nsys \
        /usr/local/bin/nsys \
        /opt/nvidia/nsight-systems/*/bin/nsys
    do
        # Intentionally allow pathname expansion here.
        for expanded in $candidate; do
            if [[ -x "$expanded" ]]; then
                printf '%s\n' "$expanded"
                return 0
            fi
        done
    done

    return 1
}

install_nsys() {
    local keyring_deb="/tmp/cuda-keyring_1.1-1_all.deb"
    local keyring_url="https://developer.download.nvidia.com/compute/cuda/repos/${NVIDIA_REPO}/x86_64/cuda-keyring_1.1-1_all.deb"

    log "Nsight Systems was not found; configuring NVIDIA's package repository"
    wget -qO "${keyring_deb}" "${keyring_url}"
    dpkg -i "${keyring_deb}"
    rm -f "${keyring_deb}"

    apt-get update

    log "Installing Nsight Systems"
    if ! apt-get install -y nsight-systems-2025.5.2; then
        warn "nsight-systems-2025.5.2 was unavailable; trying 2025.3.2."
        if ! apt-get install -y nsight-systems-2025.3.2; then
            warn "Version-specific packages were unavailable; trying the repository's default nsight-systems package."
            apt-get install -y nsight-systems
        fi
    fi
}

log "Checking for Nsight Systems"
if NSYS_PATH="$(find_nsys)"; then
    echo "Found nsys at: ${NSYS_PATH}"
else
    install_nsys
    NSYS_PATH="$(find_nsys)" || die "Nsight Systems installation completed, but the nsys executable could not be found."
fi

if [[ "${NSYS_PATH}" != "/usr/local/bin/nsys" ]]; then
    ln -sf "${NSYS_PATH}" /usr/local/bin/nsys
fi

log "Verifying Nsight Systems"
command -v nsys
nsys --version

log "Creating lab directory and Python virtual environment"
mkdir -p "${LAB_DIR}"

# RunPod images often already contain a CUDA-compatible PyTorch installation.
# --system-site-packages lets the lab reuse it instead of downloading another
# multi-gigabyte PyTorch build.
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    python3 -m venv --system-site-packages "${VENV_DIR}"
fi

PYTHON="${VENV_DIR}/bin/python"

log "Upgrading Python packaging tools"
"${PYTHON}" -m pip install --upgrade pip setuptools wheel

log "Checking PyTorch"
if ! "${PYTHON}" -c "import torch" >/dev/null 2>&1; then
    warn "PyTorch was not found in the RunPod image; installing torch, torchvision, and torchaudio from PyPI."
    "${PYTHON}" -m pip install torch torchvision torchaudio
else
    echo "Reusing the PyTorch installation already available in the RunPod image."
fi

log "Installing the remaining lab dependencies"
"${PYTHON}" -m pip install --upgrade \
    transformers \
    datasets \
    accelerate \
    matplotlib

log "Checking GPU visibility in PyTorch"
"${PYTHON}" - <<'PY'
import sys
import torch

print("python:", sys.version.split()[0])
print("pytorch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("torch_cuda_version:", torch.version.cuda)

if not torch.cuda.is_available():
    raise SystemExit(
        "PyTorch cannot see the GPU. Confirm that this is a GPU RunPod and "
        "that the container has access to the NVIDIA driver."
    )

print("gpu_count:", torch.cuda.device_count())
print("gpu:", torch.cuda.get_device_name(0))

properties = torch.cuda.get_device_properties(0)
print("gpu_memory_gib:", round(properties.total_memory / (1024**3), 2))
PY

log "Displaying NVIDIA GPU information"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    warn "nvidia-smi is not on PATH, although PyTorch may still be able to use CUDA."
fi

cat <<EOF