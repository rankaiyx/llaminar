#!/bin/bash
# Devcontainer onstart script
# This runs every time the container starts (not just on creation)

set -e

echo "[onstart] Configuring system limits and permissions..."

# Set unlimited locked memory for GPU P2P transfers and DMA operations
# This is required for ROCm/CUDA peer-to-peer memory transfers
# We set it in limits.conf so it applies to all new processes/terminals

LIMITS_FILE="/etc/security/limits.d/99-gpu-memlock.conf"
if [ ! -f "$LIMITS_FILE" ]; then
    echo "# GPU P2P transfer requires unlimited locked memory" | sudo tee "$LIMITS_FILE"
    echo "* soft memlock unlimited" | sudo tee -a "$LIMITS_FILE"
    echo "* hard memlock unlimited" | sudo tee -a "$LIMITS_FILE"
    echo "root soft memlock unlimited" | sudo tee -a "$LIMITS_FILE"
    echo "root hard memlock unlimited" | sudo tee -a "$LIMITS_FILE"
    echo "[onstart] Created $LIMITS_FILE for unlimited memlock"
else
    echo "[onstart] $LIMITS_FILE already exists"
fi

# Also try to set it for the current shell session via prlimit
if sudo prlimit --memlock=unlimited:unlimited --pid $$ 2>/dev/null; then
    echo "[onstart] Set memlock unlimited for current process"
fi

# Fix render device permissions (host may assign different GIDs)
if [ -d /dev/dri ]; then
    sudo chmod 666 /dev/dri/render* 2>/dev/null || true
    echo "[onstart] Fixed /dev/dri/render* permissions"
fi

# Ensure kfd device is accessible
if [ -e /dev/kfd ]; then
    sudo chmod 666 /dev/kfd 2>/dev/null || true
    echo "[onstart] Fixed /dev/kfd permissions"
fi

# Report GPU status
echo "[onstart] Checking GPU availability..."

# Check CUDA GPUs
if command -v nvidia-smi &>/dev/null; then
    CUDA_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    echo "[onstart] Found $CUDA_COUNT CUDA GPU(s)"
fi

# Check ROCm GPUs
if command -v rocm-smi &>/dev/null; then
    ROCM_COUNT=$(rocm-smi --showid 2>/dev/null | grep -c "GPU" || echo 0)
    echo "[onstart] Found $ROCM_COUNT ROCm GPU(s)"
fi

echo "[onstart] Done."
