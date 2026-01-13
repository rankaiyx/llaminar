#!/bin/bash
# Setup PCIe BAR access capabilities for cross-vendor GPU P2P
#
# This script:
# 1. Configures passwordless sudo for setcap (required for CMake auto-capability)
# 2. Sets permissions on AMD GPU BAR sysfs files
# 3. Can be called by CMake post-build to set capabilities on binaries
#
# Usage:
#   ./setup-pcie-bar-caps.sh              # Full setup (sudo + BAR permissions)
#   ./setup-pcie-bar-caps.sh <binary>     # Set CAP_SYS_ADMIN on binary

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# If a binary path is provided, set capabilities on it
if [ -n "$1" ]; then
    BINARY="$1"
    if [ -f "$BINARY" ]; then
        # Check if we can set capabilities (need sudo or CAP_SETFCAP)
        if sudo -n /usr/sbin/setcap cap_sys_admin+ep "$BINARY" 2>/dev/null; then
            echo -e "${GREEN}✓ Set CAP_SYS_ADMIN on $BINARY${NC}"
        elif sudo setcap cap_sys_admin+ep "$BINARY" 2>/dev/null; then
            echo -e "${GREEN}✓ Set CAP_SYS_ADMIN on $BINARY${NC}"
        else
            echo -e "${YELLOW}⚠ Could not set capability on $BINARY${NC}"
        fi
    fi
    exit 0
fi

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}PCIe BAR P2P Setup for Cross-Vendor GPU${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# Step 1: Configure passwordless sudo for setcap
echo -e "${CYAN}[1/2] Configuring passwordless sudo for setcap...${NC}"

SUDOERS_FILE="/etc/sudoers.d/setcap-nopasswd"
SETCAP_PATH=$(which setcap 2>/dev/null || echo "/usr/sbin/setcap")

if [ -f "$SUDOERS_FILE" ]; then
    echo -e "${GREEN}  ✓ Sudoers file already exists${NC}"
else
    if sudo -n true 2>/dev/null || sudo true; then
        # Create sudoers entry for passwordless setcap
        echo "$(whoami) ALL=(root) NOPASSWD: $SETCAP_PATH" | sudo tee "$SUDOERS_FILE" >/dev/null
        sudo chmod 440 "$SUDOERS_FILE"
        echo -e "${GREEN}  ✓ Created $SUDOERS_FILE${NC}"
    else
        echo -e "${YELLOW}  ⚠ Cannot configure sudoers (no sudo access)${NC}"
    fi
fi

# Verify setcap works without password by testing on a real file
TEST_FILE=$(mktemp)
echo '#!/bin/bash' > "$TEST_FILE"
chmod +x "$TEST_FILE"
if sudo -n "$SETCAP_PATH" cap_sys_admin+ep "$TEST_FILE" >/dev/null 2>&1; then
    echo -e "${GREEN}  ✓ Passwordless setcap verified working${NC}"
else
    echo -e "${YELLOW}  ⚠ Passwordless setcap not working - builds may require manual setup${NC}"
fi
rm -f "$TEST_FILE"
echo ""

# Step 2: Setup AMD GPU BAR permissions
echo -e "${CYAN}[2/2] Setting up AMD GPU BAR access...${NC}"

# Find AMD GPUs (vendor 0x1002) and set BAR permissions
AMD_GPUS=$(find /sys/bus/pci/devices -maxdepth 1 -type l -exec sh -c '
    dev="$1"
    if [ -f "$dev/vendor" ] && grep -q "0x1002" "$dev/vendor" 2>/dev/null; then
        echo "$dev"
    fi
' _ {} \; 2>/dev/null)

if [ -z "$AMD_GPUS" ]; then
    echo -e "${YELLOW}No AMD GPUs found${NC}"
    exit 0
fi

BAR_COUNT=0
for gpu in $AMD_GPUS; do
    BAR_FILE="$gpu/resource0"
    if [ -f "$BAR_FILE" ]; then
        # Get BAR size
        BAR_SIZE=$(stat -c%s "$BAR_FILE" 2>/dev/null || echo 0)
        BAR_SIZE_GB=$((BAR_SIZE / 1024 / 1024 / 1024))
        
        if [ "$BAR_SIZE_GB" -gt 0 ]; then
            GPU_NAME=$(basename "$gpu")
            echo "Found AMD GPU: $GPU_NAME (BAR0: ${BAR_SIZE_GB}GB)"
            
            # Set permissions
            if sudo -n true 2>/dev/null; then
                sudo chmod 666 "$BAR_FILE" 2>/dev/null && \
                    echo -e "${GREEN}  ✓ Set permissions on $BAR_FILE${NC}" || \
                    echo -e "${RED}  ✗ Failed to set permissions on $BAR_FILE${NC}"
            else
                echo -e "${YELLOW}  ⚠ Need sudo to set BAR permissions${NC}"
            fi
            BAR_COUNT=$((BAR_COUNT + 1))
        fi
    fi
done

if [ "$BAR_COUNT" -gt 0 ]; then
    echo -e "${GREEN}Setup complete: $BAR_COUNT AMD GPU BAR(s) configured${NC}"
else
    echo -e "${YELLOW}No usable AMD GPU BARs found (need large BAR support)${NC}"
fi

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${GREEN}Setup complete!${NC}"
echo ""
echo "CMake will now automatically set CAP_SYS_ADMIN on all built executables."
echo "This enables PCIe BAR direct P2P transfers between CUDA and ROCm GPUs."
echo -e "${CYAN}========================================${NC}"
