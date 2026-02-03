#!/bin/bash
# shellcheck disable=SC2029
# shellcheck disable=SC2015
# Validation script for multi-node training setup
# Checks all prerequisites before launching training
# Aligned with launch_multinode.sh requirements

set -e

#==================================================================================
# USAGE
#==================================================================================

usage() {
    echo "Usage: $0 <node1> <node2> [node3] ..."
    echo ""
    echo "Validate multi-node training setup before launching."
    echo ""
    echo "Arguments:"
    echo "  node1, node2, ...   List of compute nodes to validate (at least 2 required)"
    echo ""
    echo "Environment variables (same as launch_multinode.sh):"
    echo "  LOCAL_WORKSPACE     Workspace path on nodes (default: /scratch/a.palmas/code-interp-benchmark)"
    echo "  ACCELERATE_CONFIG   Accelerate config file (default: pyine/configs/accelerate/deepspeed_zero3_multinode_2x8gpu.yaml)"
    echo "  CACHE_BASE          Cache directory on nodes (default: /scratch/a.palmas/tmp/cache)"
    echo "  LOG_DIR             Log directory on NAS (default: /nas/users/a.palmas/logs/multinode_validation)"
    echo "  ENABLE_PROFILING    Check for nsys if true (default: false)"
    echo ""
    echo "Examples:"
    echo "  $0 gpu05 gpu06"
    echo "  $0 gpu01 gpu02 gpu03 gpu04"
    echo "  LOCAL_WORKSPACE=/scratch/user/project $0 gpu05 gpu06"
    exit 1
}

# Check for help flag or insufficient arguments
if [ "$1" = "-h" ] || [ "$1" = "--help" ] || [ $# -lt 2 ]; then
    usage
fi

#==================================================================================
# CONFIGURATION (aligned with launch_multinode.sh)
#==================================================================================

COMPUTE_NODES=("$@")
NUM_NODES=${#COMPUTE_NODES[@]}
EXPECTED_GPUS_PER_NODE=8

# Local workspace on each node's /scratch (must be synced manually!)
LOCAL_WORKSPACE="${LOCAL_WORKSPACE:-/scratch/a.palmas/code-interp-benchmark}"

# Accelerate config (relative to LOCAL_WORKSPACE)
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-pyine/configs/accelerate/deepspeed_zero3_multinode_2x8gpu.yaml}"

# Cache directories (local scratch for performance)
LOCAL_SCRATCH_BASE="${LOCAL_SCRATCH_BASE:-/scratch/a.palmas/tmp}"
CACHE_BASE="${CACHE_BASE:-${LOCAL_SCRATCH_BASE}/cache}"

# NAS log directory for validation
LOG_DIR="${LOG_DIR:-/nas/users/a.palmas/logs/multinode_validation}"

# Profiling check (optional)
ENABLE_PROFILING="${ENABLE_PROFILING:-false}"

#==================================================================================
# COLORS
#==================================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_check() {
    echo -e "${GREEN}✓${NC} $1"
}

print_fail() {
    echo -e "${RED}✗${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

#==================================================================================
# VALIDATION FUNCTIONS
#==================================================================================

validate_ssh() {
    print_header "1. SSH Connectivity"
    local all_passed=true

    for node in "${COMPUTE_NODES[@]}"; do
        print_info "Testing SSH to $node..."
        if timeout 10 ssh -o ConnectTimeout=10 -o BatchMode=yes "$node" "echo 'OK'" &>/dev/null; then
            print_check "$node is reachable"
        else
            print_fail "$node is NOT reachable"
            print_info "  Try: ssh-copy-id $node"
            all_passed=false
        fi
    done

    [ "$all_passed" = true ] && return 0 || return 1
}

validate_node_connectivity() {
    print_header "2. Node-to-Node Connectivity"

    print_info "Getting node IPs..."
    for node in "${COMPUTE_NODES[@]}"; do
        local ip
        ip=$(timeout 10 ssh -o ConnectTimeout=5 "$node" "hostname -I | awk '{print \$1}'" 2>/dev/null)
        [ -n "$ip" ] && print_info "  $node: $ip" || print_warn "  Could not get IP for $node"
    done

    print_info "Testing connectivity..."
    for from_node in "${COMPUTE_NODES[@]}"; do
        for to_node in "${COMPUTE_NODES[@]}"; do
            if [ "$from_node" != "$to_node" ]; then
                if timeout 10 ssh -o ConnectTimeout=5 "$from_node" "timeout 5 ping -c 1 -W 2 $to_node" &>/dev/null; then
                    print_check "$from_node can reach $to_node"
                else
                    print_warn "$from_node cannot ping $to_node (may be OK if ICMP blocked)"
                fi
            fi
        done
    done

    return 0
}

validate_workspace() {
    print_header "3. Local Workspace"
    local all_passed=true

    for node in "${COMPUTE_NODES[@]}"; do
        print_info "Checking workspace on $node..."
        if ssh "$node" "test -d ${LOCAL_WORKSPACE}" 2>/dev/null; then
            print_check "$node has ${LOCAL_WORKSPACE}"
        else
            print_fail "$node missing ${LOCAL_WORKSPACE}"
            print_info "  Run: ./scripts/sync_code_to_nodes.sh"
            all_passed=false
        fi
    done

    if [ "$all_passed" = true ]; then
        # Check sync
        print_info "Checking code sync..."
        local main_node
        main_node="${COMPUTE_NODES[0]}"
        local main_checksum
        main_checksum=$(ssh "$main_node" "find ${LOCAL_WORKSPACE} -name '*.py' -type f -exec md5sum {} \; 2>/dev/null | sort | md5sum" 2>/dev/null || echo "")

        for node in "${COMPUTE_NODES[@]:1}"; do
            local node_checksum
            node_checksum=$(ssh "$node" "find ${LOCAL_WORKSPACE} -name '*.py' -type f -exec md5sum {} \; 2>/dev/null | sort | md5sum" 2>/dev/null || echo "")
            if [ "$main_checksum" = "$node_checksum" ]; then
                print_check "$node code matches $main_node"
            else
                print_fail "$node code DIFFERS from $main_node"
                print_info "  Run: ./scripts/sync_code_to_nodes.sh"
                all_passed=false
            fi
        done
    fi

    [ "$all_passed" = true ] && return 0 || return 1
}

validate_git() {
    print_header "4. Git Access"
    local all_passed=true

    for node in "${COMPUTE_NODES[@]}"; do
        print_info "Checking git on $node..."
        # Check git is available
        if ! ssh "$node" "command -v git" &>/dev/null; then
            print_fail "$node: git not found"
            all_passed=false
            continue
        fi

        # Check git pull works (dry-run via fetch)
        if ssh "$node" "cd ${LOCAL_WORKSPACE} && git fetch --dry-run" &>/dev/null; then
            print_check "$node: git fetch works"
        else
            print_fail "$node: git fetch failed (check SSH keys or network)"
            all_passed=false
        fi

        # Check current branch
        local branch
        branch=$(ssh "$node" "cd ${LOCAL_WORKSPACE} && git branch --show-current" 2>/dev/null || echo "unknown")
        print_info "  $node: on branch '$branch'"
    done

    [ "$all_passed" = true ] && return 0 || return 1
}

validate_uv() {
    print_header "5. UV and Python"
    local all_passed=true

    for node in "${COMPUTE_NODES[@]}"; do
        print_info "Checking UV on $node..."
        if timeout 15 ssh -o ConnectTimeout=10 "$node" "bash -l -c 'which uv'" &>/dev/null; then
            local uv_path
            uv_path=$(timeout 15 ssh -o ConnectTimeout=10 "$node" "bash -l -c 'which uv'" 2>&1)
            print_check "$node: UV at $uv_path"

            print_info "  Testing Python..."
            if timeout 60 ssh -o ConnectTimeout=10 "$node" "bash -l -c 'cd $LOCAL_WORKSPACE && uv run python --version'" &>/dev/null; then
                local py_version
                py_version=$(timeout 30 ssh -o ConnectTimeout=10 "$node" "bash -l -c 'cd $LOCAL_WORKSPACE && uv run python --version'" 2>&1)
                print_check "$node: $py_version"
            else
                print_fail "$node: Cannot run Python via UV"
                all_passed=false
            fi
        else
            print_fail "$node: UV not found"
            all_passed=false
        fi
    done

    [ "$all_passed" = true ] && return 0 || return 1
}

validate_accelerate_cli() {
    print_header "6. Accelerate CLI"
    local all_passed=true

    for node in "${COMPUTE_NODES[@]}"; do
        print_info "Checking accelerate CLI on $node..."
        if timeout 30 ssh -o ConnectTimeout=10 "$node" "bash -l -c 'cd $LOCAL_WORKSPACE && uv run accelerate --version'" &>/dev/null; then
            local acc_version
            acc_version=$(timeout 30 ssh -o ConnectTimeout=10 "$node" "bash -l -c 'cd $LOCAL_WORKSPACE && uv run accelerate --version'" 2>&1)
            print_check "$node: accelerate $acc_version"
        else
            print_fail "$node: accelerate CLI not working"
            all_passed=false
        fi
    done

    [ "$all_passed" = true ] && return 0 || return 1
}

validate_gpus() {
    print_header "7. GPU Availability"
    local all_passed=true

    for node in "${COMPUTE_NODES[@]}"; do
        print_info "Checking GPUs on $node..."
        if ssh "$node" "command -v nvidia-smi" &>/dev/null; then
            local gpu_count
            gpu_count=$(ssh "$node" "nvidia-smi --query-gpu=index --format=csv,noheader | wc -l" 2>/dev/null | tr -d ' ')
            if [ "$gpu_count" -eq "$EXPECTED_GPUS_PER_NODE" ]; then
                print_check "$node has $gpu_count GPUs"
            else
                print_warn "$node has $gpu_count GPUs (expected $EXPECTED_GPUS_PER_NODE)"
            fi
        else
            print_fail "$node: nvidia-smi not found"
            all_passed=false
        fi
    done

    [ "$all_passed" = true ] && return 0 || return 1
}

validate_packages() {
    print_header "8. Required Packages"
    local all_passed=true
    local packages=("torch" "transformers" "accelerate" "trl" "vllm" "deepspeed")

    for node in "${COMPUTE_NODES[@]}"; do
        print_info "Checking packages on $node..."
        for pkg in "${packages[@]}"; do
            if ssh "$node" "bash -l -c 'cd $LOCAL_WORKSPACE && uv run python -c \"import ${pkg}\" 2>/dev/null'" &>/dev/null; then
                print_check "$node: $pkg installed"
            else
                print_fail "$node: $pkg NOT installed"
                all_passed=false
            fi
        done
    done

    [ "$all_passed" = true ] && return 0 || return 1
}

validate_cuda() {
    print_header "9. CUDA Functionality"
    local all_passed=true

    for node in "${COMPUTE_NODES[@]}"; do
        print_info "Testing CUDA on $node..."
        if ssh "$node" "bash -l -c 'cd $LOCAL_WORKSPACE && uv run python -c \"import torch; print(f\\\"CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}\\\")\"'" &>/dev/null; then
            local cuda_info
            cuda_info=$(ssh "$node" "bash -l -c 'cd $LOCAL_WORKSPACE && uv run python -c \"import torch; print(f\\\"CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}\\\")\"'" 2>&1)
            print_check "$node: $cuda_info"
        else
            print_fail "$node: Cannot test CUDA"
            all_passed=false
        fi
    done

    [ "$all_passed" = true ] && return 0 || return 1
}

validate_config() {
    print_header "10. Accelerate Configuration"

    local full_path="$LOCAL_WORKSPACE/$ACCELERATE_CONFIG"

    # Check on first node
    if ssh "${COMPUTE_NODES[0]}" "test -f $full_path" 2>/dev/null; then
        print_check "Found: $ACCELERATE_CONFIG"
        print_info "Config details:"
        ssh "${COMPUTE_NODES[0]}" "grep -E 'num_machines|num_processes|zero_stage' $full_path" 2>/dev/null | while read -r line; do
            print_info "  $line"
        done
        return 0
    else
        print_fail "Missing: $ACCELERATE_CONFIG"
        print_info "  Check ACCELERATE_CONFIG env var or ensure file exists"
        return 1
    fi
}

validate_scratch_space() {
    print_header "11. Scratch Space"
    local all_passed=true

    for node in "${COMPUTE_NODES[@]}"; do
        print_info "Checking /scratch on $node..."
        local usage
        usage=$(ssh "$node" "df -h /scratch/a.palmas/ 2>/dev/null | tail -1" 2>&1)
        if [ -n "$usage" ]; then
            print_check "$node: $usage"
            # Check for low space
            local avail_pct
            avail_pct=$(echo "$usage" | awk '{print $5}' | tr -d '%')
            if [ "$avail_pct" -gt 90 ]; then
                print_warn "$node: Less than 10% space available!"
            fi
        else
            print_fail "$node: Cannot check /scratch space"
            all_passed=false
        fi
    done

    [ "$all_passed" = true ] && return 0 || return 1
}

validate_cache_dirs() {
    print_header "12. Cache Directory Access"
    local all_passed=true

    for node in "${COMPUTE_NODES[@]}"; do
        print_info "Checking cache directory on $node..."
        # Try to create and write to cache directory
        if ssh "$node" "mkdir -p ${CACHE_BASE} && touch ${CACHE_BASE}/.validate_test && rm ${CACHE_BASE}/.validate_test" 2>/dev/null; then
            print_check "$node: ${CACHE_BASE} is writable"
        else
            print_fail "$node: Cannot write to ${CACHE_BASE}"
            all_passed=false
        fi
    done

    [ "$all_passed" = true ] && return 0 || return 1
}

validate_nas_access() {
    print_header "13. NAS Log Directory Access"
    local all_passed=true

    # First check if we can create the directory from login node
    print_info "Checking NAS access from login node..."
    if mkdir -p "$LOG_DIR" 2>/dev/null; then
        print_check "Login node: Can create $LOG_DIR"
    else
        print_fail "Login node: Cannot create $LOG_DIR"
        all_passed=false
    fi

    # Check NAS access from each compute node
    for node in "${COMPUTE_NODES[@]}"; do
        print_info "Checking NAS access from $node..."
        if ssh "$node" "test -d ${LOG_DIR} && touch ${LOG_DIR}/.validate_test_${node} && rm ${LOG_DIR}/.validate_test_${node}" 2>/dev/null; then
            print_check "$node: NAS accessible at ${LOG_DIR}"
        else
            print_fail "$node: Cannot access NAS at ${LOG_DIR}"
            all_passed=false
        fi
    done

    [ "$all_passed" = true ] && return 0 || return 1
}

validate_krenew() {
    print_header "14. Kerberos Ticket Renewal (krenew)"
    local all_passed=true

    for node in "${COMPUTE_NODES[@]}"; do
        print_info "Checking krenew on $node..."
        if ssh "$node" "command -v krenew" &>/dev/null; then
            print_check "$node: krenew available"
            # Check if we have a valid Kerberos ticket
            if ssh "$node" "klist -s" &>/dev/null; then
                print_check "$node: Valid Kerberos ticket"
            else
                print_warn "$node: No valid Kerberos ticket (run kinit)"
            fi
        else
            print_fail "$node: krenew not found"
            print_info "  krenew is needed for long-running training to maintain NAS access"
            all_passed=false
        fi
    done

    [ "$all_passed" = true ] && return 0 || return 1
}

validate_nsys() {
    print_header "15. Nsight Systems (Optional - Profiling)"

    if [ "$ENABLE_PROFILING" != "true" ]; then
        print_info "Profiling not enabled (ENABLE_PROFILING=false)"
        print_info "Skipping nsys validation"
        return 0
    fi

    local all_passed=true

    for node in "${COMPUTE_NODES[@]}"; do
        print_info "Checking nsys on $node..."
        if ssh "$node" "command -v nsys" &>/dev/null; then
            local nsys_version
            nsys_version=$(ssh "$node" "nsys --version 2>&1 | head -1" 2>/dev/null || echo "unknown")
            print_check "$node: nsys available ($nsys_version)"
        else
            print_fail "$node: nsys not found"
            print_info "  nsys is needed for ENABLE_PROFILING=true"
            all_passed=false
        fi
    done

    [ "$all_passed" = true ] && return 0 || return 1
}

#==================================================================================
# MAIN
#==================================================================================

print_header "Multi-Node Training Setup Validation"
echo "Nodes: ${COMPUTE_NODES[*]}"
echo "Num nodes: $NUM_NODES"
echo "Workspace: $LOCAL_WORKSPACE"
echo "Accelerate config: $ACCELERATE_CONFIG"
echo "Cache base: $CACHE_BASE"
echo "Log directory: $LOG_DIR"
echo ""

PASSED=0
FAILED=0
TOTAL=15

validate_ssh && ((PASSED++)) || ((FAILED++))
validate_node_connectivity && ((PASSED++)) || ((FAILED++))
validate_workspace && ((PASSED++)) || ((FAILED++))
validate_git && ((PASSED++)) || ((FAILED++))
validate_uv && ((PASSED++)) || ((FAILED++))
validate_accelerate_cli && ((PASSED++)) || ((FAILED++))
validate_gpus && ((PASSED++)) || ((FAILED++))
validate_packages && ((PASSED++)) || ((FAILED++))
validate_cuda && ((PASSED++)) || ((FAILED++))
validate_config && ((PASSED++)) || ((FAILED++))
validate_scratch_space && ((PASSED++)) || ((FAILED++))
validate_cache_dirs && ((PASSED++)) || ((FAILED++))
validate_nas_access && ((PASSED++)) || ((FAILED++))
validate_krenew && ((PASSED++)) || ((FAILED++))
validate_nsys && ((PASSED++)) || ((FAILED++))

#==================================================================================
# SUMMARY
#==================================================================================

print_header "Validation Summary"

if [ $FAILED -eq 0 ]; then
    print_check "All validations passed! ($PASSED/$TOTAL)"
    echo ""
    print_info "Ready to launch training:"
    echo ""
    echo "  ./scripts/launch_multinode.sh ${COMPUTE_NODES[*]}"
    echo ""
    print_info "With custom config:"
    echo "  TRAIN_ARGS='+experiment=your_experiment' ./scripts/launch_multinode.sh ${COMPUTE_NODES[*]}"
    echo ""
    print_info "Available environment variables for launch_multinode.sh:"
    echo "  LOCAL_WORKSPACE     - Workspace path (current: $LOCAL_WORKSPACE)"
    echo "  ACCELERATE_CONFIG   - Accelerate config file (current: $ACCELERATE_CONFIG)"
    echo "  TRAIN_ARGS          - Additional training arguments"
    echo "  CHECKPOINT_DIR      - Custom checkpoint directory"
    echo "  LOG_DIR             - Log directory on NAS"
    echo "  CACHE_BASE          - Cache directory (current: $CACHE_BASE)"
    echo "  RECREATE_CACHE      - Delete and recreate cache (default: false)"
    echo "  ENABLE_DEBUG        - Enable NCCL/PyTorch debug logging (default: false)"
    echo "  ENABLE_PROFILING    - Enable Nsight Systems profiling (default: false)"
    echo "  PROFILE_LEVEL       - Profiling level: minimal, moderate, full (default: minimal)"
    echo "  RESUME_FROM_RUN_DIR - Path to resume from"
    echo "  RESUME_CHECKPOINT   - Specific checkpoint name to resume"
    echo ""
    exit 0
else
    print_fail "Some validations failed! (Passed: $PASSED, Failed: $FAILED)"
    echo ""
    print_info "Fix the issues above before running training"
    echo ""
    exit 1
fi
