#!/bin/bash
# ------------------------------------------------------------------------------------------
# launch_wandb_agents.sh - Launches WandB sweep agents across multiple GPU cluster nodes.
#
# IMPORTANT:
#   - This script REQUIRES bash and tmux. Ensure both are available on your system.
#   - Run from the login node; the script SSHs into each specified GPU node.
#   - Each GPU node spawns 8 agents (one per GPU) in a tmux session for easy monitoring.
#
# This script is designed for rapid prototyping and iteration on GPU clusters:
#   - Creates organized tmux sessions (one per GPU node) with 8 panes (one per GPU);
#   - Handles session name collisions automatically with random suffixes;
#   - Validates inputs (command file exists and non-empty, valid node indexes);
#   - Substitutes template variables (REPO_ROOT, CUDA_DEVICE, SWEEP_ID, TARGET_GPU);
#   - All agents connect to the same centralized WandB sweep server for coordination.
#
# For a more detailed documentation on WandB sweeps, see:
#   EXPERIMENTATION_GUIDE.md (Step 6b - W&B Agents: Run Hyperparameter Sweeps)
#
# Usage examples:
#   # Launch agents on GPU nodes 1, 2, and 3
#   bash launch_wandb_agents.sh <sweep-id> \
#     --cmd-file ./scripts/my_command \
#     --repo-root /scratch/user/repo \
#     1 2 3
#
#   # Launch on a single GPU node
#   bash launch_wandb_agents.sh <sweep-id> \
#     --cmd-file ./scripts/my_command \
#     --repo-root /scratch/user/repo \
#     1
# ------------------------------------------------------------------------------------------

# Bash strict mode for better error handling
set -euo pipefail
IFS=$'\n\t'

# Check if the correct number of arguments is provided
if [ "$#" -lt 6 ]; then
    echo "Usage: $0 <WandB Sweep ID> --cmd-file <path> --repo-root <path> <Node Index 1> [Node Index 2] ... [Node Index N]"
    echo "  Node indexes: 1-8"
    echo "  --cmd-file: Path to file containing command template (required)"
    echo "  --repo-root: Path to repository root directory (required)"
    echo "  e.g.: $0 lawzero-default/code-interp-benchmark-pyine_apps_trainers/4vp5ivg4 --cmd-file ./my_command --repo-root /scratch/a.palmas/code-interp-benchmark 1 2 3"
    echo ""
    echo "Command template variables:"
    echo "  \${TARGET_GPU} - GPU node number"
    echo "  \${CUDA_DEVICE} - CUDA device index (0-7)"
    echo "  \${SWEEP_ID} - WandB sweep ID"
    echo "  \${REPO_ROOT} - Repository root path"
    exit 1
fi

# Assign input arguments to variables
WANDB_SWEEP_ID="$1"
shift  # Remove first argument, rest are flags and node indexes

# Parse arguments for command file, repo root, and node indexes
NODE_INDEXES=()
CMD_FILE=""
REPO_ROOT=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --cmd-file)
            CMD_FILE="$2"
            shift 2
            ;;
        --repo-root)
            REPO_ROOT="$2"
            shift 2
            ;;
        *)
            NODE_INDEXES+=("$1")
            shift
            ;;
    esac
done

# Validate command file is provided
if [ -z "$CMD_FILE" ]; then
    echo "Error: --cmd-file parameter is required"
    exit 1
fi

# Validate repo root is provided
if [ -z "$REPO_ROOT" ]; then
    echo "Error: --repo-root parameter is required"
    exit 1
fi

# Check if command file exists and read it
if [ ! -f "$CMD_FILE" ]; then
    echo "Error: Command file not found: $CMD_FILE"
    exit 1
fi

COMMAND_TEMPLATE=$(cat "$CMD_FILE")

# Validate command file is not empty
if [ -z "$COMMAND_TEMPLATE" ]; then
    echo "Error: Command file is empty: $CMD_FILE"
    exit 1
fi

echo "Using command template from: $CMD_FILE"

# Validate node indexes
for node in "${NODE_INDEXES[@]}"; do
    if [ "$node" -lt 1 ] || [ "$node" -gt 8 ]; then
        echo "Error: Node index $node is out of range (1-8)"
        exit 1
    fi
done

# Function to configure tmux for a specific GPU node
configure_tmux_for_node() {
    local target_gpu=$1
    local sweep_id=$2
    local session_name="wandb_sweep_gpu${target_gpu}"

    # Check if session already exists, if so add random suffix
    if tmux has-session -t "$session_name" 2>/dev/null; then
        # Generate random 3-character suffix
        local suffix
        suffix=$(LC_ALL=C tr -dc 'a-z0-9' < /dev/urandom | head -c 3)
        local original_name="$session_name"
        session_name="${session_name}_${suffix}"
        echo "⚠️  Warning: Session '$original_name' already exists. Using '$session_name' instead."
    fi

    # Create new session
    tmux new-session -d -s "$session_name"

    # Create 8 panes in 2 rows x 4 columns layout
    # First row: split horizontally 3 times to get 4 panes
    tmux split-window -h -t "$session_name"
    tmux split-window -h -t "$session_name"
    tmux split-window -h -t "$session_name"

    tmux select-layout -t "$session_name" tiled

    # Select first pane and split vertically to create second row
    tmux select-pane -t "${session_name}:0.0"
    tmux split-window -h -t "$session_name"

    # Split the bottom row horizontally 3 times to get 4 panes
    tmux select-pane -t "${session_name}:0.2"
    tmux split-window -h -t "$session_name"
    tmux select-pane -t "${session_name}:0.4"
    tmux split-window -h -t "$session_name"
    tmux select-pane -t "${session_name}:0.6"
    tmux split-window -h -t "$session_name"

    # Ensure equal sizing (2 rows x 4 columns)
    #tmux select-layout -t "$session_name" tiled

    # Send commands to each pane (8 panes, CUDA devices 0-7)
    for pane in {0..7}; do
        # Substitute variables in command template
        local cmd="${COMMAND_TEMPLATE}"
        cmd="${cmd//\$\{TARGET_GPU\}/${target_gpu}}"
        cmd="${cmd//\$\{CUDA_DEVICE\}/${pane}}"
        cmd="${cmd//\$\{SWEEP_ID\}/${sweep_id}}"
        cmd="${cmd//\$\{REPO_ROOT\}/${REPO_ROOT}}"

        echo "  Pane ${pane}: Connecting via SSH to gpu0${target_gpu}..."
        tmux send-keys -t "${session_name}:0.${pane}" "ssh gpu0${target_gpu}" C-m
        sleep 3
        echo "  Pane ${pane}: Executing command on remote host..."
        tmux send-keys -t "${session_name}:0.${pane}" "$cmd" C-m
    done

    echo "Configured tmux session '$session_name' for GPU node $target_gpu"
}

# Main loop: iterate through each node
for node in "${NODE_INDEXES[@]}"; do
    echo "Setting up wandb agents for GPU node $node..."
    configure_tmux_for_node "$node" "$WANDB_SWEEP_ID"
done

echo ""
echo "All tmux sessions created successfully!"
echo "Available sessions:"
tmux list-sessions
echo ""
echo "To attach to a session, use: tmux attach-session -t wandb_sweep_gpu<N>"
echo "To switch between sessions: Ctrl+b then s"
