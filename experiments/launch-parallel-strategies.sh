
MODEL=8b
NODES=1
STEPS=50
export PARTITION=normal

# format: DP,TP,PP
STRATEGIES=(
    "4,1,1"
    "1,4,1"
    "1,1,4"
    "2,2,1"
    "2,1,2"
    "1,2,2"
)


for STRATEGY in "${STRATEGIES[@]}"; do
    IFS=',' read -r DP TP PP <<< "$STRATEGY"
    echo "Launching with DP=$DP, TP=$TP, PP=$PP"
    
    export JOB_NAME="${MODEL}-${STEPS}s-${NODES}n-dp${DP}-tp${TP}-pp${PP}"
    export PROJECT_NAME="lsaie-parallel-strategies"

    # DP is automatically inferred
    CMD="./launch.sh throughput $MODEL $STEPS $NODES $TP $PP"
    
    echo "Command: $CMD"
    eval "$CMD"

done