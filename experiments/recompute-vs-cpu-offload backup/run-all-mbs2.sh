#!/bin/bash
MODEL=22b
NODES=1
STEPS=50
TP=4
export MBS=2
export PARTITION=normal
export PROJECT_NAME="lsaie-cpu-activation-offloading"

# --- recompute ---
export JOB_NAME="${MODEL}-${STEPS}s-${NODES}n-tp${TP}-full-recompute-mbs2"
export EXTRA_ARGS="--recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1"
$(dirname $0)/launch.sh throughput $MODEL $STEPS $NODES $TP

# --- cpu offload ---
export JOB_NAME="${MODEL}-${STEPS}s-${NODES}n-tp${TP}-full-offload-mbs2"
export EXTRA_ARGS="--cpu-offloading-num-layers 39"
$(dirname $0)/launch.sh throughput $MODEL $STEPS $NODES $TP

$(dirname $0)/launch.sh throughput $MODEL $STEPS $NODES $TP
