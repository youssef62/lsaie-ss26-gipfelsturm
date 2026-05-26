#!/bin/bash
MODEL=22b
NODES=1
STEPS=25
TP=4


export PARTITION=normal
export PROJECT_NAME="lsaie-cpu-activation-offloading-1"

for MBS in 1 2; do
    export MBS

    # --- recompute ---
    export JOB_NAME="${MODEL}-${STEPS}s-${NODES}n-tp${TP}-full-recompute-mbs${MBS}"
    export EXTRA_ARGS="--recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1"
    $(dirname $0)/launch.sh throughput $MODEL $STEPS $NODES $TP

    # --- cpu offload ---
    export JOB_NAME="${MODEL}-${STEPS}s-${NODES}n-tp${TP}-full-offload-mbs${MBS}"
    export EXTRA_ARGS="--cpu-offloading-num-layers 39"
    $(dirname $0)/launch.sh throughput $MODEL $STEPS $NODES $TP
done
