MODEL=22b
NODES=1
STEPS=50
TP=1
PP=4

export PARTITION=debug
export EXTRA_ARGS="--recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1"
export JOB_NAME="${MODEL}-${STEPS}s-${NODES}n-tp${TP}-pp${PP}-full-recompute"
export PROJECT_NAME="lsaie-cpu-activation-offloading"


$(dirname $0)/launch.sh throughput $MODEL $STEPS $NODES $TP $PP
