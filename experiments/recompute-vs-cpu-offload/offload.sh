MODEL=18b
NODES=1
STEPS=50
TP=4

export PARTITION=normal

export JOB_NAME="${MODEL}-${STEPS}s-${NODES}n-tp${TP}-pp${PP}-full-offload"
export PROJECT_NAME="lsaie-cpu-activation-offloading"


export EXTRA_ARGS="--cpu-offloading-num-layers 39"
$(dirname $0)/launch.sh throughput $MODEL $STEPS $NODES $TP
