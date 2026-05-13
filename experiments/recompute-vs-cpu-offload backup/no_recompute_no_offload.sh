MODEL=22b
NODES=1
STEPS=50
TP=4
export PARTITION=debug


export JOB_NAME="${MODEL}-${STEPS}s-${NODES}n-tp${TP}-pp${PP}-no-offload-no-recompute"
export PROJECT_NAME="lsaie-cpu-activation-offloading"


$(dirname $0)/launch.sh throughput $MODEL $STEPS $NODES $TP
