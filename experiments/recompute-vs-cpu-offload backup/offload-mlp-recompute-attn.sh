MODEL=22b
NODES=1
STEPS=50
TP=4

export PARTITION=debug

export JOB_NAME="${MODEL}-${STEPS}s-${NODES}n-tp${TP}-pp${PP}-offload-mlp-recompute-attn"
export PROJECT_NAME="lsaie-cpu-activation-offloading"


export EXTRA_ARGS="--cpu-offloading-num-layers 39 --cpu-offloading-recompute-attention"
$(dirname $0)/launch.sh throughput $MODEL $STEPS $NODES $TP
