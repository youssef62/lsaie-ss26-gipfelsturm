MODEL=18b
NODES=1
STEPS=50
TP=4

export PARTITION=normal
export NVTE_CPU_OFFLOAD_V1=1

export JOB_NAME="${MODEL}-${STEPS}s-${NODES}n-tp${TP}-finegrained-offload-attn-proj-qkv_linear"
export PROJECT_NAME="lsaie-cpu-activation-offloading"

export EXTRA_ARGS="--fine-grained-activation-offloading
    --offload-modules core_attn qkv_linear
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1"
$(dirname $0)/launch.sh throughput $MODEL $STEPS $NODES $TP
