MODEL=22b
NODES=1
STEPS=50
TP=1
PP=4
export PARTITION=debug
export NVTE_CPU_OFFLOAD_V1=1

export JOB_NAME="${MODEL}-${STEPS}s-${NODES}n-pp${PP}-finegrained-offload-more-recompute"
export PROJECT_NAME="lsaie-cpu-activation-offloading"

export EXTRA_ARGS="--fine-grained-activation-offloading
    --offload-modules core_attn qkv_linear attn_proj attn_norm mlp_norm
    --recompute-granularity selective
    --recompute-modules layernorm mlp
"

$(dirname $0)/launch.sh throughput $MODEL $STEPS $NODES $TP $PP
