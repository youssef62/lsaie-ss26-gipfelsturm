#!/bin/bash
MODEL=22b
NODES=1
STEPS=25
TP=1
PP=4

export PARTITION=normal
export PROJECT_NAME="lsaie-cpu-activation-offloading-1"

# --- full recompute ---
export EXTRA_ARGS="--recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1"
export JOB_NAME="${MODEL}-${STEPS}s-${NODES}n-tp${TP}-pp${PP}-full-recompute"
$(dirname $0)/launch.sh throughput $MODEL $STEPS $NODES $TP $PP

# --- partial offload (fine-grained activation offloading + selective recompute) ---
export NVTE_CPU_OFFLOAD_V1=1
export EXTRA_ARGS="--fine-grained-activation-offloading
    --offload-modules core_attn qkv_linear attn_proj attn_norm mlp_norm
    --recompute-granularity selective
    --recompute-modules layernorm mlp
"
export JOB_NAME="${MODEL}-${STEPS}s-${NODES}n-pp${PP}-finegrained-offload-more-recompute"
$(dirname $0)/launch.sh throughput $MODEL $STEPS $NODES $TP $PP
