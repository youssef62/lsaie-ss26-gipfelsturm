#!/bin/bash
#
# Submit reproducible Challenge 2 throughput experiments.
#
# Usage:
#   scripts/challenge2_submit.sh sanity [steps]
#   scripts/challenge2_submit.sh ozan-cuda [steps]
#   scripts/challenge2_submit.sh repeat-8b-cuda [steps]
#   scripts/challenge2_submit.sh mbs-8b [steps]
#   scripts/challenge2_submit.sh cuda-32b [steps]
#   scripts/challenge2_submit.sh mbs-32b [steps]
#   scripts/challenge2_submit.sh fit-32b [steps]
#   scripts/challenge2_submit.sh fit-140b [steps]
#   scripts/challenge2_submit.sh fit-optimizer [steps]
#   scripts/challenge2_submit.sh fit-cpuopt-frac [steps]
#   scripts/challenge2_submit.sh tiers [steps]
#   SUBMIT=0 scripts/challenge2_submit.sh ozan-cuda 60   # generate sbatch files only

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PROFILE=${1:-ozan-cuda}
STEPS=${2:-60}
SUBMIT=${SUBMIT:-1}

submit() {
    local tag=$1
    local model=$2
    local nodes=$3
    local tp=$4
    local pp=$5
    local mbs=$6
    local extra_args=${7:-}

    printf '\n==> %s: model=%s nodes=%s TP=%s PP=%s MBS=%s\n' "$tag" "$model" "$nodes" "$tp" "$pp" "$mbs"
    RUN_TAG="$tag" \
    TP="$tp" \
    PP="$pp" \
    MICRO_BATCH_SIZE="$mbs" \
    SEQUENCE_PARALLEL=auto \
    SUBMIT="$SUBMIT" \
    MEGATRON_EXTRA_ARGS="$extra_args" \
        ./launch.sh throughput "$model" "$STEPS" "$nodes"
}

case "$PROFILE" in
    sanity)
        submit "sanity-125m" "125m" 1 1 1 16 ""
        submit "baseline-8b" "8b" 1 1 1 2 ""
        ;;

    ozan-cuda)
        # Ozan's lane from the proposal: compare no CUDA graph vs attention/MLP/full-iteration graphs.
        # Keep CPU offloading out of these runs; Megatron rejects CUDA graphs with CPU offloading.
        submit "baseline-8b" "8b" 1 1 1 2 ""
        submit "cg-attn-8b" "8b" 1 1 1 2 "--cuda-graph-impl transformer_engine --cuda-graph-scope attn"
        submit "cg-mlp-8b" "8b" 1 1 1 2 "--cuda-graph-impl transformer_engine --cuda-graph-scope mlp"
        submit "cg-attn-mlp-8b" "8b" 1 1 1 2 "--cuda-graph-impl transformer_engine --cuda-graph-scope attn mlp"
        submit "cg-fulliter-8b" "8b" 1 1 1 2 "--cuda-graph-impl local --cuda-graph-scope full_iteration"
        ;;

    repeat-8b-cuda)
        # Repeat the only potentially positive 8B CUDA graph result to estimate run-to-run noise.
        submit "repeat-baseline-8b" "8b" 1 1 1 2 ""
        submit "repeat-cg-attn-8b" "8b" 1 1 1 2 "--cuda-graph-impl transformer_engine --cuda-graph-scope attn"
        ;;

    mbs-8b)
        # Micro-batch sweep for the official 8B TP=1 PP=1 tier.
        # Larger MBS can reduce gradient accumulation overhead but may OOM.
        submit "mbs1-8b" "8b" 1 1 1 1 ""
        submit "mbs2-8b" "8b" 1 1 1 2 ""
        submit "mbs3-8b" "8b" 1 1 1 3 ""
        submit "mbs4-8b" "8b" 1 1 1 4 ""
        ;;

    cuda-32b)
        # CUDA graph check on the 32B single-node tensor-parallel tier.
        # TP=4 implies sequence parallelism via SEQUENCE_PARALLEL=auto.
        submit "baseline-32b-tp4" "32b" 1 4 1 1 ""
        submit "cg-attn-32b-tp4" "32b" 1 4 1 1 "--cuda-graph-impl transformer_engine --cuda-graph-scope attn"
        submit "cg-mlp-32b-tp4" "32b" 1 4 1 1 "--cuda-graph-impl transformer_engine --cuda-graph-scope mlp"
        submit "cg-attn-mlp-32b-tp4" "32b" 1 4 1 1 "--cuda-graph-impl transformer_engine --cuda-graph-scope attn mlp"
        submit "cg-fulliter-32b-tp4" "32b" 1 4 1 1 "--cuda-graph-impl local --cuda-graph-scope full_iteration"
        ;;

    mbs-32b)
        # Small MBS sweep for 32B TP=4. MBS>2 may OOM; failure is useful data.
        submit "mbs1-32b-tp4" "32b" 1 4 1 1 ""
        submit "mbs2-32b-tp4" "32b" 1 4 1 2 ""
        submit "mbs3-32b-tp4" "32b" 1 4 1 3 ""
        ;;

    fit-32b)
        # The plain 32B TP=4 run can OOM at seq=4096. Try recompute variants first.
        submit "fit-32b-selective-recompute" "32b" 1 4 1 1 "--recompute-granularity selective"
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            submit "fit-32b-full-recompute" "32b" 1 4 1 1 "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1"
        ;;

    fit-140b)
        # The plain 140B TP=4 PP=4 run can OOM at seq=4096. Try full activation recompute.
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            submit "fit-140b-full-recompute" "140b" "${RUN_140B_NODES:-4}" 4 4 1 "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1"
        ;;

    fit-optimizer)
        # Recompute did not fit, so reduce optimizer-state memory. These flags keep the
        # precision-aware optimizer but store main params and Adam moments in lower precision.
        PA_ARGS="--main-params-dtype fp16 --exp-avg-dtype bf16 --exp-avg-sq-dtype bf16"
        submit "fit-32b-pa-optimizer" "32b" 1 4 1 1 "$PA_ARGS"
        submit "fit-32b-pa-optimizer-selective" "32b" 1 4 1 1 "$PA_ARGS --recompute-granularity selective"
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            submit "fit-140b-pa-optimizer-full-recompute" "140b" "${RUN_140B_NODES:-4}" 4 4 1 "$PA_ARGS --recompute-granularity full --recompute-method uniform --recompute-num-layers 1"
        ;;

    fit-cpuopt-frac)
        # Full optimizer CPU offload can exceed node memory. Sweep fractions to trade
        # GPU memory relief against host-memory pressure.
        PA_ARGS="--main-params-dtype fp16 --exp-avg-dtype bf16 --exp-avg-sq-dtype bf16"
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            submit "fit-32b-cpuopt-f025" "32b" 1 4 1 1 "--optimizer-cpu-offload --optimizer-offload-fraction 0.25 $PA_ARGS"
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            submit "fit-32b-cpuopt-f050" "32b" 1 4 1 1 "--optimizer-cpu-offload --optimizer-offload-fraction 0.50 $PA_ARGS"
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            submit "fit-32b-cpuopt-f075" "32b" 1 4 1 1 "--optimizer-cpu-offload --optimizer-offload-fraction 0.75 $PA_ARGS"
        ;;

    tiers)
        # Initial official Challenge 2 tier baselines.
        # 32b/140b are candidate architectures added to launch.sh for systems benchmarking.
        submit "tier-8b-tp1pp1" "8b" 1 1 1 2 ""
        submit "tier-32b-tp4pp1" "32b" 1 4 1 1 ""
        submit "tier-140b-tp4pp4" "140b" "${RUN_140B_NODES:-4}" 4 4 1 ""
        ;;

    all)
        "$0" sanity "$STEPS"
        "$0" ozan-cuda "$STEPS"
        "$0" repeat-8b-cuda "$STEPS"
        "$0" mbs-8b "$STEPS"
        "$0" fit-32b "$STEPS"
        "$0" fit-optimizer "$STEPS"
        "$0" fit-cpuopt-frac "$STEPS"
        "$0" cuda-32b "$STEPS"
        "$0" mbs-32b "$STEPS"
        "$0" fit-140b "$STEPS"
        "$0" tiers "$STEPS"
        ;;

    *)
        cat >&2 <<USAGE
Unknown profile: $PROFILE

Use one of:
  sanity      - 125m sanity + 8b baseline
  ozan-cuda  - 8b baseline plus CUDA graph variants
  repeat-8b-cuda - repeat baseline vs attention CUDA graph at 8B
  mbs-8b     - 8B micro-batch sweep, MBS 1..4
  cuda-32b   - 32B TP=4 baseline plus CUDA graph variants
  mbs-32b    - 32B TP=4 micro-batch sweep, MBS 1..3
  fit-32b    - 32B TP=4 memory-fit sweep with activation recompute
  fit-140b   - 140B TP=4 PP=4 memory-fit sweep with activation recompute
  fit-optimizer - 32B/140B memory-fit sweep with lower-precision optimizer state
  fit-cpuopt-frac - 32B optimizer CPU-offload fraction sweep
  tiers      - 8b, 32b TP=4, 140b TP=4 PP=4 baselines
  all        - submit all profiles
USAGE
        exit 1
        ;;
esac
