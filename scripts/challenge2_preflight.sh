#!/bin/bash
#
# Check that the repo is ready to submit Challenge 2 throughput jobs on Clariden.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

pass() { printf '[ok] %s\n' "$1"; }
warn() { printf '[warn] %s\n' "$1"; }
fail() { printf '[fail] %s\n' "$1"; exit 1; }

[ -f config.sh ] || fail "config.sh is missing. Copy config.sh.example to config.sh and fill it in."
# shellcheck disable=SC1091
source config.sh

[ -n "${WORKDIR:-}" ] || fail "WORKDIR is not set in config.sh."
[ -n "${SBATCH_ACCOUNT:-}" ] || fail "SBATCH_ACCOUNT is not set in config.sh."
pass "config.sh loaded: WORKDIR=$WORKDIR, SBATCH_ACCOUNT=$SBATCH_ACCOUNT"

if [ -f "$HOME/.edf/alps3.toml" ]; then
    pass "~/.edf/alps3.toml exists"
    grep -q "workdir = \"$HOME\"" "$HOME/.edf/alps3.toml" \
        && pass "~/.edf/alps3.toml workdir points at HOME" \
        || warn "~/.edf/alps3.toml workdir does not look like HOME; inspect it with: grep workdir ~/.edf/alps3.toml"
else
    fail "~/.edf/alps3.toml is missing. Run the sed command from the README setup."
fi

command -v sbatch >/dev/null 2>&1 && pass "sbatch is available" || fail "sbatch not found; run this on Clariden."
command -v squeue >/dev/null 2>&1 && pass "squeue is available" || warn "squeue not found"

[ -d Megatron-LM/megatron ] || fail "Megatron-LM is not initialized. Run: git submodule update --init"
pass "Megatron-LM submodule is populated"

if (cd Megatron-LM && git apply --check ../patches/*.patch); then
    pass "Megatron patches apply cleanly"
else
    fail "Megatron patches do not apply cleanly"
fi

DATA_PREFIX=/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small_megatron/climbmix_small
if [ -f "${DATA_PREFIX}.bin" ] && [ -f "${DATA_PREFIX}.idx" ]; then
    pass "Megatron dataset files are visible"
else
    warn "Dataset not visible at ${DATA_PREFIX}.{bin,idx}; this is expected only if you are not on Clariden."
fi

mkdir -p logs results
pass "logs/ and results/ directories are ready"

printf '\nReady. First cheap test:\n  ./launch.sh throughput 125m 50 1\n'
