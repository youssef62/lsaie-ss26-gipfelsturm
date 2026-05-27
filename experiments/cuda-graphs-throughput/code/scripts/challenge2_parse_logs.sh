#!/bin/bash
#
# Parse Challenge 2 SLURM logs into a compact CSV table.
#
# Usage:
#   scripts/challenge2_parse_logs.sh
#   scripts/challenge2_parse_logs.sh results/my_results.csv

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT=${1:-results/challenge2_results.csv}
mkdir -p "$(dirname "$OUT")"

extract_arg() {
    local flag=$1
    local cmd=$2
    awk -v flag="$flag" '{
        for (i = 1; i <= NF; i++) {
            if ($i == flag && i < NF) {
                print $(i + 1)
                exit
            }
        }
    }' <<<"$cmd"
}

series_best() {
    awk 'NF { n += 1; if (n == 1 || $1 > best) best = $1 } END { if (n > 0) printf "%.0f", best }'
}

series_avg_last() {
    local count=${1:-10}
    awk -v count="$count" 'NF {
        n += 1
        vals[n] = $1
    } END {
        if (n == 0) exit
        start = n - count + 1
        if (start < 1) start = 1
        for (i = start; i <= n; i++) sum += vals[i]
        printf "%.0f", sum / (n - start + 1)
    }'
}

printf 'log,job_id,model,tag,nodes,tp,pp,mbs,gbs,seq_len,best_tokens_sec_gpu,avg_last10_tokens_sec_gpu,best_tflops_gpu,status\n' > "$OUT"

shopt -s nullglob
for log in logs/gipfel-throughput-*.log; do
    base=$(basename "$log" .log)
    job_id=""
    job_name=$base
    if [[ $base =~ ^(.+)-([0-9]+)$ ]]; then
        job_name=${BASH_REMATCH[1]}
        job_id=${BASH_REMATCH[2]}
    fi

    model=""
    tag=""
    nodes=""
    tp=""
    pp=""
    mbs=""
    if [[ $job_name =~ ^gipfel-throughput-([^-]+)-(.+)-([0-9]+)s-([0-9]+)n-tp([0-9]+)-pp([0-9]+)-mbs([0-9]+)$ ]]; then
        model=${BASH_REMATCH[1]}
        tag=${BASH_REMATCH[2]}
        nodes=${BASH_REMATCH[4]}
        tp=${BASH_REMATCH[5]}
        pp=${BASH_REMATCH[6]}
        mbs=${BASH_REMATCH[7]}
    fi

    cmd=$(grep -m1 '^CMD:' "$log" | sed 's/^CMD: //' || true)
    gbs=$(extract_arg "--global-batch-size" "$cmd")
    seq_len=$(extract_arg "--seq-length" "$cmd")

    token_series=$(grep -Eo 'tokens/sec/GPU: [0-9.]+' "$log" | awk '{print $2}' || true)
    tflop_series=$(grep -Eo 'throughput per GPU \(TFLOP/s/GPU\): [0-9.]+' "$log" | awk '{print $5}' || true)

    best_tokens=$(printf '%s\n' "$token_series" | series_best)
    avg_last10=$(printf '%s\n' "$token_series" | series_avg_last 10)
    best_tflops=$(printf '%s\n' "$tflop_series" | series_best)

    status="unknown"
    if grep -Eiq 'OutOfMemoryError|CUDA out of memory|out of memory|oom-kill|oom killed' "$log"; then
        status="oom"
    elif [ -n "$token_series" ] && grep -q 'END TIME:' "$log"; then
        status="finished"
    elif [ -n "$token_series" ]; then
        status="has_metrics"
    elif grep -Eiq 'Traceback|ChildFailedError|FAILED$|Exception|RuntimeError|ValueError|AssertionError|Exited with exit code [1-9]' "$log"; then
        status="failed"
    elif [ -z "$token_series" ]; then
        status="no_metrics"
    fi

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$log" "$job_id" "$model" "$tag" "$nodes" "$tp" "$pp" "$mbs" \
        "$gbs" "$seq_len" "$best_tokens" "$avg_last10" "$best_tflops" "$status" >> "$OUT"
done

printf 'Wrote %s\n' "$OUT"
