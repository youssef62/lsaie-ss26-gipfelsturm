#!/bin/bash
#
# Re-parse logs and produce report-friendly Challenge 2 summaries.
#
# Usage:
#   scripts/challenge2_collect_results.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

scripts/challenge2_parse_logs.sh
python3 scripts/challenge2_summarize_results.py

printf '\nTop throughput rows:\n'
{
    head -1 results/challenge2_results.csv
    awk -F, 'NR > 1 && $11 != ""' results/challenge2_results.csv \
        | sort -t, -k12,12nr \
        | head -20
} | column -s, -t

printf '\nCUDA graph speedups:\n'
column -s, -t results/challenge2_cuda_speedups.csv

printf '\nMarkdown summary:\n  results/challenge2_summary.md\n'
