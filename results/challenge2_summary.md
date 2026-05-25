# Challenge 2 Results Summary

Generated from `results/challenge2_results.csv`.

## Top Throughput Runs

| model | tag | nodes | tp | pp | mbs | avg_last10_tokens_sec_gpu | best_tflops_gpu | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 760m | cgtest-760m-mbs1-te-attn | 1 | 1 | 1 | 1 | 42595 | 235 | finished |
| 760m | cgtest-760m-mbs1-te-full-layer | 1 | 1 | 1 | 1 | 42591 | 236 | finished |
| 350m | cgtest-350m-mbs1-te-mlp | 1 | 1 | 1 | 1 | 42565 | 121 | finished |
| 760m | cgtest-760m-mbs1-te-mlp | 1 | 1 | 1 | 1 | 42528 | 228 | finished |
| 760m | cgtest-760m-mbs1-baseline | 1 | 1 | 1 | 1 | 41750 | 215 | finished |
| 125m | cgtest-125m-mbs1-baseline | 1 | 1 | 1 | 1 | 41271 | 42 | finished |
| 350m | cgtest-350m-mbs1-te-full-layer | 1 | 1 | 1 | 1 | 41270 | 118 | finished |
| 125m | cgtest-125m-mbs1-te-attn | 1 | 1 | 1 | 1 | 41269 | 42 | finished |
| 125m | cgtest-125m-mbs1-te-mlp | 1 | 1 | 1 | 1 | 40777 | 44 | finished |
| 125m | cgtest-125m-mbs1-te-full-layer | 1 | 1 | 1 | 1 | 40774 | 44 | finished |
| 125m | sanity-125m | 1 | 1 | 1 | 16 | 40718 | 48 | finished |
| 350m | cgtest-350m-mbs1-baseline | 1 | 1 | 1 | 1 | 40405 | 109 | finished |
| 350m | cgtest-350m-mbs1-te-attn | 1 | 1 | 1 | 1 | 40389 | 118 | finished |
| 8b | cg-mlp-8b | 1 | 1 | 1 | 2 | 10990 | 517 | finished |
| 8b | repeat-baseline-8b | 1 | 1 | 1 | 2 | 10973 | 516 | finished |
| 8b | mbs2-8b | 1 | 1 | 1 | 2 | 10938 | 515 | finished |
| 8b | cg-te-full-layer-8b | 1 | 1 | 1 | 2 | 10917 | 509 | has_metrics |
| 8b | cg-attn-8b | 1 | 1 | 1 | 2 | 10873 | 510 | finished |
| 8b | baseline-8b | 1 | 1 | 1 | 2 | 10839 | 508 | finished |
| 8b | cg-te-full-layer-8b | 1 | 1 | 1 | 2 | 10825 | 505 | has_metrics |

## Best Run Per Shape

| model | nodes | tp | pp | mbs | steps | best_tag | avg_last10_tokens_sec_gpu | best_tflops_gpu |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 125m | 1 | 1 | 1 | 1 | 100 | cgtest-125m-mbs1-baseline | 41271 | 42 |
| 125m | 1 | 1 | 1 | 16 | 50 | sanity-125m | 40718 | 48 |
| 32b | 4 | 4 | 4 | 1 | 60 | fit-32b-4n-tp4pp4 | 1908 | 403 |
| 350m | 1 | 1 | 1 | 1 | 100 | cgtest-350m-mbs1-te-mlp | 42565 | 121 |
| 760m | 1 | 1 | 1 | 1 | 100 | cgtest-760m-mbs1-te-attn | 42595 | 235 |
| 8b | 1 | 1 | 1 | 1 | 60 | mbs1-8b | 10656 | 507 |
| 8b | 1 | 1 | 1 | 2 | 100 | cg-te-full-layer-8b | 10917 | 509 |
| 8b | 1 | 1 | 1 | 2 | 50 | baseline-8b | 10839 | 508 |
| 8b | 1 | 1 | 1 | 2 | 60 | cg-mlp-8b | 10990 | 517 |

## CUDA Graph Speedups

| model | tag | mbs | cuda_graph_impl | cuda_graph_scope | baseline_avg_last10 | graph_avg_last10 | speedup_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 125m | cgtest-125m-mbs1-te-attn | 1 | transformer_engine | attn | 41271 | 41269 | -0.00 |
| 125m | cgtest-125m-mbs1-te-full-layer | 1 | transformer_engine | full-layer/default | 41271 | 40774 | -1.20 |
| 125m | cgtest-125m-mbs1-te-mlp | 1 | transformer_engine | mlp | 41271 | 40777 | -1.20 |
| 350m | cgtest-350m-mbs1-te-attn | 1 | transformer_engine | attn | 40405 | 40389 | -0.04 |
| 350m | cgtest-350m-mbs1-te-full-layer | 1 | transformer_engine | full-layer/default | 40405 | 41270 | 2.14 |
| 350m | cgtest-350m-mbs1-te-mlp | 1 | transformer_engine | mlp | 40405 | 42565 | 5.35 |
| 760m | cgtest-760m-mbs1-te-attn | 1 | transformer_engine | attn | 41750 | 42595 | 2.02 |
| 760m | cgtest-760m-mbs1-te-full-layer | 1 | transformer_engine | full-layer/default | 41750 | 42591 | 2.01 |
| 760m | cgtest-760m-mbs1-te-mlp | 1 | transformer_engine | mlp | 41750 | 42528 | 1.86 |
| 8b | cg-attn-8b | 2 | transformer_engine | attn | 10819 | 10873 | 0.50 |
| 8b | cg-attn-8b | 2 | transformer_engine | attn | 10819 | 10632 | -1.72 |
| 8b | cg-attn-mlp-8b | 2 | transformer_engine | attn mlp | 10819 | 10670 | -1.37 |
| 8b | cg-attn-mlp-8b | 2 | transformer_engine | attn mlp | 10819 | 10789 | -0.27 |
| 8b | cg-fulliter-8b | 2 | local | full_iteration | 10819 | 8064 | -25.46 |
| 8b | cg-fulliter-8b | 2 | local | full_iteration | 10819 | 7376 | -31.82 |
| 8b | cg-mlp-8b | 2 | transformer_engine | mlp | 10819 | 10705 | -1.05 |
| 8b | cg-mlp-8b | 2 | transformer_engine | mlp | 10819 | 10990 | 1.58 |
| 8b | cg-te-full-layer-8b | 2 | transformer_engine | full-layer/default | 10734 | 10825 | 0.85 |
| 8b | cg-te-full-layer-8b | 2 | transformer_engine | full-layer/default | 10734 | 10917 | 1.70 |
| 8b | repeat-cg-attn-8b | 2 | transformer_engine | attn | 10878 | 10801 | -0.71 |
| 8b | repeat-cg-attn-8b | 2 | transformer_engine | attn | 10878 | 10560 | -2.93 |

## Status Counts

| model | status | count |
| --- | --- | --- |
| 125m | finished | 5 |
| 140b | oom | 4 |
| 32b | failed | 1 |
| 32b | has_metrics | 3 |
| 32b | oom | 20 |
| 350m | finished | 4 |
| 760m | finished | 4 |
| 8b | failed | 1 |
| 8b | finished | 18 |
| 8b | has_metrics | 3 |
| 8b | oom | 1 |
