# Challenge 2 Automation

Run these from the repo root on Clariden.

## 1. Preflight

```bash
scripts/challenge2_preflight.sh
```

## 2. First Sanity Jobs

```bash
scripts/challenge2_submit.sh sanity 50
```

## 3. Ozan: CUDA Graph Sweep

```bash
scripts/challenge2_submit.sh ozan-cuda 60
```

This submits:

- `baseline-8b`
- `cg-attn-8b`
- `cg-mlp-8b`
- `cg-attn-mlp-8b`
- `cg-fulliter-8b`

Use `SUBMIT=0` to only generate sbatch files:

```bash
SUBMIT=0 scripts/challenge2_submit.sh ozan-cuda 60
```

## 4. Challenge Tier Baselines

```bash
scripts/challenge2_submit.sh tiers 60
```

This submits:

- 8B, TP=1, PP=1, 1 node
- 32B candidate, TP=4, PP=1, 1 node
- 140B candidate, TP=4, PP=4, 4 nodes by default

For 140B on 8 nodes:

```bash
RUN_140B_NODES=8 scripts/challenge2_submit.sh tiers 60
```

## 5. Parse Results

```bash
scripts/challenge2_parse_logs.sh
column -s, -t results/challenge2_results.csv
```

Use `avg_last10_tokens_sec_gpu` as the main comparison number so CUDA graph warmup
does not dominate the result.
