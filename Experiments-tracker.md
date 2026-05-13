# Experiments Tracker

## Baseline 1 — 8B, no CUDA graphs (FAILED — missing .edf)

```bash
./launch.sh throughput 8b 50 1
```

- **Status:** Failed (job 2089530) — `alps3.toml` not found in `~/.edf/`
- **Fix:** `mkdir -p ~/.edf && cp alps3.toml ~/.edf/`

---

## Baseline 2 — 8B, no CUDA graphs (FAILED — missing submodule)

```bash
./launch.sh throughput 8b 50 1
```

- **Status:** Failed (job 2176739) — Megatron-LM submodule not initialized
- **Fix:** `rm -rf Megatron-LM/ && git submodule update --init --recursive`

---

## Baseline 3 — 8B, no CUDA graphs (pending)

```bash
./launch.sh throughput 8b 50 1
```

- **Status:** Pending
- **Goal:** Clean throughput baseline with no CUDA graphs to compare against

---

## Experiment 1 — 8B, CUDA graphs full scope (planned)

```bash
./launch.sh throughput 8b 50 1 "--cuda-graph-impl transformer_engine --cuda-graph-scope full"
```

- **Status:** Planned
- **Expected:** Higher tokens/sec vs. baseline due to reduced CPU kernel-launch overhead

---

## Experiment 2 — 8B, CUDA graphs attn only (planned)

```bash
./launch.sh throughput 8b 50 1 "--cuda-graph-impl transformer_engine --cuda-graph-scope attn"
```

- **Status:** Planned

---

## Results

| Job | Config | Tokens/sec | Notes |
|-----|--------|-----------|-------|
| 2089530 | baseline | — | Failed (.edf) |
| 2176739 | baseline | — | Failed (submodule) |
| TBD | baseline (no cuda graph) | — | Pending |
| TBD | cuda graph full | — | Planned |
| TBD | cuda graph attn | — | Planned |
