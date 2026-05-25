#!/usr/bin/env python3
"""Summarize Challenge 2 throughput logs.

Reads results/challenge2_results.csv produced by challenge2_parse_logs.sh and
writes report-friendly CSV and Markdown files under results/.
"""

import csv
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = ROOT / "results" / "challenge2_results.csv"
OUT_DIR = ROOT / "results"


def as_int(value):
    if value == "":
        return None
    return int(float(value))


def as_float(value):
    if value == "":
        return None
    return float(value)


def read_cmd(log_path):
    path = ROOT / log_path
    if not path.exists():
        return ""
    with path.open("r", errors="replace") as handle:
        for line in handle:
            if line.startswith("CMD: "):
                return line[len("CMD: ") :].strip()
    return ""


def extract_flag(cmd, flag):
    parts = cmd.split()
    if flag not in parts:
        return ""
    idx = parts.index(flag)
    if idx + 1 >= len(parts):
        return "true"
    if parts[idx + 1].startswith("--"):
        return "true"
    return parts[idx + 1]


def extract_multi_flag(cmd, flag):
    parts = cmd.split()
    if flag not in parts:
        return ""
    idx = parts.index(flag) + 1
    values = []
    while idx < len(parts) and not parts[idx].startswith("--"):
        values.append(parts[idx])
        idx += 1
    return " ".join(values) if values else "true"


def has_flag(cmd, flag):
    return "yes" if flag in cmd.split() else "no"


def parse_steps(log_path):
    match = re.search(r"-(\d+)s-\d+n-tp\d+-pp\d+-mbs\d+-\d+\.log$", log_path)
    return match.group(1) if match else ""


def load_rows():
    if not RESULTS_CSV.exists():
        raise SystemExit(
            f"Missing {RESULTS_CSV}. Run scripts/challenge2_parse_logs.sh first."
        )
    with RESULTS_CSV.open() as handle:
        rows = list(csv.DictReader(handle))

    enriched = []
    for row in rows:
        cmd = read_cmd(row["log"])
        row = dict(row)
        row["steps"] = parse_steps(row["log"])
        row["cuda_graph_impl"] = extract_flag(cmd, "--cuda-graph-impl")
        row["cuda_graph_scope"] = extract_multi_flag(cmd, "--cuda-graph-scope")
        row["recompute_granularity"] = extract_flag(cmd, "--recompute-granularity")
        row["recompute_method"] = extract_flag(cmd, "--recompute-method")
        row["recompute_num_layers"] = extract_flag(cmd, "--recompute-num-layers")
        row["optimizer_cpu_offload"] = has_flag(cmd, "--optimizer-cpu-offload")
        row["optimizer_offload_fraction"] = extract_flag(cmd, "--optimizer-offload-fraction")
        row["main_params_dtype"] = extract_flag(cmd, "--main-params-dtype")
        row["main_grads_dtype"] = extract_flag(cmd, "--main-grads-dtype")
        row["exp_avg_dtype"] = extract_flag(cmd, "--exp-avg-dtype")
        row["exp_avg_sq_dtype"] = extract_flag(cmd, "--exp-avg-sq-dtype")
        row["cmd"] = cmd
        enriched.append(row)
    return enriched


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def valid_metric_rows(rows):
    return [r for r in rows if r.get("avg_last10_tokens_sec_gpu")]


def best_by_config(rows):
    groups = defaultdict(list)
    for row in valid_metric_rows(rows):
        key = (
            row["model"],
            row["nodes"],
            row["tp"],
            row["pp"],
            row["mbs"],
            row["gbs"],
            row["seq_len"],
            row["steps"],
        )
        groups[key].append(row)

    best_rows = []
    for key, group_rows in sorted(groups.items()):
        best = max(group_rows, key=lambda r: as_float(r["avg_last10_tokens_sec_gpu"]) or 0)
        best_rows.append(
            {
                "model": key[0],
                "nodes": key[1],
                "tp": key[2],
                "pp": key[3],
                "mbs": key[4],
                "gbs": key[5],
                "seq_len": key[6],
                "steps": key[7],
                "best_tag": best["tag"],
                "avg_last10_tokens_sec_gpu": best["avg_last10_tokens_sec_gpu"],
                "best_tokens_sec_gpu": best["best_tokens_sec_gpu"],
                "best_tflops_gpu": best["best_tflops_gpu"],
                "log": best["log"],
            }
        )
    return best_rows


def baseline_candidates(rows, target):
    same_shape = [
        r
        for r in valid_metric_rows(rows)
        if r["model"] == target["model"]
        and r["nodes"] == target["nodes"]
        and r["tp"] == target["tp"]
        and r["pp"] == target["pp"]
        and r["mbs"] == target["mbs"]
        and r["gbs"] == target["gbs"]
        and r["seq_len"] == target["seq_len"]
        and r["steps"] == target["steps"]
    ]

    tag = target["tag"]
    cgtest = re.match(r"cgtest-(.+)-mbs(\d+)-", tag)
    if cgtest:
        base_tag = f"cgtest-{cgtest.group(1)}-mbs{cgtest.group(2)}-baseline"
        exact = [r for r in same_shape if r["tag"] == base_tag]
        if exact:
            return exact

    if tag.startswith("repeat-cg-"):
        exact = [r for r in same_shape if r["tag"] == "repeat-baseline-8b"]
        if exact:
            return exact

    if tag == "cg-te-full-layer-8b":
        exact = [r for r in same_shape if r["tag"] == "baseline-8b-100"]
        if exact:
            return exact

    return [r for r in same_shape if "baseline" in r["tag"] or re.match(r"mbs\d+-", r["tag"])]


def cuda_speedups(rows):
    out = []
    graph_rows = [
        r
        for r in valid_metric_rows(rows)
        if r["cuda_graph_impl"] or "cg-" in r["tag"] or "-te-" in r["tag"]
    ]
    for row in graph_rows:
        baselines = baseline_candidates(rows, row)
        if not baselines:
            continue
        baseline_values = [
            as_float(b["avg_last10_tokens_sec_gpu"])
            for b in baselines
            if as_float(b["avg_last10_tokens_sec_gpu"]) is not None
        ]
        if not baseline_values:
            continue
        baseline_avg = statistics.mean(baseline_values)
        graph_value = as_float(row["avg_last10_tokens_sec_gpu"])
        if graph_value is None or baseline_avg == 0:
            continue
        out.append(
            {
                "model": row["model"],
                "tag": row["tag"],
                "steps": row["steps"],
                "nodes": row["nodes"],
                "tp": row["tp"],
                "pp": row["pp"],
                "mbs": row["mbs"],
                "cuda_graph_impl": row["cuda_graph_impl"] or "unknown",
                "cuda_graph_scope": row["cuda_graph_scope"] or "full-layer/default",
                "baseline_tags": ";".join(sorted({b["tag"] for b in baselines})),
                "baseline_avg_last10": f"{baseline_avg:.0f}",
                "graph_avg_last10": row["avg_last10_tokens_sec_gpu"],
                "speedup_pct": f"{((graph_value / baseline_avg) - 1.0) * 100:.2f}",
                "log": row["log"],
            }
        )
    return sorted(out, key=lambda r: (r["model"], r["mbs"], r["tag"]))


def failure_summary(rows):
    counts = Counter((r["model"], r["status"]) for r in rows)
    out = []
    for (model, status), count in sorted(counts.items()):
        out.append({"model": model, "status": status, "count": str(count)})
    return out


def markdown_table(rows, fields):
    if not rows:
        return "_No rows._\n"
    lines = []
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
    for row in rows:
        values = [str(row.get(field, "")).replace("|", "\\|") for field in fields]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_markdown(rows, best_rows, speedup_rows, failures):
    report = OUT_DIR / "challenge2_summary.md"
    valid = valid_metric_rows(rows)
    top = sorted(
        valid,
        key=lambda r: as_float(r["avg_last10_tokens_sec_gpu"]) or 0,
        reverse=True,
    )[:20]
    with report.open("w") as handle:
        handle.write("# Challenge 2 Results Summary\n\n")
        handle.write("Generated from `results/challenge2_results.csv`.\n\n")
        handle.write("## Top Throughput Runs\n\n")
        handle.write(
            markdown_table(
                top,
                [
                    "model",
                    "tag",
                    "nodes",
                    "tp",
                    "pp",
                    "mbs",
                    "avg_last10_tokens_sec_gpu",
                    "best_tflops_gpu",
                    "status",
                ],
            )
        )
        handle.write("\n## Best Run Per Shape\n\n")
        handle.write(
            markdown_table(
                best_rows,
                [
                    "model",
                    "nodes",
                    "tp",
                    "pp",
                    "mbs",
                    "steps",
                    "best_tag",
                    "avg_last10_tokens_sec_gpu",
                    "best_tflops_gpu",
                ],
            )
        )
        handle.write("\n## CUDA Graph Speedups\n\n")
        handle.write(
            markdown_table(
                speedup_rows,
                [
                    "model",
                    "tag",
                    "mbs",
                    "cuda_graph_impl",
                    "cuda_graph_scope",
                    "baseline_avg_last10",
                    "graph_avg_last10",
                    "speedup_pct",
                ],
            )
        )
        handle.write("\n## Status Counts\n\n")
        handle.write(markdown_table(failures, ["model", "status", "count"]))


def main():
    OUT_DIR.mkdir(exist_ok=True)
    rows = load_rows()
    enriched_fields = list(rows[0].keys()) if rows else []
    write_csv(OUT_DIR / "challenge2_enriched_results.csv", rows, enriched_fields)

    best_rows = best_by_config(rows)
    write_csv(
        OUT_DIR / "challenge2_best_by_config.csv",
        best_rows,
        [
            "model",
            "nodes",
            "tp",
            "pp",
            "mbs",
            "gbs",
            "seq_len",
            "steps",
            "best_tag",
            "avg_last10_tokens_sec_gpu",
            "best_tokens_sec_gpu",
            "best_tflops_gpu",
            "log",
        ],
    )

    speedup_rows = cuda_speedups(rows)
    write_csv(
        OUT_DIR / "challenge2_cuda_speedups.csv",
        speedup_rows,
        [
            "model",
            "tag",
            "steps",
            "nodes",
            "tp",
            "pp",
            "mbs",
            "cuda_graph_impl",
            "cuda_graph_scope",
            "baseline_tags",
            "baseline_avg_last10",
            "graph_avg_last10",
            "speedup_pct",
            "log",
        ],
    )

    failures = failure_summary(rows)
    write_csv(OUT_DIR / "challenge2_status_counts.csv", failures, ["model", "status", "count"])
    write_markdown(rows, best_rows, speedup_rows, failures)

    print("Wrote:")
    for name in [
        "challenge2_enriched_results.csv",
        "challenge2_best_by_config.csv",
        "challenge2_cuda_speedups.csv",
        "challenge2_status_counts.csv",
        "challenge2_summary.md",
    ]:
        print(f"  results/{name}")


if __name__ == "__main__":
    main()
