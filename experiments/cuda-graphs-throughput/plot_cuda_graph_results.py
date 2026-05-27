#!/usr/bin/env python3
"""Generate summary plots for the CUDA graph throughput experiment.

The script intentionally uses Pillow instead of matplotlib so it can run in the
minimal local tooling environment used for the report handoff.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def draw_title(draw: ImageDraw.ImageDraw, title: str, subtitle: str, width: int) -> None:
    title_font = font(30, bold=True)
    sub_font = font(17)
    draw.text((40, 26), title, fill=(24, 24, 24), font=title_font)
    draw.text((40, 66), subtitle, fill=(88, 88, 88), font=sub_font)
    draw.line((40, 102, width - 40, 102), fill=(205, 205, 205), width=2)


def save_speedup_plot() -> None:
    rows = [r for r in read_csv("challenge2_cuda_speedups.csv") if r["speedup_pct"]]
    grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append((row["cuda_graph_scope"], float(row["speedup_pct"])))

    models = ["125m", "350m", "760m", "8b"]
    values: list[tuple[str, str, float]] = []
    for model in models:
        for scope, speedup in grouped.get(model, []):
            values.append((model, scope, speedup))

    top, row_h = 136, 32
    width = 1400
    height = max(900, 180 + len(values) * row_h + 80)
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw_title(
        draw,
        "CUDA graph speedup by model and capture scope",
        "Speedup is computed from average tokens/sec/GPU over the last 10 logged iterations.",
        width,
    )

    left, right = 330, width - 90
    zero_x = left + int((0 - (-35)) / (10 - (-35)) * (right - left))
    draw.line((zero_x, top - 12, zero_x, height - 70), fill=(70, 70, 70), width=2)

    for pct in range(-30, 11, 10):
        x = left + int((pct - (-35)) / (10 - (-35)) * (right - left))
        draw.line((x, top - 12, x, height - 86), fill=(232, 232, 232), width=1)
        draw.text((x - 16, height - 72), f"{pct}%", fill=(95, 95, 95), font=font(14))

    label_font = font(15)
    small_font = font(13)
    colors = {
        "attn": (58, 123, 213),
        "mlp": (42, 157, 143),
        "attn mlp": (111, 86, 189),
        "full-layer/default": (229, 126, 49),
        "full_iteration": (191, 69, 69),
    }

    y = top
    last_model = None
    for model, scope, speedup in values:
        if last_model is not None and model != last_model:
            y += 18
            draw.line((40, y - 8, width - 40, y - 8), fill=(238, 238, 238), width=1)
        if model != last_model:
            draw.text((40, y + 2), model, fill=(24, 24, 24), font=font(17, bold=True))
            last_model = model

        label = scope.replace("full-layer/default", "TE full-layer").replace("full_iteration", "local full-iteration")
        draw.text((108, y + 4), label, fill=(44, 44, 44), font=label_font)
        x = left + int((speedup - (-35)) / (10 - (-35)) * (right - left))
        x0, x1 = sorted((zero_x, x))
        draw.rounded_rectangle((x0, y, x1, y + 22), radius=4, fill=colors.get(scope, (100, 100, 100)))
        text_x = x + 8 if speedup >= 0 else x - 62
        draw.text((text_x, y + 2), f"{speedup:+.2f}%", fill=(24, 24, 24), font=small_font)
        y += row_h

    PLOTS.mkdir(parents=True, exist_ok=True)
    img.save(PLOTS / "cuda_graph_speedups_by_model.png")


def save_throughput_plot() -> None:
    rows = [
        r
        for r in read_csv("challenge2_results.csv")
        if r["avg_last10_tokens_sec_gpu"] and r["model"] in {"125m", "350m", "760m", "8b"}
    ]
    rows = sorted(rows, key=lambda r: float(r["avg_last10_tokens_sec_gpu"]), reverse=True)[:18]

    width, height = 1400, 820
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw_title(
        draw,
        "Top measured throughput runs",
        "Average tokens/sec/GPU over the last 10 logged iterations.",
        width,
    )

    left, right = 420, width - 110
    top, row_h = 134, 34
    max_value = max(float(r["avg_last10_tokens_sec_gpu"]) for r in rows)
    for tick in range(0, int(max_value) + 10000, 10000):
        x = left + int(tick / max_value * (right - left))
        draw.line((x, top - 12, x, height - 70), fill=(234, 234, 234), width=1)
        draw.text((x - 22, height - 58), f"{tick//1000}k", fill=(95, 95, 95), font=font(14))

    label_font = font(14)
    value_font = font(13)
    for idx, row in enumerate(rows):
        y = top + idx * row_h
        label = f"{row['model']}  {row['tag']}"
        value = float(row["avg_last10_tokens_sec_gpu"])
        x = left + int(value / max_value * (right - left))
        fill = (58, 123, 213) if "cg" in row["tag"] else (110, 110, 110)
        draw.text((40, y + 3), label[:48], fill=(38, 38, 38), font=label_font)
        draw.rounded_rectangle((left, y, x, y + 23), radius=4, fill=fill)
        draw.text((x + 8, y + 3), f"{value:,.0f}", fill=(24, 24, 24), font=value_font)

    PLOTS.mkdir(parents=True, exist_ok=True)
    img.save(PLOTS / "top_throughput_runs.png")


def main() -> None:
    save_speedup_plot()
    save_throughput_plot()
    print("Wrote plots to", PLOTS)


if __name__ == "__main__":
    main()
