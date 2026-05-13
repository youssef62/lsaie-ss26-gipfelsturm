import re
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs", "run-full-recompute-vs-full-offload")

CONFIGS = [
    ("full-offload-mbs2",    "22b-25s-1n-tp4-full-offload-mbs2",    "offload",    2),
    ("full-recompute-mbs2",  "22b-25s-1n-tp4-full-recompute-mbs2",  "recompute",  2),
    ("full-offload",         "22b-25s-1n-tp4-full-offload-mbs1",    "offload",    1),
    ("full-recompute",       "22b-25s-1n-tp4-full-recompute-mbs1",  "recompute",  1),
]

WARMUP_STEPS = 10
COLOR = {"offload": "#4472C4", "recompute": "#ED7D31"}


def find_log(prefix):
    for fname in os.listdir(LOG_DIR):
        if fname.startswith(prefix):
            return os.path.join(LOG_DIR, fname)
    raise FileNotFoundError(f"No log found for prefix {prefix!r} in {LOG_DIR}")


def parse_mfu(path):
    values = []
    with open(path) as f:
        for line in f:
            m = re.search(r"MFU:\s*([0-9.]+)%", line)
            if m:
                values.append(float(m.group(1)))
    return values


labels, means, colors = [], [], []
for label, prefix, kind, mbs in CONFIGS:
    path = find_log(prefix)
    mfu_vals = parse_mfu(path)
    stable = mfu_vals[WARMUP_STEPS:]
    mean_mfu = np.mean(stable) if stable else np.mean(mfu_vals)
    labels.append(label)
    means.append(mean_mfu)
    colors.append(COLOR[kind])
    print(f"{label}: n={len(stable)} steps, mean MFU = {mean_mfu:.2f}%")

fig, ax = plt.subplots(figsize=(7, 3.2))
y = np.arange(len(labels))
bars = ax.barh(y, means, color=colors, height=0.55, edgecolor="none")

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("mfu", fontsize=9)
ax.set_xlim(0, max(means) * 1.18)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))

for bar, val in zip(bars, means):
    ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", ha="left", fontsize=8)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.invert_yaxis()

fig.suptitle("MFU — Full Offload vs. Full Recompute (TP=4, 23B model, 1 node)", fontsize=10, y=1.01)
fig.tight_layout()

out = os.path.join(os.path.dirname(__file__), "mfu_recompute_vs_offload.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
