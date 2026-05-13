import re
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_DIR = os.path.dirname(__file__)

# (label, log_path_glob_prefix, color, linestyle)
CONFIGS = [
    (
        "pp4-mixed-offload-recompute-mbs-1",
        "logs/run-pp-recompute-vs-partial-offload/22b-25s-1n-pp4-finegrained-offload-more-recompute",
        "#C00000", (0, (6, 1, 1, 1)),   # dash-dot, unique
    ),
    (
        "pp4-full-recompute-mbs-1",
        "logs/run-pp-recompute-vs-partial-offload/22b-25s-1n-tp1-pp4-full-recompute",
        "#C00000", "--",
    ),
    (
        "tp4-full-offload-mbs2",
        "logs/run-full-recompute-vs-full-offload/22b-25s-1n-tp4-full-offload-mbs2",
        "#4CAF50", "-",
    ),
    (
        "tp4-full-offload-mbs-1",
        "logs/run-full-recompute-vs-full-offload/22b-25s-1n-tp4-full-offload-mbs1",
        "#4CAF50", "--",
    ),
    (
        "tp4-full-recompute-mbs2",
        "logs/run-full-recompute-vs-full-offload/22b-25s-1n-tp4-full-recompute-mbs2",
        "#2196F3", "-",
    ),
    (
        "tp4-full-recompute-mbs-1",
        "logs/run-full-recompute-vs-full-offload/22b-25s-1n-tp4-full-recompute-mbs1",
        "#2196F3", "--",
    ),
]


def find_log(prefix):
    full_prefix = os.path.join(LOG_DIR, prefix)
    base_dir = os.path.dirname(full_prefix)
    base_name = os.path.basename(full_prefix)
    for fname in os.listdir(base_dir):
        if fname.startswith(base_name):
            return os.path.join(base_dir, fname)
    raise FileNotFoundError(f"No log found for prefix {prefix!r}")


def parse_steps_mfu(path):
    steps, mfus = [], []
    with open(path) as f:
        for line in f:
            m = re.search(r"iteration\s+(\d+)/\s*\d+.*MFU:\s*([0-9.]+)%", line)
            if m:
                steps.append(int(m.group(1)))
                mfus.append(float(m.group(2)))
    return steps, mfus


fig, ax = plt.subplots(figsize=(7, 4.5))

for label, prefix, color, ls in CONFIGS:
    path = find_log(prefix)
    steps, mfus = parse_steps_mfu(path)
    ax.plot(steps, mfus, color=color, linestyle=ls, linewidth=1.8, label=label)

ax.set_xlabel("Step", fontsize=9)
ax.set_ylabel("mfu", fontsize=9)
ax.set_title(
    "MFU across all experiments (23B model, 1 node)",
    fontsize=10,
)
ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
ax.grid(axis="y", color="lightgrey", linewidth=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
out = os.path.join(LOG_DIR, "mfu_curves.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
