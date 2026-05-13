import re
import glob
import numpy as np
import matplotlib.pyplot as plt

logs = glob.glob("logs/*.log")

data = {}
for path in logs:
    name = re.search(r"(dp\d+-tp\d+-pp\d+)", path).group(1)
    values = [float(m) for m in re.findall(r"MFU:\s+([\d.]+)%", open(path, errors="ignore").read())]
    if values:
        data[name] = sum(values[-5:]) / min(5, len(values))

def sort_key(item):
    has4 = int(any(int(v) == 4 for v in re.findall(r"\d+", item[0])))
    return (has4, item[1])

labels, means = zip(*sorted(data.items(), key=sort_key))
colors = plt.cm.tab10.colors[:len(labels)]
x = np.arange(len(labels)) * 0.65

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(x, means, width=0.4, color=colors)
ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9)
ax.set_xticks(x)
def fmt_label(s):
    vals = {k: int(v) for k, v in zip(["DP", "TP", "PP"], re.findall(r"[a-z]+(\d+)", s))}
    if 4 in vals.values():
        return "\n".join(f"{k}={v}" for k, v in vals.items() if v == 4)
    return "\n".join(f"{k}={v}" for k, v in vals.items() if v == 2)

ax.set_xticklabels([fmt_label(l) for l in labels])
ax.set_ylabel("MFU (%)")
ax.set_title("Avg last-5-step MFU per parallelism strategy\n(8B model, 4 GPUs)")
ax.set_ylim(0, max(means) * 1.18)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
ax.spines[["top", "right"]].set_visible(False)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("throughput_comparison.png", dpi=150)
print("Saved throughput_comparison.png")
