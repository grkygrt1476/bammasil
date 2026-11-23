"""Plot the five completed YOLO26 runs from their published summary."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
runs = json.loads((ROOT / "docs/experiments/yolo26-summary.json").read_text(encoding="utf-8"))["runs"]
labels = [f"YOLO26{k[0]} / {r['settings']['imgsz']}" for k, r in runs.items()]
colors = ["#2563eb" if k == "s960" else "#94a3b8" for k in runs]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold"})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained", sharey=True)
series = [([r["metrics"]["metrics/mAP50-95(B)"] for r in runs.values()], "Development mAP50-95", .8, ".5f"), ([r["elapsed_seconds"]/60 for r in runs.values()], "Training time / minutes", 165, ".1f")]
for ax, (values, title, limit, fmt) in zip(axes, series):
    ax.barh(range(5), values, height=.53, color=colors)
    for i, value in enumerate(values):
        ax.text(value + limit*.025, i, format(value, fmt), va="center", fontsize=12, fontweight="bold", color="#0f172a")
    ax.set_xlim(0, limit)
    ax.set_title(title, loc="left", pad=18, fontsize=14)
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=.15)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", length=0)
axes[0].set_yticks(range(5), labels)
axes[0].invert_yaxis()
fig.savefig(ROOT / "docs/assets/experiments/yolo26-comparison.png", dpi=180)
plt.close(fig)
