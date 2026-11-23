"""Redraw proposal figures from the archived benchmark and epoch CSV files."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/experiments/preliminary"
OUT = ROOT / "docs/assets/experiments"
rows = list(csv.DictReader((DATA / "benchmark.csv").open()))
colors = ["#64748b", "#2563eb", "#0f766e", "#7c3aed", "#94a3b8", "#b45309"]
labels = [f"{r['model']} / {r['imgsz']}" + (" / no negatives" if "noneg" in r['name'] else "") for r in rows]
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":11, "axes.spines.top":False, "axes.spines.right":False, "axes.titleweight":"bold", "figure.facecolor":"white"})

fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
positions = [(.88, .732), (1.26, .787), (1.50, .821), (2.10, .858), (1.48, .750), (2.16, .810)]
for i, row in enumerate(rows):
    x, y = float(row["inf_ms"]), float(row["v2146_map50"])
    ax.scatter(x, y, s=115, color=colors[i], edgecolor="white", linewidth=1.5, zorder=4)
    label = labels[i].replace(" / no negatives", " (no negatives)")
    ax.annotate(f"{label}\n{x:.2f} ms | {y:.4f}", (x, y), xytext=positions[i],
                textcoords="data", fontsize=11, color=colors[i],
                arrowprops={"arrowstyle": "-", "color": colors[i], "alpha": .5, "lw": 1},
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 3}, zorder=5)
ax.set(xlabel="Inference time (ms / image, desktop)", ylabel="mAP50 (person + bollard)",
       xlim=(.8, 2.9), ylim=(.72, .88))
ax.set_title("Model accuracy vs. inference time", loc="left", pad=18, fontsize=16)
ax.grid(alpha=.18)
fig.savefig(OUT/'proposal-model-resolution.png',dpi=180)
plt.close(fig)

bins=['<4px','4~8px','8~16px','>16px']
values=np.array([[float(r['v2146_bollard_R_'+b]) for b in bins] for r in rows])
fig,ax=plt.subplots(figsize=(10,4.8),layout="constrained")
ax.imshow(values,cmap="Blues",vmin=0,vmax=1,aspect="auto")
ax.set_xticks(range(4),bins)
ax.set_yticks(range(6),[f"{label} ({r['epochs']} ep)" for label,r in zip(labels,rows)])
for i in range(6):
    for j in range(4):ax.text(j,i,f"{values[i,j]:.4f}",ha="center",va="center",fontsize=13,color="white" if values[i,j]>.65 else "#0f172a")
ax.set(xlabel="Ground-truth bollard width (source image pixels)")
ax.set_title("Bollard recall by object size",loc="left",pad=15)
fig.savefig(OUT/'proposal-bollard-size.png',dpi=180)
plt.close(fig)

fig,ax=plt.subplots(figsize=(10,5),layout="constrained")
for i,r in enumerate(rows):
    curve=list(csv.DictReader((DATA/(r['name']+'.csv')).open()))
    ax.plot([int(v['epoch']) for v in curve],[float(v['metrics/mAP50(B)']) for v in curve],label=f"{labels[i]} ({r['epochs']} ep)",color=colors[i],lw=1.8)
ax.set(xlabel="Epoch",ylabel="Development mAP50",title="Historical training curves",ylim=(.5,.92))
ax.grid(alpha=.18)
ax.legend(loc="lower right",fontsize=9)
fig.savefig(OUT/'proposal-learning-curves.png',dpi=180)
plt.close(fig)
print("Redrew three figures from archived CSV values.")
