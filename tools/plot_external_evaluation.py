"""Plot external AP at a common 640-pixel evaluation input."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"docs/experiments/external-640"
rows=list(csv.DictReader((DATA/"comparison.csv").open()))
runs=["n640","s640","m640","n960","s960"]
labels=["YOLO26n / 640", "YOLO26s / 640", "YOLO26m / 640", "YOLO26n / 960", "YOLO26s / 960"]
fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout="constrained",sharey=True)
fig.suptitle("External evaluation at 640 px (labels show training resolution)", fontsize=12)
for ax,dataset,cls,title in [(axes[0],"bollard_dev_disjoint","bollard","Bollard / AP50-95"),(axes[1],"nightowls_rec34","person","Night pedestrian / AP50-95")]:
    values=[float(next(r['ap50_95'] for r in rows if r['run']==run and r['dataset']==dataset and r['class']==cls)) for run in runs]
    colors=["#2563eb" if v==max(values) else "#94a3b8" for v in values]
    ax.barh(range(5),values,height=.53,color=colors)
    for i,v in enumerate(values):ax.text(v+.012,i,f"{v:.5f}",va="center",fontweight="bold")
    ax.set_xlim(0,.7)
    ax.set_title(title,loc="left",pad=15)
    ax.set_axisbelow(True);ax.grid(axis="x",alpha=.15)
    for edge in ['top','right','left']:ax.spines[edge].set_visible(False)
    ax.tick_params(axis="y",length=0)
axes[0].set_yticks(range(5),labels)
axes[0].invert_yaxis()
fig.savefig(DATA/"comparison.png",dpi=180)
plt.close(fig)
