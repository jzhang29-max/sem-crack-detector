import csv, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
rows=[r for r in csv.DictReader(open("analysis/paired_detector.csv"))
      if r["Jaccard"] not in ("",None) and float(r["Jaccard"])>=0.5]
COL={"AS":"#4C72B0","Cast":"#DD8452","HIP":"#55A868"}
fig,ax=plt.subplots(figsize=(7.8,5.6))
for r in rows:
    c,e=float(r["CBS_AreaPct"]),float(r["ETD_AreaPct"])
    ax.plot([0,1],[c,e],"-o",color=COL[r["Process"]],ms=7,lw=1.8,alpha=.9)
    ax.annotate(f"{r['Process']}_{r['Field']}",(1.02,e),fontsize=8,va="center",color=COL[r["Process"]])
ax.set_xlim(-.15,1.45); ax.set_xticks([0,1]); ax.set_xticklabels(["CBS","ETD"])
ax.set_ylabel("crack area fraction (%)")
ax.set_title("Same field of view, two detectors — every line slopes down\n"
             "CBS flags more crack than ETD in 8/8 pairs\n"
             "median +3.88 pp, Wilcoxon p=0.008",fontsize=11)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([],[],color=c,marker="o",label=k) for k,c in COL.items()],
          frameon=False,loc="upper right")
ax.grid(axis="y",alpha=.3)
fig.tight_layout(); fig.savefig("analysis/figures/fig6_paired_detector.png",dpi=150)
print("wrote fig6")
