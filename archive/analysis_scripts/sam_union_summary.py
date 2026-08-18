"""
Summarise the 63-image SAM-union sweep into one figure.

The headline question is not "how much does the union add" but "is what it
adds real". Those diverge sharply here: the union adds a visible amount of
area, and every pixel of it that has a human verdict is wrong. So the ground
-truth panel is the one that matters and is placed first.

    python3 sam_union_summary.py
Writes ../../figures/sam_union_summary.png
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import PROJECT_ROOT

STATS = os.path.join(PROJECT_ROOT, "figures", "sam_union", "sam_union_stats.csv")
OUT = os.path.join(PROJECT_ROOT, "figures", "sam_union_summary.png")


def main():
    d = pd.read_csv(STATS).sort_values("SourceImage").reset_index(drop=True)
    d["is_mar"] = d.SourceImage.str.startswith("MAR")
    g = d.dropna(subset=["sam_only_on_crack"])
    oc = int(g.sam_only_on_crack.sum())
    on = int(g.sam_only_on_notcrack.sum())
    ou = int(g.sam_only_unreviewed.sum())

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.2))

    # --- 1. the verdict ---
    ax = axes[0]
    bars = ["on confirmed\nCRACK\n(real gain)", "on confirmed\nNOT-crack\n(false positive)",
            "never\nreviewed\n(unknown)"]
    vals = [oc, on, ou]
    cols = ["#27ae60", "#c0392b", "#95a5a6"]
    b = ax.bar(bars, vals, color=cols)
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + max(vals) * 0.015, f"{v:,}",
                ha="center", fontsize=10, weight="bold")
    ax.set_ylabel("pixels SAM adds beyond the pipeline")
    adj = oc + on
    sub = (f"of adjudicated pixels: {100.0*oc/adj:.1f}% real crack, {100.0*on/adj:.1f}% false positive"
           if adj else "no adjudicated pixels")
    ax.set_title(f"What the union ADDS, vs your correction masks\n"
                 f"{len(g)} reviewed images -- {sub}")
    ax.grid(alpha=0.3, axis="y")

    # --- 2. per-image area, pipeline vs union ---
    ax = axes[1]
    y = np.arange(len(d))
    ax.barh(y, d.pipeline_area_pct, color="#c0392b", label="pipeline crack area")
    ax.barh(y, d.sam_only_area_pct, left=d.pipeline_area_pct, color="#f1c40f",
            label="added by SAM (union)")
    ax.set_yticks(y)
    ax.set_yticklabels([s if len(s) <= 26 else ".." + s[-24:] for s in d.SourceImage],
                       fontsize=5.6)
    for t, mar in zip(ax.get_yticklabels(), d.is_mar):
        t.set_color("#8e44ad" if mar else "#2c3e50")
    ax.set_xlabel("% of image area")
    ax.set_title(f"Per-image crack area, all {len(d)} images\n"
                 "purple labels = MAR superalloy, dark = steel")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="x")

    # --- 3. relative gain distribution ---
    ax = axes[2]
    for sub_d, lbl, c in [(d[~d.is_mar], f"steel (n={int((~d.is_mar).sum())})", "#2980b9"),
                          (d[d.is_mar], f"MAR (n={int(d.is_mar.sum())})", "#8e44ad")]:
        if len(sub_d):
            v = np.sort(sub_d.union_gain_pct_of_pipeline.values)
            ax.plot(v, np.linspace(0, 100, len(v)), lw=2.2, marker="o", ms=3, label=lbl, color=c)
    ax.axvline(d.union_gain_pct_of_pipeline.median(), color="k", ls="--", lw=1.3,
               label=f"overall median {d.union_gain_pct_of_pipeline.median():.1f}%")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("SAM-only area, % of pipeline crack area")
    ax.set_ylabel("cumulative % of images")
    ax.set_title("How much the union adds\n(larger is WORSE: adjudicated additions are"
                 " false positives)")
    ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")

    print(f"\n{len(d)} images | pipeline {int(d.pipeline_px.sum()):,} px, "
          f"union {int(d.union_px.sum()):,} px, SAM-only {int(d.sam_only_px.sum()):,} px")
    print(f"median SAM-only gain: steel {d[~d.is_mar].union_gain_pct_of_pipeline.median():.2f}%, "
          f"MAR {d[d.is_mar].union_gain_pct_of_pipeline.median():.2f}%")
    print(f"SAM masks: {int(d.sam_masks_accepted.sum()):,} accepted of "
          f"{int(d.sam_masks_scored.sum()):,} scored "
          f"({int(d.rej_too_big.sum()):,} rejected as tile-background, "
          f"{int(d.rej_border.sum()):,} as border-hugging)")
    if adj:
        print(f"\nGROUND TRUTH: {oc:,} px on confirmed crack / {on:,} px on confirmed not-crack")
        print(f"  -> {100.0*oc/adj:.1f}% of adjudicated additions are real crack")


if __name__ == "__main__":
    main()
