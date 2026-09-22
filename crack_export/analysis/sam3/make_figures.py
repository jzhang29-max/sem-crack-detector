#!/usr/bin/env python3
"""Figures for the leak-gated SAM 3 run. Replaces sam3_examples.png, which had NO generator
in the repo and was built from the contaminated input.

  fig_presence_gate.png  -- the finding: one global scalar per (image, prompt) decides whether
                            an image returns anything at all
  fig_leak_evidence.png  -- the postmortem's evidence: the old input beside the new one
  (fig_qualitative.png was removed in the 2026-09-19 cleanup along with masks/: it showed the
   SAM 3 tile arm, which is closed, and detector_contact_sheet.png supersedes it.)

Every panel is generated from committed artifacts so it can be rebuilt.
"""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

SC = os.path.dirname(os.path.abspath(__file__))
PROMPTS = ["crack", "a crack in metal", "thin dark line", "fracture"]
TAU = 0.3


def fig_presence_gate():
    pres = json.load(open(f"{SC}/sam3_presence.json"))
    res = {(r["tile"], r["prompt"]): r for r in json.load(open(f"{SC}/sam3_results.json"))
           if "error" not in r}
    tiles = sorted({k.split("|")[0] for k in pres})
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1.35, 1]})

    # LEFT: gated score per (tile, prompt) against tau
    cols = {"crack": "#1b6ca8", "a crack in metal": "#f0a202",
            "thin dark line": "#7d4f9c", "fracture": "#c1292e"}
    for p in PROMPTS:
        y = [pres[f"{t}|{p}"]["gated_max"] for t in tiles if f"{t}|{p}" in pres]
        x = np.arange(len(y))
        empty = [res.get((t, p), {}).get("n", 0) == 0 for t in tiles if f"{t}|{p}" in pres]
        ax[0].plot(x, y, "o-", color=cols[p], lw=1.2, ms=5, label=p, alpha=0.9)
        xe = x[np.array(empty)]
        ye = np.array(y)[np.array(empty)]
        ax[0].plot(xe, ye, "x", color="k", ms=9, mew=1.8,
                   label="returned 0 instances" if p == "crack" else None)
    ax[0].axhline(TAU, color="k", ls="--", lw=1.4)
    ax[0].text(0.3, TAU + 0.015, r"confidence threshold $\tau=0.3$", fontsize=9)
    ax[0].set_yscale("log")
    # 15, not 16: the 16-tile set is the withdrawn contaminated run. sam3_presence.json,
    # which this figure plots, holds 15 distinct tiles.
    ax[0].set_xlabel("tile (15 hand-labelled SEM tiles)")
    ax[0].set_ylabel(r"$s_i \cdot \max_j q_{ij}$  (gated top score)")
    ax[0].set_title("Whether an image returns ANYTHING is set by $s_i\\cdot\\max_j q_{ij}$\n"
                    "black x = returned nothing. This separation is DEFINITIONAL, not a result",
                    fontsize=10)
    ax[0].legend(fontsize=8, loc="lower right")
    ax[0].grid(alpha=0.25)

    # RIGHT: the presence scalar by prompt
    data = [[pres[f"{t}|{p}"]["presence"] for t in tiles if f"{t}|{p}" in pres] for p in PROMPTS]
    bp = ax[1].boxplot(data, vert=True, patch_artist=True, widths=0.6,
                       medianprops=dict(color="k", lw=1.6))
    for patch, p in zip(bp["boxes"], PROMPTS):
        patch.set_facecolor(cols[p]); patch.set_alpha(0.55)
    for i, d in enumerate(data, 1):
        ax[1].plot(np.full(len(d), i) + np.linspace(-0.13, 0.13, len(d)), d, ".",
                   color="k", ms=3.5, alpha=0.6)
    ax[1].set_yscale("log")
    ax[1].set_xticks(range(1, len(PROMPTS) + 1))
    ax[1].set_xticklabels([p.replace(" ", "\n") for p in PROMPTS], fontsize=8)
    ax[1].set_ylabel(r"global presence scalar $s_i$")
    med = {p: float(np.median(d)) for p, d in zip(PROMPTS, data)}
    # WRAPPED ONTO THREE LINES. As one long line this ran off the right edge of the figure
    # and the rendered PNG stopped mid-word at "...between the two rea", so the shipped
    # artefact's caption was truncated where the qualifier lives -- the half that says the
    # 105x headline is carried by one out-of-vocabulary noun.
    ax[1].set_title("THE MEASUREMENT: prompt sensitivity is one scalar\n"
                    f"median {med['crack']:.3f} → {med['fracture']:.4f} = "
                    f"{med['crack']/med['fracture']:.1f}×\n"
                    f"but 6.3× without 'fracture', 3.0× between the two real synonyms",
                    fontsize=9.5)
    ax[1].grid(alpha=0.25, axis="y")
    fig.suptitle("SAM 3 on SEM cracks: the presence head collapses on one out-of-vocabulary noun "
                 "while the decoder barely moves (max$_j q_{ij}$ spans 1.31×)", fontsize=11, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{SC}/fig_presence_gate.png", dpi=160)
    print(f"  fig_presence_gate.png   (medians: " +
          ", ".join(f"{p}={med[p]:.4f}" for p in PROMPTS) + ")")


def fig_leak_evidence():
    meta = json.load(open(f"{SC}/tiles/meta.json"))
    # pick the tile with the largest burn-in coverage, and one clean-by-accident tile
    pick = [m for m in meta if m["frame"] == "AS_24hr_BSE_Side_008"][:1] + \
           [m for m in meta if m["frame"] == "260622_316_H_b4_CBS_02"][:1]
    fig, axes = plt.subplots(len(pick), 4, figsize=(13, 3.4 * len(pick)))
    axes = np.atleast_2d(axes)
    for r, m in enumerate(pick):
        t = m["tile"]
        ovl = np.array(Image.open(f"{SC}/tiles/{t}_overlay_REFERENCE_DO_NOT_FEED.png").convert("RGB"))
        old = ovl[..., 1]
        new = np.array(Image.open(f"{SC}/tiles/{t}_gray.png").convert("L"))
        gt = np.array(Image.open(f"{SC}/tiles/{t}_gt.png")) > 127
        for c, (im, ttl, kw) in enumerate([
            (ovl, "annotated overlay\n(reference only)", {}),
            (old, f"THE INVALID INPUT\ngreen channel; std inside label = {old[gt].std():.2f}",
             dict(cmap="gray", vmin=0, vmax=255)),
            (new, f"the clean input\nraw original; std inside label = {new[gt].std():.2f}",
             dict(cmap="gray", vmin=0, vmax=255)),
            (gt, "hand label (scored)", dict(cmap="gray")),
        ]):
            axes[r, c].imshow(im, **kw)
            axes[r, c].set_title(ttl, fontsize=8.5)
            axes[r, c].axis("off")
        axes[r, 0].set_ylabel(t, fontsize=7)
    fig.suptitle("Why every score from the first run is void: the label was written into the "
                 "model input\n"
                 "top row: burn-in covers 100% of the label (std exactly 0) · "
                 "bottom row: this frame's overlay is disjoint from its correction mask",
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(f"{SC}/fig_leak_evidence.png", dpi=150)
    print("  fig_leak_evidence.png")


def best_threshold_mask(g, gt):
    best = (-1, None)
    for pol in ("dark", "bright"):
        for th in range(0, 256, 2):
            p = (g <= th) if pol == "dark" else (g >= th)
            i = np.logical_and(p, gt).sum()
            if not i:
                continue
            iou = i / np.logical_or(p, gt).sum()
            if iou > best[0]:
                best = (iou, p)
    return best


def fig_qualitative():
    if not glob.glob(f"{SC}/masks/*.npz"):
        print("  (skipping fig_qualitative.png: no masks/ -- run SAM3_SAVE_MASKS=1 run_real.py)")
        return
    res = {(r["tile"], r["prompt"]): r for r in json.load(open(f"{SC}/sam3_results.json"))
           if "error" not in r}
    cand = sorted([t for (t, p) in res if p == "crack"],
                  key=lambda t: -res[(t, "crack")].get("union_IoU", 0))
    pick = cand[:2] + cand[-2:]
    fig, axes = plt.subplots(len(pick), 4, figsize=(13, 3.35 * len(pick)))
    for r, t in enumerate(pick):
        g = np.array(Image.open(f"{SC}/tiles/{t}_gray.png").convert("L"))
        gt = np.array(Image.open(f"{SC}/tiles/{t}_gt.png")) > 127
        tiou, tmask = best_threshold_mask(g, gt)
        f = f"{SC}/masks/{t}__crack.npz"
        union = np.load(f)["union"] if os.path.exists(f) else np.zeros_like(gt)
        sres = res[(t, "crack")]
        for c, (im, ttl) in enumerate([
            (g, f"{t[:30]}\nclean SEM input"),
            (gt, f"hand label ({100*gt.mean():.2f}% of tile)"),
            (tmask, f"best global threshold\nIoU {tiou:.3f}"),
            (union, f"SAM 3 'crack' union (n={sres.get('n',0)})\nIoU {sres.get('union_IoU',0):.3f}"),
        ]):
            axes[r, c].imshow(im, cmap="gray", **({"vmin": 0, "vmax": 255} if c == 0 else {}))
            axes[r, c].set_title(ttl, fontsize=8.5)
            axes[r, c].axis("off")
    fig.suptitle("Leak-gated SAM 3 versus the trivial baseline it must beat "
                 "(two best and two worst tiles by union IoU)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{SC}/fig_qualitative.png", dpi=150)
    print("  fig_qualitative.png")


if __name__ == "__main__":
    fig_presence_gate()
    fig_leak_evidence()
    # fig_qualitative() -- removed with masks/; see the module docstring
