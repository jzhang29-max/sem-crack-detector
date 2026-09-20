#!/usr/bin/env python3
"""One PDF with every frame's detector overlay and its binary mask, side by side.

Mirrors the convention of the hand-annotation exports (all_overlays.pdf, all_masks_bw.pdf) but
for the DETECTOR's output. One page per frame: orange overlay on the left, black-and-white mask
on the right, captioned with the frame name, clIoU_adapt where a hand label exists, and the
predicted area.

Pages are ordered worst-scoring LAST -- the low scorers are where the real disagreements are,
and two frames with nearly identical scores turned out to be a label-completeness artefact and
a genuine false-positive mode. Unlabelled frames go at the end, captioned as unlabelled rather
than given a number that would mean something different.

Masks are read at full resolution and downsampled for the page; the full-resolution binaries
stay in detector_masks/.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

SC = os.path.dirname(os.path.abspath(__file__))
LONG = 1500          # page raster long edge


def thumb(p, long_edge=LONG):
    im = Image.open(p)
    im.thumbnail((long_edge, long_edge), Image.LANCZOS)
    return np.array(im)


def main():
    rows = json.load(open(f"{SC}/detector_all.json"))
    scored = sorted([r for r in rows if "clIoU_adapt" in r], key=lambda r: -r["clIoU_adapt"])
    rest = sorted([r for r in rows if "clIoU_adapt" not in r], key=lambda r: -r["pred_frac"])
    allr = scored + rest
    out = f"{SC}/detector_all_frames.pdf"
    with PdfPages(out) as pdf:
        # contents page
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.text(.5, .93, "SEM crack detector — all 62 frames", ha="center", size=17)
        fig.text(.5, .89, "Meijering ridge, $\\sigma$ 1–4, top 1% of response, components $\\leq$ 32 px dropped",
                 ha="center", size=10, color="#555")
        ns = [r["clIoU_adapt"] for r in scored]
        fig.text(.5, .84, f"{len(allr)} frames · {len(scored)} with a hand label · "
                          f"median clIoU_adapt {np.median(ns):.4f} · range {min(ns):.3f}–{max(ns):.3f}",
                 ha="center", size=10)
        body = ("Left panel: detector overlay, orange = prediction.   Right panel: the binary mask.\n\n"
                "Pages run best-scoring first; unlabelled frames last. clIoU_adapt is a tolerant\n"
                "centreline IoU with the tolerance set per frame to half that frame's label brush.\n"
                "It is NOT pixel IoU: a perfect 3 px trace scores pixel IoU 0.1662 on this corpus,\n"
                "so pixel IoU cannot rank methods here.\n\n"
                "Every score is against hand labels that are region assertions, not pixel-precise\n"
                "ground truth. They rank; they do not measure correctness.")
        fig.text(.5, .66, body, ha="center", va="top", size=9.5, linespacing=1.9, color="#333")
        pdf.savefig(fig); plt.close(fig)

        for i, r in enumerate(allr):
            n = r["frame"]
            ov = f"{SC}/detector_overlays/{n}.png"
            mk = f"{SC}/detector_masks/{n}.png"
            if not (os.path.exists(ov) and os.path.exists(mk)):
                continue
            fig, ax = plt.subplots(1, 2, figsize=(11.7, 5.2))
            ax[0].imshow(thumb(ov)); ax[0].set_title("detector overlay", size=9)
            ax[1].imshow(thumb(mk), cmap="gray"); ax[1].set_title("binary mask", size=9)
            for a in ax:
                a.axis("off")
            cap = (f"{n}    clIoU_adapt {r['clIoU_adapt']:.4f}" if "clIoU_adapt" in r
                   else f"{n}    (no hand label)")
            fig.suptitle(f"{cap}    ·    {100*r['pred_frac']:.2f}% of frame predicted", size=11)
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            pdf.savefig(fig, dpi=110); plt.close(fig)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(allr)}", flush=True)
        d = pdf.infodict()
        d["Title"] = "SEM crack detector — all frames"
    print(f"\nwrote {out}  ({os.path.getsize(out)/1e6:.1f} MB, {len(allr)+1} pages)")


if __name__ == "__main__":
    main()
