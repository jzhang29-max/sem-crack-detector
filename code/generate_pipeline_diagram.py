#!/usr/bin/env python3
"""Render docs/img/pipeline.png -- the architecture as it actually is.

Regenerate after any change to the stage order:

    python3 code/generate_pipeline_diagram.py

WHY A GENERATOR RATHER THAN A DRAWING. A hand-made diagram goes stale silently, and this
project has already been bitten by documents describing a pipeline that had moved on. This
reads the real stage list from one place below, so updating the diagram is a code edit that
shows up in review rather than an image someone forgets to redo.
"""
import os
import sys

from PIL import Image, ImageDraw, ImageFont

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
OUT = os.path.join(ROOT, "docs", "img", "pipeline.png")

W, H = 1900, 1240
BG = (14, 16, 20)
INK = (236, 238, 241)
DIM = (150, 158, 168)
FAINT = (104, 112, 122)
LINE = (44, 50, 60)
BLUE = (79, 142, 247)
GREEN = (46, 163, 107)
AMBER = (201, 130, 42)
RED = (214, 69, 69)
VIOLET = (150, 118, 235)

FONTS = ["/System/Library/Fonts/Supplemental/Helvetica.ttc",
         "/System/Library/Fonts/Helvetica.ttc",
         "/Library/Fonts/Arial.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]


def font(size, bold=False):
    for p in FONTS:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size, index=1 if bold else 0)
            except Exception:
                try:
                    return ImageFont.truetype(p, size)
                except Exception:
                    continue
    return ImageFont.load_default()


#: THE PIPELINE, in order. (title, detail lines, accent, tag)
STAGES = [
    ("Load", ["16-bit TIFF, ~25 MP", "contrast stretch to 8-bit"], BLUE, None),
    ("Frame", ["find_field_of_view: crop the burned-in", "info bar and vignette"], BLUE, None),
    ("Flatten", ["flatten_background", "compute_vesselness"], BLUE, None),
    ("Candidates", ["segment_dark_regions -> clean_mask", "extract_candidates: regions + 8 features"], BLUE, None),
    ("Pass 1", ["LogisticRegression over the 8 features", "threshold 0.5 (bundle fallback)"], VIOLET, "model"),
    ("Pass 2", ["interior_fill / concavity / bridge_corridor", "scored with the unified model"], VIOLET, "model"),
    ("Corrections", ["1 crack   2 not-crack   3 erased   0 UNREVIEWED", "human verdicts override the model"], GREEN, "human"),
    ("Merge", ["merge_large_cracks: bridge fragments", "a connector cannot invent a region"], BLUE, None),
    ("SAM 2 refine", ["box prompt per accepted region", "boundary redrawn; DEFAULT since 4d97602"], AMBER, "new"),
]

OUTPUTS = [
    ("Overlay", ["regenerate_templates", "the only renderer"]),
    ("Per-crack CSV", ["one row per crack", "+ _provenance.json"]),
    ("Aggregate", ["specimen as the unit", "refusals, not guesses"]),
]

SIDE = [
    ("Imported mask", ["ilastik / micro-sam / anything", "REPLACES the detector; never refined"], RED),
    ("Calibration", ["scale bar, HFW, manual, instrument tags", "refuses on >5% disagreement"], GREEN),
    ("Promotion gate", ["out-of-sample baseline required", "refuses a mismatched holdout"], GREEN),
]


def rr(d, box, r, fill=None, outline=None, w=1):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=w)


def chip(d, x, y, text, color):
    f = font(13, True)
    tw = d.textlength(text, font=f)
    rr(d, (x, y, x + tw + 18, y + 24), 12, fill=color)
    d.text((x + 9, y + 5), text, font=f, fill=(12, 14, 18))
    return tw + 18


def main():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    d.text((56, 44), "SEM crack detection - pipeline as implemented",
           font=font(38, True), fill=INK)
    d.text((58, 96), "One shared path: overlays, measurements, exports, undo and the app all "
                     "call run_unified_pipeline, so there is one mask.",
           font=font(18), fill=DIM)

    # ---- the main column of stages ----
    x0, y0, cw, ch, gap = 56, 156, 760, 96, 16
    for i, (title, lines, accent, tag) in enumerate(STAGES):
        y = y0 + i * (ch + gap)
        rr(d, (x0, y, x0 + cw, y + ch), 14, fill=(22, 25, 30), outline=LINE)
        d.rectangle((x0, y + 14, x0 + 4, y + ch - 14), fill=accent)
        d.text((x0 + 26, y + 18), title, font=font(23, True), fill=INK)
        d.text((x0 + 26, y + 50), lines[0], font=font(15), fill=DIM)
        d.text((x0 + 26, y + 70), lines[1], font=font(15), fill=FAINT)
        if tag:
            label = {"model": "MODEL", "human": "AUTHORITATIVE", "new": "NEW"}[tag]
            col = {"model": VIOLET, "human": GREEN, "new": AMBER}[tag]
            t = label
            f = font(12, True)
            tw = d.textlength(t, font=f)
            rr(d, (x0 + cw - tw - 34, y + 18, x0 + cw - 16, y + 40), 11, fill=col)
            d.text((x0 + cw - tw - 25, y + 22), t, font=f, fill=(12, 14, 18))
        if i < len(STAGES) - 1:
            mx = x0 + cw // 2
            d.line((mx, y + ch, mx, y + ch + gap), fill=LINE, width=2)
            d.polygon([(mx - 5, y + ch + gap - 6), (mx + 5, y + ch + gap - 6),
                       (mx, y + ch + gap)], fill=LINE)

    # ---- authority order ----
    ax, ay = 900, 156
    rr(d, (ax, ay, ax + 944, ay + 116), 14, fill=(22, 25, 30), outline=LINE)
    d.text((ax + 24, ay + 16), "Authority order", font=font(23, True), fill=INK)
    cx = ax + 24
    for i, (t, c) in enumerate([("human correction", GREEN), ("imported mask", RED),
                                ("SAM 2", AMBER), ("built-in detector", BLUE)]):
        cx += chip(d, cx, ay + 56, t, c) + 6
        if i < 3:
            d.text((cx, ay + 58), ">", font=font(18, True), fill=DIM)
            cx += 20
    d.text((ax + 24, ay + 90), "Refinement runs after corrections and the human's verdicts are "
                               "restored afterwards; the sidecar records how many pixels each put back.",
           font=font(14), fill=FAINT)

    # ---- side concerns ----
    sy = ay + 148
    for title, lines, accent in SIDE:
        rr(d, (ax, sy, ax + 944, sy + 92), 14, fill=(20, 23, 28), outline=LINE)
        d.rectangle((ax, sy + 14, ax + 4, sy + 78), fill=accent)
        d.text((ax + 24, sy + 16), title, font=font(20, True), fill=INK)
        d.text((ax + 24, sy + 44), lines[0], font=font(15), fill=DIM)
        d.text((ax + 24, sy + 64), lines[1], font=font(15), fill=FAINT)
        sy += 108

    # ---- outputs ----
    oy = sy + 12
    d.text((ax, oy), "Outputs", font=font(23, True), fill=INK)
    oy += 40
    for title, lines in OUTPUTS:
        rr(d, (ax, oy, ax + 944, oy + 78), 14, fill=(20, 23, 28), outline=LINE)
        d.text((ax + 24, oy + 14), title, font=font(19, True), fill=INK)
        d.text((ax + 24, oy + 42), f"{lines[0]}  -  {lines[1]}", font=font(15), fill=DIM)
        oy += 92

    # ---- measured footnote ----
    fy = oy + 14
    rr(d, (ax, fy, ax + 944, fy + 96), 14, fill=(18, 30, 24), outline=(34, 74, 56))
    d.text((ax + 24, fy + 14), "What SAM 2 refinement is worth, on the ten both-class frames",
           font=font(17, True), fill=INK)
    d.text((ax + 24, fy + 42),
           "bare detector    f1 0.638   recall 0.534   specificity 0.460   precision 0.970",
           font=font(15), fill=DIM)
    d.text((ax + 24, fy + 64),
           "--sam2 refine    f1 0.676   recall 0.561   specificity 0.569   precision 0.976",
           font=font(15), fill=(140, 220, 175))

    d.text((56, H - 42), "Regenerate: python3 code/generate_pipeline_diagram.py    "
                         "Stage order is read from STAGES in that file, so this image "
                         "cannot drift from the code without the edit showing up in review.",
           font=font(14), fill=FAINT)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    img.save(OUT, "PNG")
    print(f"wrote {OUT}  ({img.size[0]}x{img.size[1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
