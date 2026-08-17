"""
A hand-authored SVG scientific graphical-abstract figure for the crack-
detection pipeline: real gradients, real blurred drop shadows, a two-row
serpentine layout (the layout most Cell/Nature graphical abstracts use),
line-style icons, and an actual journal-style figure caption underneath --
built directly in SVG rather than matplotlib so coordinates are natively
isotropic (no aspect-ratio correction hacks) and shadows/gradients are real
rather than faked with stacked patches.

Requires the `rsvg-convert` CLI (already present on this machine) to
rasterize the final PNG; the SVG itself is also kept as a deliverable since
it's fully vector (infinitely scalable, editable in Illustrator/Inkscape).

Usage
-----
    python3 generate_scientific_diagram.py [image_name_without_extension]
"""
import base64
import io
import os
import subprocess
import sys
import textwrap

import numpy as np
from PIL import Image
import matplotlib
try:
    from matplotlib import colormaps as _colormaps
    def get_cmap(name):
        return _colormaps[name]
except ImportError:
    import matplotlib.cm as cm
    def get_cmap(name):
        return cm.get_cmap(name)

from pipeline_stages import compute_pipeline_stages, ROOT, RESULTS_DIR

OUT_DIR = os.path.join(ROOT, "pipeline_diagram")
os.makedirs(OUT_DIR, exist_ok=True)
IMAGE_NAME = sys.argv[1] if len(sys.argv) > 1 else "260708_316_H_b2_front_CBS_002"

# Cool -> warm progression, a refined "sunset" palette (raw data to finished result)
STAGE_COLORS = ["#264653", "#2A9D8F", "#5E9C6C", "#E9B44C", "#F0954A", "#E76F51"]
DOC_COLOR = "#A9812F"
INK = "#1B242C"
SUBTEXT = "#5B6B76"


def to_data_uri(arr, cmap=None):
    """numpy array -> base64 PNG data URI, ready to drop straight into an <image> tag."""
    if arr.dtype != np.uint8:
        a = arr.astype(np.float64)
        a = (a - a.min()) / (np.ptp(a) + 1e-9)
        if cmap:
            arr = (get_cmap(cmap)(a)[..., :3] * 255).astype(np.uint8)
        else:
            arr = (a * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}", img.size


# ------------------------------------------------------------------ icons
def icon_svg(kind, color, size=1.0):
    """Icons are authored in a -1..1 box and scaled/translated by the caller
    via a <g transform>, so SVG's native isotropic coordinates keep every
    shape a true circle/square -- no aspect-ratio correction needed."""
    sw = 0.16 * size
    if kind == "sliders":
        parts = []
        for x, ky in zip([-0.55, 0, 0.55], [0.25, -0.35, 0.4]):
            parts.append(f'<line x1="{x}" y1="-0.7" x2="{x}" y2="0.7" stroke="{color}" '
                         f'stroke-width="{sw}" stroke-linecap="round"/>')
            parts.append(f'<circle cx="{x}" cy="{ky}" r="0.16" fill="white" stroke="{color}" stroke-width="{sw}"/>')
        return "".join(parts)
    if kind == "sun":
        parts = [f'<circle cx="0" cy="0" r="0.42" fill="none" stroke="{color}" stroke-width="{sw}"/>']
        for ang in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            x0, y0 = np.cos(ang) * 0.6, np.sin(ang) * 0.6
            x1, y1 = np.cos(ang) * 0.95, np.sin(ang) * 0.95
            parts.append(f'<line x1="{x0:.3f}" y1="{y0:.3f}" x2="{x1:.3f}" y2="{y1:.3f}" '
                         f'stroke="{color}" stroke-width="{sw}" stroke-linecap="round"/>')
        return "".join(parts)
    if kind == "magnifier":
        return (f'<circle cx="-0.15" cy="-0.15" r="0.45" fill="none" stroke="{color}" stroke-width="{sw}"/>'
                f'<line x1="0.18" y1="0.18" x2="0.6" y2="0.6" stroke="{color}" '
                f'stroke-width="{sw * 1.3}" stroke-linecap="round"/>')
    if kind == "network":
        pts = [(-0.6, 0.55), (0.6, 0.6), (-0.4, -0.5), (0.55, -0.45), (0, 0.05)]
        parts = []
        for i, (x0, y0) in enumerate(pts):
            for (x1, y1) in pts[i + 1:]:
                parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="{color}" '
                             f'stroke-width="{sw * 0.4}" opacity="0.55"/>')
        for (x0, y0) in pts:
            parts.append(f'<circle cx="{x0}" cy="{y0}" r="0.13" fill="{color}"/>')
        return "".join(parts)
    if kind == "classifier":
        left = [(-0.55, 0.6), (-0.55, 0), (-0.55, -0.6)]
        right = [(0.55, 0.35), (0.55, -0.35)]
        parts = []
        for (x0, y0) in left:
            for (x1, y1) in right:
                parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="{color}" '
                             f'stroke-width="{sw * 0.4}" opacity="0.6"/>')
        for (x0, y0) in left:
            parts.append(f'<circle cx="{x0}" cy="{y0}" r="0.15" fill="white" stroke="{color}" stroke-width="{sw}"/>')
        for (x0, y0) in right:
            parts.append(f'<circle cx="{x0}" cy="{y0}" r="0.17" fill="{color}"/>')
        return "".join(parts)
    if kind == "grid":
        parts = []
        for gx in (-0.62, 0, 0.62):
            for gy in (-0.62, 0, 0.62):
                parts.append(f'<rect x="{gx-0.24}" y="{gy-0.24}" width="0.48" height="0.48" '
                             f'fill="none" stroke="{color}" stroke-width="{sw}"/>')
        return "".join(parts)
    if kind == "check":
        return (f'<circle cx="0" cy="0" r="0.85" fill="none" stroke="{color}" stroke-width="{sw}"/>'
                f'<polyline points="-0.38,0 -0.05,0.35 0.5,-0.35" fill="none" stroke="{color}" '
                f'stroke-width="{sw * 1.3}" stroke-linecap="round" stroke-linejoin="round"/>')
    return ""


def doc_glyph_svg(kind, color):
    if kind == "image":
        return (f'<rect x="-0.5" y="-0.42" width="1" height="0.84" fill="none" stroke="{color}" stroke-width="0.06"/>'
                f'<circle cx="-0.22" cy="-0.14" r="0.09" fill="{color}"/>'
                f'<polygon points="-0.5,0.05 -0.05,-0.28 0.15,-0.05 0.5,-0.3 0.5,0.42 -0.5,0.42" fill="{color}" opacity="0.35"/>')
    if kind == "table":
        parts = []
        for x in (-0.45, 0, 0.45):
            parts.append(f'<line x1="{x}" y1="-0.45" x2="{x}" y2="0.45" stroke="{color}" stroke-width="0.055"/>')
        for y in (-0.45, 0, 0.45):
            parts.append(f'<line x1="-0.45" y1="{y}" x2="0.45" y2="{y}" stroke="{color}" stroke-width="0.055"/>')
        return "".join(parts)
    if kind == "model":
        parts = [f'<rect x="-0.32" y="-0.32" width="0.64" height="0.64" fill="none" stroke="{color}" stroke-width="0.07"/>']
        for dx, dy in [(-1, 0.5), (-1, -0.5), (1, 0.5), (1, -0.5), (0.5, 1), (-0.5, 1), (0.5, -1), (-0.5, -1)]:
            x0, y0 = dx * 0.32, dy * 0.32
            x1 = x0 + (0.18 if abs(dx) > abs(dy) else 0) * np.sign(dx)
            y1 = y0 + (0.18 if abs(dy) >= abs(dx) else 0) * np.sign(dy)
            parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="{color}" stroke-width="0.06"/>')
        return "".join(parts)
    return ""


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def wrap_tspans(text, width, x, line_h, first_dy=0):
    lines = textwrap.wrap(text, width=width)
    out = []
    for i, ln in enumerate(lines):
        dy = first_dy if i == 0 else line_h
        out.append(f'<tspan x="{x}" dy="{dy}">{esc(ln)}</tspan>')
    return "".join(out), len(lines)


class SVG:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.defs = []
        self.body = []

    def add_gradient(self, gid, c0, c1, angle="vertical"):
        x2, y2 = ("0%", "100%") if angle == "vertical" else ("100%", "0%")
        self.defs.append(
            f'<linearGradient id="{gid}" x1="0%" y1="0%" x2="{x2}" y2="{y2}">'
            f'<stop offset="0%" stop-color="{c0}"/><stop offset="100%" stop-color="{c1}"/>'
            f'</linearGradient>')

    def add_shadow_filter(self, fid, dx=0, dy=6, blur=10, opacity=0.22):
        self.defs.append(f'''
        <filter id="{fid}" x="-50%" y="-50%" width="200%" height="200%">
          <feDropShadow dx="{dx}" dy="{dy}" stdDeviation="{blur}" flood-color="#1B242C" flood-opacity="{opacity}"/>
        </filter>''')

    def add_arrowhead(self, mid, color):
        self.defs.append(f'''
        <marker id="{mid}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="{color}"/>
        </marker>''')

    def raw(self, s):
        self.body.append(s)

    def render(self):
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                f'viewBox="0 0 {self.w} {self.h}">'
                f'<defs>{"".join(self.defs)}</defs>{"".join(self.body)}</svg>')


def rounded_rect(x, y, w, h, r, fill, stroke=None, sw=0, filter_id=None, opacity=1.0):
    f = f'filter="url(#{filter_id})"' if filter_id else ""
    s = f'stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" ry="{r}" fill="{fill}" {s} {f} opacity="{opacity}"/>'


def main():
    print(f"Building scientific SVG diagram for {IMAGE_NAME} ...")
    s = compute_pipeline_stages(IMAGE_NAME, half_w=520, half_h=430)

    stages = [
        dict(key="A", title="Preprocessing", icon="sliders",
             subtitle="16-bit → 8-bit contrast stretch; auto-crop info bar / vignette",
             thumbs=[("Raw (16-bit)", s["raw_display"], None), ("Stretched + cropped", s["img8_crop"], None)]),
        dict(key="B", title="Illumination Correction", icon="sun",
             subtitle="Large-kernel background estimate subtracted (high-pass filter)",
             thumbs=[("Before", s["img8_crop"], None), ("Flattened", s["flat_crop"], None)]),
        dict(key="C", title="Dark-Region Segmentation", icon="magnifier",
             subtitle="Relative + absolute darkness tests, OR'd; morphological cleanup",
             thumbs=[("Raw dark mask", s["dark_mask_crop"], None), ("Cleaned mask", s["clean_crop"], None)]),
        dict(key="D", title="Feature Extraction", icon="network",
             subtitle="Frangi vesselness + connected components → 8-D feature vectors",
             thumbs=[("Vesselness map", s["vesselness_crop"], "inferno"),
                     (f"{s['n_candidates']} candidates", s["label_rgb_crop"], None)]),
        dict(key="E", title="ML Classification & Merging", icon="classifier",
             subtitle="Logistic regression + MST-based fragment bridging",
             thumbs=[("Classified", s["overlay_pre_merge_crop"], None),
                     ("Merged", s["overlay_post_merge_crop"], None)]),
        dict(key="F", title="Output Generation", icon="check",
             subtitle="Final mask + numbered overlay + measurements",
             thumbs=[("B&W mask", s["final_bw_crop"], None), ("Annotated overlay", s["final_overlay_crop"], None)]),
    ]

    # ---------------------------------------------------------- geometry
    W = 1760
    MARGIN = 70
    CARD_W = (W - 2 * MARGIN - 2 * 60) / 3
    CARD_H = 470
    ROW_GAP = 190
    TITLE_H = 260
    EXTRA_BAND_H = 480   # output-document cluster + the review-sheet showcase card
    CAPTION_H = 230
    H = TITLE_H + CARD_H + ROW_GAP + CARD_H + EXTRA_BAND_H + CAPTION_H + 60

    svg = SVG(W, H)
    svg.add_shadow_filter("cardShadow", dy=10, blur=16, opacity=0.16)
    svg.add_shadow_filter("thumbShadow", dy=4, blur=7, opacity=0.22)
    svg.add_shadow_filter("badgeShadow", dy=3, blur=5, opacity=0.30)
    for i, color in enumerate(STAGE_COLORS):
        svg.add_gradient(f"grad{i}", color, _darken(color, 0.72))
        svg.add_arrowhead(f"arrow{i}", color)
    svg.add_gradient("bg", "#F7FAFC", "#EAF1F3")
    svg.add_gradient("gradReview", DOC_COLOR, _darken(DOC_COLOR, 0.72))
    svg.add_arrowhead("arrowDoc", DOC_COLOR)

    # background + faint blueprint dot grid
    svg.raw(f'<rect x="0" y="0" width="{W}" height="{H}" fill="url(#bg)"/>')
    dots = []
    step = 26
    for gx in range(0, W, step):
        for gy in range(0, H - CAPTION_H, step):
            dots.append(f'<circle cx="{gx}" cy="{gy}" r="1" fill="#1B242C" opacity="0.035"/>')
    svg.raw("".join(dots))

    svg.raw(f'<text x="{W/2}" y="58" text-anchor="middle" font-family="Georgia, \'Times New Roman\', serif" '
            f'font-size="34" font-weight="700" fill="{INK}">Automated Crack Detection in SEM/TXM Micrographs</text>')
    svg.raw(f'<text x="{W/2}" y="90" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" '
            f'font-size="16" font-style="italic" fill="{SUBTEXT}">Pipeline overview, worked example: {esc(IMAGE_NAME)}</text>')

    row_y = [TITLE_H, TITLE_H + CARD_H + ROW_GAP]
    col_x = [MARGIN, MARGIN + CARD_W + 60, MARGIN + 2 * (CARD_W + 60)]
    centers = {}

    # Classic serpentine (boustrophedon) reading order: row 0 goes A->B->C
    # left-to-right, row 1 goes D->E->F RIGHT-TO-LEFT (D lands directly under
    # C). That keeps the row-wrap connector a short same-column drop instead
    # of a diagonal sweeping across the whole figure and colliding with the
    # document icons above row 1.
    positions = [(col_x[0], row_y[0]), (col_x[1], row_y[0]), (col_x[2], row_y[0]),
                 (col_x[2], row_y[1]), (col_x[1], row_y[1]), (col_x[0], row_y[1])]

    for i, stage in enumerate(stages):
        x, y = positions[i]
        color = STAGE_COLORS[i]
        centers[i] = (x, y, x + CARD_W, y + CARD_H)
        draw_card(svg, x, y, CARD_W, CARD_H, stage, color, i)

    def connect(i, j):
        """Arrow from stage i to stage j, whichever side actually faces the other."""
        xi0, yi0, xi1, yi1 = centers[i]
        xj0, yj0, xj1, yj1 = centers[j]
        ay = yi0 + 60
        if xj0 >= xi1:
            x_from, x_to = xi1 - 6, xj0 + 6
        else:
            x_from, x_to = xi0 + 6, xj1 - 6
        svg.raw(f'<path d="M {x_from} {ay} L {x_to} {ay}" stroke="{STAGE_COLORS[j]}" stroke-width="4" '
                f'fill="none" marker-end="url(#arrow{j})"/>')

    connect(0, 1); connect(1, 2)   # row 0, left -> right
    connect(3, 4); connect(4, 5)   # row 1, right -> left

    # serpentine wrap: bottom of C drops straight into top of D (same column)
    x2_0, y2_0, x2_1, y2_1 = centers[2]
    x3_0, y3_0, x3_1, y3_1 = centers[3]
    wrap_color = STAGE_COLORS[3]
    cxw = (x2_0 + x2_1) / 2
    svg.raw(f'<path d="M {cxw} {y2_1} C {cxw+40} {y2_1+ROW_GAP*0.4}, {cxw-40} {y3_0-ROW_GAP*0.4}, {cxw} {y3_0}" '
            f'stroke="{wrap_color}" stroke-width="4" fill="none" marker-end="url(#arrow3)"/>')

    # ---- input/output documents -------------------------------------------
    docA_size = 78
    docA_x, docA_y = centers[0][0] + CARD_W * 0.5, doc_cy_above(row_y[0], docA_size)
    draw_document(svg, docA_x, docA_y, docA_size, "Raw 16-bit TIFF\nmicrograph", "image")
    svg.raw(f'<path d="M {docA_x} {doc_arrow_start_y(docA_y, docA_size)} L {docA_x} {row_y[0]-4}" '
            f'stroke="{DOC_COLOR}" stroke-width="3" fill="none" marker-end="url(#arrowDoc)"/>')

    # All three of these feed into E (the classifier + merge step): D's
    # feature vectors, the trained model, and the corrections ledger. Space
    # them evenly WITHIN E's own column rather than straddling the narrow
    # gaps to its neighbors, so nothing overlaps the adjacent stage cards.
    ex, ey0, ex1, ey1 = centers[4]
    mcx = (ex + ex1) / 2
    doc_size = 66
    doc_y = doc_cy_above(row_y[1], doc_size)
    doc_arrow_y = doc_arrow_start_y(doc_y, doc_size)
    doc_spacing = (CARD_W - 90) / 2  # keeps all 3 docs comfortably inside CARD_W
    e_docs = [
        (mcx - doc_spacing, f"Feature vectors\n(8 x {s['n_candidates']})", "table"),
        (mcx, "Trained LogReg\nclassifier (.joblib)", "model"),
        (mcx + doc_spacing, "Manual corrections\nledger (.csv)", "table"),
    ]
    for x, label, glyph in e_docs:
        draw_document(svg, x, doc_y, doc_size, label, glyph)
        svg.raw(f'<path d="M {x} {doc_arrow_y} L {x} {row_y[1]-4}" stroke="{DOC_COLOR}" '
                f'stroke-width="3" fill="none" marker-end="url(#arrowDoc)"/>')

    fx0, fy0, fx1, fy1 = centers[5]
    fcx2 = (fx0 + fx1) / 2
    out_y = fy1 + 90
    conv_y = fy1 + 42
    svg.raw(f'<path d="M {fcx2} {fy1} L {fcx2} {conv_y}" stroke="{DOC_COLOR}" stroke-width="3" fill="none"/>')
    for ddx, lbl, glyph in [(-150, "Black & white\nmask (.tif)", "image"),
                             (0, "Measurements\nCSV", "table"),
                             (150, "Annotated\noverlay (.png)", "image")]:
        svg.raw(f'<path d="M {fcx2} {conv_y} L {fcx2+ddx} {out_y-38}" stroke="{DOC_COLOR}" stroke-width="3" '
                f'fill="none" marker-end="url(#arrowDoc)"/>')
        draw_document(svg, fcx2 + ddx, out_y, 70, lbl, glyph)

    # ---- quality-control review-sheet showcase (real output, cropped) ------
    review_paths = sorted(
        p for p in os.listdir(RESULTS_DIR) if p.startswith(f"{IMAGE_NAME}_review_page") and p.endswith(".png")
    )
    if review_paths:
        review_x0, review_x1 = centers[4][0], centers[3][2]
        review_y0 = row_y[1] + CARD_H + 30
        review_h = 430
        header_h = 68
        pad = 20

        svg.raw(rounded_rect(review_x0, review_y0, review_x1 - review_x0, review_h, 16,
                              "white", stroke="#E1E8EB", sw=1.5, filter_id="cardShadow"))
        svg.raw(f'<path d="M {review_x0} {review_y0+16} Q {review_x0} {review_y0} {review_x0+16} {review_y0} '
                f'L {review_x1-16} {review_y0} Q {review_x1} {review_y0} {review_x1} {review_y0+16} '
                f'L {review_x1} {review_y0+header_h} L {review_x0} {review_y0+header_h} Z" fill="url(#gradReview)"/>')
        badge_r = 26
        bcx, bcy = review_x0 + 46, review_y0 + header_h / 2
        svg.raw(f'<circle cx="{bcx}" cy="{bcy}" r="{badge_r}" fill="white" filter="url(#badgeShadow)"/>')
        svg.raw(f'<g transform="translate({bcx},{bcy}) scale({badge_r*0.62})">{icon_svg("grid", DOC_COLOR)}</g>')
        svg.raw(f'<text x="{review_x0+82}" y="{bcy+7}" font-family="Helvetica, Arial, sans-serif" '
                f'font-size="19" font-weight="700" fill="white">Quality-Control Review Sheet</text>')

        content_w = review_x1 - review_x0 - 2 * pad
        content_h = review_h - header_h - 2 * pad - 22
        rp_img = Image.open(os.path.join(RESULTS_DIR, review_paths[0])).convert("RGB")
        rp_w, rp_h = rp_img.size
        crop_h = min(rp_h, int(rp_w * (content_h / content_w)))
        rp_crop = rp_img.crop((0, 0, rp_w, crop_h))
        uri, _ = to_data_uri(np.array(rp_crop))
        ix, iy = review_x0 + pad, review_y0 + header_h + pad
        svg.raw(f'<g filter="url(#thumbShadow)"><rect x="{ix-3}" y="{iy-3}" width="{content_w+6}" '
                f'height="{content_h+6}" fill="white"/></g>')
        svg.raw(f'<image x="{ix}" y="{iy}" width="{content_w}" height="{content_h}" href="{uri}" '
                f'preserveAspectRatio="xMidYMin slice"/>')
        svg.raw(f'<rect x="{ix}" y="{iy}" width="{content_w}" height="{content_h}" fill="none" '
                f'stroke="{DOC_COLOR}" stroke-width="1.5"/>')
        n_kept_total = s["n_kept_final"]
        cap_text = (f"Every candidate across all {len(review_paths)} page(s) ({s['n_candidates']} total, "
                    f"{n_kept_total} kept), bordered red = kept / cyan = rejected -- used to spot-check "
                    f"and hand-correct labels for the next training round.")
        cap_tspans, _ = wrap_tspans(cap_text, 92, review_x0 + content_w / 2 + pad, 15)
        svg.raw(f'<text x="{review_x0+content_w/2+pad}" y="{iy+content_h+18}" text-anchor="middle" '
                f'font-family="Helvetica, Arial, sans-serif" font-size="11.5" fill="{SUBTEXT}">{cap_tspans}</text>')

    # ---------------------------------------------------------- caption
    cap_y = H - CAPTION_H + 30
    svg.raw(rounded_rect(MARGIN, cap_y - 24, W - 2 * MARGIN, CAPTION_H - 40, 10, "white",
                          stroke="#D8E0E4", sw=1.5, filter_id="cardShadow"))
    caption = (
        f"Figure 1. Automated crack-detection pipeline for SEM/TXM micrographs of annealed stainless steel. "
        f"Raw 16-bit micrographs are contrast-stretched and auto-cropped to remove instrument overlays (A), "
        f"background-flattened to remove uneven illumination (B), and segmented into candidate dark regions via "
        f"combined relative/absolute darkness thresholds (C). A Frangi vesselness filter and connected-component "
        f"analysis extract an 8-feature vector per candidate (D), which a logistic-regression classifier trained on "
        f"manually-corrected labels uses to separate cracks from artifacts (pores, inclusions); a minimum-spanning-tree "
        f"then bridges fragments of the same physical crack using locally-measured connector widths (E). The final "
        f"output is a binary crack mask, a numbered annotated overlay, and per-candidate measurements (F). "
        f"Worked example shown: {IMAGE_NAME}."
    )
    tspans, nlines = wrap_tspans(caption, 150, MARGIN + 24, 22)
    svg.raw(f'<text x="{MARGIN+24}" y="{cap_y+6}" font-family="Georgia, serif" font-size="14.5" fill="{INK}">{tspans}</text>')

    svg_path = os.path.join(OUT_DIR, f"scientific_diagram_{IMAGE_NAME}.svg")
    with open(svg_path, "w") as f:
        f.write(svg.render())
    print(f"Saved SVG: {svg_path}")

    png_path = os.path.join(OUT_DIR, f"scientific_diagram_{IMAGE_NAME}.png")
    subprocess.run(["rsvg-convert", "-w", str(W * 2), "-h", str(H * 2), svg_path, "-o", png_path], check=True)
    print(f"Saved PNG: {png_path}")


def _darken(hexcolor, factor):
    h = hexcolor.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"#{int(r*factor):02x}{int(g*factor):02x}{int(b*factor):02x}"


def draw_card(svg, x, y, w, h, stage, color, idx):
    svg.raw(rounded_rect(x, y, w, h, 16, "white", stroke="#E1E8EB", sw=1.5, filter_id="cardShadow"))
    header_h = 108
    svg.raw(f'<path d="M {x} {y+16} Q {x} {y} {x+16} {y} L {x+w-16} {y} Q {x+w} {y} {x+w} {y+16} '
            f'L {x+w} {y+header_h} L {x} {y+header_h} Z" fill="url(#grad{idx})"/>')

    badge_r = 34
    bcx, bcy = x + 58, y + header_h / 2
    svg.raw(f'<circle cx="{bcx}" cy="{bcy}" r="{badge_r}" fill="white" filter="url(#badgeShadow)"/>')
    svg.raw(f'<g transform="translate({bcx},{bcy}) scale({badge_r*0.62})">{icon_svg(stage["icon"], color)}</g>')

    svg.raw(f'<text x="{x+104}" y="{bcy-6}" font-family="Helvetica, Arial, sans-serif" font-size="20.5" '
            f'font-weight="700" fill="white">({stage["key"]}) {esc(stage["title"])}</text>')
    sub_tspans, _ = wrap_tspans(stage["subtitle"], 40, x + 104, 16)
    svg.raw(f'<text x="{x+104}" y="{bcy+14}" font-family="Helvetica, Arial, sans-serif" font-size="12.3" '
            f'fill="#EAF3F1">{sub_tspans}</text>')

    thumb_y = y + header_h + 18
    thumb_h = h - header_h - 18 - 34
    gap = 12
    thumb_w = (w - 2 * 18 - gap) / 2
    for j, (cap, arr, cmap) in enumerate(stage["thumbs"]):
        tx = x + 18 + j * (thumb_w + gap)
        uri, (pw, ph) = to_data_uri(arr, cmap=cmap)
        ar = ph / pw
        draw_w, draw_h = thumb_w, thumb_w * ar
        if draw_h > thumb_h:
            draw_h = thumb_h
            draw_w = thumb_h / ar
        ox = tx + (thumb_w - draw_w) / 2
        oy = thumb_y + (thumb_h - draw_h) / 2
        svg.raw(f'<g filter="url(#thumbShadow)"><rect x="{ox-3}" y="{oy-3}" width="{draw_w+6}" height="{draw_h+6}" '
                f'fill="white"/></g>')
        svg.raw(f'<image x="{ox}" y="{oy}" width="{draw_w}" height="{draw_h}" href="{uri}" '
                f'style="image-rendering:auto"/>')
        svg.raw(f'<rect x="{ox}" y="{oy}" width="{draw_w}" height="{draw_h}" fill="none" stroke="{color}" stroke-width="1.5"/>')
        svg.raw(f'<text x="{tx+thumb_w/2}" y="{thumb_y+thumb_h+20}" text-anchor="middle" '
                f'font-family="Helvetica, Arial, sans-serif" font-size="12" fill="{SUBTEXT}">{esc(cap)}</text>')


# A document's 2-line label sits below it (see draw_document); an outgoing
# arrow must clear that text, not just the doc's own bottom edge. These two
# helpers keep every doc-to-box placement/arrow consistent with the actual
# label height instead of hand-tuned numbers that drift out of sync.
DOC_LABEL_GAP = 16      # gap between doc bottom edge and first line of label
DOC_LINE_H = 14         # per-line spacing used in draw_document's tspans
DOC_LABEL_PAD = 10      # breathing room below the last line of label text
DOC_ARROW_MIN = 22      # minimum visible arrow length feeding into a box


def doc_label_block_h(n_lines=2):
    return DOC_LABEL_GAP + n_lines * DOC_LINE_H + DOC_LABEL_PAD


def doc_cy_above(row_top, doc_size, n_lines=2):
    """Center-y for a document so its label clears row_top with room for an arrow."""
    return row_top - DOC_ARROW_MIN - doc_label_block_h(n_lines) - doc_size / 2


def doc_arrow_start_y(doc_cy, doc_size, n_lines=2):
    """Where an outgoing arrow below this doc may safely start (below its label)."""
    return doc_cy + doc_size / 2 + doc_label_block_h(n_lines)


def draw_document(svg, cx, cy, size, label, glyph):
    w = h = size
    x0, y0 = cx - w / 2, cy - h / 2
    fold = 0.16 * w
    svg.raw(f'<g filter="url(#thumbShadow)">'
            f'<polygon points="{x0},{y0} {x0+w-fold},{y0} {x0+w},{y0+fold} {x0+w},{y0+h} {x0},{y0+h}" '
            f'fill="#FFF8E4" stroke="{DOC_COLOR}" stroke-width="1.6"/></g>')
    svg.raw(f'<polygon points="{x0+w-fold},{y0} {x0+w-fold},{y0+fold} {x0+w},{y0+fold}" '
            f'fill="#F0E0AE" stroke="{DOC_COLOR}" stroke-width="1"/>')
    svg.raw(f'<g transform="translate({cx},{cy-h*0.08}) scale({w*0.38})">{doc_glyph_svg(glyph, DOC_COLOR)}</g>')
    lines = label.split("\n")
    tspans = "".join(f'<tspan x="{cx}" dy="{0 if i==0 else 14}">{esc(l)}</tspan>' for i, l in enumerate(lines))
    svg.raw(f'<text x="{cx}" y="{y0+h+16}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" '
            f'font-size="11.5" fill="{SUBTEXT}">{tspans}</text>')


if __name__ == "__main__":
    main()
