"""
UNIFIED-MODEL variant of generate_full_workflow_diagram.py: same visual
language and stage grid, but reflects the single-shared-model architecture
validated in this experiment folder -- stage (E) and stage (H) now run the
literal SAME unified_model.joblib (one 11-feature logistic regression,
reused for both the original darkness-threshold candidates AND the
concavity/bridge/interior-fill candidates), instead of two separately
trained models. See unified_pipeline.py's module docstring for the full
rationale, and MODEL_VALIDATION_BENCHMARK.md / this session's benchmark
figures for the cross-validated evidence this doesn't perform worse than
the two-model system.

Usage
-----
    python3 generate_full_workflow_diagram_unified.py [image_name_without_extension]
"""
import os
import subprocess
import sys

if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
    print(__doc__ or "")
    print("usage: generate_full_workflow_diagram_unified.py [IMAGE_NAME]\n"
          "  IMAGE_NAME  a frame in original/, without the .tif "
          "(default 260708_316_H_b2_front_CBS_002)")
    sys.exit(0)
# argv[1] is DATA, an image name -- so a flag-shaped argument must be rejected by name rather
# than looked up as a frame. Otherwise `--help` is treated as an image and the error names a
# missing file, which reads like a broken checkout.
if len(sys.argv) > 1 and sys.argv[1].startswith("-"):
    print(f"unknown option {sys.argv[1]!r}. Pass an image name, or --help.")
    sys.exit(2)

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# generate_scientific_diagram.py was moved to archive/superseded_code/ when it was
# superseded, which silently broke THIS script: the module is still tracked, so a clone has
# the file, but it is no longer on sys.path and the import died with ModuleNotFoundError --
# a documented command that could not run at all. Point at where it actually lives rather
# than un-archiving it.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "archive", "superseded_code"))

from generate_scientific_diagram import (
    SVG, icon_svg as _base_icon_svg, doc_glyph_svg, esc, wrap_tspans, rounded_rect,
    to_data_uri, draw_card, draw_document, doc_cy_above, doc_arrow_start_y,
    DOC_COLOR, INK, SUBTEXT, _darken,
)
from pipeline_stages_unified import compute_pipeline_stages_unified, ROOT, RESULTS_DIR

IAL_ROOT = os.path.join(ROOT, "interior_active_learning")
OUT_DIR = os.path.join(ROOT, "pipeline_diagram")
os.makedirs(OUT_DIR, exist_ok=True)
IMAGE_NAME = sys.argv[1] if len(sys.argv) > 1 else "260708_316_H_b2_front_CBS_002"

STAGE_COLORS = ["#264653", "#2A9D8F", "#5E9C6C", "#E9B44C", "#F0954A", "#E76F51"]
PART2_COLORS = ["#7A4C9E", "#C1272D"]

FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD_PATH = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
# Fallbacks for a machine without the macOS fonts; see the note in build_figures.font().
FONT_FALLBACKS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                  "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                  "/usr/share/fonts/dejavu/DejaVuSans.ttf")
FONT_BOLD_FALLBACKS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                       "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                       "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf")


def _font(size, bold=False):
    try:
        return ImageFont.truetype(FONT_BOLD_PATH if bold else FONT_PATH, size)
    except Exception:
        for _fb in (FONT_BOLD_FALLBACKS if bold else FONT_FALLBACKS):
            try:
                return ImageFont.truetype(_fb, size)
            except Exception:
                pass
        try:
            return ImageFont.load_default(size=size)
        except (TypeError, OSError):   # no size arg (Pillow < 10.1), or no font to load
            return ImageFont.load_default()


def crop_center(img, cx_frac, cy_frac, w_frac, h_frac):
    W, H = img.size
    cx, cy = int(W * cx_frac), int(H * cy_frac)
    w, h = int(W * w_frac), int(H * h_frac)
    x0, y0 = max(0, cx - w // 2), max(0, cy - h // 2)
    x1, y1 = min(W, x0 + w), min(H, y0 + h)
    return img.crop((x0, y0, x1, y1))


def legend_card(title, swatches, accent):
    W, H = 520, 430
    img = Image.new("RGB", (W, H), "#FFFFFF")
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, W - 1, H - 1], outline="#E1E8EB", width=2)
    draw.rectangle([0, 0, W - 1, 64], fill=accent)
    draw.text((22, 18), title, fill="white", font=_font(24, bold=True))
    y, r = 110, 22
    for color, label, note in swatches:
        draw.ellipse([40, y, 40 + 2 * r, y + 2 * r], fill=color, outline=INK, width=2)
        draw.text((100, y - 2), label, fill=INK, font=_font(26, bold=True))
        draw.text((100, y + 32), note, fill=SUBTEXT, font=_font(15))
        y += 92
    return img


def stat_card(title, lines, accent):
    W, H = 520, 430
    img = Image.new("RGB", (W, H), "#FFFFFF")
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, W - 1, H - 1], outline="#E1E8EB", width=2)
    draw.rectangle([0, 0, W - 1, 64], fill=accent)
    draw.text((22, 18), title, fill="white", font=_font(24, bold=True))
    y = 96
    for line, sub in lines:
        draw.text((26, y), line, fill=INK, font=_font(28, bold=True))
        draw.text((26, y + 38), sub, fill=SUBTEXT, font=_font(15))
        y += 82
    return img


def icon_svg(kind, color, size=1.0):
    sw = 0.16 * size
    if kind == "layers":
        parts = []
        for i, dy in enumerate([-0.32, 0, 0.32]):
            op = 1.0 - i * 0.18
            parts.append(f'<polygon points="-0.7,{dy} 0,{dy-0.22} 0.7,{dy} 0,{dy+0.22}" '
                         f'fill="none" stroke="{color}" stroke-width="{sw}" opacity="{op:.2f}"/>')
        return "".join(parts)
    if kind == "brush":
        return (f'<path d="M -0.55 0.6 L -0.15 0.15 L 0.15 0.45 L -0.25 0.85 Z" fill="{color}"/>'
                f'<rect x="-0.05" y="-0.65" width="0.75" height="0.32" rx="0.08" '
                f'transform="rotate(45 -0.05 -0.65)" fill="none" stroke="{color}" stroke-width="{sw}"/>'
                f'<circle cx="0.55" cy="-0.55" r="0.1" fill="{color}"/>')
    if kind == "loop":
        parts = [f'<path d="M -0.6 -0.15 A 0.65 0.65 0 1 1 -0.6 0.2" fill="none" '
                 f'stroke="{color}" stroke-width="{sw*1.2}" stroke-linecap="round"/>']
        parts.append(f'<polygon points="-0.6,0.55 -0.85,0.1 -0.35,0.15" fill="{color}"/>')
        return "".join(parts)
    return _base_icon_svg(kind, color, size)


def draw_card_v2(svg, x, y, w, h, stage, color, grad_id):
    svg.raw(rounded_rect(x, y, w, h, 16, "white", stroke="#E1E8EB", sw=1.5, filter_id="cardShadow"))
    header_h = 108
    svg.raw(f'<path d="M {x} {y+16} Q {x} {y} {x+16} {y} L {x+w-16} {y} Q {x+w} {y} {x+w} {y+16} '
            f'L {x+w} {y+header_h} L {x} {y+header_h} Z" fill="url(#{grad_id})"/>')
    badge_r = 34
    bcx, bcy = x + 58, y + header_h / 2
    svg.raw(f'<circle cx="{bcx}" cy="{bcy}" r="{badge_r}" fill="white" filter="url(#badgeShadow)"/>')
    svg.raw(f'<g transform="translate({bcx},{bcy}) scale({badge_r*0.62})">{icon_svg(stage["icon"], color)}</g>')
    svg.raw(f'<text x="{x+104}" y="{bcy-10}" font-family="Helvetica, Arial, sans-serif" font-size="17.5" '
            f'font-weight="700" fill="white">({stage["key"]}) {esc(stage["title"])}</text>')
    sub_tspans, _ = wrap_tspans(stage["subtitle"], 34, x + 104, 15)
    svg.raw(f'<text x="{x+104}" y="{bcy+10}" font-family="Helvetica, Arial, sans-serif" font-size="11.8" '
            f'fill="#F3E9F5">{sub_tspans}</text>')

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


def main():
    print(f"Building UNIFIED-MODEL full-workflow SVG diagram for {IMAGE_NAME} ...")
    s = compute_pipeline_stages_unified(IMAGE_NAME, half_w=520, half_h=430)

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
             subtitle="Frangi vesselness + connected components → 11-D feature vectors",
             thumbs=[("Vesselness map", s["vesselness_crop"], "inferno"),
                     (f"{s['n_candidates']} candidates", s["label_rgb_crop"], None)]),
        dict(key="E", title="Unified ML Classification & Merging", icon="classifier",
             subtitle="Unified LogReg (same model as H) + MST-based fragment bridging",
             thumbs=[("Classified", s["overlay_pre_merge_crop"], None),
                     ("Merged", s["overlay_post_merge_crop"], None)]),
        dict(key="F", title="Output Generation (Pass 1)", icon="check",
             subtitle="Base mask + numbered overlay -- before manual correction / interior candidates",
             thumbs=[("B&W mask", s["final_bw_crop"], None), ("Annotated overlay", s["final_overlay_crop"], None)]),
    ]

    paint_template = Image.open(os.path.join(IAL_ROOT, "paint", f"{IMAGE_NAME}_paint_template.png")).convert("RGB")
    paint_crop = crop_center(paint_template, 0.5, 0.46, 0.6, 0.34)
    paint_legend = legend_card("Paint colors", [
        ("#ff0000", "Red", "add or correct to crack"),
        ("#00ccff", "Cyan", "add or correct to artifact"),
        ("#ff00ff", "Magenta", "erase from candidacy"),
    ], PART2_COLORS[0])

    review_dir = os.path.join(IAL_ROOT, "review", "round_4")
    review_files = sorted(p for p in os.listdir(review_dir) if p.startswith(IMAGE_NAME) and p.endswith(".png")) \
        if os.path.isdir(review_dir) else []
    review_crop = None
    if review_files:
        review_img = Image.open(os.path.join(review_dir, review_files[0])).convert("RGB")
        review_crop = crop_center(review_img, 0.5, 0.35, 1.0, 0.55)

    # Read the live UNIFIED model's own calibration + pooled training set --
    # same "don't hardcode a snapshot" discipline as the two-stage diagram,
    # now pointed at the single shared model and its pooled (Step-E +
    # Step-H) 11-feature dataset instead of the interior-only one.
    import joblib
    sys.path.insert(0, os.path.join(IAL_ROOT, "code"))
    experiments_dir = os.path.join(IAL_ROOT, "code", "experiments")
    sys.path.insert(0, experiments_dir)
    from unified_data import load_unified_pooled
    _bundle = joblib.load(os.path.join(IAL_ROOT, "models", "unified_model.joblib"))
    _rule = _bundle.get("interior_fill_rule")
    _pooled = load_unified_pooled()
    _n_pos, _n_neg = int(_pooled["IsCrack"].sum()), int((~_pooled["IsCrack"]).sum())
    if _rule is not None:
        calib_card = stat_card("unified model calibration", [
            (f"{_n_pos} pos / {_n_neg} neg", f"pooled examples, {_pooled['SourceImage'].nunique()} images"),
            (f"{_rule['recall_on_known_positives']*100:.1f}% recall", "interior_fill, known positives"),
            (f"{_rule['full_pool_accept_rate']*100:.1f}% accept", "interior_fill, full candidate pool"),
        ], PART2_COLORS[1])
    else:
        calib_card = stat_card("unified model calibration", [
            (f"{_n_pos} pos / {_n_neg} neg", f"pooled examples, {_pooled['SourceImage'].nunique()} images"),
            ("no rule calibrated", "falls back to a flat threshold"),
        ], PART2_COLORS[1])

    applied_dir = os.path.join(IAL_ROOT, "review", "applied")
    final_overlay = Image.open(os.path.join(applied_dir, f"{IMAGE_NAME}_unified_final_result.png")).convert("RGB")
    final_overlay_crop = crop_center(final_overlay, 0.5, 0.46, 0.55, 0.34)

    g_thumbs = [("Paint colors", np.array(paint_legend), None)]
    g_thumbs.append(("Harder candidates", np.array(review_crop), None) if review_crop is not None
                     else ("Same red/cyan as (F)", np.array(paint_crop), None))

    part2_stages = [
        dict(key="G", title="Manual Correction", icon="brush",
             subtitle="Paint app (add/correct/erase) plus review sheets for harder candidates the base mask missed",
             thumbs=g_thumbs),
        dict(key="H", title="Unified ML Classification (Pass 2)", icon="check",
             subtitle="SAME model as (E) + hybrid rule (leave-one-out validated) -- "
                       "one shared classifier, not two",
             thumbs=[("Calibration", np.array(calib_card), None), ("Final result", np.array(final_overlay_crop), None)]),
    ]

    W = 1760
    MARGIN = 70
    CARD_W = (W - 2 * MARGIN - 2 * 60) / 3
    CARD_H = 470
    ROW_GAP = 190
    TITLE_H = 260
    EXTRA_BAND_H = 480
    PART2_TITLE_H = 110
    PART2_GAP = 80
    PART2_CARD_W = CARD_W
    PART2_CARD_H = 460
    PART2_BAND_H = PART2_TITLE_H + PART2_CARD_H + 90
    CAPTION_H = 340
    H = TITLE_H + CARD_H + ROW_GAP + CARD_H + EXTRA_BAND_H + PART2_BAND_H + CAPTION_H + 60

    svg = SVG(W, H)
    svg.add_shadow_filter("cardShadow", dy=10, blur=16, opacity=0.16)
    svg.add_shadow_filter("thumbShadow", dy=4, blur=7, opacity=0.22)
    svg.add_shadow_filter("badgeShadow", dy=3, blur=5, opacity=0.30)
    for i, color in enumerate(STAGE_COLORS):
        svg.add_gradient(f"grad{i}", color, _darken(color, 0.72))
        svg.add_arrowhead(f"arrow{i}", color)
    for i, color in enumerate(PART2_COLORS):
        svg.add_gradient(f"grad2_{i}", color, _darken(color, 0.72))
        svg.add_arrowhead(f"arrow2_{i}", color)
    svg.add_gradient("bg", "#F7FAFC", "#EAF1F3")
    svg.add_gradient("gradReview", DOC_COLOR, _darken(DOC_COLOR, 0.72))
    svg.add_arrowhead("arrowDoc", DOC_COLOR)

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
            f'font-size="16" font-style="italic" fill="{SUBTEXT}">UNIFIED single-model architecture: detection → '
            f'manual correction → machine learning (same model, two passes), worked example: {esc(IMAGE_NAME)}</text>')

    row_y = [TITLE_H, TITLE_H + CARD_H + ROW_GAP]
    col_x = [MARGIN, MARGIN + CARD_W + 60, MARGIN + 2 * (CARD_W + 60)]
    centers = {}
    positions = [(col_x[0], row_y[0]), (col_x[1], row_y[0]), (col_x[2], row_y[0]),
                 (col_x[2], row_y[1]), (col_x[1], row_y[1]), (col_x[0], row_y[1])]

    for i, stage in enumerate(stages):
        x, y = positions[i]
        color = STAGE_COLORS[i]
        centers[i] = (x, y, x + CARD_W, y + CARD_H)
        draw_card(svg, x, y, CARD_W, CARD_H, stage, color, i)

    def connect(i, j, colors=STAGE_COLORS, prefix="arrow"):
        xi0, yi0, xi1, yi1 = centers[i]
        xj0, yj0, xj1, yj1 = centers[j]
        ay = yi0 + 60
        if xj0 >= xi1:
            x_from, x_to = xi1 - 6, xj0 + 6
        else:
            x_from, x_to = xi0 + 6, xj1 - 6
        svg.raw(f'<path d="M {x_from} {ay} L {x_to} {ay}" stroke="{colors[j % len(colors)]}" stroke-width="4" '
                f'fill="none" marker-end="url(#{prefix}{j})"/>')

    connect(0, 1); connect(1, 2)
    connect(3, 4); connect(4, 5)

    x2_0, y2_0, x2_1, y2_1 = centers[2]
    x3_0, y3_0, x3_1, y3_1 = centers[3]
    wrap_color = STAGE_COLORS[3]
    cxw = (x2_0 + x2_1) / 2
    svg.raw(f'<path d="M {cxw} {y2_1} C {cxw+40} {y2_1+ROW_GAP*0.4}, {cxw-40} {y3_0-ROW_GAP*0.4}, {cxw} {y3_0}" '
            f'stroke="{wrap_color}" stroke-width="4" fill="none" marker-end="url(#arrow3)"/>')

    docA_size = 78
    docA_x, docA_y = centers[0][0] + CARD_W * 0.5, doc_cy_above(row_y[0], docA_size)
    draw_document(svg, docA_x, docA_y, docA_size, "Raw 16-bit TIFF\nmicrograph", "image")
    svg.raw(f'<path d="M {docA_x} {doc_arrow_start_y(docA_y, docA_size)} L {docA_x} {row_y[0]-4}" '
            f'stroke="{DOC_COLOR}" stroke-width="3" fill="none" marker-end="url(#arrowDoc)"/>')

    ex, ey0, ex1, ey1 = centers[4]
    mcx = (ex + ex1) / 2
    doc_size = 66
    doc_y = doc_cy_above(row_y[1], doc_size)
    doc_arrow_y = doc_arrow_start_y(doc_y, doc_size)
    doc_spacing = (CARD_W - 90) / 2
    e_docs = [
        (mcx - doc_spacing, f"Feature vectors\n(11 x {s['n_candidates']})", "table"),
        (mcx, "unified_model.joblib\n(reused again in H)", "model"),
        (mcx + doc_spacing, "Manual corrections +\noriginal-candidate features\n(pooled, .csv)", "table"),
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

    review_paths = sorted(
        p for p in os.listdir(RESULTS_DIR) if p.startswith(f"{IMAGE_NAME}_review_page") and p.endswith(".png")
    ) if os.path.isdir(RESULTS_DIR) else []
    review_y0 = row_y[1] + CARD_H + 30
    review_h = 430
    if review_paths:
        review_x0, review_x1 = centers[4][0], centers[3][2]
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

    part2_y0 = review_y0 + review_h + 50
    svg.raw(f'<line x1="{MARGIN}" y1="{part2_y0-14}" x2="{W-MARGIN}" y2="{part2_y0-14}" '
            f'stroke="#D8E0E4" stroke-width="2" stroke-dasharray="2,6"/>')
    svg.raw(f'<text x="{W/2}" y="{part2_y0+18}" text-anchor="middle" font-family="Georgia, serif" '
            f'font-size="24" font-weight="700" fill="{PART2_COLORS[1]}">Manual Correction &amp; Retraining '
            f'(shared model)</text>')
    svg.raw(f'<text x="{W/2}" y="{part2_y0+42}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" '
            f'font-size="14" font-style="italic" fill="{SUBTEXT}">Read-only w.r.t. the base pipeline above (F) -- '
            f'lets a human correct what it missed, then retrains the SAME model (E) reuses for the next round</text>')

    p2_row_y = part2_y0 + PART2_TITLE_H
    p2_total_w = 2 * PART2_CARD_W + PART2_GAP
    p2_x0 = (W - p2_total_w) / 2
    p2_col_x = [p2_x0, p2_x0 + PART2_CARD_W + PART2_GAP]
    p2_centers = {}
    for i, stage in enumerate(part2_stages):
        x = p2_col_x[i]
        p2_centers[i] = (x, p2_row_y, x + PART2_CARD_W, p2_row_y + PART2_CARD_H)
        draw_card_v2(svg, x, p2_row_y, PART2_CARD_W, PART2_CARD_H, stage, PART2_COLORS[i], f"grad2_{i}")

    gx0, gy0, gx1, gy1 = p2_centers[0]
    hx0, hy0, hx1, hy1 = p2_centers[1]
    gcx, hcx = (gx0 + gx1) / 2, (hx0 + hx1) / 2
    ay = gy0 + 60
    svg.raw(f'<path d="M {gx1-6} {ay} L {hx0+6} {ay}" stroke="{PART2_COLORS[1]}" stroke-width="4" '
            f'fill="none" marker-end="url(#arrow2_1)"/>')

    svg.raw(f'<path d="M {fcx2} {out_y+45} C {fcx2-60} {(out_y+p2_row_y)/2}, {gcx+60} {(out_y+p2_row_y)/2}, '
            f'{gcx} {p2_row_y-4}" stroke="{PART2_COLORS[0]}" stroke-width="3.5" stroke-dasharray="1,7" '
            f'stroke-linecap="round" fill="none" marker-end="url(#arrow2_0)"/>')
    svg.raw(f'<text x="{(fcx2+gcx)/2-40}" y="{(out_y+p2_row_y)/2+40}" text-anchor="middle" '
            f'font-family="Helvetica, Arial, sans-serif" font-size="11.5" font-style="italic" '
            f'fill="{SUBTEXT}">reads production result (read-only)</text>')

    loop_y = hy1 + 55
    svg.raw(f'<path d="M {hx1-30} {hy1} L {hx1-30} {loop_y} L {gx0+30} {loop_y} L {gx0+30} {gy1}" '
            f'stroke="{PART2_COLORS[1]}" stroke-width="3" stroke-dasharray="8,6" fill="none" '
            f'marker-end="url(#arrow2_0)"/>')
    svg.raw(f'<text x="{(gx0+hx1)/2}" y="{loop_y+18}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" '
            f'font-size="12.5" font-style="italic" fill="{PART2_COLORS[1]}">newly labeled corrections retrain the '
            f'ONE shared model for the next round</text>')

    cap_y = H - CAPTION_H + 30
    svg.raw(rounded_rect(MARGIN, cap_y - 24, W - 2 * MARGIN, CAPTION_H - 40, 10, "white",
                          stroke="#D8E0E4", sw=1.5, filter_id="cardShadow"))
    caption = (
        f"Figure 1. UNIFIED single-model crack-detection pipeline for SEM/TXM micrographs of annealed stainless "
        f"steel, plus its manual-correction and retraining loop. Raw 16-bit micrographs are contrast-stretched and "
        f"auto-cropped (A), background-flattened (B), and segmented into candidate dark regions (C). A Frangi "
        f"vesselness filter and connected-component analysis extract an 11-feature vector per candidate (D) -- 9 "
        f"real (size, shape, brightness, vesselness) plus 2 neutral placeholders for the two crack-context features "
        f"(distance to / boundary contact with a confirmed crack) that cannot yet be computed before any candidate "
        f"has been confirmed. ONE shared 11-feature logistic-regression classifier -- the SAME .joblib file reused "
        f"again in stage (H) -- separates these into cracks and artifacts; a minimum-spanning-tree bridges fragments "
        f"of the same physical crack (E), producing a Pass-1 mask, overlay, and measurements (F). Read-only against "
        f"that result, a human can then directly add, correct, or erase regions in a browser paint app (red=crack/"
        f"cyan=artifact/magenta=erase, two touching red regions always merge into one crack) or work through review "
        f"sheets of harder candidates the base mask's brightness threshold missed -- notches along a crack's own "
        f"boundary, corridors between bridged fragments, or the wider gradient-fade interior a flattening filter can "
        f"suppress (G). Those candidates get their REAL crack-context features computed relative to the now-real "
        f"Pass-1 crack mask, and are scored by the SAME shared model from (E) plus an extra calibrated distance/"
        f"brightness rule for the hardest candidate type, leave-one-out validated against every known negative; "
        f"accepted regions are folded into the crack mask, and the freshly labeled corrections (both original-"
        f"candidate and interior-candidate) retrain that one shared model for the next round (H). Using one model "
        f"instead of two collapses a training-data split that no longer reflected a real architectural difference "
        f"between candidate types -- see MODEL_VALIDATION_BENCHMARK.md for the cross-validated evidence this "
        f"performs comparably to the previous two-model system. Worked example shown: {IMAGE_NAME}."
    )
    tspans, nlines = wrap_tspans(caption, 156, MARGIN + 24, 21)
    svg.raw(f'<text x="{MARGIN+24}" y="{cap_y+6}" font-family="Georgia, serif" font-size="13.5" fill="{INK}">{tspans}</text>')

    svg_path = os.path.join(OUT_DIR, f"full_workflow_unified_{IMAGE_NAME}.svg")
    with open(svg_path, "w") as f:
        f.write(svg.render())
    print(f"Saved SVG: {svg_path}")

    png_path = os.path.join(OUT_DIR, f"full_workflow_unified_{IMAGE_NAME}.png")
    subprocess.run(["rsvg-convert", "-w", str(W * 2), "-h", str(H * 2), svg_path, "-o", png_path], check=True)
    print(f"Saved PNG: {png_path}")


if __name__ == "__main__":
    main()
