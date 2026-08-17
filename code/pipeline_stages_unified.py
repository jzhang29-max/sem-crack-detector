"""
UNIFIED-MODEL variant of pipeline_stages.py: identical Steps A-D (image
loading/contrast, background flattening, dark-region segmentation, vesselness
+ candidate extraction -- none of that depends on which classifier scores
the candidates), but stage E/F's classification step calls
unified_pipeline.score_pass1_candidates() -- the SAME unified_model.joblib
that also handles the interior/concavity/bridge candidates in stage H,
instead of a separate 8-feature crack_classifier.joblib -- so this can never
silently drift from what unified_pipeline.py actually does (shared function,
not a re-implementation).

Only used by generate_full_workflow_diagram_unified.py.
"""
import os
import sys

import numpy as np
import pandas as pd
import tifffile
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect_cracks import (
    load_as_uint8, find_field_of_view, flatten_background, segment_dark_regions,
    clean_mask, compute_vesselness, extract_candidates, merge_large_cracks,
)

IAL_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                             "interior_active_learning", "code")
sys.path.insert(0, os.path.abspath(IAL_CODE_DIR))
from unified_pipeline import score_pass1_candidates, _load_unified_bundle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIGINAL_DIR = os.path.join(ROOT, "original")
RESULTS_DIR = os.path.join(ROOT, "results")
LEDGER_PATH = os.path.join(ROOT, "manual_corrections_ledger.csv")


def load_hard_overrides(image_name):
    if not os.path.exists(LEDGER_PATH):
        return None
    ledger = pd.read_csv(LEDGER_PATH)
    g = ledger[ledger["SourceImage"] == image_name]
    if len(g) == 0:
        return None
    is_crack = g["CorrectedTo"].astype(str).str.strip().str.lower().isin(["true", "1"])
    return dict(zip(g["Label"].astype(int), is_crack))


def detail_window(cx, cy, w, h, half_w=800, half_h=300):
    x0 = max(0, int(cx - half_w))
    x1 = min(w, int(cx + half_w))
    y0 = max(0, int(cy - half_h))
    y1 = min(h, int(cy + half_h))
    return x0, y0, x1, y1


def make_overlay_rgb(img8, labeled, df_, bridge=None, crop_box=None, bridge_color=(1, 1, 0)):
    kept = df_[df_["IsCrack"]]
    rejected = df_[~df_["IsCrack"]]
    kept_mask = np.isin(labeled, kept["Label"].tolist()) if len(kept) else np.zeros(labeled.shape, dtype=bool)
    if bridge is not None:
        kept_mask = kept_mask | bridge
    rejected_mask = np.isin(labeled, rejected["Label"].tolist()) if len(rejected) else np.zeros(labeled.shape, dtype=bool)
    base = np.stack([img8] * 3, axis=-1).astype(float) / 255.0
    out = base.copy()
    red = np.array([1, 0, 0]); cyan = np.array([0, 0.8, 1])
    out[kept_mask] = out[kept_mask] * 0.45 + red * 0.55
    out[rejected_mask] = out[rejected_mask] * 0.55 + cyan * 0.45
    if bridge is not None and bridge.any():
        bc = np.array(bridge_color)
        out[bridge] = out[bridge] * 0.3 + bc * 0.7
    if crop_box:
        cx0, cy0, cx1, cy1 = crop_box
        out = out[cy0:cy1, cx0:cx1]
    return out


def compute_pipeline_stages_unified(image_name, half_w=800, half_h=300):
    """Run Steps A-D + the unified model's Pass 1 (classify + merge) on one
    image, returning every intermediate array -- same field names as
    pipeline_stages.compute_pipeline_stages() so the diagram generator's
    draw_card() calls need no changes for stages A-D."""
    image_path = os.path.join(ORIGINAL_DIR, f"{image_name}.tif")

    raw16 = tifffile.imread(image_path)
    if raw16.ndim == 3:
        raw16 = raw16[..., 0]
    raw_display = load_as_uint8(image_path, low_pct=0.5, high_pct=99.9)

    img8_full = load_as_uint8(image_path, low_pct=1.0, high_pct=99.5)

    x0, y0, x1, y1 = find_field_of_view(img8_full)
    img8 = img8_full[y0:y1, x0:x1]
    was_cropped = (x0, y0, x1, y1) != (0, 0, img8_full.shape[1], img8_full.shape[0])

    flat = flatten_background(img8, sigma=40)
    dark_mask = segment_dark_regions(flat, denoise_sigma=1.0, img8=img8, absolute_dark_thresh=10)
    clean = clean_mask(dark_mask, open_radius=1, close_radius=3, min_area_px=13)
    vesselness = compute_vesselness(flat, sigma_min=1, sigma_max=6)
    labeled, df = extract_candidates(clean, flat, vesselness, min_area_px=40)
    n_candidates = len(df)

    # Same unified_model.joblib stage H reuses for interior candidates --
    # see unified_pipeline.score_pass1_candidates()'s docstring.
    bundle = _load_unified_bundle()
    df = score_pass1_candidates(df, labeled, img8, flat, vesselness, bundle)
    overrides = load_hard_overrides(image_name)
    if overrides:
        for label, is_crack in overrides.items():
            m = df["Label"] == label
            if m.any():
                df.loc[m, "IsCrack"] = is_crack
    df_pre_merge = df.copy()
    n_kept_pre_merge = int(df["IsCrack"].sum())

    df_final, bridge_mask = merge_large_cracks(labeled, df, flat, min_area_px=1000, max_gap_px=80)
    df_final = df_final.sort_values("Area", ascending=False).reset_index(drop=True)
    n_kept_final = int(df_final["IsCrack"].sum())

    # Pass-1-only B&W mask / overlay -- computed live (not read from disk),
    # since results/ only has the OLD two-stage system's production output.
    crack_mask_full = np.isin(labeled, df_final.loc[df_final["IsCrack"], "Label"].tolist())
    final_bw = np.where(crack_mask_full, 0, 255).astype(np.uint8)
    final_overlay = (make_overlay_rgb(img8, labeled, df_final, bridge=bridge_mask) * 255).astype(np.uint8)

    kept_main = df_final[df_final["IsCrack"]].sort_values("Area", ascending=False)
    cx, cy = float(kept_main.iloc[0]["X"]), float(kept_main.iloc[0]["Y"])
    h, w = img8.shape
    dwin = detail_window(cx, cy, w, h, half_w=half_w, half_h=half_h)
    dx0, dy0, dx1, dy1 = dwin

    def dcrop(arr2d):
        return arr2d[dy0:dy1, dx0:dx1]

    final_overlay_crop = final_overlay[dy0:dy1, dx0:dx1]

    bridge_crop = bridge_mask[dy0:dy1, dx0:dx1] if bridge_mask is not None else None
    labeled_crop = labeled[dy0:dy1, dx0:dx1]
    img8_crop = dcrop(img8)
    overlay_pre_merge_crop = make_overlay_rgb(img8_crop, labeled_crop, df_pre_merge, bridge=None, crop_box=None)
    overlay_post_merge_crop = make_overlay_rgb(img8_crop, labeled_crop, df_final, bridge=bridge_crop, crop_box=None)

    from skimage.color import label2rgb
    label_rgb_crop = label2rgb(labeled_crop, image=dcrop(flat), bg_label=0, alpha=0.45, bg_color=None)

    return {
        "image_name": image_name,
        "raw16": raw16,
        "raw_display": raw_display,
        "img8_full": img8_full,
        "crop_box": (x0, y0, x1, y1),
        "was_cropped": was_cropped,
        "img8": img8,
        "img8_crop": img8_crop,
        "flat": flat,
        "flat_crop": dcrop(flat),
        "dark_mask_crop": dcrop(dark_mask),
        "clean_crop": dcrop(clean),
        "vesselness_crop": dcrop(vesselness),
        "labeled_crop": labeled_crop,
        "label_rgb_crop": label_rgb_crop,
        "overlay_pre_merge_crop": overlay_pre_merge_crop,
        "overlay_post_merge_crop": overlay_post_merge_crop,
        "final_bw_crop": dcrop(final_bw),
        "final_overlay_crop": final_overlay_crop,
        "detail_window": dwin,
        "n_candidates": n_candidates,
        "n_kept_pre_merge": n_kept_pre_merge,
        "n_kept_final": n_kept_final,
        "n_merge_bridges": int(bridge_mask.any()) if bridge_mask is not None else 0,
    }
