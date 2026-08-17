"""
Apply the trained interior model to one image: regenerate its interior
candidates fresh, score each with models/interior_model.joblib, fold the
accepted ones into the production crack mask, and save a FINAL RESULT
overlay -- red=crack (original + newly-accepted interior, indistinguishable,
since accepting a region means it now genuinely counts as the same crack),
cyan=rejected artifact. Same two colors as the production overlay itself;
no third color for "newly accepted" -- if you want to see what the model
is proposing before trusting it, that's what the threshold/console output
below are for, not a different color in the image.

Two connected red pixels are ALWAYS one crack, whether the connection comes
from the original production candidates or a newly-accepted interior patch
bridging two of them -- both the on-image numbering and the printed crack
count are based on connected-component GROUPS of the final red mask, not
raw candidate counts, so touching regions are never double-counted.

Saves to TWO places:
  - interior_active_learning/review/applied/ (as before -- this tool's own
    scratch space)
  - ../../results/, as {image}_final_result.png / _bw.png, ALONGSIDE (never
    overwriting) the production pipeline's own {image}_cracks_overlay.png
    etc. -- so there's one place to look for "the current best answer" that
    includes accepted interior corrections, without discarding the pure
    production baseline as a reference point. This is a deliberate, narrow
    exception to this experiment folder's usual read-only-w.r.t.-production
    rule (see common.py's module docstring) -- additive only, nothing in
    results/ is ever modified or deleted by this tool.
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image
from skimage import measure

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import REVIEW_DIR, PROD_RESULTS_DIR
from interior_candidates import run_production_pipeline, run_enhanced_pipeline
from labeling_overlay import draw_labels

# concavity/bridge_corridor have enough negative examples (20 and 8 as of
# this writing) for the plain learned boundary to be reasonably trusted at
# the standard 0.5 cut -- confirmed via a multi-agent experiment comparing
# 5 different fixes for the (much worse) interior_fill imbalance, which also
# checked these two types don't need adjusting. interior_fill itself is
# handled by its own calibrated hybrid rule regardless of this value -- see
# interior_candidates.score_interior_candidates's docstring.
THRESHOLD = 0.5


def apply_to_image(image_name, threshold=THRESHOLD):
    base_stage = run_production_pipeline(image_name)
    crack_mask = np.isin(base_stage["labeled"], base_stage["df"].loc[base_stage["df"]["IsCrack"], "Label"].tolist())

    stage = run_enhanced_pipeline(image_name, stage=base_stage, threshold=threshold)
    labeled, df, img8 = stage["labeled"], stage["df"], stage["img8"]
    final_crack_mask = np.isin(labeled, df.loc[df["IsCrack"], "Label"].tolist())
    artifact_mask = np.isin(labeled, df.loc[~df["IsCrack"], "Label"].tolist())
    n_accepted = len(stage["interior_origin"])
    n_total = stage["n_interior_total"]

    # Connected-component GROUPS of the final red mask -- this is what makes
    # two touching red regions (whether both original, both newly-accepted,
    # or one of each) count and get numbered as ONE crack, never several.
    before_groups = measure.label(crack_mask, connectivity=2)
    after_groups = measure.label(final_crack_mask, connectivity=2)
    n_cracks_before = before_groups.max()
    n_cracks_after = after_groups.max()

    # Exactly matches production save_overlay()'s own colors/alphas (see
    # interior_candidates.build_simple_overlay's docstring) -- red/cyan
    # only, no third color, so this looks like a normal production overlay.
    rgb = np.stack([img8] * 3, axis=-1).astype(float) / 255.0
    red = np.array([1, 0, 0])
    cyan = np.array([0, 0.8, 1.0])
    rgb[artifact_mask] = rgb[artifact_mask] * 0.55 + cyan * 0.45
    rgb[final_crack_mask] = rgb[final_crack_mask] * 0.45 + red * 0.55
    pil_img = Image.fromarray((rgb * 255).astype(np.uint8))

    # One number per crack GROUP (its largest connected sub-area's centroid),
    # not one per original production Label plus one per accepted patch.
    crack_group_props = measure.regionprops(after_groups)
    crack_items = [(p.centroid[1], p.centroid[0], i + 1) for i, p in enumerate(crack_group_props)]
    rejected = df[~df["IsCrack"]]

    draw_labels(pil_img, crack_items, color=(0, 255, 0))
    draw_labels(pil_img, zip(rejected["X"], rejected["Y"], rejected["Label"]), color=(0, 255, 255))

    # Same black-crack-on-white convention as production's own
    # <image>_cracks_bw.tif (detect_cracks.save_bw_image: crack=0, background=255),
    # but reflecting the FINAL mask (original + accepted interior), not just
    # the original production candidates.
    bw = np.where(final_crack_mask, 0, 255).astype(np.uint8)
    bw_img = Image.fromarray(bw, mode="L")

    out_paths = []
    for out_dir in (os.path.join(REVIEW_DIR, "applied"), PROD_RESULTS_DIR):
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{image_name}_final_result.png")
        bw_path = os.path.join(out_dir, f"{image_name}_final_result_bw.png")
        pil_img.save(out_path)
        bw_img.save(bw_path)
        out_paths.append((out_path, bw_path))

    old_frac = crack_mask.mean() * 100
    new_frac = final_crack_mask.mean() * 100
    print(f"{image_name}: accepted {n_accepted}/{n_total} interior candidates (threshold={threshold})")
    print(f"  crack count (connected groups): {n_cracks_before} -> {n_cracks_after}")
    print(f"  crack area: {old_frac:.2f}% -> {new_frac:.2f}%")
    print(f"  result: {out_paths[0][0]}  (also written to {out_paths[1][0]})")
    print(f"  bw mask: {out_paths[0][1]}  (also written to {out_paths[1][1]})")
    print("  colors: red=crack (green numbers, one per connected crack), cyan=rejected artifact (cyan numbers)")
    return out_paths[0]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("image_name")
    ap.add_argument("--threshold", type=float, default=THRESHOLD,
                     help="concavity/bridge_corridor only -- interior_fill uses its own calibrated rule")
    args = ap.parse_args()
    apply_to_image(args.image_name, args.threshold)
