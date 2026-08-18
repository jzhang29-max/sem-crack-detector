"""
Apply the retrained model to all 63 images and compare it, per image, against
the current production model.

This is a MEASUREMENT pass, not a deployment: models/crack_classifier.joblib
is not touched and no paint template is rewritten, so an in-progress painting
session is unaffected. It answers "how does the retrained model actually
behave on the whole set", which a held-out AUC on one image cannot.

Both models are run over the SAME candidate set per image -- identical
segmentation, identical features, only the classifier differs -- so every
difference reported here is attributable to the model and nothing else.

For images that have human corrections, agreement with those corrections is
reported too. Note the asymmetry when reading it: 24 of the 32 reviewed images
have only crack marks, so on those images "agreement" can only measure recall,
never false positives. Images with both classes are flagged with *.

    python3 apply_v3_all_images.py
Writes ../../training_data/v3_vs_production_per_image.csv
"""
import os
import sys
import warnings
from multiprocessing import Pool

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "code"))

import joblib
import numpy as np
import pandas as pd

from common import (ORIGINAL_DIR, PAINT_DIR, PROJECT_ROOT, PROD_MODEL_PATH,
                     contrast_kwargs_for, load_correction_mask)
from detect_cracks import (load_as_uint8, find_field_of_view, flatten_background,
                            segment_dark_regions, clean_mask, compute_vesselness,
                            exclude_border_background, extract_candidates)

V3_PATH = os.path.join(PROJECT_ROOT, "models", "crack_classifier_v3_weighted.joblib")
OUT_CSV = os.path.join(PROJECT_ROOT, "training_data", "v3_vs_production_per_image.csv")
FORCE_KEEP_AREA = 50000     # matches classify_with_model's own default


def process(image_name):
    try:
        img8 = load_as_uint8(os.path.join(ORIGINAL_DIR, f"{image_name}.tif"),
                             **contrast_kwargs_for(image_name))
        x0, y0, x1, y1 = find_field_of_view(img8)
        img8 = img8[y0:y1, x0:x1]
        flat = flatten_background(img8)
        clean = clean_mask(segment_dark_regions(flat, img8=img8), min_area_px=13)
        ves = compute_vesselness(flat)
        clean = exclude_border_background(clean, ves)
        labeled, df = extract_candidates(clean, flat, ves, min_area_px=40)
        if not len(df):
            return None

        prod = joblib.load(PROD_MODEL_PATH)
        v3 = joblib.load(V3_PATH)
        area = df["Area"].values
        big = area >= FORCE_KEEP_AREA

        p_prod = prod["clf"].predict_proba(
            prod["scaler"].transform(df[prod["feature_names"]].values))[:, 1]
        p_v3 = v3["clf"].predict_proba(
            v3["scaler"].transform(df[v3["feature_names"]].values))[:, 1]
        acc_prod = (p_prod >= 0.5) | big
        acc_v3 = (p_v3 >= v3["threshold"]) | big

        total_px = labeled.size
        rec = {
            "SourceImage": image_name, "n_candidates": len(df),
            "prod_accepted": int(acc_prod.sum()), "v3_accepted": int(acc_v3.sum()),
            "prod_area_pct": 100.0 * area[acc_prod].sum() / total_px,
            "v3_area_pct": 100.0 * area[acc_v3].sum() / total_px,
            "flipped_to_crack": int((acc_v3 & ~acc_prod).sum()),
            "flipped_to_not": int((~acc_v3 & acc_prod).sum()),
            "n_reviewed": 0, "prod_agree": np.nan, "v3_agree": np.nan,
            "has_both_classes": False,
        }

        mask = load_correction_mask(image_name, labeled.shape)
        if mask is not None:
            truth, idx = [], []
            for i, lbl in enumerate(df["Label"].values):
                vals = mask[labeled == int(lbl)]
                if vals.size == 0:
                    continue
                c = np.bincount(vals, minlength=4)
                if c[1:].sum() == 0:
                    continue
                v = int(c[1:].argmax() + 1)
                if v in (1, 2):
                    truth.append(v == 1); idx.append(i)
            if truth:
                t = np.array(truth); idx = np.array(idx)
                rec["n_reviewed"] = len(t)
                rec["prod_agree"] = float((acc_prod[idx] == t).mean())
                rec["v3_agree"] = float((acc_v3[idx] == t).mean())
                rec["has_both_classes"] = bool(t.any() and (~t).any())

        star = "*" if rec["has_both_classes"] else " "
        ag = ("" if rec["n_reviewed"] == 0 else
              f" | reviewed {rec['n_reviewed']:>4d}{star} agree prod {rec['prod_agree']:.1%} "
              f"-> v3 {rec['v3_agree']:.1%}")
        print(f"{image_name:32s} cand {rec['n_candidates']:>5d} | accept "
              f"{rec['prod_accepted']:>5d} -> {rec['v3_accepted']:>5d} | area "
              f"{rec['prod_area_pct']:>5.2f}% -> {rec['v3_area_pct']:>5.2f}%{ag}", flush=True)
        return rec
    except Exception as e:
        print(f"{image_name:32s} FAILED: {e}", flush=True)
        return None


if __name__ == "__main__":
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(ORIGINAL_DIR)
                    if f.lower().endswith(".tif"))
    print(f"applying retrained model to {len(names)} images "
          f"(production model NOT modified)\n")
    with Pool(3) as pool:
        rows = [r for r in pool.map(process, names) if r]

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    print("\n" + "=" * 84)
    print(f"{len(out)} images processed")
    print(f"candidates total          {out['n_candidates'].sum():>8d}")
    print(f"accepted, production      {out['prod_accepted'].sum():>8d}")
    print(f"accepted, retrained       {out['v3_accepted'].sum():>8d}")
    print(f"  newly accepted           {out['flipped_to_crack'].sum():>8d}")
    print(f"  newly rejected           {out['flipped_to_not'].sum():>8d}")
    print(f"mean crack-area %, prod   {out['prod_area_pct'].mean():>8.3f}")
    print(f"mean crack-area %, v3     {out['v3_area_pct'].mean():>8.3f}")

    rev = out[out["n_reviewed"] > 0]
    both = rev[rev["has_both_classes"]]
    print(f"\nagreement with human corrections, {len(rev)} reviewed images "
          f"({int(rev['n_reviewed'].sum())} regions):")
    w = rev["n_reviewed"]
    print(f"  production  {np.average(rev['prod_agree'], weights=w):.1%}")
    print(f"  retrained   {np.average(rev['v3_agree'], weights=w):.1%}")
    print(f"\nrestricted to the {len(both)} images that have BOTH crack and not-crack marks "
          f"({int(both['n_reviewed'].sum())} regions) -- the only ones where")
    print("agreement can be lowered by a false positive rather than only by a miss:")
    wb = both["n_reviewed"]
    print(f"  production  {np.average(both['prod_agree'], weights=wb):.1%}")
    print(f"  retrained   {np.average(both['v3_agree'], weights=wb):.1%}")
    print(f"\nwrote {OUT_CSV}")
