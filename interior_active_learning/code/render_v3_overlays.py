"""
Render every image with the RETRAINED model so its output can actually be
looked at, side by side with the current production model.

Until now the retrained model (models/crack_classifier_v3_weighted.joblib,
LogisticRegression + per-image weights, threshold 0.578) had only ever been
measured -- apply_v3_all_images.py produced a statistics CSV and no pictures.
Everything visual in this project so far, paint templates included, was scored
by the old production model.

Both models are run over the IDENTICAL candidate set per image -- same
segmentation, same features, only the classifier differs -- so every colour
difference is attributable to the model and nothing else:

    red    both models call it crack
    green  ONLY the retrained model accepts it   (what v3 adds)
    blue   ONLY production accepts it            (what v3 drops)

SCOPE, stated plainly: this is Pass 1 only. The paint app's templates also
include Pass-2 interior/concavity/bridge regions, which are 31% of all crack
regions, and Pass 2 costs ~117 min per image as currently written (it scores
each candidate against full-image arrays), so it is out of reach for a 63-image
sweep. These overlays therefore show what the new model does to the primary
darkness-threshold candidates, not the app's complete output.

    python3 render_v3_overlays.py
Writes ../../figures/model_compare/<image>_v3_vs_prod.png + a stats CSV
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
from PIL import Image

from common import ORIGINAL_DIR, PROJECT_ROOT, PROD_MODEL_PATH, contrast_kwargs_for
from detect_cracks import (load_as_uint8, find_field_of_view, flatten_background,
                            segment_dark_regions, clean_mask, compute_vesselness,
                            exclude_border_background, extract_candidates)

Image.MAX_IMAGE_PIXELS = None
V3_PATH = os.path.join(PROJECT_ROOT, "models", "crack_classifier_v3_weighted.joblib")
OUT_DIR = os.path.join(PROJECT_ROOT, "figures", "model_compare")
STATS = os.path.join(OUT_DIR, "v3_vs_prod_render_stats.csv")
FORCE_KEEP_AREA = 50000
VIEW_MAX = 2048


def process(name):
    try:
        img8 = load_as_uint8(os.path.join(ORIGINAL_DIR, f"{name}.tif"),
                             **contrast_kwargs_for(name))
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
        big = df["Area"].values >= FORCE_KEEP_AREA
        p_prod = prod["clf"].predict_proba(
            prod["scaler"].transform(df[prod["feature_names"]].values))[:, 1]
        p_v3 = v3["clf"].predict_proba(
            v3["scaler"].transform(df[v3["feature_names"]].values))[:, 1]
        acc_prod = (p_prod >= 0.5) | big
        acc_v3 = (p_v3 >= v3["threshold"]) | big

        lp = df["Label"].values[acc_prod]
        lv = df["Label"].values[acc_v3]
        m_prod = np.isin(labeled, lp)
        m_v3 = np.isin(labeled, lv)

        rgb = np.stack([img8] * 3, -1).astype(np.uint8)
        rgb[m_prod & m_v3] = (220, 30, 30)      # agreement
        rgb[m_v3 & ~m_prod] = (40, 220, 80)     # v3 adds
        rgb[m_prod & ~m_v3] = (60, 130, 255)    # v3 drops
        im = Image.fromarray(rgb)
        if max(im.size) > VIEW_MAX:
            s = VIEW_MAX / max(im.size)
            im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
        os.makedirs(OUT_DIR, exist_ok=True)
        im.save(os.path.join(OUT_DIR, f"{name}_v3_vs_prod.png"))

        tot = img8.size
        rec = {"SourceImage": name, "n_candidates": len(df),
               "prod_accepted": int(acc_prod.sum()), "v3_accepted": int(acc_v3.sum()),
               "added_by_v3": int((acc_v3 & ~acc_prod).sum()),
               "dropped_by_v3": int((acc_prod & ~acc_v3).sum()),
               "prod_area_pct": 100.0 * m_prod.sum() / tot,
               "v3_area_pct": 100.0 * m_v3.sum() / tot,
               "added_px": int((m_v3 & ~m_prod).sum()),
               "dropped_px": int((m_prod & ~m_v3).sum())}
        print(f"{name:32s} cand {len(df):>5d} | accept {rec['prod_accepted']:>5d} -> "
              f"{rec['v3_accepted']:>5d} (+{rec['added_by_v3']} / -{rec['dropped_by_v3']}) "
              f"| area {rec['prod_area_pct']:5.2f}% -> {rec['v3_area_pct']:5.2f}%", flush=True)
        return rec
    except Exception as e:
        print(f"{name:32s} FAILED {type(e).__name__}: {e}", flush=True)
        return None


if __name__ == "__main__":
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(ORIGINAL_DIR)
                    if f.lower().endswith(".tif"))
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"rendering {len(names)} images: production vs retrained (Pass 1)\n"
          f"  red = both, green = only retrained, blue = only production\n", flush=True)
    with Pool(3) as pool:
        rows = [r for r in pool.map(process, names) if r]
    d = pd.DataFrame(rows)
    d.to_csv(STATS, index=False)
    print("\n" + "=" * 80)
    print(f"{len(d)} images rendered")
    print(f"candidates            {int(d.n_candidates.sum()):>10,d}")
    print(f"accepted, production  {int(d.prod_accepted.sum()):>10,d}")
    print(f"accepted, retrained   {int(d.v3_accepted.sum()):>10,d}")
    print(f"  added by retrained  {int(d.added_by_v3.sum()):>10,d} regions "
          f"({int(d.added_px.sum()):,} px)")
    print(f"  dropped by retrained{int(d.dropped_by_v3.sum()):>10,d} regions "
          f"({int(d.dropped_px.sum()):,} px)")
    print(f"mean crack area%  production {d.prod_area_pct.mean():.3f} -> "
          f"retrained {d.v3_area_pct.mean():.3f}")
    print(f"\noverlays in {OUT_DIR}")
