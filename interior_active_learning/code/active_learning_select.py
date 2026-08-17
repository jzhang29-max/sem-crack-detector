"""
Pick the next batch of candidates for the user to manually label, then
render review contact sheets (reusing the production save_review_sheets
layout: thumbnail crops, most-ambiguous-first, bordered by current guess).

Two candidate pools, combined into one batch:

  - "original" region candidates from the production pipeline (results/,
    with each candidate's current CrackProbability) that haven't already
    been reviewed (per manual_corrections_ledger.csv or a prior round of
    this tool) -- ranked by |CrackProbability - 0.5| (uncertainty
    sampling): the ones the CURRENT model is least sure about.

  - "interior" candidates (concavity / bridge_corridor / interior_fill,
    from interior_candidates.py) that haven't been labeled yet. Before an
    interior model exists (round 1, "cold start"), there's no probability
    to rank by, so this samples across candidate types and area deciles
    for coverage. Once models/interior_model.joblib exists, later rounds
    rank these the same way -- by predicted-probability uncertainty.

Output: labels/round_<N>/<image>_review_page*.png contact sheets, plus
labels/round_<N>_template.csv for the user to fill in a UserVerdict column
(TRUE/FALSE/SKIP) and hand back to ingest_labels.py.
"""
import os
import sys
import glob
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    ORIGINAL_DIR, PROD_RESULTS_DIR, LEDGER_PATH, CANDIDATES_DIR, LABELS_DIR, MODELS_DIR,
    REVIEW_DIR, contrast_kwargs_for, ORIGINAL_PAINT_CORRECTIONS_PATH,
)
from detect_cracks import load_as_uint8, find_field_of_view, save_review_sheets

INTERIOR_FEATURE_COLUMNS = [
    "LogArea", "Elongation", "Solidity", "Eccentricity", "Extent", "Circularity",
    "MeanRawBrightness", "MeanFlatBrightness", "MeanVesselness",
    "FracBoundaryTouchingCrack", "MeanDistToCrack",
]


def already_reviewed_original_pairs():
    """(SourceImage, Label) pairs already covered by manual_corrections_ledger,
    a previous round of this tool's own labeling, or a paint-app erasure
    (labels/original_paint_corrections.csv) -- any of these already gave a
    verdict, so none should be re-offered for review."""
    pairs = set()
    if os.path.exists(LEDGER_PATH):
        ledger = pd.read_csv(LEDGER_PATH)
        pairs |= set(zip(ledger["SourceImage"], ledger["Label"].astype(int)))
    for f in glob.glob(os.path.join(LABELS_DIR, "round_*_filled.csv")):
        d = pd.read_csv(f)
        d = d[d["CandidateType"] == "original"]
        pairs |= set(zip(d["SourceImage"], d["Label"].astype(int)))
    if os.path.exists(ORIGINAL_PAINT_CORRECTIONS_PATH):
        d = pd.read_csv(ORIGINAL_PAINT_CORRECTIONS_PATH)
        pairs |= set(zip(d["SourceImage"], d["Label"].astype(int)))
    return pairs


def already_labeled_interior_pairs():
    """(SourceImage, Label) pairs already given a verdict in a previous
    round. Every image's interior candidates number from the same fixed
    offset (see INTERIOR_LABEL_OFFSET), so plain Label values collide
    constantly across different images -- matching on Label alone would
    mean reviewing one image's candidate #1002 silently blocks a totally
    different image's unrelated candidate #1002 from ever being offered
    for review again. Always pair with SourceImage, same as
    already_reviewed_original_pairs() already correctly does."""
    pairs = set()
    for f in glob.glob(os.path.join(LABELS_DIR, "round_*_filled.csv")):
        d = pd.read_csv(f)
        d = d[d["CandidateType"] != "original"]
        pairs |= set(zip(d["SourceImage"], d["Label"].astype(int)))
    return pairs


def load_original_pool(max_per_image=None):
    rows = []
    reviewed = already_reviewed_original_pairs()
    for f in sorted(glob.glob(os.path.join(PROD_RESULTS_DIR, "*_cracks.csv"))):
        name = os.path.basename(f).replace("_cracks.csv", "")
        d = pd.read_csv(f)
        if len(d) == 0:
            continue
        d = d.copy()
        d["SourceImage"] = name
        d["CandidateType"] = "original"
        d = d[~d.apply(lambda r: (r["SourceImage"], int(r["Label"])) in reviewed, axis=1)]
        d["Uncertainty"] = (d["CrackProbability"] - 0.5).abs()
        if max_per_image:
            d = d.sort_values("Uncertainty").head(max_per_image)
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_interior_pool():
    rows = []
    reviewed_pairs = already_labeled_interior_pairs()
    for f in sorted(glob.glob(os.path.join(CANDIDATES_DIR, "*_interior.csv"))):
        name = os.path.basename(f).replace("_interior.csv", "")
        d = pd.read_csv(f)
        if len(d) == 0:
            continue
        # user_painted candidates get IsCrack written directly by
        # apply_paint_annotations.py, bypassing round_*_filled.csv entirely
        # -- excluding only via reviewed_pairs (round history) would let
        # them resurface for review even though painting them in already
        # WAS the verdict. Filtering on "IsCrack already set" catches those
        # (and is a harmless no-op for already-round-labeled rows, which
        # reviewed_pairs already excludes).
        already_has_verdict = d["IsCrack"].astype(str).str.strip().str.upper().isin(["TRUE", "FALSE"])
        already_reviewed = d["Label"].astype(int).apply(lambda lbl: (name, lbl) in reviewed_pairs)
        d = d[~already_reviewed & ~already_has_verdict]
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def score_interior_pool(pool):
    model_path = os.path.join(MODELS_DIR, "interior_model.joblib")
    if len(pool) == 0:
        return pool
    if os.path.exists(model_path):
        bundle = joblib.load(model_path)
        X = bundle["scaler"].transform(pool[INTERIOR_FEATURE_COLUMNS].values)
        proba = bundle["clf"].predict_proba(X)[:, 1]
        pool = pool.copy()
        pool["CrackProbability"] = proba
        pool["Uncertainty"] = np.abs(proba - 0.5)
    else:
        # cold start: no model yet -- stratify by type and area so the
        # first round covers a diverse cross-section rather than whatever
        # happened to be generated first.
        pool = pool.copy()
        pool["Uncertainty"] = 0.0  # ties -> keep insertion order within stratified sample
    return pool


def select_batch(n_original=80, n_interior=80, seed=0):
    orig = load_original_pool()
    if len(orig):
        orig = orig.sort_values("Uncertainty").head(n_original)

    pool = load_interior_pool()
    pool = score_interior_pool(pool)
    if len(pool):
        if os.path.exists(os.path.join(MODELS_DIR, "interior_model.joblib")):
            interior = pool.sort_values("Uncertainty").head(n_interior)
        else:
            # Plain loop instead of groupby(...).apply(...): pandas
            # groupby-apply on raw Series keys folds those values into the
            # result's index rather than keeping them as columns (verified:
            # silently dropped the CandidateType column entirely), and the
            # include_groups=False workaround needs the keys rejoined
            # afterward anyway -- simpler to just not use it here.
            pool = pool.copy()
            pool["AreaBin"] = pd.qcut(pool["Area"], q=min(5, pool["Area"].nunique()), duplicates="drop")
            per_group = max(1, n_interior // 15)
            parts = [
                g.sample(min(len(g), per_group), random_state=seed)
                for _, g in pool.groupby(["CandidateType", "AreaBin"], observed=True)
            ]
            interior = pd.concat(parts, ignore_index=True) if parts else pool.iloc[0:0]
            if len(interior) > n_interior:
                interior = interior.sample(n_interior, random_state=seed)
            interior = interior.drop(columns=["AreaBin"], errors="ignore")
    else:
        interior = pool

    return orig, interior


def next_round_number():
    existing = glob.glob(os.path.join(LABELS_DIR, "round_*_template.csv"))
    nums = [int(os.path.basename(f).split("_")[1]) for f in existing]
    return (max(nums) + 1) if nums else 1


def render_round(orig, interior, round_num):
    out_dir = os.path.join(REVIEW_DIR, f"round_{round_num}")
    os.makedirs(out_dir, exist_ok=True)

    combined = pd.concat([orig, interior], ignore_index=True) if len(orig) or len(interior) else pd.DataFrame()
    if len(combined) == 0:
        print("Nothing left to review -- both pools are empty (or fully labeled).")
        return

    template_rows = []
    for name, g in combined.groupby("SourceImage"):
        img_path = os.path.join(ORIGINAL_DIR, f"{name}.tif")
        img8 = load_as_uint8(img_path, **contrast_kwargs_for(name))
        x0, y0, x1, y1 = find_field_of_view(img8)
        img8 = img8[y0:y1, x0:x1]

        review_df = g.copy()
        review_df["IsCrack"] = review_df["CandidateType"].eq("original") & review_df.get("IsCrack", False).astype(bool)
        paths = save_review_sheets(img8, review_df, out_dir, name, thumb_px=140)
        print(f"{name}: {len(g)} candidates ({(g['CandidateType']=='original').sum()} original, "
              f"{(g['CandidateType']!='original').sum()} interior) -> {len(paths)} page(s)")

        for _, row in g.iterrows():
            template_rows.append({
                "Label": int(row["Label"]),
                "SourceImage": name,
                "CandidateType": row["CandidateType"],
                "CurrentGuess": bool(row.get("IsCrack", False)) if row["CandidateType"] == "original" else "",
                "CrackProbability": round(float(row["CrackProbability"]), 3),
                "UserVerdict": "",  # fill in TRUE / FALSE / SKIP
            })

    template = pd.DataFrame(template_rows)
    template_path = os.path.join(LABELS_DIR, f"round_{round_num}_template.csv")
    template.to_csv(template_path, index=False)
    print(f"\nWrote {len(template)}-row labeling template: {template_path}")
    print(f"Review sheets: {out_dir}/")
    print("\nNext step: open the review sheets, fill in UserVerdict (TRUE/FALSE/SKIP) in the "
          "template CSV using each thumbnail's 'L<Label>' caption to find its row, then run:")
    print(f"  python3 ingest_labels.py {round_num}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-original", type=int, default=80)
    ap.add_argument("--n-interior", type=int, default=80)
    args = ap.parse_args()
    orig, interior = select_batch(args.n_original, args.n_interior)
    round_num = next_round_number()
    render_round(orig, interior, round_num)
