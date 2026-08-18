"""
Apply a trained crack classifier to every image matching a glob pattern.

Example (from this folder, running on the whole parent-folder series):
    python3 batch_apply.py "../../*.tif" --model crack_clf.joblib --out-dir ../batch_results

Manual corrections (--corrections-ledger)
------------------------------------------
Sample-weighting a correction during training only *encourages* the model to
agree -- for a genuinely borderline candidate (e.g. probability 0.57) it can
still lose to the broader pattern learned from everything else. Passing a
ledger CSV (columns: SourceImage, Label, CorrectedTo) forces those exact
rows to the value a human set, no matter what the model's probability says.
SourceImage must match each image's filename without extension or "_cracks"
suffix (e.g. "260708_316_H_b2_front_CBS_003").
"""

import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect_cracks import detect_cracks

EXCLUDE_SUBSTRINGS = ("Probabilities", "_cracks_bw", "_cracks_overlay", "_review_page")


def load_overrides_by_image(ledger_path):
    ledger = pd.read_csv(ledger_path)
    ledger.columns = [c.strip() for c in ledger.columns]
    overrides = {}
    for name, g in ledger.groupby("SourceImage"):
        is_crack = g["CorrectedTo"].astype(str).str.strip().str.lower().isin(["true", "1"])
        overrides[name] = dict(zip(g["Label"].astype(int), is_crack))
    return overrides


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pattern", help='Glob pattern for input images, e.g. "../../*.tif" (quote it!).')
    parser.add_argument("--model", required=True, help="Trained classifier from train_classifier.py.")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: alongside each image).")
    parser.add_argument("--proba-threshold", type=float, default=0.5)
    parser.add_argument("--corrections-ledger", default=None,
                         help="CSV with SourceImage,Label,CorrectedTo columns -- forces these exact rows "
                              "regardless of the model's own prediction. See module docstring.")
    parser.add_argument("--review-sheet", action="store_true")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.pattern))
    paths = [p for p in paths if not any(s in p for s in EXCLUDE_SUBSTRINGS)]
    if not paths:
        print(f"No images matched pattern: {args.pattern}")
        sys.exit(1)

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    overrides_by_image = load_overrides_by_image(args.corrections_ledger) if args.corrections_ledger else {}
    if overrides_by_image:
        total = sum(len(v) for v in overrides_by_image.values())
        print(f"Loaded {total} manual corrections across {len(overrides_by_image)} image(s) from ledger.")

    print(f"Found {len(paths)} image(s).")
    summary = []
    for p in paths:
        name_no_ext = os.path.splitext(os.path.basename(p))[0]
        print(f"\n--- {os.path.basename(p)} ---")
        try:
            result = detect_cracks(
                p, out_dir=args.out_dir, mode="apply",
                model_path=args.model, proba_threshold=args.proba_threshold,
                make_review_sheets=args.review_sheet,
                hard_overrides=overrides_by_image.get(name_no_ext),
            )
            summary.append((os.path.basename(p), result["count"]))
        except Exception as e:
            print(f"FAILED: {e}")
            summary.append((os.path.basename(p), None))

    print("\n=== Summary ===")
    for name, count in summary:
        print(f"  {name}: {count if count is not None else 'FAILED'}")


if __name__ == "__main__":
    main()
