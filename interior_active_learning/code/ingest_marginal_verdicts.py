"""
Turn spoken verdicts on the marginal-calls sheet into real training labels.

    python3 ingest_marginal_verdicts.py --crack 1,3,5,6 --not-crack rest
    python3 ingest_marginal_verdicts.py --crack 1,3 --not-crack 2,4,5 --dry-run

Verdicts are written into the per-pixel correction masks -- the same
authoritative record the paint tool writes (1 = crack, 2 = not-crack, 3 =
erased, 0 = unreviewed) -- rather than into a side-channel CSV. That matters
for three reasons: run_unified_pipeline applies these masks after scoring, so
the verdicts immediately override the model everywhere; build_training_data.py
reads them, so they flow into the next retrain automatically; and the paint
tool will show them, so nothing is invisible to the person reviewing.

"rest" is accepted for either flag and means every index on the sheet not
named by the other flag, so the common case ("these four are cracks, the
others aren't") is one short command.

The region geometry is recovered by re-running the same pipeline and selecting
the recorded Label. That is deliberate rather than storing masks in the
manifest: the Label is only meaningful together with the segmentation that
produced it, so it is re-derived from that segmentation, and the script
verifies the region's area still matches the manifest before writing anything.
A mismatch means the pipeline changed underneath the sheet, and it aborts
rather than labelling the wrong pixels.
"""
import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "code"))

import numpy as np
import pandas as pd

from common import (ORIGINAL_DIR, PAINT_DIR, PROJECT_ROOT, contrast_kwargs_for,
                     load_correction_mask, save_correction_mask)
from detect_cracks import (load_as_uint8, find_field_of_view, flatten_background,
                            segment_dark_regions, clean_mask, compute_vesselness,
                            extract_candidates)

MANIFEST = os.path.join(PROJECT_ROOT, "figures", "marginal_calls_0.4_0.45_manifest.json")
LOG = os.path.join(PROJECT_ROOT, "interior_active_learning", "labels",
                    "original_paint_corrections.csv")


def parse_set(spec, all_idx, other):
    if not spec:
        return set()
    if spec.strip().lower() == "rest":
        return set(all_idx) - other
    out = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crack", default="")
    ap.add_argument("--not-crack", dest="notcrack", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    man = json.load(open(MANIFEST))
    regions = {r["index"]: r for r in man["regions"]}
    all_idx = sorted(regions)

    # resolve "rest" on whichever side uses it, using the other side as the complement
    if a.crack.strip().lower() == "rest":
        nc = parse_set(a.notcrack, all_idx, set())
        cr = set(all_idx) - nc
    else:
        cr = parse_set(a.crack, all_idx, set())
        nc = parse_set(a.notcrack, all_idx, cr)

    overlap = cr & nc
    if overlap:
        print(f"ABORT: indices in both lists: {sorted(overlap)}"); sys.exit(1)
    unknown = (cr | nc) - set(all_idx)
    if unknown:
        print(f"ABORT: no such index on the sheet: {sorted(unknown)}"); sys.exit(1)
    if not (cr or nc):
        print("nothing to do -- pass --crack and/or --not-crack"); sys.exit(1)

    print(f"crack     ({len(cr)}): {sorted(cr)}")
    print(f"not-crack ({len(nc)}): {sorted(nc)}")
    unlabelled = set(all_idx) - cr - nc
    if unlabelled:
        print(f"left unreviewed ({len(unlabelled)}): {sorted(unlabelled)}")
    print()

    by_image = {}
    for i in sorted(cr | nc):
        by_image.setdefault(regions[i]["SourceImage"], []).append(i)

    log_rows = []
    for name, idxs in by_image.items():
        img8 = load_as_uint8(os.path.join(ORIGINAL_DIR, f"{name}.tif"),
                             **contrast_kwargs_for(name))
        x0, y0, x1, y1 = find_field_of_view(img8)
        img8 = img8[y0:y1, x0:x1]
        flat = flatten_background(img8)
        clean = clean_mask(segment_dark_regions(flat, img8=img8), min_area_px=13)
        ves = compute_vesselness(flat)
        # Not called, matching unified_pipeline and build_training_data. Region Label
        # IDs are assigned in scan order over the surviving regions, so a segmenter that
        # deletes regions here produces IDs that mean something different from the ones
        # the app shows -- and this script's output is keyed by ID.
        labeled, df = extract_candidates(clean, flat, ves, min_area_px=40)

        existing = load_correction_mask(name, labeled.shape)
        mask = existing.copy() if existing is not None else np.zeros(labeled.shape, np.uint8)

        for i in idxs:
            r = regions[i]
            m = labeled == r["Label"]
            got = int(m.sum())
            if got != r["area"]:
                print(f"ABORT: #{i} {name} label {r['Label']} area {got} != manifest "
                      f"{r['area']} -- segmentation changed since the sheet was made.")
                sys.exit(1)
            val = 1 if i in cr else 2
            mask[m] = val
            log_rows.append({"SourceImage": name, "Label": r["Label"],
                             "CorrectedTo": bool(val == 1)})
            print(f"  #{i:>3d} {name:28s} label {r['Label']:>5d} {got:>6d}px -> "
                  f"{'CRACK' if val == 1 else 'not-crack'}")

        if not a.dry_run:
            save_correction_mask(name, mask)
    if a.dry_run:
        print("\ndry run -- nothing written")
        return

    if log_rows:
        new = pd.DataFrame(log_rows)
        if os.path.exists(LOG):
            old = pd.read_csv(LOG)
            both = pd.concat([old, new], ignore_index=True)
            both = both.drop_duplicates(subset=["SourceImage", "Label"], keep="last")
        else:
            both = new
        both.to_csv(LOG, index=False)
        print(f"\ncorrection masks updated for {len(by_image)} image(s); "
              f"label log now {len(both)} rows")
    print("\nnext: python3 build_training_data.py && python3 train_v3_weighted.py")


if __name__ == "__main__":
    main()
