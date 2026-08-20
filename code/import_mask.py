"""Import a crack mask from another tool, from the command line.

    python3 code/import_mask.py IMAGE MASK.png --from ilastik
    python3 code/import_mask.py IMAGE --clear
    python3 code/import_mask.py --list
    python3 code/import_mask.py --shape IMAGE      # what size must the mask be?

WHY YOU WOULD DO THIS
The built-in detector is the weakest part of this project. A survey of ilastik 1.4.2,
micro-sam 1.8.9, Fiji's Trainable Weka/Labkit, and the commercial CNNs put all of them
ahead of a darkness threshold plus a LogisticRegression over 8 morphology features -- the
deployed operating point misses roughly 40% of crack pixels.

Everything this project does that those tools do not is DOWNSTREAM of the mask: a
calibration that refuses when the scale bar and HFW disagree, reporting pixels and saying so
when a group is uncalibrated, unreviewed pixels never scored as negatives, a promotion gate
that refuses without a valid baseline, and a provenance record bound to every CSV. So segment
wherever you segment best and bring the mask here for measurement and audit.

Authority order after importing: human correction > imported mask > built-in detector. Your
brush strokes still win, and importing never touches a correction mask.

EXPORTING A COMPATIBLE MASK
The mask must be a 2D image the same size as this project's CROPPED frame -- run --shape to
get it, because the crop removes any burned-in info bar. Non-zero means crack, so a binary
PNG, a 0/1 mask, or a labelled instance map all work. Masks are never resampled: a resampled
foreign segmentation is subtly wrong everywhere and nothing downstream would reveal it.

  ilastik    Pixel Classification -> Prediction Export -> Simple Segmentation, as TIFF/PNG
  micro-sam  napari layer -> save the labels layer
  Fiji       the binary result of TWS/Labkit -> File > Save As > PNG
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "interior_active_learning", "code"))
sys.path.insert(0, HERE)

import external_mask
from common import ORIGINAL_DIR


def _frame_shape(image_name):
    """The cropped frame's shape -- what the mask must match."""
    from unified_pipeline import run_unified_pipeline
    return run_unified_pipeline(image_name)["labeled"].shape


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", help="image name, without .tif")
    ap.add_argument("mask", nargs="?", help="mask file to import")
    ap.add_argument("--from", dest="source_tool", default="custom",
                    help=f"which tool produced it, e.g. {', '.join(external_mask.KNOWN_SOURCES)}")
    ap.add_argument("--note", default="")
    ap.add_argument("--clear", action="store_true",
                    help="drop the import and fall back to the built-in detector")
    ap.add_argument("--list", action="store_true", help="show every imported mask")
    ap.add_argument("--shape", action="store_true",
                    help="print the frame shape the mask must match, then exit")
    a = ap.parse_args()

    if a.list:
        names = sorted(f[:-len("_external_mask.json")]
                       for f in os.listdir(external_mask.PAINT_DIR)
                       if f.endswith("_external_mask.json"))
        if not names:
            print("no imported masks; every image uses the built-in detector")
            return 0
        for n in names:
            r = external_mask.provenance_for(n).get("mask_import", {})
            print(f"  {n:36s} {r.get('source_tool','?'):12s} "
                  f"{r.get('nonzero_px',0):>10,} px  {r.get('imported_at','')}")
        return 0

    if not a.image:
        ap.print_help()
        return 1
    if not os.path.exists(os.path.join(ORIGINAL_DIR, f"{a.image}.tif")):
        print(f"no such image: {a.image}")
        return 1

    if a.shape:
        h, w = _frame_shape(a.image)
        print(f"{a.image}: export your mask at {h} x {w} (height x width). This is the "
              f"CROPPED frame, which excludes any burned-in info bar.")
        return 0

    if a.clear:
        print("cleared; back to the built-in detector" if external_mask.clear(a.image)
              else "nothing was imported for this image")
        return 0

    if not a.mask:
        ap.print_help()
        return 1
    if not os.path.exists(a.mask):
        print(f"no such mask file: {a.mask}")
        return 1

    shape = _frame_shape(a.image)
    try:
        rec = external_mask.store(a.image, a.mask, a.source_tool, shape, note=a.note)
    except ValueError as e:
        print(f"REFUSED: {e}")
        return 1
    print(f"imported {rec['nonzero_px']:,} crack px ({100 * rec['nonzero_fraction']:.2f}% "
          f"of the frame) from {rec['source_tool']}")
    print(f"  regions: {rec['n_regions']}"
          f"{' (labelled instances preserved)' if rec['labelled_instances'] else ''}")
    print(f"  sha256:  {rec['source_sha256'][:16]}")
    print(f"  This replaces the built-in detector for {a.image}. Your corrections still "
          f"override it, and every exported CSV will name the source.")
    print(f"  Re-render the overlay in the app, or run crack_measurements.py {a.image}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
