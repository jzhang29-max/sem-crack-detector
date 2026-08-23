"""
Regenerate every paint template by running the ACTUAL pipeline the app runs.

This replaces seed_templates_with_steel_model.py, which reimplemented only
Pass 1 (original darkness-threshold candidates scored by the production
classifier) and wrote that as the template. The app, however, renders
run_unified_pipeline() output -- Pass 1 PLUS Pass 2's accepted interior /
concavity / bridge candidates -- and rewrites the template from it on the
first click or ingest. On one real image that meant a template showing 282
crack regions was replaced by 501 the moment anything was clicked, which is
what looked like "regions randomly changing colour".

Any template generator that is not literally the app's own pipeline will
drift from it again, so this calls run_unified_pipeline + build_simple_overlay
-- the same two calls paint_server.py makes. Template and app output are then
identical by construction rather than by careful maintenance.

Hand corrections are preserved automatically: run_unified_pipeline applies
the per-pixel correction mask itself, after scoring, so a reviewed verdict
still overrides any model proposal.

    python3 regenerate_templates.py [--only IMAGE_NAME]
"""
import json
import os
import sys
import warnings
from multiprocessing import Pool

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "code"))

from common import PAINT_DIR, ORIGINAL_DIR
from interior_candidates import build_simple_overlay
from unified_pipeline import run_unified_pipeline

COUNTS_PATH = os.path.join(PAINT_DIR, "candidate_counts.json")


USE_SAM = "--with-sam" in sys.argv


def process(image_name):
    try:
        if USE_SAM:
            # Route through the SAME helper the interactive path uses, so a
            # batch re-render and a drag-drop produce identical overlays. They
            # did not before: only /api/process ran SAM, so Re-apply and the
            # post-retrain re-render silently discarded every SAM region and
            # quietly downgraded the detector from f1 0.776 to 0.715.
            from hybrid_detect import render_and_record
            r = render_and_record(image_name, use_sam=True)
            n_total, n_crack = r["n_candidates"], r["n_crack"]
            n_interior = r["n_interior"]
            print(f"OK   {image_name:30s} {n_total:5d} cand | {n_crack:5d} crack "
                  f"({n_crack - n_interior - r['n_sam_regions']} pass1 + {n_interior} interior "
                  f"+ {r['n_sam_regions']} SAM)", flush=True)
            return (image_name, n_total, n_crack, n_interior, None)
        stage = run_unified_pipeline(image_name)
        df = stage["df"]
        build_simple_overlay(stage).save(
            os.path.join(PAINT_DIR, f"{image_name}_paint_template.png"))
        n_total = len(df)
        n_crack = int(df["IsCrack"].sum())
        n_interior = len(stage.get("interior_origin", {}))
        print(f"OK   {image_name:30s} {n_total:5d} cand | {n_crack:5d} crack "
              f"({n_crack - n_interior} pass1 + {n_interior} interior)", flush=True)
        return (image_name, n_total, n_crack, n_interior, None)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"FAIL {image_name:30s} {e}", flush=True)
        return (image_name, 0, 0, 0, str(e))


if __name__ == "__main__":
    # --help USED TO RENDER THE WHOLE CORPUS. There was no argument handling here at all, so
    # `--help` fell through to the default branch and started re-rendering every image --
    # roughly 40 s each, unattended, with no way to tell it was not printing help. Anything
    # unrecognised now refuses instead of doing an hour of work.
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__ or "")
        print("usage: regenerate_templates.py [--only IMAGE_NAME] [--with-sam]\n"
              "  no arguments  re-render every image in original/ (~40 s each)\n"
              "  --only NAME   re-render one image\n"
              "  --with-sam    include the SAM stage (~3 min each)")
        sys.exit(0)
    _KNOWN = {"--only", "--with-sam"}
    _bad = [a for a in sys.argv[1:] if a.startswith("-") and a not in _KNOWN]
    if _bad:
        print(f"unknown option(s) {_bad}. Use --help.")
        sys.exit(2)
    if "--only" in sys.argv:
        names = [sys.argv[sys.argv.index("--only") + 1]]
    else:
        names = sorted(os.path.splitext(f)[0] for f in os.listdir(ORIGINAL_DIR)
                        if f.lower().endswith(".tif"))
    print(f"regenerating {len(names)} template(s) "
          f"{'WITH SAM (~3 min each)' if USE_SAM else 'pipeline only (~40s each)'}\n")

    if USE_SAM:
        # One process: SAM holds a ViT-Huge on the GPU, and three workers
        # would load three copies and contend for the same device.
        results = [process(n) for n in names]
    else:
        with Pool(3) as pool:
            results = pool.map(process, names)

    counts = {}
    if os.path.exists(COUNTS_PATH):
        try:
            with open(COUNTS_PATH) as f:
                counts = json.load(f)
        except Exception:
            counts = {}
    ok = [r for r in results if r[4] is None]
    for name, n_total, n_crack, _n_int, _e in ok:
        counts[name] = {"n_candidates": n_total, "n_crack": n_crack}
    # drop entries for images that no longer exist
    counts = {k: v for k, v in counts.items()
              if os.path.exists(os.path.join(ORIGINAL_DIR, f"{k}.tif"))}
    with open(COUNTS_PATH, "w") as f:
        json.dump(counts, f, indent=2)

    print(f"\n{len(ok)}/{len(names)} regenerated; counts written for {len(counts)} images")
    if ok:
        tot_i = sum(r[3] for r in ok)
        tot_c = sum(r[2] for r in ok)
        print(f"total crack regions {tot_c}, of which {tot_i} come from Pass-2 interior "
              f"candidates ({tot_i / max(tot_c, 1):.0%})")
    for r in results:
        if r[4]:
            print(f"  FAILED {r[0]}: {r[4]}")
    print("DONE")
