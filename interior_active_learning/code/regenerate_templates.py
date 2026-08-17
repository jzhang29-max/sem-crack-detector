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


def process(image_name):
    try:
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
    if "--only" in sys.argv:
        names = [sys.argv[sys.argv.index("--only") + 1]]
    else:
        names = sorted(os.path.splitext(f)[0] for f in os.listdir(ORIGINAL_DIR)
                        if f.lower().endswith(".tif"))
    print(f"regenerating {len(names)} template(s) via run_unified_pipeline\n")

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
