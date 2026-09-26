"""Machine-only masks for every frame in the export, not just the corrected ones.

The 47 frames carrying human corrections obviously need this. The other 95 are regenerated
too, deliberately: their exported mask is read back out of the paint TEMPLATE rather than
recomputed, so it can be stale relative to the current model, and a folder that mixes
"freshly recomputed" with "whatever was rendered last time" is not one thing. Re-running all
142 through the same suppressed-correction path makes every file in the folder the same kind
of object, which is the only way the comparison means anything.
"""
import os, sys, json, glob
import numpy as np
from multiprocessing import Pool
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# Paths derive from this file, and outputs go under a directory this repo owns. An earlier
# draft of this tool hardcoded /Users/jiamingzhang/... and /tmp, which is the same defect
# three repos here were repaired for on 2026-09-24: an absolute path is an undeclared
# dependency on one machine's layout, and /tmp is swept. Override with SEMCRACK_DERIVED.
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(_HERE))
DERIVED = os.environ.get("SEMCRACK_DERIVED", os.path.join(REPO, "crack_export", "derived"))
os.makedirs(DERIVED, exist_ok=True)
sys.path.insert(0, f"{REPO}/interior_active_learning/code")
sys.path.insert(0, f"{REPO}/code")
OUT = os.path.join(DERIVED, "machine_masks")
os.makedirs(OUT, exist_ok=True)


def one(name):
    dst = os.path.join(OUT, f"{name}_machine.png")
    if os.path.exists(dst):
        return {"frame": name, "cached": True}
    try:
        import unified_pipeline as up
        import common
        up.load_correction_mask = lambda *a, **k: None
        common.load_correction_mask = lambda *a, **k: None
        up.load_hard_overrides = lambda *a, **k: {}
        st = up.run_unified_pipeline(name)
        labeled, df = st["labeled"], st["df"]
        keep = set(df.loc[df["IsCrack"] == True, "Label"].tolist())      # noqa: E712
        m = np.isin(labeled, list(keep)) if keep else np.zeros(labeled.shape, bool)
        Image.fromarray(np.where(m, 0, 255).astype(np.uint8), "L").save(dst, optimize=True)
        return {"frame": name, "machine_pct": 100 * float(m.mean()),
                "h": int(m.shape[0]), "w": int(m.shape[1])}
    except Exception as e:
        return {"frame": name, "error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    E = os.path.expanduser("~/Desktop/_Outputs/MAR_crack_identifications/masks_bw")
    names = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(f"{E}/*.png"))
    # seed from the run already done
    for p in glob.glob(os.path.join(DERIVED, "machine_masks_partial", "*_machine.png")):
        d = os.path.join(OUT, os.path.basename(p))
        if not os.path.exists(d):
            os.link(p, d)
    todo = [n for n in names if not os.path.exists(os.path.join(OUT, f"{n}_machine.png"))]
    print(f"  {len(names)} frames, {len(names)-len(todo)} already done, {len(todo)} to run", flush=True)
    res = []
    with Pool(5) as pool:
        for i, r in enumerate(pool.imap_unordered(one, todo), 1):
            res.append(r)
            tag = r.get("error") or f"{r.get('machine_pct', 0):.2f}%"
            print(f"  [{i}/{len(todo)}] {r['frame']:<34} {tag}", flush=True)
    json.dump(res, open(os.path.join(DERIVED, "machine_masks.json"), "w"))
    bad = [r for r in res if "error" in r]
    print(f"\n  errors: {len(bad)}")
    for r in bad[:5]:
        print(f"    {r['frame']}: {r['error']}")
    print(f"  masks present: {len(glob.glob(OUT+'/*_machine.png'))}")
