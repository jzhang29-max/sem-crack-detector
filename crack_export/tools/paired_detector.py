#!/usr/bin/env python3
"""CBS vs ETD on the SAME field of view, and field-level (not frame-level) set sizes.

Established by measurement: MAR_Amb_<proc>_CBS_000N and _ETD_000N are the same
field imaged with two detectors. Index-matched mask pairs have median Jaccard
0.505; mismatched indices from the same specimen give 0.024 (n=6 control). So
counting 11 AS "frames" as 11 samples double-counts 5 fields.

Consequence for every number in this analysis: the effective sample size in the
superalloy family is FIELDS (18), not frames (34).
"""
import csv, os, numpy as np, sys
from collections import defaultdict
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
# --- portable roots -------------------------------------------------------------
# Resolved from this file's own location so the analysis runs from a fresh clone.
# crack_export used to be a separate repo beside sem-crack-detector, and every script
# hard-coded /Users/jiamingzhang/Desktop/... Now that it lives inside the repo, those
# literals would have made a clone unrunnable for anyone but this laptop.
import os as _os
_CE = _os.path.dirname(_os.path.abspath(__file__))
while _os.path.basename(_CE) != "crack_export" and _os.path.dirname(_CE) != _CE:
    _CE = _os.path.dirname(_CE)
_REPO = _os.path.dirname(_CE)
# --------------------------------------------------------------------------------
sys.path.insert(0, f"{_REPO}/interior_active_learning/code")
from aggregate import parse_name

ROOT = _CE
os.chdir(ROOT)
F = {m["SourceImage"]: m for m in csv.DictReader(open("analysis/per_frame_metrics_with_review.csv"))}

def load(n):
    a = np.array(Image.open(f"masks/{n}_mask.png").resize((768, 512), Image.NEAREST))
    while a.ndim > 2: a = a[..., 0]
    return a < 128

rows = []
for proc in ("AS", "Cast", "HIP"):
    for i in range(1, 12):
        c, e = f"MAR_Amb_{proc}_CBS_{i:04d}", f"MAR_Amb_{proc}_ETD_{i:04d}"
        hc, he = os.path.exists(f"masks/{c}_mask.png"), os.path.exists(f"masks/{e}_mask.png")
        if not (hc or he): continue
        r = {"Process": proc, "Field": f"{i:04d}", "HasCBS": hc, "HasETD": he,
             "CBS_AreaPct": float(F[c]["CrackAreaPct"]) if hc else "",
             "ETD_AreaPct": float(F[e]["CrackAreaPct"]) if he else "", "Jaccard": ""}
        if hc and he:
            A, B = load(c), load(e)
            u = (A | B).sum()
            r["Jaccard"] = round(float((A & B).sum() / u), 4) if u else ""
            r["Delta_CBS_minus_ETD"] = round(r["CBS_AreaPct"] - r["ETD_AreaPct"], 3)
        rows.append(r)

with open("analysis/paired_detector.csv", "w", newline="") as f:
    cols = ["Process", "Field", "HasCBS", "HasETD", "CBS_AreaPct", "ETD_AreaPct",
            "Jaccard", "Delta_CBS_minus_ETD"]
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader(); w.writerows(rows)

good = [r for r in rows if r.get("Jaccard") not in ("", None) and r["Jaccard"] >= 0.5]
d = np.array([r["Delta_CBS_minus_ETD"] for r in good])
print("=== CBS vs ETD, PAIRED on the same field (Jaccard >= 0.5 => registration confirmed) ===")
print(f"{'field':<14}{'CBS%':>8}{'ETD%':>8}{'CBS-ETD':>10}{'Jaccard':>9}")
for r in sorted(good, key=lambda r: (r["Process"], r["Field"])):
    print(f"{r['Process']+'_'+r['Field']:<14}{r['CBS_AreaPct']:>8.2f}{r['ETD_AreaPct']:>8.2f}"
          f"{r['Delta_CBS_minus_ETD']:>10.2f}{r['Jaccard']:>9.3f}")
print(f"\nn={len(d)} confirmed-registration pairs")
print(f"median CBS-ETD = {np.median(d):+.2f} pp;  CBS higher in {int((d>0).sum())}/{len(d)} pairs")
from scipy.stats import wilcoxon, binomtest
if len(d) >= 5:
    try:
        st = wilcoxon(d)
        print(f"Wilcoxon signed-rank on paired differences: W={st.statistic:.1f}, p={st.pvalue:.4f}")
    except Exception as ex:
        print("wilcoxon:", ex)
    bt = binomtest(int((d > 0).sum()), len(d), 0.5)
    print(f"sign test: p={bt.pvalue:.4f}")
print("Interpretation: a positive median means CBS flags MORE crack area than ETD on\n"
      "the identical field, i.e. part of every between-set difference is detector gain,\n"
      "not material. It is small next to the process spread, but it is not zero.")

amb = [r for r in rows if r.get("Jaccard") not in ("", None) and r["Jaccard"] < 0.5]
print(f"\n{len(amb)} index-matched pairs did NOT register (Jaccard < 0.5): "
      f"{', '.join(r['Process']+'_'+r['Field'] for r in amb)}")
print("  Either a renumbered field or one detector missing most of the damage. Excluded\n"
      "  from the paired test rather than averaged in.")

print("\n=== effective sample size: FIELDS, not frames ===")
print(f"{'set':<16}{'frames':>8}{'fields':>8}{'field-level median area%':>26}")
for proc in ("AS", "Cast", "HIP"):
    rr = [r for r in rows if r["Process"] == proc]
    nf = sum(1 for r in rr for k in ("HasCBS", "HasETD") if r[k])
    vals = []
    for r in rr:                     # one value per field: mean of detectors present
        v = [r[k] for k in ("CBS_AreaPct", "ETD_AreaPct") if r[k] != ""]
        vals.append(float(np.mean(v)))
    print(f"{'MAR_Amb_'+proc:<16}{nf:>8}{len(rr):>8}{np.median(vals):>26.2f}")
print("\nAll three are ONE specimen each. Frames and fields are replicates WITHIN a\n"
      "specimen; neither is a replicate OF the processing route.")
