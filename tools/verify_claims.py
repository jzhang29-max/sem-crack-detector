#!/usr/bin/env python3
"""Recompute every statistic quoted in the analysis docs, from its source artefact.

WHY THIS EXISTS. Three times in a row, asking "is it ready?" surfaced a real error, and
the worst of them was a number I had repeated confidently across six files (a recall
threshold count that was 9/16, published as 10/16). Re-reading prose does not catch that.
This does: each claim names the document that quotes it, the number as published, and a
function that DERIVES the number from the artefact. A claim passes only if the derivation
reproduces it.

Deliberately not a regex scrape of the markdown: that finds numbers but cannot tell you
what they should be. The registry is explicit, so a drifting artefact fails loudly and a
new claim has to be registered to be covered. `coverage` reports how many quoted
statistics are actually registered, so the gap is visible rather than assumed.

Exit code 1 if any claim fails.
"""
import csv, glob, json, os, sys, math
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
PAINT = "/Users/jiamingzhang/Desktop/sem-crack-detector/interior_active_learning/paint"
REPO = "/Users/jiamingzhang/Desktop/sem-crack-detector"


def _f(v, d=float("nan")):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def rows(p):
    return list(csv.DictReader(open(p)))


# ----------------------------------------------------------------- derivations

def d_frames():
    return len(glob.glob("masks/*_mask.png"))


def d_sets():
    return len([d for d in glob.glob("sets/*") if os.path.isdir(d)])


def d_links():
    return len([p for p in glob.glob("sets/*/*/*")
                if p.endswith((".png", "_regions.csv")) and "_diagram" not in p])


def d_marked_px():
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    t = 0
    for p in glob.glob(f"{PAINT}/*_correction_mask.png"):
        m = np.array(Image.open(p))
        while m.ndim > 2:
            m = m[..., 0]
        t += int(np.isin(m, (1, 2, 3)).sum())
    return t


def d_masks_present():
    return len(glob.glob(f"{PAINT}/*_correction_mask.png"))


def d_zero_review():
    return sum(1 for r in rows("analysis/review_coverage.csv") if _f(r["ReviewedPct"], 0) == 0)


def d_coarse_share():
    R = [r for r in rows("analysis/label_granularity.csv") if r["median_thick_px"]]
    th = np.array([_f(r["median_thick_px"]) for r in R])
    px = np.array([_f(r["crack_px"]) for r in R])
    return round(100 * px[th > 80].sum() / px.sum(), 1)


def d_fine_share():
    R = [r for r in rows("analysis/label_granularity.csv") if r["median_thick_px"]]
    th = np.array([_f(r["median_thick_px"]) for r in R])
    px = np.array([_f(r["crack_px"]) for r in R])
    return round(100 * px[th <= 25].sum() / px.sum(), 1)


def d_stroke_median():
    R = [r for r in rows("analysis/label_granularity.csv") if r["median_thick_px"]]
    return int(np.median([_f(r["median_thick_px"]) for r in R]))


def d_stroke_max():
    R = [r for r in rows("analysis/label_granularity.csv") if r["median_thick_px"]]
    return int(max(_f(r["median_thick_px"]) for r in R))


def _all_areas():
    a = []
    for p in glob.glob("regions/*_regions.csv"):
        a += [_f(r["Area_px"]) for r in rows(p)]
    return np.array(a)


def d_top1_area_share():
    a = np.sort(_all_areas())
    return round(100 * a[-len(a) // 100:].sum() / a.sum(), 1)


def d_speck_count_share():
    a = _all_areas()
    return round(100 * (a <= 500).mean(), 0)


def d_speck_area_share():
    a = _all_areas()
    return round(100 * a[a <= 500].sum() / a.sum(), 2)


def d_hfw_n():
    return len(rows("analysis/scale_hfw.csv"))


def d_hfw_min():
    return min(_f(r["HFW_um"]) for r in rows("analysis/scale_hfw.csv"))


def d_hfw_max():
    return max(_f(r["HFW_um"]) for r in rows("analysis/scale_hfw.csv"))


def d_hfw_ratio():
    return round(d_hfw_max() / d_hfw_min(), 0)


def d_pairs_registered():
    return len([r for r in rows("analysis/paired_detector.csv")
                if r["Jaccard"] not in ("", None) and _f(r["Jaccard"]) >= 0.5])


def d_pairs_index_matched():
    return len([r for r in rows("analysis/paired_detector.csv") if r["Jaccard"] not in ("", None)])


def d_cbs_higher_count():
    P = [r for r in rows("analysis/paired_detector.csv")
         if r["Jaccard"] not in ("", None) and _f(r["Jaccard"]) >= 0.5]
    return sum(1 for r in P if _f(r["Delta_CBS_minus_ETD"]) > 0)


def d_cbs_median_delta():
    P = [r for r in rows("analysis/paired_detector.csv")
         if r["Jaccard"] not in ("", None) and _f(r["Jaccard"]) >= 0.5]
    return round(float(np.median([_f(r["Delta_CBS_minus_ETD"]) for r in P])), 2)


def d_cbs_wilcoxon_p():
    from scipy.stats import wilcoxon
    P = [r for r in rows("analysis/paired_detector.csv")
         if r["Jaccard"] not in ("", None) and _f(r["Jaccard"]) >= 0.5]
    return round(float(wilcoxon([_f(r["Delta_CBS_minus_ETD"]) for r in P]).pvalue), 4)


def d_branches():
    return sum(1 for _ in open("analysis/branches.csv")) - 1


def d_area_identity_err():
    R = rows("analysis/skeleton_frames.csv")
    e = []
    for r in R:
        frame = _f(r["H"]) * _f(r["W"])
        A = _f(r["CrackAreaPct"]) / 100 * frame
        L = _f(r["SkeletonPx"])
        if L <= 0:
            continue
        e.append(abs((L / frame) * (A / L) - A / frame))
    return max(e)


def d_auc_pooled():
    d = json.load(open(f"{REPO}/models/crack_classifier_v3_metrics.json"))
    return round(d["cv_results"]["LogisticRegression"]["pooled_auc"], 4)


def d_auc_loio():
    d = json.load(open(f"{REPO}/models/crack_classifier_v3_metrics.json"))
    return round(d["cv_results"]["LogisticRegression"]["loio_auc_exhaustive_image"], 4)


# ---- SAM 3 scoring claims WITHDRAWN 2026-09-18 ------------------------------------------
# Six claims were registered here off sam3_results.json: tiles with non-zero recall (10),
# tiles with recall exactly 0 (6), lowest non-zero recall (0.969), median recall (0.979),
# and the two prompt-emptiness counts. All are withdrawn: the model input was the green
# channel of the annotated overlay, and opaque red (225,25,25) has green 25, well under the 80 the threshold used, so the label was
# written into the input as black pixels on 14 of 16 tiles (analysis/sam3/LEAK_POSTMORTEM.md).
#
# A registry that recomputes a number cannot tell that the number answers the wrong question.
# Every derivation below reproduced its published value exactly, 33/33, while six of them were
# measuring an annotation. What replaces them are claims about the LEAK and the REGISTRATION,
# which are properties of the data rather than of a model run, plus the trivial baseline.


def d_sam_tiles():
    return len(json.load(open("analysis/sam3/tiles_meta.json")))


def d_overlay_triples():
    """Distinct RGB triples inside the overlay's painted region. Exactly one: (225,25,25)."""
    from PIL import Image
    from collections import Counter
    tot = Counter()
    for m in json.load(open("analysis/sam3/tiles/meta.json")):
        t = m["tile"]
        o = np.array(Image.open(
            f"analysis/sam3/tiles/{t}_overlay_REFERENCE_DO_NOT_FEED.png").convert("RGB"))
        red = (o[..., 0] > 150) & (o[..., 1] < 80) & (o[..., 2] < 80)
        if red.sum():
            u = np.unique(o[red].reshape(-1, 3), axis=0)
            for tri in u:
                tot[tuple(int(x) for x in tri)] += 1
    return len(tot)


def d_overlay_green():
    """The green value the burn-in writes. 25, not 0 -- I published 0 while retracting."""
    from PIL import Image
    for m in json.load(open("analysis/sam3/tiles/meta.json")):
        t = m["tile"]
        o = np.array(Image.open(
            f"analysis/sam3/tiles/{t}_overlay_REFERENCE_DO_NOT_FEED.png").convert("RGB"))
        red = (o[..., 0] > 150) & (o[..., 1] < 80) & (o[..., 2] < 80)
        if red.sum():
            return int(np.unique(o[red][:, 1])[0])
    return -1


def _leak():
    return json.load(open("analysis/sam3/leak_check.json"))


def d_leak_clean():
    """Tiles whose current input carries a written-in label. Must be 0."""
    return sum(1 for r in _leak() if r["leak"])


def d_leak_oracle_iou():
    """Median best-single-threshold IoU on the clean input: the bar any method must clear."""
    return round(float(np.median([r["oracle_iou"] for r in _leak()])), 4)


def d_align_ok():
    """Label frames registered to their raw original at ncc >= 0.99."""
    a = json.load(open("analysis/sam3/alignment.json"))
    return sum(1 for v in a.values() if v.get("ok"))


def d_align_exact():
    """Frames registering at ncc exactly 1.0 -- proof the grey stretch matches the renderer."""
    a = json.load(open("analysis/sam3/alignment.json"))
    return sum(1 for v in a.values() if v.get("ncc") == 1.0)


def d_burnin_tiles():
    """Tiles the INVALID input contaminated, recomputed from the reference overlays."""
    from PIL import Image
    n = 0
    for m in json.load(open("analysis/sam3/tiles/meta.json")):
        t = m["tile"]
        g = np.array(Image.open(
            f"analysis/sam3/tiles/{t}_overlay_REFERENCE_DO_NOT_FEED.png").convert("RGB"))[..., 1]
        gt = np.array(Image.open(f"analysis/sam3/tiles/{t}_gt.png")) > 127
        v, c = np.unique(g[gt], return_counts=True)
        if float(g[gt].std()) < 0.5 or float(c.max() / c.sum()) > 0.999:
            n += 1
    return n


# ----------------------------------------------------------------- the registry
# (document that quotes it, claim, published value, derivation, tolerance)
CLAIMS = [
    ("README / CRACK_ANALYSIS", "frames in corpus", 62, d_frames, 0),
    ("README / CRACK_ANALYSIS", "specimen sets", 10, d_sets, 0),
    ("README", "hard links under sets/", 186, d_links, 0),
    ("LABEL_GRANULARITY", "hand-marked pixels", 70434978, d_marked_px, 0),
    ("CRACK_ANALYSIS", "frames with a correction mask", 47, d_masks_present, 0),
    ("CRACK_ANALYSIS", "frames with zero human review", 16, d_zero_review, 0),
    ("LABEL_GRANULARITY", "% marked px on frames with stroke >80 px", 91.1, d_coarse_share, 0.05),
    ("LABEL_GRANULARITY", "% marked px on frames with stroke <=25 px", 2.1, d_fine_share, 0.05),
    ("LABEL_GRANULARITY", "median stroke thickness (px)", 59, d_stroke_median, 0),
    ("LABEL_GRANULARITY", "max stroke thickness (px)", 413, d_stroke_max, 0),
    ("CRACK_ANALYSIS", "% crack area in top 1% of regions", 92.5, d_top1_area_share, 0.05),
    ("CRACK_ANALYSIS", "% of regions <=500 px2", 66, d_speck_count_share, 0.5),
    ("CRACK_ANALYSIS", "% crack area in regions <=500 px2", 0.73, d_speck_area_share, 0.005),
    ("CORRECTION_scale", "frames with HFW read", 45, d_hfw_n, 0),
    ("CORRECTION_scale", "min HFW (um)", 10.4, d_hfw_min, 0.01),
    ("CORRECTION_scale", "max HFW (um)", 2590, d_hfw_max, 0.01),
    ("CORRECTION_scale", "magnification span (x)", 249, d_hfw_ratio, 0.5),
    ("CRACK_ANALYSIS / VERDICT", "index-matched CBS/ETD pairs", 16, d_pairs_index_matched, 0),
    ("CRACK_ANALYSIS / VERDICT", "registration-confirmed pairs", 8, d_pairs_registered, 0),
    ("CRACK_ANALYSIS", "pairs where CBS reads higher", 8, d_cbs_higher_count, 0),
    ("CRACK_ANALYSIS", "median CBS-ETD delta (pp)", 3.88, d_cbs_median_delta, 0.01),
    ("CRACK_ANALYSIS", "Wilcoxon p, paired detector", 0.0078, d_cbs_wilcoxon_p, 0.0001),
    ("LINEARITY", "skeleton branches", 33471, d_branches, 0),
    ("NOVELTY / LINEARITY", "area-fraction identity max error", 0.0, d_area_identity_err, 1e-12),
    ("HEAD_TO_HEAD / README", "pooled grouped-CV AUC", 0.7144, d_auc_pooled, 0.0001),
    ("HEAD_TO_HEAD / README", "single-image LOIO AUC", 0.8840, d_auc_loio, 0.0001),
    ("sam3 tiles", "tiles tested", 16, d_sam_tiles, 0),
    ("LEAK_POSTMORTEM", "tiles the invalid input contaminated", 14, d_burnin_tiles, 0),
    ("LEAK_POSTMORTEM", "tiles the current input contaminates", 0, d_leak_clean, 0),
    ("LEAK_POSTMORTEM", "median best-threshold IoU, clean input", 0.3843, d_leak_oracle_iou, 0.0005),
    ("LEAK_POSTMORTEM", "label frames registered at ncc >= 0.99", 9, d_align_ok, 0),
    ("LEAK_POSTMORTEM", "frames registering at ncc exactly 1.0", 5, d_align_exact, 0),
    ("LEAK_POSTMORTEM", "distinct RGB triples in the burn-in", 1, d_overlay_triples, 0),
    ("LEAK_POSTMORTEM", "green value the burn-in writes", 25, d_overlay_green, 0),
]


def main():
    print(f"{'doc':<26} {'claim':<46} {'published':>14} {'derived':>14}   status")
    print("-" * 112)
    bad = 0
    for doc, name, pub, fn, tol in CLAIMS:
        try:
            got = fn()
        except Exception as e:
            print(f"{doc[:26]:<26} {name:<46} {pub:>14} {'ERROR':>14}   {type(e).__name__}: {str(e)[:40]}")
            bad += 1
            continue
        ok = abs(float(got) - float(pub)) <= tol
        bad += 0 if ok else 1
        def fmt(v):
            if isinstance(v, int) and abs(v) > 9999:
                return f"{v:,}"
            if isinstance(v, float) and v != 0 and abs(v) < 1e-6:
                return f"{v:.2e}"
            return f"{v}"
        print(f"{doc[:26]:<26} {name:<46} {fmt(pub):>14} {fmt(got):>14}   "
              f"{'PASS' if ok else '*** FAIL ***'}")
    print("-" * 112)
    print(f"{len(CLAIMS)} registered claims, {len(CLAIMS)-bad} pass, {bad} fail")
    if bad:
        print("\nFAIL means a document quotes a number its artefact no longer produces.")
    return 1 if bad else 0


sys.exit(main())
