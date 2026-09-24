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
PAINT = f"{_REPO}/interior_active_learning/paint"
REPO = _REPO


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
    # was analysis/sam3/tiles_meta.json, a byte-identical copy of the file the
    # pipeline actually writes. One source of truth.
    return len(json.load(open("analysis/sam3/tiles/meta.json")))


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


def _clean(prompt="crack"):
    return [r for r in json.load(open("analysis/sam3/sam3_results.json"))
            if "error" not in r and r["prompt"] == prompt]


def _pres():
    return json.load(open("analysis/sam3/sam3_presence.json"))


def d_clean_fired():
    """Tiles where the 'crack' prompt returned at least one instance, LEAK-GATED run."""
    return sum(1 for r in _clean() if r.get("n", 0) > 0)


def d_clean_union_iou():
    """Median union IoU, 'crack' prompt, over all 16 tiles."""
    return round(float(np.median([r.get("union_IoU", 0) for r in _clean()])), 4)


def d_clean_oracle_iou():
    return round(float(np.median([r.get("oracle_IoU", 0) for r in _clean()])), 4)


def d_gate_agreement():
    """INSTRUMENTATION SELF-CHECK, NOT A RESULT.

    (tile, prompt) pairs where 'returned nothing' == 'presence*max_q <= tau'. This is an
    ALGEBRAIC IDENTITY: the wrapper reads the same tensors sam3_image_processor.py:197-199
    multiplies and thresholds with the same constant, and sigmoid(presence) > 0 so
    max_j(s*q_j) = s*max_j q_j. It cannot fail except through an instrumentation bug, which
    is exactly what it is here to catch. Registered so the wrapper stays wired correctly --
    NOT as evidence about the model. An earlier draft published it as a 64-trial result.
    """
    pres = _pres()
    n = 0
    for r in json.load(open("analysis/sam3/sam3_results.json")):
        if "error" in r:
            continue
        p = pres.get(f"{r['tile']}|{r['prompt']}")
        if p is not None and p["survives"] == (r.get("n", 0) > 0):
            n += 1
    return n


def d_presence_span():
    """THE empirical claim: ratio of median presence for the best vs worst of four synonyms.

    Computed from UNROUNDED medians. An earlier version divided the 4-dp rounded values and a
    1.0 tolerance hid the difference; the figure's ".0f" then printed 113 for a value of 112.55.
    """
    pres = json.load(open("analysis/sam3/sam3_presence.json"))
    med = {}
    for q in ("crack", "fracture"):
        med[q] = float(np.median([v["presence"] for k, v in pres.items()
                                  if k.split("|", 1)[1] == q]))
    return round(med["crack"] / med["fracture"], 2)


def d_always_empty_baseline():
    """Information-free baseline for 'did this pair return anything?' -- always say empty."""
    n = sum(1 for r in json.load(open("analysis/sam3/sam3_results.json"))
            if "error" not in r and r.get("n", 0) == 0)
    return n


def d_prompt_identity_baseline():
    """Best rule using ONLY prompt identity. The honest reference point, not 50%."""
    rows = [r for r in json.load(open("analysis/sam3/sam3_results.json")) if "error" not in r]
    prompts = sorted({r["prompt"] for r in rows})
    best = 0
    for mask in range(1 << len(prompts)):
        pred = {p: bool(mask >> i & 1) for i, p in enumerate(prompts)}
        best = max(best, sum(1 for r in rows if pred[r["prompt"]] == (r.get("n", 0) > 0)))
    return best


def d_max_instances():
    """Largest instance count returned. If the scalar flipped all instances this would be 200."""
    return max(r.get("n", 0) for r in json.load(open("analysis/sam3/sam3_results.json"))
               if "error" not in r)


def d_prompt_r2():
    """R^2 of logit(presence) by prompt identity. Chance for a 4-level factor here is 0.038."""
    pres = json.load(open("analysis/sam3/sam3_presence.json"))
    lg, lab = [], []
    for k, x in pres.items():
        s = min(max(x["presence"], 1e-6), 1 - 1e-6)
        lg.append(np.log(s / (1 - s))); lab.append(k.split("|", 1)[1])
    lg = np.array(lg)
    g = {}
    for val, k in zip(lg, lab):
        g.setdefault(k, []).append(val)
    sst = ((lg - lg.mean()) ** 2).sum()
    ssw = sum(((np.array(v) - np.mean(v)) ** 2).sum() for v in g.values())
    return round(1 - ssw / sst, 4)


def d_presence_median(prompt):
    v = [x["presence"] for k, x in _pres().items() if k.split("|", 1)[1] == prompt]
    return round(float(np.median(v)), 4)


def d_pres_crack():
    return d_presence_median("crack")


def d_pres_fracture():
    return d_presence_median("fracture")


def d_leak_paired_p():
    """Wilcoxon p on paired IoU deltas, contaminated vs leak-gated. Recovered from git."""
    import subprocess
    from scipy.stats import wilcoxon
    out = subprocess.run(["git", "show", "3ae43eb:analysis/sam3/sam3_results.json"],
                         capture_output=True, text=True, check=True).stdout
    cm = {(r["tile"], r["prompt"]): r for r in json.loads(out) if "error" not in r}
    d = [r.get("union_IoU", 0) - cm[(r["tile"], r["prompt"])].get("union_IoU", 0)
         for r in _clean() if (r["tile"], r["prompt"]) in cm]
    d = [x for x in d if x != 0]
    return round(float(wilcoxon(d).pvalue), 4)


# ---- the 2026-09-19 benchmark round ---------------------------------------------------------
def _bench():
    return json.load(open("analysis/sam3/methods_bench.json"))


def d_bench_thr_iou():
    """Best IoU arm: a plain global threshold at leave-one-frame-out."""
    return round(_bench()["global threshold|IoU"]["LOFO"], 4)


def d_bench_meij_cldice():
    """Best clDice arm: a Meijering ridge filter. The IoU and clDice leaders differ."""
    return round(_bench()["Meijering ridge|clDice"]["LOFO"], 4)


def d_bench_nested_cldice():
    """Nested LOFO clDice -- method AND parameters chosen on the training frames."""
    return round(_bench()["NESTED LOFO|clDice"]["LOFO"], 4)


def d_selection_premium():
    """clDice gained by picking the method with hindsight instead of nested selection."""
    b = _bench()
    best = max(v["LOFO"] for k, v in b.items()
               if k.endswith("|clDice") and isinstance(v, dict) and "LOFO" in v
               and not k.startswith("NESTED"))
    return round(best - b["NESTED LOFO|clDice"]["LOFO"], 4)


def d_iou_ceiling():
    """Median IoU of a PERFECT 3 px trace down the label centreline.

    The single most consequential number here: pixel IoU on this corpus cannot exceed about
    0.17 for a physically correct crack, so every IoU above it was bought by being thicker
    than a crack.
    """
    return round(json.load(open("analysis/sam3/iou_ceiling.json"))["ceiling_IoU"], 4)


def d_omnicrack_cldice():
    """OmniCrack30k's released nnU-Net, LOFO clDice, on our tiles. Single fold, no mirroring."""
    return round(json.load(open("analysis/sam3/omnicrack_eval.json"))["clDice"]["LOFO"], 4)


def d_omnicrack_iou():
    return round(json.load(open("analysis/sam3/omnicrack_eval.json"))["IoU"]["LOFO"], 4)


def d_serd_cldice():
    """SERD + Sobel, reading SAM 3's field before the presence gate."""
    return round(json.load(open("analysis/sam3/serd_eval.json"))["SERD +Sobel|clDice"]["LOFO"], 4)


def d_corridor_allones():
    """Containment scored by predicting the ENTIRE tile. Shows containment is not gameable."""
    n = json.load(open("analysis/sam3/corridor_scores.json"))["nulls"]
    return round(n["all-ones"][0], 3)


def d_corridor_random_coverage():
    """Coverage reached by an area-matched RANDOM SCATTER -- why coverage is not quotable."""
    n = json.load(open("analysis/sam3/corridor_scores.json"))["nulls"]
    return round(n["area-matched random"][1], 2)


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
    ("sam3 tiles", "disjoint tiles tested", 15, d_sam_tiles, 0),
    ("LEAK_POSTMORTEM", "tiles the invalid input contaminated", 13, d_burnin_tiles, 0),
    ("LEAK_POSTMORTEM", "tiles the current input contaminates", 0, d_leak_clean, 0),
    ("LEAK_POSTMORTEM", "median best-threshold IoU (OIS), clean input", 0.4090, d_leak_oracle_iou, 0.0005),
    ("LEAK_POSTMORTEM", "label frames registered at ncc >= 0.99", 9, d_align_ok, 0),
    ("LEAK_POSTMORTEM", "frames registering at ncc exactly 1.0", 5, d_align_exact, 0),
    ("LEAK_POSTMORTEM", "distinct RGB triples in the burn-in", 1, d_overlay_triples, 0),
    ("LEAK_POSTMORTEM", "green value the burn-in writes", 25, d_overlay_green, 0),
    ("sam3 clean run", "tiles where 'crack' returned instances", 13, d_clean_fired, 0),
    ("sam3 clean run", "median union IoU, 'crack', 15 disjoint tiles", 0.1914, d_clean_union_iou, 0.0005),
    ("sam3 clean run", "median oracle IoU, 'crack', 15 disjoint tiles", 0.3871, d_clean_oracle_iou, 0.0005),
    ("sam3 clean run", "gate self-check (IDENTITY, not a result)", 60, d_gate_agreement, 0),
    ("sam3 clean run", "presence span, best/worst prompt", 105.3, d_presence_span, 0.05),
    ("sam3 clean run", "R2 of logit(presence) by prompt", 0.8148, d_prompt_r2, 0.0005),
    ("sam3 clean run", "always-empty baseline (of 60)", 39, d_always_empty_baseline, 0),
    ("sam3 clean run", "prompt-identity-only baseline (of 60)", 50, d_prompt_identity_baseline, 0),
    ("sam3 clean run", "max instances returned (of 200 queries)", 62, d_max_instances, 0),
    ("sam3 clean run", "median presence scalar, 'crack'", 0.9062, d_pres_crack, 0.0005),
    ("sam3 clean run", "median presence scalar, 'fracture'", 0.0086, d_pres_fracture, 0.0005),
    ("POSITION_VS_2026", "IoU ceiling, perfect 3 px trace", 0.1662, d_iou_ceiling, 0.0005),
    ("POSITION_VS_2026", "best IoU arm (global threshold, LOFO)", 0.2579, d_bench_thr_iou, 0.0005),
    ("POSITION_VS_2026", "best clDice arm (Meijering, LOFO)", 0.3382, d_bench_meij_cldice, 0.0005),
    ("POSITION_VS_2026", "nested-LOFO clDice", 0.1739, d_bench_nested_cldice, 0.0005),
    ("POSITION_VS_2026", "hindsight selection premium, clDice", 0.1643, d_selection_premium, 0.0005),
    ("POSITION_VS_2026", "OmniCrack30k LOFO IoU", 0.2069, d_omnicrack_iou, 0.0005),
    ("POSITION_VS_2026", "OmniCrack30k LOFO clDice", 0.3304, d_omnicrack_cldice, 0.0005),
    ("POSITION_VS_2026", "SERD+Sobel LOFO clDice", 0.3234, d_serd_cldice, 0.0005),
    ("corridor_metric", "containment of an all-ones prediction", 0.055, d_corridor_allones, 0.002),
    ("corridor_metric", "coverage of area-matched random", 0.90, d_corridor_random_coverage, 0.01),
]


# Artefacts that are regenerable and therefore not committed. If one is absent the affected
# claims cannot be checked -- which is NOT the same thing as a claim having drifted, and must
# not be reported as a failure. A fresh clone has none of these until the reproduce sequence in
# analysis/README.md has been run, and it should not look broken for that reason.
REGENERABLE = {
    "masks": "masks/ (run the export)",
    "sets": "sets/ (tools/split_sets.py)",
    "analysis/regions.csv": "analysis/regions.csv (tools/analyse_sets.py)",
    "analysis/sam3/tiles": "analysis/sam3/tiles/ (analysis/sam3/make_tiles.py)",
    "analysis/sam3/masks": "analysis/sam3/masks/ (SAM3_SAVE_MASKS=1 run_real.py)",
}


def missing_artefacts():
    return {p: how for p, how in REGENERABLE.items() if not _os.path.exists(f"{_CE}/{p}")}


def main():
    absent = missing_artefacts()
    if absent:
        print("NOTE: regenerable artefacts are absent, so some claims cannot be checked.")
        print("      This is not drift. Regenerate with:")
        for how in absent.values():
            print(f"        - {how}")
        print()
    print(f"{'doc':<26} {'claim':<46} {'published':>14} {'derived':>14}   status")
    print("-" * 112)
    bad = skipped = 0
    for doc, name, pub, fn, tol in CLAIMS:
        try:
            got = fn()
        except (FileNotFoundError, StopIteration, IndexError) as e:
            print(f"{doc[:26]:<26} {name:<46} {pub:>14} {'--':>14}   SKIP (artefact absent)")
            skipped += 1
            continue
        except Exception as e:
            print(f"{doc[:26]:<26} {name:<46} {pub:>14} {'ERROR':>14}   {type(e).__name__}: {str(e)[:40]}")
            bad += 1
            continue
        # an empty glob silently yields 0 or nan; that is an absent artefact, not a drifted value
        if absent and (got == 0 and pub != 0 or (isinstance(got, float) and got != got)):
            print(f"{doc[:26]:<26} {name:<46} {pub:>14} {'--':>14}   SKIP (artefact absent)")
            skipped += 1
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
    print(f"{len(CLAIMS)} registered claims, {len(CLAIMS)-bad-skipped} pass, {bad} fail"
          + (f", {skipped} skipped (artefact absent)" if skipped else ""))
    if bad:
        print("\nFAIL means a document quotes a number its artefact no longer produces.")

    bad += check_prose()
    return 1 if bad else 0


# ---------------------------------------------------------------------------------------
# THE REGISTRY ABOVE CHECKS ARTEFACTS, NEVER THE PROSE THAT QUOTES THEM. That gap is not
# hypothetical: on 2026-09-21 LEAK_POSTMORTEM.md said the trivial-baseline IoU was 0.384
# while the registry recomputed 0.4090 from leak_check.json and PASSED, because the two
# never meet. Three more numbers had drifted the same way in POSITION_VS_2026.md.
#
# The obvious fix -- scan each document for the registered value as a substring -- was
# tried and thrown away: it reports FOUND for 0.4090 because the unrelated string "0.41"
# appears somewhere in the file, and FOUND for 0 because "0.0" does. A check that passes
# by coincidence is worse than no check.
#
# So each entry here names the EXACT sentence fragment that must appear, verbatim. If the
# underlying number moves, the registry above fails on the artefact AND this fails on the
# sentence, so the document cannot be left quoting the old one. Adding a claim here is
# cheap; do it whenever a number makes it into prose.
PROSE = [
    ("analysis/sam3/LEAK_POSTMORTEM.md",
     "reaches median **IoU 0.4090** over the 15 disjoint tiles",
     "the trivial baseline every method must beat; said 0.384 until 2026-09-21"),
    ("analysis/sam3/POSITION_VS_2026.md",
     "Brush widths across the 47 run from **4 px to 413 px**",
     "said 10-288 px, understating the coarsest brush by 1.43x"),
    ("analysis/sam3/POSITION_VS_2026.md",
     "**55 published numbers are recomputed from source artefacts by `verify_claims.py`,**",
     "the registry's own size, quoted in prose; was 45"),
    ("analysis/sam3/POSITION_VS_2026.md",
     "ours **passed** all 33 of its own checks",
     "said 'failed', which inverts the lesson: passing is what hid the leak"),
    ("analysis/sam3/best_detector.py",
     "keep the top 1% of the response (quantile 99)",
     "the shipped config; the docstring said quantile 98 for four commits"),
    ("analysis/sam3/best_config.json",
     '"quantile": 99',
     "the value the docstring above must agree with"),
    ("../README.md",
     "367 passed, 0 failed, 3 skipped, 370 total",
     "the fresh-clone suite result, re-measured 2026-09-22 by cloning the public remote; "
     "read 356/1/357 before that"),
    ("analysis/VERDICT_2026.md",
     "**Without a\nnormaliser the effect does not reach significance.**",
     "the raw crack/matrix ratio gives 32/49 pairs and p=0.125; the claim is CNR, not darkness"),
    ("analysis/VERDICT_2026.md",
     "**51 / 52** usable pairs, geometric mean **2.74x**",
     "the gain-free, segmenter-free detector effect over all 56 pairs; recomputed 2026-09-23"),
    ("analysis/VERDICT_2026.md",
     "7/7 positive, p = 0.0156",
     "the specimen-level test at 7 cells, which n=3 could not reach (floor 0.25)"),
    ("analysis/VERDICT_2026.md",
     "The 8 confirmed pairs come from 3 specimens, not 8",
     "specimen_key() gives MAR_Amb_AS/Cast/HIP; per-specimen aggregation drops the exact p "
     "from the n=8 floor 0.0078125 to the n=3 floor 0.2500"),
    ("analysis/sam3/CLEAN_RUN_RESULTS.md",
     "p = 0.0312 is not obtainable here",
     "5 discordant pairs put the exact two-sided signed-rank floor at 0.0625; the document "
     "claimed significance the shipped data cannot produce"),
    ("analysis/sam3/CLEAN_RUN_RESULTS.md",
     "F(14,42) = 3.94, p = 2.76e-4",
     "the crossed-design ANOVA on 15 tiles x 4 prompts; (15,45) is the withdrawn 16-tile design"),
    ("analysis/sam3/best_detector.py",
     "deletes 455 components holding 56.9% of predicted pixels",
     "recomputed on AS_24hr_BSE_Side_008: 1217 -> 762 components, 225,086 -> 97,124 px. "
     "The component count was right and the share said 60.5%"),
]


def check_prose():
    """Every sentence in PROSE must still be present, verbatim, in its document."""
    print()
    print(f"{'document':<44} {'the sentence that must still be true':<52} ")
    print("-" * 112)
    missing = 0
    for path, fragment, why in PROSE:
        try:
            text = open(path).read()
        except OSError:
            print(f"{path[:43]:<44} {'':<52} SKIP (file absent)")
            continue
        ok = fragment in text
        missing += 0 if ok else 1
        print(f"{path.split('/')[-1][:43]:<44} {fragment[:50]:<52} "
              f"{'PASS' if ok else '*** FAIL ***'}")
        if not ok:
            print(f"{'':<44} why it matters: {why}")
    print("-" * 112)
    print(f"{len(PROSE)} prose claims, {len(PROSE)-missing} pass, {missing} fail")
    if missing:
        print("\nFAIL here means a document no longer states the number its artefact produces.")
    return missing


sys.exit(main())
