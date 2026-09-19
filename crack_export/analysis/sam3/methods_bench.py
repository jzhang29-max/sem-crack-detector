#!/usr/bin/env python3
"""Method comparison on the leak-gated SEM crack tiles, at three tuning budgets.

WHY THIS FILE. The earlier write-up compared an ORACLE-tuned per-tile threshold against SAM 3
un-tuned at a fixed tau and called the difference a result. It was not: at matched budgets the
two are indistinguishable. So every method here is reported at all three budgets, and the one
that matters for a claim of usefulness is the third.

  OIS   per-tile oracle          parameters chosen per tile USING that tile's label. Upper
                                 bound only. This is the protocol of Arbelaez et al., TPAMI
                                 33(5):898-916 2011, and it is not an achievable accuracy.
  ODS   one parameter set, fitted on ALL tiles including the test tile. Still optimistic.
  LOFO  leave-one-FRAME-out: parameters fitted on the other 8 frames, applied to the held-out
        frame's tiles. Frame level, not tile level, because 15 tiles come from 9 frames and
        two tiles of one frame share a specimen, an instrument setting and an operator.

  NESTED LOFO  the number to actually quote. Reporting "the best method's LOFO score" still
        selects the METHOD using every frame, including the held-out one -- a second-order
        leak that survives a correct first-order LOFO. Here the method AND its parameters are
        both chosen on the 8 training frames, so the held-out frame contributes nothing to any
        choice. This is what a deployed pipeline with no prior knowledge of the new frame gets.

Uncertainty is a frame-clustered bootstrap (resample the 9 frames, not the 15 tiles).

Cracks are DARK curvilinear structures, so all ridge filters run with black_ridges=True.
"""
import os, json, itertools
import numpy as np
from PIL import Image
from skimage.filters import frangi, sato, meijering, threshold_sauvola, apply_hysteresis_threshold
from skimage.morphology import skeletonize, remove_small_objects

SC = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(20260919)


def load():
    meta = json.load(open(f"{SC}/tiles/meta.json"))
    out = []
    for m in meta:
        g = np.array(Image.open(f"{SC}/tiles/{m['tile']}_gray.png").convert("L"))
        gt = np.array(Image.open(f"{SC}/tiles/{m['tile']}_gt.png")) > 127
        out.append((m["tile"], m["frame"], g, gt))
    return out


def iou(p, gt):
    u = np.logical_or(p, gt).sum()
    return float(np.logical_and(p, gt).sum() / u) if u else 0.0


def cldice(p, gt):
    if p.sum() == 0 or gt.sum() == 0:
        return 0.0
    if p.mean() > 0.25:          # a "crack" covering a quarter of the tile is not a crack
        return 0.0
    sp, sg = skeletonize(p), skeletonize(gt)
    tp = (sp & gt).sum() / sp.sum() if sp.sum() else 0.0
    ts = (sg & p).sum() / sg.sum() if sg.sum() else 0.0
    return float(2 * tp * ts / (tp + ts)) if (tp + ts) > 0 else 0.0


# ----------------------------------------------------------------- response maps
def resp_raw(g):
    return {(): 255 - g.astype(np.float32)}          # dark = high response


def resp_sauvola(g):
    out = {}
    for w in (25, 51, 101, 201):
        out[(w,)] = (threshold_sauvola(g, window_size=w, k=0.2) - g).astype(np.float32)
    return out


def _ridge(fn, g, **kw):
    out = {}
    for smax in (2, 3, 4, 6):
        sig = np.arange(1, smax + 1, 1.0)
        r = fn(g.astype(np.float32) / 255.0, sigmas=sig, black_ridges=True, **kw)
        r = r.astype(np.float32)
        m = r.max()
        out[(smax,)] = r / m if m > 0 else r
    return out


def resp_frangi(g):
    return _ridge(frangi, g)


def resp_sato(g):
    return _ridge(sato, g)


def resp_meijering(g):
    return _ridge(meijering, g)


METHODS = {
    "global threshold": resp_raw,
    "Sauvola local": resp_sauvola,
    "Frangi ridge": resp_frangi,
    "Sato ridge": resp_sato,
    "Meijering ridge": resp_meijering,
}
# binarisation parameters swept on top of every response map
# Quantiles below 90 predict >10% of a tile as crack. The densest hand label in this corpus is
# 10.16% of a tile, so those settings are physically impossible for a crack and they dominate
# runtime, because skeletonize on a near-half-filled mask is orders of magnitude slower than on
# a thin one. Restricting the search to <=10% foreground is a modelling decision, stated here,
# not a convenience: it can only hurt a method that wanted to predict a third of the frame.
QS = [90, 92, 94, 95, 96, 97, 98, 98.5, 99, 99.3, 99.5, 99.7, 99.9]
MINSZ = [0, 32, 128]


def binarise(r, q, minsz, hyst):
    hi = np.percentile(r, q)
    if hyst:
        lo = np.percentile(r, max(q - 4.0, 0.0))
        p = apply_hysteresis_threshold(r, lo, hi)
    else:
        p = r >= hi
    if minsz:
        # skimage 0.26 renamed min_size -> max_size AND changed it to remove objects
        # smaller than OR EQUAL TO the value. Verified both spellings behave
        # identically in this version; using the non-deprecated one explicitly.
        p = remove_small_objects(p, max_size=minsz)
    return p


def build(data):
    """score[(method, params)][tile] = (iou, cldice)"""
    score, params = {}, {}
    for mname, fn in METHODS.items():
        print(f"  {mname} ...", flush=True)
        maps = {t: fn(g) for t, _, g, _ in data}
        keys = list(next(iter(maps.values())).keys())
        for rk, q, ms, hy in itertools.product(keys, QS, MINSZ, (False, True)):
            key = (mname, rk + (q, ms, hy))
            score[key] = {}
            for t, _, _, gt in data:
                p = binarise(maps[t][rk], q, ms, hy)
                score[key][t] = (iou(p, gt), cldice(p, gt))
        params[mname] = keys
    return score


def main():
    data = load()
    frames = sorted({f for _, f, _, _ in data})
    tiles = [t for t, _, _, _ in data]
    tframe = {t: f for t, f, _, _ in data}
    print(f"{len(tiles)} tiles, {len(frames)} frames\n")
    score = build(data)
    mnames = list(METHODS)

    def best_over(keys, subset, idx=0):
        """parameter set maximising the median metric over `subset` tiles"""
        return max(keys, key=lambda k: np.median([score[k][t][idx] for t in subset]))

    # ---- nested LOFO: choose method AND parameters on the training frames only
    nested = {}
    for idx, metric in ((0, "IoU"), (1, "clDice")):
        vals, picks = [], []
        for t in tiles:
            tr = [u for u in tiles if tframe[u] != tframe[t]]
            best_m, best_k, best_v = None, None, -1
            for m in mnames:
                k = best_over([q for q in score if q[0] == m], tr, idx)
                v = np.median([score[k][u][idx] for u in tr])
                if v > best_v:
                    best_m, best_k, best_v = m, k, v
            vals.append(score[best_k][t][idx]); picks.append(best_m)
        nested[metric] = (vals, picks)

    rows = []
    for m in mnames:
        keys = [k for k in score if k[0] == m]
        for idx, metric in ((0, "IoU"), (1, "clDice")):
            ois = [max(score[k][t][idx] for k in keys) for t in tiles]
            ods_k = best_over(keys, tiles, idx)
            ods = [score[ods_k][t][idx] for t in tiles]
            lofo = []
            for t in tiles:
                tr = [u for u in tiles if tframe[u] != tframe[t]]
                lofo.append(score[best_over(keys, tr, idx)][t][idx])
            rows.append((m, metric, ois, ods, lofo))

    # SAM 3 for reference
    res = {(r["tile"], r["prompt"]): r for r in json.load(open(f"{SC}/sam3_results.json"))
           if "error" not in r}
    sam_u = [res.get((t, "crack"), {}).get("union_IoU", 0.0) for t in tiles]
    sam_c = [res.get((t, "crack"), {}).get("union_clDice", 0.0) for t in tiles]
    sam_o = [res.get((t, "crack"), {}).get("oracle_IoU", 0.0) for t in tiles]

    def boot(v):
        """frame-clustered bootstrap CI on the median"""
        out = []
        for _ in range(4000):
            pick = rng.choice(frames, len(frames), replace=True)
            sub = [v[tiles.index(t)] for f in pick for t in tiles if tframe[t] == f]
            if sub:
                out.append(np.median(sub))
        return np.percentile(out, 2.5), np.percentile(out, 97.5)

    print(f"\n{'method':<20} {'metric':<7} {'OIS':>7} {'ODS':>7} {'LOFO':>7}   {'LOFO 95% CI (frame-clustered)':>30}")
    print("-" * 92)
    out = {}
    for m, metric, ois, ods, lofo in rows:
        lo, hi = boot(lofo)
        print(f"{m:<20} {metric:<7} {np.median(ois):>7.4f} {np.median(ods):>7.4f} "
              f"{np.median(lofo):>7.4f}   [{lo:.4f}, {hi:.4f}]")
        out[f"{m}|{metric}"] = {"OIS": float(np.median(ois)), "ODS": float(np.median(ods)),
                                "LOFO": float(np.median(lofo)), "LOFO_ci": [float(lo), float(hi)],
                                "per_tile_LOFO": [float(x) for x in lofo]}
    for metric, (vals, picks) in nested.items():
        lo_, hi_ = boot(vals)
        from collections import Counter
        c = Counter(picks).most_common()
        print(f"{'NESTED LOFO (best)':<20} {metric:<7} {'--':>7} {'--':>7} {np.median(vals):>7.4f}   "
              f"[{lo_:.4f}, {hi_:.4f}]   picked: {', '.join(f'{k}x{v}' for k, v in c)}")
        out[f"NESTED LOFO|{metric}"] = {"LOFO": float(np.median(vals)),
                                        "LOFO_ci": [float(lo_), float(hi_)],
                                        "per_tile_LOFO": [float(x) for x in vals],
                                        "method_picked": picks}
    lo, hi = boot(sam_u)
    print(f"{'SAM 3 union tau=.3':<20} {'IoU':<7} {'--':>7} {'--':>7} {np.median(sam_u):>7.4f}   [{lo:.4f}, {hi:.4f}]")
    print(f"{'SAM 3 union tau=.3':<20} {'clDice':<7} {'--':>7} {'--':>7} {np.median(sam_c):>7.4f}")
    print(f"{'SAM 3 oracle inst.':<20} {'IoU':<7} {np.median(sam_o):>7.4f} {'--':>7} {'--':>7}")
    out["SAM 3 union|IoU"] = {"LOFO": float(np.median(sam_u)), "LOFO_ci": [float(lo), float(hi)],
                              "per_tile_LOFO": [float(x) for x in sam_u]}
    out["SAM 3 union|clDice"] = {"LOFO": float(np.median(sam_c))}
    out["SAM 3 oracle|IoU"] = {"OIS": float(np.median(sam_o))}
    out["_tiles"] = tiles
    out["_frames"] = {t: tframe[t] for t in tiles}
    json.dump(out, open(f"{SC}/methods_bench.json", "w"), indent=1)
    print(f"\nwrote methods_bench.json")


if __name__ == "__main__":
    main()
