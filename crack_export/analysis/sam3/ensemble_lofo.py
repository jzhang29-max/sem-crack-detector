#!/usr/bin/env python3
"""Do the methods combine? Union / intersection / majority vote, all under nested LOFO.

Motivation, measured rather than assumed: under clDice all five candidate methods win on at
least one of the 15 tiles (global threshold 3, Sato 4, Frangi 3, Meijering 1, Sauvola 4), and
under IoU four of five do. When no single method dominates, combining them is the obvious next
move -- and it is also the obvious way to fool yourself, so the combination rule and every
parameter are chosen on the TRAINING frames only, exactly like the nested-LOFO arm.

If an ensemble does not beat the best single method under nested LOFO, that is worth reporting
too: it means the methods fail on the same tiles rather than on different ones, which is a
statement about the data, not about the methods.
"""
import os, json, itertools
import numpy as np
from PIL import Image
from skimage.filters import frangi, sato, meijering, threshold_sauvola, apply_hysteresis_threshold
from skimage.morphology import skeletonize, remove_small_objects
from scipy import ndimage as ndi

SC = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(20260919)
QS = [90, 94, 96, 97, 98, 99, 99.5]
SMAX = {"frangi": 4, "sato": 4, "meijering": 4}


def iou(p, gt):
    u = np.logical_or(p, gt).sum()
    return float(np.logical_and(p, gt).sum() / u) if u else 0.0


def cldice(p, gt):
    if p.sum() == 0 or gt.sum() == 0 or p.mean() > 0.25:
        return 0.0
    sp, sg = skeletonize(p), skeletonize(gt)
    tp = (sp & gt).sum() / sp.sum() if sp.sum() else 0.0
    ts = (sg & p).sum() / sg.sum() if sg.sum() else 0.0
    return float(2 * tp * ts / (tp + ts)) if (tp + ts) > 0 else 0.0


def responses(g):
    f = g.astype(np.float32) / 255.0
    out = {"threshold": (255 - g).astype(np.float32),
           "sauvola": (threshold_sauvola(g, window_size=101, k=0.2) - g).astype(np.float32)}
    for nm, fn in (("frangi", frangi), ("sato", sato), ("meijering", meijering)):
        r = fn(f, sigmas=np.arange(1, SMAX[nm] + 1, 1.0), black_ridges=True).astype(np.float32)
        m = r.max()
        out[nm] = r / m if m > 0 else r
    return out


def main():
    meta = json.load(open(f"{SC}/tiles/meta.json"))
    data = []
    print("computing responses ...", flush=True)
    for m in meta:
        g = np.array(Image.open(f"{SC}/tiles/{m['tile']}_gray.png").convert("L"))
        gt = np.array(Image.open(f"{SC}/tiles/{m['tile']}_gt.png")) > 127
        data.append({"tile": m["tile"], "frame": m["frame"], "R": responses(g), "gt": gt})
    names = list(data[0]["R"])
    frames = sorted({d["frame"] for d in data})
    print(f"{len(data)} tiles, {len(frames)} frames, {len(names)} responses\n")

    # candidate configs: (subset of methods, quantile, rule)
    subsets = [s for r in (1, 2, 3) for s in itertools.combinations(names, r)]
    RULES = ("union", "inter", "majority")
    cache = {}

    def predict(d, sub, q, rule):
        key = (d["tile"], sub, q, rule)
        if key in cache:
            return cache[key]
        ms = [d["R"][n] >= np.percentile(d["R"][n], q) for n in sub]
        if rule == "union":
            p = np.logical_or.reduce(ms)
        elif rule == "inter":
            p = np.logical_and.reduce(ms)
        else:
            p = (np.sum(ms, 0) * 2 > len(ms))
        p = remove_small_objects(p, max_size=32)
        cache[key] = p
        return p

    configs = [(s, q, r) for s in subsets for q in QS for r in RULES
               if not (len(s) == 1 and r != "union")]
    print(f"{len(configs)} ensemble configs\n")

    sc = {}
    for i, cfg in enumerate(configs):
        if i % 200 == 0:
            print(f"  {i}/{len(configs)}", flush=True)
        sc[cfg] = {d["tile"]: (iou(predict(d, *cfg), d["gt"]),
                               cldice(predict(d, *cfg), d["gt"])) for d in data}

    tiles = [d["tile"] for d in data]
    tf = {d["tile"]: d["frame"] for d in data}
    out = {}
    for idx, metric in ((0, "IoU"), (1, "clDice")):
        vals, picks = [], []
        for t in tiles:
            tr = [u for u in tiles if tf[u] != tf[t]]
            best = max(configs, key=lambda c: np.median([sc[c][u][idx] for u in tr]))
            vals.append(sc[best][t][idx]); picks.append(f"{'+'.join(best[0])}|q{best[1]}|{best[2]}")
        ois = [max(sc[c][t][idx] for c in configs) for t in tiles]
        bs = []
        for _ in range(4000):
            pk = rng.choice(frames, len(frames), replace=True)
            sub = [vals[tiles.index(t)] for f in pk for t in tiles if tf[t] == f]
            if sub:
                bs.append(np.median(sub))
        lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
        from collections import Counter
        print(f"\nENSEMBLE {metric}: nested LOFO {np.median(vals):.4f}  [{lo:.4f}, {hi:.4f}]"
              f"   (OIS upper bound {np.median(ois):.4f})")
        for k, v in Counter(picks).most_common(4):
            print(f"    picked {k:<44} x{v}")
        out[metric] = {"nested_LOFO": float(np.median(vals)), "ci": [float(lo), float(hi)],
                       "OIS": float(np.median(ois)), "per_tile": vals, "picks": picks}
    json.dump(out, open(f"{SC}/ensemble_lofo.json", "w"), indent=1)
    print("\nwrote ensemble_lofo.json")


if __name__ == "__main__":
    main()
