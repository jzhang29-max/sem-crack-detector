#!/usr/bin/env python3
"""Score OmniCrack30k's released nnU-Net on our 15 disjoint tiles, under our protocol.

This is the arm that answers "is our model the best" honestly: the published cross-material
state of the art, run on OUR data, scored with OUR metrics at OUR tuning budgets -- rather than
setting our IoU beside the IoU in their paper, which would be meaningless.

OmniCrack30k: Benz & Rodehorst, CVPRW 2024, 10.1109/CVPRW63382.2024.00392. nnU-Net over 30,017
samples / 9.03 Gpx / 20 subsets including steel. Their reported figure is clIoU4px 64% ON THEIR
OWN TEST DATA; it is not a bar for this corpus and is not treated as one.

LOWER BOUND, not a fair fight: inference here used folds=(0,) instead of their (0,1,2,4)
ensemble and no test-time mirroring, both forced by CPU-only. Both choices can only hurt their
model. Grey was replicated to three channels.
"""
import os, json, glob
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize, remove_small_objects, disk, binary_dilation

SC = os.path.dirname(os.path.abspath(__file__))
QS = [50, 70, 80, 90, 94, 96, 97, 98, 99, 99.5]   # quantiles of the crack probability
PROB = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]       # absolute probability cuts
rng = np.random.default_rng(20260919)


def iou(p, g):
    u = np.logical_or(p, g).sum()
    return float(np.logical_and(p, g).sum() / u) if u else 0.0


def cldice(p, g):
    if p.sum() == 0 or g.sum() == 0 or p.mean() > 0.3:
        return 0.0
    sp, sg = skeletonize(p), skeletonize(g)
    tp = (sp & g).sum() / sp.sum() if sp.sum() else 0.0
    ts = (sg & p).sum() / sg.sum() if sg.sum() else 0.0
    return float(2 * tp * ts / (tp + ts)) if (tp + ts) > 0 else 0.0


def cliou(p, g, tau=4):
    if p.sum() == 0 or g.sum() == 0:
        return 0.0
    sp, sg = skeletonize(p), skeletonize(g)
    if sp.sum() == 0 or sg.sum() == 0:
        return 0.0
    k = disk(tau)
    tp = int((sg & binary_dilation(sp, k)).sum())
    fp = int((sp & ~(sp & binary_dilation(sg, k))).sum())
    fn = int((sg & ~(sg & binary_dilation(sp, k))).sum())
    d = tp + fp + fn
    return float(tp / d) if d else 0.0


def main():
    if not glob.glob(f"{SC}/omnicrack/*.npz"):
        raise SystemExit("no omnicrack maps; run run_omnicrack.py first")
    meta = json.load(open(f"{SC}/tiles/meta.json"))
    tf = {m["tile"]: m["frame"] for m in meta}
    data = {}
    for m in meta:
        t = m["tile"]
        f = f"{SC}/omnicrack/{t}.npz"
        if not os.path.exists(f):
            continue
        data[t] = {"p": np.load(f)["prob"].astype(np.float32),
                   "gt": np.array(Image.open(f"{SC}/tiles/{t}_gt.png")) > 127}
    tiles = list(data)
    print(f"{len(tiles)} tiles\n")
    print("raw crack-probability stats (before any threshold):")
    for t in tiles:
        p = data[t]["p"]
        print(f"  {t[:34]:<36} mean {p.mean():.4f}  p99 {np.percentile(p,99):.4f}  max {p.max():.4f}")

    cfgs = [("q", q) for q in QS] + [("p", v) for v in PROB]
    sc = {}
    for c in cfgs:
        sc[c] = {}
        for t in tiles:
            d = data[t]
            thr = np.percentile(d["p"], c[1]) if c[0] == "q" else c[1]
            b = remove_small_objects(d["p"] >= thr, max_size=32)
            sc[c][t] = (iou(b, d["gt"]), cldice(b, d["gt"]), cliou(b, d["gt"]))

    def boot(v):
        frames = sorted({tf[t] for t in tiles}); out = []
        for _ in range(4000):
            pk = rng.choice(frames, len(frames), replace=True)
            s = [v[tiles.index(t)] for f in pk for t in tiles if tf[t] == f]
            if s:
                out.append(np.median(s))
        return np.percentile(out, 2.5), np.percentile(out, 97.5)

    print(f"\n{'metric':<10} {'OIS':>8} {'ODS':>8} {'LOFO':>8}   {'LOFO 95% CI':>20}")
    out = {}
    for idx, metric in ((0, "IoU"), (1, "clDice"), (2, "clIoU4")):
        ois = [max(sc[c][t][idx] for c in cfgs) for t in tiles]
        odsk = max(cfgs, key=lambda c: np.median([sc[c][t][idx] for t in tiles]))
        ods = [sc[odsk][t][idx] for t in tiles]
        lofo = []
        for t in tiles:
            tr = [u for u in tiles if tf[u] != tf[t]]
            bk = max(cfgs, key=lambda c: np.median([sc[c][u][idx] for u in tr]))
            lofo.append(sc[bk][t][idx])
        lo, hi = boot(lofo)
        print(f"{metric:<10} {np.median(ois):>8.4f} {np.median(ods):>8.4f} {np.median(lofo):>8.4f}"
              f"   [{lo:.4f}, {hi:.4f}]")
        out[metric] = {"OIS": float(np.median(ois)), "ODS": float(np.median(ods)),
                       "LOFO": float(np.median(lofo)), "ci": [float(lo), float(hi)],
                       "per_tile": lofo, "ods_cfg": str(odsk)}
    json.dump(out, open(f"{SC}/omnicrack_eval.json", "w"), indent=1)
    print("\nwrote omnicrack_eval.json")
    print("\nREMINDER: single fold, no mirroring -> a LOWER BOUND on OmniCrack30k.")


if __name__ == "__main__":
    main()
