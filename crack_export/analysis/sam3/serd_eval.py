#!/usr/bin/env python3
"""Score SAM 3's PRE-GATE dense response (SERD) against its post-gate instance masks.

SERD, arXiv:2607.12292: "the internal response achieves 82.66% average crack-pixel recall,
compared with 74.66% for the retained SAM3 proposals" over six public crack datasets.

Why it should matter specifically here, checkable in the vendored source: in
sam3_image_processor.py:193-201 the presence scalar multiplies `pred_logits` ONLY.
`outputs["pred_masks"]` is never touched by the presence head; the mask tensor is filtered
solely by the `keep` line. So the 105x presence span measured on this corpus, and the tiles
that returned nothing at all, are artefacts of `keep`. Reading pred_masks before `keep` is
presence-invariant BY CONSTRUCTION.

Enhancement follows the paper: Re = Norm( Rq * (1 + E) ), E = normalised 3x3 Sobel magnitude.
Both the raw field and the Sobel-enhanced field are scored, so the enhancement is not assumed
to help.

Protocol is the same as everything else here: OIS (per-tile oracle, upper bound only), ODS (one
quantile for all tiles) and LOFO (quantile chosen on the other 8 frames). Frame-clustered
bootstrap. The headline is LOFO.
"""
import os, json, glob
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize, remove_small_objects, disk, binary_dilation

SC = os.path.dirname(os.path.abspath(__file__))
QS = [90, 94, 96, 97, 98, 99, 99.3, 99.5, 99.7, 99.9]
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


def sobel_enh(r, g8):
    gx = ndi.sobel(g8.astype(np.float32), 0)
    gy = ndi.sobel(g8.astype(np.float32), 1)
    e = np.hypot(gx, gy)
    e = (e - e.min()) / max(np.ptp(e), 1e-9)
    out = r * (1.0 + e)
    return (out - out.min()) / max(np.ptp(out), 1e-9)


def main():
    files = glob.glob(f"{SC}/serd/*__crack.npz")
    if not files:
        raise SystemExit("no SERD maps; run SAM3_SAVE_SERD=1 run_real.py first")
    meta = json.load(open(f"{SC}/tiles/meta.json"))
    tf = {m["tile"]: m["frame"] for m in meta}
    res = {(r["tile"], r["prompt"]): r for r in json.load(open(f"{SC}/sam3_results.json"))
           if "error" not in r}

    data = {}
    for m in meta:
        t = m["tile"]
        f = f"{SC}/serd/{t}__crack.npz"
        if not os.path.exists(f):
            continue
        r = np.load(f)["r"].astype(np.float32)
        r = (r - r.min()) / max(np.ptp(r), 1e-9)
        g8 = np.array(Image.open(f"{SC}/tiles/{t}_gray.png").convert("L"))
        gt = np.array(Image.open(f"{SC}/tiles/{t}_gt.png")) > 127
        data[t] = {"raw": r, "enh": sobel_enh(r, g8), "gt": gt}
    tiles = list(data)
    print(f"{len(tiles)} tiles with a SERD map\n")

    # THE MECHANISTIC TEST: tiles where the gate returned nothing
    empty = [t for t in tiles if res.get((t, "crack"), {}).get("n", 0) == 0]
    print(f"Tiles where the presence gate returned NOTHING: {len(empty)}")
    for t in empty:
        d = data[t]
        best = max(iou(d["enh"] >= np.percentile(d["enh"], q), d["gt"]) for q in QS)
        bc = max(cldice(d["enh"] >= np.percentile(d["enh"], q), d["gt"]) for q in QS)
        print(f"  {t[:40]:<42} gated IoU 0.0000 -> SERD best IoU {best:.4f}, clDice {bc:.4f}")

    sc = {}
    for field in ("raw", "enh"):
        for q in QS:
            sc[(field, q)] = {}
            for t in tiles:
                d = data[t]
                p = remove_small_objects(d[field] >= np.percentile(d[field], q), max_size=32)
                sc[(field, q)][t] = (iou(p, d["gt"]), cldice(p, d["gt"]), cliou(p, d["gt"]))

    def boot(v):
        out = []
        frames = sorted({tf[t] for t in tiles})
        for _ in range(4000):
            pk = rng.choice(frames, len(frames), replace=True)
            s = [v[tiles.index(t)] for f in pk for t in tiles if tf[t] == f]
            if s:
                out.append(np.median(s))
        return np.percentile(out, 2.5), np.percentile(out, 97.5)

    print(f"\n{'arm':<22} {'metric':<8} {'OIS':>7} {'ODS':>7} {'LOFO':>7}   {'LOFO 95% CI':>20}")
    out = {}
    for field in ("raw", "enh"):
        keys = [(field, q) for q in QS]
        for idx, metric in ((0, "IoU"), (1, "clDice"), (2, "clIoU4")):
            ois = [max(sc[k][t][idx] for k in keys) for t in tiles]
            odsk = max(keys, key=lambda k: np.median([sc[k][t][idx] for t in tiles]))
            ods = [sc[odsk][t][idx] for t in tiles]
            lofo = []
            for t in tiles:
                tr = [u for u in tiles if tf[u] != tf[t]]
                bk = max(keys, key=lambda k: np.median([sc[k][u][idx] for u in tr]))
                lofo.append(sc[bk][t][idx])
            lo, hi = boot(lofo)
            nm = f"SERD {'+Sobel' if field=='enh' else 'raw'}"
            print(f"{nm:<22} {metric:<8} {np.median(ois):>7.4f} {np.median(ods):>7.4f} "
                  f"{np.median(lofo):>7.4f}   [{lo:.4f}, {hi:.4f}]")
            out[f"{nm}|{metric}"] = {"OIS": float(np.median(ois)), "ODS": float(np.median(ods)),
                                     "LOFO": float(np.median(lofo)), "ci": [float(lo), float(hi)],
                                     "per_tile": lofo}
    # gated reference
    gi = [res.get((t, "crack"), {}).get("union_IoU", 0.0) for t in tiles]
    gc = [res.get((t, "crack"), {}).get("union_clDice", 0.0) for t in tiles]
    print(f"{'SAM 3 gated union':<22} {'IoU':<8} {'--':>7} {'--':>7} {np.median(gi):>7.4f}")
    print(f"{'SAM 3 gated union':<22} {'clDice':<8} {'--':>7} {'--':>7} {np.median(gc):>7.4f}")
    out["gated|IoU"] = float(np.median(gi)); out["gated|clDice"] = float(np.median(gc))

    from scipy.stats import wilcoxon
    print("\nPAIRED, SERD+Sobel LOFO vs the gated union:")
    for idx, metric, ref in ((0, "IoU", gi), (1, "clDice", gc)):
        a = out[f"SERD +Sobel|{metric}"]["per_tile"]
        d = [x - y for x, y in zip(a, ref)]
        nz = [x for x in d if x != 0]
        p = wilcoxon(nz).pvalue if len(nz) >= 6 else float("nan")
        print(f"  {metric:<8} median d {np.median(d):+.4f}  wins {sum(1 for x in d if x>0)}/{len(d)}  p={p:.4f}")
    json.dump(out, open(f"{SC}/serd_eval.json", "w"), indent=1)
    print("\nwrote serd_eval.json")


if __name__ == "__main__":
    main()
