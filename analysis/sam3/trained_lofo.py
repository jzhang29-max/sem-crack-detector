#!/usr/bin/env python3
"""A trained pixel classifier on ridge features, leave-one-FRAME-out.

The question this answers: on 9 frames of SEM crack data, does training beat the best
training-free ridge filter? It is deliberately a SMALL model, because n is 9 frames, not 9000
images, and because the labels are region assertions -- a high-capacity model trained on a
59 px brush learns to predict a 59 px brush.

Protocol. Features are computed per pixel; the model trains on pixels from 8 frames and is
evaluated on the held-out frame's tiles. Frame level, never tile level: two tiles from one
frame share specimen, instrument settings and operator, so a tile-level split leaks.
The decision threshold is ALSO chosen on the training frames only -- choosing it on the test
frame would silently rebuild the OIS oracle this project already mistook for a result once.

Pixels are subsampled for training (all positives, a matched number of negatives) because a
1024^2 tile has a million pixels and the positive rate is 0.2-10%.
"""
import os, json
import numpy as np
from PIL import Image
from skimage.filters import frangi, sato, meijering, threshold_sauvola
from skimage.morphology import skeletonize, remove_small_objects
from scipy import ndimage as ndi
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

SC = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(20260919)


def features(g):
    """Per-pixel features. Everything here is computable from the image alone."""
    f = g.astype(np.float32) / 255.0
    feats, names = [], []
    feats.append(f); names.append("intensity")
    for smax in (2, 3, 4, 6):
        sig = np.arange(1, smax + 1, 1.0)
        for fn, nm in ((frangi, "frangi"), (sato, "sato"), (meijering, "meijering")):
            r = fn(f, sigmas=sig, black_ridges=True).astype(np.float32)
            m = r.max()
            feats.append(r / m if m > 0 else r); names.append(f"{nm}{smax}")
    for w in (25, 101):
        feats.append((threshold_sauvola(g, window_size=w, k=0.2) - g).astype(np.float32) / 255.0)
        names.append(f"sauvola{w}")
    for s in (1, 2, 4):
        feats.append(ndi.gaussian_gradient_magnitude(f, s)); names.append(f"grad{s}")
        feats.append(f - ndi.gaussian_filter(f, s)); names.append(f"dog{s}")
    return np.stack(feats, -1), names


def iou(p, gt):
    u = np.logical_or(p, gt).sum()
    return float(np.logical_and(p, gt).sum() / u) if u else 0.0


def cldice(p, gt):
    if p.sum() == 0 or gt.sum() == 0:
        return 0.0
    sp, sg = skeletonize(p), skeletonize(gt)
    tp = (sp & gt).sum() / sp.sum() if sp.sum() else 0.0
    ts = (sg & p).sum() / sg.sum() if sg.sum() else 0.0
    return float(2 * tp * ts / (tp + ts)) if (tp + ts) > 0 else 0.0


def main():
    meta = json.load(open(f"{SC}/tiles/meta.json"))
    data = []
    print("computing features ...", flush=True)
    for m in meta:
        g = np.array(Image.open(f"{SC}/tiles/{m['tile']}_gray.png").convert("L"))
        gt = np.array(Image.open(f"{SC}/tiles/{m['tile']}_gt.png")) > 127
        X, names = features(g)
        data.append({"tile": m["tile"], "frame": m["frame"], "X": X, "gt": gt})
    frames = sorted({d["frame"] for d in data})
    print(f"{len(data)} tiles, {len(frames)} frames, {data[0]['X'].shape[-1]} features\n")

    MODELS = {
        "logreg": lambda: LogisticRegression(max_iter=2000, C=1.0),
        "hist-gbdt": lambda: HistGradientBoostingClassifier(max_depth=4, max_iter=150,
                                                            learning_rate=0.1,
                                                            random_state=0),
    }
    results = {k: {} for k in MODELS}
    for held in frames:
        tr = [d for d in data if d["frame"] != held]
        te = [d for d in data if d["frame"] == held]
        Xs, ys = [], []
        for d in tr:
            X = d["X"].reshape(-1, d["X"].shape[-1]); y = d["gt"].ravel()
            pos = np.flatnonzero(y)
            neg = np.flatnonzero(~y)
            take = min(len(pos), 20000)
            pi = rng.choice(pos, take, replace=False) if len(pos) > take else pos
            ni = rng.choice(neg, take, replace=False)
            idx = np.concatenate([pi, ni])
            Xs.append(X[idx]); ys.append(y[idx])
        Xtr = np.concatenate(Xs); ytr = np.concatenate(ys)
        sc = StandardScaler().fit(Xtr)
        for mk, mf in MODELS.items():
            clf = mf().fit(sc.transform(Xtr), ytr)
            # threshold chosen on TRAINING frames only
            best_t, best_v = 0.5, -1
            probs_tr = []
            for d in tr[:6]:
                p = clf.predict_proba(sc.transform(d["X"].reshape(-1, d["X"].shape[-1])))[:, 1]
                probs_tr.append((p.reshape(d["gt"].shape), d["gt"]))
            for t in np.arange(0.1, 0.96, 0.05):
                v = np.median([iou(remove_small_objects(p >= t, max_size=32), gt) for p, gt in probs_tr])
                if v > best_v:
                    best_v, best_t = v, t
            for d in te:
                p = clf.predict_proba(sc.transform(d["X"].reshape(-1, d["X"].shape[-1])))[:, 1]
                pred = remove_small_objects(p.reshape(d["gt"].shape) >= best_t, max_size=32)
                results[mk][d["tile"]] = {"IoU": iou(pred, d["gt"]),
                                          "clDice": cldice(pred, d["gt"]),
                                          "thr": float(best_t)}
            print(f"  held-out {held:<34} {mk:<10} thr={best_t:.2f} "
                  f"IoU={np.median([results[mk][d['tile']]['IoU'] for d in te]):.4f}", flush=True)

    print(f"\n{'model':<12} {'median IoU (LOFO)':>18} {'median clDice':>15}")
    out = {}
    for mk in MODELS:
        i = [v["IoU"] for v in results[mk].values()]
        c = [v["clDice"] for v in results[mk].values()]
        print(f"{mk:<12} {np.median(i):>18.4f} {np.median(c):>15.4f}")
        out[mk] = {"median_IoU": float(np.median(i)), "median_clDice": float(np.median(c)),
                   "per_tile": results[mk]}
    json.dump(out, open(f"{SC}/trained_lofo.json", "w"), indent=1)
    print("\nwrote trained_lofo.json")


if __name__ == "__main__":
    main()
