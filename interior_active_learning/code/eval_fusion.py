"""
Evaluate ONE SAM+classifier combination against human ground truth.

Deterministic harness for the combination search: every trial goes through this
same code so results are comparable, and the search only varies its arguments.

    python3 eval_fusion.py --eval-image AS_24hr_BSE_Side_008 \
        --model LogisticRegression --sources pipeline,sam --fusion union \
        --sam-filter void_bright

Metrics are PIXEL-level, not region-level. Pipeline candidates and SAM masks
come from different segmentations, so they cannot be compared region-for-region;
pixels are the common ground. They are computed only over ADJUDICATED pixels --
those a human marked crack (1) or not-crack (2) -- because unreviewed pixels
cannot confirm or refute anything, and counting them as negatives would reward
whichever method predicts least.

The classifier trains on labelled pipeline regions from OTHER images and never
on the evaluation image, so a combination cannot score well by memorising it.
The evaluation image's own labels are used only as the answer key.

Reported at two operating points: the fixed 0.5 threshold, and the threshold
maximising F1 on this image. The latter is optimistic (chosen on the test image)
and is labelled as such -- it is there to separate "this combination ranks well"
from "this combination happens to sit at a good default".
"""
import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from common import PROJECT_ROOT

CACHE = os.path.join(PROJECT_ROOT, "fusion_cache")
TRAIN_CSV = os.path.join(PROJECT_ROOT, "training_data", "labeled_regions.csv")
FEATURES = ["LogArea", "Elongation", "Solidity", "Eccentricity",
            "Extent", "Circularity", "MeanDarkness", "MeanVesselness"]

MODELS = {
    "LogisticRegression": lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                    random_state=0, n_jobs=2),
    "GradientBoosting": lambda: GradientBoostingClassifier(n_estimators=200, random_state=0),
    "SVC": lambda: SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=0),
    "MLP": lambda: MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1500, random_state=0),
}


def unpack(bits, shape):
    n = shape[0] * shape[1]
    return np.unpackbits(bits)[:n].reshape(shape).astype(bool)


def load_cache(name):
    z = np.load(os.path.join(CACHE, f"{name}.npz"), allow_pickle=True)
    return z


def build_mask(z, accept_idx, shape):
    m = np.zeros(shape, bool)
    bbox, rshape, bits = z["bbox"], z["rshape"], z["bits"]
    for i in accept_idx:
        y0, y1, x0, x1 = bbox[i]
        sub = unpack(bits[i], tuple(rshape[i]))
        m[y0:y1, x0:x1] |= sub
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-image", required=True)
    ap.add_argument("--model", default="LogisticRegression", choices=list(MODELS))
    ap.add_argument("--sources", default="pipeline")       # pipeline | sam | pipeline,sam
    ap.add_argument("--fusion", default="union", choices=["union", "intersection"])
    ap.add_argument("--sam-filter", default="none", choices=["none", "void_bright"])
    ap.add_argument("--weighted", action="store_true", default=True)
    a = ap.parse_args()

    z = load_cache(a.eval_image)
    shape = tuple(int(v) for v in z["shape"])
    src = z["source"].astype(str)
    feats = z["feats"]
    mean_raw = z["mean_raw"]
    img_med = float(z["img_median"])

    # ---- train on OTHER images only ----
    tr = pd.read_csv(TRAIN_CSV)
    tr = tr[tr.SourceImage != a.eval_image]
    Xtr, ytr = tr[FEATURES].values, tr["IsCrack"].astype(bool).values
    cnt = tr.SourceImage.value_counts()
    w = np.array([1.0 / cnt[s] for s in tr.SourceImage]) if a.weighted else None
    sc = StandardScaler().fit(Xtr)
    clf = MODELS[a.model]()
    try:
        clf.fit(sc.transform(Xtr), ytr, sample_weight=w)
    except TypeError:
        clf.fit(sc.transform(Xtr), ytr)          # MLP takes no sample_weight

    p_all = clf.predict_proba(sc.transform(feats))[:, 1]

    wanted = set(a.sources.split(","))
    keep = np.isin(src, list(wanted))
    if a.sam_filter == "void_bright":
        bad = (src == "sam") & ((mean_raw <= 25) | (mean_raw >= img_med))
        keep &= ~bad

    hc = unpack(z["human_crack_bits"], shape)
    hn = unpack(z["human_notcrack_bits"], shape)
    adjudicated = hc | hn
    n_c, n_n = int(hc.sum()), int(hn.sum())

    def metrics(thr):
        acc = keep & (p_all >= thr)
        if a.fusion == "intersection" and wanted == {"pipeline", "sam"}:
            mp = build_mask(z, np.where(acc & (src == "pipeline"))[0], shape)
            ms = build_mask(z, np.where(acc & (src == "sam"))[0], shape)
            pred = mp & ms
        else:
            pred = build_mask(z, np.where(acc)[0], shape)
        tp = int((pred & hc).sum()); fn = int((~pred & hc).sum())
        fp = int((pred & hn).sum()); tn = int((~pred & hn).sum())
        rec = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        prec = tp / max(tp + fp, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        iou = tp / max(tp + fp + fn, 1)
        return dict(threshold=float(thr), n_regions=int(acc.sum()), recall=rec,
                    specificity=spec, precision=prec, f1=f1, iou=iou,
                    tp=tp, fp=fp, fn=fn, tn=tn)

    at_half = metrics(0.5)
    grid = [metrics(t) for t in np.linspace(0.05, 0.95, 19)]
    best = max(grid, key=lambda d: d["f1"])

    out = {
        "eval_image": a.eval_image, "model": a.model, "sources": a.sources,
        "fusion": a.fusion, "sam_filter": a.sam_filter,
        "n_pipeline_regions": int((src == "pipeline").sum()),
        "n_sam_regions": int((src == "sam").sum()),
        "n_regions_considered": int(keep.sum()),
        "human_crack_px": n_c, "human_notcrack_px": n_n,
        "adjudicated_px": int(adjudicated.sum()),
        "at_threshold_0.5": at_half,
        "best_f1_on_this_image_OPTIMISTIC": best,
        "train_rows": int(len(tr)), "train_images": int(tr.SourceImage.nunique()),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
