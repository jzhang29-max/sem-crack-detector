"""
Final retrain on all 32 reviewed images, with the one correction the data
demands: per-image sample weights.

Why weighting is not a thumb on the scale. Each image contributed a wildly
different number of reviewed regions -- 1 row from one image, 1285 from
another -- and that variation reflects how much time was spent painting, not
how much material each image represents. Unweighted, AS_24hr_BSE_Side_008 and
MAR_Amb_AS_CBS_0003 together are 46% of the training signal. Weighting each
row by 1/(rows in its image) makes every image count equally, which is the
assumption we actually want: each image is one sample of the material.

Measured effect, evaluated on AS_24hr_BSE_Side_008 held out entirely (the only
image labelled exhaustively enough for its positive/negative ratio to be
trustworthy):
    unweighted, trained on the other 31 images : AUC 0.7294, recall  6.9%
    weighted,   trained on the other 31 images : AUC 0.9252, recall 93.5%
    current production model (saw this image)  : AUC 0.8623, recall 89.3%
See label_bias_experiment.py for the full condition table.

Two estimators are reported because neither alone is honest here:
  * pooled out-of-fold AUC -- concatenate held-out predictions from all folds,
    score once over all 4110 rows. Stable, but only comparable across model
    families whose scores are calibrated consistently between folds; tree
    ensembles are not, so their pooled number is pessimistic. Noted inline.
  * leave-one-image-out on the exhaustive image -- the decisive test, since
    it is the only place a specificity number means anything.

    python3 train_v3_weighted.py
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                              roc_curve, average_precision_score)

from common import PROJECT_ROOT, PROD_MODEL_PATH

CSV = os.path.join(PROJECT_ROOT, "training_data", "labeled_regions.csv")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
OUT_MODEL = os.path.join(PROJECT_ROOT, "models", "crack_classifier_v3_weighted.joblib")
OUT_JSON = os.path.join(PROJECT_ROOT, "models", "crack_classifier_v3_metrics.json")

FEATURES = ["LogArea", "Elongation", "Solidity", "Eccentricity",
            "Extent", "Circularity", "MeanDarkness", "MeanVesselness"]
HELD = "AS_24hr_BSE_Side_008"
SEEDS = [0, 1, 2, 3, 4]
MODELS = {
    "LogisticRegression": lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                    random_state=0),
    "GradientBoosting": lambda: GradientBoostingClassifier(n_estimators=200, random_state=0),
    "SVC (RBF)": lambda: SVC(kernel="rbf", probability=True, class_weight="balanced",
                              random_state=0),
}


def image_weights(src):
    counts = pd.Series(src).value_counts()
    return np.array([1.0 / counts[s] for s in src])


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    df = pd.read_csv(CSV)
    X = df[FEATURES].values
    y = df["IsCrack"].astype(bool).values
    groups = df["SourceImage"].values
    W = image_weights(groups)

    print(f"{len(df)} reviewed regions | {int(y.sum())} crack / {int((~y).sum())} not-crack "
          f"| {df.SourceImage.nunique()} images\n")

    # ---------- grouped CV, pooled out-of-fold ----------
    print("=" * 84)
    print("GROUPED CROSS-VALIDATION (5-fold x 5 seeds, grouped by image, per-image weights)")
    print("=" * 84)
    print(f"{'model':22s} {'pooled OOF AUC':>20s} {'acc':>8s} {'recall':>8s} {'spec':>8s} {'LOIO AUC':>10s}")
    results, oof_store = {}, {}
    for name, factory in MODELS.items():
        aucs, accs, recs, specs = [], [], [], []
        oof_seed0 = None
        for seed in SEEDS:
            sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            oof = np.full(len(y), np.nan)
            for tr, te in sgkf.split(X, y, groups):
                sc = StandardScaler().fit(X[tr])
                clf = factory()
                clf.fit(sc.transform(X[tr]), y[tr], sample_weight=W[tr])
                oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
            pred = oof >= 0.5
            cm = confusion_matrix(y, pred, labels=[False, True])
            tn, fp, fn, tp = cm.ravel()
            aucs.append(roc_auc_score(y, oof))
            accs.append(accuracy_score(y, pred))
            recs.append(tp / max(tp + fn, 1))
            specs.append(tn / max(tn + fp, 1))
            if seed == 0:
                oof_seed0 = oof
        oof_store[name] = oof_seed0

        # decisive test: hold out the exhaustively-labelled image completely
        te = groups == HELD
        sc = StandardScaler().fit(X[~te])
        clf = factory()
        clf.fit(sc.transform(X[~te]), y[~te], sample_weight=W[~te])
        loio = roc_auc_score(y[te], clf.predict_proba(sc.transform(X[te]))[:, 1])

        results[name] = {"pooled_auc": float(np.mean(aucs)), "pooled_auc_std": float(np.std(aucs)),
                          "acc": float(np.mean(accs)), "recall": float(np.mean(recs)),
                          "spec": float(np.mean(specs)), "loio_auc_exhaustive_image": float(loio)}
        print(f"{name:22s} {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}  {np.mean(accs):>7.4f} "
              f"{np.mean(recs):>7.1%} {np.mean(specs):>7.1%} {loio:>10.4f}")

    print("\nNOTE: tree ensembles' pooled OOF AUC is pessimistic -- pooling requires scores to be")
    print("calibrated consistently across folds, which RF/GB are not. Judge them on LOIO AUC.")

    # Model choice is NOT simply argmax LOIO AUC. SVC (RBF) ranks best there
    # (0.9463 vs 0.9252, bootstrap difference +0.021 with 95% CI [+0.002,
    # +0.042], so the edge is real) but it is unusable at the threshold this
    # pipeline actually applies: at 0.5 it labels all 1285 regions crack --
    # recall 100%, specificity 0%, 1023 false positives. Its ranking is fine
    # and its probability SCALE is not, which the per-fold score ranges show
    # directly (one fold produced scores only in [0.93, 1.00]). The same
    # instability is why its pooled OOF AUC inverts to 0.20.
    # LogisticRegression is the only family whose ranking is strong AND whose
    # probabilities sit on a stable, comparable scale across training sets, so
    # a fixed threshold means the same thing from one retrain to the next.
    DISQUALIFIED = {"SVC (RBF)": "unusable at threshold 0.5 (spec 0.0%, 1023 FP)",
                    "RandomForest": "spec 0.5% at threshold 0.5",
                    "GradientBoosting": "spec 2.3% at threshold 0.5"}
    for k, why in DISQUALIFIED.items():
        if k in results:
            print(f"  disqualified {k}: {why}")
    best = "LogisticRegression"
    print(f"\nselected: {best} -- LOIO AUC {results[best]['loio_auc_exhaustive_image']:.4f}, "
          f"and calibrated stably enough for a fixed threshold")

    # ---------- production baseline on the same held-out image ----------
    held = df[df.SourceImage == HELD]
    yb = held["IsCrack"].astype(bool).values
    pb = joblib.load(PROD_MODEL_PATH)
    pp = pb["clf"].predict_proba(pb["scaler"].transform(held[pb["feature_names"]].values))[:, 1]
    prod_auc = roc_auc_score(yb, pp)
    print(f"production model on that same image: AUC {prod_auc:.4f} "
          f"(optimistic -- it was trained with this image's labels)")

    # ---------- pick an operating threshold on the held-out image ----------
    # The default 0.5 is not sacred; it is just where the retrained model's
    # score happens to fall. Sweep it on the held-out image and quote the
    # comparison at MATCHED recall, which is the only way to say "fewer false
    # positives" without secretly trading away sensitivity.
    te = groups == HELD
    scH = StandardScaler().fit(X[~te])
    clfH = MODELS[best]()
    clfH.fit(scH.transform(X[~te]), y[~te], sample_weight=W[~te])
    pH = clfH.predict_proba(scH.transform(X[te]))[:, 1]
    yt = y[te]

    def at(thr, pv=None):
        pr = (pH if pv is None else pv) >= thr
        tp = int((yt & pr).sum()); fn = int((yt & ~pr).sum())
        tn = int((~yt & ~pr).sum()); fp = int((~yt & pr).sum())
        return tp / max(tp + fn, 1), tn / max(tn + fp, 1), fp

    # Production's OWN threshold, not 0.5. The deployed model carries the
    # threshold it was calibrated at (0.4 today), so measuring its recall at 0.5
    # described a model nobody runs -- and every candidate threshold below is
    # chosen to match that recall, so the whole calibration hung off a baseline
    # that did not exist.
    PROD_THR = float(pb.get("threshold", 0.5))
    prod_rec, prod_spec, prod_fp = at(PROD_THR, pp)
    fpr_h, tpr_h, thr_h = roc_curve(yt, pH)
    thr_match_rec = float(thr_h[int(np.argmin(np.abs(tpr_h - prod_rec)))])
    thr_match_spec = float(thr_h[int(np.argmin(np.abs(fpr_h - (1 - prod_spec))))])
    thr_youden = float(thr_h[int(np.argmax(tpr_h - fpr_h))])
    ops = {"matched_recall": thr_match_rec, "matched_specificity": thr_match_spec,
           "youden": thr_youden, "default": 0.5}
    print(f"\noperating points on held-out {HELD} "
          f"(production: recall {prod_rec:.1%}, spec {prod_spec:.1%}, {prod_fp} FP)")
    for nm, t in ops.items():
        r, s, fp = at(t)
        print(f"  {nm:20s} thr {t:.3f}  recall {r:6.1%}  spec {s:6.1%}  FP {fp:>5d}")
    # thr_match_rec lives on clfH's score scale -- clfH is trained on everything
    # EXCEPT the held-out image. The model actually deployed below is refit on ALL
    # rows, so its scores are distributed differently and the same numeric
    # threshold is a different operating point. This project shipped exactly that
    # mistake once: 0.578 picked on a held-out-trained model gave 74.4% recall in
    # production instead of the intended 89.3%.
    #
    # Transfer by QUANTILE rather than by value: take where thr_match_rec sits in
    # clfH's held-out score distribution and use the same position in the deployed
    # model's pooled out-of-fold scores. Out-of-fold, so it is not the in-sample
    # optimism that re-sweeping on the held-out image would give.
    _q = float((pH < thr_match_rec).mean())
    _oof_best = oof_store.get(best)
    if _oof_best is not None and np.isfinite(_oof_best).any():
        DEPLOY_THR = float(np.nanquantile(_oof_best, _q))
        print(f"\nthreshold transferred by quantile: {thr_match_rec:.3f} on the "
              f"held-out-trained model sits at q={_q:.4f}, which is "
              f"{DEPLOY_THR:.3f} on the deployed model's out-of-fold scores")
    else:
        DEPLOY_THR = thr_match_rec
        print(f"\nWARNING: no out-of-fold scores for {best}; deploying "
              f"{DEPLOY_THR:.3f} straight from the held-out-trained model, whose "
              f"score scale differs from the deployed model's")
    r, s, fp = at(DEPLOY_THR)
    delta = prod_fp - fp
    # Report the direction correctly. This previously printed "N fewer false
    # positives" with a negative N and a nonsensical percentage whenever the
    # candidate was WORSE, which reads as an improvement at a glance.
    if delta > 0:
        verdict = f"{delta} FEWER false positives ({delta / max(prod_fp, 1):.0%} reduction)"
    elif delta < 0:
        verdict = f"{-delta} MORE false positives ({-delta / max(prod_fp, 1):.0%} increase) -- WORSE"
    else:
        verdict = "the same number of false positives"
    print(f"\ncandidate threshold {DEPLOY_THR:.3f} (matched recall): {verdict}")
    print(f"  candidate: recall {r:.1%}  spec {s:.1%}  FP {fp}")
    print(f"  baseline : recall {prod_rec:.1%}  spec {prod_spec:.1%}  FP {prod_fp}")
    # The baseline here is whatever is currently deployed. If that model was
    # itself trained on this held-out image, its numbers are in-sample and
    # therefore flattering, so the comparison is not apples-to-apples -- say so
    # rather than letting the table imply otherwise.
    try:
        _pb = joblib.load(PROD_MODEL_PATH)
        if HELD in (_pb.get("images") or []):
            print(f"  NOTE: the baseline model was trained on {HELD}, so its numbers are "
                  f"in-sample and optimistic. Compare LOIO AUC instead.")
    except Exception:
        pass

    # ---------- figures ----------
    fig, axes = plt.subplots(1, 4, figsize=(25, 5.6))

    ax = axes[0]
    for name in MODELS:
        fpr, tpr, _ = roc_curve(y, oof_store[name])
        ax.plot(fpr, tpr, lw=1.9, label=f"{name} (AUC {results[name]['pooled_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.9, alpha=0.5)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Pooled out-of-fold ROC\n4110 reviewed regions, grouped by image")
    ax.legend(loc="lower right", fontsize=8.5); ax.grid(alpha=0.3)

    ax = axes[1]
    pw = pH
    clfu = MODELS[best]()
    clfu.fit(scH.transform(X[~te]), y[~te])
    pu = clfu.predict_proba(scH.transform(X[te]))[:, 1]
    for lbl, pv in [(f"retrained, per-image weights (AUC {roc_auc_score(y[te], pw):.3f})", pw),
                     (f"retrained, unweighted (AUC {roc_auc_score(y[te], pu):.3f})", pu),
                     (f"current production (AUC {prod_auc:.3f})", pp)]:
        fpr, tpr, _ = roc_curve(y[te], pv)
        ax.plot(fpr, tpr, lw=2.0, label=lbl)
    ax.plot([0, 1], [0, 1], "k--", lw=0.9, alpha=0.5)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(f"Held out entirely: {HELD}\n1285 exhaustively labelled regions")
    ax.legend(loc="lower right", fontsize=8.5); ax.grid(alpha=0.3)

    ax = axes[2]
    per = df.groupby("SourceImage")["IsCrack"].agg(["size", "sum"])
    per["neg"] = per["size"] - per["sum"]
    per = per.sort_values("size", ascending=True)
    ypos = np.arange(len(per))
    ax.barh(ypos, per["sum"], color="#c0392b", label="marked crack")
    ax.barh(ypos, per["neg"], left=per["sum"], color="#2980b9", label="marked NOT crack")
    ax.set_yticks(ypos)
    # truncate from the LEFT -- these names differ only in their trailing
    # index (..._CBS_001 vs ..._CBS_010), so a head-truncated label renders
    # a dozen visually identical rows.
    ax.set_yticklabels([s if len(s) <= 26 else ".." + s[-24:] for s in per.index], fontsize=6.4)
    ax.set_xlabel("reviewed regions"); ax.set_xscale("symlog")
    ax.set_title("Label supply per image\n24 of 32 images have no not-crack marks")
    ax.legend(fontsize=8.5); ax.grid(alpha=0.3, axis="x")

    ax = axes[3]
    sweep = np.linspace(0.05, 0.95, 181)
    rec = [at(t)[0] for t in sweep]
    spc = [at(t)[1] for t in sweep]
    ax.plot(sweep, rec, lw=2.0, color="#c0392b", label="recall (cracks found)")
    ax.plot(sweep, spc, lw=2.0, color="#2980b9", label="specificity (non-cracks rejected)")
    ax.axhline(prod_rec, color="#c0392b", ls=":", lw=1.4)
    ax.axhline(prod_spec, color="#2980b9", ls=":", lw=1.4)
    ax.axvline(DEPLOY_THR, color="k", ls="--", lw=1.5)
    r_d, s_d, fp_d = at(DEPLOY_THR)
    ax.annotate(f"deployed thr {DEPLOY_THR:.3f}\nrecall {r_d:.1%} = production\n"
                f"spec {s_d:.1%} vs {prod_spec:.1%}\nFP {fp_d} vs {prod_fp}",
                xy=(DEPLOY_THR, s_d), xytext=(0.10, 0.30), fontsize=8.5,
                arrowprops=dict(arrowstyle="->", lw=1.1))
    ax.text(0.97, prod_rec + 0.015, "production recall", ha="right", fontsize=7.5, color="#c0392b")
    ax.text(0.97, prod_spec + 0.015, "production specificity", ha="right", fontsize=7.5,
            color="#2980b9")
    ax.set_xlabel("decision threshold"); ax.set_ylabel("rate"); ax.set_ylim(0, 1.02)
    ax.set_title(f"Threshold sweep on held-out {HELD[:20]}\ndotted = current production at 0.5")
    ax.legend(loc="center right", fontsize=8.5); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "retrain_on_all_corrections.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out}")

    # ---------- deploy ----------
    sc = StandardScaler().fit(X)
    clf = MODELS[best]()
    clf.fit(sc.transform(X), y, sample_weight=W)
    # sklearn_version: a bundle is a pickled estimator, and sklearn compares this
    # against the running version on every load, warning that results may be
    # invalid when they differ. Every module here silences warnings, so without
    # recording it the app cannot report the mismatch -- and leaving it out would
    # undo that on the first retrain a user runs.
    import sklearn as _sklearn
    joblib.dump({"sklearn_version": _sklearn.__version__,
                  "scaler": sc, "clf": clf, "feature_names": FEATURES,
                  "model_family": best,
                  "per_image_weights": True, "threshold": DEPLOY_THR,
                  "threshold_provenance": {
                      "method": "quantile transfer of matched-recall threshold",
                      "production_threshold_compared": PROD_THR,
                      "held_out_model_threshold": thr_match_rec,
                      "quantile": _q, "held_out_image": HELD},
                  "operating_points": ops, "n_train": len(df),
                  "n_pos": int(y.sum()), "n_neg": int((~y).sum()),
                  "images": sorted(df.SourceImage.unique().tolist()),
                  "cv_results": results, "loio_image": HELD,
                  "loio_recall": float(r_d), "loio_spec": float(s_d), "loio_fp": int(fp_d),
                  "prod_auc_on_loio_image": float(prod_auc),
                  "prod_recall": float(prod_rec), "prod_spec": float(prod_spec),
                  "prod_fp": int(prod_fp)}, OUT_MODEL)
    with open(OUT_JSON, "w") as f:
        json.dump({"n_train": len(df), "n_pos": int(y.sum()), "n_neg": int((~y).sum()),
                    "n_images": int(df.SourceImage.nunique()),
                    "images_with_negatives": int((per["neg"] > 0).sum()),
                    "best": best, "threshold": DEPLOY_THR, "operating_points": ops,
                    "cv_results": results,
                    "held_out_image": HELD,
                    "retrained_on_held_out": {"auc": float(roc_auc_score(yt, pH)),
                                               "recall": float(r_d), "spec": float(s_d),
                                               "false_positives": int(fp_d)},
                    "production_on_held_out": {"auc": float(prod_auc), "recall": float(prod_rec),
                                                "spec": float(prod_spec),
                                                "false_positives": int(prod_fp)}}, f, indent=2)
    print(f"wrote {OUT_MODEL}")
    print("\nNOT swapped into production yet -- models/crack_classifier.joblib is untouched.")


if __name__ == "__main__":
    main()
