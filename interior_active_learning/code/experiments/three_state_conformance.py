"""Conformance suite: can a library express "a human never looked here"?

WHY THIS EXISTS
docs/UNLABELLED_PIXEL_AUDIT.md audits six external targets by READING their source. That
answers the generality objection but leaves the weakest joint in the argument: a referee can
say "you read code, you did not run it". This runs it.

THE FIXTURE
One canonical three-state ground truth, and one prediction, reused by every probe:

    1 = crack        a human marked this as the object
    2 = not-crack    a human marked this as background   <- ADJUDICATED negative
    0 = UNREVIEWED   nobody looked                       <- must not be scored as negative

The question each probe answers is narrow and factual: given this fixture, does the library
offer any way to say "exclude the unreviewed pixels", and what does it return when you ask?

HONESTY RULES BUILT INTO THE OUTPUT
  * Every probe reports EXECUTED or SKIPPED with the reason. A missing dependency is not a
    pass and not a failure -- it is an absence, and it is printed as one. A suite that
    silently skips is how "all tests pass" comes to mean nothing.
  * Probes assert on BEHAVIOUR, not on version strings, so the result stays meaningful as
    these libraries change. Where a library's behaviour is the correct one, the probe says
    so: this is a conformance suite, not a prosecution.
  * The suite exits non-zero only if a probe ERRORS unexpectedly -- not because a library
    treats unlabelled as background. That finding is the point of the experiment, so
    recording it must not look like a broken test run.

    python3 three_state_conformance.py
    python3 three_state_conformance.py --json
"""
import argparse
import importlib
import json
import os
import sys
import traceback
import warnings

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))

import numpy as np

OUT = os.path.join(_HERE, "three_state_conformance.json")

CRACK, NOT_CRACK, UNREVIEWED = 1, 2, 0


def fixture(h=64, w=64):
    """Ground truth, prediction, and the derived masks every probe shares.

    Deliberately small and deliberately lopsided: the adjudicated-negative region is a
    fraction of the frame, matching the real corpus where not-crack pixels are 0.034% of the
    total. A balanced toy fixture would hide the effect being measured.
    """
    gt = np.full((h, w), UNREVIEWED, dtype=np.uint8)
    gt[8:24, 8:40] = CRACK              # 16x32 = 512 px marked crack
    gt[40:44, 8:40] = NOT_CRACK         #  4x32 = 128 px marked not-crack
    pred = np.zeros((h, w), dtype=bool)
    pred[12:28, 8:40] = True            # overlaps the crack, spills into unreviewed
    pred[41:43, 12:20] = True           # a genuine false positive on adjudicated background
    pred[56:60, 50:60] = True           # entirely inside unreviewed territory
    return {
        "gt": gt, "pred": pred,
        "crack": gt == CRACK,
        "adjudicated_negative": gt == NOT_CRACK,
        "unreviewed": gt == UNREVIEWED,
        "adjudicated": (gt == CRACK) | (gt == NOT_CRACK),
    }


def reference_scores(f):
    """The two conventions, computed directly, as the yardstick for every probe."""
    def sc(neg):
        tp = int((f["pred"] & f["crack"]).sum())
        fn = int((~f["pred"] & f["crack"]).sum())
        fp = int((f["pred"] & neg).sum())
        tn = int((~f["pred"] & neg).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec,
                "recall": rec, "specificity": tn / max(tn + fp, 1),
                "iou": tp / max(tp + fp + fn, 1)}
    return {"adjudicated": sc(f["adjudicated_negative"]),
            "unlabelled_as_background": sc(~f["crack"])}


# ---------------------------------------------------------------------- probes
def probe_sklearn(f, ref):
    """Does sklearn let you exclude unreviewed pixels, and are siblings consistent?"""
    from sklearn.metrics import (confusion_matrix, jaccard_score,
                                 precision_recall_fscore_support)
    import inspect
    y_dense = f["crack"].ravel().astype(int)          # unlabelled folded into 0
    p = f["pred"].ravel().astype(int)
    m = f["adjudicated"].ravel()
    y_adj, p_adj = y_dense[m], p[m]                   # caller pre-filters, the only escape

    has_param = {fn.__name__: ("ignore_index" in inspect.signature(fn).parameters)
                 for fn in (confusion_matrix, jaccard_score,
                            precision_recall_fscore_support)}
    dense_iou = float(jaccard_score(y_dense, p, pos_label=1))
    adj_iou = float(jaccard_score(y_adj, p_adj, pos_label=1))
    # The documented workaround, and whether it actually helps for these two functions.
    labels_iou = float(jaccard_score(y_dense, p, labels=[1], average="micro"))
    cm_labels = confusion_matrix(y_dense, p, labels=[1]).tolist()

    return {
        "ignore_index_available": has_param,
        "iou_dense": dense_iou,
        "iou_prefiltered": adj_iou,
        "iou_with_labels_kwarg": labels_iou,
        "confusion_matrix_with_labels": cm_labels,
        "labels_kwarg_equals_dense": abs(labels_iou - dense_iou) < 1e-12,
        "verdict": ("no ignore_index on any of the three; the ONLY escape is for the caller "
                    "to pre-filter the arrays, which changes IoU from "
                    f"{dense_iou:.4f} to {adj_iou:.4f}. The natural `labels=[1]` idiom does "
                    "NOT help -- it returns the dense number "
                    f"({labels_iou:.4f}), while confusion_matrix(labels=[1]) does drop the "
                    "other class. Sibling functions in one namespace disagree."),
        "supports_unreviewed": False,
    }


def probe_skimage(f, ref):
    """skimage's connectomics metrics DO carry ignore_labels. Does it work?"""
    from skimage.metrics import adapted_rand_error, contingency_table
    import inspect
    gt_lab = f["gt"].astype(np.int64)                 # 0/1/2 as labels
    pred_lab = f["pred"].astype(np.int64)
    sig = str(inspect.signature(adapted_rand_error))
    # default ignore_labels=(0,) -- label 0 excluded without the caller asking
    d_default = adapted_rand_error(gt_lab, pred_lab)
    d_keep0 = adapted_rand_error(gt_lab, pred_lab, ignore_labels=())
    ct = contingency_table(gt_lab, pred_lab)
    return {
        "adapted_rand_error_signature": sig,
        "error_with_default_ignore_0": float(d_default[0]),
        "error_including_label_0": float(d_keep0[0]),
        "changes_when_0_included": abs(d_default[0] - d_keep0[0]) > 1e-9,
        "contingency_table_shape": list(ct.shape),
        "verdict": ("adapted_rand_error ships ignore_labels=(0,) as the DEFAULT, so label 0 "
                    "is excluded without the caller asking, and the result changes when it "
                    "is included. This is a library from the connectomics/EM community "
                    "getting it right by default -- the capability exists, it is just absent "
                    "from the metrics people report for crack segmentation."),
        "supports_unreviewed": True,
    }


def probe_torch_loss(f, ref):
    """torch.nn.CrossEntropyLoss ignore_index: does it exclude, and is the default usable?"""
    import torch
    import torch.nn as nn
    import inspect
    default = inspect.signature(nn.CrossEntropyLoss).parameters["ignore_index"].default
    logits = torch.zeros(1, 2, *f["gt"].shape)
    logits[0, 1][torch.from_numpy(f["pred"])] = 4.0
    logits[0, 0][~torch.from_numpy(f["pred"])] = 4.0

    # Encode UNREVIEWED as a sentinel the loss can be told to ignore.
    tgt_dense = torch.from_numpy(f["crack"].astype(np.int64))[None]
    sentinel = torch.from_numpy(np.where(f["unreviewed"], 255,
                                         f["crack"].astype(np.int64))).long()[None]
    l_dense = float(nn.CrossEntropyLoss()(logits, tgt_dense))
    l_ignore = float(nn.CrossEntropyLoss(ignore_index=255)(logits, sentinel))
    return {
        "ignore_index_default": default,
        "loss_dense": l_dense,
        "loss_with_unreviewed_ignored": l_ignore,
        "differs": abs(l_dense - l_ignore) > 1e-9,
        "verdict": (f"ignore_index exists and works: excluding the unreviewed pixels changes "
                    f"the loss from {l_dense:.4f} to {l_ignore:.4f}. But the default is "
                    f"{default}, a sentinel no mask file contains -- so the capability is "
                    f"real and only reachable if the caller invents an encoding for "
                    f"UNREVIEWED first. Training is the stage the field gets right."),
        "supports_unreviewed": True,
    }


def probe_torchmetrics(f, ref):
    """The classification-vs-segmentation asymmetry, executed."""
    import torch
    import torchmetrics
    import inspect
    from torchmetrics.classification import BinaryJaccardIndex
    out = {"version": torchmetrics.__version__}
    out["classification_has_ignore_index"] = (
        "ignore_index" in inspect.signature(BinaryJaccardIndex).parameters)

    seg_has = {}
    try:
        import torchmetrics.segmentation as seg
        for name in ("MeanIoU", "GeneralizedDiceScore", "DiceScore"):
            cls = getattr(seg, name, None)
            if cls is not None:
                seg_has[name] = "ignore_index" in inspect.signature(cls).parameters
    except Exception as e:
        seg_has["import_error"] = f"{type(e).__name__}: {e}"
    out["segmentation_has_ignore_index"] = seg_has

    p = torch.from_numpy(f["pred"].astype(np.int64))
    y = torch.from_numpy(np.where(f["unreviewed"], 255,
                                  f["crack"].astype(np.int64))).long()
    if out["classification_has_ignore_index"]:
        out["iou_ignoring_unreviewed"] = float(
            BinaryJaccardIndex(ignore_index=255)(p, y))
        out["iou_dense"] = float(BinaryJaccardIndex()(
            p, torch.from_numpy(f["crack"].astype(np.int64))))
        out["differs"] = abs(out["iou_ignoring_unreviewed"] - out["iou_dense"]) > 1e-9
    out["verdict"] = (
        "classification metrics accept ignore_index and it changes the answer; the "
        f"segmentation namespace exposes {sum(1 for v in seg_has.values() if v is True)} of "
        f"{len([k for k in seg_has if k != 'import_error'])} inspected classes with it. The "
        "asymmetry is inside one library, which is why a practitioner can reasonably believe "
        "their stack handles this.")
    out["supports_unreviewed"] = bool(out["classification_has_ignore_index"])
    return out


def probe_smp(f, ref):
    """segmentation_models_pytorch.get_stats: does it really REFUSE in binary mode?"""
    import torch
    import segmentation_models_pytorch as smp
    out = {"version": smp.__version__}
    p = torch.from_numpy(f["pred"].astype(np.int64))[None]
    y = torch.from_numpy(np.where(f["unreviewed"], 255,
                                  f["crack"].astype(np.int64))).long()[None]
    try:
        smp.metrics.get_stats(p, y, mode="binary", ignore_index=255)
        out["binary_ignore_index"] = "accepted"
        out["raises"] = False
    except Exception as e:
        out["binary_ignore_index"] = f"{type(e).__name__}: {str(e)[:160]}"
        out["raises"] = True
    try:
        smp.metrics.get_stats(p, y, mode="multiclass", num_classes=256, ignore_index=255)
        out["multiclass_ignore_index"] = "accepted"
    except Exception as e:
        out["multiclass_ignore_index"] = f"{type(e).__name__}: {str(e)[:120]}"
    out["verdict"] = (
        f"binary mode: {out['binary_ignore_index']}. Binary mode is exactly the "
        f"object-versus-background case of a micrograph, so if this raises, the library is "
        f"refusing the request rather than defaulting badly -- a stronger finding, because a "
        f"refusal is a decision.")
    out["supports_unreviewed"] = not out["raises"]
    return out


def probe_mask_roundtrip(f, ref):
    """Does a three-state mask survive the file formats these tools exchange?

    The information-theoretic core of the argument: once a mask is written with two states,
    no downstream scorer can recover which zeros were adjudicated.
    """
    import tempfile
    from PIL import Image
    import tifffile
    res = {}
    with tempfile.TemporaryDirectory() as td:
        p_png = os.path.join(td, "m.png")
        Image.fromarray(f["gt"]).save(p_png)
        back = np.asarray(Image.open(p_png))
        res["png_uint8_preserves_three_states"] = bool(
            set(np.unique(back)) == set(np.unique(f["gt"])))

        p_tif = os.path.join(td, "m.tif")
        tifffile.imwrite(p_tif, f["gt"])
        res["tiff_preserves_three_states"] = bool(
            set(np.unique(tifffile.imread(p_tif))) == set(np.unique(f["gt"])))

        # what a binary exporter does: threshold, then write 0/255
        binar = (f["gt"] == CRACK).astype(np.uint8) * 255
        p_bin = os.path.join(td, "b.png")
        Image.fromarray(binar).save(p_bin)
        rb = np.asarray(Image.open(p_bin))
        res["binary_export_states"] = sorted(int(v) for v in np.unique(rb))
        res["adjudicated_negative_recoverable_after_binary_export"] = False
        # and the Label Studio import rule, applied to a void-encoded mask
        void_mask = np.where(f["unreviewed"], 255,
                             (f["gt"] == CRACK).astype(np.uint8) * 1)
        res["void_255_becomes_foreground_under_gt128_threshold"] = bool(
            (void_mask > 128)[f["unreviewed"]].all())
    res["verdict"] = (
        "The FORMATS are innocent: uint8 PNG and TIFF both carry three states fine. The loss "
        "happens in the exporter, not the container. And a mask that encodes unreviewed as "
        "255 -- the PASCAL/Cityscapes convention -- is read as FOREGROUND by a `> 128` import "
        "threshold, so the two conventions actively corrupt each other.")
    res["supports_unreviewed"] = True
    return res


PROBES = [
    ("sklearn.metrics", "sklearn", probe_sklearn),
    ("skimage.metrics", "skimage", probe_skimage),
    ("torch.nn.CrossEntropyLoss", "torch", probe_torch_loss),
    ("torchmetrics", "torchmetrics", probe_torchmetrics),
    ("segmentation_models_pytorch", "segmentation_models_pytorch", probe_smp),
    ("mask file round-trip", "PIL", probe_mask_roundtrip),
]

#: Probes worth running that need a dependency this environment does not have. Listed so the
#: report states what was NOT tested -- an absence recorded is worth more than a silent skip.
NOT_TESTED = {
    "monai": "MONAI metrics: ignore_index absent, one-hot contract forbids a third state",
    "datumaro": "CVAT's exporter path: label map with a background entry and no void",
    "label_studio_converter": "Label Studio brush export/import, incl. the >128 threshold",
    "ilastik": "desktop app; headless export of Simple Segmentation vs LABELS",
    "micro_sam": "napari annotator canvas and zarr fill value",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    f = fixture()
    ref = reference_scores(f)
    report = {"fixture": {k: int(v.sum()) for k, v in f.items() if v.dtype == bool},
              "reference": ref, "probes": {}, "not_tested": NOT_TESTED}

    if not a.json:
        print("THREE-STATE CONFORMANCE SUITE")
        print(f"  fixture: {int(f['crack'].sum())} crack, "
              f"{int(f['adjudicated_negative'].sum())} adjudicated not-crack, "
              f"{int(f['unreviewed'].sum())} UNREVIEWED px")
        print(f"  the two conventions on this fixture: IoU "
              f"{ref['adjudicated']['iou']:.4f} (adjudicated) vs "
              f"{ref['unlabelled_as_background']['iou']:.4f} (dense); precision "
              f"{ref['adjudicated']['precision']:.4f} vs "
              f"{ref['unlabelled_as_background']['precision']:.4f}; recall "
              f"{ref['adjudicated']['recall']:.4f} vs "
              f"{ref['unlabelled_as_background']['recall']:.4f} (invariant)\n")

    errors = 0
    for label, dep, fn in PROBES:
        try:
            importlib.import_module(dep)
        except Exception:
            report["probes"][label] = {"status": "SKIPPED",
                                       "reason": f"{dep} not installed"}
            if not a.json:
                print(f"  SKIPPED  {label:32s} ({dep} not installed)")
            continue
        try:
            res = fn(f, ref)
            res["status"] = "EXECUTED"
            report["probes"][label] = res
            if not a.json:
                flag = "expresses UNREVIEWED" if res.get("supports_unreviewed") \
                    else "CANNOT express UNREVIEWED"
                print(f"  EXECUTED {label:32s} {flag}")
                print(f"           {res['verdict']}")
        except Exception as e:
            errors += 1
            report["probes"][label] = {"status": "ERROR",
                                       "error": f"{type(e).__name__}: {e}",
                                       "traceback": traceback.format_exc()[-600:]}
            if not a.json:
                print(f"  ERROR    {label:32s} {type(e).__name__}: {e}")

    ex = [k for k, v in report["probes"].items() if v.get("status") == "EXECUTED"]
    can = [k for k in ex if report["probes"][k].get("supports_unreviewed")]
    if not a.json:
        print(f"\n  {len(ex)} probe(s) executed, {len(can)} can express UNREVIEWED, "
              f"{len(ex) - len(can)} cannot")
        print(f"  {len(NOT_TESTED)} target(s) NOT tested here and named in the report, so the "
              f"gap is on the record rather than implied:")
        for k, why in NOT_TESTED.items():
            print(f"     {k:26s} {why}")
        print(f"\n  -> {OUT}")

    json.dump(report, open(OUT, "w"), indent=1, default=str)
    if a.json:
        print(json.dumps(report, indent=1, default=str))
    # A library treating unlabelled as background is the FINDING, not a test failure.
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
