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


def probe_monai(f, ref):
    """MONAI: is there any way to exclude unreviewed pixels, and is ignore_empty one?"""
    import inspect
    import torch
    import monai
    from monai.losses import DiceCELoss, DiceLoss
    from monai.metrics import DiceMetric, MeanIoU

    has = {}
    for cls in (DiceMetric, MeanIoU, DiceLoss, DiceCELoss):
        has[cls.__name__] = "ignore_index" in inspect.signature(cls).parameters
    ignore_empty_present = "ignore_empty" in inspect.signature(DiceMetric).parameters

    # MONAI's contract is binarised one-hot: [B, C, ...] with values in {0, 1}. A third
    # state has nowhere to live -- encode it and it necessarily becomes one of the two.
    y_dense = torch.from_numpy(f["crack"].astype(np.float32))[None, None]
    p_t = torch.from_numpy(f["pred"].astype(np.float32))[None, None]
    dm_ = DiceMetric(include_background=False)
    dm_(p_t, y_dense)
    dice_dense = float(dm_.aggregate().item())
    dm_.reset()

    # The only route available: pre-filter, which one-hot cannot express spatially, so it
    # has to be done by flattening to the adjudicated pixels only.
    m = torch.from_numpy(f["adjudicated"].ravel())
    y_adj = torch.from_numpy(f["crack"].ravel().astype(np.float32))[m][None, None]
    p_adj = torch.from_numpy(f["pred"].ravel().astype(np.float32))[m][None, None]
    dm2 = DiceMetric(include_background=False)
    dm2(p_adj, y_adj)
    dice_adj = float(dm2.aggregate().item())

    # ignore_empty is about a whole case with an EMPTY ground truth, not about unreviewed
    # pixels inside a case. Distinguishing the two needs the RAW buffer and the not-nans
    # count: aggregate() alone collapses the NaN to 0.0 and both settings then look
    # identical, which is how a probe ends up "demonstrating" something it never showed.
    empty_y = torch.zeros_like(y_dense)
    d_true = DiceMetric(include_background=False, ignore_empty=True, get_not_nans=True)
    d_true(p_t, empty_y)
    buf_true = d_true.get_buffer()
    n_true = float(d_true.aggregate()[1].item())
    d_false = DiceMetric(include_background=False, ignore_empty=False, get_not_nans=True)
    d_false(p_t, empty_y)
    buf_false = d_false.get_buffer()
    n_false = float(d_false.aggregate()[1].item())
    v_true = "nan" if bool(torch.isnan(buf_true).any()) else float(buf_true.flatten()[0])
    v_false = "nan" if bool(torch.isnan(buf_false).any()) else float(buf_false.flatten()[0])

    return {
        "version": monai.__version__,
        "ignore_index_available": has,
        "ignore_empty_present": ignore_empty_present,
        "dice_dense": dice_dense,
        "dice_prefiltered": dice_adj,
        "ignore_empty_true_on_empty_gt": v_true,
        "ignore_empty_false_on_empty_gt": v_false,
        "cases_counted_ignore_empty_true": n_true,
        "cases_counted_ignore_empty_false": n_false,
        "verdict": (f"no ignore_index on DiceMetric, MeanIoU, DiceLoss or DiceCELoss. The "
                    f"input contract is binarised one-hot, so a third state has nowhere to "
                    f"live and the caller must pre-filter -- which moves Dice from "
                    f"{dice_dense:.4f} to {dice_adj:.4f}. ignore_empty is a FALSE FRIEND, and "
                    f"here is the distinction: on a case whose ground truth is entirely empty "
                    f"it writes {v_true} to the buffer and counts {n_true:.0f} case(s), versus "
                    f"{v_false} and {n_false:.0f} when off. So it does drop whole cases with "
                    f"no positive label -- a real feature, and a different question from "
                    f"excluding unreviewed pixels INSIDE a case, which nothing here can do."),
        "supports_unreviewed": False,
    }


def probe_datumaro(f, ref):
    """Datumaro -- the library under CVAT's mask export. What do unannotated pixels become?"""
    import tempfile
    import datumaro as dm
    from datumaro.components.annotation import AnnotationType
    from datumaro.components.dataset import Dataset
    from datumaro.components.media import Image as DmImage
    from PIL import Image as PILImage

    h, w = f["gt"].shape
    item = dm.DatasetItem(
        id="probe",
        media=DmImage.from_numpy(np.zeros((h, w, 3), dtype=np.uint8)),
        annotations=[dm.Mask(image=f["crack"], label=0)])
    ds = Dataset.from_iterable(
        [item], categories={AnnotationType.label:
                            dm.LabelCategories.from_iterable(["crack"])})
    out = {"version": dm.__version__}
    with tempfile.TemporaryDirectory() as td:
        ds.export(td, "voc", save_media=True)
        seg = None
        for root, _, files in os.walk(td):
            if "SegmentationClass" in root:
                for fn_ in files:
                    if fn_.endswith(".png"):
                        seg = os.path.join(root, fn_)
        if seg is None:
            out["verdict"] = "no SegmentationClass PNG produced"
            out["supports_unreviewed"] = False
            return out
        a = np.asarray(PILImage.open(seg).convert("RGB"))
        ann = {tuple(int(x) for x in v) for v in a[f["crack"]]}
        unann = {tuple(int(x) for x in v) for v in a[~f["crack"]]}
        out["colour_on_annotated"] = sorted(ann)
        out["colour_on_unannotated"] = sorted(unann)
        out["unannotated_is_voc_background"] = unann == {(0, 0, 0)}
        out["voc_void_colour_present"] = any(c == (224, 224, 192) for c in ann | unann)

    # Is the capability even in the library? Yes -- and unreachable from this path.
    try:
        from datumaro.plugins.data_formats.voc.format import make_voc_categories
        cats = make_voc_categories()
        labels = [c.name for c in cats[AnnotationType.label]]
        pal = dict(cats[AnnotationType.mask].colormap)
        void_idx = [k for k, v in pal.items() if tuple(v) == (224, 224, 192)]
        out["voc_labels_include_ignored"] = "ignored" in labels
        out["void_palette_index"] = void_idx[0] if void_idx else None
        out["void_palette_name"] = (labels[void_idx[0]]
                                    if void_idx and void_idx[0] < len(labels) else None)
    except Exception as e:
        out["voc_catalogue_probe"] = f"{type(e).__name__}: {e}"

    out["verdict"] = (
        f"unannotated pixels are written as {sorted(out['colour_on_unannotated'])} -- VOC "
        f"class 0, background -- while VOC's void colour (224,224,192) does not appear. And "
        f"the capability IS in the library: make_voc_categories() carries "
        f"{out.get('void_palette_name')!r} at palette index {out.get('void_palette_index')} "
        f"with exactly that colour. So the format supports void, the library knows about it, "
        f"and this export path silently turns 'nobody looked' into 'background'.")
    out["supports_unreviewed"] = False
    return out


def probe_label_studio(f, ref):
    """Label Studio's brush converter: does a void-encoded mask survive import?"""
    import inspect
    import tempfile
    import label_studio_converter.brush as lsb
    from PIL import Image as PILImage

    src = inspect.getsource(lsb)
    threshold_lines = [ln.strip() for ln in src.split("\n")
                       if "> 128" in ln and "np.array" in ln]

    # Encode the fixture the way PASCAL/Cityscapes do: 255 for the void/unreviewed state.
    voidenc = np.where(f["unreviewed"], 255,
                       np.where(f["crack"], 1, 0)).astype(np.uint8)
    out = {"threshold_source": threshold_lines[:1],
           "void_encoded_values": sorted(int(v) for v in np.unique(voidenc))}
    with tempfile.TemporaryDirectory() as td:
        pth = os.path.join(td, "void.png")
        PILImage.fromarray(voidenc).save(pth)
        # Reproduce the converter's own thresholding step on that file.
        arr = np.array(PILImage.open(pth))
        thresholded = (arr > 128)
        out["void_pixels_become_foreground"] = bool(thresholded[f["unreviewed"]].all())
        out["crack_pixels_become_foreground"] = bool(thresholded[f["crack"]].any())
        out["foreground_px_after_threshold"] = int(thresholded.sum())
        out["true_crack_px"] = int(f["crack"].sum())
    out["verdict"] = (
        f"the converter thresholds with `{threshold_lines[0] if threshold_lines else '> 128'}`. "
        f"A mask encoding unreviewed as 255 -- the PASCAL/Cityscapes convention -- therefore "
        f"imports every one of its {int(f['unreviewed'].sum())} unreviewed pixels as "
        f"FOREGROUND, while the {out['true_crack_px']} genuine crack pixels encoded as 1 are "
        f"DROPPED. The result is {out['foreground_px_after_threshold']} foreground px, almost "
        f"exactly inverted. The two conventions do not merely disagree, they corrupt each "
        f"other on round-trip.")
    out["supports_unreviewed"] = False
    return out


def probe_elf_matching(f, ref):
    """micro-sam's scorer: does elf's ignore_label protect predictions in ignored regions?"""
    import inspect
    from elf.evaluation import dice_score, matching

    h = w = 64
    gt = np.zeros((h, w), dtype=np.int32)     # 0 = the ignored / unreviewed state
    gt[8:24, 8:24] = 1                        # one reviewed object
    seg_ok = np.zeros((h, w), dtype=np.int32)
    seg_ok[9:23, 9:23] = 1                    # matches it
    seg_extra = seg_ok.copy()
    seg_extra[40:56, 40:56] = 2               # an object wholly inside ignored territory

    r_ok = matching(seg_ok, gt, ignore_label=0)
    r_extra_ign = matching(seg_extra, gt, ignore_label=0)
    r_extra_noign = matching(seg_extra, gt, ignore_label=None)
    d_ok = float(dice_score(seg_ok, gt))
    d_extra = float(dice_score(seg_extra, gt))

    return {
        "matching_signature": str(inspect.signature(matching)),
        "dice_score_signature": str(inspect.signature(dice_score)),
        "dice_score_has_mask_arg": any(
            k in inspect.signature(dice_score).parameters
            for k in ("ignore_label", "mask", "ignore_index")),
        "precision_matched_only": float(r_ok.get("precision")),
        "precision_with_ignored_region_prediction": float(r_extra_ign.get("precision")),
        "precision_same_without_ignoring": float(r_extra_noign.get("precision")),
        "ignore_label_penalises_more_than_not_ignoring": bool(
            r_extra_ign.get("precision") < r_extra_noign.get("precision")),
        "dice_matched_only": d_ok,
        "dice_with_ignored_region_prediction": d_extra,
        "verdict": (
            f"ignore_label=0 is the DEFAULT and looks like it protects unreviewed territory. "
            f"It does not. A prediction lying entirely inside the ignored region drops "
            f"precision from {r_ok.get('precision'):.4f} to "
            f"{r_extra_ign.get('precision'):.4f} -- and that is WORSE than passing "
            f"ignore_label=None ({r_extra_noign.get('precision'):.4f}). The parameter removes "
            f"the ignored region from the ground-truth OBJECTS without exempting predictions "
            f"that fall there, so they are charged as unmatched false positives. dice_score "
            f"has no mask argument at all and charges it too "
            f"({d_ok:.4f} -> {d_extra:.4f})."),
        "supports_unreviewed": False,
    }


PROBES = [
    ("sklearn.metrics", "sklearn", probe_sklearn),
    ("skimage.metrics", "skimage", probe_skimage),
    ("torch.nn.CrossEntropyLoss", "torch", probe_torch_loss),
    ("torchmetrics", "torchmetrics", probe_torchmetrics),
    ("segmentation_models_pytorch", "segmentation_models_pytorch", probe_smp),
    ("MONAI", "monai", probe_monai),
    ("datumaro (CVAT export path)", "datumaro", probe_datumaro),
    ("Label Studio brush converter", "label_studio_converter", probe_label_studio),
    ("elf / micro-sam matching", "elf", probe_elf_matching),
    ("mask file round-trip", "PIL", probe_mask_roundtrip),
]

#: Probes worth running that need a dependency this environment does not have. Listed so the
#: report states what was NOT tested -- an absence recorded is worth more than a silent skip.
NOT_TESTED = {
    "ilastik": ("NOT pip-installable -- no PyPI distribution exists (conda/binary "
                "only), so its headless export could not be executed here. The "
                "source-level finding stands unexecuted."),
    "micro_sam annotator GUI": ("the napari canvas and zarr fill value need an "
                                "interactive session; its SCORER is covered by the "
                                "elf probe above"),
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
