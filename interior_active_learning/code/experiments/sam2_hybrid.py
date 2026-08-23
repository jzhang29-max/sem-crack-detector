"""Does SAM 2 segment these cracks better than the shipped detector, alone or hybridised?

WHY ASK
The survey put this project's detector last: a darkness threshold plus a LogisticRegression
over 8 features, missing roughly 40% of crack pixels. The measurement layer is
detector-agnostic by design, so if a foundation model draws better crack boundaries the right
move is to take them. SAM 1 was tried and switched off; SAM 2 is a different model and the
question is open.

HOW SAM 2 IS PROMPTED, and why not automatically
transformers exposes Sam2Model as a PROMPTABLE segmenter, with no automatic mask generator.
That is a better fit here anyway. Automatic mask generation proposes whole-object regions and
a crack is a thin dark filament; SAM 1's automatic proposals are why that experiment
disappointed. Measured on a synthetic 3-px filament, prompt geometry decides the outcome:

    box around the structure      IoU 0.742
    single point on it            IoU 0.604
    ten points along it           IoU 0.545

So each of the pipeline's own candidate regions is passed to SAM 2 as a BOX prompt and SAM 2
redraws the boundary inside a window around it. The pipeline proposes, SAM 2 refines.

THE VARIANT CHOICE IS NOT ALLOWED TO CHEAT. SAM 2 returns three mask variants per prompt.
Picking the one that best matches the human mask would score SAM 2 on an oracle it will not
have at inference, so the variant is chosen by the model's OWN predicted IoU and the ground
truth is touched only when scoring.

ARMS, all scored on identical adjudicated pixels so the numbers sit beside naive_baselines.py
  pipeline          the shipped two-pass detector
  sam2_refine       SAM 2's boundaries for the candidates the pipeline ACCEPTED
  sam2_all          SAM 2's boundaries for EVERY candidate, accepted or rejected -- asks
                    whether SAM 2 can recover cracks the classifier threw away
  hybrid_or         pipeline OR sam2_refine, the same union the existing SAM 1 path uses

    python3 sam2_hybrid.py --frames 260708_316_H_b2_front_CBS_012
    python3 sam2_hybrid.py --model facebook/sam2.1-hiera-small --max-candidates 400
"""
import argparse
import contextlib
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..", "code")))

import numpy as np
from PIL import Image

import detector_config as _dc
import unified_pipeline as up
from common import load_correction_mask
from scoring_convention_bias import eligible

Image.MAX_IMAGE_PIXELS = None
OUT = os.path.join(_HERE, "sam2_hybrid.json")


#: Which detector this experiment measures. this experiment applies its OWN refinement, so the pipeline must arrive bare -- otherwise the 'pipeline' arm is refined and the 'sam2_refine' arm is refined twice, and the comparison measures nothing.
DETECTOR = "off"


def _out_for(model_id):
    """One file per checkpoint. The report dedupes by image, so mixing two checkpoints in
    one file would average them together and call it a single result."""
    tag = model_id.split("/")[-1].replace(".", "").replace("-", "_")
    return OUT.replace(".json", f".{tag}.json") if "tiny" not in tag else OUT

#: Window padding around a candidate's box, in pixels. SAM 2 needs context around the object;
#: too little and it cannot tell filament from field, too much and the window stops being
#: about this candidate.
PAD = 48
MIN_WIN = 128


@contextlib.contextmanager
def _without_human_input():
    a, b = up.load_correction_mask, up.load_hard_overrides
    up.load_correction_mask = lambda *x, **k: None
    up.load_hard_overrides = lambda *x, **k: None
    try:
        yield
    finally:
        up.load_correction_mask, up.load_hard_overrides = a, b


def _shape_cost(pred):
    """What an adjudicated-pixel score cannot see: is the mask solid or lacy?

    This exists because the four-metric comparison below was used to make SAM 2 the default,
    and it was wrong to. Refinement raised f1 while nearly doubling the number of connected
    components and more than doubling total skeleton length -- turning solid crack regions
    into ragged, broken ones. Crack COUNT and crack LENGTH are the two headline quantities
    this tool produces, and both degrade in a way f1 on 8% of the frame is blind to. Any
    future comparison has to report these beside the metrics.
    """
    from skimage import measure as _m, morphology as _mo
    lab = _m.label(pred, connectivity=2)
    return {"components": int(lab.max()),
            "skeleton_px": int(_mo.skeletonize(pred).sum()),
            "predicted_px": int(pred.sum())}


def _score(pred, crack, neg):
    tp = int((pred & crack).sum()); fn = int((~pred & crack).sum())
    fp = int((pred & neg).sum());   tn = int((~pred & neg).sum())
    rec = tp / max(tp + fn, 1); prec = tp / max(tp + fp, 1)
    return {"f1": 2 * prec * rec / max(prec + rec, 1e-9), "recall": rec,
            "specificity": tn / max(tn + fp, 1), "precision": prec,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _load_sam2(model_id):
    import torch
    from transformers import Sam2Model, Sam2Processor
    proc = Sam2Processor.from_pretrained(model_id)
    model = Sam2Model.from_pretrained(model_id).eval()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    return proc, model.to(dev), dev, torch


def _sam2_mask_for_boxes(img8, boxes, proc, model, dev, torch, log_every=100):
    """Union of SAM 2's boundary for each box prompt, in full-frame coordinates."""
    H, W = img8.shape
    out = np.zeros((H, W), dtype=bool)
    t0 = time.time()
    for i, (y0, x0, y1, x1) in enumerate(boxes):
        wy0 = max(0, y0 - PAD); wx0 = max(0, x0 - PAD)
        wy1 = min(H, y1 + PAD); wx1 = min(W, x1 + PAD)
        if (wy1 - wy0) < MIN_WIN:
            c = (wy0 + wy1) // 2
            wy0, wy1 = max(0, c - MIN_WIN // 2), min(H, c + MIN_WIN // 2)
        if (wx1 - wx0) < MIN_WIN:
            c = (wx0 + wx1) // 2
            wx0, wx1 = max(0, c - MIN_WIN // 2), min(W, c + MIN_WIN // 2)
        win = img8[wy0:wy1, wx0:wx1]
        if win.size == 0:
            continue
        rgb = np.dstack([win] * 3)
        box = [[[float(x0 - wx0), float(y0 - wy0),
                 float(x1 - wx0), float(y1 - wy0)]]]
        try:
            inp = proc(images=rgb, input_boxes=box, return_tensors="pt").to(dev)
            with torch.no_grad():
                res = model(**inp, multimask_output=True)
            masks = proc.post_process_masks(res.pred_masks.cpu(), inp["original_sizes"])[0]
            m = np.asarray(masks).astype(bool)
            while m.ndim > 3:
                m = m[0]
            # THE MODEL'S OWN score picks the variant. Using the human mask here would score
            # SAM 2 on an oracle it does not have at inference time.
            sc = res.iou_scores.detach().cpu().numpy().reshape(-1)
            k = int(np.argmax(sc[:m.shape[0]])) if sc.size else 0
            chosen = m[k]
            # A variant that claims most of the window is the background object, not a crack.
            if chosen.mean() > 0.5:
                continue
            out[wy0:wy1, wx0:wx1] |= chosen
        except Exception:
            continue
        if log_every and (i + 1) % log_every == 0:
            print(f"      {i + 1}/{len(boxes)} prompts, {time.time() - t0:.0f}s",
                  flush=True)
    return out


def _boxes_from(labeled, labels):
    from scipy import ndimage as ndi
    boxes = []
    if not len(labels):
        return boxes
    objs = ndi.find_objects(labeled)
    for lb in labels:
        sl = objs[int(lb) - 1] if 0 < int(lb) <= len(objs) else None
        if sl is None:
            continue
        boxes.append((sl[0].start, sl[1].start, sl[0].stop, sl[1].stop))
    return boxes


def run(frames, model_id, max_candidates, shard=0, nshard=1):
    # PINNED, not inherited. run_unified_pipeline defaults to SAM 2 refinement, so an experiment that does not say which detector it wants silently measures a
    # different one than its output is labelled with. There is no default here on purpose.
    up.SAM2_MODE = DETECTOR
    base = _out_for(model_id)
    out_path = base if nshard == 1 else base.replace(".json", f".shard{shard}.json")
    rows = []
    if os.path.exists(out_path):
        try:
            rows = json.load(open(out_path)).get("per_image", [])
        except (OSError, ValueError):
            rows = []
    done = {r["image"] for r in rows}
    frames = [f for f in frames[shard::nshard] if f not in done]
    if done:
        print(f"  resuming: {len(done)} frame(s) already recorded", flush=True)
    if not frames:
        print("  nothing left to do")
        return rows
    proc, model, dev, torch = _load_sam2(model_id)
    print(f"  SAM 2: {model_id} on {dev}, {len(frames)} frame(s)"
          f"{f'  [shard {shard}/{nshard}]' if nshard > 1 else ''}\n", flush=True)
    for n in frames:
        with _without_human_input():
            st = up.run_unified_pipeline(n)
        lab, df, img8 = st["labeled"], st["df"], st["img8"]
        m = load_correction_mask(n, lab.shape)
        if m is None:
            print(f"  {n}: no mask"); continue
        crack, neg = (m == 1), (m == 2)
        if not (crack.any() and neg.any()):
            print(f"  {n}: not both classes"); continue

        acc = df.loc[df["IsCrack"], "Label"].tolist()
        alll = df["Label"].tolist()
        pipe = np.isin(lab, acc)
        print(f"  {n}  {len(alll)} candidates, {len(acc)} accepted, "
              f"{int(crack.sum()):,} crack px adjudicated", flush=True)

        def cap(xs):
            return xs[:max_candidates] if max_candidates else xs

        t = time.time()
        b_acc = cap(_boxes_from(lab, acc))
        s_ref = _sam2_mask_for_boxes(img8, b_acc, proc, model, dev, torch)
        t_ref = time.time() - t
        t = time.time()
        b_all = cap(_boxes_from(lab, alll))
        s_all = _sam2_mask_for_boxes(img8, b_all, proc, model, dev, torch)
        t_all = time.time() - t

        arms = {"pipeline": pipe, "sam2_refine": s_ref, "sam2_all": s_all,
                "hybrid_or": pipe | s_ref}
        rec = {"image": n, "model": model_id, "n_candidates": len(alll),
               "n_accepted": len(acc), "n_prompts_refine": len(b_acc),
               "n_prompts_all": len(b_all),
               "seconds_refine": t_ref, "seconds_all": t_all, "arms": {}}
        for k, pm in arms.items():
            s = _score(pm, crack, neg)
            s.update(_shape_cost(pm))
            rec["arms"][k] = s
            print(f"     {k:14s} f1 {s['f1']:.3f}  recall {s['recall']:.3f}  "
                  f"spec {s['specificity']:.3f}  prec {s['precision']:.3f}", flush=True)
        rows.append(rec)
        tmp = out_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"detector": _dc.stamp(DETECTOR), "per_image": rows}, fh, indent=1)
        os.replace(tmp, out_path)
    if rows:
        print(f"\n  MEANS over {len(rows)} frame(s)")
        print(f"  {'arm':14s} {'f1':>7s} {'recall':>7s} {'spec':>7s} {'prec':>7s}")
        for k in ("pipeline", "sam2_refine", "sam2_all", "hybrid_or"):
            f = lambda key: float(np.mean([r["arms"][k][key] for r in rows]))
            print(f"  {k:14s} {f('f1'):7.3f} {f('recall'):7.3f} {f('specificity'):7.3f} "
                  f"{f('precision'):7.3f}")
        print(f"  -> {out_path}")
    return rows


def report(model_id=None):
    import glob as _g
    base = _out_for(model_id) if model_id else OUT
    paths = sorted(_g.glob(base.replace(".json", ".shard*.json"))) or [base]
    rows = []
    for p in paths:
        try:
            rows += json.load(open(p))["per_image"]
        except (OSError, ValueError, KeyError):
            continue
    if not rows:
        print("no results")
        return None
    ARMS = ["pipeline", "sam2_refine", "sam2_all", "hybrid_or"]
    seen, uniq = set(), []
    for r in rows:
        if r["image"] not in seen:
            seen.add(r["image"]); uniq.append(r)
    print(f"{len(uniq)} frame(s), SAM 2 = "
          f"{sorted({r.get('model', '?') for r in uniq})}\n")
    print(f"  {'image':30s} " + " ".join(f"{a[:11]:>11s}" for a in ARMS) + "   (f1)")
    for r in sorted(uniq, key=lambda x: x["image"]):
        print(f"  {r['image'][:30]:30s} "
              + " ".join(f"{r['arms'][a]['f1']:11.3f}" for a in ARMS))
    print(f"\n  {'arm':14s} {'f1':>7s} {'recall':>7s} {'spec':>7s} {'prec':>7s} "
          f"{'pred px':>12s} {'compnts':>9s} {'skel px':>11s}")
    print(f"  (components and skeleton are the columns that showed refinement fragments the "
          f"mask; f1 alone made it look like a clear win)")
    means = {}
    for a in ARMS:
        f = lambda k: float(np.mean([r["arms"][a][k] for r in uniq]))
        means[a] = {k: f(k) for k in ("f1", "recall", "specificity", "precision")}
        for k in ("components", "skeleton_px"):
            if k in uniq[0]["arms"][a]:
                means[a][k] = f(k)
        print(f"  {a:14s} {f('f1'):7.3f} {f('recall'):7.3f} {f('specificity'):7.3f} "
              f"{f('precision'):7.3f} {f('predicted_px'):12,.0f}"
              + (f" {f('components'):9,.0f} {f('skeleton_px'):11,.0f}"
                 if "components" in means[a] else ""))
    best = max(ARMS, key=lambda a: means[a]["f1"])
    d = means[best]["f1"] - means["pipeline"]["f1"]
    print(f"\n  Best f1: {best} ({means[best]['f1']:.3f}), "
          f"{d:+.3f} against the shipped pipeline.")
    n_win = sum(1 for r in uniq
                if r["arms"]["hybrid_or"]["f1"] > r["arms"]["pipeline"]["f1"])
    print(f"  The hybrid beats the pipeline on {n_win} of {len(uniq)} frames.")
    print(f"  SAM 2 refinement trades recall for specificity: recall "
          f"{means['sam2_refine']['recall']:.3f} against "
          f"{means['pipeline']['recall']:.3f}, specificity "
          f"{means['sam2_refine']['specificity']:.3f} against "
          f"{means['pipeline']['specificity']:.3f}. Whether that is an improvement depends "
          f"on which error a study cannot afford, which is exactly the choice this repo "
          f"argues should be stated rather than defaulted.")
    json.dump({"n_frames": len(uniq), "means": means, "best_arm": best,
               "delta_f1_vs_pipeline": d, "hybrid_wins_on_n_frames": n_win,
               "per_image": uniq},
              open(base.replace(".json", "_report.json"), "w"), indent=1)
    print(f"\n  -> {base.replace('.json', '_report.json')}")
    return means


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", nargs="*")
    ap.add_argument("--model", default="facebook/sam2.1-hiera-tiny")
    ap.add_argument("--max-candidates", type=int, default=0,
                    help="cap prompts per arm; 0 means all")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report(a.model)
    else:
        run(a.frames or eligible(), a.model, a.max_candidates, a.shard, a.nshard)
