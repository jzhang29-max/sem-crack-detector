#!/usr/bin/env python3
"""SAM 3 (official Meta code, community-mirrored weights) on labelled SEM crack tiles.

PROVENANCE. facebook/sam3 is gated-manual and this account is NOT authorised (403
verified). Weights are the community mirror 1038lab/sam3 (`sam3.safetensors`, 1465
tensors, header-validated), used at the user's explicit direction. Model code is Meta's
official `sam3` package from GitHub. NOT citable until official access is granted.

ENVIRONMENT NOTES (all contained, nothing written to site-packages):
  * sam3.model.edt is pre-empted in sys.modules — it imports triton at module load for a
    kernel used only by the VIDEO tracker, and triton has no Apple-Silicon build.
  * torch tensor factories are patched during build only, because
    sam3/model/position_encoding.py:55 hardcodes device="cuda".

LEAK GATE. This script REFUSES TO RUN unless leak_check.py reports every tile clean. The
first version of this experiment fed the green channel of the annotated overlay -- pure red
(225,25,25) has green 0, so the label was written into the input as black pixels on 14 of 16
tiles. See LEAK_POSTMORTEM.md. The gate is not optional and not a warning.

PRESENCE CAPTURE. model.forward_grounding is wrapped to record, per (tile, prompt), the global
presence scalar s_i and max_j q_ij BEFORE gating. This makes the "empty output" behaviour
measurable rather than merely inferred: sam3_image_processor.py:195-200 computes
    out_probs = sigmoid(pred_logits) * sigmoid(presence_logit_dec);  keep = out_probs > tau
so ONE scalar per (image, prompt) rescales all num_queries=200 per-query scores before a
PER-QUERY threshold. It does NOT flip every instance together: survivors return 1 to 62 of 200.
What is all-or-nothing is only whether the image returns ANYTHING, via s_i * max_j q_ij.

Be careful what this instrumentation can prove. Because the wrapper reads the same tensors the
library multiplies and thresholds with the same constant, "n == 0 iff s_i*max_j q_ij <= tau" is
an ALGEBRAIC IDENTITY, not a prediction -- sigmoid(presence) > 0, so max_j(s*q_j) = s*max_j q_j.
There is no NMS, dedup or area filter after the gate (verified: no such call exists in
sam3_image_processor.py), so the instance COUNT is algebra too. Treat any agreement figure as an
instrumentation self-check. The EMPIRICAL content is the scalar's MAGNITUDE per prompt.

SCORING. Ground truth is the fine-stroke correction subset (median brush <=25 px) — the
only labels here that approximate an outline rather than a region assertion. They are
still not pixel-precise, so IoU/Dice are INDICATIVE. clDice is primary (it does not
punish a thin prediction against a thick label). An ORACLE arm reports the single
best-matching returned instance, chosen using the ground truth: a deliberate upper bound,
so a poor oracle result cannot be blamed on proposal ranking.
"""
import os, sys, json, time
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SC, "shim"))
import sam3_preload  # noqa: F401  (must precede sam3)
import numpy as np
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from skimage.morphology import skeletonize
from cuda_redirect import redirect_cuda, no_pin_memory
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

PROMPTS = ["crack", "a crack in metal", "thin dark line", "fracture"]
CONF = 0.3
SAVE_MASKS = os.environ.get("SAM3_SAVE_MASKS") == "1"   # dump union/oracle masks for figures
SAVE_SERD = os.environ.get("SAM3_SAVE_SERD") == "1"     # dump the pre-gate dense response


def cldice(pred, gt):
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0
    sp, sg = skeletonize(pred), skeletonize(gt)
    tp = (sp & gt).sum() / sp.sum() if sp.sum() else 0.0
    ts = (sg & pred).sum() / sg.sum() if sg.sum() else 0.0
    return float(2 * tp * ts / (tp + ts)) if (tp + ts) > 0 else 0.0


def sc(pred, gt):
    i = int((pred & gt).sum()); u = int((pred | gt).sum())
    return {"IoU": round(i / u, 4) if u else 0.0,
            "Dice": round(2 * i / (pred.sum() + gt.sum()), 4) if (pred.sum() + gt.sum()) else 0.0,
            "clDice": round(cldice(pred, gt), 4),
            "prec": round(i / pred.sum(), 4) if pred.sum() else 0.0,
            "rec": round(i / gt.sum(), 4) if gt.sum() else 0.0,
            "pred_frac": round(float(pred.mean()), 5)}


def gate():
    """Refuse to run on an input that encodes the label."""
    import leak_check
    rows, nl = leak_check.check("LEAK GATE -- model input", leak_check.new_gray)
    if nl:
        sys.exit(f"REFUSING TO RUN: {nl} tiles carry a written-in label. "
                 f"Regenerate with make_tiles.py (which reads the raw original), not from overlays.")
    print("gate passed: no tile encodes its label\n", flush=True)


def main():
    t0 = time.time()
    gate()
    if SAVE_MASKS:
        os.makedirs(f"{SC}/masks", exist_ok=True)
    with redirect_cuda("cpu"):
        model = build_sam3_image_model(checkpoint_path=f"{SC}/sam3_original.pt",
                                       load_from_HF=False, device="cpu")
    # the mirrored checkpoint is bfloat16; CPU matmul here needs a single dtype
    model = model.float()
    model.eval()
    proc = Sam3Processor(model, resolution=1008, device="cpu", confidence_threshold=CONF)

    # record the presence scalar and the top instance logit before gating
    presence = {}
    serd = {}
    cur = {"tile": None, "prompt": None}
    _orig_fg = model.forward_grounding

    def fg(*a, **k):
        out = _orig_fg(*a, **k)
        # SERD (arXiv:2607.12292): SAM 3's post-gate instance masks discard crack evidence that
        # survives in the dense prompt-conditioned response. That paper measures 82.66% internal
        # crack-pixel recall against 74.66% for the retained proposals over six public crack
        # datasets. Capture the dense field BEFORE the presence multiply and BEFORE the tau cut,
        # which is exactly the stage our own measurements showed the gate throwing away.
        if SAVE_SERD:
            try:
                import torch.nn.functional as F
                pm = out["pred_masks"].float()          # [Q, h, w] logits
                pl = out["pred_logits"].float().sigmoid().flatten()   # [Q]
                q = (pm.sigmoid() * pl[:, None, None]).max(0).values  # NO presence, NO keep
                q = F.interpolate(q[None, None], size=(1024, 1024), mode="bilinear",
                                  align_corners=False)[0, 0]
                q = q.detach().cpu().numpy()
                serd[f"{cur['tile']}|{cur['prompt']}"] = q.astype(np.float16)
            except Exception as e:
                print(f"    SERD capture failed: {type(e).__name__}: {str(e)[:90]}", flush=True)
        try:
            s_i = float(out["presence_logit_dec"].sigmoid().flatten()[0])
            q = out["pred_logits"].sigmoid().flatten()
            presence[f"{cur['tile']}|{cur['prompt']}"] = {
                "presence": round(s_i, 6), "max_q": round(float(q.max()), 6),
                "gated_max": round(s_i * float(q.max()), 6),
                "tau": CONF,
                "survives": bool(s_i * float(q.max()) > CONF)}
        except Exception:
            pass
        return out

    model.forward_grounding = fg
    print(f"model ready in {time.time()-t0:.0f}s", flush=True)

    meta = json.load(open(f"{SC}/tiles/meta.json"))
    rows = []
    for j, m in enumerate(meta):
        tid = m["tile"]
        img = Image.open(f"{SC}/tiles/{tid}_gray.png").convert("RGB")  # PIL: processor reads .size
        gt = np.array(Image.open(f"{SC}/tiles/{tid}_gt.png")) > 127
        te = time.time()
        try:
            with no_pin_memory(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                state = proc.set_image(img)
        except Exception as e:
            print(f"  [{j+1}/{len(meta)}] {tid}: set_image FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        enc = time.time() - te
        cur["tile"] = tid
        for p in PROMPTS:
            cur["prompt"] = p
            try:
                with no_pin_memory(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    st = proc.set_text_prompt(p, state)
                M = st["masks"]
                M = M.squeeze(1).float().cpu().numpy() > 0.5 if hasattr(M, "cpu") else np.asarray(M).astype(bool)
                if M.ndim == 2:
                    M = M[None]
                n = len(M)
                if n == 0:
                    r = {"tile": tid, "prompt": p, "n": 0}
                    r.update({f"union_{k}": v for k, v in sc(np.zeros_like(gt), gt).items()})
                    r.update({f"oracle_{k}": v for k, v in sc(np.zeros_like(gt), gt).items()})
                else:
                    union = M.any(0)
                    b = max(range(n), key=lambda k: (M[k] & gt).sum() / max((M[k] | gt).sum(), 1))
                    r = {"tile": tid, "prompt": p, "n": int(n)}
                    r.update({f"union_{k}": v for k, v in sc(union, gt).items()})
                    r.update({f"oracle_{k}": v for k, v in sc(M[b], gt).items()})
                if SAVE_MASKS and n:
                    np.savez_compressed(f"{SC}/masks/{tid}__{p.replace(' ','_')}.npz",
                                        union=M.any(0), oracle=M[b], n=np.int32(n))
                rows.append(r)
                print(f"  [{j+1}/{len(meta)}] {tid[:34]:<34} '{p[:16]:<16}' n={r['n']:<3} "
                      f"union IoU {r['union_IoU']:.3f} clD {r['union_clDice']:.3f} | "
                      f"oracle IoU {r['oracle_IoU']:.3f} clD {r['oracle_clDice']:.3f}", flush=True)
            except Exception as e:
                print(f"  [{j+1}/{len(meta)}] {tid[:34]:<34} '{p[:16]:<16}' ERR {type(e).__name__}: {str(e)[:110]}", flush=True)
                rows.append({"tile": tid, "prompt": p, "error": f"{type(e).__name__}: {str(e)[:200]}"})
        print(f"      (encode {enc:.1f}s)", flush=True)
        json.dump(rows, open(f"{SC}/sam3_results.json", "w"), indent=1)
    json.dump(rows, open(f"{SC}/sam3_results.json", "w"), indent=1)
    json.dump(presence, open(f"{SC}/sam3_presence.json", "w"), indent=1)
    if SAVE_SERD and serd:
        os.makedirs(f"{SC}/serd", exist_ok=True)
        for k, v in serd.items():
            t, pr = k.split("|", 1)
            np.savez_compressed(f"{SC}/serd/{t}__{pr.replace(' ', '_')}.npz", r=v)
        print(f"wrote {len(serd)} SERD response maps to {SC}/serd/")
    print(f"\nwrote {SC}/sam3_results.json  ({len(rows)} rows, {time.time()-t0:.0f}s total)")
    print(f"wrote {SC}/sam3_presence.json  ({len(presence)} (tile,prompt) presence records)")


main()
