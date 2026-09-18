#!/usr/bin/env python3
"""clIoU_tau -- the tolerant, width-insensitive metric for region-assertion labels.

Definition follows OmniCrack30k (Benz & Rodehorst, CVPRW 2024, 10.1109/CVPRW63382.2024.00392),
which introduces it precisely because hand annotation of thin structures is imprecise:

    S_T = skeleton(GT),  S_P = skeleton(prediction),  K_tau = disk(tau)
    TP_cl = S_T  AND  dilate(S_P, K_tau)
    FP_cl = S_P  \\  (S_P AND dilate(S_T, K_tau))
    FN_cl = S_T  \\  TP_cl
    clIoU_tau = |TP_cl| / (|TP_cl| + |FP_cl| + |FN_cl|)

Both sides are skeletonised and each is compared against the OTHER side dilated, which is what
makes it insensitive to the width of the annotation stroke -- the exact pathology here, where
the label brush is a median 16 px against a crack of order 3 px.

Also reported: PAR = |prediction| / |GT|, a bias diagnostic that separates "found the crack but
too fat" (PAR >> 1) from "missed it" (PAR << 1). Pixel IoU cannot tell those apart.

GUARD, and it matters: skeletonising a broad-brush GT returns the BRUSH's centreline, which is
not necessarily the crack's. So this metric is not a free pass -- it trades a width bias for a
centreline-placement assumption. Tau is a stated convention (sweep reported, tau=4 highlighted
as OmniCrack30k's empirical knee), never fitted per dataset, because fitting it would rebuild
the oracle this project has already mistaken for a result once.
"""
import os, json
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize, disk, binary_dilation

SC = os.path.dirname(os.path.abspath(__file__))
TAUS = [0, 1, 2, 4, 8, 16, 32, 64]


def cliou(pred, gt, tau):
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0
    sp, st = skeletonize(pred), skeletonize(gt)
    if sp.sum() == 0 or st.sum() == 0:
        return 0.0
    k = disk(tau) if tau > 0 else None
    dsp = binary_dilation(sp, k) if tau > 0 else sp
    dst = binary_dilation(st, k) if tau > 0 else st
    tp = int((st & dsp).sum())
    fp = int((sp & ~(sp & dst)).sum())
    fn = int((st & ~(st & dsp)).sum())
    d = tp + fp + fn
    return float(tp / d) if d else 0.0


def main():
    meta = json.load(open(f"{SC}/tiles/meta.json"))
    bench = json.load(open(f"{SC}/methods_bench.json"))
    tiles = bench["_tiles"]

    # predictions to score: SAM 3's saved masks, plus the LOFO-selected classical arms are not
    # re-derivable from the json, so we score SAM 3 and the simple reproducible baselines here.
    from skimage.filters import threshold_otsu, sato, meijering
    from skimage.morphology import remove_small_objects
    preds = {}
    for m in meta:
        t = m["tile"]
        g = np.array(Image.open(f"{SC}/tiles/{t}_gray.png").convert("L"))
        gt = np.array(Image.open(f"{SC}/tiles/{t}_gt.png")) > 127
        f = f"{SC}/masks/{t}__crack.npz"
        sam = np.load(f)["union"] if os.path.exists(f) else np.zeros_like(gt)
        gf = g.astype(np.float32) / 255.0
        sa = sato(gf, sigmas=np.arange(1, 5.0), black_ridges=True)
        me = meijering(gf, sigmas=np.arange(1, 5.0), black_ridges=True)
        preds[t] = {
            "SAM 3 union": sam,
            "Otsu": remove_small_objects(g <= threshold_otsu(g), max_size=32),
            "global thr q98": remove_small_objects(g <= np.percentile(g, 2), max_size=32),
            "Sato q98": remove_small_objects(sa >= np.percentile(sa, 98), max_size=32),
            "Meijering q98": remove_small_objects(me >= np.percentile(me, 98), max_size=32),
        }
    names = list(next(iter(preds.values())))

    print("clIoU_tau -- width-insensitive tolerant IoU (OmniCrack30k definition)\n")
    print(f"  {'method':<16} " + " ".join(f"{'t='+str(t):>7}" for t in TAUS) + f" {'PAR':>7}")
    out = {}
    for nm in names:
        row, pars = [], []
        for tau in TAUS:
            v = []
            for m in meta:
                t = m["tile"]
                gt = np.array(Image.open(f"{SC}/tiles/{t}_gt.png")) > 127
                v.append(cliou(preds[t][nm], gt, tau))
            row.append(float(np.median(v)))
        for m in meta:
            t = m["tile"]
            gt = np.array(Image.open(f"{SC}/tiles/{t}_gt.png")) > 127
            pars.append(preds[t][nm].sum() / max(gt.sum(), 1))
        print(f"  {nm:<16} " + " ".join(f"{x:>7.4f}" for x in row) + f" {np.median(pars):>7.2f}")
        out[nm] = {"cliou": dict(zip(map(str, TAUS), row)), "PAR": float(np.median(pars))}

    print("\n  Leader at each tolerance:")
    for i, tau in enumerate(TAUS):
        best = max(names, key=lambda n: out[n]["cliou"][str(tau)])
        print(f"    tau={tau:<3} {best:<16} {out[best]['cliou'][str(tau)]:.4f}")
    json.dump(out, open(f"{SC}/tolerance_sweep.json", "w"), indent=1)
    print("\nwrote tolerance_sweep.json")


if __name__ == "__main__":
    main()
