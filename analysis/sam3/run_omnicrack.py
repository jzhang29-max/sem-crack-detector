#!/usr/bin/env python3
"""Run OmniCrack30k's released nnU-Net on our 15 disjoint SEM tiles.

This is the honest way to answer "is our model the best": the published state of the art for
cross-material crack segmentation, run on OUR data under OUR protocol, instead of comparing our
IoU to their paper's IoU on their data.

OmniCrack30k, Benz & Rodehorst, CVPRW 2024 (10.1109/CVPRW63382.2024.00392): nnU-Net trained on
30,017 samples / 9.03 billion px / 20 subsets spanning asphalt, ceramic, concrete, masonry AND
steel, reporting "a mean clIoU4px of 64% outperforming all other approaches by at least 10%
points".

DEVIATIONS FROM THEIR INFERENCE SETTINGS, STATED BECAUSE THEY MATTER:
  * folds=(0,) instead of (0,1,2,4)  -- a single fold instead of a 4-fold ensemble
  * use_mirroring=False              -- no test-time augmentation
Both were forced by CPU-only inference. Both can only HURT their model, so whatever it scores
here is a LOWER BOUND on OmniCrack30k's performance on this data, and must be reported as one.

Our tiles are greyscale SEM; the model expects 3-channel. The grey is replicated to 3 channels,
which is the standard way to feed a mono image to an RGB-trained network, and is stated because
it is a choice.
"""
import os, sys, json, time
import numpy as np
from PIL import Image

SC = "/Users/jiamingzhang/Desktop/crack_export/analysis/sam3"
sys.path.insert(0, "/private/tmp/claude-501/-Users-jiamingzhang-Desktop-APP/48e14b5c-6bee-4570-a55e-3f87da7069da/scratchpad/omnicrack30k/src")
from omnicrack30k.inference import OmniCrack30kModel

def main():
    t0 = time.time()
    model = OmniCrack30kModel(folds=(0,), allow_tqdm=False)
    model.predictor.use_mirroring = False
    print(f"model ready in {time.time()-t0:.0f}s", flush=True)
    meta = json.load(open(f"{SC}/tiles/meta.json"))
    os.makedirs(f"{SC}/omnicrack", exist_ok=True)
    for i, m in enumerate(meta):
        t = m["tile"]
        g = np.array(Image.open(f"{SC}/tiles/{t}_gray.png").convert("L"))
        img = np.stack([g, g, g], -1)          # mono -> 3 channel
        te = time.time()
        soft, arg = model(img)
        p = soft[model.classes.index("crack")].numpy().astype(np.float16)
        np.savez_compressed(f"{SC}/omnicrack/{t}.npz", prob=p)
        print(f"  [{i+1}/{len(meta)}] {t[:38]:<40} crack-prob mean {p.mean():.4f} "
              f"max {p.max():.4f}  ({time.time()-te:.0f}s)", flush=True)
    print(f"\nwrote {len(meta)} probability maps to {SC}/omnicrack/  ({time.time()-t0:.0f}s)")

main()
