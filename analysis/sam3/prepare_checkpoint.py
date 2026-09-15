#!/usr/bin/env python3
"""Regenerate sam3_original.pt, which run_real.py loads.

Meta's build_sam3_image_model calls torch.load, so the safetensors checkpoint has to be
re-saved as a .pt. That file is ~3.4 GB and is deleted after a run because it is fully
regenerable from the cached safetensors in a few seconds.

facebook/sam3 is gated-manual. If access has since been granted, prefer it:
    hf_hub_download("facebook/sam3", "<checkpoint>.safetensors")
and record that in the provenance, because only then is the result citable.
"""
import os, sys, glob, hashlib
import torch
from safetensors.torch import load_file

MIRROR_SHA = None  # 1038lab/sam3 sha256 was not published; size/header were validated
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam3_original.pt")


def find_checkpoint():
    pats = [os.path.expanduser("~/.cache/huggingface/hub/models--1038lab--sam3/snapshots/*/sam3.safetensors")]
    for p in pats:
        hits = glob.glob(p)
        if hits:
            return os.path.realpath(hits[0])
    return None


def main():
    ck = sys.argv[1] if len(sys.argv) > 1 else find_checkpoint()
    if not ck or not os.path.exists(ck):
        sys.exit("no checkpoint found; pass a path, or:\n"
                 "  HF_HUB_DISABLE_XET=1 python -c \""
                 "from huggingface_hub import hf_hub_download as d; d('1038lab/sam3','sam3.safetensors')\"")
    sd = load_file(ck)
    print(f"loaded {len(sd)} tensors from {ck}")
    n_det = sum(1 for k in sd if k.startswith("detector"))
    print(f"  detector.* keys: {n_det}  (build_sam3_image_model strips this prefix)")
    torch.save(sd, OUT)
    print(f"wrote {OUT}  {os.path.getsize(OUT)/1e9:.2f} GB")


main()
