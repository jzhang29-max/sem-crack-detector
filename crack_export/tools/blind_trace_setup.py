#!/usr/bin/env python3
"""Build the blinded dual-channel tracing experiment.

WHAT THIS IS FOR. Every CBS-vs-ETD number in this repo is segmentation-derived, so the
standing objection is "your segmenter just responds differently to the two channels".
The gain-free dark-tail statistic answers that for the IMAGE, not for the MEASURAND. Only a
human tracing each channel, blind to which channel it is, separates "the detector changes
the image" from "the detector changes the crack you would measure".

Schmies 2023 used the two channels as model inputs; RODARE merges them at ratio 0.5;
Ayadi 2026 ships per-modality masks and never compares them. Nobody has annotated the two
channels independently and blind. That is the experiment this sets up.

DESIGN, fixed before any outcome was looked at:

  selection   Paired fields only. Index-0010 frames excluded -- they are half-resolution
              overviews of a different field of view, not grid frames. Of the rest, keep
              fields with <10% off-specimen background, then take the two per specimen cell
              with the LOWEST black clipping. Both criteria are image-quality properties;
              neither references the CBS/ETD difference under test. Yields 13 fields over
              7 cells (MAR_Amb_Cast contributes one -- only one of its five fields passes
              the off-specimen screen).

  blinding    Filenames are opaque ids. Channel, specimen, condition and field index appear
              nowhere in the annotator's folder. The key is written OUTSIDE that folder.

  display     Both channels get the SAME percentile-anchored stretch (1st/99th) to 8-bit.
              This does double duty: it removes the operator's per-channel gain -- CBS was
              acquired at Contrast=45.5 and ETD at 73.5, which is itself a channel cue --
              and it is the common histogram protocol a metrology referee will ask for.
              It is monotone per image, so it cannot create or destroy dark-tail ordering.

  order       Each annotator gets an independent random order, and the two channels of one
              field are forced at least MIN_GAP apart in the sequence so the second is not
              traced from memory of the first.

  annotators  Two, tracing independently. Inter-annotator agreement on the SAME image is
              the noise floor against which any between-channel difference is judged; a
              difference smaller than that floor means nothing.

OUTPUT is deliberately not committed: four of the seven cells are unreleased frames.

    python3 blind_trace_setup.py --out ~/Desktop/MAR_blind_tracing
"""
import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

SC = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(SC))
sys.path.insert(0, os.path.join(REPO, "code"))
from detect_cracks import load_as_uint8, find_field_of_view, off_specimen_mask  # noqa: E402

ANNOTATORS = ("A", "B")
MIN_GAP = 6          # minimum separation, in sequence positions, between a field's two channels
SEED = 20260923      # fixed so the whole package is reproducible from this file alone


def clip_frac(path):
    # find_field_of_view, not a fixed fraction. A crude a[:0.88*h] discards 246 rows on
    # these 4376-row frames -- 6.0% of the usable field -- because the databar is 280 rows,
    # not 526. Every other measurement in this repo uses the detector; so does this.
    a = load_as_uint8(path)
    x0, y0, x1, y1 = find_field_of_view(a)
    a = a[y0:y1, x0:x1]
    return float((a <= a.min() + 1e-6).mean())


def select_fields():
    pairs = {}
    for p in sorted(os.listdir(os.path.join(REPO, "original"))):
        m = re.match(r"(MAR_[A-Za-z]+_[A-Za-z]+)_(CBS|ETD)_(\d+)\.tif$", p)
        if m:
            pairs.setdefault((m.group(1), m.group(3)), {})[m.group(2)] = \
                os.path.join(REPO, "original", p)
    both = {k: v for k, v in pairs.items() if len(v) == 2 and k[1] != "0010"}
    rows = []
    for (cell, idx), v in sorted(both.items()):
        g = load_as_uint8(v["CBS"])
        x0, y0, x1, y1 = find_field_of_view(g)
        rows.append({"cell": cell, "idx": idx,
                     "clip": max(clip_frac(v["CBS"]), clip_frac(v["ETD"])),
                     "off": float(off_specimen_mask(g[y0:y1, x0:x1]).mean()),
                     "CBS": v["CBS"], "ETD": v["ETD"]})
    elig = [r for r in rows if r["off"] < 0.10]
    out = []
    for c in sorted({r["cell"] for r in rows}):
        out += sorted([r for r in elig if r["cell"] == c],
                      key=lambda r: (r["clip"], r["idx"]))[:2]
    return out


def to_display(path):
    """8-bit, common 1st/99th-percentile stretch over the detected field of view.

    Monotone per image, so it cannot create or destroy dark-tail ordering. The crop is
    find_field_of_view's, not a fixed fraction: 0.88*h threw away 246 rows of real specimen
    on every frame and computed the stretch over a different region than the rest of the
    pipeline measures.
    """
    g = load_as_uint8(path)
    x0, y0, x1, y1 = find_field_of_view(g)
    a = g[y0:y1, x0:x1].astype(np.float32)
    lo, hi = np.percentile(a, [1.0, 99.0])
    return Image.fromarray(np.clip((a - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8))


def opaque_id(cell, idx, det):
    h = hashlib.sha256(f"{SEED}|{cell}|{idx}|{det}".encode()).hexdigest()[:10]
    return f"F{h}"


def spaced_order(items, key_of_field, rng, min_gap):
    """Shuffle so the two entries sharing a field are >= min_gap apart. Falls back to the
    best of many tries rather than looping forever on a short list."""
    best, best_gap = None, -1
    for _ in range(4000):
        cand = items[:]
        rng.shuffle(cand)
        pos = {}
        gap = 10 ** 6
        for i, it in enumerate(cand):
            f = key_of_field(it)
            if f in pos:
                gap = min(gap, i - pos[f])
            pos[f] = i
        if gap > best_gap:
            best, best_gap = cand, gap
        if gap >= min_gap:
            return cand, gap
    return best, best_gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = os.path.expanduser(a.out)
    if os.path.exists(out) and os.listdir(out):
        sys.exit(f"{out} exists and is not empty. Refusing to overwrite a tracing package "
                 f"that may already contain work; move it aside or choose another path.")

    sel = select_fields()
    units = [(r, d) for r in sel for d in ("CBS", "ETD")]
    print(f"{len(sel)} fields, {len(units)} images per annotator", flush=True)

    keyrows = []
    shared = os.path.join(out, "_images")
    os.makedirs(shared, exist_ok=True)
    for r, det in units:
        oid = opaque_id(r["cell"], r["idx"], det)
        to_display(r[det]).save(os.path.join(shared, f"{oid}.png"))
        keyrows.append({"opaque_id": oid, "cell": r["cell"], "field": r["idx"],
                        "detector": det, "source": os.path.relpath(r[det], REPO),
                        "clip_frac": round(r["clip"], 5), "off_specimen_frac": round(r["off"], 5)})
    rng = random.Random(SEED)
    orders = {}
    for ann in ANNOTATORS:
        seq, gap = spaced_order(units, lambda u: (u[0]["cell"], u[0]["idx"]), rng, MIN_GAP)
        d = os.path.join(out, f"annotator_{ann}")
        os.makedirs(os.path.join(d, "traced"), exist_ok=True)
        rows = []
        for i, (r, det) in enumerate(seq, 1):
            oid = opaque_id(r["cell"], r["idx"], det)
            rows.append({"position": i, "image": f"{oid}.png"})
            os.link(os.path.join(shared, f"{oid}.png"), os.path.join(d, f"{i:02d}_{oid}.png"))
        with open(os.path.join(d, "ORDER.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["position", "image"]); w.writeheader(); w.writerows(rows)
        orders[ann] = {"min_gap_achieved": gap, "n": len(seq)}
        print(f"  annotator {ann}: {len(seq)} images, channels of a field >= {gap} apart")

    keydir = os.path.join(REPO, "crack_export", "analysis", "blind_trace_key")
    os.makedirs(keydir, exist_ok=True)
    kp = os.path.join(keydir, "KEY_do_not_open_until_tracing_complete.csv")
    with open(kp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(keyrows[0])); w.writeheader(); w.writerows(keyrows)
    json.dump({"seed": SEED, "min_gap": MIN_GAP, "annotators": list(ANNOTATORS),
               "n_fields": len(sel), "n_images": len(units), "orders": orders},
              open(os.path.join(keydir, "design.json"), "w"), indent=1)
    print(f"\nkey written OUTSIDE the annotator folders: {kp}")
    print(f"images:  {out}")


if __name__ == "__main__":
    main()
