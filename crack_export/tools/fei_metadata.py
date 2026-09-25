#!/usr/bin/env python3
"""Recover per-frame acquisition metadata from the FEI/Thermo block inside each TIFF.

Why this exists. The 2026-09-15 hydrogen batch shipped with a hand-written capture-notes
file (stage stops, a 3x3 grid map, "6144x4096, 3 us, 10 mm WD"). That file was lost when
the capture folder was removed. Everything in it -- and more -- is still inside the TIFFs
themselves: Thermo Scientific SEMs append an ini-style text block after the image data.

This matters beyond provenance. Two load-bearing claims in the analysis were previously
*inferred* from image agreement and are now *certified* by the instrument:

  1. "CBS_000N and ETD_000N are the same field imaged twice" was established from mask
     Jaccard (0.505 matched vs 0.024 mismatched). The metadata shows the two channels
     share StageX, StageY AND the acquisition timestamp to the second -- they are one
     scan read out on two detectors, not two scans of one place.
  2. Scale was read by OCR off the databar, which only 47 of the older originals carry.
     HorFieldsize gives it exactly, in metres, for all 80 frames of the new batch.

Only the 2026-09-15 batch retains the block; the 71 older originals were re-exported
without it. So this is a property of that batch, not of the corpus -- which is itself
worth recording, because it makes those 80 frames the metrologically strongest subset.

Writes analysis/fei_metadata_260915.csv. Read-only with respect to the TIFFs.
"""
import csv, glob, os, sys

_CE = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(_CE) != "crack_export" and os.path.dirname(_CE) != _CE:
    _CE = os.path.dirname(_CE)
REPO = os.path.dirname(_CE)

# The block is appended AFTER the pixel data, so it lives in the tail of the file. 400 kB is
# comfortably more than any block observed (~9 kB) and avoids reading 25 MB per frame.
TAIL = 400_000
FIELDS = ("StageX", "StageY", "StageZ", "StageR", "StageT", "WD", "HorFieldsize",
          "Dwell", "HV", "BeamCurrent", "SpotSize", "Date", "Time", "PixelWidth",
          "PixelHeight", "ResolutionX", "ResolutionY", "SystemType", "Software")


def fei_block(path, tail=TAIL):
    """Parse the trailing ini-style FEI block. Returns {} when the TIFF has none."""
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        fh.seek(max(0, size - tail))
        txt = fh.read().decode("latin-1")
    start = txt.find("[User]")
    if start < 0:
        start = txt.find("[System]")
    if start < 0:
        return {}
    out, section = {}, ""
    for line in txt[start:].splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
        elif "=" in line:
            k, _, v = line.partition("=")
            k = k.strip()
            # First writer wins: later sections repeat generic keys like "Name".
            out.setdefault(k, v.strip())
            out.setdefault(f"{section}.{k}", v.strip())
    return out


def main():
    originals = sorted(glob.glob(f"{REPO}/original/*.tif")) + \
                sorted(glob.glob(f"{REPO}/original/*.tiff"))
    if not originals:
        sys.exit(f"no originals under {REPO}/original -- nothing to read")

    rows, without = [], []
    for p in originals:
        d = fei_block(p)
        if not d:
            without.append(os.path.basename(p))
            continue
        r = {"frame": os.path.basename(p)}
        for f in FIELDS:
            r[f] = d.get(f, "")
        # nm/px, derived. HorFieldsize is metres; ResolutionX is the stored pixel width.
        try:
            r["nm_per_px"] = round(float(d["HorFieldsize"]) / float(d["ResolutionX"]) * 1e9, 4)
        except (KeyError, ValueError, ZeroDivisionError):
            r["nm_per_px"] = ""
        rows.append(r)

    out = f"{_CE}/analysis/fei_metadata_260915.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["frame"] + list(FIELDS) + ["nm_per_px"])
        w.writeheader()
        w.writerows(rows)

    print(f"originals scanned : {len(originals)}")
    print(f"  with FEI block  : {len(rows)}")
    print(f"  without         : {len(without)}  (older re-exports; databar OCR remains "
          f"the only scale source for these)")
    print(f"wrote {out}")
    return rows


if __name__ == "__main__":
    main()
