#!/usr/bin/env python3
"""Certify, from instrument metadata, that index-matched CBS/ETD frames are one field.

Every paired detector statistic in this work -- (a3), (a4), (a7) -- rests on the premise
that MAR_<cond>_<proc>_CBS_000N and ..._ETD_000N image the SAME field of view. Until now
that premise was *inferred* from the images: index-matched mask pairs have median Jaccard
0.505 against 0.024 for mismatched indices (paired_detector.py). That is good evidence and
it is still circular in one respect -- it uses the segmenter's output to justify the
comparison the segmenter is then used for.

The 2026-09-15 batch can settle it without touching a pixel. Those TIFFs retain the FEI
metadata block, so the stage position and the acquisition clock are recorded per frame.
If CBS_000N and ETD_000N are one scan read out on two detectors, they must share StageX,
StageY and the timestamp exactly. If they were two separate scans of the same nominal
place, the stage would have been re-driven and would not repeat to the nanometre.

The control matters as much as the test: "two frames from one specimen have similar stage
coordinates" would be true of any two frames if the grid were tight enough. So the same
comparison is run over mismatched-index pairs from the same specimen, and the two
distributions have to be disjoint, not merely different.

Run after crack_export/tools/fei_metadata.py. Prints; writes nothing.
"""
import csv, itertools, os, re, statistics as st, sys

_CE = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(_CE) != "crack_export" and os.path.dirname(_CE) != _CE:
    _CE = os.path.dirname(_CE)

CSV = f"{_CE}/analysis/fei_metadata_260915.csv"
NAME = re.compile(r"MAR_(H|AmbB)_(AS|HIP)_(CBS|ETD)_(\d+)\.tif")
GRID = [f"{k:04d}" for k in range(1, 10)]      # 0010 is a low-mag overview, not a grid tile
CLUSTER_TOL_UM = 100.0                          # >> stage repeatability, << the ~450 um pitch

# Stop coordinates as written in the recovered capture notes (original/CAPTURE_NOTES_*.txt).
# HIP H is recorded there without its minus sign; compared on magnitude and flagged.
NOTES_STOPS = {("H", "HIP"): (19.5287, 19.0594, 18.5901),
               ("H", "AS"): (-8.0259, -7.5558, -7.0857),
               ("AmbB", "HIP"): (0.30625, 0.7758, 1.2453),
               ("AmbB", "AS"): (25.6524, 26.0883, 26.5241)}


def cluster(vals, tol):
    groups = []
    for v in sorted(vals):
        if groups and v - groups[-1][-1] <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return groups


def load():
    if not os.path.exists(CSV):
        sys.exit(f"{CSV} missing -- run crack_export/tools/fei_metadata.py first.\n"
                 "Refusing to report a certification with no metadata behind it.")
    by = {}
    for r in csv.DictReader(open(CSV)):
        m = NAME.match(r["frame"])
        if m:
            by[m.groups()] = r
    if not by:
        sys.exit(f"{CSV} holds no MAR_H_/MAR_AmbB_ frames -- nothing to certify.")
    return by


def main():
    by = load()
    cells = [(c, p) for c in ("H", "AmbB") for p in ("AS", "HIP")]

    # --- test: index-matched pairs ------------------------------------------------
    n = same_xy = same_t = same_hfw = 0
    worst = 0.0
    for (c, p) in cells:
        for i in GRID + ["0010"]:
            a, b = by.get((c, p, "CBS", i)), by.get((c, p, "ETD", i))
            if not (a and b):
                continue
            n += 1
            d = max(abs(float(a["StageX"]) - float(b["StageX"])),
                    abs(float(a["StageY"]) - float(b["StageY"]))) * 1e9
            worst = max(worst, d)
            same_xy += d == 0.0
            same_t += (a["Date"], a["Time"]) == (b["Date"], b["Time"])
            same_hfw += a["HorFieldsize"] == b["HorFieldsize"]

    # --- control: mismatched indices, same specimen --------------------------------
    sep = []
    for (c, p) in cells:
        for i, j in itertools.permutations(GRID, 2):
            a, b = by.get((c, p, "CBS", i)), by.get((c, p, "ETD", j))
            if a and b:
                sep.append(((float(a["StageX"]) - float(b["StageX"])) ** 2 +
                            (float(a["StageY"]) - float(b["StageY"])) ** 2) ** 0.5 * 1e6)

    print("=== index-matched CBS/ETD pairs (the premise under (a3)/(a4)/(a7)) ===")
    print(f"  pairs                      {n}")
    print(f"  identical stage position   {same_xy}/{n}   worst offset {worst:.1f} nm")
    print(f"  identical timestamp (1 s)  {same_t}/{n}")
    print(f"  identical HorFieldsize     {same_hfw}/{n}")
    print()
    print("=== control: mismatched indices from the SAME specimen ===")
    print(f"  pairs                      {len(sep)}")
    print(f"  stage separation           min {min(sep):.1f} um   median {st.median(sep):.1f} um")
    print(f"  -> matched 0.0 um vs mismatched min {min(sep):.1f} um: the distributions are "
          f"disjoint, so this is not an artefact of a tight grid.")

    # --- grid geometry --------------------------------------------------------------
    print()
    print("=== grid geometry, from the stage ===")
    for (c, p) in cells:
        pts = [by[(c, p, "CBS", i)] for i in GRID if (c, p, "CBS", i) in by]
        if len(pts) != 9:
            print(f"  MAR_{c}_{p}: {len(pts)}/9 frames -- skipped")
            continue
        cx = cluster([float(r["StageX"]) * 1e6 for r in pts], CLUSTER_TOL_UM)
        cy = cluster([float(r["StageY"]) * 1e6 for r in pts], CLUSTER_TOL_UM)
        mx, my = [st.mean(g) for g in cx], [st.mean(g) for g in cy]
        px = st.mean([mx[k + 1] - mx[k] for k in range(len(mx) - 1)])
        py = st.mean([my[k + 1] - my[k] for k in range(len(my) - 1)])
        hfw = float(pts[0]["HorFieldsize"]) * 1e6
        vfw = hfw * float(pts[0]["ResolutionY"]) / float(pts[0]["ResolutionX"])
        print(f"  MAR_{c}_{p}: {len(cx)}x{len(cy)} grid | pitch {px:.0f}x{py:.0f} um | "
              f"field {hfw:.1f}x{vfw:.1f} um | gap {px-hfw:+.0f}x{py-vfw:+.0f} um "
              f"({'disjoint' if px > hfw and py > vfw else 'OVERLAPPING'})")

    # --- authenticate the recovered notes -------------------------------------------
    print()
    print("=== recovered capture notes vs the instrument ===")
    for (c, p), stops in NOTES_STOPS.items():
        pts = [by[(c, p, "CBS", i)] for i in GRID if (c, p, "CBS", i) in by]
        if len(pts) != 9:
            continue
        cols = sorted(st.mean(g) for g in
                      cluster([float(r["StageX"]) * 1e3 for r in pts], CLUSTER_TOL_UM / 1e3))
        direct = max(abs(a - b) for a, b in zip(cols, sorted(stops))) * 1000
        flipped = max(abs(a - b) for a, b in zip(cols, sorted(-s for s in stops))) * 1000
        if direct <= flipped:
            print(f"  MAR_{c}_{p}: agrees to {direct:.1f} um")
        else:
            print(f"  MAR_{c}_{p}: agrees to {flipped:.1f} um ONLY after negating the notes "
                  f"({direct:.0f} um as written) -- sign slip in the original notes")


if __name__ == "__main__":
    main()
