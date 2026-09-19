#!/usr/bin/env python3
"""Split the flat export into one folder per specimen set.

The set definition is not invented here -- it is the pipeline's own
aggregate.specimen_key(), so a set folder holds exactly the frames the
trainer treats as one specimen (siblings that must be held out together).

Files are HARD LINKED, not copied: the overlays alone are 860 MB and a copy
would double that for no benefit. Each set folder holds real files, and
editing one in place would edit the original -- so treat sets/ as read-only.
"""
import csv, os, re, sys, shutil
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_CODE = "/Users/jiamingzhang/Desktop/sem-crack-detector/interior_active_learning/code"
sys.path.insert(0, REPO_CODE)
from aggregate import parse_name, specimen_key   # the project's own taxonomy

OUT = os.path.join(ROOT, "sets")

def main():
    names = sorted(n[:-len("_mask.png")] for n in os.listdir(os.path.join(ROOT, "masks"))
                   if n.endswith("_mask.png"))
    groups = defaultdict(list)
    unparsed = []
    for n in names:
        k = specimen_key(n)
        if k is None:
            unparsed.append(n)
            continue
        groups[k].append(n)
    if unparsed:
        print(f"!! {len(unparsed)} frames the taxonomy cannot parse -> sets/_unparsed/")
        for n in unparsed:
            print("   ", n)
        groups["_unparsed"] = unparsed

    summary = {r["SourceImage"]: r for r in csv.DictReader(open(os.path.join(ROOT, "summary.csv")))}
    sum_cols = list(next(iter(summary.values())).keys())

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)          # rebuilt from scratch; contains only hard links
    linked = kept = 0
    manifest = []

    for key in sorted(groups):
        frames = sorted(groups[key])
        d = os.path.join(OUT, key)
        for sub in ("masks", "overlays", "regions"):
            os.makedirs(os.path.join(d, sub), exist_ok=True)
        pooled = []
        for n in frames:
            for sub, suffix in (("masks", "_mask.png"), ("overlays", "_overlay.png"),
                                ("regions", "_regions.csv")):
                src = os.path.join(ROOT, sub, n + suffix)
                dst = os.path.join(d, sub, n + suffix)
                if not os.path.exists(src):
                    print(f"!! missing {src}")
                    continue
                os.link(src, dst)
                linked += 1
            rp = os.path.join(ROOT, "regions", n + "_regions.csv")
            if os.path.exists(rp):
                for r in csv.DictReader(open(rp)):
                    pooled.append(dict(r, SourceImage=n))
        # per-set summary slice
        with open(os.path.join(d, "set_summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sum_cols)
            w.writeheader()
            for n in frames:
                if n in summary:
                    w.writerow(summary[n])
        # per-set pooled regions, with the frame each row came from
        if pooled:
            cols = ["SourceImage"] + [c for c in pooled[0] if c != "SourceImage"]
            with open(os.path.join(d, "regions_pooled.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                w.writerows(pooled)
        t = parse_name(frames[0])
        with open(os.path.join(d, "SET.txt"), "w") as f:
            f.write(f"set (aggregate.specimen_key): {key}\n")
            f.write(f"family: {t.get('family')}\n")
            for k2 in ("date", "alloy", "condition", "block", "project", "process", "exposure"):
                if t.get(k2):
                    f.write(f"{k2}: {t[k2]}\n")
            dets = sorted({parse_name(n).get("detector", "?") for n in frames})
            f.write(f"detectors present: {', '.join(dets)}\n")
            f.write(f"frames: {len(frames)}\n")
            for n in frames:
                f.write(f"  {n}\n")
        kept += len(frames)
        manifest.append((key, t.get("family"), len(frames), ",".join(dets)))

    with open(os.path.join(OUT, "SETS.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Set", "Family", "Frames", "Detectors"])
        w.writerows(manifest)

    print(f"\n{len(groups)} sets, {kept} frames, {linked} hard links")
    assert kept == len(names), f"frame accounting: {kept} != {len(names)}"
    for key, fam, n, dets in manifest:
        print(f"  {key:<22} {fam:<11} n={n:<3} [{dets}]")

main()
