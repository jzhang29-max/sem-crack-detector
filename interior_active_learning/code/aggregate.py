"""Specimen- and condition-level statistics across images.

WHY THIS EXISTS
Everything else in this project stops at one frame. But nobody asks "how long is the
crack in this micrograph" -- they ask "does the HIP condition crack less than the cast
condition", which is a question about a POPULATION of frames with a spread, not a single
number. Without this the researcher exports 62 per-image CSVs and does the grouping by
hand in a spreadsheet, which is where transcription errors and silently-mixed units enter
a manuscript.

THE RULE THAT MATTERS: NEVER MIX UNITS
A mean crack length computed over a group where some images are calibrated and some are
not is meaningless -- it averages micrometres with pixels. Every physical statistic here
is computed over the CALIBRATED SUBSET ONLY, and the record says how many images were
excluded and why. A group with no calibrated images gets pixel statistics and an explicit
`units: "px"`, never a physical number.

This is the same principle as the correction masks (0 means UNREVIEWED, not "not a
crack") and the calibration store (uncalibrated is None, never 1.0): a silent default is
indistinguishable from a real measurement.

DISPERSION, NOT JUST MEANS
Each statistic carries n, mean, sd, median and IQR. A mean crack length with no spread
cannot support a claim that two conditions differ, and quoting one is how a plot ends up
with no error bars.

GROUPING
Specimen identity lives in the filenames, and this dataset has three naming families:

  260622_316_H_b2_front_CBS_01   date, alloy 316, condition H, block b2, face, detector
  MAR_Amb_Cast_CBS_0002          project MAR, condition Amb, process Cast, detector
  AS_24hr_BSE_Side_008           condition AS, exposure 24hr, detector BSE, view Side

parse_name() extracts what it can and labels the family; anything it cannot parse is
reported under family "unparsed" rather than being forced into a bucket, because a
mis-grouped specimen is worse than an ungrouped one. Callers can also pass an explicit
mapping and skip parsing entirely.
"""
import csv
import json
import os
import re
import statistics as stats

import calibration as cal
from common import EXP_ROOT

MEAS_DIR = os.path.join(EXP_ROOT, "measurements")
OUT_DIR = os.path.join(EXP_ROOT, "measurements")

#: Statistics are reported for these columns. Physical variants are substituted
#: automatically when the group is calibrated.
METRICS = ["SkeletonLength_px", "MaxWidth_px", "MeanWidth_px", "Area_px", "Tortuosity"]


def parse_name(name):
    """Best-effort specimen tokens from a filename. Never guesses across families."""
    # 260622_316_H_b2_front_CBS_01 / 260708_316_H_b2_front_CBS_014
    # Detectors are named explicitly, longest first. A generic [A-Z]{2,3} was greedy in
    # the wrong direction: it split "..._amb_b3_CBS_02" into face="C", detector="BS",
    # which would have silently mis-grouped every frame with no face token.
    m = re.match(r"^(\d{6})_(\d{3}[A-Za-z]?)_([A-Za-z]+)_(b\d+|amb_b\d+)_"
                 r"(?:([A-Za-z]+)_)?(CBS|BSE|ETD|TLD|SED|SE)_(\d+)$", name)
    if m:
        return {"family": "steel", "date": m.group(1), "alloy": m.group(2),
                "condition": m.group(3), "block": m.group(4),
                "face": m.group(5) or "", "detector": m.group(6), "index": m.group(7)}
    # MAR_Amb_Cast_CBS_0002 / MAR_Amb_HIP_ETD_0010
    m = re.match(r"^MAR_([A-Za-z]+)_([A-Za-z]+)_(CBS|BSE|ETD|TLD|SED|SE)_(\d+)$", name)
    if m:
        return {"family": "superalloy", "project": "MAR", "condition": m.group(1),
                "process": m.group(2), "detector": m.group(3), "index": m.group(4)}
    # AS_24hr_BSE_Side_008 / HIP_24hr_SE_Side_006 / Cast_24hr_SE_Side_006
    m = re.match(r"^([A-Za-z]+)_(\d+hr)_(CBS|BSE|ETD|TLD|SED|SE)_([A-Za-z]+)_(\d+)$", name)
    if m:
        return {"family": "exposure", "process": m.group(1), "exposure": m.group(2),
                "detector": m.group(3), "view": m.group(4), "index": m.group(5)}
    return {"family": "unparsed"}


def specimen_key(name):
    """The specimen/session an image belongs to. Sibling frames share it.

    Used to hold out a whole specimen rather than a single frame. Leave-one-IMAGE-out over
    frames from one session is near-duplicate leakage: the model sees 14 other views of the
    same block and is then asked to generalise to the fifteenth, which it can do without
    generalising to a new specimen at all.

    Measured on this corpus: the 38 labelled images come from 8 specimens, and one of them
    (260708_316_H_b2) supplies 15 frames. Holding out a single frame from that session
    leaves 14 siblings in training.
    """
    t = parse_name(name)
    fam = t.get("family")
    if fam == "steel":
        return f"{t['date']}_{t['alloy']}_{t['condition']}_{t['block']}"
    if fam == "superalloy":
        return f"{t['project']}_{t['condition']}_{t['process']}"
    if fam == "exposure":
        return f"{t['process']}_{t['exposure']}"
    return f"unparsed:{name}"


def group_key(name, by):
    """(key, None) for a groupable image, or (None, reason) explaining why not.

    Two different situations were both reported as "unparsable_name", which sent the reader
    looking for a filename problem that did not exist: a name this module genuinely cannot
    read, versus a perfectly parsed name whose FAMILY has no such token -- an exposure-family
    frame has no 'condition', so grouping by condition silently dropped every one of them.
    The default CLI grouping is family,condition, so that was the common case.
    """
    t = parse_name(name)
    if t["family"] == "unparsed":
        return None, "unparsable_name"
    absent = [k for k in by if k not in t]
    if absent:
        return None, f"family {t['family']!r} has no token(s): {', '.join(absent)}"
    return tuple(str(t[k]) for k in by), None


def _describe(values):
    """n, mean, sd, median and IQR. sd is None for n < 2 -- not 0, which would read as
    'no spread' rather than 'not enough data to have one'."""
    vs = sorted(float(v) for v in values
                if isinstance(v, (int, float)) or (isinstance(v, str) and v.strip() != ""))
    if not vs:
        return {"n": 0}
    out = {"n": len(vs), "mean": stats.fmean(vs), "median": stats.median(vs),
           "min": vs[0], "max": vs[-1],
           "sd": (stats.stdev(vs) if len(vs) > 1 else None)}
    if len(vs) >= 4:
        q = stats.quantiles(vs, n=4)
        out["q1"], out["q3"], out["iqr"] = q[0], q[2], q[2] - q[0]
    return out


def _read_rows(image_name):
    """Rows from the image's CSV, with micrometre columns derived on read if calibrated.

    The CSV is a snapshot: one written while the image was uncalibrated has no _um
    columns, and calibrating afterwards does not go back and edit it. Deriving here means
    the CALIBRATION is authoritative rather than whatever the file happened to contain
    when it was written -- otherwise a freshly calibrated image contributes nothing to a
    physical statistic, or worse, contributes pixel values through a fallback that still
    labels the group "um".
    """
    p = os.path.join(MEAS_DIR, f"{image_name}_crack_measurements.csv")
    if not os.path.exists(p):
        return None
    with open(p, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in list(r.items()):
            if k in ("SourceImage",):
                continue
            try:
                r[k] = float(v) if v not in ("", None) else ""
            except (TypeError, ValueError):
                pass
    umpx = cal.get_um_per_px(image_name)
    if umpx:
        rows = [cal.convert_row(r, umpx) for r in rows]
    return rows


def aggregate(images, by=("condition",), require_calibrated=True):
    """Group `images` by parsed tokens and summarise each group.

    require_calibrated=True computes physical statistics over the calibrated subset and
    records how many images were dropped. Set it False to force pixel statistics over
    everything, which is honest only if nothing physical is claimed from the result.
    """
    groups = {}
    skipped = {"no_measurements": [], "unparsable_name": [],
               "missing_group_token": [], "uncalibrated": []}

    for name in images:
        rows = _read_rows(name)
        if rows is None:
            skipped["no_measurements"].append(name)
            continue
        key, why = group_key(name, by)
        if key is None:
            bucket = ("unparsable_name" if why == "unparsable_name"
                      else "missing_group_token")
            skipped.setdefault(bucket, []).append(f"{name} ({why})"
                                                  if bucket != "unparsable_name" else name)
            continue
        umpx = cal.get_um_per_px(name)
        if require_calibrated and not umpx:
            skipped["uncalibrated"].append(name)
            continue
        g = groups.setdefault(key, {"images": [], "rows": [], "calibrated": [],
                                    "uncalibrated": []})
        g["images"].append(name)
        g["rows"].extend(rows)
        (g["calibrated"] if umpx else g["uncalibrated"]).append(name)

    out = []
    for key, g in sorted(groups.items()):
        physical = bool(g["calibrated"]) and not g["uncalibrated"]
        rec = {
            "group": dict(zip(by, key)),
            "n_images": len(g["images"]),
            "n_cracks": len(g["rows"]),
            "images": sorted(g["images"]),
            # Units are a property of the GROUP, not of the caller's intent. A group
            # holding even one uncalibrated image cannot report micrometres.
            "units": "um" if physical else "px",
            "n_calibrated": len(g["calibrated"]),
            "n_uncalibrated": len(g["uncalibrated"]),
            "metrics": {},
        }
        if g["uncalibrated"] and g["calibrated"]:
            rec["mixed_calibration_warning"] = (
                f"{len(g['uncalibrated'])} of {len(g['images'])} images in this group are "
                f"uncalibrated, so statistics are reported in PIXELS. Calibrate them, or "
                f"split the group, before quoting a physical dimension.")
        for m in METRICS:
            physical_col = m in cal.LENGTH_POWERS
            col = cal.um_column_name(m) if (physical and physical_col) else m
            vals = [r.get(col, "") for r in g["rows"]]
            if physical and physical_col and not any(
                    isinstance(v, (int, float)) for v in vals):
                # Do NOT fall back to the pixel column here. That would report pixel
                # numbers under a micrometre name -- the exact unit-mixing this module
                # exists to prevent. Report the metric as absent instead.
                rec["metrics"][col] = {"n": 0, "note": "no calibrated values available"}
                continue
            rec["metrics"][col] = _describe(vals)
        # Per-image maxima, because "the longest crack per frame" is the quantity a
        # fatigue study reports, and its spread across frames is the interesting part --
        # pooling every crack from every frame buries it.
        per_image_max = []
        for name in g["images"]:
            rr = _read_rows(name) or []
            lcol = ("SkeletonLength_um" if physical else "SkeletonLength_px")
            vals = [r.get(lcol) for r in rr
                    if isinstance(r.get(lcol), (int, float))]
            if vals:
                per_image_max.append(max(vals))
        rec["longest_crack_per_image"] = _describe(per_image_max)
        out.append(rec)

    return {"grouped_by": list(by), "groups": out, "skipped": skipped,
            "tool_version": __import__("common").VERSION}


def write_report(result, stem="aggregate"):
    os.makedirs(OUT_DIR, exist_ok=True)
    jp = os.path.join(OUT_DIR, f"{stem}.json")
    with open(jp, "w") as fh:
        json.dump(result, fh, indent=1)

    cp = os.path.join(OUT_DIR, f"{stem}.csv")
    fields = ["group", "units", "n_images", "n_cracks", "n_calibrated", "n_uncalibrated",
              "longest_crack_mean", "longest_crack_sd", "longest_crack_median",
              "longest_crack_n"]
    with open(cp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for g in result["groups"]:
            lc = g["longest_crack_per_image"]
            w.writerow({
                "group": "|".join(f"{k}={v}" for k, v in g["group"].items()),
                "units": g["units"], "n_images": g["n_images"],
                "n_cracks": g["n_cracks"], "n_calibrated": g["n_calibrated"],
                "n_uncalibrated": g["n_uncalibrated"],
                "longest_crack_mean": lc.get("mean"), "longest_crack_sd": lc.get("sd"),
                "longest_crack_median": lc.get("median"), "longest_crack_n": lc.get("n"),
            })
    return jp, cp


if __name__ == "__main__":
    import sys
    from crack_measurements import all_images

    by = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else ("family", "condition")
    res = aggregate(all_images(), by=by, require_calibrated=False)
    jp, cp = write_report(res)
    print(f"grouped by {by}: {len(res['groups'])} group(s)")
    for g in res["groups"]:
        lc = g["longest_crack_per_image"]
        sd = f" +/- {lc['sd']:.1f}" if lc.get("sd") is not None else ""
        mean = f"{lc['mean']:.1f}{sd}" if lc.get("mean") is not None else "n/a"
        print(f"  {'|'.join(f'{k}={v}' for k, v in g['group'].items()):38s} "
              f"{g['n_images']:3d} images  {g['n_cracks']:5d} cracks  "
              f"longest/frame {mean} {g['units']}")
        if g.get("mixed_calibration_warning"):
            print(f"      WARNING {g['mixed_calibration_warning']}")
    for why, names in res["skipped"].items():
        if names:
            print(f"  skipped ({why}): {len(names)}")
    print(f"-> {jp}\n-> {cp}")
