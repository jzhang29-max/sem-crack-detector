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

import numpy as np

import calibration as cal
from common import MEASUREMENTS_DIR

#: One directory, named once. There were two constants here -- MEAS_DIR for reading and
#: OUT_DIR for writing -- pointing at the same place by coincidence. When OUT_DIR learned
#: to follow SEMCRACK_MEASUREMENTS_DIR and MEAS_DIR did not, a batch run wrote its
#: aggregate into the batch directory while reading the repository's CSVs, so it produced
#: a header-only file and reported success. Reading and writing must not be able to
#: disagree about where the measurements are.
OUT_DIR = MEAS_DIR = MEASUREMENTS_DIR

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
    # None, not a per-frame sentinel. This used to return f"unparsed:{name}", which is
    # UNIQUE PER FRAME -- so for any filename this module cannot parse, every frame became
    # its own "specimen": n_specimens equalled n_frames, dispersion_is_estimable flipped to
    # True at three frames, and the sd columns that the pseudo-replication guard exists to
    # blank were written after all. The guard inverted for exactly the users who need it,
    # since a stranger's filenames never match this corpus's convention.
    return None


def group_key(name, by):
    """(key, None) for a groupable image, or (None, reason) explaining why not.

    Two different situations were both reported as "unparsable_name", which sent the reader
    looking for a filename problem that did not exist: a name this module genuinely cannot
    read, versus a perfectly parsed name whose FAMILY has no such token -- an exposure-family
    frame has no 'condition', so grouping by condition silently dropped every one of them.
    The default CLI grouping is family,condition, so that was the common case.
    """
    # Grouping by nothing means ONE group, and that must not depend on the filename. A
    # stranger running the batch CLI on their own files gets names this module cannot parse,
    # and every one of them was dropped -- so the aggregate came out empty for everybody
    # outside this corpus's naming convention, with no way to ask for a single pooled group.
    if not by:
        return (), None
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


def _fmt_mean(d):
    """Format a _describe() mean, or 'n/a'."""
    v = (d or {}).get("mean")
    return f"{v:.1f}" if isinstance(v, (int, float)) else "n/a"


def _is_censored(row):
    """True / False / None, where None means the CSV predates the censoring column.

    Returning False for a missing column would be the same mistake this whole project is
    about: treating "not recorded" as "not the case". A CSV written before censoring was
    tracked carries no evidence either way, and 49 such files sit in this corpus -- reading
    them as uncensored made only 2 of 23 frames look censored while both spot-checked frames
    had a censored LONGEST crack. Unknown must propagate, not default.
    """
    v = row.get("LengthIsCensored", row.get("TouchesBoundary"))
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return None
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return bool(v)


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
        # LONGEST CRACK PER FRAME, WITH CENSORING SEPARATED.
        #
        # A crack touching the frame edge continues outside it, so its length is a lower
        # bound. The bias is not random: the longest cracks are the most likely to be
        # censored, so the statistic most damaged is exactly the one a fatigue study reports.
        # Both are computed -- uncensored-only, which is interpretable but biased downward by
        # excluding the big ones, and all-cracks, which mixes bounds with measurements -- and
        # the count of censored frames is carried so neither can be quoted without it.
        per_image_max, per_image_max_uncens = [], []
        n_cens_frames = n_unknown_frames = 0
        for name in g["images"]:
            rr = _read_rows(name) or []
            lcol = ("SkeletonLength_um" if physical else "SkeletonLength_px")
            vals = [r.get(lcol) for r in rr if isinstance(r.get(lcol), (int, float))]
            flags = [_is_censored(r) for r in rr
                     if isinstance(r.get(lcol), (int, float))]
            unc = [v for v, c in zip(vals, flags) if c is False]
            if vals:
                per_image_max.append(max(vals))
            if unc:
                per_image_max_uncens.append(max(unc))
            if any(c is True for c in flags):
                n_cens_frames += 1
            if any(c is None for c in flags):
                # No censoring column: this frame's CSV predates the flag. Re-measure it.
                n_unknown_frames += 1
        rec["frames_with_unknown_censoring"] = n_unknown_frames
        # THE STATISTICAL UNIT.
        #
        # Pooling every crack from every frame reports n in the thousands while the
        # independent units are specimens, of which this corpus has three per condition.
        # Quoting 16,075 as n invites a pseudo-replication objection and deserves it: cracks
        # within one frame are not independent, and frames within one specimen are not either.
        # So the same statistic is computed at all three levels and the specimen count is
        # carried, because a spread over 3 units is a different claim from a spread over
        # 16,075 and only one of them is defensible.
        by_spec, unknown_spec_frames = {}, []
        for name in g["images"]:
            rr = _read_rows(name) or []
            lcol = ("SkeletonLength_um" if physical else "SkeletonLength_px")
            # UNCENSORED cracks only. A crack touching the frame edge continues outside it,
            # so its SkeletonLength is a lower bound, and averaging bounds with lengths
            # gives neither. Measured on family=steel|condition=H: pooling them gave
            # 297.09 +/- 162.18 px where the uncensored cracks alone give 102.04 +/- 10.48,
            # because censored cracks carry 61.9% of the group's total length. The refused
            # longest-crack figure was already blanked for this reason while this one -- the
            # figure statistical_unit_note tells the reader to USE -- was not.
            usable, n_cens, n_unk = [], 0, 0
            for r in rr:
                v = r.get(lcol)
                if not isinstance(v, (int, float)):
                    continue
                c = _is_censored(r)
                if c is True:
                    n_cens += 1
                elif c is None:
                    n_unk += 1
                else:
                    usable.append(float(v))
            sk = specimen_key(name)
            if sk is None:
                unknown_spec_frames.append(name)
            if usable:
                by_spec.setdefault(sk, []).append(sum(usable) / len(usable))
            rec.setdefault("_censoring_tally", [0, 0, 0])
            rec["_censoring_tally"][0] += len(usable)
            rec["_censoring_tally"][1] += n_cens
            rec["_censoring_tally"][2] += n_unk

        # A frame whose specimen cannot be identified cannot be shown independent of any
        # other, so it does not get to be counted as its own unit.
        known = {k: v for k, v in by_spec.items() if k is not None}
        spec_means = [sum(v) / len(v) for v in known.values()]
        rec["n_specimens"] = len(known) if not unknown_spec_frames else None
        rec["n_frames_with_unknown_specimen"] = len(unknown_spec_frames)
        rec["specimens"] = sorted(known)
        _tal = rec.pop("_censoring_tally", [0, 0, 0])
        rec["n_cracks_uncensored"] = _tal[0]
        rec["n_cracks_censored"] = _tal[1]
        rec["n_cracks_censoring_unknown"] = _tal[2]
        rec["mean_crack_length_by_specimen"] = _describe(spec_means)
        rec["mean_crack_length_by_specimen_note"] = (
            f"Computed over UNCENSORED cracks only ({_tal[0]} of "
            f"{_tal[0] + _tal[1] + _tal[2]}; {_tal[1]} censored, {_tal[2]} with censoring "
            f"not recorded). Excluding censored cracks removes the longest ones, so this "
            f"UNDERSTATES mean crack length -- it is a mean over cracks that fit inside the "
            f"frame, not a mean crack length. A survival-style estimator would be the "
            f"correct treatment and is not attempted here.")
        rec["statistical_unit_note"] = (
            f"n_cracks={len(g['rows'])} is a POOLED count and must not be used as a sample "
            f"size: cracks within a frame are not independent, nor are frames within a "
            f"specimen. The independent unit here is the specimen, and this group has "
            f"{len(known) if not unknown_spec_frames else 'an UNKNOWN number of'}. Use "
            f"mean_crack_length_by_specimen for any cross-condition claim, and read its "
            f"note first: it covers uncensored cracks only.")
        if unknown_spec_frames:
            rec["dispersion_is_estimable"] = False
            rec["dispersion_refusal"] = (
                f"REFUSED: {len(unknown_spec_frames)} frame(s) have no identifiable "
                f"specimen, so independence cannot be established for any of them and the "
                f"number of independent units is UNKNOWN, not equal to the frame count. "
                f"Name files so the specimen can be read, or supply the grouping.")
        elif len(known) < 3:
            rec["dispersion_is_estimable"] = False
            rec["dispersion_refusal"] = (
                f"REFUSED: a spread cannot be estimated from {len(known)} independent "
                f"unit(s). Per-crack and per-frame dispersion look reassuringly small only "
                f"because they measure variation WITHIN specimens, which is not the "
                f"variation a cross-condition comparison is about.")
        else:
            rec["dispersion_is_estimable"] = True

        rec["longest_crack_per_image"] = _describe(per_image_max)
        rec["longest_crack_per_image_uncensored_only"] = _describe(per_image_max_uncens)
        rec["frames_with_a_censored_crack"] = n_cens_frames
        rec["n_frames"] = len(g["images"])
        # Measured on this corpus: on both spot-checked frames the LONGEST crack is censored,
        # and the longest uncensored crack is 83-97% shorter. So neither figure is a valid
        # cross-condition comparable -- the overall max quotes a lower bound as a length, and
        # the uncensored max excludes precisely the cracks the study is about. Rather than
        # print a number that cannot be interpreted, say so: this is the same refusal rule the
        # calibration and the promotion gate follow.
        if n_unknown_frames:
            rec["longest_crack_is_valid_comparable"] = False
            rec["longest_crack_refusal"] = (
                f"REFUSED: {n_unknown_frames} of {len(g['images'])} frames have CSVs written "
                f"before boundary censoring was recorded, so whether their cracks were cut "
                f"off by the frame edge is UNKNOWN. Re-run crack_measurements.py on them. "
                f"Treating unknown as uncensored is the same error this project exists to "
                f"prevent, so it is not done here.")
        elif n_cens_frames:
            rec["longest_crack_is_valid_comparable"] = False
            rec["longest_crack_refusal"] = (
                f"REFUSED as a cross-condition comparable: {n_cens_frames} of "
                f"{len(g['images'])} frames contain a crack cut off by the frame edge. Both "
                f"variants are reported for inspection, and neither should be quoted: the "
                f"all-cracks max mixes lower bounds with measurements, and the uncensored-only "
                f"max excludes the longest cracks by construction. A survival-style estimator "
                f"on censored lengths is the proper treatment and is not implemented here.")
        else:
            rec["longest_crack_is_valid_comparable"] = True
        if n_cens_frames:
            rec["censoring_warning"] = (
                f"{n_cens_frames} of {len(g['images'])} frames contain at least one crack "
                f"touching the frame edge, whose length is a LOWER BOUND. "
                f"longest_crack_per_image mixes bounds with measurements; "
                f"longest_crack_per_image_uncensored_only excludes them but is biased "
                f"downward, because the longest cracks are the most likely to be censored. "
                f"Neither is a valid cross-condition comparable on its own.")
        out.append(rec)

    return {"grouped_by": list(by), "groups": out, "skipped": skipped,
            "tool_version": __import__("common").VERSION}


def write_report(result, stem="aggregate"):
    os.makedirs(OUT_DIR, exist_ok=True)
    jp = os.path.join(OUT_DIR, f"{stem}.json")
    with open(jp, "w") as fh:
        json.dump(result, fh, indent=1)

    cp = os.path.join(OUT_DIR, f"{stem}.csv")
    fields = ["group", "units", "n_specimens", "n_frames_with_unknown_specimen",
              "n_frames", "n_cracks", "n_cracks_uncensored", "n_cracks_censored",
              "n_cracks_censoring_unknown",
              "n_calibrated", "n_uncalibrated", "dispersion_is_estimable",
              "mean_length_by_specimen_uncensored_only", "sd_length_by_specimen",
              "longest_crack_is_valid_comparable", "longest_crack_mean",
              "longest_crack_sd", "frames_with_unknown_censoring"]
    with open(cp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for g in result["groups"]:
            lc = g["longest_crack_per_image"]
            spec = g.get("mean_crack_length_by_specimen", {})
            valid = g.get("longest_crack_is_valid_comparable", True)
            w.writerow({
                "group": "|".join(f"{k}={v}" for k, v in g["group"].items()),
                "units": g["units"],
                # Specimens first, frames second, cracks last -- in the order of how much
                # weight each may carry, so a reader scanning the CSV meets the real n before
                # the impressive one.
                "n_specimens": g.get("n_specimens"),
                "n_frames_with_unknown_specimen":
                    g.get("n_frames_with_unknown_specimen"),
                # The censoring split, in the CSV and not only the JSON. The specimen mean
                # is computed over the uncensored column alone, so a reader needs to see how
                # many cracks that left out before quoting it.
                "n_cracks_uncensored": g.get("n_cracks_uncensored"),
                "n_cracks_censored": g.get("n_cracks_censored"),
                "n_cracks_censoring_unknown": g.get("n_cracks_censoring_unknown"),
                "n_frames": g["n_images"], "n_cracks": g["n_cracks"],
                "n_calibrated": g["n_calibrated"],
                "n_uncalibrated": g["n_uncalibrated"],
                "dispersion_is_estimable": g.get("dispersion_is_estimable"),
                # Named for what it is. "mean_length_by_specimen" invited the reader to
                # treat it as mean crack length; it is a mean over cracks that fit inside
                # the frame, which understates the real mean by excluding the longest.
                "mean_length_by_specimen_uncensored_only": spec.get("mean"),
                # Blank, not 0.0, when dispersion has been refused. Two specimens gave
                # "sd = 0.0" next to a False flag, and a reader quoting that number would
                # be quoting a spread this group is not entitled to report at all -- the
                # same trap the longest-crack column had.
                "sd_length_by_specimen": (spec.get("sd")
                                          if g.get("dispersion_is_estimable") else ""),
                "longest_crack_is_valid_comparable": valid,
                # Blank rather than a number when the figure has been refused: a CSV cell is
                # exactly where a refused statistic gets quoted anyway.
                "longest_crack_mean": (lc.get("mean") if valid else ""),
                # Two guards, and both must pass. `valid` is about censoring -- whether
                # these are lengths at all. `dispersion_is_estimable` is about independence
                # -- an sd over frames from one or two specimens measures variation WITHIN
                # a specimen, and that is unearned here for exactly the reason it is
                # unearned for the per-specimen sd above.
                "longest_crack_sd": (lc.get("sd") if valid
                                     and g.get("dispersion_is_estimable") else ""),
                "frames_with_unknown_censoring": g.get("frames_with_unknown_censoring"),
            })
    return jp, cp


#: Everything parse_name() can key on. Anything else is a typo, and it used to be accepted
#: silently: the grouping matched nothing, aggregate() returned zero groups, and write_report()
#: then OVERWROTE aggregate.csv with a header and no rows -- exit 0, no warning. `--help` hit
#: this exact path, so asking for help destroyed the output file.
GROUPABLE = ("family", "condition", "alloy", "block", "face", "detector", "date", "index")

USAGE = """usage: aggregate.py [GROUP_BY]

  GROUP_BY   comma-separated fields, default "family,condition"
             one or more of: family, condition, alloy, block, face, detector, date, index
             pass "none" to pool every frame into a single group

Writes aggregate.json and aggregate.csv next to the per-image measurement CSVs.
Group statistics use the SPECIMEN as the unit, and refuse rather than guess: dispersion is
withheld below three independent specimens, and a longest-crack mean is withheld when any
crack in the group touches a frame edge (its length is a lower bound, not a measurement)."""


if __name__ == "__main__":
    import sys
    from crack_measurements import all_images

    _argv = sys.argv[1:]
    if _argv and _argv[0] in ("-h", "--help"):
        print(USAGE)
        sys.exit(0)
    if len(_argv) > 1:
        print(f"expected at most one argument, got {len(_argv)}: {_argv}\n\n{USAGE}")
        sys.exit(2)

    if _argv and _argv[0].strip().lower() == "none":
        by = ()
    elif _argv:
        by = tuple(f.strip() for f in _argv[0].split(",") if f.strip())
    else:
        by = ("family", "condition")

    # REFUSE AN UNKNOWN FIELD instead of grouping by it. Writing an empty CSV is worse than
    # failing, because the file still looks like a result.
    _unknown = [f for f in by if f not in GROUPABLE]
    if _unknown:
        print(f"unknown group-by field(s) {_unknown}. Valid: {', '.join(GROUPABLE)}.\n"
              f"Refusing rather than grouping by nothing and writing an empty table.\n\n"
              f"{USAGE}")
        sys.exit(2)

    res = aggregate(all_images(), by=by, require_calibrated=False)
    # REFUSE TO WRITE AN EMPTY TABLE over a good one. semcrack.py already refuses on
    # aggregate_empty; this path did not, so a bad argument silently replaced real output.
    if not res["groups"]:
        print("no groups produced -- refusing to overwrite the existing report with an empty "
              "one. Nothing was written.")
        sys.exit(1)
    jp, cp = write_report(res)
    print(f"grouped by {by}: {len(res['groups'])} group(s)")
    for g in res["groups"]:
        lc = g["longest_crack_per_image"]
        sd = f" +/- {lc['sd']:.1f}" if lc.get("sd") is not None else ""
        mean = f"{lc['mean']:.1f}{sd}" if lc.get("mean") is not None else "n/a"
        valid = g.get("longest_crack_is_valid_comparable", True)
        _unk = g.get("frames_with_unknown_censoring", 0)
        shown = (f"longest/frame {mean} {g['units']}" if valid
                 else f"longest/frame REFUSED ("
                      + (f"censoring UNKNOWN in {_unk}/{g.get('n_frames')} frames"
                         if _unk else
                         f"censored in {g.get('frames_with_a_censored_crack')}"
                         f"/{g.get('n_frames')} frames") + ")")
        print(f"  {'|'.join(f'{k}={v}' for k, v in g['group'].items()):38s} "
              f"{g['n_images']:3d} frames  {g['n_cracks']:5d} cracks  "
              f"{g.get('n_specimens', 0):2d} specimen(s)  {shown}")
        if not g.get("dispersion_is_estimable", True):
            print(f"      dispersion REFUSED: {g.get('n_specimens')} independent unit(s); "
                  f"per-crack spread measures variation WITHIN a specimen, not between")
        if not valid:
            lu = g.get("longest_crack_per_image_uncensored_only", {})
            print(f"      for inspection only -- all-cracks mean {mean}, uncensored-only mean "
                  f"{_fmt_mean(lu)}; neither is quotable")
        if g.get("mixed_calibration_warning"):
            print(f"      WARNING {g['mixed_calibration_warning']}")
    for why, names in res["skipped"].items():
        if names:
            print(f"  skipped ({why}): {len(names)}")
    print(f"-> {jp}\n-> {cp}")
