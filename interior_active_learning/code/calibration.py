"""Physical-unit calibration: pixels -> micrometres, with provenance.

WHY THIS EXISTS
Every number this project exported was in pixels: Area_px, SkeletonLength_px,
MaxWidth_px. A crack length in pixels is not a publishable quantity -- no materials
journal takes it -- so the tool could produce a better segmentation than ilastik or
Dragonfly and still be unusable for the actual paper. app_exports.py:70 admitted as much
in a comment ("Pixel units only -- no micron conversion") and left the reader to
multiply by hand, which is exactly where unit errors enter a manuscript.

WHY CALIBRATION IS NOT AUTOMATIC
It could have been, and three attempts got three different wrong answers:

  longest contiguous bright run in the panel   1000 px -> 0.400 um/px
  widest bright extent in the panel's right half 2816 px -> 0.142 um/px
  tick-to-tick, measured by eye on a crop      ~2379 px -> 0.168 um/px

Only the third agrees with the independent HFW route (field width 1.04 mm over 6144 px =
0.169 um/px, within 0.7%). The first overshot because the "400 um" label INTERRUPTS the
bar, so the longest contiguous run is one segment of it; the second overshot because the
panel's right border is also bright. A silently wrong calibration is worse than none: it
propagates into every exported length and nobody notices, because the numbers still look
plausible. So calibration is explicit, recorded with its provenance, and cross-checked.

The vendor metadata is not available as a fallback. The shipped TIFFs carry
`Software: tifffile.py` and `XResolution: (1, 1)` because the corpus was losslessly
recompressed (pixels are bit-identical; the FEI tags were not preserved), and the
pre-compression copies in the archived repo only carry print DPI -- 768 and 384 with
ResolutionUnit=2 -- which is not specimen scale.

HOW A VALUE GETS SET
Either route records where it came from, so any exported number can be traced:

  "scale_bar"  the user reads the label off the burned-in bar ("400 um") and marks its
               two end ticks; um_per_px = label / |x2 - x1|. This is ImageJ's Set Scale,
               except the span is measured from the marks rather than typed, so it does
               not inherit a hand-drawn line's aiming error.
  "hfw"        horizontal field width from the panel divided by image width.
  "manual"     typed directly, for a user who already knows the value.

CROSS-CHECK
set_from_scale_bar accepts an optional hfw_um. When given, it recomputes um/px the other
way and refuses the calibration if the two disagree by more than tolerance (default 5%),
because that disagreement means one of the two readings is wrong and there is no way to
tell which from inside the program.

UNCALIBRATED IS A STATE, NOT A ZERO
get_um_per_px returns None for an image nobody calibrated, and callers must emit pixel
columns and say `calibrated=false` rather than defaulting to 1.0. This is the same rule
the correction masks follow -- 0 means UNREVIEWED, not "not a crack" -- and it exists for
the same reason: a silent default is indistinguishable from a real measurement.
"""
import json
import os
import time

from common import PAINT_DIR, VERSION, save_json_atomic

CALIB_PATH = os.path.join(PAINT_DIR, "calibration.json")

#: Two independent readings of the same frame must agree this closely, or the
#: calibration is refused. 5% is generous for a hand-marked span and still catches the
#: 16-25% errors the automatic bar detectors produced.
CROSS_CHECK_TOL = 0.05


def _load_all():
    if not os.path.exists(CALIB_PATH):
        return {}
    try:
        with open(CALIB_PATH) as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except (ValueError, OSError):
        # A corrupt calibration file must not be silently treated as "no calibration" --
        # that would downgrade calibrated exports to pixels with no warning.
        raise ValueError(f"{CALIB_PATH} is unreadable; refusing to treat it as empty")


def _write_all(d):
    save_json_atomic(d, CALIB_PATH)


def get_um_per_px(image_name):
    """Micrometres per pixel, or None if this image has never been calibrated."""
    rec = _load_all().get(image_name)
    if not isinstance(rec, dict):
        return None
    v = rec.get("um_per_px")
    return float(v) if isinstance(v, (int, float)) and v > 0 else None


def get_record(image_name):
    """The full calibration record including provenance, or None."""
    rec = _load_all().get(image_name)
    return rec if isinstance(rec, dict) else None


def _store(image_name, um_per_px, source, detail):
    if not (isinstance(um_per_px, (int, float)) and um_per_px > 0):
        raise ValueError(f"um_per_px must be a positive number, got {um_per_px!r}")
    d = _load_all()
    d[image_name] = {
        "um_per_px": float(um_per_px),
        "source": source,
        "detail": detail,
        "set_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
    }
    _write_all(d)
    return d[image_name]


def set_manual(image_name, um_per_px, note=""):
    """Record a value the user already knows."""
    return _store(image_name, um_per_px, "manual", {"note": note})


def set_from_hfw(image_name, hfw_um, image_width_px):
    """Horizontal field width over image width."""
    if image_width_px <= 0:
        raise ValueError("image_width_px must be positive")
    return _store(image_name, float(hfw_um) / float(image_width_px), "hfw",
                  {"hfw_um": float(hfw_um), "image_width_px": int(image_width_px)})


def set_from_scale_bar(image_name, label_um, x1, x2, hfw_um=None,
                       image_width_px=None, tol=CROSS_CHECK_TOL):
    """Calibrate from the burned-in bar: label length over marked pixel span.

    Pass hfw_um and image_width_px to cross-check. If the two routes disagree by more
    than `tol`, this raises instead of storing -- the disagreement means one reading is
    wrong and the program cannot tell which.
    """
    span = abs(float(x2) - float(x1))
    # A jittered double-click, or clicking the same tick twice, lands 1-3 px apart. The old
    # guard was `span < 1` with the message "the two marks are the same point", which is
    # both wrong for 0 < span < 1 and useless at span == 2: label 400 over span 2 stores
    # 200 um/px against a true 0.169, so every exported length comes out ~1180x too large,
    # with a green readout and full provenance saying it is calibrated. A real scale bar
    # occupies a substantial part of the frame, so require that when the width is known,
    # and an absolute floor otherwise.
    MIN_SPAN_PX = 50
    MIN_SPAN_FRAC = 0.02
    floor = MIN_SPAN_PX
    if image_width_px:
        floor = max(floor, MIN_SPAN_FRAC * float(image_width_px))
    if span < floor:
        raise ValueError(
            f"the two marks are only {span:.1f} px apart, below the {floor:.0f} px minimum "
            f"for a believable scale bar. Mark the bar's two END TICKS -- a span this short "
            f"would store {label_um / max(span, 1e-9):.3f} um/px and inflate every exported "
            f"length by orders of magnitude.")
    if label_um <= 0:
        raise ValueError("label_um must be positive")
    bar = float(label_um) / span
    detail = {"label_um": float(label_um), "span_px": span,
              "x1": float(x1), "x2": float(x2)}
    if hfw_um and image_width_px:
        other = float(hfw_um) / float(image_width_px)
        rel = abs(bar - other) / other
        detail.update({"hfw_um": float(hfw_um), "hfw_um_per_px": other,
                       "cross_check_rel_diff": rel})
        if rel > tol:
            raise ValueError(
                f"{image_name}: scale bar gives {bar:.5f} um/px but HFW gives "
                f"{other:.5f} um/px, a {100 * rel:.1f}% disagreement (tolerance "
                f"{100 * tol:.0f}%). One of the two readings is wrong -- refusing to "
                f"store a calibration that would silently corrupt every exported length.")
    return _store(image_name, bar, "scale_bar", detail)


def clear(image_name):
    d = _load_all()
    if image_name in d:
        del d[image_name]
        _write_all(d)
        return True
    return False


# --------------------------------------------------------------------- conversion
#: Exported measurement columns and the power of length each carries, so one table drives
#: every conversion instead of each call site remembering that area goes as um^2.
LENGTH_POWERS = {
    "Area_px": 2,
    "SkeletonLength_px": 1,
    "MeanWidth_px": 1,
    "MaxWidth_px": 1,
    "EllipseMajorAxis_px": 1,
    "EllipseMinorAxis_px": 1,
    "CentroidX_px": 1,
    "CentroidY_px": 1,
}

#: Dimensionless -- must NOT be scaled. Tortuosity is a ratio of two lengths,
#: Orientation is an angle, BranchPointCount is a count. Scaling any of these was the
#: obvious way to get this wrong.
DIMENSIONLESS = {"Tortuosity", "Orientation_deg", "BranchPointCount",
                 "NFragmentsMerged",
                 "BoundaryRoughness", "AreaPct_of_image", "CrackID", "SourceImage"}


def um_column_name(px_column):
    """Area_px -> Area_um2, SkeletonLength_px -> SkeletonLength_um."""
    base = px_column[:-3] if px_column.endswith("_px") else px_column
    return f"{base}_um2" if LENGTH_POWERS.get(px_column) == 2 else f"{base}_um"


def convert_row(row, um_per_px):
    """Add micrometre columns for every length column present. Returns a new dict.

    Dimensionless quantities are passed through untouched -- see DIMENSIONLESS.
    """
    out = dict(row)
    if not um_per_px:
        return out
    for col, power in LENGTH_POWERS.items():
        if col in row and isinstance(row[col], (int, float)):
            out[um_column_name(col)] = row[col] * (um_per_px ** power)
    return out


def provenance_header(image_name, model_path=None, threshold=None):
    """A one-record description of how the numbers for this image were produced.

    Exports previously carried no provenance at all: once a zip left the browser, no
    number in it could be traced to a model, a threshold, or a calibration. A reviewer
    asking "which model produced this length, and in what units" had no answer.
    """
    rec = get_record(image_name)
    # Use the SAME predicate the conversion uses. `bool(rec)` is weaker than
    # get_um_per_px, which requires a positive number, so a record holding 0, a negative,
    # null or the string "0.169" produced pixel columns in the CSV while the sidecar
    # announced calibrated=true with that value echoed back and the UNCALIBRATED note
    # suppressed -- pixel numbers labelled as micrometres, which is the one outcome this
    # module exists to prevent.
    umpx = get_um_per_px(image_name)
    out = {
        "image": image_name,
        "tool_version": VERSION,
        "calibrated": umpx is not None,
        "um_per_px": umpx,
        "calibration_source": (rec or {}).get("source"),
        "calibration_set_at": (rec or {}).get("set_at"),
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
    }
    if model_path:
        out["model"] = os.path.basename(model_path)
        try:
            out["model_mtime"] = time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(model_path)))
        except OSError:
            pass
    if threshold is not None:
        out["threshold"] = threshold
    if umpx is None:
        out["note"] = ("UNCALIBRATED: lengths are in PIXELS. Set a calibration before "
                       "reporting physical dimensions.")
        if rec:
            # A record exists but the converter rejects it -- say so, rather than letting
            # it read as "never calibrated".
            out["calibration_record_invalid"] = True
            out["invalid_um_per_px"] = rec.get("um_per_px")
    return out
