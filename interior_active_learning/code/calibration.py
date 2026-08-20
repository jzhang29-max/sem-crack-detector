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

#: A batch run keeps its calibration beside its own outputs. Without this override the
#: store lives in PAINT_DIR, so a headless run pointed at someone's read-only corrections
#: directory would try to write there, and two batches sharing a corrections directory
#: would share a scale they never agreed on.
CALIB_PATH = (os.path.abspath(os.path.expanduser(os.environ["SEMCRACK_CALIB_PATH"]))
              if os.environ.get("SEMCRACK_CALIB_PATH")
              else os.path.join(PAINT_DIR, "calibration.json"))

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


#: Where instrument vendors record the field width. FEI/Thermo writes an INI block, ZEISS
#: writes its own tag set, and both are wildly non-standard -- hence a table rather than a
#: parser.
_FEI_TAGS = (34682, 34683)          # FEI_HELIOS / FEI_TITAN INI blocks
_ZEISS_TAG = 34118                  # ZEISS CZ_SEM
_HFW_KEYS = ("HorizontalFieldWidth", "HFW", "Width", "PixelWidth", "AP_WIDTH",
             "AP_IMAGE_PIXEL_SIZE", "ap_image_pixel_size")


def read_instrument_metadata(image_path):
    """Pixel size straight from the instrument, or None if the file does not carry it.

    THE HONEST STATUS OF THIS FUNCTION, because it would otherwise read as a validated
    feature. Every TIFF in this repository's own corpus has had its vendor metadata
    DESTROYED: all 70 carry an ImageDescription of exactly `{"shape": [h, w]}`, the
    fingerprint of a numpy/tifffile re-save, and none carries an FEI or ZEISS block. So
    this reader returns None on every image here and cannot be regression-tested against a
    real vendor file in this repo. It is written for users whose files are still originals,
    and it is deliberately conservative: it reads, it never infers.

    That destruction is worth stating rather than working around. The microscope recorded
    the field width; a re-save with a tool that does not preserve private TIFF tags threw
    it away silently; and the consequence is that 0 of 62 images can be calibrated from
    metadata and every scale here has to be marked by hand off the burned-in bar. It is
    the same failure this project is about -- a default that substituted convenience for a
    measurement and said nothing -- and it happened to the data before any of this code
    ran.

    Returns {"um_per_px": float, "source": str, "detail": dict} or None.
    """
    try:
        from PIL import Image
        im = Image.open(image_path)
        tags = getattr(im, "tag_v2", {}) or {}
        width_px = int(im.size[0])
    except Exception:
        return None
    if width_px <= 0:
        return None

    def _ini_lookup(blob):
        """FEI/ZEISS blocks are INI-ish: `Key = value` lines under [Section] headers."""
        out = {}
        for line in str(blob).replace("\r", "\n").split("\n"):
            if "=" in line:
                k, _, v = line.partition("=")
                out[k.strip()] = v.strip()
        return out

    for tag in _FEI_TAGS + (_ZEISS_TAG,):
        if tag not in tags:
            continue
        kv = _ini_lookup(tags[tag])
        for key in _HFW_KEYS:
            raw = kv.get(key)
            if raw is None:
                continue
            try:
                val = float(str(raw).split()[0])
            except (TypeError, ValueError):
                continue
            if not (val > 0):
                continue
            # FEI records metres. A "HorizontalFieldWidth" of 0.0002 is 200 um, not
            # 0.0002 um, and mistaking one for the other is a 1e6 error -- so the unit is
            # decided by the key's own meaning, never by which magnitude looks plausible.
            if key in ("PixelWidth", "AP_IMAGE_PIXEL_SIZE", "ap_image_pixel_size"):
                um_per_px = val * 1e6 if val < 1e-3 else val
            else:
                hfw_um = val * 1e6 if val < 1e-3 else val
                um_per_px = hfw_um / width_px
            if 1e-6 < um_per_px < 1e4:
                return {"um_per_px": float(um_per_px),
                        "source": f"instrument_metadata:tag{tag}:{key}",
                        "detail": {"tiff_tag": tag, "key": key, "raw": str(raw)[:80],
                                   "image_width_px": width_px}}
    return None


def set_from_instrument_metadata(image_name, image_path, tol=CROSS_CHECK_TOL):
    """Calibrate from vendor metadata, cross-checked against any existing calibration.

    Metadata is the most trustworthy source available -- the instrument wrote it -- but
    "most trustworthy" is not "unquestionable", so if this image already has a scale from
    a hand-marked bar and the two disagree by more than `tol`, this REFUSES rather than
    silently overwriting a human's reading with a machine's.
    """
    got = read_instrument_metadata(image_path)
    if got is None:
        return None
    prev = get_um_per_px(image_name)
    if prev and abs(got["um_per_px"] - prev) / prev > tol:
        raise ValueError(
            f"{image_name}: instrument metadata says {got['um_per_px']:.6g} um/px but the "
            f"stored calibration says {prev:.6g} um/px, a "
            f"{100 * abs(got['um_per_px'] - prev) / prev:.1f}% disagreement. One of them is "
            f"wrong and this cannot tell which, so neither is used.")
    return _store(image_name, got["um_per_px"], got["source"], got["detail"])


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
