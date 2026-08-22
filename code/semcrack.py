#!/usr/bin/env python3
"""Headless batch crack measurement: a directory of micrographs in, a table of cracks out.

WHY THIS EXISTS
Every comparable tool has a batch mode and this one did not, which meant the only way to
measure a hundred frames was to click through a browser a hundred times. That is not a
missing convenience, it is a reason the tool cannot be used for the studies it was built
for.

WHAT IT REFUSES TO DO, AND WHY EACH REFUSAL IS THERE
A batch runner is where provenance goes to die: it is the step that turns images into
numbers with nobody watching, so every default it picks silently becomes a published
quantity. So this one refuses rather than guesses.

  * It will not write into an output directory produced by a DIFFERENT model, threshold or
    input directory. Two runs sharing a folder produce a CSV set that is half one
    configuration and half another, with no way to tell which row came from which. Pass
    --force if you actually meant it.
  * It does not apply human corrections unless you name their directory with
    --corrections. Folding another session's hand edits into a batch export would make the
    numbers part human judgement without saying which part.
  * A single --um-per-px is cross-checked against instrument metadata when the files still
    carry it, and REFUSED on disagreement over 5%. It is also refused outright if the
    images are not all the same pixel size, because one scale across differently-sized
    frames is a magnification assumption, not a measurement.
  * Uncalibrated images get pixel columns and a sidecar that says so. They never get
    micrometre columns filled from a default, which would be indistinguishable from a real
    measurement.
  * It exits non-zero if any image failed. A batch that measured 40 of 62 frames and
    exited 0 reads as success in every script that calls it.

The manifest is the point of the whole thing: run_manifest.json records the code version,
the model file's SHA-256, the resolved threshold, both directories, whether corrections
were applied, the calibration source per image, and every refusal. An output directory that
cannot say what produced it is the hole this project exists to complain about.

    semcrack.py --in ./micrographs --out ./results
    semcrack.py --in ./micrographs --out ./results --um-per-px 0.0431 --jobs 6
    semcrack.py --in ./micrographs --out ./results --threshold 0.6 --model my.joblib
    semcrack.py --in ./micrographs --out ./results --from-metadata
    semcrack.py --in ./micrographs --out ./results --dry-run
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import traceback

EXIT_OK, EXIT_FAILURES, EXIT_NOTHING, EXIT_CONFLICT = 0, 1, 2, 3


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha(root):
    head = os.path.join(root, ".git", "HEAD")
    try:
        with open(head) as fh:
            ref = fh.read().strip()
        if ref.startswith("ref: "):
            with open(os.path.join(root, ".git", ref[5:])) as fh:
                return fh.read().strip()
        return ref
    except OSError:
        return None


def _parse_args(argv):
    ap = argparse.ArgumentParser(
        prog="semcrack",
        description="Batch crack measurement with refusals instead of defaults.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="indir", required=True, help="directory of micrographs")
    ap.add_argument("--out", dest="outdir", required=True, help="output directory")
    ap.add_argument("--glob", default="*.tif", help="filename pattern (default *.tif)")
    ap.add_argument("--group-by", dest="group_by", default="family,condition",
                    help="comma-separated name tokens to group by, or 'none' for one "
                         "pooled group. Grouping parses tokens out of the FILENAME, so "
                         "'none' is the right answer for files not named like this "
                         "corpus (default: family,condition)")
    ap.add_argument("--model", help="Pass 1 classifier bundle (.joblib)")
    ap.add_argument("--sam2", choices=["off", "refine", "hybrid"], default="refine",
                    help="SAM 2 refinement of the detector's candidate boundaries. DEFAULT "
                         "'refine': it beats the shipped detector on f1, recall, specificity "
                         "AND precision, so it is on unless you turn it off. 'hybrid' unions "
                         "the two for a higher f1 at lower specificity. 'off' is the bare "
                         "detector. Costs roughly 0.2s per candidate region and degrades to "
                         "the bare detector, on the record, if the checkpoint is unavailable")
    ap.add_argument("--threshold", type=float,
                    help="decision threshold for BOTH passes; default is each bundle's own")
    ap.add_argument("--um-per-px", type=float, dest="um_per_px",
                    help="one scale for every image, cross-checked where possible")
    ap.add_argument("--scale-csv", dest="scale_csv",
                    help="per-image scale; columns image,um_per_px")
    ap.add_argument("--from-metadata", action="store_true",
                    help="read pixel size from FEI/ZEISS TIFF tags where present")
    ap.add_argument("--corrections", help="directory of *_correction_mask.png to apply")
    ap.add_argument("--jobs", type=int, default=1, help="parallel workers (default 1)")
    ap.add_argument("--force", action="store_true",
                    help="write into an output directory from a different configuration")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run")
    a = ap.parse_args(argv)
    if a.um_per_px is not None and a.scale_csv:
        ap.error("--um-per-px and --scale-csv are mutually exclusive: pick one source of "
                 "scale, because two sources that disagree cannot both be right")
    if a.um_per_px is not None and a.um_per_px <= 0:
        ap.error("--um-per-px must be positive")
    if a.threshold is not None and not (0.0 < a.threshold < 1.0):
        ap.error("--threshold must lie strictly between 0 and 1")
    if a.jobs < 1:
        ap.error("--jobs must be at least 1")
    return a


def _configure_env(a):
    """Point the pipeline at the caller's directories. MUST run before the imports below.

    common.py resolves its paths at import time, so setting these afterwards would leave
    the run reading the repository's own images while reporting the caller's -- a silent
    wrong-data failure rather than a crash.
    """
    os.environ["SEMCRACK_ORIGINAL_DIR"] = a.indir
    os.environ["SEMCRACK_MEASUREMENTS_DIR"] = a.outdir
    os.environ["SEMCRACK_CALIB_PATH"] = os.path.join(a.outdir, "calibration.json")
    # No corrections directory means an empty one, not the repo's. Defaulting to the repo's
    # paint/ would quietly apply this project's own hand marks to a stranger's images.
    os.environ["SEMCRACK_PAINT_DIR"] = (
        os.path.abspath(os.path.expanduser(a.corrections)) if a.corrections
        else os.path.join(a.outdir, "_no_corrections"))
    if a.model:
        os.environ["SEMCRACK_MODEL"] = os.path.abspath(os.path.expanduser(a.model))
    # UNCONDITIONALLY, even when no --threshold was given. This is a private parent->child
    # channel, and it used to be written only when the flag was present -- so a value left
    # in the caller's environment was inherited, silently moved the detector, and was
    # invisible to _effective_threshold and the config fingerprint, both of which report
    # the bundle's own value. The run then produced numbers at one threshold and a manifest
    # claiming another, and two runs at different thresholds were judged the same
    # configuration and allowed to share an output directory. Every other SEMCRACK_* var
    # here is set unconditionally; this one was the exception.
    os.environ["SEMCRACK_THRESHOLD"] = ("" if a.threshold is None
                                        else repr(float(a.threshold)))
    # Same hermetic rule as the threshold: written unconditionally so an inherited value
    # cannot change the detector while the manifest reports something else.
    os.environ["SEMCRACK_SAM2"] = ("" if a.sam2 in (None, "off") else a.sam2)


def _effective_threshold(a, model_path):
    """The number that will actually decide, whether or not the caller named it.

    Fingerprinting on the raw flag was wrong in both directions. A first run with no
    --threshold recorded null, so a second run passing the bundle's own value explicitly
    was refused as a different configuration when it is the same one; and a manifest
    reading `"threshold": null` never states the number its CSVs were produced at, which
    is the thing a reader most needs and the thing this repo keeps finding missing.
    """
    if a.threshold is not None:
        return float(a.threshold), "given on the command line"
    try:
        import joblib
        bundle = joblib.load(model_path)
        if isinstance(bundle, dict) and "threshold" in bundle:
            return float(bundle["threshold"]), "read from the model bundle"
    except Exception:
        pass
    return 0.5, ("the bundle carries no threshold key, so this is the 0.5 fallback -- a "
                 "library default reached by omission, not a calibration decision")


def _config_fingerprint(a, model_path, threshold):
    """What must match for two runs to be allowed to share an output directory."""
    return {"input_dir": os.path.abspath(os.path.expanduser(a.indir)),
            "model_sha256": _sha256(model_path) if os.path.exists(model_path) else None,
            # The EFFECTIVE threshold, so "unspecified" and "specified as the same value"
            # are correctly the same configuration.
            "threshold": threshold,
            # The detector is part of the configuration. Two runs that differ in it produce
            # incomparable rows, so they must not share an output directory unnoticed.
            "sam2": (a.sam2 if a.sam2 != "off" else None),
            "corrections_dir": (os.path.abspath(os.path.expanduser(a.corrections))
                                if a.corrections else None)}


def _measure_one(name):
    """One image, in this process. Top level so multiprocessing can pickle it."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "interior_active_learning", "code"))
    import crack_measurements as cm
    import unified_pipeline as up
    t = os.environ.get("SEMCRACK_THRESHOLD")
    if t:
        try:
            up.THRESHOLD_OVERRIDE = float(t)
        except (TypeError, ValueError):
            # float() used to sit outside the try below, so a non-numeric value aborted the
            # whole batch with a bare ValueError instead of failing one image with a reason.
            return {"image": name, "status": "failed",
                    "error": f"SEMCRACK_THRESHOLD is not a number: {t!r}"}
    try:
        df = cm.measure_image(name)
        import calibration as cal
        rec = cal.get_record(name) or {}
        return {"image": name, "status": "ok", "n_cracks": int(len(df)),
                "n_censored": int(df["LengthIsCensored"].sum()) if len(df) else 0,
                "calibration_source": rec.get("source"),
                "units": "um" if rec.get("um_per_px") else "px"}
    except Exception as e:
        return {"image": name, "status": "failed",
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[-1200:]}


def main(argv=None):
    a = _parse_args(argv if argv is not None else sys.argv[1:])
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    a.indir = os.path.abspath(os.path.expanduser(a.indir))
    a.outdir = os.path.abspath(os.path.expanduser(a.outdir))
    if not os.path.isdir(a.indir):
        print(f"semcrack: --in is not a directory: {a.indir}", file=sys.stderr)
        return EXIT_NOTHING

    paths = sorted(p for p in glob.glob(os.path.join(a.indir, a.glob))
                   if os.path.isfile(p))
    names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    if not names:
        print(f"semcrack: no files matching {a.glob!r} in {a.indir}", file=sys.stderr)
        return EXIT_NOTHING

    # Every output file is named from the basename, so two inputs that share one -- a.tif
    # beside a.tiff, or foo.TIF beside foo.tif -- write the same CSV twice and the second
    # silently wins. There is no correct guess about which the user meant.
    # Case-FOLDED, because the filesystem is what decides whether two output paths
    # collide. On macOS and Windows, Blob_01.tif and blob_01.tiff write to the same CSV;
    # a case-sensitive check passed them both, one silently overwrote the other, and
    # all_cracks.csv then counted the survivor twice while the lost frame vanished.
    _folded = [n.casefold() for n in names]
    _dupes = sorted({n for n in names if _folded.count(n.casefold()) > 1})
    if _dupes:
        print(f"semcrack: {len(_dupes)} basename(s) collide in {a.indir} (ignoring case, "
              f"because the filesystem does): {', '.join(_dupes[:6])}. Every output file is "
              f"named from the basename, so these would overwrite each other and the last "
              f"one would silently win. Rename them or narrow --glob.", file=sys.stderr)
        return EXIT_CONFLICT

    # A mistyped --corrections path used to be CREATED, empty, by common.py's import-time
    # makedirs. Zero corrections were then applied while the manifest asserted
    # corrections_applied=true and "these numbers are part human judgement". Refuse before
    # anything is created: a path the user got wrong is not a path to work around.
    if a.corrections:
        _cdir = os.path.abspath(os.path.expanduser(a.corrections))
        if not os.path.isdir(_cdir):
            print(f"semcrack: --corrections {_cdir} is not a directory. It will not be "
                  f"created: a run that reports corrections_applied=true having applied "
                  f"none is worse than a run that stops.", file=sys.stderr)
            return EXIT_NOTHING
        if not glob.glob(os.path.join(_cdir, "*_correction_mask.png")):
            print(f"semcrack: --corrections {_cdir} holds no *_correction_mask.png. Nothing "
                  f"would be applied, so the manifest would claim human judgement the "
                  f"output does not contain. Drop the flag to run detector-only.",
                  file=sys.stderr)
            return EXIT_NOTHING

    os.makedirs(a.outdir, exist_ok=True)
    _configure_env(a)
    os.makedirs(os.environ["SEMCRACK_PAINT_DIR"], exist_ok=True)

    sys.path.insert(0, os.path.join(root, "interior_active_learning", "code"))
    sys.path.insert(0, os.path.join(root, "code"))
    from common import PROD_MODEL_PATH, VERSION
    import calibration as cal

    refusals = []
    eff_threshold, eff_why = _effective_threshold(a, PROD_MODEL_PATH)
    fp = _config_fingerprint(a, PROD_MODEL_PATH, eff_threshold)
    prev_path = os.path.join(a.outdir, "run_manifest.json")
    # The comparison runs ALWAYS. It used to be gated on `not a.force`, so a forced run
    # never computed the conflict and therefore could not record it: the directory ended up
    # half one threshold and half another, with a manifest naming only the new one, an empty
    # refusals list, and the previous manifest overwritten -- no trace of either half. That
    # is the exact outcome this module's docstring promises to prevent, produced by the flag
    # meant to acknowledge it.
    prev_full = None
    if os.path.exists(prev_path):
        try:
            prev_full = json.load(open(prev_path))
            prev = (prev_full or {}).get("config_fingerprint")
        except (ValueError, OSError):
            prev = None
        if prev and prev != fp:
            differing = [k for k in fp if prev.get(k) != fp.get(k)]
            if a.force:
                # Keep the superseded manifest instead of destroying it, and say which
                # earlier images it covers, so a reader can tell the halves apart.
                _keep = os.path.join(
                    a.outdir, f"run_manifest.superseded.{len(glob.glob(os.path.join(a.outdir, 'run_manifest.superseded.*.json')))}.json")
                try:
                    with open(_keep, "w") as _fh:
                        json.dump(prev_full, _fh, indent=1)
                except OSError:
                    _keep = None
                refusals.append({
                    "kind": "config_conflict_forced", "forced": True,
                    "differing_keys": differing,
                    "previous_fingerprint": prev,
                    "previous_images": [r.get("image") for r in
                                        (prev_full or {}).get("images", [])
                                        if r.get("status") == "ok"],
                    "previous_manifest_kept_as": (os.path.basename(_keep) if _keep
                                                  else None),
                    "note": ("--force was passed, so this directory now holds CSVs from "
                             "more than one configuration. The rows measured under the "
                             "earlier one are listed in previous_images and their manifest "
                             "is kept beside this one. Per-image *_provenance.json records "
                             "each CSV's own threshold, so the halves remain "
                             "distinguishable per row.")})
                print(f"semcrack: --force: writing over a different configuration "
                      f"({', '.join(differing)} differ). Recorded as a refusal; the "
                      f"superseded manifest is kept.", file=sys.stderr)
            else:
                print(f"semcrack: {a.outdir} was produced by a different configuration "
                      f"({', '.join(differing)} differ). Mixing them gives a CSV set with "
                      f"no way to tell which row came from which run. Use a new --out, or "
                      f"--force if you meant it.", file=sys.stderr)
                return EXIT_CONFLICT

    # ---- scale, before any measuring, so a refusal costs nothing ----
    sizes, unreadable = set(), []
    if a.um_per_px is not None or a.from_metadata:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        for p in paths:
            wh = None
            try:
                wh = Image.open(p).size
            except Exception:
                # The pipeline reads with tifffile, which decodes compressions PIL cannot
                # (LERC, some tiled/deflate variants). Probing with PIL alone and swallowing
                # the error dropped those frames from the size set, so the
                # differently-sized-frames refusal could not fire and one scale was applied
                # across two magnifications. A guard that silently loses its inputs cannot
                # fail, which is the failure mode this repo keeps finding.
                try:
                    import tifffile as _tf
                    with _tf.TiffFile(p) as _t:
                        _sh = _t.pages[0].shape
                    wh = (int(_sh[1]), int(_sh[0]))
                except Exception:
                    unreadable.append(os.path.basename(p))
            if wh:
                sizes.add(wh)

    if a.um_per_px is not None and unreadable:
        # Cannot show the frames share a magnification, so do not assume it.
        print(f"semcrack: could not read the pixel dimensions of {len(unreadable)} file(s) "
              f"({', '.join(unreadable[:4])}), so --um-per-px cannot be shown to apply to "
              f"them. Use --scale-csv, or --force.", file=sys.stderr)
        if not a.force:
            return EXIT_CONFLICT
        refusals.append({"kind": "size_unreadable_scale_forced", "forced": True,
                         "files": unreadable[:20]})

    if a.um_per_px is not None and len(sizes) > 1:
        print(f"semcrack: --um-per-px {a.um_per_px} was given for images of "
              f"{len(sizes)} different pixel sizes {sorted(sizes)[:4]}. One scale across "
              f"differently-sized frames assumes they share a magnification, which is an "
              f"assumption and not a measurement. Use --scale-csv, or --force.",
              file=sys.stderr)
        if not a.force:
            return EXIT_CONFLICT
        refusals.append({"kind": "one_scale_many_sizes", "forced": True,
                         "distinct_sizes": len(sizes)})

    per_image_scale = {}
    if a.scale_csv:
        import csv
        with open(a.scale_csv, newline="") as fh:
            for row in csv.DictReader(fh):
                nm = (row.get("image") or "").strip()
                try:
                    v = float(row.get("um_per_px"))
                except (TypeError, ValueError):
                    continue
                if nm and v > 0:
                    per_image_scale[os.path.splitext(nm)[0]] = v

    # Which images this run actually established a scale for. Anything else must be
    # CLEARED: calibration.json lives in --out, so a record left by an earlier run survived
    # and silently supplied micrometre columns for a frame this run's manifest said was
    # "measured in PIXELS". The refusal skipped set_manual but never cleared, so the
    # refusal did not refuse -- it just declined to overwrite.
    _scaled = set()

    for p, nm in zip(paths, names):
        want = per_image_scale.get(nm, a.um_per_px)
        meta = cal.read_instrument_metadata(p) if (a.from_metadata or want) else None
        if want is not None:
            if meta and abs(meta["um_per_px"] - want) / want > cal.CROSS_CHECK_TOL:
                refusals.append({
                    "kind": "scale_disagrees_with_metadata", "image": nm,
                    "given_um_per_px": want, "metadata_um_per_px": meta["um_per_px"],
                    "note": "instrument metadata and the supplied scale disagree by more "
                            "than 5%; one is wrong and this cannot tell which, so this "
                            "image is measured in PIXELS"})
                continue
            cal.set_manual(nm, want, note="semcrack --um-per-px/--scale-csv" +
                           (" (agrees with instrument metadata)" if meta else
                            " (NO independent cross-check available)"))
            _scaled.add(nm)
        elif a.from_metadata:
            if meta:
                cal.set_from_instrument_metadata(nm, p)
                _scaled.add(nm)
            else:
                refusals.append({"kind": "no_instrument_metadata", "image": nm,
                                 "note": "file carries no FEI/ZEISS pixel size; measured "
                                         "in PIXELS rather than assuming one"})
        elif a.scale_csv:
            # An image the scale CSV never mentions falls through to pixels. Saying nothing
            # would make it indistinguishable from an image the user meant to leave
            # uncalibrated, and the whole point of the sidecar is that those differ.
            refusals.append({"kind": "image_absent_from_scale_csv", "image": nm,
                             "note": f"{os.path.basename(a.scale_csv)} has no row for this "
                                     f"image, so it is measured in PIXELS. If that was not "
                                     f"intended, add a row."})

    # Enforce it. Every image this run did not scale is uncalibrated, whatever an earlier
    # run left behind, so any stale record is removed before measuring.
    _stale = []
    for nm in names:
        if nm not in _scaled and cal.get_um_per_px(nm) is not None:
            _rec = cal.get_record(nm) or {}
            cal.clear(nm)
            _stale.append({"image": nm, "cleared_um_per_px": _rec.get("um_per_px"),
                           "cleared_source": _rec.get("source")})
    if _stale:
        refusals.append({
            "kind": "stale_calibration_cleared", "n": len(_stale), "images": _stale[:20],
            "note": ("These images carried a calibration from an earlier run in this output "
                     "directory but this run established no scale for them, so the records "
                     "were removed and the images are measured in PIXELS. Leaving them would "
                     "have produced micrometre columns at a scale this run never chose, "
                     "beside a manifest saying the images were measured in pixels.")})

    if a.dry_run:
        print(f"semcrack --dry-run\n  images     : {len(names)}\n  input      : {a.indir}\n"
              f"  output     : {a.outdir}\n  model      : {PROD_MODEL_PATH}\n"
              f"  threshold  : {eff_threshold}  ({eff_why})\n"
              f"  detector   : {'pipeline + SAM 2 (' + a.sam2 + ')' if a.sam2 != 'off' else 'pipeline only'}\n"
              f"  corrections: {a.corrections or 'NONE (detector only)'}\n"
              f"  calibrated : {sum(1 for n in names if cal.get_um_per_px(n))}/{len(names)}\n"
              f"  refusals   : {len(refusals)}")
        for r in refusals[:10]:
            print(f"    - {r['kind']}: {r.get('image', '')}")
        return EXIT_OK

    print(f"semcrack {VERSION}: {len(names)} image(s) -> {a.outdir}", flush=True)
    if a.jobs > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.jobs) as pool:
            results = pool.map(_measure_one, names)
    else:
        results = [_measure_one(n) for n in names]
    for r in results:
        tag = (f"{r['n_cracks']:6d} cracks ({r['n_censored']} censored) [{r['units']}]"
               if r["status"] == "ok" else f"FAILED {r['error'][:70]}")
        print(f"  {r['image'][:44]:46s} {tag}", flush=True)

    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] != "ok"]

    # ---- combined table + cross-image aggregate ----
    import pandas as pd
    frames = []
    for r in ok:
        p = os.path.join(a.outdir, f"{r['image']}_crack_measurements.csv")
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    if frames:
        allp = os.path.join(a.outdir, "all_cracks.csv")
        pd.concat(frames, ignore_index=True).to_csv(allp, index=False)
        print(f"  -> {allp}")
    try:
        # aggregate reads OUT_DIR from the same SEMCRACK_MEASUREMENTS_DIR this run set, so
        # it groups the CSVs just written rather than the repository's.
        import aggregate as ag
        # Check the READ path, not just the write path. Asserting only on OUT_DIR passed
        # while aggregate read the repository's CSVs through a second constant and wrote a
        # header-only file here -- a guard that could not fail, certifying nothing.
        for _nm in ("OUT_DIR", "MEAS_DIR"):
            _d = getattr(ag, _nm, None)
            assert _d and os.path.abspath(_d) == a.outdir, (
                f"aggregate.{_nm} is {_d}, not {a.outdir}")
        _by = () if a.group_by.strip().lower() in ("none", "") else tuple(
            x.strip() for x in a.group_by.split(",") if x.strip())
        res = ag.aggregate([r["image"] for r in ok], by=_by, require_calibrated=False)
        ag.write_report(res)
        _ng = len(res.get("groups", []))
        print(f"  -> {os.path.join(a.outdir, 'aggregate.csv')} ({_ng} group(s))")
        if ok and not _ng:
            # An empty aggregate beside 60 measured frames is a bug or an unparseable
            # naming convention, and either way it must not read as a success.
            refusals.append({
                "kind": "aggregate_empty",
                "note": f"{len(ok)} image(s) measured but no group was formed. Grouping "
                        f"parses family/condition/specimen out of the FILENAME; names that "
                        f"do not follow that convention produce no groups. The per-image "
                        f"CSVs and all_cracks.csv are complete and unaffected."})
            print(f"  aggregate produced NO groups from {len(ok)} measured image(s) -- "
                  f"filenames did not parse into {a.group_by}. Re-run with "
                  f"--group-by none for a single pooled group.")
    except Exception as e:
        # Grouping depends on parsing a filename convention that a stranger's files will
        # not follow. That must not lose the per-image tables, which are the real output.
        refusals.append({"kind": "aggregate_skipped", "error": f"{type(e).__name__}: {e}",
                         "note": "per-image CSVs and all_cracks.csv are unaffected; "
                                 "cross-image grouping needs filenames this run could "
                                 "not parse into family/condition/specimen"})
        print(f"  aggregate skipped: {type(e).__name__}: {e}")

    manifest = {
        "tool": "semcrack", "version": VERSION, "git_sha": _git_sha(root),
        "python": sys.version.split()[0],
        "input_dir": a.indir, "output_dir": a.outdir, "glob": a.glob,
        "model_path": PROD_MODEL_PATH,
        "model_sha256": fp["model_sha256"],
        "threshold": eff_threshold,
        "threshold_as_given": a.threshold,
        "threshold_note": eff_why,
        "sam2_mode": a.sam2,
        "sam2_note": ("SAM 2 refined the detector's candidate boundaries; measured on this "
                      "corpus that beats the shipped detector on f1, recall, specificity and "
                      "precision. Per-CSV provenance records it per image."
                      if a.sam2 == "refine" else
                      "SAM 2's mask was unioned with the detector's: higher f1, lower "
                      "specificity than refine." if a.sam2 == "hybrid" else
                      "shipped detector only; no SAM 2"),
        "corrections_dir": a.corrections,
        "corrections_applied": bool(a.corrections),
        "corrections_note": ("human corrections from the named directory OVERRIDE the "
                             "detector, so these numbers are part human judgement"
                             if a.corrections else
                             "detector only; no human correction was applied"),
        "n_images": len(names), "n_ok": len(ok), "n_failed": len(failed),
        "n_calibrated": sum(1 for r in ok if r.get("units") == "um"),
        "group_by": a.group_by,
        "config_fingerprint": fp,
        "images": results, "refusals": refusals,
    }
    json.dump(manifest, open(prev_path, "w"), indent=1)
    print(f"  -> {prev_path}")
    print(f"  {len(ok)} measured, {len(failed)} failed, {len(refusals)} refusal(s), "
          f"{manifest['n_calibrated']} calibrated")
    if failed:
        print(f"semcrack: {len(failed)} image(s) failed; see run_manifest.json",
              file=sys.stderr)
        return EXIT_FAILURES
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
