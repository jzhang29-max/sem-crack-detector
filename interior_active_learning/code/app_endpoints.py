"""
The endpoints that turn the paint tool into a self-contained app: drag-and-drop
upload, hybrid detection with live progress, and one-click retrain.

Kept in its own module and registered onto the existing Flask app rather than
edited into paint_server.py. paint_server's painting/ingest path has been
debugged carefully (click isolation, the stage cache, correction-mask
precedence) and there is no reason to risk it to add uploads.

Long jobs run on a background thread and report through a small in-process job
registry, so the browser can poll rather than hold a multi-minute request open.
SAM takes ~3 min/image, which no browser should be asked to wait on
synchronously.

Registered routes:
    POST /api/upload            multipart file(s) -> converted into original/
    POST /api/process/<name>    run detection, write the paint template
    GET  /api/job/<job_id>      progress for either of the above
    POST /api/retrain           rebuild training data, retrain, regenerate
    GET  /api/pipeline_info     what model is live, and is SAM usable
"""
import os
import shutil
import sys
import threading
import time
import traceback
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image
from flask import jsonify, request

from common import ORIGINAL_DIR, PAINT_DIR, PROJECT_ROOT, PROD_MODEL_PATH, VERSION

# Serialises upload name reservation; see _ingest_upload.
_UPLOAD_LOCK = threading.Lock()

Image.MAX_IMAGE_PIXELS = None
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
COUNTS_PATH = os.path.join(PAINT_DIR, "candidate_counts.json")
ALLOWED = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

_jobs = {}
_jobs_lock = threading.Lock()


def label_balance():
    """How the not-crack labels are distributed across images.

    This matters for interpreting every reported score. One image
    (AS_24hr_BSE_Side_008) currently supplies 1023 of 1084 negatives -- 94.4% -- and 22 of
    32 labelled images contain no not-crack label at all. The number the retrain result
    used to print as "held-out AUC" is train-without-that-image / test-on-that-image, so it
    largely measures "can the model rank the one frame that has negatives", and the model
    actually shipped is refit on all rows including it. Surfacing the concentration is what
    lets a reader discount the figure appropriately instead of taking 0.915 at face value.
    """
    import csv as _csv
    path = os.path.join(PROJECT_ROOT, "training_data", "labeled_regions.csv")
    if not os.path.exists(path):
        return None
    per, total_neg, images = {}, 0, set()
    try:
        with open(path, newline="") as fh:
            for row in _csv.DictReader(fh):
                img = row.get("SourceImage", "")
                images.add(img)
                v = str(row.get("IsCrack", "")).strip().lower()
                if v in ("false", "0", "f", "no"):
                    per[img] = per.get(img, 0) + 1
                    total_neg += 1
    except (OSError, ValueError):
        return None
    if not total_neg:
        return {"total_negatives": 0, "n_images": len(images),
                "images_with_negatives": 0,
                "warning": "no not-crack labels at all: specificity is unmeasurable"}
    top_img, top_n = max(per.items(), key=lambda kv: kv[1])
    frac = top_n / total_neg
    out = {"total_negatives": total_neg, "n_images": len(images),
           "images_with_negatives": len(per),
           "top_image": top_img, "top_image_negatives": top_n,
           "top_image_share": frac}
    if frac > 0.5:
        out["warning"] = (
            f"{top_n} of {total_neg} not-crack labels ({100 * frac:.0f}%) come from a "
            f"single image ({top_img}), and only {len(per)} of {len(images)} labelled "
            f"images carry any. Specificity and any held-out AUC therefore describe that "
            f"one frame more than the corpus. Marking not-crack regions on more images is "
            f"the single highest-value labelling you can do.")
    return out


def comparable_baseline(bundle, candidate_image):
    """The deployed bundle's out-of-sample baseline, if it is comparable. (value, reason)

    Pulled out of the retrain closure for the same reason promotion_decision was: the
    logic was reachable only by actually retraining, so nothing tested it. It also makes
    "a bundle carrying a baseline must name its source" a rule the code ENFORCES rather
    than one a comment asserts -- the test for that invariant used to build a dict literal
    containing the key and then check the key was there, which could not fail.

    Every refusal returns None with a reason, and the reason is surfaced in the API
    response, because a baseline silently ignored looks identical to no baseline at all.
    """
    if not isinstance(bundle, dict):
        return None, "absent"
    v = bundle.get("loio_out_of_sample")
    if v is None:
        return None, "absent"
    if not bundle.get("loio_out_of_sample_source"):
        # A figure with no provenance cannot be distinguished from a refit-same-family
        # estimate, which is not the shipped weights' out-of-sample performance at all.
        return None, ("ignored: the bundle records a baseline without naming its source, "
                      "so it cannot be told apart from a refit estimate")
    base_img = bundle.get("loio_out_of_sample_image")
    if not base_img:
        return None, ("ignored: the bundle records a baseline without saying which image "
                      "it was held out on, so comparability cannot be established")
    # A baseline measured on a different held-out image is a different quantity, and
    # comparing across images is the class of error this gate exists to prevent.
    if candidate_image and base_img != candidate_image:
        return None, (f"ignored: baseline was measured on {base_img} but the candidate was "
                      f"scored on {candidate_image}")
    try:
        return float(v), "loio_out_of_sample recorded in the deployed bundle"
    except (TypeError, ValueError):
        return None, "ignored: the recorded baseline is not a number"


def promotion_decision(new_loio, cur_loio):
    """Should a freshly trained candidate replace production? (promote, reason)

    Pulled out of the retrain closure so it can be tested without a 10-minute training
    run -- the previous version was only reachable by actually retraining, which is why
    a fail-OPEN gate survived in the codebase unnoticed.

    The rule that matters: a MISSING baseline is a refusal, not a pass. It used to read
    `cur_loio is None or new_loio >= cur_loio`, and because the deployed bundle carries no
    cv_results the left branch was always true, so every retrain was promoted while the UI
    promised a held-out comparison.
    """
    if new_loio is None:
        return False, "new model reported no held-out AUC; not deployed"
    if cur_loio is None:
        return False, (
            "NOT deployed: the deployed model has no recorded out-of-sample score "
            "(loio_out_of_sample), so there is nothing valid to compare against -- "
            "comparing against its in-sample score would bias every retrain toward "
            "refusal. Run `python3 code/establish_baseline.py` once to measure and "
            "record it, then retrain again. Production left unchanged.")
    if new_loio >= cur_loio - 1e-9:
        return True, None       # caller fills in the reason, it names the backup file
    return False, (f"NOT deployed: held-out AUC {new_loio:.4f} is worse than "
                   f"current {cur_loio:.4f}; production left unchanged")


def _running_job_of_kind(*kinds):
    """A job of one of these kinds that is still running, or None.

    Retrain is not re-entrant: two concurrent runs share CODE_DIR and the same
    crack_classifier_v3_weighted.joblib intermediate, so the second overwrites the
    first's candidate mid-promotion. That also defeats the atomic promote and the
    timestamped PREV backup, which is why this guard has to exist for those to mean
    anything.
    """
    with _jobs_lock:
        for j in _jobs.values():
            if j.get("kind") in kinds and j.get("state") == "running":
                return j
    return None


def _new_job(kind, note=""):
    jid = uuid.uuid4().hex[:12]
    with _jobs_lock:
        # Finished records are never read again after the frontend stops polling, but they
        # were kept for the life of the process -- one per detect, export, reapply and
        # retrain across a long session. Reap the settled ones on the way in.
        # Reap on FINISH time, not start time. A retrain that ran for over an hour was
        # eligible the instant it completed, so its result could be deleted before the UI
        # polled for it -- the user would see the job vanish rather than a verdict. Also
        # keep the most recent retrain regardless of age, because that record is the only
        # place the promotion decision and its reason are reported.
        now = time.time()
        newest_retrain = None
        for k, j in _jobs.items():
            if j.get("kind") == "retrain":
                if newest_retrain is None or (j.get("finished") or j.get("started", 0)) > (
                        _jobs[newest_retrain].get("finished")
                        or _jobs[newest_retrain].get("started", 0)):
                    newest_retrain = k
        stale = [k for k, j in _jobs.items()
                 if j.get("state") in ("done", "error")
                 and k != newest_retrain
                 and (now - (j.get("finished") or j.get("started", 0))) > 3600]
        for k in stale:
            del _jobs[k]
        _jobs[jid] = {"id": jid, "kind": kind, "state": "running", "frac": 0.0,
                      "stage": "starting", "note": note, "started": time.time(),
                      "result": None, "error": None}
    return jid


def _update(jid, **kw):
    with _jobs_lock:
        if jid in _jobs:
            _jobs[jid].update(kw)


def _run_bg(jid, fn):
    def wrapper():
        try:
            res = fn(lambda **kw: _update(jid, **kw))
            # Stamp when it FINISHED. Reaping is keyed on this; without it the fallback to
            # "started" made a long retrain eligible for deletion the moment it completed.
            _update(jid, state="done", frac=1.0, result=res, finished=time.time())
        except Exception as e:
            _update(jid, state="error", error=f"{type(e).__name__}: {e}",
                    note=traceback.format_exc()[-1200:], finished=time.time())
    t = threading.Thread(target=wrapper, daemon=True)
    t.start()
    return t


def _safe_name(fn):
    base = os.path.splitext(os.path.basename(fn))[0]
    keep = "".join(c if (c.isalnum() or c in "_-") else "_" for c in base)
    return keep or "image"


def _ingest_upload(storage):
    """Convert an uploaded file into original/<name>.tif.

    Everything downstream (load_as_uint8, contrast_kwargs_for) expects a TIFF in
    original/, so uploads are normalised here rather than teaching the whole
    pipeline about other formats. Multi-page TIFFs keep page 0; anything with
    colour is converted to greyscale, because every feature in the model is
    intensity-based.
    """
    import tifffile
    name = _safe_name(storage.filename)
    ext = os.path.splitext(storage.filename)[1].lower()
    if ext not in ALLOWED:
        raise ValueError(f"unsupported file type {ext}")
    # A name is "taken" if ANYTHING still belongs to it, not just the .tif.
    # /api/remove deliberately leaves the correction mask behind so a misclick is
    # recoverable -- but the collision check only looked at original/, so a later
    # upload could seize that name and inherit a stranger's hand-marked labels,
    # silently transplanting them onto different pixels.
    from common import _correction_mask_path
    import template_writer as _tw
    def _taken(nm):
        return (os.path.exists(os.path.join(ORIGINAL_DIR, f"{nm}.tif"))
                or os.path.exists(_correction_mask_path(nm))
                or os.path.exists(_tw.path_for_read(
                    os.path.join(PAINT_DIR, f"{nm}_paint_template.png"))))

    # Reserve the name under a lock and create the file immediately with O_EXCL.
    # Two uploads of the same filename used to both pass the existence check and
    # then both write the same destination, so one image vanished with ok:true.
    with _UPLOAD_LOCK:
        cand, n = name, 1
        while _taken(cand):
            cand = f"{name}_{n}"
            n += 1
        dest = os.path.join(ORIGINAL_DIR, f"{cand}.tif")
        os.close(os.open(dest, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    final_name = cand

    tmp = os.path.join(PROJECT_ROOT, f".upload_{uuid.uuid4().hex[:8]}{ext}")
    part = dest + ".part"
    storage.save(tmp)
    try:
        # Image.MAX_IMAGE_PIXELS is set to None project-wide (app_endpoints.py:41)
        # because the real corpus is up to 27 megapixels and PIL's default guard trips at
        # ~89 megapixels wholesale. That disables the decompression-bomb check for
        # UPLOADS too, where it is genuinely wanted: a ~6 KB PNG whose header declares
        # 60000x60000 asks the allocator for ~3.6 GB in one go at im.convert("L") and
        # takes the Flask process with it, losing every warm stage and any in-flight
        # retrain. Read the declared size from the header first -- this does not decode
        # pixels -- and refuse before allocating.
        # SIZE GUARD, BEFORE ANY DECODE.
        #
        # Image.MAX_IMAGE_PIXELS is None project-wide (line 41) because the real corpus
        # reaches 27 megapixels and PIL's default guard trips wholesale -- which also
        # disables the decompression-bomb check for uploads, where it is wanted.
        #
        # Two earlier versions of this guard failed OPEN on the format the app is built
        # around. The first only asked PIL. The second asked tifffile only when PIL failed
        # to read a size -- but PIL reads multi-page TIFFs fine and reports PAGE 0, so a
        # 40-page 3000x3000 file (360 MP total, 0.4 MB on the wire because it compresses)
        # measured as 9 MP and sailed through. tifffile, not PIL, is what decodes .tif
        # below, so for TIFFs the page total is what matters and it is always computed.
        # TiffFile reads the directory without materialising pixels.
        _MAX_UPLOAD_MPX = 300
        _px = 0
        if ext in (".tif", ".tiff"):
            try:
                with tifffile.TiffFile(tmp) as _tf:
                    _px = sum(int(np.prod(pg.shape)) for pg in _tf.pages if pg.shape)
            except Exception as _e:
                # Unreadable as a TIFF -- fail CLOSED rather than hand it to the decoder.
                raise ValueError(f"could not read TIFF structure: {type(_e).__name__}")
        else:
            try:
                with Image.open(tmp) as _probe:
                    _w, _h = _probe.size
                _px = int(_w) * int(_h)
            except Exception as _e:
                raise ValueError(f"could not read image header: {type(_e).__name__}")
        if _px > _MAX_UPLOAD_MPX * 1_000_000:
            raise ValueError(
                f"{_px / 1e6:.0f} megapixels across all pages, over the "
                f"{_MAX_UPLOAD_MPX} MP upload limit")
        if ext in (".tif", ".tiff"):
            arr = np.asarray(tifffile.imread(tmp))
            # Collapse PAGE axes only. `while arr.ndim > 2: arr = arr[0]` treated an
            # interleaved colour axis as a page axis, so an (H, W, 3) RGB TIFF became
            # (W, 3) -- row 0 only, axes transposed, 0.15% of a 6.3-megapixel frame --
            # and detection then "succeeded" on that strip with no warning anywhere.
            while arr.ndim > 3:
                arr = arr[0]
            if arr.ndim == 3:
                if arr.shape[-1] in (2, 3, 4):        # interleaved colour / alpha
                    rgb = arr[..., :3]
                    if rgb.dtype == np.uint8:
                        arr = np.asarray(Image.fromarray(rgb).convert("L"))
                    else:                              # 16-bit colour: luminance by hand
                        arr = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1]
                               + 0.114 * rgb[..., 2]).astype(rgb.dtype)
                else:                                  # a genuine page/plane axis
                    arr = arr[0]
            if arr.ndim != 2:
                raise ValueError(f"unsupported TIFF shape {arr.shape}")
            tifffile.imwrite(part, arr)
        else:
            im = Image.open(tmp)
            if im.mode not in ("L", "I;16", "I"):
                im = im.convert("L")
            tifffile.imwrite(part, np.array(im))
        os.replace(part, dest)
    except Exception:
        # Do not leave the reserved zero-byte placeholder behind: the sidebar would
        # list it as an image and every click on it would fail.
        for f in (part, dest):
            if os.path.exists(f):
                os.remove(f)
        raise
    finally:
        for f in (tmp,):
            if os.path.exists(f):
                os.remove(f)
    return final_name


def register(app, get_stage, invalidate_stage=None):
    """Attach the app routes. get_stage/invalidate_stage come from paint_server
    so uploads and retrains can drop its cached pipeline results -- otherwise a
    freshly retrained model would keep rendering from a stale cached stage,
    which is exactly the class of bug that made regions appear to change colour
    on click earlier in this project."""

    @app.route("/api/external_mask/<image_name>", methods=["GET"])
    def api_get_external_mask(image_name):
        """Whether this image's mask came from another tool, and from which."""
        import external_mask as _em
        return jsonify({"ok": True, "external": _em.has_external(image_name),
                        **_em.provenance_for(image_name)})

    @app.route("/api/external_mask/<image_name>", methods=["POST"])
    def api_set_external_mask(image_name):
        """Import a crack mask produced elsewhere (ilastik, micro-sam, Fiji, anything).

        This is the composability path: the surveyed alternatives all segment better than
        the built-in detector, and everything this project does that they do not --
        refusing calibration, unreviewed-aware metrics, a gated retrain, per-CSV provenance
        -- is downstream of the mask. So take the better mask.

        Multipart with a `mask` file, or JSON {"clear": true} to fall back to the detector.
        """
        import external_mask as _em
        if (request.get_json(silent=True) or {}).get("clear"):
            return jsonify({"ok": True, "cleared": _em.clear(image_name)})
        f = request.files.get("mask") or request.files.get("file")
        if f is None:
            return jsonify({"ok": False, "error": "no mask file in request"}), 400
        tool = (request.form.get("source_tool") or "custom").strip()
        note = (request.form.get("note") or "").strip()
        tmp = os.path.join(PROJECT_ROOT, f".extmask_{uuid.uuid4().hex[:8]}")
        f.save(tmp)
        try:
            from paint_server import get_stage
            shape = get_stage(image_name)["labeled"].shape
            rec = _em.store(image_name, tmp, tool, shape, note=note)
        except ValueError as e:
            # A shape mismatch or a wrong-layer export is the user's to fix, not a 500.
            return jsonify({"ok": False, "error": str(e)}), 409
        except Exception as e:
            return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 400
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return jsonify({"ok": True, "record": rec})

    @app.route("/api/calibration/<image_name>", methods=["GET"])
    def api_get_calibration(image_name):
        """The image's um/px and where it came from, or calibrated=false."""
        import calibration as _cal
        rec = _cal.get_record(image_name)
        return jsonify({"ok": True, "calibrated": bool(rec), "record": rec})

    @app.route("/api/calibration/<image_name>", methods=["POST"])
    def api_set_calibration(image_name):
        """Set or clear the calibration.

        Three routes, mirroring calibration.py: scale_bar (label plus the two marked x
        positions), hfw, and manual. The scale_bar route accepts an optional hfw_um and
        will REFUSE a pair that disagrees by more than 5% -- that is the whole point, so
        the refusal is surfaced as a 409 with the two values rather than swallowed.
        """
        import calibration as _cal
        body = request.get_json(silent=True) or {}
        if body.get("clear"):
            return jsonify({"ok": True, "cleared": _cal.clear(image_name)})
        mode = body.get("mode")
        try:
            if mode == "scale_bar":
                rec = _cal.set_from_scale_bar(
                    image_name, float(body["label_um"]), float(body["x1"]),
                    float(body["x2"]),
                    hfw_um=(float(body["hfw_um"]) if body.get("hfw_um") else None),
                    image_width_px=(int(body["image_width_px"])
                                    if body.get("image_width_px") else None))
            elif mode == "hfw":
                rec = _cal.set_from_hfw(image_name, float(body["hfw_um"]),
                                        int(body["image_width_px"]))
            elif mode == "manual":
                rec = _cal.set_manual(image_name, float(body["um_per_px"]),
                                      body.get("note", ""))
            else:
                return jsonify({"ok": False, "error":
                                "mode must be scale_bar, hfw or manual"}), 400
        except ValueError as e:
            # A cross-check failure is not a server fault and not a silent success --
            # 409 so the UI can show the two disagreeing numbers and let the user redo
            # the marks or correct the label.
            return jsonify({"ok": False, "error": str(e)}), 409
        except (KeyError, TypeError) as e:
            return jsonify({"ok": False, "error": f"missing or bad field: {e}"}), 400
        return jsonify({"ok": True, "record": rec})

    #: Performance the app can state on screen, read from the committed experiment
    #: artifacts rather than recomputed. Cached on the artifacts' mtimes because
    #: benchmark_results.json is ~190 KB of per-fold predictions and /api/pipeline_info
    #: is polled. Returns None rather than raising when an artifact is absent: a clone
    #: has them, but the app must still start for someone who deleted one.
    _PERF_CACHE = {}

    def _perf_summary():
        import json as _json
        exp = os.path.join(PROJECT_ROOT, "interior_active_learning", "code", "experiments")
        bench = os.path.join(exp, "benchmark_results.json")
        hyb = os.path.join(exp, "sam2_hybrid_report.json")
        try:
            key = (os.path.getmtime(bench), os.path.getmtime(hyb))
        except OSError:
            return None
        if _PERF_CACHE.get("key") == key:
            return _PERF_CACHE.get("val")
        out = {}
        try:
            d = _json.load(open(bench))
            FAM = "LogisticRegression"        # the family that ships
            folds = d["oof"][FAM]
            rec, spec, prec, bal = [], [], [], []
            for k in sorted(folds):
                yt = np.array(folds[k]["y_true"]); yp = np.array(folds[k]["y_pred"])
                tp = int(((yp == 1) & (yt == 1)).sum()); fp = int(((yp == 1) & (yt == 0)).sum())
                fn = int(((yp == 0) & (yt == 1)).sum()); tn = int(((yp == 0) & (yt == 0)).sum())
                r = tp / (tp + fn) if tp + fn else float("nan")
                sp = tn / (tn + fp) if tn + fp else float("nan")
                pr = tp / (tp + fp) if tp + fp else float("nan")
                rec.append(r); spec.append(sp); prec.append(pr); bal.append((r + sp) / 2)
            out["grouped_cv"] = {
                "model": FAM,
                "auc": round(float(d["results"][FAM]["auc_mean"]), 3),
                "auc_sd": round(float(d["results"][FAM]["auc_std"]), 3),
                "balacc": round(float(np.mean(bal)), 3),
                "balacc_sd": round(float(np.std(bal)), 3),
                "balacc_worst": round(float(np.min(bal)), 3),
                "recall": round(float(np.mean(rec)), 3),
                "specificity": round(float(np.mean(spec)), 3),
                "precision": round(float(np.mean(prec)), 3),
                "n_regions": int(d["n_examples"]), "n_pos": int(d["n_pos"]),
                "n_neg": int(d["n_neg"]), "n_groups": int(d["n_groups"]),
                "repeats": len(folds),
            }
        except Exception:
            pass
        try:
            h = _json.load(open(hyb))
            m = h["means"]["pipeline"]
            out["pixel"] = {k: round(float(m[k]), 3)
                            for k in ("f1", "recall", "specificity", "precision") if k in m}
            out["pixel"]["n_frames"] = int(h.get("n_frames") or 0)
        except Exception:
            pass
        # STATED, not omitted. The sibling TXM app shows false indications per frame on
        # crack-free specimens; this corpus has no frame established as crack-free, so the
        # row would have no denominator. Saying so beats leaving a gap to be noticed.
        # WHICH MODEL DID THESE DESCRIBE? held_out_auc is read live from the deployed bundle,
        # but grouped_cv and pixel come from committed artifacts measured against whatever was
        # deployed then. Retrain and the panel would show one fresh number beside two
        # describing the previous model, silently -- the exact failure this project refuses.
        #
        # Compared by DECISION SURFACE, not by file mtime. An mtime comparison cries stale for
        # a metadata-only rewrite: recording the held-out baseline rewrote the bundle on
        # 2026-08-20 without changing a single coefficient. Artifacts written before this
        # existed carry no fingerprint, and "unknown" is reported as unknown rather than
        # guessed either way.
        try:
            import hashlib as _h
            import joblib as _jl
            _bb = _jl.load(PROD_MODEL_PATH)
            _ee = _bb.get("clf") or _bb
            if hasattr(_ee, "steps"):
                _ee = _ee.steps[-1][1]
            _live_fp = _h.sha256(
                np.round(np.concatenate([np.ravel(_ee.coef_), np.ravel(_ee.intercept_),
                                         [float(_bb.get("threshold", 0.5))]]), 8).tobytes()
            ).hexdigest()[:12]
            out["model_fingerprint"] = _live_fp
            for _k, _f in (("grouped_cv", bench), ("pixel", hyb)):
                if _k not in out:
                    continue
                try:
                    _art = _json.load(open(_f))
                except Exception:
                    continue
                _afp = ((_art.get("detector") or {}).get("model_fingerprint")
                        or (_art.get("detector_arms") or {}).get("bare", {}).get("model_fingerprint"))
                out[_k]["describes_this_model"] = (None if not _afp else _afp == _live_fp)
        except Exception:
            pass
        out["false_calls"] = None
        out["false_calls_reason"] = (
            "no frame in this corpus is established as crack-free, so a false-call rate has "
            "no denominator. Two masks hold zero crack marks, but one is 100% UNREVIEWED and "
            "the other 99.89%. Specificity is the nearest substitute.")
        _PERF_CACHE.update(key=key, val=out or None)
        return out or None


    @app.route("/api/pipeline_info")
    def api_pipeline_info():
        import joblib
        from hybrid_detect import sam_available
        info = {"sam_available": sam_available(), "version": VERSION, "model": None}
        try:
            b = joblib.load(PROD_MODEL_PATH)
            info["model"] = {
                "family": b.get("model_family", type(b["clf"]).__name__),
                "threshold": float(b.get("threshold", 0.5)),
                "n_train": int(b.get("n_train", 0)),
                "n_images": len(b.get("images", []) or []),
                "per_image_weights": bool(b.get("per_image_weights", False)),
                "mtime": os.path.getmtime(PROD_MODEL_PATH),
            }
            # A model is a pickled scikit-learn estimator, and sklearn compares the
            # version it was pickled under against the running one on EVERY load,
            # warning "this might lead to breaking code or invalid results". Every
            # module here calls warnings.filterwarnings("ignore"), so that warning
            # was swallowed: the shipped models were built on 1.7.2 while
            # requirements.txt's `scikit-learn>=1.7` installs whatever is newest,
            # so every clone silently ran a mismatch. Report it instead of hiding
            # it -- the bundles now record the version they were saved under.
            # The gate's own bar. It was recorded in the bundle and surfaced only in the
            # model-picker label, so the card itself said nothing about how well the model does.
            for _src, _dst in (("loio_out_of_sample", "held_out_auc"),
                               ("loio_in_sample_for_reference", "in_sample_auc")):
                _v = b.get(_src)
                if isinstance(_v, (int, float)):
                    info["model"][_dst] = round(float(_v), 4)
            for _src, _dst in (("loio_out_of_sample_image", "held_out_image"),
                               ("loio_out_of_sample_holdout_kind", "held_out_kind")):
                if b.get(_src):
                    info["model"][_dst] = str(b[_src])
            info["performance"] = _perf_summary()
            
            import sklearn
            built = b.get("sklearn_version")
            info["model"]["sklearn_built"] = built
            info["label_balance"] = label_balance()
            info["model"]["sklearn_running"] = sklearn.__version__
            if built and built != sklearn.__version__:
                info["model"]["version_mismatch"] = (
                    f"model pickled with scikit-learn {built}, running "
                    f"{sklearn.__version__} -- predictions may differ; re-save with "
                    f"python3 code/resave_models.py")
        except Exception as e:
            info["error"] = str(e)
        return jsonify(info)

    @app.route("/api/upload", methods=["POST"])
    def api_upload():
        files = request.files.getlist("files") or request.files.getlist("file")
        if not files:
            return jsonify({"ok": False, "error": "no files in request"}), 400
        added, failed = [], []
        for f in files:
            try:
                added.append(_ingest_upload(f))
            except Exception as e:
                failed.append({"file": f.filename, "error": str(e)})
        return jsonify({"ok": bool(added), "added": added, "failed": failed})

    @app.route("/api/process/<image_name>", methods=["POST"])
    def api_process(image_name):
        use_sam = bool((request.get_json(silent=True) or {}).get("use_sam", True))
        if not os.path.exists(os.path.join(ORIGINAL_DIR, f"{image_name}.tif")):
            return jsonify({"ok": False, "error": "no such image"}), 404
        jid = _new_job("process", image_name)

        def work(report):
            from hybrid_detect import render_and_record

            def prog(stage, frac, note):
                report(stage=stage, frac=float(frac), note=note)

            # Shared with regenerate_templates.py so the interactive and batch
            # paths cannot diverge again.
            res = render_and_record(image_name, use_sam=use_sam, progress=prog)
            report(stage="rendering", frac=0.98, note="writing overlay")
            if invalidate_stage:
                invalidate_stage(image_name)
            try:
                import paint_server
                paint_server.warm_stage_async(image_name)
            except Exception:
                pass
            return res

        _run_bg(jid, work)
        return jsonify({"ok": True, "job": jid})

    @app.route("/api/retrain", methods=["POST"])
    def api_retrain():
        body = request.get_json(silent=True) or {}
        regen = bool(body.get("regenerate", True))
        regen_sam = bool(body.get("regenerate_with_sam", False))
        # Single-flight. A double-click, a second tab, or a Retrain fired while a
        # Re-apply is still running previously started a second training run against the
        # same intermediate files.
        busy = _running_job_of_kind("retrain", "reapply")
        if busy is not None:
            return jsonify({"ok": False, "error": f"a {busy['kind']} job is already "
                            f"running ({busy['stage']})", "job": busy["id"]}), 409
        jid = _new_job("retrain")

        def work(report):
            import subprocess
            out = {}
            steps = [("building training data from your corrections",
                      ["python3", "build_training_data.py"], 0.35),
                     ("retraining and calibrating the model",
                      ["python3", "train_v3_weighted.py"], 0.70)]
            for note, cmd, frac in steps:
                report(stage="retrain", frac=frac, note=note)
                r = subprocess.run(cmd, cwd=CODE_DIR, capture_output=True, text=True,
                                   timeout=7200)
                out[cmd[1]] = (r.stdout or "")[-3000:]
                if r.returncode != 0:
                    # build_training_data.py prints its diagnostic ("NO ROWS
                    # PRODUCED ...") to stdout, so reporting stderr alone gave the
                    # user an empty reason for the failure.
                    _why = ((r.stderr or "").strip() or (r.stdout or "").strip()
                            or f"exit code {r.returncode}")
                    raise RuntimeError(f"{cmd[1]} failed: {_why[-600:]}")

            # train_v3_weighted writes a candidate model but deliberately does
            # NOT overwrite production. Promote it only if it is at least as
            # good out-of-sample on the held-out image, so a retrain on thin
            # new data cannot silently make detection worse -- that check
            # already caught one regression (LOIO 0.9153 vs 0.9252).
            report(stage="retrain", frac=0.80, note="checking the new model before deploying")
            import json as _json
            import joblib
            mj = os.path.join(PROJECT_ROOT, "models", "crack_classifier_v3_metrics.json")
            cand = os.path.join(PROJECT_ROOT, "models", "crack_classifier_v3_weighted.joblib")
            promoted = False
            reason = "no metrics file, left production unchanged"
            if os.path.exists(mj) and os.path.exists(cand):
                m = _json.load(open(mj))
                cv = (m.get("cv_results") or {}).get(m.get("best", ""), {})
                new_loio = cv.get("loio_auc_exhaustive_image")
                # BOTH SIDES MUST BE OUT-OF-SAMPLE, AND MISSING MUST MEAN REFUSE.
                #
                # Two wrong versions preceded this one. The first read the incumbent's
                # score from its own bundle and promoted when that came back None, so with
                # the archived bundle (4 keys, no cv_results) every retrain was promoted
                # unconditionally while the UI promised a held-out comparison.
                #
                # The second was mine, and worse because it looked principled: it compared
                # against production_on_held_out.auc, which is the DEPLOYED model scored on
                # the held-out image it was TRAINED ON. train_v3_weighted.py prints
                # "optimistic -- it was trained with this image's labels" beside that number
                # (line 162) and "in-sample and optimistic. Compare LOIO AUC instead."
                # (line 245). The gap is visible on disk: retrained_on_held_out 0.9153
                # against production_on_held_out 0.9535. Grading an honest out-of-sample
                # candidate against an in-sample incumbent biases the gate to refuse EVERY
                # retrain, permanently and silently -- the user paints for weeks, clicks
                # Retrain, and is told their labels made it worse when nothing about their
                # labels was measured.
                #
                # The comparable quantity is each model's own out-of-sample LOIO, recorded
                # in its bundle at promotion time. A bundle without it cannot be compared,
                # so promotion is refused -- but refusing FOREVER would brick retraining,
                # which is what code/establish_baseline.py exists to unblock.
                cur_loio = None
                _cur_src = "absent"
                try:
                    _cand_img = m.get("held_out_image") or cv.get("loio_image")
                    cur_loio, _cur_src = comparable_baseline(
                        joblib.load(PROD_MODEL_PATH), _cand_img)
                except Exception:
                    pass
                out["baseline_source"] = _cur_src
                out["loio_new"], out["loio_current"] = new_loio, cur_loio
                # The single-image LOIO is the best-case figure. Report the grouped
                # cross-image estimate beside it, and say what each one is, so the user is
                # not handed 0.915 when the corpus-level number is 0.676 +/- 0.119.
                out["pooled_auc"] = cv.get("pooled_auc")
                out["pooled_auc_std"] = cv.get("pooled_auc_std")
                out["spec_cross_image"] = cv.get("spec")
                out["metric_note"] = (
                    "loio_* is train-without-one-image / test-on-that-image, the "
                    "best-case figure and the one the promotion gate compares. "
                    "pooled_auc is the grouped cross-image estimate and is the number to "
                    "quote. The deployed model is refit on all rows including the held-out "
                    "image, so neither figure is a property of the shipped model.")
                out["label_balance"] = label_balance()
                _go, _why = promotion_decision(new_loio, cur_loio)
                if not _go:
                    reason = _why
                else:
                    # Keep every superseded model. This used to write one fixed
                    # crack_classifier_PREV.joblib, so a second Retrain overwrote the only
                    # recoverable copy with the first retrain's output and the model the
                    # user had been working with was gone from disk entirely.
                    _stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
                    shutil.copy2(PROD_MODEL_PATH, os.path.join(
                        PROJECT_ROOT, "models", f"crack_classifier_PREV-{_stamp}.joblib"))
                    shutil.copy2(PROD_MODEL_PATH, os.path.join(
                        PROJECT_ROOT, "models", "crack_classifier_PREV.joblib"))
                    # Atomic: a plain copy2 onto the live path leaves a window where
                    # /api/pipeline_info and every pipeline run can load a truncated pickle.
                    _tmp = PROD_MODEL_PATH + ".tmp"
                    shutil.copy2(cand, _tmp)
                    os.replace(_tmp, PROD_MODEL_PATH)
                    promoted = True
                    reason = (f"deployed: held-out AUC {new_loio:.4f} >= current "
                              f"{round(cur_loio, 4)}; previous model kept as "
                              f"crack_classifier_PREV-{_stamp}.joblib")
            out["promoted"], out["reason"] = promoted, reason

            if regen and promoted:
                report(stage="regenerate", frac=0.85,
                       note=("re-rendering every image with the new model"
                             + (" WITH SAM (~3 min each)" if regen_sam else " (~40s each)")))
                cmd = ["python3", "regenerate_templates.py"] + (["--with-sam"] if regen_sam else [])
                r = subprocess.run(cmd, cwd=CODE_DIR,
                                   capture_output=True, text=True, timeout=86400)
                out["regenerate_templates.py"] = (r.stdout or "")[-2000:]
                # regenerate_templates.py exits 0 even when individual images fail,
                # and this never checked the status either, so a re-render that
                # rendered nothing reported success and left every overlay stale.
                _failed = [ln for ln in (r.stdout or "").splitlines()
                           if "FAIL" in ln or "Traceback" in ln]
                if r.returncode != 0 or _failed:
                    out["regenerate_warning"] = (
                        f"re-render reported {len(_failed)} failure(s)"
                        + (f", exit code {r.returncode}" if r.returncode else "")
                        + "; some overlays may still be stale")
                if invalidate_stage:
                    invalidate_stage(None)
            return out

        _run_bg(jid, work)
        return jsonify({"ok": True, "job": jid})

    @app.route("/api/job/<job_id>")
    def api_job(job_id):
        with _jobs_lock:
            j = _jobs.get(job_id)
        if not j:
            return jsonify({"ok": False, "error": "unknown job"}), 404
        return jsonify({"ok": True, **j})

    return app
