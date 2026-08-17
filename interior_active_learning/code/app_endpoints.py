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

from common import ORIGINAL_DIR, PAINT_DIR, PROJECT_ROOT, PROD_MODEL_PATH

Image.MAX_IMAGE_PIXELS = None
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
COUNTS_PATH = os.path.join(PAINT_DIR, "candidate_counts.json")
ALLOWED = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

_jobs = {}
_jobs_lock = threading.Lock()


def _new_job(kind, note=""):
    jid = uuid.uuid4().hex[:12]
    with _jobs_lock:
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
            _update(jid, state="done", frac=1.0, result=res)
        except Exception as e:
            _update(jid, state="error", error=f"{type(e).__name__}: {e}",
                    note=traceback.format_exc()[-1200:])
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
    dest = os.path.join(ORIGINAL_DIR, f"{name}.tif")
    n = 1
    while os.path.exists(dest):
        dest = os.path.join(ORIGINAL_DIR, f"{name}_{n}.tif")
        n += 1
    final_name = os.path.splitext(os.path.basename(dest))[0]

    tmp = os.path.join(PROJECT_ROOT, f".upload_{uuid.uuid4().hex[:8]}{ext}")
    storage.save(tmp)
    try:
        if ext in (".tif", ".tiff"):
            arr = tifffile.imread(tmp)
            arr = np.asarray(arr)
            while arr.ndim > 2:
                arr = arr[0]
            tifffile.imwrite(dest, arr)
        else:
            im = Image.open(tmp)
            if im.mode not in ("L", "I;16", "I"):
                im = im.convert("L")
            tifffile.imwrite(dest, np.array(im))
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return final_name


def register(app, get_stage, invalidate_stage=None):
    """Attach the app routes. get_stage/invalidate_stage come from paint_server
    so uploads and retrains can drop its cached pipeline results -- otherwise a
    freshly retrained model would keep rendering from a stale cached stage,
    which is exactly the class of bug that made regions appear to change colour
    on click earlier in this project."""

    @app.route("/api/pipeline_info")
    def api_pipeline_info():
        import joblib
        from hybrid_detect import sam_available
        info = {"sam_available": sam_available(), "model": None}
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
                    raise RuntimeError(f"{cmd[1]} failed: {(r.stderr or '')[-600:]}")

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
                cur = joblib.load(PROD_MODEL_PATH)
                cur_loio = (cur.get("cv_results", {}).get(cur.get("model_family", ""), {})
                            or {}).get("loio_auc_exhaustive_image")
                out["loio_new"], out["loio_current"] = new_loio, cur_loio
                if new_loio is None:
                    reason = "new model reported no held-out AUC; not deployed"
                elif cur_loio is None or new_loio >= cur_loio - 1e-9:
                    shutil.copy2(PROD_MODEL_PATH, os.path.join(
                        PROJECT_ROOT, "models", "crack_classifier_PREV.joblib"))
                    shutil.copy2(cand, PROD_MODEL_PATH)
                    promoted = True
                    reason = (f"deployed: held-out AUC {new_loio:.4f} >= current "
                              f"{cur_loio if cur_loio is None else round(cur_loio,4)}")
                else:
                    reason = (f"NOT deployed: held-out AUC {new_loio:.4f} is worse than "
                              f"current {cur_loio:.4f}; production left unchanged")
            out["promoted"], out["reason"] = promoted, reason

            if regen and promoted:
                report(stage="regenerate", frac=0.85,
                       note=("re-rendering every image with the new model"
                             + (" WITH SAM (~3 min each)" if regen_sam else " (~40s each)")))
                cmd = ["python3", "regenerate_templates.py"] + (["--with-sam"] if regen_sam else [])
                r = subprocess.run(cmd, cwd=CODE_DIR,
                                   capture_output=True, text=True, timeout=86400)
                out["regenerate_templates.py"] = (r.stdout or "")[-2000:]
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
