"""
Endpoints the TXM-style frontend needs and this app did not have.

Ported because the sibling TXM app's current UI depends on them, and matching
its layout without them would mean shipping dead controls:

    GET  /api/thumb/<image>?w=128   sidebar preview with the result burned in
    GET  /api/raw/<image>           the image WITHOUT the overlay ("Show result" off)
    GET  /api/models                every model available, current one flagged
    POST /api/model/select          switch/roll back to one of them
    POST /api/reapply               re-render every image with the current model
    POST /api/remove/<image>        take an image out of the app

Thumbnails are generated once and cached on disk. TXM's own docstring records
why they exist: its list pointed each row's <img> at the full-resolution PNG,
"a 2.5 MB full-resolution PNG decoded down to a 38 px box, once per row. At 71
images that is ~180 MB of transfer to draw a sidebar." This app's templates are
6-33 MB, so the same mistake would be worse. A thumb is ~6 KB.

Removing an image MOVES it to removed_images/ rather than deleting, matching how
image removal was already handled in this project, so a misclick is recoverable.
"""
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "code"))

import numpy as np
from flask import jsonify, request, send_file
from PIL import Image

from common import ORIGINAL_DIR, PAINT_DIR, PROJECT_ROOT, PROD_MODEL_PATH, contrast_kwargs_for
from detect_cracks import load_as_uint8, find_field_of_view

Image.MAX_IMAGE_PIXELS = None
THUMB_DIR = os.path.join(PAINT_DIR, ".thumbs")
MODELS_TOP = os.path.join(PROJECT_ROOT, "models")
REMOVED_DIR = os.path.join(PROJECT_ROOT, "removed_images")
CODE_DIR = os.path.dirname(os.path.abspath(__file__))


def _template(image_name):
    return os.path.join(PAINT_DIR, f"{image_name}_paint_template.png")


def _crack_mask_from_template(image_name):
    p = _template(image_name)
    if not os.path.exists(p):
        return None
    a = np.array(Image.open(p).convert("RGB")).astype(np.int16)
    d = a[..., 0] - a[..., 1]
    return (a[..., 1] == a[..., 2]) & ((d == 140) | (d == 141))


def _prewarm_thumbs(list_images, widths=(128,)):
    """Generate missing thumbnails in the background at server start.

    Each thumbnail costs a full decode of a 6-33 MB template PNG, so 62 rows
    requesting theirs at once takes minutes and the sidebar renders blank in the
    meantime. Doing it on a daemon thread means the list fills in progressively
    instead of stalling, and every later load is a cache hit.
    """
    import threading

    def work():
        try:
            names = list_images()
        except Exception:
            return
        os.makedirs(THUMB_DIR, exist_ok=True)
        made = 0
        for n in names:
            tpl = _template(n)
            if not os.path.exists(tpl):
                continue
            for w in widths:
                out = os.path.join(THUMB_DIR, f"{n}_{w}.png")
                if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(tpl):
                    continue
                try:
                    im = Image.open(tpl).convert("RGB")
                    im.thumbnail((w, w), Image.LANCZOS)
                    im.save(out, format="PNG")
                    made += 1
                except Exception:
                    pass
        if made:
            print(f"prewarmed {made} sidebar thumbnail(s)")

    t = threading.Thread(target=work, daemon=True)
    t.start()


def register(app, list_images, invalidate_stage):
    _prewarm_thumbs(list_images)

    @app.route("/api/thumb/<image_name>")
    def api_thumb(image_name):
        w = max(32, min(int(request.args.get("w", 128)), 512))
        os.makedirs(THUMB_DIR, exist_ok=True)
        tpl = _template(image_name)
        if not os.path.exists(tpl):
            return jsonify({"ok": False, "error": "not processed"}), 404
        out = os.path.join(THUMB_DIR, f"{image_name}_{w}.png")
        # regenerate only when the template is newer than the cached thumb
        if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(tpl):
            im = Image.open(tpl).convert("RGB")
            im.thumbnail((w, w), Image.LANCZOS)
            im.save(out, format="PNG")
        return send_file(out, mimetype="image/png")

    @app.route("/api/raw/<image_name>")
    def api_raw(image_name):
        """The field-of-view-cropped image with no overlay, so the result can be
        toggled off and the underlying microstructure inspected. Same crop as the
        template, otherwise the two would not register when swapped."""
        src = os.path.join(ORIGINAL_DIR, f"{image_name}.tif")
        if not os.path.exists(src):
            return jsonify({"ok": False, "error": "no such image"}), 404
        os.makedirs(THUMB_DIR, exist_ok=True)
        out = os.path.join(THUMB_DIR, f"{image_name}_raw.png")
        if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(src):
            img8 = load_as_uint8(src, **contrast_kwargs_for(image_name))
            x0, y0, x1, y1 = find_field_of_view(img8)
            Image.fromarray(img8[y0:y1, x0:x1]).save(out, format="PNG", compress_level=1)
        return send_file(out, mimetype="image/png")

    @app.route("/api/models")
    def api_models():
        """Every classifier on disk, newest first, with the live one flagged.

        Rollback previously meant finding a .joblib by hand and copying it over
        crack_classifier.joblib. The retrain gate already refuses to deploy a
        worse model, but that does not help when a deployed model turns out to
        be wrong for reasons the held-out image cannot see.
        """
        import joblib
        out = []
        for f in sorted(os.listdir(MODELS_TOP)):
            if not f.endswith(".joblib"):
                continue
            p = os.path.join(MODELS_TOP, f)
            rec = {"file": f, "mtime": os.path.getmtime(p), "size": os.path.getsize(p),
                   "is_current": f == "crack_classifier.joblib"}
            try:
                b = joblib.load(p)
                rec.update({"family": b.get("model_family", type(b.get("clf")).__name__),
                            "threshold": float(b.get("threshold", 0.5)),
                            "n_train": int(b.get("n_train", 0)),
                            "n_images": len(b.get("images", []) or [])})
                cv = (b.get("cv_results") or {}).get(b.get("model_family", ""), {}) or {}
                if cv.get("loio_auc_exhaustive_image") is not None:
                    rec["held_out_auc"] = round(float(cv["loio_auc_exhaustive_image"]), 4)
            except Exception as e:
                rec["error"] = f"{type(e).__name__}"
            out.append(rec)
        out.sort(key=lambda r: (not r["is_current"], -r["mtime"]))
        return jsonify({"ok": True, "models": out})

    @app.route("/api/model/select", methods=["POST"])
    def api_model_select():
        f = (request.get_json(silent=True) or {}).get("file", "")
        src = os.path.join(MODELS_TOP, os.path.basename(f))
        if not f.endswith(".joblib") or not os.path.exists(src):
            return jsonify({"ok": False, "error": "no such model"}), 404
        if os.path.basename(f) == "crack_classifier.joblib":
            return jsonify({"ok": True, "message": "already current"})
        # keep the outgoing model recoverable
        stamp = time.strftime("%Y%m%d_%H%M%S")
        shutil.copy2(PROD_MODEL_PATH,
                     os.path.join(MODELS_TOP, f"crack_classifier_replaced_{stamp}.joblib"))
        shutil.copy2(src, PROD_MODEL_PATH)
        if invalidate_stage:
            invalidate_stage(None)
        return jsonify({"ok": True,
                        "message": f"now using {os.path.basename(f)}; overlays are stale "
                                   f"until you re-apply"})

    @app.route("/api/reapply", methods=["POST"])
    def api_reapply():
        from app_endpoints import _new_job, _run_bg
        from hybrid_detect import sam_available
        want_sam = bool((request.get_json(silent=True) or {}).get("use_sam", False))
        if want_sam and not sam_available():
            return jsonify({"ok": False, "error": "SAM is not installed"}), 409
        jid = _new_job("reapply")

        def work(report):
            import subprocess
            # Pass SAM through. Previously this always ran the pipeline-only
            # renderer, so re-applying after processing images WITH SAM silently
            # threw those regions away and downgraded f1 from 0.776 to 0.715
            # with nothing on screen to say so.
            cmd = ["python3", "regenerate_templates.py"] + (["--with-sam"] if want_sam else [])
            report(stage="reapply", frac=0.05,
                   note=("re-rendering every image WITH SAM (~3 min each)" if want_sam
                         else "re-rendering every image, pipeline only (~40s each)"))
            r = subprocess.run(cmd, cwd=CODE_DIR,
                               capture_output=True, text=True, timeout=86400)
            if r.returncode != 0:
                raise RuntimeError((r.stderr or "")[-500:])
            if invalidate_stage:
                invalidate_stage(None)
            tail = [l for l in (r.stdout or "").splitlines() if "regenerated" in l]
            return {"message": tail[-1] if tail else "re-applied"}

        _run_bg(jid, work)
        return jsonify({"ok": True, "job": jid})

    @app.route("/api/install_sam", methods=["POST"])
    def api_install_sam():
        """Install the optional SAM dependencies into the running virtualenv.

        This existed only as a line in the README telling the user to open a
        terminal and run pip -- which is the difference between the default
        detector (f1 0.715) and the best one measured (0.776). For a local
        single-user tool there is no reason that has to be a manual step.

        Installs into sys.executable's own environment, so it lands in the venv
        ./run created rather than wherever a system pip points. torchvision is
        included because SAM's mask post-processing calls its NMS -- omitting it
        produces an import error at first predict, not at install.
        """
        from app_endpoints import _new_job, _run_bg
        from hybrid_detect import sam_available
        if sam_available():
            return jsonify({"ok": True, "already": True, "message": "SAM is already available"})
        jid = _new_job("install_sam")

        def work(report):
            import subprocess
            report(stage="install", frac=0.05,
                   note="downloading PyTorch, transformers, torchvision (~2.5 GB, several minutes)")
            r = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                                "torch", "transformers", "torchvision"],
                               capture_output=True, text=True, timeout=7200)
            if r.returncode != 0:
                raise RuntimeError((r.stderr or "")[-600:])
            report(stage="install", frac=0.9, note="verifying")
            chk = subprocess.run([sys.executable, "-c",
                                  "import torch, transformers, torchvision; print('ok')"],
                                 capture_output=True, text=True, timeout=600)
            ok = "ok" in (chk.stdout or "")
            return {"installed": ok,
                    "message": ("SAM installed. It applies to images you process from now on; "
                                "use Re-apply model to redo existing ones.")
                               if ok else "install finished but the import check failed",
                    "restart_needed": True}

        _run_bg(jid, work)
        return jsonify({"ok": True, "job": jid})

    @app.route("/api/remove/<image_name>", methods=["POST"])
    def api_remove(image_name):
        src = os.path.join(ORIGINAL_DIR, f"{image_name}.tif")
        if not os.path.exists(src):
            return jsonify({"ok": False, "error": "no such image"}), 404
        d = os.path.join(REMOVED_DIR, image_name)
        os.makedirs(d, exist_ok=True)
        moved = []
        # Move, never delete: the same convention used when images were removed
        # from this project before, so a misclick is a mv away from undone.
        for p in [src, _template(image_name),
                  os.path.join(PAINT_DIR, f"{image_name}_painted.png")]:
            if os.path.exists(p):
                shutil.move(p, os.path.join(d, os.path.basename(p)))
                moved.append(os.path.basename(p))
        # correction masks are labels -- keep them where they are, so re-adding
        # the image restores its verdicts
        cp = os.path.join(PAINT_DIR, "candidate_counts.json")
        if os.path.exists(cp):
            try:
                c = json.load(open(cp))
                c.pop(image_name, None)
                json.dump(c, open(cp, "w"), indent=2)
            except Exception:
                pass
        if invalidate_stage:
            invalidate_stage(image_name)
        return jsonify({"ok": True, "moved": moved, "to": d,
                        "note": "corrections kept; restore by moving the .tif back"})

    return app
