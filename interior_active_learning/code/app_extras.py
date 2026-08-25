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
    # Every caller of this helper reads the file, so the barrier belongs here rather than
    # repeated at each call site. See template_writer.
    import template_writer
    return template_writer.path_for_read(
        os.path.join(PAINT_DIR, f"{image_name}_paint_template.png"))


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
                elif b.get("loio_out_of_sample") is not None:
                    # FALL BACK so every row can show a score. The deployed bundle carries
                    # loio_out_of_sample but no model_family, so the cv lookup above missed and
                    # the LIVE model displayed no AUC at all -- while a freshly trained
                    # candidate displayed one. A reviewer deciding whether to try the candidate
                    # was shown its number and nothing to compare it against.
                    rec["held_out_auc"] = round(float(b["loio_out_of_sample"]), 4)
                    rec["held_out_auc_note"] = (
                        "measured when this model was trained, on the corpus as it stood then; "
                        "not necessarily the same held-out rows as a newer candidate")
                # NOT surfacing prod_auc_on_loio_image here, deliberately. It is the model
                # that was live at training time re-scored on the candidate's held-out image --
                # an image that model was TRAINED on. train_v3_weighted.py prints it with the
                # words "optimistic -- it was trained with this image's labels". Putting it in
                # this list beside two honest out-of-sample figures invites exactly the
                # in-sample-vs-out-of-sample comparison the promotion gate exists to prevent.
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
        # Second-resolution stamps collide: two selects inside the same second
        # produced the same backup name, so the second overwrote the first and the
        # model being replaced stopped being recoverable. Also refuse to write the
        # backup over the file we are about to read.
        stamp = time.strftime("%Y%m%d_%H%M%S")
        backup = os.path.join(MODELS_TOP, f"crack_classifier_replaced_{stamp}.joblib")
        _n = 1
        while os.path.exists(backup):
            backup = os.path.join(MODELS_TOP, f"crack_classifier_replaced_{stamp}_{_n}.joblib")
            _n += 1
        if os.path.realpath(backup) == os.path.realpath(src):
            return jsonify({"ok": False,
                            "error": "refusing to overwrite the model being selected"}), 400
        shutil.copy2(PROD_MODEL_PATH, backup)
        shutil.copy2(src, PROD_MODEL_PATH)
        # copy2 preserves the SOURCE's mtime, so rolling back to an older model could
        # move production's mtime backwards and nothing downstream looked stale.
        os.utime(PROD_MODEL_PATH, None)
        if invalidate_stage:
            invalidate_stage(None)
        return jsonify({"ok": True,
                        "message": f"now using {os.path.basename(f)}; overlays are stale "
                                   f"until you re-apply"})

    @app.route("/api/reapply", methods=["POST"])
    def api_reapply():
        # Symmetric with /api/retrain. The guard was one-directional: retrain
        # refused while a reapply ran, but a reapply could start mid-retrain and
        # re-render every overlay from a model that was being replaced underneath it.
        from app_endpoints import _running_job_of_kind as _busy_of
        _busy = _busy_of("retrain", "reapply")
        if _busy is not None:
            return jsonify({"ok": False, "error": f"a {_busy['kind']} job is "
                            f"already running ({_busy['stage']})",
                            "job": _busy["id"]}), 409
        from app_endpoints import _new_job, _run_bg
        from hybrid_detect import sam_available
        # Defaults to SAM when SAM is installed. It used to default to False, so a
        # Re-apply silently downgraded every overlay from f1 0.776 to 0.715 unless the
        # user happened to opt in -- which is how the shipped overlays lost their SAM
        # regions in the first place.
        from hybrid_detect import sam_available as _sam_ok
        _body = request.get_json(silent=True) or {}
        want_sam = bool(_body.get("use_sam", _sam_ok()))
        if want_sam and not sam_available():
            return jsonify({"ok": False, "error": "SAM is not installed"}), 409
        jid = _new_job("reapply")

        def work(report):
            import subprocess
            # Pass SAM through. Previously this always ran the pipeline-only
            # renderer, so re-applying after processing images WITH SAM silently
            # threw those regions away and downgraded f1 from 0.776 to 0.715
            # with nothing on screen to say so.
            cmd = [sys.executable, "regenerate_templates.py"] + (["--with-sam"] if want_sam else [])
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
                # Keep the entry rather than popping it. Removal is reversible by design --
                # the correction mask deliberately stays -- but dropping the counts made a
                # restored image read "not processed yet, 0 candidates", contradicting the
                # note this endpoint returns ("restore by moving the .tif back").
                pass
                # Atomic: this was rewritten in place, so a concurrent reader (the
                # sidebar polls /api/images) could catch a 0-byte file and show every
                # image as having no candidates.
                from common import save_json_atomic
                save_json_atomic(c, cp)
            except Exception:
                pass
        if invalidate_stage:
            invalidate_stage(image_name)
        return jsonify({"ok": True, "moved": moved, "to": d,
                        "note": "corrections kept; restore by moving the .tif back"})

    return app
