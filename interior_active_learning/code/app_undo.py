"""
Server-side undo of a COMMITTED correction.

Autosave created a hole this fills. Before autosave, a mistaken stroke could be
taken back with Cmd-Z because nothing was written until Save was pressed. Now a
stroke commits about a second after drawing stops, the page reloads to show the
committed result, and that reload resets the browser's stroke-layer undo stack --
so the mark was permanent and there was no way back. The strokes are baked into
the server's correction mask by then, which the client cannot reach.

The sibling TXM app has exactly this endpoint (/api/image/<iid>/undo, popping a
correction delta under the same lock as painting), which is what prompted
checking for it here.

Approach: snapshot the correction mask immediately BEFORE each ingest, keep a
bounded per-image stack of those snapshots on disk, and pop one to restore.
Snapshots are the mask PNGs themselves -- ~17 KB each, so a 10-deep stack per
image is trivial, unlike the 20-33 MB templates which are regenerated instead
of stored.

"No mask existed yet" is recorded explicitly rather than as an absent file, so
undoing the very first correction on an image removes the mask rather than
leaving the previous state guessed at.

    POST /api/undo_correction/<image_name>   -> job id
    GET  /api/undo_depth/<image_name>
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import jsonify

from common import PAINT_DIR

UNDO_ROOT = os.path.join(PAINT_DIR, ".undo")
MAX_DEPTH = 10
NONE_MARKER = "__none__"


def _dir(image_name):
    return os.path.join(UNDO_ROOT, image_name)


def _mask_path(image_name):
    return os.path.join(PAINT_DIR, f"{image_name}_correction_mask.png")


def _entries(image_name):
    d = _dir(image_name)
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.endswith((".png", ".none")))


def depth(image_name):
    return len(_entries(image_name))


def snapshot(image_name):
    """Record the pre-ingest correction state. Cheap and safe to call often;
    failures are swallowed because losing an undo step must never block a save."""
    try:
        d = _dir(image_name)
        os.makedirs(d, exist_ok=True)
        existing = _entries(image_name)
        n = 0
        if existing:
            n = max(int(f.split(".")[0]) for f in existing) + 1
        src = _mask_path(image_name)
        # Skip a snapshot identical to the one on top. Every commit used to push an
        # entry even when the ingest turned out to have nothing to ingest -- which
        # is the normal outcome of pressing Cmd-Z inside the 1.1 s autosave window --
        # so the next Cmd-Z "restored" a byte-identical mask while the UI reported
        # success, and ten such commits evicted the one snapshot that mattered.
        top = existing[-1] if existing else None
        if top is not None:
            tp = os.path.join(d, top)
            same = ((not os.path.exists(src) and top.endswith(".none"))
                    or (os.path.exists(src) and top.endswith(".png")
                        and os.path.getsize(tp) == os.path.getsize(src)
                        and open(tp, "rb").read() == open(src, "rb").read()))
            if same:
                return True
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(d, f"{n:04d}.png"))
        else:
            open(os.path.join(d, f"{n:04d}.none"), "w").close()
        # trim oldest
        while len(_entries(image_name)) > MAX_DEPTH:
            oldest = _entries(image_name)[0]
            os.remove(os.path.join(d, oldest))
        return True
    except Exception as e:
        print(f"undo snapshot failed for {image_name}: {type(e).__name__}: {e}")
        return False


def pop(image_name):
    """Restore the most recent snapshot. Returns (ok, message).

    Runs under the same per-image lock the writers hold. Without it an undo that
    overlapped an in-flight autosave could be overwritten by that save the moment
    it finished, so the restore was silently discarded while the UI reported it had
    worked.
    """
    from common import mask_lock
    with mask_lock(image_name):
        ents = _entries(image_name)
        if not ents:
            return False, "nothing to undo"
        latest = ents[-1]
        path = os.path.join(_dir(image_name), latest)
        dest = _mask_path(image_name)
        try:
            if latest.endswith(".none"):
                if os.path.exists(dest):
                    os.remove(dest)
                msg = "reverted to no corrections on this image"
            else:
                shutil.copy2(path, dest)
                msg = "previous corrections restored"
            # only now, once the restore is actually on disk
            os.remove(path)
            return True, msg
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"


def register(app, invalidate_stage):
    @app.route("/api/undo_depth/<image_name>")
    def api_undo_depth(image_name):
        return jsonify({"ok": True, "depth": depth(image_name)})

    @app.route("/api/undo_correction/<image_name>", methods=["POST"])
    def api_undo_correction(image_name):
        from app_endpoints import _new_job, _run_bg
        if depth(image_name) == 0:
            return jsonify({"ok": False, "error": "nothing left to undo"}), 409
        jid = _new_job("undo", image_name)

        def work(report):
            report(stage="undo", frac=0.2, note="restoring previous corrections")
            ok, msg = pop(image_name)
            if not ok:
                raise RuntimeError(msg)
            # The cached stage and the on-disk template both describe the state
            # we just reverted, so both must be rebuilt or the UI keeps showing
            # the undone marks -- the same staleness that has bitten this
            # project at three other layers.
            if invalidate_stage:
                invalidate_stage(image_name)
            report(stage="undo", frac=0.5, note="re-rendering overlay")
            from interior_candidates import build_simple_overlay
            from unified_pipeline import run_unified_pipeline
            import json
            stage = run_unified_pipeline(image_name)
            build_simple_overlay(stage).save(
                os.path.join(PAINT_DIR, f"{image_name}_paint_template.png"))
            df = stage["df"]
            counts_path = os.path.join(PAINT_DIR, "candidate_counts.json")
            counts = {}
            if os.path.exists(counts_path):
                try:
                    counts = json.load(open(counts_path))
                except Exception:
                    counts = {}
            counts[image_name] = {"n_candidates": int(len(df)),
                                  "n_crack": int(df["IsCrack"].sum())}
            json.dump(counts, open(counts_path, "w"), indent=2)
            if invalidate_stage:
                invalidate_stage(image_name)
            return {"image": image_name, "message": msg,
                    "n_candidates": int(len(df)), "n_crack": int(df["IsCrack"].sum()),
                    "undo_depth": depth(image_name)}

        _run_bg(jid, work)
        return jsonify({"ok": True, "job": jid})

    return app
