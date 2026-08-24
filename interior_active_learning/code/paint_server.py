"""
Local web app for painting crack-interior annotations directly in the
browser, instead of exporting a PNG, editing it in an external image
editor, and re-importing it. Run with:

    python3 paint_server.py

then open http://127.0.0.1:8765 in a browser.

The frontend paints on a transparent canvas (only strokes, no base image
pixels) and POSTs just that layer -- the server composites it onto the
actual on-disk template and calls the exact same ingest() used by the CLI
tool, so behavior (color meaning, merging, feature extraction) is
identical either way.

The template is the production ("round4") red=crack/cyan=artifact overlay
only (see interior_candidates.build_simple_overlay), and painting uses the
SAME two colors: red=crack, cyan=not-crack. Painting one over blank area
adds a new candidate; painting it over an EXISTING opposite-colored
candidate corrects that candidate's verdict instead. The Eraser tool
removes a region from candidacy entirely (back to plain background)
rather than recoloring it -- for when a region is neither a real crack
nor a meaningful artifact.
"""
import os
import sys
import io
import base64
import threading
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_file, Response

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (PAINT_DIR, ORIGINAL_DIR, CANDIDATES_DIR, MODELS_DIR,
                    PROD_MODEL_PATH, load_correction_mask, save_correction_mask,
                    mask_lock, save_png_atomic)
from apply_paint_annotations import (
    make_template as _make_template, ingest as _ingest, RED, CYAN, ERASE_MARKER, _color_mask,
    _log_touched_labels, _log_interior_origin_corrections,
)
from interior_candidates import build_simple_overlay, apply_pixel_corrections
# UNIFIED MODEL EXPERIMENT: one shared model instead of two -- see
# unified_pipeline.py's module docstring. Otherwise identical to production.
from unified_pipeline import run_unified_pipeline as run_enhanced_pipeline
from paint_frontend import INDEX_HTML

app = Flask(__name__)
# Cap the request body. Without this an upload of any size is buffered to disk before
# anything looks at it. 2 GiB is well above the largest real capture here (a 6144x4376
# uint16 TIFF is ~54 MB uncompressed) while stopping a single request from filling the
# disk the correction masks live on.
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024


@app.errorhandler(413)
def _too_large(_e):
    """Say what happened. The cap is per REQUEST, and the frontend posts a whole drop in one
    request, so one oversized batch rejects every file in it -- and without this handler
    Flask returns an HTML error page that the uploader's response.json() cannot parse, so
    the UI showed nothing at all."""
    return jsonify({"ok": False, "error":
                    "this upload exceeds the 2 GB per-request limit. The limit is per "
                    "request, not per file, so drop the images in smaller batches."}), 413

# Caches the (labeled, df, ...) "stage" dict per image so clicking to flip a
# whole region -- or a plain Save & Ingest -- doesn't re-run the full
# production pipeline (background flattening, vesselness, ML classification,
# merging -- can take minutes on the largest images) every time. Both
# api_flip_region and api_ingest update a cached entry's labeled/df IN PLACE
# after applying their own correction, rather than dropping it, so the cache
# stays valid (and fast) across a whole editing session instead of paying for
# one full recompute per action.
#
# Each entry also remembers the interior_model.joblib mtime it was computed
# against (_model_mtime_at_cache) -- retraining the model doesn't touch this
# cache or any already-written *_paint_template.png on its own, so without
# checking this, an image opened before a retrain would keep showing the
# PREVIOUS model's guesses indefinitely, even after switching away and back,
# and a brand new server process would still serve a stale on-disk template
# that predates the latest retrain. See get_stage() and api_template() below.
_stage_cache = {}


def _model_path():
    # UNIFIED MODEL EXPERIMENT: staleness-detection now watches the one
    # shared model file instead of interior_model.joblib.
    return os.path.join(MODELS_DIR, "unified_model.joblib")


def _model_mtime():
    """Newest mtime across every model a render depends on, or None if none exist.

    This watched only unified_model.joblib (Pass 2). But an overlay also depends on
    the Pass 1 classifier, models/crack_classifier.joblib, which is what Retrain
    replaces and what the model picker swaps on a rollback -- so after switching
    models the cached stage and the on-disk template still looked fresh, and the
    next click rewrote the template from the OLD model's output.

    Rollback made that worse: api_model_select uses shutil.copy2, which preserves
    the source file's mtime, so restoring an older model could move the mtime
    BACKWARDS and nothing looked stale at all. Hence max() over both, and the
    select endpoint now touches the file it deploys.
    """
    times = []
    for p in (_model_path(), PROD_MODEL_PATH):
        if p and os.path.exists(p):
            times.append(os.path.getmtime(p))
    return max(times) if times else None


_warming = set()
_warm_lock = __import__("threading").Lock()


def warm_stage_async(image_name):
    """Build the pipeline stage for an image in the background.

    Autosave is only fast when this cache is warm. Measured on
    260622_316_H_b2_back_CBS_01 (6144x4096): with a warm stage an ingest is
    ~1.5s, but on a cache MISS run_unified_pipeline costs 203s -- and opening an
    image for viewing did not warm it, because api_template only builds a stage
    when the template file is missing or stale. So the first correction of every
    session paid a 3.5-minute pipeline run while the user waited.

    Warming starts the moment an image is opened, which is dead time anyway --
    nobody paints before looking. Guarded by a set so two rapid opens do not
    start two pipeline runs for the same image.
    """
    import threading
    with _warm_lock:
        if image_name in _stage_cache or image_name in _warming:
            return False
        _warming.add(image_name)

    def work():
        started_gen = _stage_gen
        try:
            # Do not warm an image that has since been removed, and discard the
            # result if anything invalidated the cache while we were building it.
            if not os.path.exists(os.path.join(ORIGINAL_DIR, f"{image_name}.tif")):
                return
            get_stage(image_name)
            if _stage_gen != started_gen:
                _stage_cache.pop(image_name, None)
            print(f"stage warm: {image_name}")
        except Exception as e:
            print(f"stage warm failed for {image_name}: {type(e).__name__}: {e}")
        finally:
            with _warm_lock:
                _warming.discard(image_name)

    threading.Thread(target=work, daemon=True).start()
    return True


def _deps_mtime(image_name):
    """Newest mtime of everything a rendered overlay depends on for THIS image.

    The models, plus the image's own correction mask. The mask matters because
    /api/stroke now commits a brush stroke straight into it without re-rendering the
    overlay -- that is what took an autosave from 8.0 s to ~0.1 s -- so the overlay
    is only correct if "the mask is newer than the overlay" counts as stale. Without
    this a reload showed the pre-stroke overlay and the mark looked lost, even though
    it was safely recorded.
    """
    times = []
    m = _model_mtime()
    if m is not None:
        times.append(m)
    mp = os.path.join(PAINT_DIR, f"{image_name}_correction_mask.png")
    if os.path.exists(mp):
        times.append(os.path.getmtime(mp))
    return max(times) if times else None


def _stage_is_fresh(image_name):
    """True only if get_stage would REUSE the cached entry.

    stage_ready used to be a bare `in _stage_cache`, but get_stage discards an entry
    whose model mtime has moved -- so after a retrain or a model switch the UI was
    told the next save would be fast when it was actually about to recompute the
    whole pipeline.
    """
    cached = _stage_cache.get(image_name)
    if cached is None:
        return False
    # Model only, deliberately. A brush stroke writes the correction mask, and if
    # that counted here every fast stroke would discard the warm stage and make the
    # next Whole-region click or undo pay a full pipeline run.
    m = _model_mtime()
    if m is None:
        return True
    return not (cached[1] is None or m > cached[1])


def stage_ready(image_name):
    return _stage_is_fresh(image_name)


_STAGE_LOCKS = {}
_STAGE_LOCKS_GUARD = threading.Lock()


def _stage_lock(image_name):
    """One lock per image, created on demand.

    Without this, get_stage was check-then-act with no mutual exclusion, and the app runs
    threaded=True with a background warm thread. Opening a large image while its warm-up
    was already in flight ran the whole ~200 s pipeline TWICE on the same frame -- the
    click waited ~400 s instead of reusing the warm result, and peak memory doubled on a
    6144x4096 frame, which on a laptop is an OOM kill that drops every warm stage.
    Guarded by its own lock so two threads cannot create two locks for one image.
    """
    with _STAGE_LOCKS_GUARD:
        lk = _STAGE_LOCKS.get(image_name)
        if lk is None:
            lk = _STAGE_LOCKS[image_name] = threading.Lock()
        return lk


def get_stage(image_name, force=False):
    def _fresh():
        cached = _stage_cache.get(image_name)
        m = _model_mtime()
        if cached is None:
            return None, m
        stale = m is not None and (cached[1] is None or m > cached[1])
        return (None if stale else cached[0]), m

    if not force:
        hit, _ = _fresh()
        if hit is not None:
            return hit
    with _stage_lock(image_name):
        # Re-check inside the lock: while this thread waited, the holder may have
        # installed exactly the entry it wanted, and the whole point is not to run the
        # pipeline a second time for it.
        if not force:
            hit, current_model_mtime = _fresh()
            if hit is not None:
                return hit
        else:
            current_model_mtime = _model_mtime()
        # run_enhanced_pipeline (not the bare production pipeline) so
        # whatever models/interior_model.joblib currently accepts renders as
        # plain red and can be corrected exactly like any other candidate --
        # see its docstring in interior_candidates.py.
        _stage_cache[image_name] = (run_enhanced_pipeline(image_name), current_model_mtime)
        return _stage_cache[image_name][0]


def list_images():
    return sorted(os.path.splitext(f)[0] for f in os.listdir(ORIGINAL_DIR) if f.lower().endswith(".tif"))


def _png_response(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return Response(buf.read(), mimetype="image/png")


_frontend_mtime = None


@app.route("/")
def index():
    """Serve the frontend, re-reading paint_frontend.py if it changed on disk.

    Without this, editing the frontend has no effect until the server process is
    restarted: INDEX_HTML is bound at import time, so a browser reload re-fetches
    the same stale string. That cost real debugging time -- two UI fixes appeared
    not to work and were investigated as bugs when the edited code simply was not
    being served. It is the same staleness that bit the paint templates and the
    stage cache, one level up.

    Cheap: one stat() per page load, and reimport only when the mtime moves.
    """
    global INDEX_HTML, _frontend_mtime
    try:
        import importlib
        import paint_frontend
        path = paint_frontend.__file__
        mtime = os.path.getmtime(path)
        if _frontend_mtime is None:
            _frontend_mtime = mtime
        elif mtime > _frontend_mtime:
            importlib.reload(paint_frontend)
            INDEX_HTML = paint_frontend.INDEX_HTML
            _frontend_mtime = mtime
            print(f"frontend reloaded from disk ({path})")
    except Exception as e:
        print(f"frontend reload skipped: {type(e).__name__}: {e}")
    return INDEX_HTML


@app.route("/api/images")
def api_images():
    names = list_images()

    # paint/candidate_counts.json is written when templates are generated
    # (seed_templates_with_steel_model.py). Prefer it: the original count
    # here came from candidates/<name>_interior.csv, which only exists once
    # run_all_candidates.py has produced INTERIOR candidates -- a step this
    # project has never run -- so every image read "0 candidates" even
    # though the images have hundreds of real segmentation candidates
    # apiece. Reporting 0 for an image with 1285 candidates is worse than
    # reporting nothing, since it reads as "nothing was detected here".
    counts_path = os.path.join(PAINT_DIR, "candidate_counts.json")
    counts = {}
    if os.path.exists(counts_path):
        try:
            import json as _json
            with open(counts_path) as f:
                counts = _json.load(f)
        except Exception:
            counts = {}

    info = []
    for name in names:
        entry = counts.get(name)
        if entry:
            info.append({"name": name,
                          "n_candidates": entry.get("n_candidates", 0),
                          "n_crack": entry.get("n_crack")})
            continue
        candidates_csv = os.path.join(CANDIDATES_DIR, f"{name}_interior.csv")
        n_candidates = 0
        if os.path.exists(candidates_csv):
            import pandas as pd
            n_candidates = len(pd.read_csv(candidates_csv))
        info.append({"name": name, "n_candidates": n_candidates, "n_crack": None})

    # Which images have no overlay yet. The package ships the source images but
    # NOT the templates (derived, and ~30 MB each), so on a fresh clone every
    # image is unrendered. /api/template would build one on demand inside the
    # request -- correct, but it blocks 40 s to 200 s with no progress, and
    # loadImageList opens the first image automatically, so the app looked hung
    # on first launch. The frontend uses this flag to run the background
    # detection job instead, which reports progress.
    for row in info:
        row["has_template"] = os.path.exists(
            os.path.join(PAINT_DIR, f"{row['name']}_paint_template.png"))
    return jsonify(info)


@app.route("/api/template/<image_name>")
def api_template(image_name):
    # Every sibling endpoint answers a bad name with a JSON 404; this one fell
    # through into the pipeline and returned a 500 traceback, which reads as "the
    # app is broken" rather than "no such image".
    if not os.path.exists(os.path.join(ORIGINAL_DIR, f"{image_name}.tif")):
        return jsonify({"ok": False, "error": "no such image"}), 404
    template_path = os.path.join(PAINT_DIR, f"{image_name}_paint_template.png")
    model_mtime = _model_mtime()
    is_stale = os.path.exists(template_path) and model_mtime is not None and \
        os.path.getmtime(template_path) < model_mtime
    # A mask newer than the overlay also makes it stale -- /api/stroke commits into
    # the mask without re-rendering -- but that case does NOT need a pipeline run:
    # the cached stage is still valid for the current model, so re-applying the mask
    # to it and redrawing costs about a second instead of ~200.
    mask_only_stale = False
    if os.path.exists(template_path) and not is_stale:
        _mp = os.path.join(PAINT_DIR, f"{image_name}_correction_mask.png")
        if os.path.exists(_mp) and os.path.getmtime(template_path) < os.path.getmtime(_mp):
            mask_only_stale = True
    regenerated = not os.path.exists(template_path) or is_stale or mask_only_stale
    if regenerated:
        # get_stage() itself already detects "this cache entry predates the
        # current model" -- force=True here additionally covers the case
        # where the CACHE is already fresh (this server process retrained
        # the model itself, e.g. via a script sharing this process) but the
        # on-disk template from a PRIOR process/session is still the stale
        # artifact -- without forcing, get_stage() would see a fresh cache
        # entry and skip recomputing, leaving this stale file untouched.
        if mask_only_stale and not is_stale:
            # Cheap path: keep the stage, re-apply the current mask, redraw.
            stage = get_stage(image_name)
            m = load_correction_mask(image_name, stage["labeled"].shape)
            if m is not None:
                lab, df = apply_pixel_corrections(stage["labeled"].copy(),
                                                  stage["df"].copy(), m)
                stage = dict(stage, labeled=lab, df=df)
        else:
            stage = get_stage(image_name, force=is_stale)
        build_simple_overlay(stage).save(template_path)
    # Opening an image is the right moment to start building its stage: the
    # user needs seconds to look before painting, and autosave is 130x faster
    # with a warm cache (1.5s vs 203s on the largest images).
    warm_stage_async(image_name)
    resp = send_file(template_path, mimetype="image/png")
    # Tells the frontend whether THIS load reflects a freshly retrained
    # model, purely so it can say so instead of silently swapping the
    # picture out from under you -- see loadImage()'s use of this header.
    resp.headers["X-Regenerated"] = "true" if regenerated else "false"
    return resp


def _resync_painted_file(image_name, old_template_arr, new_template_arr):
    """When the base template's own colors change under an ALREADY-SAVED
    painted.png (e.g. after a click-to-flip correction changes what color a
    region renders as), any pixel the user never actually hand-painted (i.e.
    it's still exactly the OLD template's color) has to be updated to the
    NEW template's color too. Otherwise the next ingest/resume would
    misread that now-stale leftover color as a fresh brush stroke against
    the new template baseline (since _color_mask only checks "does this
    differ from the CURRENT template", not "did the user actually paint
    this"). Pixels that are already genuinely painted (different from the
    old template) are left untouched."""
    painted_path = os.path.join(PAINT_DIR, f"{image_name}_painted.png")
    if not os.path.exists(painted_path):
        return
    painted = np.array(Image.open(painted_path).convert("RGB"))
    if painted.shape != old_template_arr.shape:
        return
    unpainted = np.all(painted == old_template_arr, axis=-1)
    if not unpainted.any():
        return
    painted[unpainted] = new_template_arr[unpainted]
    Image.fromarray(painted).save(painted_path)


@app.route("/api/stroke/<image_name>", methods=["POST"])
def api_stroke(image_name):
    """Commit one brush stroke as GEOMETRY, not as a re-uploaded canvas.

    Body: {mode: 'crack'|'not_crack'|'erase', points: [[x, y], ...], radius: int}

    Why this exists. The autosave path uploaded the whole 25-megapixel paint layer
    as a PNG dataURL, colour-matched it against the template three times to work out
    which pixels were new, re-rendered the overlay, wrote a 35.7 MB PNG, and then had
    the browser re-download that overlay. Measured on 6144x4096 for a single 40 px
    dot: /api/save 1.0 s, /api/ingest 7.0 s. None of that work is proportional to the
    stroke, which touches a few thousand pixels.

    A stroke's meaning is exactly a mask value, so stamping discs straight into the
    correction mask is both faster and more direct -- there is nothing to infer. The
    sibling TXM app does the same thing and that is why its autosave feels instant.

    The overlay is NOT re-rendered here. The user's own canvas already shows the
    stroke, and /api/template rebuilds the overlay when the mask is newer than it,
    so a reload still shows committed colours.
    """
    d = request.get_json(silent=True) or {}
    mode = d.get("mode", "crack")
    pts = d.get("points") or []
    r = max(1, int(d.get("radius", 20)))
    value = {"crack": 1, "not_crack": 2, "erase": 3}.get(mode)
    if value is None:
        return jsonify({"ok": False, "error": f"unknown mode {mode}"}), 400
    if not os.path.exists(os.path.join(ORIGINAL_DIR, f"{image_name}.tif")):
        return jsonify({"ok": False, "error": "no such image"}), 404
    if not pts:
        return jsonify({"ok": True, "changed": 0})

    # Shape from the template header -- PIL is lazy, so .size costs no decode. Falls
    # back to the cached stage only if there is no template yet.
    tpl = os.path.join(PAINT_DIR, f"{image_name}_paint_template.png")
    if os.path.exists(tpl):
        with Image.open(tpl) as im:
            W, H = im.size
    else:
        lab = get_stage(image_name)["labeled"]
        H, W = lab.shape

    with mask_lock(image_name):
        # One undo entry per stroke, taken before the change, like flip_region.
        try:
            import app_undo
            app_undo.snapshot(image_name)
        except Exception as e:
            print(f"undo snapshot skipped for {image_name}: {type(e).__name__}: {e}")

        m = load_correction_mask(image_name, (H, W))
        m = m.copy() if m is not None else np.zeros((H, W), dtype=np.uint8)
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        disk = (xx * xx + yy * yy) <= r * r
        changed = 0
        for pt in pts:
            try:
                x, y = int(round(float(pt[0]))), int(round(float(pt[1])))
            except Exception:
                continue
            y0, y1 = max(0, y - r), min(H, y + r + 1)
            x0, x1 = max(0, x - r), min(W, x + r + 1)
            if y0 >= y1 or x0 >= x1:
                continue
            sub = disk[(y0 - (y - r)):(y1 - (y - r)), (x0 - (x - r)):(x1 - (x - r))]
            before = m[y0:y1, x0:x1][sub]
            changed += int((before != value).sum())
            m[y0:y1, x0:x1][sub] = value
        save_correction_mask(image_name, m)

    # Deliberately does NOT invalidate the cached stage. Dropping it made every
    # fast stroke expensive for whatever came next: the stage is what Whole region
    # and undo's re-render need, and rebuilding it costs a full pipeline run (~200 s
    # on a 25-megapixel frame). Measured: a stroke took 0.1 s and then an undo took
    # 198 s, purely because the stroke had discarded the warm stage.
    #
    # Keeping it is sound. A brush stroke writes mask values; the mask is what
    # rendering and training read, and it is read fresh each time. The stage's
    # `labeled` map is only used to answer "which connected region did I click",
    # which a brush stroke does not change. The overlay itself is rebuilt on next
    # open because staleness now includes the mask mtime.
    return jsonify({"ok": True, "changed": changed,
                    "crack_px": int((m == 1).sum()), "not_crack_px": int((m == 2).sum()),
                    "erased_px": int((m == 3).sum())})


@app.route("/api/flip_region/<image_name>", methods=["POST"])
def api_flip_region(image_name):
    """Click-to-flip (or click-to-erase): correct/remove an ENTIRE existing
    candidate region in one action instead of having to brush over every
    one of its pixels by hand -- important when that region is huge (e.g.
    one connected component spanning most of the image's dark background,
    which is exactly what made painting-only correction impractical for
    large regions). Body: {x, y, mode}, mode is "toggle" (default -- flip
    crack<->not-crack) or "erase" (remove from candidacy entirely)."""
    data = request.get_json()
    x, y = int(round(data["x"])), int(round(data["y"]))
    mode = data.get("mode", "toggle")

    template_path = os.path.join(PAINT_DIR, f"{image_name}_paint_template.png")
    # Decode the old template ONLY when _resync_painted_file can actually use it: it returns
    # immediately unless <image>_painted.png exists, and 58 of the 62 shipped frames have no
    # painted file. Decoding a 22 MB PNG into a 72 MB array to hand it to a function that
    # discards it cost 0.28 s of the 1.43 s every Whole-region click took.
    _painted_exists = os.path.exists(os.path.join(PAINT_DIR, f"{image_name}_painted.png"))
    old_template_arr = (np.array(Image.open(template_path).convert("RGB"))
                        if _painted_exists and os.path.exists(template_path) else None)

    stage = get_stage(image_name)
    labeled, df = stage["labeled"], stage["df"]
    if not (0 <= y < labeled.shape[0] and 0 <= x < labeled.shape[1]):
        return jsonify({"ok": False, "error": "Click is outside the image"}), 400

    label = int(labeled[y, x])
    if label == 0:
        return jsonify({"ok": False, "error": "No candidate there -- click inside a red or cyan region"}), 400
    row = df[df["Label"] == label]
    if len(row) == 0:
        return jsonify({"ok": False, "error": "No candidate there"}), 400

    label_mask = labeled == label
    area = int(label_mask.sum())

    if mode == "erase":
        correction_value = 3
        forced_is_crack = None
    elif mode == "crack":
        forced_is_crack = True
        correction_value = 1
    elif mode == "not_crack":
        # Explicit SET, not a flip. A plain toggle can't express "I looked
        # at this already-cyan region and confirm it is not a crack":
        # toggling it would turn it red, and painting cyan over cyan is a
        # no-op by design (_color_mask ignores paint matching the existing
        # colour). Without this mode there is no way to record a negative
        # example for a region the pipeline defaulted to not-crack -- which
        # is exactly the data a classifier needs and exactly what this
        # dataset was missing (every label collected so far was positive).
        forced_is_crack = False
        correction_value = 2
    else:
        forced_is_crack = not bool(row.iloc[0]["IsCrack"])
        correction_value = 1 if forced_is_crack else 2

    click_mask = np.zeros(labeled.shape, dtype=np.uint8)
    click_mask[label_mask] = correction_value

    # apply to the cached in-memory stage so the regenerated template
    # reflects this immediately, without a full pipeline re-run
    new_labeled, new_df = apply_pixel_corrections(labeled, df, click_mask)
    stage["labeled"], stage["df"] = new_labeled, new_df

    # persist so ingest/quicklook/training (each a fresh pipeline run) see it too
    # Under the lock: this is a read-modify-write, and the frontend calls
    # flip_region straight from mousedown with no in-flight flag, so two quick
    # clicks arrive concurrently. Both used to read the same mask and both write,
    # so the second silently discarded the first click's verdict.
    with mask_lock(image_name):
        # Snapshot first, so Cmd-Z can reverse a Whole-region click. This path never
        # snapshotted, so the tool a reviewer uses most on large regions -- one click
        # setting thousands of pixels -- was the one edit with no way back, and an
        # undo pressed after it silently reverted an EARLIER brush correction instead.
        # Guarded like api_ingest's: losing an undo step must not block the edit.
        try:
            import app_undo
            app_undo.snapshot(image_name)
        except Exception as _e:
            print(f"undo snapshot skipped for {image_name}: {type(_e).__name__}: {_e}")
        existing = load_correction_mask(image_name, labeled.shape)
        merged = existing.copy() if existing is not None else np.zeros(labeled.shape, dtype=np.uint8)
        merged[label_mask] = correction_value
        save_correction_mask(image_name, merged)
    _log_touched_labels(image_name, [label], "erased" if mode == "erase" else forced_is_crack)
    if mode != "erase":
        _log_interior_origin_corrections(image_name, stage.get("interior_origin", {}), [label], forced_is_crack)

    new_template_img = build_simple_overlay(stage)
    if old_template_arr is not None:
        _resync_painted_file(image_name, old_template_arr, np.array(new_template_img))
    new_template_img.save(template_path)

    # Hand back ONLY the rectangle that changed. The client used to re-fetch
    # /api/template, which is the whole 15-31 MB overlay, and decode a 25-megapixel PNG, to
    # show an edit confined to one region's bounding box. A flip of a small region now sends
    # a few KB. A region spanning the frame sends about what it did before, which is the
    # honest worst case rather than a hidden one.
    ys, xs = np.where(label_mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    import base64, io as _io
    _buf = _io.BytesIO()
    new_template_img.crop((x0, y0, x1, y1)).save(_buf, format="PNG")
    _patch = base64.b64encode(_buf.getvalue()).decode("ascii")

    return jsonify({
        "ok": True,
        "label": label,
        "area": area,
        "erased": mode == "erase",
        "newIsCrack": forced_is_crack,
        "bbox": [x0, y0, x1 - x0, y1 - y0],
        "patch_png_b64": _patch,
        "patch_bytes": len(_patch),
    })


@app.route("/api/paintlayer/<image_name>")
def api_paintlayer(image_name):
    """If a previous painting session exists for this image, return just
    the painted strokes as a transparent-background PNG (diffed against
    the template) so the browser can resume from where it left off."""
    template_path = os.path.join(PAINT_DIR, f"{image_name}_paint_template.png")
    painted_path = os.path.join(PAINT_DIR, f"{image_name}_painted.png")
    if not os.path.exists(template_path) or not os.path.exists(painted_path):
        return ("", 204)

    template = np.array(Image.open(template_path).convert("RGB"))
    painted = np.array(Image.open(painted_path).convert("RGB"))
    if template.shape != painted.shape:
        return ("", 204)

    # _color_mask (shared with apply_paint_annotations.ingest()) excludes
    # any pixel already that color in the TEMPLATE -- this is what makes
    # "new red paint" mean only freshly-painted pixels, never pixels that
    # were already part of an existing red/crack region.
    red_mask = _color_mask(painted, template, RED)
    cyan_mask = _color_mask(painted, template, CYAN)
    erase_mask = _color_mask(painted, template, ERASE_MARKER)

    h, w = template.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[red_mask] = [255, 0, 0, 255]
    rgba[cyan_mask] = [0, 204, 255, 255]
    rgba[erase_mask] = [255, 0, 255, 255]
    return _png_response(Image.fromarray(rgba, mode="RGBA"))


@app.route("/api/save/<image_name>", methods=["POST"])
def api_save(image_name):
    """Body: {dataURL: 'data:image/png;base64,...'} -- a transparent PNG
    containing only the painted strokes (native resolution, same size as
    the template). Composited onto the actual on-disk template and saved
    as <image>_painted.png, which is exactly what apply_paint_annotations
    ingest() reads."""
    data = request.get_json()
    header, b64 = data["dataURL"].split(",", 1)
    layer = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")

    template_path = os.path.join(PAINT_DIR, f"{image_name}_paint_template.png")
    if not os.path.exists(template_path):
        _make_template(image_name)
    template = Image.open(template_path).convert("RGB")

    if layer.size != template.size:
        return jsonify({"ok": False, "error": f"layer size {layer.size} != template size {template.size}"}), 400

    composited = template.copy()
    composited.paste(layer, (0, 0), mask=layer)  # only paints where layer alpha > 0
    painted_path = os.path.join(PAINT_DIR, f"{image_name}_painted.png")
    composited.save(painted_path)
    return jsonify({"ok": True})


@app.route("/api/ingest/<image_name>", methods=["POST"])
def api_ingest(image_name):
    try:
        # Pass the cached stage so ingest() doesn't re-run the ENTIRE
        # production pipeline (background flattening, vesselness, ML
        # classification, merge_large_cracks) from scratch -- that full
        # re-run is what made every single Save & Ingest take a minute-plus
        # even for a small paint/erase edit. ingest() also updates this same
        # stage dict's labeled/df in place and rewrites the on-disk template
        # to reflect what it just committed, so the cache stays valid for
        # the next click-to-flip or template reload -- nothing further to
        # do here.
        # Record the pre-ingest correction state so a committed mark can still
        # be taken back. Autosave made this necessary: it commits ~1s after
        # drawing stops and the reload clears the browser's stroke-layer undo,
        # which left no way to reverse a mistake. See app_undo.py.
        try:
            import app_undo
            app_undo.snapshot(image_name)
        except Exception as _e:
            print(f"undo snapshot skipped: {type(_e).__name__}: {_e}")
        stage = get_stage(image_name)
        result = _ingest(image_name, stage=stage)
        return jsonify({"ok": True, **(result or {})})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/stage_ready/<image_name>")
def api_stage_ready(image_name):
    """Whether a correction on this image will commit fast. The UI uses it to
    say "preparing" rather than letting the first save look like a hang."""
    return jsonify({"ok": True, "ready": stage_ready(image_name),
                    "warming": image_name in _warming})


@app.route("/api/model_status")
def api_model_status():
    model_path = os.path.join(MODELS_DIR, "interior_model.joblib")
    return jsonify({"exists": os.path.exists(model_path)})


_stage_gen = 0


def invalidate_stage(image_name=None):
    """Drop cached pipeline results so the next load recomputes.

    Needed by the app endpoints: after an upload+process or a retrain, a cached
    stage describes the OLD model's output. The template on disk would then get
    rewritten from that stale cache on the first click, which is precisely how
    regions appeared to spontaneously change colour earlier in this project.
    Passing None clears everything (after a retrain, every image is stale)."""
    # Bump the generation as well as clearing. A warm started before the
    # invalidation was still running, and assigned its (now stale) result into the
    # cache when it finished -- so a removed image's stage came back from the dead,
    # and a retrained model's first click could still render from the old one.
    global _stage_gen
    _stage_gen += 1
    if image_name is None:
        _stage_cache.clear()
    else:
        _stage_cache.pop(image_name, None)


# Upload / hybrid-detect / retrain live in app_endpoints.py so this module's
# painting and ingest paths -- which took real debugging to get right -- are not
# disturbed by app plumbing. Registered here, optional so the tool still runs if
# that module is missing.
try:
    from app_endpoints import register as _register_app_endpoints
    _register_app_endpoints(app, get_stage, invalidate_stage)
    from app_exports import register as _register_exports
    _register_exports(app, list_images)
    from app_undo import register as _register_undo
    _register_undo(app, invalidate_stage, get_stage)
    from app_extras import register as _register_extras
    _register_extras(app, list_images, invalidate_stage)
    print("app endpoints registered: /api/upload, /api/process, /api/retrain, "
          "/api/export*, /api/undo_correction, /api/thumb, /api/models")
except Exception as _e:  # pragma: no cover
    print(f"app endpoints NOT registered ({type(_e).__name__}: {_e}) -- "
          f"painting still works")


if __name__ == "__main__":
    # threaded=True so generating a slow (multi-minute, for the largest
    # images) template for one image doesn't block loading a different,
    # already-cached image in another tab/request.
    # MAR SUPERALLOY PROJECT: different port from the steel CBS project's
    # paint app (8767) and the unrelated TXM project's (8766) so all can run
    # side by side without colliding.
    # Port from the environment so a clone can run alongside anything else
    # without editing source -- the packaged app is meant to need no backend
    # tweaks. 8767 stays the default: the steel/MAR project's historical port,
    # distinct from the unrelated TXM project's 8766.
    port = int(os.environ.get("PORT", "8767"))
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
