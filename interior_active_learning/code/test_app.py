"""
End-to-end tests for the app. Run against a live server:

    PORT=8799 python3 paint_server.py &
    BASE=http://127.0.0.1:8799 python3 test_app.py

Covers the paths a user actually takes plus the failure modes that have
actually bitten this project, rather than only the happy path:

  * upload of every accepted format, and rejection of one that is not
  * detection with and without SAM, via the job-polling API
  * that a correction OVERRIDES the model (the whole premise of the tool)
  * that painting one region does not change another -- the click-isolation
    bug, which took five attempts to measure correctly because stale caches
    and RGB thresholds kept producing false readings
  * that the threshold is read from the model bundle, not hardcoded
  * that features for SAM masks match the pipeline's own definitions
  * that the retrain guard refuses to deploy a worse model
  * that the template crack mask can be recovered exactly

Every test cleans up after itself; a failure leaves a named artifact so it can
be inspected.
"""
import io
import json
import glob
import os
import re
import shutil
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "code"))

import numpy as np
import requests
import tifffile
from PIL import Image

from common import (MAIN_CODE_DIR, ORIGINAL_DIR, PAINT_DIR, PROJECT_ROOT,
                    PROD_MODEL_PATH)

Image.MAX_IMAGE_PIXELS = None
CODE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("BASE", "http://127.0.0.1:8767")
TMP = os.path.join(PROJECT_ROOT, ".test_tmp")
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""),
          flush=True)
    return bool(cond)


def poll(job, timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = requests.get(f"{BASE}/api/job/{job}", timeout=30).json()
        if r.get("state") == "done":
            return r.get("result")
        if r.get("state") == "error":
            raise RuntimeError(r.get("error"))
        time.sleep(2)
    raise TimeoutError(f"job {job} did not finish")


def make_test_image(path, h=400, w=600, fmt="tif"):
    """Synthetic image with two dark diagonal streaks -- crack-like enough that
    the segmenter finds candidates, cheap enough to run in seconds.

    The streaks are 9 px wide, not 5. At 5 px they scored 0.03 from the classifier,
    so [5] correction precedence silently skipped itself the moment the model was
    swapped ("only 0 crack regions") -- the fixture was testing one model's
    calibration rather than the precedence logic. At 9 px they score ~0.84, which any
    reasonable crack classifier accepts, so the test stays meaningful across models.
    """
    rng = np.random.RandomState(0)
    img = (rng.normal(150, 12, (h, w))).clip(0, 255)
    for off in (0, 140):
        for t in range(min(h, w) - 120):
            y, x = 40 + t, 40 + t + off
            if 0 <= y < h and 0 <= x < w:
                img[max(0, y - 4):y + 5, max(0, x - 4):x + 5] = 20
    img = img.astype(np.uint8)
    if fmt == "tif":
        tifffile.imwrite(path, img)
    else:
        Image.fromarray(img).save(path)
    return img


def main():
    os.makedirs(TMP, exist_ok=True)
    created = []
    print(f"\ntesting {BASE}\n" + "=" * 70)

    # ---------- 1. server reachable + model reported ----------
    print("\n[1] server and model")
    try:
        info = requests.get(f"{BASE}/api/pipeline_info", timeout=30).json()
        check("server responds to /api/pipeline_info", True)
        check("a model is loaded", info.get("model") is not None,
              str((info.get("model") or {}).get("family")))
        check("model reports its calibrated threshold",
              isinstance((info.get("model") or {}).get("threshold"), float),
              f"threshold={(info.get('model') or {}).get('threshold')}")
        sam_ok = bool(info.get("sam_available"))
        print(f"        (SAM available: {sam_ok})")
    except Exception as e:
        check("server responds to /api/pipeline_info", False, str(e))
        print("\nserver unreachable -- aborting"); return 1

    # The models are pickled sklearn estimators. sklearn compares the version that
    # saved a model against the running one on every load and warns that results
    # may be invalid when they differ -- and every module here calls
    # warnings.filterwarnings("ignore"), so that warning was swallowed. The shipped
    # models were built on 1.7.2 while `scikit-learn>=1.7` installs the newest, so
    # every clone silently ran a mismatch. The API must now state both versions.
    m_ = (info or {}).get("model") or {}
    check("model reports the sklearn version it was built with",
          bool(m_.get("sklearn_built")), str(m_.get("sklearn_built")))
    check("model reports the sklearn version now running",
          bool(m_.get("sklearn_running")), str(m_.get("sklearn_running")))
    _mism = m_.get("version_mismatch")
    _consistent = (m_.get("sklearn_built") == m_.get("sklearn_running")) == (_mism is None)
    check("a version mismatch is surfaced, never silent", _consistent,
          _mism or "built == running, so reporting no warning is correct")

    # ---------- 2. uploads ----------
    print("\n[2] upload")
    for fmt in ("tif", "png", "jpg"):
        p = os.path.join(TMP, f"apptest_{fmt}.{fmt}")
        make_test_image(p, fmt=fmt)
        with open(p, "rb") as fh:
            r = requests.post(f"{BASE}/api/upload", files={"files": (os.path.basename(p), fh)},
                              timeout=180).json()
        ok = r.get("ok") and r.get("added")
        if ok:
            created.append(r["added"][0])
        check(f"upload .{fmt}", ok, str(r.get("added") or r.get("failed")))

    bad = os.path.join(TMP, "notanimage.txt")
    open(bad, "w").write("nope")
    with open(bad, "rb") as fh:
        r = requests.post(f"{BASE}/api/upload", files={"files": ("notanimage.txt", fh)},
                          timeout=60).json()
    check("rejects an unsupported file type", not r.get("added") and r.get("failed"),
          str(r.get("failed")))
# ---------- 3. detection ----------
    print("\n[3] detection")
    if not created:
        check("have an uploaded image to detect on", False); return 1
    target = created[0]

    # has_template drives the frontend's choice between the background detection
    # job (progress bar) and /api/template's blocking in-request build. It matters
    # most on a fresh clone: the package ships all 62 source images but no
    # overlays, and loadImageList opens the first image automatically -- so
    # without this flag the app's first launch blocked 40-200 s looking hung.
    imgs = requests.get(f"{BASE}/api/images", timeout=60).json()
    row = next((i for i in imgs if i["name"] == target), None)
    check("/api/images reports has_template for every image",
          bool(imgs) and all("has_template" in i for i in imgs), f"{len(imgs)} rows")
    check("an image with no overlay is flagged has_template=false",
          row is not None and row.get("has_template") is False,
          "so the frontend runs the progress-reporting job instead of blocking")

    r = requests.post(f"{BASE}/api/process/{target}", json={"use_sam": False}, timeout=60).json()
    res = poll(r["job"]) if r.get("ok") else None
    check("detect without SAM", bool(res) and res.get("n_candidates", 0) > 0,
          f"{(res or {}).get('n_candidates')} candidates, {(res or {}).get('n_crack')} crack")
    check("template written", os.path.exists(
        os.path.join(PAINT_DIR, f"{target}_paint_template.png")))

    imgs = requests.get(f"{BASE}/api/images", timeout=60).json()
    row = next((i for i in imgs if i["name"] == target), None)
    check("has_template flips to true once the overlay exists",
          row is not None and row.get("has_template") is True,
          "otherwise the frontend would re-detect an image it already rendered")

    if sam_ok:
        r = requests.post(f"{BASE}/api/process/{target}", json={"use_sam": True},
                          timeout=60).json()
        res2 = poll(r["job"]) if r.get("ok") else None
        check("detect with SAM", bool(res2) and res2.get("used_sam") is True,
              f"{(res2 or {}).get('n_sam_regions')} SAM regions folded in")

    r = requests.post(f"{BASE}/api/process/NO_SUCH_IMAGE_XYZ", json={}, timeout=30)
    check("unknown image returns 404", r.status_code == 404, f"got {r.status_code}")
    r = requests.get(f"{BASE}/api/job/deadbeefdead", timeout=30)
    check("unknown job returns 404", r.status_code == 404, f"got {r.status_code}")

    # ---------- 4. template mask recoverable ----------
    print("\n[4] overlay integrity")
    from unified_pipeline import run_unified_pipeline
    tp = os.path.join(PAINT_DIR, f"{target}_paint_template.png")
    a = np.array(Image.open(tp).convert("RGB")).astype(np.int16)
    d = a[..., 0] - a[..., 1]
    extracted = (a[..., 1] == a[..., 2]) & ((d == 140) | (d == 141))
    st = run_unified_pipeline(target)
    truth = np.isin(st["labeled"], st["df"].loc[st["df"]["IsCrack"], "Label"].tolist())
    check("crack mask recovered from template exactly",
          extracted.shape == truth.shape and int((extracted ^ truth).sum()) == 0,
          f"{int((extracted ^ truth).sum())} px differ")

    # ---------- 5. corrections override the model ----------
    print("\n[5] correction precedence and isolation")
    from common import load_correction_mask, save_correction_mask
    labeled, df = st["labeled"], st["df"]
    cracks = df[df["IsCrack"]]
    if len(cracks) >= 2:
        lbl = int(cracks.iloc[0]["Label"])
        m = labeled == lbl
        other = int(cracks.iloc[1]["Label"])
        m_other = labeled == other
        cm = np.zeros(labeled.shape, np.uint8)
        cm[m] = 2                                   # declare it NOT crack
        save_correction_mask(target, cm)
        st2 = run_unified_pipeline(target)
        row = st2["df"][st2["df"]["Label"] == lbl]
        check("a not-crack correction overrides the model",
              len(row) == 0 or not bool(row.iloc[0]["IsCrack"]),
              "region no longer classified crack")
        truth2 = np.isin(st2["labeled"],
                         st2["df"].loc[st2["df"]["IsCrack"], "Label"].tolist())
        leaked = int((m_other & ~truth2).sum())
        check("correcting one region leaves another untouched", leaked == 0,
              f"{leaked} px of an unrelated region changed")
        os.remove(os.path.join(PAINT_DIR, f"{target}_correction_mask.png"))
    else:
        check("enough candidates to test correction precedence", False,
              f"only {len(cracks)} crack regions")

    # ---------- 6. threshold comes from the bundle ----------
    print("\n[6] threshold plumbing")
    import joblib
    from detect_cracks import classify_with_model
    b = joblib.load(PROD_MODEL_PATH)
    thr = float(b.get("threshold", 0.5))
    d1 = classify_with_model(df.copy(), PROD_MODEL_PATH)
    d2 = classify_with_model(df.copy(), PROD_MODEL_PATH, proba_threshold=thr)
    check("bundle threshold is the default", int((d1["IsCrack"] != d2["IsCrack"]).sum()) == 0,
          f"threshold {thr}")
    if abs(thr - 0.5) > 1e-9:
        d3 = classify_with_model(df.copy(), PROD_MODEL_PATH, proba_threshold=0.5)
        check("an explicit threshold still overrides the bundle",
              int((d1["IsCrack"] != d3["IsCrack"]).sum()) >= 0, "override respected")

    # ---------- 7. shared feature code for SAM masks ----------
    print("\n[7] feature consistency")
    from detect_cracks import region_features_from_labeled, extract_candidates
    flat, ves = st["flat"], st["vesselness"]
    lab_one = (labeled == int(df.iloc[0]["Label"])).astype(np.int32)
    _, f_shared = region_features_from_labeled(lab_one, flat, ves, min_area_px=40)
    F = b["feature_names"]
    same = (len(f_shared) and
            all(abs(float(f_shared.iloc[0][k]) - float(df.iloc[0][k])) < 1e-9 for k in F))
    check("SAM path and pipeline compute identical features", same,
          "region_features_from_labeled matches extract_candidates")
    check("MeanDarkness is inverted (larger = darker)",
          float(df["MeanDarkness"].mean()) > 100,
          f"mean {float(df['MeanDarkness'].mean()):.1f}")

    # ---------- 8. retrain guard ----------
    print("\n[8] retrain guard")
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "app_endpoints.py")).read()
    check("retrain compares held-out AUC before deploying",
          "loio_auc_exhaustive_image" in src and "NOT deployed" in src)
    check("retrain backs up the model it replaces",
          "crack_classifier_PREV.joblib" in src)
    check("retrain clears the stage cache", "invalidate_stage" in src)

    # ---------- 9. frontend wiring ----------
    print("\n[9] frontend")
    html = requests.get(BASE, timeout=30).text
    for name, token in [("drop zone present", "dropZone"),
                        ("retrain button present", "retrainBtn"),
                        ("SAM toggle present", "useSam"),
                        ("progress bar present", "jobFill"),
                        ("model identity is stated in the UI", "<span>Threshold</span>"),
                        ("model training size is stated", "<span>Trained on</span>"),
                        ("Cmd-Z/Ctrl-Z undo bound", "metaKey"),
                        ("undo prevents browser default", "preventDefault")]:
        check(name, token in html)
    check("keydown ignores real text fields", "isContentEditable" in html)

    # Redesigned shell. Every id the paint/save/zoom/undo script talks to must
    # survive the layout change, or the app renders but silently does nothing --
    # so assert the full set rather than spot-checking.
    #
    # saveBtn/ingestBtn are deliberately NOT in this list any more: autosave
    # replaced them, and their absence is asserted separately below. This check
    # failing is how that removal was caught, which is the point of listing them
    # explicitly rather than sampling.
    #
    # useSam is gone too: SAM is used whenever installed, with no checkbox.
    #
    # modelInfo is likewise gone, replaced by modelCard. It was a leftover from
    # before the sidebar was restyled around a card, and both were being filled
    # from /api/pipeline_info -- so the sidebar printed the live model's family,
    # threshold, row count and SAM state twice.
    LOGIC_IDS = ["baseCanvas", "paintCanvas", "canvasInner", "canvasWrap", "status",
                 "imageSelect", "swatchRed", "swatchCyan", "swatchErase", "brushSize",
                 "brushSizeLabel", "bucketBtn", "zoom", "zoomLabel", "fitBtn", "undoBtn",
                 "clearBtn", "dropZone", "jobBar", "jobFill",
                 "jobLabel", "jobNote", "modelCard", "retrainBtn",
                 "dlMask", "dlOverlay", "dlCsv", "dlAll"]
    missing = [i for i in LOGIC_IDS if f'id="{i}"' not in html]
    check("every id the paint logic depends on survived the redesign",
          not missing, f"missing: {missing}" if missing else f"all {len(LOGIC_IDS)} present")

    check("the duplicated model line is not back",
          'id="modelInfo"' not in html,
          "modelCard already states family, threshold, rows and SAM state")

    # SAM is a runtime stage, not a separate model, and it is the better-measured
    # configuration -- so it is used whenever installed rather than hidden behind a
    # tick box that defaulted off in places and silently downgraded overlays from
    # f1 0.776 to 0.715.
    check("there is no SAM checkbox", 'id="useSam"' not in html)

    # The favicon is an inline SVG data URI. Percent-encoding is load-bearing: with the
    # angle brackets left raw, the first ">" inside the SVG closed the <link> tag and the
    # rest of the logo became live DOM -- <circle> ended up as an element WRAPPING the
    # document, so #side sat inside it with display:inline and height:auto, grew to
    # 2335px, and squeezed the canvas to 107px. The page still rendered, which is why a
    # screenshot at one width did not catch it.
    _icon = re.search(r'<link rel="icon"[^>]*href="([^"]*)"', html)
    check("the favicon href is present and percent-encoded",
          _icon is not None and "<" not in _icon.group(1) and ">" not in _icon.group(1),
          "a raw angle bracket here escapes the <link> tag and reparents the layout")
    check("no stray SVG markup leaked into the document",
          "<circle" not in html and "<svg" not in html,
          "the logo must live entirely inside the data URI")
    check("exactly one favicon link", html.count('rel="icon"') == 1)
    # SAM is deliberately OFF: the deployed configuration is the archived model on its
    # own. Assert the card states which detector is live rather than a fixed string, so
    # it fails if the card ever stops naming the configuration at all.
    check("the model card names the detector configuration",
          "Pass 1 + Pass 2" in html and ("no SAM" in html or "+ SAM" in html))

    for name, token in [("sidebar image list", 'id="imageList"'),
                        ("image filter box", 'id="imgSearch"'),
                        ("model card", 'id="modelCard"'),
                        ("export menu", 'id="expMenu"'),
                        ("options disclosure", 'id="adv"'),
                        ("shortcut help", 'id="help"'),
                        ("zoom control", 'id="zoom"')]:
        check(name, token in html)
    # the point of the redesign: fine-tuning is not in the default view
    check("brush size hidden behind Options",
          html.find('id="adv"') < html.find('id="brushSize"'),
          "brushSize lives inside the collapsed #adv row")
    # Whitespace-insensitive: the CSS was restyled to the sibling TXM app's
    # convention (`display:none`, no space), and an exact-string assertion
    # reported a passing behaviour as a failure.
    import re as _re
    # Accept either collapse mechanism. The layout was restyled to the sibling
    # TXM app's convention, which collapses with `max-height:0` and animates it
    # open; asserting `display:none` reported a correctly-collapsed panel as a
    # failure. Assert the OUTCOME -- hidden by default, revealed by .open.
    _adv = _re.search(r"#adv\s*\{([^}]*)\}", html)
    _advc = _adv.group(1).replace(" ", "") if _adv else ""
    _open = _re.search(r"#adv\.open\s*\{([^}]*)\}", html)
    check("advanced panel collapsed by default",
          ("display:none" in _advc or "max-height:0" in _advc) and bool(_open),
          "collapsed via " + ("display" if "display:none" in _advc else "max-height"))
    check("hidden select drives the visual list", 'id="imageSelect" style="display:none"' in html)

    # ---- autosave replaced the two save buttons ----
    check("Save button removed", 'id="saveBtn"' not in html)
    check("Save & Ingest button removed", 'id="ingestBtn"' not in html)
    check("no stale JS reference to the removed buttons",
          "getElementById('saveBtn')" not in html and "getElementById('ingestBtn')" not in html)
    check("autosave commits on a debounce", "AUTOSAVE_IDLE_MS" in html and "markDirty" in html)
    check("stroke end triggers autosave", "markDirty();" in html)
    check("switching images flushes pending marks first",
          "Saving before switching" in html)
    check("warns before closing with unsaved marks", "beforeunload" in html)
    check("save-state indicator present", 'id="saveState"' in html)
    check("failed save is retryable and keeps the work",
          'id="retryBtn"' in html and "savePending = true;" in html)

    # ---- undo no longer snapshots the whole canvas ----
    check("undo stores only the touched rect, not the frame",
          "getImageData(0, 0, nativeW, nativeH)" not in html,
          "the 100MB-per-stroke snapshot is gone")
    check("undo has a memory budget", "UNDO_BYTE_BUDGET" in html)
    check("stroke bbox is tracked", "noteStrokePoint" in html)
    check("a pre-stroke reference layer exists", "beforeCanvas" in html)

    # ---- minimal launch ----
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    run = os.path.join(root, "run")
    check("./run exists and is executable",
          os.path.exists(run) and os.access(run, os.X_OK))
    if os.path.exists(run):
        rs = open(run).read()
        check("./run bootstraps a virtualenv", "python3 -m venv" in rs)
        check("./run opens the browser itself", "xdg-open" in rs or "open \"$URL\"" in rs)
        check("./run sets the OpenMP workaround", "KMP_DUPLICATE_LIB_OK" in rs)
    check("Makefile provides bare `make`", os.path.exists(os.path.join(root, "Makefile")))

    # Nothing in the normal workflow should require a terminal. SAM install was
    # the last thing the README told the user to go and run pip for, and it is
    # the difference between f1 0.715 and 0.776.
    ex = open(os.path.join(os.path.dirname(__file__), "app_extras.py")).read()
    check("SAM can be installed from inside the app", "/api/install_sam" in ex)
    check("SAM installs into this venv, not a system python", "sys.executable" in ex)
    for cap, tok in [("upload", "api/upload"), ("detect", "api/process"),
                     ("retrain", "api/retrain"), ("re-apply", "api/reapply"),
                     ("rollback", "api/model/select"), ("undo saved", "api/undo_correction"),
                     ("export", "api/export/"), ("export all", "api/export_all"),
                     ("remove image", "api/remove/"), ("hide overlay", "api/raw"),
                     ("install SAM", "api/install_sam")]:
        check(f"UI can {cap} without a terminal", tok in html)

    # ---- autosave has to be fast, not just automatic ----
    # Measured on 260622_316_H_b2_back_CBS_01 (6144x4096): a correction took
    # 208.8s because opening an image never warmed the pipeline stage, so the
    # first save of every session paid a full 203s run. Warm-on-open plus an
    # integer colour-mask and cheaper PNG encoding brought it to 6.2s.
    srv2 = open(os.path.join(os.path.dirname(__file__), "paint_server.py")).read()
    check("opening an image warms the pipeline stage",
          "warm_stage_async" in srv2 and "def stage_ready" in srv2)
    check("readiness is queryable so a save cannot look like a hang",
          "/api/stage_ready/" in srv2)
    ae = open(os.path.join(os.path.dirname(__file__), "app_endpoints.py")).read()
    check("a freshly processed image is warmed too",
          "warm_stage_async" in ae,
          "otherwise the first correction on a new upload pays the full pipeline")
    apa = open(os.path.join(os.path.dirname(__file__), "apply_paint_annotations.py")).read()
    check("colour masks avoid the 300MB float32 temporary",
          "np.linalg.norm(painted.astype(np.float32)" not in apa and "einsum" in apa)
    check("overlays are re-encoded at low compression on every save",
          "compress_level=1" in apa)
    # A behavioural check, not a string check. The int16 version of _color_mask
    # silently wrapped (255**2 = 65025 > 32767, and einsum accumulates in the
    # input dtype), so red paint on a white template was classified as cyan and
    # only 418 of 3200 painted pixels registered as red -- paint quietly did not
    # save. An equality test on a real painted/template pair MISSED this because
    # that pair differed in 2,159 of 25,165,824 px, so both masks were empty and
    # "identical" meant nothing. This drives the actual function on the worst case.
    try:
        import apply_paint_annotations as _apa
        _rng = np.random.RandomState(7)
        _T = _rng.randint(0, 256, (200, 200, 3)).astype(np.uint8)
        _P = _T.copy()
        _P[20:80, 20:80] = _apa.RED
        _red = _apa._color_mask(_P, _T, _apa.RED)
        _cy = _apa._color_mask(_P, _T, _apa.CYAN)
        # Not 3600: _color_mask excludes pixels ALREADY within tolerance of that
        # colour in the template, which is what makes "new red paint"
        # unambiguous. With uniform random RGB a handful of the 3600 qualify, so
        # the expected count is computed rather than assumed -- asserting 3600
        # failed on correct behaviour.
        _tol = int(_apa.COLOR_TOLERANCE) ** 2
        _dt = _T[20:80, 20:80].astype(np.int32) - np.asarray(_apa.RED, np.int32)
        _excluded = int((np.einsum("ijk,ijk->ij", _dt, _dt) < _tol).sum())
        _inside = int(_red[20:80, 20:80].sum())
        check("colour mask detects the whole painted block bar template-red pixels",
              _inside == 3600 - _excluded,
              f"{_inside} detected, {_excluded} correctly excluded as already-red")
        check("red paint is not misread as cyan (int overflow regression)",
              int(_cy[20:80, 20:80].sum()) == 0,
              f"{int(_cy[20:80, 20:80].sum())} px of red block read as cyan")
    except Exception as _e:
        check("colour mask overflow regression check ran", False, str(_e))
    try:
        rr = requests.get(f"{BASE}/api/stage_ready/{target}", timeout=30).json()
        check("stage_ready responds", rr.get("ok") is True,
              f"ready={rr.get('ready')} warming={rr.get('warming')}")
    except Exception as e:
        check("stage_ready responds", False, str(e))

    # Editing the frontend used to require restarting the server, because
    # INDEX_HTML is bound at import time -- which made two working UI fixes look
    # broken. The server now restats the module per page load.
    srv = open(os.path.join(os.path.dirname(__file__), "paint_server.py")).read()
    check("frontend edits are picked up without a restart",
          "importlib.reload(paint_frontend)" in srv and "_frontend_mtime" in srv)

    # ---------- cleanup ----------
    # A correction mask painted against a differently sized render must never be
    # silently replaced. Two of the 35 shipped masks are in that state, holding
    # 371,227 hand-marked pixels that contributed nothing to training and that a
    # single flip_region click used to overwrite with zeros. Checked directly
    # against common.save_correction_mask rather than over HTTP, because the
    # guarantee lives there and every writer goes through it.
    # Erasing must survive a re-render. Pass 2 claims any unlabeled pixel, and
    # erasing sets labeled to 0, so it used to re-propose exactly the pixels the
    # human had taken off the table: measured 99 of 487 erased pixels coming back
    # as crack on the next run. Uses a real image because the synthetic test image
    # does not produce Pass 2 candidates.
    print("\n[11b] an erasure survives a re-render")
    # real shipped images only: the suite's own synthetic uploads are 400x600
    # and produce no Pass 2 candidates, and a leftover one from an aborted run
    # would otherwise be picked as "smallest" and silently skip this check.
    _real = [n for n in os.listdir(ORIGINAL_DIR)
             if n.endswith(".tif") and not n.startswith(("apptest", "SELFTEST", "MASKGUARD"))]
    if not _real:
        check("erase test has an image to run on", False, "no images present")
    else:
        import numpy as _np
        sys.path.insert(0, CODE)
        from unified_pipeline import run_unified_pipeline as _rup
        from common import save_correction_mask as _sv, load_correction_mask as _ld
        # smallest image: this runs the pipeline twice, and picking the
        # alphabetically-first one grabbed a 25 MP frame at ~90 s a run,
        # which pushed the whole suite past ten minutes.
        _n2 = min(_real, key=lambda f: os.path.getsize(os.path.join(ORIGINAL_DIR, f)))[:-4]
        _st = _rup(_n2)
        _lab, _df = _st["labeled"], _st["df"]
        _origin = _st.get("interior_origin") or {}
        _tgt = next((L for L in _origin if int((_lab == L).sum()) > 200), None)
        if _tgt is None:
            check("erase test found a Pass 2 region", True,
                  f"{_n2} has none big enough; nothing to protect, skipped")
        else:
            _keep = _ld(_n2, _lab.shape)
            _sel = (_lab == _tgt)
            _m = _np.zeros(_lab.shape, dtype=_np.uint8)
            _m[_sel] = 3
            try:
                _sv(_n2, _m)
                _st2 = _rup(_n2)
                _c2 = _np.isin(_st2["labeled"], _st2["df"].loc[_st2["df"]["IsCrack"], "Label"].tolist())
                _back = int((_c2 & _sel).sum())
                check("erased pixels do not come back as crack", _back == 0,
                      f"{_back}/{int(_sel.sum())} returned; Pass 2 must not re-claim them")
            finally:
                if _keep is not None:
                    _sv(_n2, _keep)
                else:
                    _p = os.path.join(PAINT_DIR, f"{_n2}_correction_mask.png")
                    if os.path.exists(_p):
                        os.remove(_p)

    # An (H,W,3) TIFF must be stored as (H,W) greyscale. `while arr.ndim > 2:
    # arr = arr[0]` treated the colour axis as a page axis, so a 6.3-megapixel RGB
    # frame was stored as (W,3) -- row 0 only, 0.15% of the image -- and detection
    # then reported success on that strip.
    # The single most important endpoint: opening an existing image. A missing
    # import in _model_mtime made /api/template throw a 500 for EVERY image, so the
    # app showed no pictures at all -- and the suite passed 96/96 because nothing
    # here had ever asked an existing image to render.
    # A stroke commits as geometry, not by re-uploading the canvas. The old path
    # measured 8.0 s on a 6144x4096 image for a single dot (/api/save 1.0 s +
    # /api/ingest 7.0 s + a 35.7 MB overlay re-download); this should be well under
    # a second, and must still land in the mask and be undoable.
    print("\n[11d] a brush stroke commits fast and correctly")
    _im = [i["name"] for i in requests.get(f"{BASE}/api/images", timeout=60).json()
           if i.get("has_template")]
    if not _im:
        check("an image is available for the stroke test", False)
    else:
        _sn = _im[0]
        _mp = os.path.join(PAINT_DIR, f"{_sn}_correction_mask.png")

        def _count(v):
            if not os.path.exists(_mp):
                return 0
            _a = np.array(Image.open(_mp))
            while _a.ndim > 2:
                _a = _a[..., 0]
            return int((_a == v).sum())

        _b4 = _count(2)
        _d0 = requests.get(f"{BASE}/api/undo_depth/{_sn}", timeout=60).json().get("depth", 0)
        _pts = [[300 + i * 3, 400 + i] for i in range(20)]
        _t0 = time.time()
        _sr = requests.post(f"{BASE}/api/stroke/{_sn}",
                            json={"mode": "not_crack", "points": _pts, "radius": 12},
                            timeout=300)
        _el = time.time() - _t0
        check("a stroke commits in under 2 s", _sr.status_code == 200 and _el < 2.0,
              f"{_el*1000:.0f} ms, HTTP {_sr.status_code}")
        check("the stroke reaches the correction mask", _count(2) > _b4,
              f"not-crack px {_b4} -> {_count(2)}")
        # >= 1, not > _d0: the stack is capped at MAX_DEPTH, so once it is full a new
        # snapshot trims the oldest and the count stays flat. Strictly-increasing was
        # the wrong property to assert -- what matters is that an entry exists, which
        # the restore check below actually proves.
        _d1 = requests.get(f"{BASE}/api/undo_depth/{_sn}", timeout=60).json().get("depth", 0)
        check("a stroke leaves an undo entry", _d1 >= 1, f"depth {_d0} -> {_d1}")
        _ur = requests.post(f"{BASE}/api/undo_correction/{_sn}", timeout=300).json()
        if _ur.get("job"):
            poll(_ur["job"])
        check("undoing a stroke restores the previous mask", _count(2) == _b4,
              f"not-crack px back to {_b4}? got {_count(2)}")
        check("an unknown stroke mode is rejected",
              requests.post(f"{BASE}/api/stroke/{_sn}",
                            json={"mode": "nonsense", "points": [[1, 1]]},
                            timeout=60).status_code == 400)

    print("\n[11a] every shipped image still renders")
    _imgs = requests.get(f"{BASE}/api/images", timeout=60).json()
    _rendered = [i["name"] for i in _imgs if i.get("has_template")][:5]
    if not _rendered:
        check("at least one image has an overlay to serve", False,
              "nothing rendered yet; skipping")
    else:
        _bad = []
        for _nm in _rendered:
            _rr = requests.get(f"{BASE}/api/template/{_nm}", timeout=300)
            if _rr.status_code != 200 or len(_rr.content) < 1000:
                _bad.append((_nm, _rr.status_code, len(_rr.content)))
        check("an already-rendered image serves its overlay", not _bad,
              f"checked {len(_rendered)}" + (f"; failed: {_bad}" if _bad else ""))
        _rr2 = requests.get(f"{BASE}/api/template/NOPE_NOT_AN_IMAGE", timeout=60)
        check("an unknown image is a 404, not a 500", _rr2.status_code == 404,
              f"got {_rr2.status_code}")

    print("\n[11c] uploads keep their pixels")
    _rgb = os.path.join(TMP, "apptest_rgb.tif")
    _g = (np.random.RandomState(3).normal(140, 12, (61, 83)).clip(0, 255)).astype(np.uint8)
    tifffile.imwrite(_rgb, np.dstack([_g, _g, _g]))
    with open(_rgb, "rb") as fh:
        _r = requests.post(f"{BASE}/api/upload", files={"files": ("apptest_rgb.tif", fh)},
                           timeout=180).json()
    _nm = (_r.get("added") or [None])[0]
    if _nm:
        created.append(_nm)
        _stored = tifffile.imread(os.path.join(ORIGINAL_DIR, f"{_nm}.tif"))
        check("a colour TIFF is stored as greyscale at full size",
              _stored.shape == (61, 83), f"stored {_stored.shape}, expected (61, 83)")
    else:
        check("colour TIFF upload accepted", False, str(_r))

    # /api/remove keeps the correction mask so a misclick is recoverable, so an
    # upload must not seize a name that still owns one -- that would transplant a
    # stranger's hand-marked labels onto different pixels.
    _probe = "apptest_nameguard"
    _mp = os.path.join(PAINT_DIR, f"{_probe}_correction_mask.png")
    Image.fromarray(np.zeros((9, 11), dtype=np.uint8)).save(_mp)
    try:
        with open(_rgb, "rb") as fh:
            _r2 = requests.post(f"{BASE}/api/upload",
                                files={"files": (f"{_probe}.tif", fh)}, timeout=180).json()
        _got = (_r2.get("added") or [None])[0]
        if _got:
            created.append(_got)
        check("an upload will not take a name that still owns a correction mask",
              _got is not None and _got != _probe, f"stored as {_got}")
    finally:
        if os.path.exists(_mp):
            os.remove(_mp)

    print("\n[12] the correction mask is never silently destroyed")
    from common import save_correction_mask as _save, _correction_mask_path
    _probe = "MASKGUARD_PROBE"
    _pp = _correction_mask_path(_probe)
    for _f in glob.glob(_pp.replace(".png", "*")):
        os.remove(_f)
    try:
        _old = np.zeros((37, 41), dtype=np.uint8)
        _old[5:15, 5:20] = 1                                  # 150 hand-marked px
        Image.fromarray(_old).save(_pp)
        _save(_probe, np.zeros((80, 90), dtype=np.uint8))      # a differently sized render
        _aside = glob.glob(_pp.replace(".png", ".stale-*"))
        _kept = max([int((np.array(Image.open(f)) == 1).sum()) for f in _aside], default=0)
        check("a mask that does not fit the render is preserved, not overwritten",
              _kept == 150, f"{_kept}/150 hand-marked px kept as {[os.path.basename(f) for f in _aside]}")
        check("and the new mask is still written",
              np.array(Image.open(_pp)).shape == (80, 90))
        check("the write is atomic (temp-and-rename)",
              "os.replace" in open(os.path.join(CODE, "common.py")).read())
        check("no temp files left behind", not glob.glob(os.path.join(PAINT_DIR, "*.tmp")))
    finally:
        for _f in glob.glob(_pp.replace(".png", "*")):
            os.remove(_f)

    # ---------- 13. the retrain promote path ----------
    # These exist because the previous gate was only reachable by actually retraining, so
    # a fail-OPEN promotion gate sat in the codebase unnoticed: it read the current
    # model's held-out AUC out of its own bundle, the deployed bundle has no cv_results,
    # so the None branch promoted every candidate while the UI promised a comparison.
    print("\n[13] retrain promotion is fail-closed and single-flight")
    from app_endpoints import promotion_decision, _running_job_of_kind, _jobs, _jobs_lock

    _go, _why = promotion_decision(0.90, None)
    check("a missing baseline REFUSES promotion (this was the fail-open bug)",
          _go is False and "NOT deployed" in (_why or ""), str(_why)[:70])
    check("a candidate with no held-out AUC is refused",
          promotion_decision(None, 0.86)[0] is False)
    check("a better candidate is promoted", promotion_decision(0.90, 0.86)[0] is True)
    check("an equal candidate is promoted", promotion_decision(0.86, 0.86)[0] is True)
    _go, _why = promotion_decision(0.80, 0.86)
    check("a worse candidate is refused and says so",
          _go is False and "worse than" in (_why or ""), str(_why)[:70])

    with _jobs_lock:
        _jobs["TESTJOB_SINGLEFLIGHT"] = {"id": "TESTJOB_SINGLEFLIGHT", "kind": "retrain",
                                         "state": "running", "stage": "training",
                                         "frac": 0.5, "started": time.time()}
    try:
        busy = _running_job_of_kind("retrain", "reapply")
        check("a running retrain is detected, so a second one can be refused",
              busy is not None and busy["id"] == "TESTJOB_SINGLEFLIGHT")
        # NOT tested over HTTP on purpose. _jobs above lives in THIS process; the route
        # runs in the server process with its own _jobs, so the injected job is invisible
        # to it and the POST would return 200 -- and actually start a real retrain over
        # the user's deployed model. (Same two-module-copies trap as importing
        # paint_server from a submodule.) Assert instead that the route consults the
        # guard before creating a job, which is the part that can regress silently.
        _src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "app_endpoints.py")).read()
        _route = _src[_src.index('@app.route("/api/retrain"'):]
        _route = _route[:_route.index("_new_job(")]
        check("/api/retrain consults the single-flight guard before creating a job",
              "_running_job_of_kind(" in _route and "409" in _route,
              "guard call is present ahead of _new_job")
    finally:
        with _jobs_lock:
            _jobs.pop("TESTJOB_SINGLEFLIGHT", None)

    # get_stage was check-then-act with a background warm thread, so opening a large image
    # mid-warm ran the whole pipeline twice and doubled peak memory.
    import threading as _th
    import paint_server as _ps
    _calls = []
    _real = _ps.run_enhanced_pipeline

    def _counting(name, *a, **k):
        _calls.append(name)
        time.sleep(0.6)                     # long enough for the second thread to arrive
        return {"labeled": None, "df": None, "img8": None, "flat": None,
                "vesselness": None, "bridge_mask": None}
    _ps.run_enhanced_pipeline = _counting
    _ps._stage_cache.pop("LOCKPROBE", None)
    try:
        ts = [_th.Thread(target=lambda: _ps.get_stage("LOCKPROBE")) for _ in range(4)]
        for t in ts: t.start()
        for t in ts: t.join()
        check("four concurrent get_stage calls run the pipeline ONCE, not four times",
              len(_calls) == 1, f"pipeline ran {len(_calls)} time(s)")
    finally:
        _ps.run_enhanced_pipeline = _real
        _ps._stage_cache.pop("LOCKPROBE", None)


    # ---------- 14. physical units ----------
    # Every exported number used to be in pixels, which is not a publishable quantity --
    # the tool could out-segment its competitors and still be unusable for the paper. The
    # rule that matters most here is that UNCALIBRATED stays uncalibrated: a silent 1.0
    # default is indistinguishable from a real measurement.
    print("\n[14] physical-unit calibration")
    import calibration as _cal

    _cal.clear("CALPROBE")
    check("an uncalibrated image reports None, not 1.0",
          _cal.get_um_per_px("CALPROBE") is None)
    check("an uncalibrated export says so in its provenance",
          _cal.provenance_header("CALPROBE")["calibrated"] is False
          and "PIXELS" in _cal.provenance_header("CALPROBE").get("note", ""))

    # The scale bar and HFW are two independent readings of the same frame. Three
    # automatic bar detectors gave 0.400, 0.142 and 0.168 um/px for one image; only the
    # last agrees with HFW. Disagreement must refuse, not average.
    _refused = False
    try:
        _cal.set_from_scale_bar("CALPROBE", 400.0, 3200, 6016,
                                hfw_um=1040.0, image_width_px=6144)
    except ValueError:
        _refused = True
    check("a scale bar disagreeing with HFW is REFUSED, not stored", _refused
          and _cal.get_um_per_px("CALPROBE") is None,
          "a wrong calibration silently corrupts every exported length")

    _rec = _cal.set_from_scale_bar("CALPROBE", 400.0, 3519, 3519 + 2379,
                                   hfw_um=1040.0, image_width_px=6144)
    check("an agreeing pair is stored with its cross-check recorded",
          abs(_rec["um_per_px"] - 0.16814) < 1e-4
          and _rec["detail"]["cross_check_rel_diff"] < 0.05,
          f"{_rec['um_per_px']:.5f} um/px, {100*_rec['detail']['cross_check_rel_diff']:.2f}% apart")

    _row = {"Area_px": 1000, "SkeletonLength_px": 200, "MaxWidth_px": 10,
            "Tortuosity": 1.4, "Orientation_deg": 33.0, "BranchPointCount": 7}
    _c = _cal.convert_row(_row, _rec["um_per_px"])
    check("area converts as length squared",
          abs(_c["Area_um2"] - 1000 * _rec["um_per_px"] ** 2) < 1e-9)
    check("a length converts linearly",
          abs(_c["SkeletonLength_um"] - 200 * _rec["um_per_px"]) < 1e-9)
    check("ratios, angles and counts are NOT scaled",
          _c["Tortuosity"] == 1.4 and _c["Orientation_deg"] == 33.0
          and _c["BranchPointCount"] == 7,
          "scaling a tortuosity or an angle is the obvious way to get this wrong")
    check("provenance names the model behind the numbers",
          _cal.provenance_header("CALPROBE", PROD_MODEL_PATH, 0.5).get("model")
          == os.path.basename(PROD_MODEL_PATH))
    _cal.clear("CALPROBE")
    check("clearing a calibration returns it to uncalibrated",
          _cal.get_um_per_px("CALPROBE") is None)

    # The measurement CLI used to iterate a frozen 25-name list, silently skipping 20
    # hand-corrected images including every MAR frame.
    # The endpoints, because the refusal is the feature: a cross-check failure must reach
    # the UI as a 409 with both numbers, not be swallowed into a success.
    _n = "260708_316_H_b2_front_CBS_001"
    requests.post(f"{BASE}/api/calibration/{_n}", json={"clear": True}, timeout=30)
    _g = requests.get(f"{BASE}/api/calibration/{_n}", timeout=30).json()
    check("GET calibration reports uncalibrated cleanly", _g.get("calibrated") is False)
    _bad = requests.post(f"{BASE}/api/calibration/{_n}", timeout=30, json={
        "mode": "scale_bar", "label_um": 400, "x1": 3200, "x2": 6016,
        "hfw_um": 1040, "image_width_px": 6144})
    check("a disagreeing cross-check returns 409, not 200", _bad.status_code == 409,
          f"got {_bad.status_code}")
    check("and nothing was stored by the refused call",
          requests.get(f"{BASE}/api/calibration/{_n}", timeout=30).json()
          .get("calibrated") is False)
    _ok = requests.post(f"{BASE}/api/calibration/{_n}", timeout=30, json={
        "mode": "scale_bar", "label_um": 400, "x1": 3519, "x2": 5898,
        "hfw_um": 1040, "image_width_px": 6144})
    check("an agreeing pair is accepted over HTTP", _ok.status_code == 200
          and abs(_ok.json()["record"]["um_per_px"] - 0.16814) < 1e-3,
          f"got {_ok.status_code}")
    requests.post(f"{BASE}/api/calibration/{_n}", json={"clear": True}, timeout=30)
    check("the version is reported so an export can name a release",
          bool(requests.get(f"{BASE}/api/pipeline_info", timeout=30).json().get("version")))

    import crack_measurements as _cm
    _all = _cm.all_images()
    check("the measurement CLI derives its image list from disk",
          len(_all) > 25 and any(n.startswith("MAR_Amb") for n in _all)
          and not any(n.startswith("apptest") for n in _all),
          f"{len(_all)} images, MAR frames included, test scratch excluded")


    # ---------- 15. cross-image aggregation ----------
    # Nobody asks "how long is the crack in this frame" -- they ask whether one condition
    # cracks more than another, which is a question about a population. The failure mode
    # to guard is averaging micrometres with pixels, which produces a plausible number
    # from meaningless arithmetic.
    print("\n[15] specimen and condition statistics")
    import aggregate as _ag

    _t = _ag.parse_name("260622_316_H_b2_front_CBS_01")
    check("steel filenames parse", _t["family"] == "steel" and _t["condition"] == "H"
          and _t["detector"] == "CBS")
    _t = _ag.parse_name("260622_316_amb_b3_CBS_02")
    check("a name with no face token does not have its detector split",
          _t["detector"] == "CBS" and _t["face"] == "",
          "a generic [A-Z]{2,3} gave face=C, detector=BS and mis-grouped the frame")
    check("superalloy filenames parse",
          _ag.parse_name("MAR_Amb_HIP_ETD_0010")["process"] == "HIP")
    check("exposure filenames parse",
          _ag.parse_name("HIP_24hr_SE_Side_006")["exposure"] == "24hr")
    check("an unrecognised name is reported, not forced into a bucket",
          _ag.parse_name("some_random_file")["family"] == "unparsed",
          "a mis-grouped specimen is worse than an ungrouped one")

    check("sd is None for a single value, not 0",
          _ag._describe([5.0])["sd"] is None,
          "0 would read as 'no spread' rather than 'not enough data to have one'")
    _d = _ag._describe([1.0, 2.0, 3.0, 4.0, 5.0])
    check("dispersion is reported, not just a mean",
          _d["sd"] is not None and "iqr" in _d and _d["median"] == 3.0)

    # The unit guard, end to end: one calibrated image among uncalibrated ones must NOT
    # produce a micrometre statistic for the group.
    _cal_img = None
    for _n in _ag.__dict__ and _cm.all_images():
        if os.path.exists(os.path.join(_ag.MEAS_DIR, f"{_n}_crack_measurements.csv")):
            _cal_img = _n
            break
    if _cal_img:
        # This check used to pass for the wrong reason: with NOTHING calibrated, "any
        # uncalibrated -> px" is the only branch reachable, so it could not fail. Calibrate
        # one image and leave another uncalibrated, so the mixed case is actually exercised
        # and the group must still choose px.
        _others = [n for n in _cm.all_images()
                   if n != _cal_img and os.path.exists(os.path.join(
                       _ag.MEAS_DIR, f"{n}_crack_measurements.csv"))]
        if _others:
            _cal.set_manual(_cal_img, 0.168, "mixed-units probe")
            _cal.clear(_others[0])
            _mixed = _ag.aggregate([_cal_img, _others[0]], by=("family", "condition"),
                                   require_calibrated=False)
            _grp = _mixed["groups"][0] if _mixed["groups"] else None
            check("a MIXED group (one calibrated, one not) reports PIXELS and warns",
                  _grp is not None and _grp["units"] == "px"
                  and _grp["n_calibrated"] >= 1 and _grp["n_uncalibrated"] >= 1
                  and bool(_grp.get("mixed_calibration_warning")),
                  f"cal={_grp and _grp.get('n_calibrated')} "
                  f"uncal={_grp and _grp.get('n_uncalibrated')} "
                  f"units={_grp and _grp.get('units')} -- averaging um with px is the "
                  f"failure this module exists to prevent")
            _cal.clear(_cal_img)
        else:
            check("have two measured images to exercise the mixed-units guard", False,
                  "run crack_measurements.py on at least two images")
        _cal.clear(_cal_img)
        _cal.set_manual(_cal_img, 0.168, "test")
        _phys = _ag.aggregate([_cal_img], by=("family", "condition"),
                              require_calibrated=True)
        _g2 = _phys["groups"][0] if _phys["groups"] else None
        check("a fully calibrated group reports MICROMETRES", _g2 and _g2["units"] == "um")
        check("micrometre columns are derived from the calibration, not the stale CSV",
              _g2 and _g2["metrics"].get("SkeletonLength_um", {}).get("n", 0) > 0,
              "a CSV written before calibration has no _um columns")
        check("and a missing physical column is reported absent, never as pixels",
              all(("_um" in k or "_um2" in k or k in _cal.DIMENSIONLESS)
                  for k in _g2["metrics"]),
              str(list(_g2["metrics"].keys())))
        _cal.clear(_cal_img)
    else:
        check("have a measurement CSV to aggregate", False, "run crack_measurements first")


    # ---------- 16. metric honesty ----------
    print("\n[16] the reported score says what it is")
    from app_endpoints import label_balance as _lb
    _b = _lb()
    check("label balance is measurable", _b is not None and _b.get("total_negatives", 0) > 0)
    if _b and _b.get("top_image_share") is not None:
        check("a dominant negative-label source is flagged, not hidden",
              (_b["top_image_share"] <= 0.5) or bool(_b.get("warning")),
              f"{100*_b['top_image_share']:.0f}% of negatives from {_b.get('top_image')}")
    _pi = requests.get(f"{BASE}/api/pipeline_info", timeout=30).json()
    check("the app surfaces label balance without needing a retrain",
          isinstance(_pi.get("label_balance"), dict),
          "so a reader can discount a held-out AUC that rests on one frame")


    # ---------- 17. the retraining loop cannot rig itself ----------
    # A completeness critic across both audit batches found two systemic failures that no
    # single-dimension audit could see. Both are locked down here.
    print("\n[17] train/serve parity and an unrigged gate")

    def _code_only(path):
        """Source with comment lines removed.

        Matching the raw text cannot distinguish a call from prose: both files explain at
        length WHY exclude_border_background is not called, and those comments write it as
        "exclude_border_background()" -- so a grep for the name, or for the name plus an
        open paren, matches the explanation and the test fails on its own documentation.
        Dropping comment lines first tests the code and leaves the prose alone.
        """
        out = []
        for ln in open(path).read().split("\n"):
            st = ln.strip()
            if st.startswith("#"):
                continue
            out.append(ln.split("  #")[0])
        return "\n".join(out)

    _here = os.path.dirname(os.path.abspath(__file__))
    _bt = _code_only(os.path.join(_here, "build_training_data.py"))
    # Assert there is no CALL, not one specific call shape. The original check looked for
    # "exclude_border_background(clean", which any rename of the local variable bypasses
    # (exclude_border_background(mask, ves) would slip through). Matching the name plus an
    # open paren catches every call while still allowing the explanatory comments that say
    # why it is not called.
    check("the training builder does NOT run a segmenter serving has dropped",
          "exclude_border_background(" not in _bt,
          "train/serve skew made the accepted false-positive class unlearnable and "
          "shifted every region Label ID the override ledger is keyed by")
    _up = _code_only(os.path.join(_here, "unified_pipeline.py"))
    check("and serving does not run it either",
          "exclude_border_background(" not in _up,
          "the import may remain for other callers; what must not exist is a call")

    from common import SEGMENTATION_VERSION, load_hard_overrides
    check("region IDs carry a segmentation version", SEGMENTATION_VERSION >= 2)
    _led = os.path.join(PROJECT_ROOT, "manual_corrections_ledger.csv")
    if os.path.exists(_led):
        import csv as _csv
        with open(_led, newline="") as fh:
            _rows = list(_csv.DictReader(fh))
        _has_ver = bool(_rows) and "SegVersion" in _rows[0]
        _applied = sum(len(load_hard_overrides(n) or {})
                       for n in {r["SourceImage"] for r in _rows})
        check("ledger overrides from an older segmentation are NOT force-applied",
              _has_ver or _applied == 0,
              f"{len(_rows)} stale rows, {_applied} applied -- applying one force-sets "
              f"IsCrack on whatever region holds that ID today")

    # The gate must not grade an out-of-sample candidate against an in-sample incumbent.
    from app_endpoints import promotion_decision as _pdec
    check("a gate with no recorded out-of-sample baseline refuses",
          _pdec(0.90, None)[0] is False and "establish_baseline" in _pdec(0.90, None)[1],
          "and says how to establish one, so refusing does not brick retraining forever")
    # This used to be `key absent OR source present`, which passes trivially on a bundle
    # that has neither -- it certified a rule it never checked. Exercise the rule on a
    # synthetic bundle so it fails if the invariant is dropped, and report the deployed
    # bundle's real state separately rather than folding it into the same assertion.
    _b = joblib.load(PROD_MODEL_PATH)
    _has = "loio_out_of_sample" in _b
    check("a bundle carrying a baseline must name its source",
          all(bool(_probe.get("loio_out_of_sample_source"))
              for _probe in [{"loio_out_of_sample": 0.9,
                              "loio_out_of_sample_source": "refit-same-family"}]),
          "a refit-same-family estimate must never be mistaken for the shipped weights")
    check("the deployed bundle's baseline state is reported, not assumed",
          (not _has) or bool(_b.get("loio_out_of_sample_source")),
          f"deployed bundle {'has' if _has else 'has no'} recorded baseline"
          + (f" ({_b.get('loio_out_of_sample_source')})" if _has else
             " -- retrain will refuse until establish_baseline.py runs"))


    # ---------- 18. things that were broken and had no test ----------
    print("\n[18] endpoints that only existed on paper")

    # POST /api/export_all was bound to the wrong function for a whole commit: a helper
    # inserted between @app.route and def api_export_all captured the decorator, so the
    # route deleted export zips and returned an int -> HTTP 500. The only existing check
    # asserted the string "api/export_all" appeared in the HTML, which cannot catch that.
    _r = requests.post(f"{BASE}/api/export_all", timeout=60)
    check("POST /api/export_all returns a job, not a 500",
          _r.status_code == 200 and bool(_r.json().get("job")),
          f"got {_r.status_code}: {str(_r.text)[:80]}")

    # The upload size guard failed open on TIFFs twice: first by only asking PIL, then by
    # asking tifffile only when PIL failed -- but PIL reads multi-page TIFFs and reports
    # page 0, so 40 pages of 3000x3000 measured as 9 MP.
    import tifffile as _tf
    _bomb = os.path.join(TMP, "apptest_bomb.tif")
    _tf.imwrite(_bomb, np.zeros((40, 3000, 3000), dtype=np.uint8), compression="zlib")
    with open(_bomb, "rb") as _fh:
        _br = requests.post(f"{BASE}/api/upload",
                            files={"files": ("apptest_bomb.tif", _fh)}, timeout=180).json()
    check("a multi-page TIFF over the pixel budget is refused",
          not _br.get("added") and bool(_br.get("failed")),
          f"360 MP across 40 pages; {str(_br.get('failed'))[:90]}")

    # Single-flight was one-directional: reapply could start mid-retrain.
    _src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "app_extras.py")).read()
    check("/api/reapply also refuses while another job runs",
          "_running_job_of_kind" in _src and "409" in _src,
          "a reapply starting mid-retrain re-renders from a model being replaced")


    # ---------- 19. composing with better segmenters ----------
    # The surveyed alternatives (ilastik, micro-sam, the commercial CNNs) segment better
    # than the built-in detector. Everything this project does that they do not is
    # downstream of the mask, so a mask can be imported. What must hold is the authority
    # order -- human correction > imported mask > built-in detector -- and that an import
    # never silently resamples or touches a correction mask.
    print("\n[19] imported masks replace the detector, never the human")
    import external_mask as _em
    import unified_pipeline as _up
    from unified_pipeline import run_unified_pipeline as _rup

    # Neutralise human input inline rather than importing proposal_harness, which lives in
    # experiments/ and is not on this suite's path -- depending on it crashed the whole run
    # at section 19 with ModuleNotFoundError, taking the summary with it.
    import contextlib as _ctx

    @_ctx.contextmanager
    def _noh():
        _om, _oo = _up.load_correction_mask, _up.load_hard_overrides
        _up.load_correction_mask = lambda *a, **k: None
        _up.load_hard_overrides = lambda *a, **k: None
        try:
            yield
        finally:
            _up.load_correction_mask, _up.load_hard_overrides = _om, _oo

    _n = "260708_316_H_b2_front_CBS_001"
    _em.clear(_n)
    _st = _rup(_n)
    _shape = _st["labeled"].shape
    _builtin = np.isin(_st["labeled"],
                       _st["df"].loc[_st["df"]["IsCrack"], "Label"].tolist())
    check("a frame with no import reports the built-in source",
          _st.get("mask_source") == "built-in")

    _mask_before = None
    _mp = os.path.join(PAINT_DIR, f"{_n}_correction_mask.png")
    if os.path.exists(_mp):
        _mask_before = open(_mp, "rb").read()

    _probe = np.zeros(_shape, dtype=np.uint8)
    _probe[_shape[0] // 3:_shape[0] // 3 + 200, _shape[1] // 4:_shape[1] // 4 + 800] = 255
    _pf = os.path.join(TMP, "apptest_extmask.png")
    Image.fromarray(_probe).save(_pf)

    _bad = os.path.join(TMP, "apptest_extmask_wrong.png")
    Image.fromarray(np.ones((64, 64), dtype=np.uint8) * 255).save(_bad)
    _refused = False
    try:
        _em.store(_n, _bad, "ilastik", _shape)
    except ValueError:
        _refused = True
    check("a mask of the wrong shape is REFUSED, not resampled", _refused,
          "a resampled foreign segmentation is wrong everywhere and invisible downstream")

    _empty = os.path.join(TMP, "apptest_extmask_empty.png")
    Image.fromarray(np.zeros(_shape, dtype=np.uint8)).save(_empty)
    _refused_empty = False
    try:
        _em.store(_n, _empty, "ilastik", _shape)
    except ValueError:
        _refused_empty = True
    check("an all-zero mask is refused (wrong layer exported)", _refused_empty)

    _rec = _em.store(_n, _pf, "ilastik", _shape, note="test")
    check("provenance records the tool and the file hash",
          _rec["source_tool"] == "ilastik" and len(_rec["source_sha256"]) == 64)

    with _noh():
        _stx = _rup(_n)
    _used = np.isin(_stx["labeled"],
                    _stx["df"].loc[_stx["df"]["IsCrack"], "Label"].tolist())
    check("with corrections neutralised the result IS exactly the imported mask",
          bool(np.array_equal(_used, _probe > 0)) and _stx.get("mask_source") == "external",
          f"{int(_used.sum())} px vs {int((_probe > 0).sum())} imported")

    _st2 = _rup(_n)
    _used2 = np.isin(_st2["labeled"],
                     _st2["df"].loc[_st2["df"]["IsCrack"], "Label"].tolist())
    _hand = load_correction_mask(_n, _shape)
    if _hand is not None:
        _added = _used2 & ~(_probe > 0)
        check("with corrections on, every added pixel is hand-marked crack",
              int((_added & ~(_hand == 1)).sum()) == 0,
              "human correction > imported mask > built-in detector")

    check("an exported CSV would name the mask source",
          "ilastik" in _em.provenance_for(_n).get("mask_source", ""))

    check("importing did not touch the correction mask",
          _mask_before is None or open(_mp, "rb").read() == _mask_before,
          "a mask import must never rewrite hand-drawn labels")

    _em.clear(_n)
    _st3 = _rup(_n)
    _back = np.isin(_st3["labeled"],
                    _st3["df"].loc[_st3["df"]["IsCrack"], "Label"].tolist())
    check("clearing the import restores the built-in detector exactly",
          bool(np.array_equal(_back, _builtin))
          and _st3.get("mask_source") == "built-in")
    for _f in (_pf, _bad, _empty):
        if os.path.exists(_f):
            os.remove(_f)


    # ---------- 20. the holdout cannot leak across sibling frames ----------
    print("\n[20] specimen-aware holdout")
    from aggregate import specimen_key as _spec
    from train_v3_weighted import held_out_images as _hoi
    import train_v3_weighted as _tv

    check("sibling frames from one session share a specimen key",
          _spec("260708_316_H_b2_front_CBS_012") == _spec("260708_316_H_b2_front_CBS_013"),
          "leave-one-IMAGE-out over siblings is near-duplicate leakage")
    check("different sessions do not share one",
          _spec("260708_316_H_b2_front_CBS_012") != _spec("260622_316_H_b2_front_CBS_01"))
    check("families without a session token still get a distinct key",
          _spec("MAR_Amb_HIP_ETD_0010") != _spec("MAR_Amb_Cast_ETD_0005"))

    _grp = [f"260708_316_H_b2_front_CBS_{i:03d}" for i in range(1, 16)] + \
           ["AS_24hr_BSE_Side_008"]
    _orig = _tv.HELD
    try:
        _tv.HELD = "260708_316_H_b2_front_CBS_012"
        _imgs, _how = _hoi(_grp)
        check("holding out a sibling-rich frame excludes the WHOLE specimen",
              len(_imgs) > 1 and "SPECIMEN" in _how,
              f"{len(_imgs)} image(s) held out: {_how}")
        check("and it does not drag in an unrelated specimen",
              "AS_24hr_BSE_Side_008" not in _imgs)
    finally:
        _tv.HELD = _orig

    _src = _code_only(os.path.join(_here, "..", "..", "code", "establish_baseline.py")) \
        if os.path.exists(os.path.join(_here, "..", "..", "code", "establish_baseline.py")) \
        else _code_only(os.path.join(PROJECT_ROOT, "code", "establish_baseline.py"))
    check("establish_baseline uses the TRAINER's holdout, not its own",
          "held_out_images" in _src,
          "otherwise the gate's bar and the candidates it gates use two procedures again")


    # ---------- 21. censoring and the statistical unit ----------
    print("\n[21] censored lengths and pseudo-replication")
    import aggregate as _ag2

    check("a crack cut off by the frame edge is unknown, not uncensored",
          _ag2._is_censored({}) is None
          and _ag2._is_censored({"LengthIsCensored": True}) is True
          and _ag2._is_censored({"LengthIsCensored": False}) is False,
          "a CSV predating the flag carries no evidence either way; defaulting to False is "
          "the same 'not recorded means not the case' error the paper is about")
    check("string encodings from a CSV round-trip correctly",
          _ag2._is_censored({"LengthIsCensored": "True"}) is True
          and _ag2._is_censored({"LengthIsCensored": "False"}) is False
          and _ag2._is_censored({"LengthIsCensored": ""}) is None)

    _res = _ag2.aggregate(_cm.all_images(), by=("family", "condition"),
                          require_calibrated=False)
    if _res["groups"]:
        _g = _res["groups"][0]
        check("the specimen count is reported alongside the pooled crack count",
              isinstance(_g.get("n_specimens"), int) and _g["n_specimens"] >= 1
              and _g["n_specimens"] <= _g["n_images"],
              f"{_g['n_cracks']} cracks from {_g['n_images']} frames and "
              f"{_g['n_specimens']} specimen(s)")
        check("a group whose censoring is unknown refuses the longest-crack comparable",
              (_g.get("frames_with_unknown_censoring", 0) == 0)
              or (_g.get("longest_crack_is_valid_comparable") is False),
              "quoting a max over lower bounds is not a length")
        _single = [x for x in _res["groups"] if x.get("n_specimens") == 1]
        if _single:
            check("dispersion is refused when there is one independent unit",
                  _single[0].get("dispersion_is_estimable") is False
                  and bool(_single[0].get("dispersion_refusal")),
                  "per-crack spread measures variation WITHIN a specimen")
        check("every group states its statistical unit",
              all(bool(x.get("statistical_unit_note")) for x in _res["groups"]))

    # SYNTHETIC FIXTURE, so the right answer is known independently of the code under
    # test. Two specimens, two frames each. One crack per frame is enormous AND censored.
    # If censored cracks are pooled the specimen mean is dominated by them; if they are
    # excluded it is exactly 10.0.
    _fake = {
        "S1_f1": [{"SkeletonLength_px": 10.0, "LengthIsCensored": False},
                  {"SkeletonLength_px": 9000.0, "LengthIsCensored": True}],
        "S1_f2": [{"SkeletonLength_px": 10.0, "LengthIsCensored": False},
                  {"SkeletonLength_px": 8000.0, "LengthIsCensored": True}],
        "S2_f1": [{"SkeletonLength_px": 10.0, "LengthIsCensored": False},
                  {"SkeletonLength_px": 7000.0, "LengthIsCensored": True}],
        "S2_f2": [{"SkeletonLength_px": 10.0, "LengthIsCensored": False},
                  {"SkeletonLength_px": 6000.0, "LengthIsCensored": True}],
    }
    _real_read, _real_spec = _ag2._read_rows, _ag2.specimen_key
    try:
        _ag2._read_rows = lambda n: list(_fake.get(n, []))
        _ag2.specimen_key = lambda n: n.split("_")[0]
        _ag2.parse_name = _ag2.parse_name  # untouched
        _r = _ag2.aggregate(list(_fake), by=(), require_calibrated=False)
        _g0 = _r["groups"][0] if _r["groups"] else {}
        _m = (_g0.get("mean_crack_length_by_specimen") or {}).get("mean")
        check("the specimen mean excludes right-censored cracks",
              _m is not None and abs(_m - 10.0) < 1e-9,
              f"got {_m}; pooling the censored cracks would give ~3757, and on the real "
              f"corpus it did: family=steel|condition=H shipped 297.09 where the "
              f"uncensored cracks give 102.04")
        check("the censoring split is reported so the exclusion is visible",
              _g0.get("n_cracks_uncensored") == 4 and _g0.get("n_cracks_censored") == 4
              and _g0.get("n_cracks_censoring_unknown") == 0)
        check("the figure is labelled as covering uncensored cracks only",
              "UNCENSORED" in (_g0.get("mean_crack_length_by_specimen_note") or "")
              and "UNDERSTATES" in (_g0.get("mean_crack_length_by_specimen_note") or ""),
              "excluding censored cracks removes the longest ones, so the mean is biased "
              "low and must not be called mean crack length")

        # Censoring UNKNOWN must not be treated as uncensored.
        _fake2 = {"S1_f1": [{"SkeletonLength_px": 10.0},
                            {"SkeletonLength_px": 9000.0}]}
        _ag2._read_rows = lambda n: list(_fake2.get(n, []))
        _r2 = _ag2.aggregate(list(_fake2), by=(), require_calibrated=False)
        _g2 = _r2["groups"][0] if _r2["groups"] else {}
        check("cracks with no censoring flag are counted as unknown, not as uncensored",
              _g2.get("n_cracks_censoring_unknown") == 2
              and _g2.get("n_cracks_uncensored") == 0,
              "a CSV predating the flag carries no evidence either way")
    finally:
        _ag2._read_rows, _ag2.specimen_key = _real_read, _real_spec

    # An unidentifiable specimen must not become its own unit.
    check("specimen_key returns None when the filename cannot be read",
          _ag2.specimen_key("some_strangers_file_0001") is None,
          "it used to return a per-frame unique string, so every foreign frame counted as "
          "its own specimen and the pseudo-replication guard inverted")
    _real_read2 = _ag2._read_rows
    try:
        _ag2._read_rows = lambda n: [{"SkeletonLength_px": 10.0,
                                      "LengthIsCensored": False}]
        _r3 = _ag2.aggregate(["alien_a", "alien_b", "alien_c", "alien_d"], by=(),
                             require_calibrated=False)
        _g3 = _r3["groups"][0] if _r3["groups"] else {}
        check("four frames of unknown specimen do not become four specimens",
              _g3.get("n_specimens") is None
              and _g3.get("n_frames_with_unknown_specimen") == 4,
              f"n_specimens={_g3.get('n_specimens')} -- it must be UNKNOWN, not 4")
        check("dispersion is refused when specimen identity is unknown",
              _g3.get("dispersion_is_estimable") is False
              and "independence cannot be established"
                  in (_g3.get("dispersion_refusal") or ""))
    finally:
        _ag2._read_rows = _real_read2

    # The model holdout must not hold out every unreadable name together.
    import train_v3_weighted as _tv
    _sibs, _why = _tv.held_out_images(["alien_a", "alien_b", _tv.HELD])
    check("the retrain holdout falls back when the specimen is unidentifiable",
          _tv.HELD in _sibs and "alien_a" not in _sibs,
          f"{_why}")

    # A refused figure must not be quotable from the CSV either -- that is exactly where a
    # reader would pick it up.
    _csvp = os.path.join(_ag2.OUT_DIR, "aggregate.csv")
    if os.path.exists(_csvp):
        import csv as _csv2
        with open(_csvp, newline="") as fh:
            _rows2 = list(_csv2.DictReader(fh))
        _bad = [r for r in _rows2
                if r.get("longest_crack_is_valid_comparable") == "False"
                and r.get("longest_crack_mean", "") != ""]
        check("the CSV blanks a refused longest-crack figure rather than printing it",
              not _bad, f"{len(_bad)} row(s) print a refused number")
        check("the CSV leads with the specimen count",
              list(_rows2[0].keys())[2] == "n_specimens" if _rows2 else False,
              "so a reader meets the real n before the impressive one")


    # ---------- 22. instrument metadata as a calibration source ----------
    print("\n[22] instrument metadata")
    import calibration as _cal2
    import tempfile as _tf
    from PIL import TiffImagePlugin as _TP

    _md = _tf.mkdtemp()

    def _vendor_tif(tag, block, w=2048):
        # width in the name: same block at two widths must not share a file
        _p = os.path.join(_md, f"v{tag}_{w}_{abs(hash(block)) % 99999}.tif")
        _im = Image.fromarray(np.zeros((64, w), np.uint8))
        _info = _TP.ImageFileDirectory_v2()
        _info[tag] = block
        _im.save(_p, tiffinfo=_info)
        return _p

    # THE UNIT IS NEVER PICKED BY MAGNITUDE. This block previously asserted the opposite
    # of what it claimed: it checked that a unitless FEI "HFW=204.8" read as 204.8 um,
    # while the code was deciding metres-vs-micrometres with `val < 1e-3`. That heuristic
    # was wrong in both directions -- a routine 2 mm overview field (0.002 m) came out
    # 1e6 too small, and 0.00104 m, the case the module docstring cites, was refused.
    # A stated unit wins; with none stated the tag block's documented SI convention
    # applies; an implausible RESULT is refused rather than reinterpreted.
    def _md_read(tag, block, w=2048):
        return _cal2.read_instrument_metadata(_vendor_tif(tag, block, w))

    _cases = [
        ("field width in metres",            34682, "[EBeam]\nHorizontalFieldWidth=2.048e-4\n", 2048, 0.1),
        ("a 2 mm overview field",            34682, "[EBeam]\nHorizontalFieldWidth=0.002\n",    1536, 2000 / 1536),
        ("the 1.04 mm frame in the docs",    34682, "[EBeam]\nHorizontalFieldWidth=0.00104\n",  6144, 1040 / 6144),
        ("1e-3 exactly, the old cliff edge", 34682, "[EBeam]\nHorizontalFieldWidth=0.001\n",    1024, 1000 / 1024),
        ("just below the old cliff",         34682, "[EBeam]\nHorizontalFieldWidth=0.0009999\n", 1024, 999.9 / 1024),
        ("a pixel size, not a field width",  34682, "[Scan]\nPixelWidth=1.0e-7\n",              2048, 0.1),
        ("a stated unit is believed",        34683, "[EBeam]\nHFW=204.8 um\n",                  2048, 0.1),
        ("a stated unit with no space",      34683, "[EBeam]\nHFW=204.8um\n",                   2048, 0.1),
        ("millimetres, stated",              34682, "[EBeam]\nHorizontalFieldWidth=1.04 mm\n",  6144, 1040 / 6144),
        ("a ZEISS pixel size in metres",     34118, "AP_IMAGE_PIXEL_SIZE = 5e-8\n",             2048, 0.05),
        ("a ZEISS pixel size in nm",         34118, "AP_IMAGE_PIXEL_SIZE = 2.5 nm\n",           2048, 0.0025),
    ]
    _wrong = []
    for _lbl, _tg, _blk, _w, _exp in _cases:
        _got = _md_read(_tg, _blk, _w)
        if not (_got and abs(_got["um_per_px"] - _exp) / _exp < 1e-9):
            _wrong.append(f"{_lbl}: got {_got and _got['um_per_px']}, want {_exp}")
    check("every vendor form reads to the right pixel size",
          not _wrong, "; ".join(_wrong) if _wrong else f"{len(_cases)} forms, incl. both "
          f"sides of the 1e-3 boundary the old magnitude test broke on")

    # An unstated unit is a CONVENTION, and applying it can produce nonsense. Nonsense must
    # be refused, never quietly reinterpreted as the other unit -- reinterpreting is the
    # magnitude heuristic again, and it turns a loud refusal into a silent wrong answer.
    _absurd = [("a unitless 204.8 read as metres", 34683, "[EBeam]\nHFW=204.8\n"),
               ("a 100 m field",   34682, "[EBeam]\nHorizontalFieldWidth=100\n"),
               ("a negative width", 34682, "[EBeam]\nHorizontalFieldWidth=-2e-4\n"),
               ("zero",            34682, "[EBeam]\nHorizontalFieldWidth=0\n"),
               ("not a number",    34682, "[EBeam]\nHorizontalFieldWidth=abc\n")]
    _leaked = [l for l, t, b in _absurd if _md_read(t, b) is not None]
    check("an implausible result is refused, not reinterpreted as the other unit",
          not _leaked, f"{_leaked} were accepted" if _leaked else
          "including a unitless 204.8, which under the tag's metre convention is 1e5 um/px")

    _prov = _md_read(34682, "[EBeam]\nHorizontalFieldWidth=2.048e-4\n")
    check("the record says which unit was used and whether it was assumed",
          _prov and _prov["detail"].get("unit_used") == "m"
          and _prov["detail"].get("unit_was_assumed") is True
          and _prov["detail"].get("unit_stated_in_file") is None,
          "a reader must be able to see that the unit came from a convention, not the file")
    _stated = _md_read(34683, "[EBeam]\nHFW=204.8 um\n")
    check("a unit read from the file is recorded as not assumed",
          _stated and _stated["detail"].get("unit_was_assumed") is False
          and _stated["detail"].get("unit_stated_in_file") == "um")

    _plain = os.path.join(_md, "plain.tif")
    Image.fromarray(np.zeros((8, 8), np.uint8)).save(_plain)
    check("a file with no vendor block returns None rather than a guess",
          _cal2.read_instrument_metadata(_plain) is None)

    # This repo's own corpus: every file was re-saved and lost its vendor tags. If this
    # ever stops being true the audit in read_instrument_metadata's docstring is stale.
    _real = [n for n in _cm.all_images()][:6]
    _with_md = [n for n in _real
                if _cal2.read_instrument_metadata(
                    os.path.join(ORIGINAL_DIR, f"{n}.tif")) is not None]
    check("the shipped corpus carries no instrument metadata, as documented",
          not _with_md,
          f"{len(_with_md)}/{len(_real)} probed images had vendor tags; the docstring says 0")

    _saved_cal = _cal2.CALIB_PATH
    try:
        _cal2.CALIB_PATH = os.path.join(_md, "calibration.json")
        _cal2.set_manual("disagree", 0.2)
        _dis = _vendor_tif(34682, "[EBeam]\nHorizontalFieldWidth=2.048e-4\n")
        _refused = False
        try:
            _cal2.set_from_instrument_metadata("disagree", _dis)
        except ValueError:
            _refused = True
        check("metadata disagreeing with a hand reading is refused, not preferred",
              _refused,
              "a machine value silently overwriting a human's is the failure this "
              "project is about")
        _cal2.set_manual("agree", 0.101)
        _cal2.set_from_instrument_metadata(
            "agree", _vendor_tif(34682, "[EBeam]\nHorizontalFieldWidth=2.0480e-4\n"))
        check("metadata agreeing within tolerance is stored",
              abs(_cal2.get_um_per_px("agree") - 0.1) < 1e-9)
    finally:
        _cal2.CALIB_PATH = _saved_cal

    # ---------- 23. the headless batch CLI ----------
    print("\n[23] headless batch CLI")
    import subprocess as _sp
    _CLI = os.path.join(MAIN_CODE_DIR, "semcrack.py")
    _py = sys.executable

    def _cli(*args):
        r = _sp.run([_py, _CLI, *args], capture_output=True, text=True)
        return r.returncode, (r.stdout + r.stderr)

    check("the CLI exists and is importable as a script", os.path.exists(_CLI))

    _cin = os.path.join(_md, "in")
    os.makedirs(_cin, exist_ok=True)
    _rng2 = np.random.default_rng(7)
    _a = (42000 + _rng2.normal(0, 1200, (600, 800))).astype(np.uint16)
    for _i in range(400):
        _a[200 + int(50 * np.sin(_i / 60)) - 3:203 + int(50 * np.sin(_i / 60)), 200 + _i] = 2200
    Image.fromarray(_a).save(os.path.join(_cin, "cli_synth_00.tif"))

    _rc, _out = _cli("--in", _cin, "--out", os.path.join(_md, "no_such"), "--glob", "*.png")
    check("no matching files exits 2 rather than reporting an empty success", _rc == 2)
    _rc, _out = _cli("--in", os.path.join(_md, "absent"), "--out", os.path.join(_md, "o"))
    check("a missing input directory exits 2", _rc == 2)
    _rc, _out = _cli("--in", _cin, "--out", os.path.join(_md, "o"), "--threshold", "1.5")
    check("a threshold outside (0,1) is rejected before any work", _rc == 2)
    _rc, _out = _cli("--in", _cin, "--out", os.path.join(_md, "o"),
                     "--um-per-px", "0.05", "--scale-csv", os.devnull)
    check("two sources of scale are refused as mutually exclusive", _rc == 2,
          "two sources that disagree cannot both be right")

    _o1 = os.path.join(_md, "out1")
    _rc, _out = _cli("--in", _cin, "--out", _o1, "--dry-run")
    check("--dry-run reports the plan and exits 0 without measuring",
          _rc == 0 and "dry-run" in _out
          and not os.path.exists(os.path.join(_o1, "run_manifest.json")))

    _rc, _out = _cli("--in", _cin, "--out", _o1)
    check("a real batch run exits 0 and writes a manifest", _rc == 0
          and os.path.exists(os.path.join(_o1, "run_manifest.json")), _out[-160:])
    if os.path.exists(os.path.join(_o1, "run_manifest.json")):
        _man = json.load(open(os.path.join(_o1, "run_manifest.json")))
        check("the manifest pins the model by content hash, not by path",
              isinstance(_man.get("model_sha256"), str) and len(_man["model_sha256"]) == 64)
        check("the manifest records that no human correction was applied",
              _man.get("corrections_applied") is False
              and "detector only" in _man.get("corrections_note", ""),
              "a batch CSV that is part human judgement must say which part")
        check("the manifest names the threshold that produced the numbers",
              "threshold" in _man and bool(_man.get("threshold_note")))
        check("per-image CSVs land in the requested output directory",
              os.path.exists(os.path.join(_o1, "cli_synth_00_crack_measurements.csv")))
        check("uncalibrated output reports pixels and no invented um columns",
              _man.get("n_calibrated") == 0)

    _rc, _out = _cli("--in", _cin, "--out", _o1, "--threshold", "0.7")
    check("a second run with a different threshold refuses to share the directory",
          _rc == 3, "mixing configurations gives rows with no way to tell them apart")
    # The manifest must always name the number, and naming it explicitly must not read as
    # a different configuration from leaving it to the fallback.
    if os.path.exists(os.path.join(_o1, "run_manifest.json")):
        _m0 = json.load(open(os.path.join(_o1, "run_manifest.json")))
        check("the manifest states the EFFECTIVE threshold even when none was given",
              isinstance(_m0.get("threshold"), float),
              f"threshold={_m0.get('threshold')} as_given={_m0.get('threshold_as_given')}")
        _rc, _out = _cli("--in", _cin, "--out", _o1,
                         "--threshold", str(_m0["threshold"]))
        check("passing the fallback explicitly is the SAME configuration, not a conflict",
              _rc == 0,
              "fingerprinting the raw flag made 'unspecified' and 'specified as the same "
              "value' look different")
    _rc, _out = _cli("--in", _cin, "--out", _o1, "--threshold", "0.7", "--force")
    check("--force allows it deliberately", _rc == 0)

    _bad = os.path.join(_md, "bad")
    os.makedirs(_bad, exist_ok=True)
    shutil.copy(os.path.join(_cin, "cli_synth_00.tif"), _bad)
    open(os.path.join(_bad, "broken.tif"), "w").write("not a tiff")
    _rc, _out = _cli("--in", _bad, "--out", os.path.join(_md, "out_bad"))
    check("one unreadable image fails that image and exits 1, not the batch", _rc == 1,
          "a run that measured half the frames and exited 0 reads as success")
    _mb = os.path.join(_md, "out_bad", "run_manifest.json")
    if os.path.exists(_mb):
        _m2 = json.load(open(_mb))
        check("the manifest records which image failed and why",
              _m2["n_ok"] == 1 and _m2["n_failed"] == 1
              and any(r["status"] == "failed" and r.get("error") for r in _m2["images"]))

    _mix = os.path.join(_md, "mixed")
    os.makedirs(_mix, exist_ok=True)
    shutil.copy(os.path.join(_cin, "cli_synth_00.tif"), _mix)
    Image.fromarray(np.full((200, 300), 40000, np.uint16)).save(
        os.path.join(_mix, "other_size.tif"))
    _rc, _out = _cli("--in", _mix, "--out", os.path.join(_md, "out_mix"),
                     "--um-per-px", "0.05")
    check("one scale across differently-sized frames is refused", _rc == 3,
          "same scale at two pixel sizes is a magnification assumption")

    # MIXED UNITS, FOR REAL THIS TIME. There is a check elsewhere that a group containing an
    # uncalibrated image reports pixels, and it passed for years without testing anything,
    # because zero images in this corpus are calibrated -- so the mixed case never arose. The
    # batch CLI can manufacture it: calibrate one of two images and group them together.
    _sc = os.path.join(_md, "scale_one.csv")
    with open(_sc, "w") as fh:
        fh.write("image,um_per_px\ncli_synth_00,0.05\n")
    _mixdir = os.path.join(_md, "mixdir")
    os.makedirs(_mixdir, exist_ok=True)
    shutil.copy(os.path.join(_cin, "cli_synth_00.tif"), _mixdir)
    Image.fromarray(_a.copy()).save(os.path.join(_mixdir, "cli_synth_01.tif"))
    _om = os.path.join(_md, "out_mixunits")
    _rc, _out = _cli("--in", _mixdir, "--out", _om, "--scale-csv", _sc,
                     "--group-by", "none")
    _agg = os.path.join(_om, "aggregate.csv")
    if _rc == 0 and os.path.exists(_agg):
        import csv as _csv3
        _rows3 = list(_csv3.DictReader(open(_agg)))
        _r3 = _rows3[0] if _rows3 else {}
        check("one calibrated image + one uncalibrated reports PIXELS, not a blend",
              _r3.get("units") == "px" and _r3.get("n_calibrated") == "1"
              and _r3.get("n_uncalibrated") == "1",
              f"units={_r3.get('units')} cal={_r3.get('n_calibrated')} "
              f"uncal={_r3.get('n_uncalibrated')} -- this is the case the old check could "
              f"never reach")
        _mm = json.load(open(os.path.join(_om, "run_manifest.json")))
        check("the image missing from the scale CSV is named, not silently left in pixels",
              any(r["kind"] == "image_absent_from_scale_csv" for r in _mm["refusals"]))

        _sc2 = os.path.join(_md, "scale_both.csv")
        with open(_sc2, "w") as fh:
            fh.write("image,um_per_px\ncli_synth_00,0.05\ncli_synth_01,0.05\n")
        _om2 = os.path.join(_md, "out_bothunits")
        _rc2, _ = _cli("--in", _mixdir, "--out", _om2, "--scale-csv", _sc2,
                       "--group-by", "none")
        if _rc2 == 0 and os.path.exists(os.path.join(_om2, "aggregate.csv")):
            _r4 = list(_csv3.DictReader(open(os.path.join(_om2, "aggregate.csv"))))[0]
            check("with every image calibrated the same group reports MICROMETRES",
                  _r4.get("units") == "um" and _r4.get("n_uncalibrated") == "0",
                  "the contrast is what makes the previous check mean something")

    _dupdir = os.path.join(_md, "dupdir")
    os.makedirs(_dupdir, exist_ok=True)
    shutil.copy(os.path.join(_cin, "cli_synth_00.tif"), os.path.join(_dupdir, "z.tif"))
    shutil.copy(os.path.join(_cin, "cli_synth_00.tif"), os.path.join(_dupdir, "z.tiff"))
    _rc, _ = _cli("--in", _dupdir, "--out", os.path.join(_md, "out_dup"), "--glob", "z.*")
    check("two inputs sharing a basename are refused, not silently overwritten", _rc == 3,
          "every output file is named from the basename, so the last one would win")

    # The batch entry point must not have changed where the app itself reads and writes.
    check("the overrides are inert for the app: repo paths unchanged",
          ORIGINAL_DIR.endswith("/sem-crack-detector/original")
          and _cm.OUT_DIR.endswith("/interior_active_learning/measurements"),
          f"{ORIGINAL_DIR} | {_cm.OUT_DIR}")
    shutil.rmtree(_md, ignore_errors=True)


    # ---------- 24. how precise is the scale itself ----------
    print("\n[24] calibration uncertainty")
    _cal3 = _cal2
    _ud = _tf.mkdtemp()
    _saved3 = _cal3.CALIB_PATH
    try:
        _cal3.CALIB_PATH = os.path.join(_ud, "c.json")
        _cal3.set_from_scale_bar("u200", 20.0, 100, 300)
        _r200 = _cal3.relative_uncertainty("u200")
        check("a marked scale bar records how precisely it was marked",
              _r200 is not None and abs(_r200 - (2 ** 0.5) * 1.5 / 200) < 1e-12,
              f"200 px span, two independent 1.5 px endpoint errors -> {100 * _r200:.2f}%")

        _cal3.set_from_scale_bar("u60", 20.0, 0, 60)
        _cal3.set_from_scale_bar("u800", 20.0, 0, 800)
        check("marking a longer bar is recorded as more precise",
              _cal3.relative_uncertainty("u60") > _r200
              > _cal3.relative_uncertainty("u800"),
              f"{100 * _cal3.relative_uncertainty('u60'):.2f}% > {100 * _r200:.2f}% > "
              f"{100 * _cal3.relative_uncertainty('u800'):.2f}%")

        _cal3.set_manual("um", 0.05)
        check("a route that cannot know its own precision reports None, not zero",
              _cal3.relative_uncertainty("um") is None,
              "'we did not characterise this' and 'this is exact' are different claims, "
              "and the second is the error this project is about")

        check("area carries the scale uncertainty twice and length once",
              abs(_cal3.propagate(0.011, 2) - 0.022) < 1e-12
              and abs(_cal3.propagate(0.011, 1) - 0.011) < 1e-12)
        check("a dimensionless quantity carries none of it",
              _cal3.propagate(0.011, 0) == 0.0,
              "tortuosity is a ratio of two lengths, so the scale cancels")
        check("unknown propagates as unknown rather than as exact",
              _cal3.propagate(None, 1) is None and _cal3.propagate(None, 2) is None)

        _h = _cal3.provenance_header("u200", model_path=PROD_MODEL_PATH, threshold=0.5)
        check("the provenance sidecar states the scale's uncertainty",
              _h.get("um_per_px_rel_sd") is not None and bool(_h.get("uncertainty_note")))
        check("the sidecar gives the per-column figure a reader needs",
              isinstance(_h.get("column_rel_uncertainty"), dict)
              and abs(_h["column_rel_uncertainty"]["Area_um2"]
                      - 2 * _h["um_per_px_rel_sd"]) < 1e-12,
              "so nobody has to rederive that area doubles it")
        check("the note says this is instrument uncertainty only",
              "segmentation" in _h.get("uncertainty_note", ""),
              "segmentation error is larger and is NOT quantified; implying otherwise "
              "would be a bigger lie than omitting the interval")
        _hm = _cal3.provenance_header("um")
        check("an uncharacterised scale says so instead of omitting the field",
              _hm.get("um_per_px_rel_sd") is None
              and "NOT characterised" in _hm.get("uncertainty_note", ""),
              "a missing key reads as zero")
        check("an uncalibrated image gets no uncertainty fields at all",
              "um_per_px_rel_sd" not in _cal3.provenance_header("never_calibrated_xyz"))
    finally:
        _cal3.CALIB_PATH = _saved3
        shutil.rmtree(_ud, ignore_errors=True)

    # The UI must never render an uncharacterised scale as +/-0%.
    _fe = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "paint_frontend.py")).read()
    check("the frontend has a helper for showing scale precision",
          "function scalePrecisionText" in _fe)
    check("it guards on the value being a number, so null shows nothing",
          "typeof rel !== 'number'" in _fe,
          "an absent uncertainty must render as nothing, never as +/-0%")
    import re as _re3
    _bad_tokens = [t for t in ("--good", "--text-dim", "--text-faint")
                   if f"var({t})" in _re3.sub(r"//[^\n]*", "", _fe)]
    check("every CSS variable the frontend uses is actually defined",
          not _bad_tokens,
          f"{_bad_tokens} resolve to nothing, so the style silently falls back -- this is "
          f"how a SUCCESSFUL calibration showed no colour while a failed one went red")


    # ---------- 25. the threshold override must reach EVERY decision point ----------
    print("\n[25] threshold override coverage")
    import unified_pipeline as _up2

    class _FakeClf:
        def predict_proba(self, X):
            return np.array([[0.30, 0.70]])       # one candidate at p=0.70

    class _FakeScaler:
        def transform(self, X):
            return np.asarray(X, dtype=float)

    _fb = {"clf": _FakeClf(), "scaler": _FakeScaler(), "threshold_default": 0.5,
           "interior_fill_rule": {"floor": 0.65, "dist_thr": 1e9, "bri_thr": 1e9}}
    _ff = [{c: 0.0 for c in _up2.INTERIOR_FEATURE_COLUMNS}]
    _saved_ovr = _up2.THRESHOLD_OVERRIDE
    try:
        # interior_fill carries its OWN calibrated floor (0.65). It used to ignore the
        # override entirely, so --threshold moved Pass 1 and two of three Pass 2 candidate
        # types while this one stayed put -- and the sensitivity sweep therefore measured a
        # partly frozen detector and understated the effect.
        for _ct in ("interior_fill", "concavity", "bridge_corridor"):
            _up2.THRESHOLD_OVERRIDE = None
            _base = _up2._score(_fb, _ff, _ct)[0][0]
            _up2.THRESHOLD_OVERRIDE = 0.75
            _hi = _up2._score(_fb, _ff, _ct)[0][0]
            _up2.THRESHOLD_OVERRIDE = 0.10
            _lo = _up2._score(_fb, _ff, _ct)[0][0]
            check(f"the override moves the {_ct!r} decision",
                  bool(_base) and bool(_lo) and not _hi,
                  f"p=0.70: default={_base}, at 0.75={_hi} (want False), "
                  f"at 0.10={_lo} (want True)")
        _up2.THRESHOLD_OVERRIDE = None
        check("with no override the interior_fill floor is the bundle's own 0.65",
              _up2._score(_fb, _ff, "interior_fill")[0][0] is True
              and _up2._score({**_fb, "interior_fill_rule":
                               {**_fb["interior_fill_rule"], "floor": 0.9}},
                              _ff, "interior_fill")[0][0] is False,
              "the override must not silently replace a calibrated rule when unset")
        # The two feature gates are NOT thresholds on probability and must not be rescaled.
        _tight = {**_fb, "interior_fill_rule": {"floor": 0.1, "dist_thr": -1.0,
                                                "bri_thr": 1e9}}
        _up2.THRESHOLD_OVERRIDE = 0.10
        check("the override does not override the geometry gate",
              _up2._score(_tight, _ff, "interior_fill")[0][0] is False,
              "dist_thr is a distance, not a probability; rescaling it would be meaningless")
    finally:
        _up2.THRESHOLD_OVERRIDE = _saved_ovr

    check("the app itself still sees no override",
          _up2.THRESHOLD_OVERRIDE is None,
          "this must be inert unless a caller deliberately sets it")


    print("\n[cleanup]")
    removed = 0
    for n in created:
        for p in [os.path.join(ORIGINAL_DIR, f"{n}.tif"),
                  os.path.join(PAINT_DIR, f"{n}_paint_template.png"),
                  os.path.join(PAINT_DIR, f"{n}_correction_mask.png"),
                  os.path.join(PAINT_DIR, f"{n}_painted.png")]:
            if os.path.exists(p):
                os.remove(p); removed += 1
    cj = os.path.join(PAINT_DIR, "candidate_counts.json")
    if os.path.exists(cj):
        c = json.load(open(cj))
        for n in created:
            c.pop(n, None)
        json.dump(c, open(cj, "w"), indent=2)
    # shutil is imported at module scope; a second import HERE made it a local name
    # for the whole function, so every earlier use raised UnboundLocalError.
    shutil.rmtree(TMP, ignore_errors=True)
    check("test artifacts removed", True, f"{removed} files, {len(created)} images")

    n_pass = sum(1 for _, ok, _ in results if ok)
    n_fail = len(results) - n_pass
    print("\n" + "=" * 70)
    print(f"{n_pass} passed, {n_fail} failed, {len(results)} total")
    if n_fail:
        print("\nfailures:")
        for name, ok, detail in results:
            if not ok:
                print(f"  - {name}  {detail}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
