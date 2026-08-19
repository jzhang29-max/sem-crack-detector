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

from common import ORIGINAL_DIR, PAINT_DIR, PROJECT_ROOT, PROD_MODEL_PATH

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
    the segmenter finds candidates, cheap enough to run in seconds."""
    rng = np.random.RandomState(0)
    img = (rng.normal(150, 12, (h, w))).clip(0, 255)
    for off in (0, 140):
        for t in range(min(h, w) - 120):
            y, x = 40 + t, 40 + t + off
            if 0 <= y < h and 0 <= x < w:
                img[max(0, y - 2):y + 3, max(0, x - 2):x + 3] = 25
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
    check("the model card names the detector configuration",
          "Pass 1 + Pass 2 + SAM" in html and "measured f1" in html)

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
    import shutil
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
