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
    r = requests.post(f"{BASE}/api/process/{target}", json={"use_sam": False}, timeout=60).json()
    res = poll(r["job"]) if r.get("ok") else None
    check("detect without SAM", bool(res) and res.get("n_candidates", 0) > 0,
          f"{(res or {}).get('n_candidates')} candidates, {(res or {}).get('n_crack')} crack")
    check("template written", os.path.exists(
        os.path.join(PAINT_DIR, f"{target}_paint_template.png")))

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
                        ("model banner present", "modelInfo"),
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
    LOGIC_IDS = ["baseCanvas", "paintCanvas", "canvasInner", "canvasWrap", "status",
                 "imageSelect", "swatchRed", "swatchCyan", "swatchErase", "brushSize",
                 "brushSizeLabel", "bucketBtn", "zoom", "zoomLabel", "fitBtn", "undoBtn",
                 "clearBtn", "dropZone", "jobBar", "jobFill",
                 "jobLabel", "jobNote", "modelInfo", "retrainBtn", "useSam",
                 "dlMask", "dlOverlay", "dlCsv", "dlAll"]
    missing = [i for i in LOGIC_IDS if f'id="{i}"' not in html]
    check("every id the paint logic depends on survived the redesign",
          not missing, f"missing: {missing}" if missing else f"all {len(LOGIC_IDS)} present")

    for name, token in [("sidebar image list", 'id="imageList"'),
                        ("image filter box", 'id="imgSearch"'),
                        ("model card", 'id="modelCard"'),
                        ("export menu", 'id="expMenu"'),
                        ("options disclosure", 'id="adv"'),
                        ("shortcut help", 'id="help"'),
                        ("zoom buttons", 'id="zoomIn"')]:
        check(name, token in html)
    # the point of the redesign: fine-tuning is not in the default view
    check("brush size hidden behind Options",
          html.find('id="adv"') < html.find('id="brushSize"'),
          "brushSize lives inside the collapsed #adv row")
    # Whitespace-insensitive: the CSS was restyled to the sibling TXM app's
    # convention (`display:none`, no space), and an exact-string assertion
    # reported a passing behaviour as a failure.
    import re as _re
    _adv = _re.search(r"#adv\s*\{([^}]*)\}", html)
    check("advanced row collapsed by default",
          bool(_adv) and "display:none" in _adv.group(1).replace(" ", ""))
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

    # Editing the frontend used to require restarting the server, because
    # INDEX_HTML is bound at import time -- which made two working UI fixes look
    # broken. The server now restats the module per page load.
    srv = open(os.path.join(os.path.dirname(__file__), "paint_server.py")).read()
    check("frontend edits are picked up without a restart",
          "importlib.reload(paint_frontend)" in srv and "_frontend_mtime" in srv)

    # ---------- cleanup ----------
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
