"""
In-app downloads: B&W mask, burned-in overlay, per-region CSV, and an
all-images zip with a cross-image summary.

Added because the sibling TXM app has these and this one did not -- masks and
overlays could only be produced by running scripts from a terminal, which is not
usable by someone who just wants the numbers. For anyone measuring crack growth
across a series, the per-region table IS the product; the overlay is how you
check the table is believable.

Conventions match the TXM app so results from the two are interchangeable:
    * crack is BLACK on white in exported B&W masks
    * the CSV carries one row per connected crack region
    * long operations return a job id and stream progress

The crack mask is read back out of the paint template rather than recomputed, so
what is exported is exactly what is on screen, including every human correction.
The recovery is algebraic (build_simple_overlay alpha-blends: crack pixels
satisfy G == B and R-G in {140,141}) and was verified against the pipeline at 0
pixels difference -- an RGB threshold would have been brightness-dependent and
wrong on bright images, which is a mistake already made once here.

Registered routes:
    GET  /api/export/<name>/mask.png       crack = black
    GET  /api/export/<name>/overlay.png    red crack burned into the image
    GET  /api/export/<name>/regions.csv    one row per crack region
    POST /api/export_all                   -> job id; zip of everything + summary.csv
    GET  /api/export_all/<job_id>/download
"""
import io
import os
import time
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "code"))

import numpy as np
import pandas as pd
from PIL import Image
from flask import jsonify, request, send_file
from skimage import measure

from common import ORIGINAL_DIR, PAINT_DIR, PROJECT_ROOT, contrast_kwargs_for
from detect_cracks import load_as_uint8, find_field_of_view

Image.MAX_IMAGE_PIXELS = None
EXPORT_DIR = os.path.join(PROJECT_ROOT, "figures", "exports")


def crack_mask(image_name):
    p = os.path.join(PAINT_DIR, f"{image_name}_paint_template.png")
    if not os.path.exists(p):
        return None
    a = np.array(Image.open(p).convert("RGB")).astype(np.int16)
    d = a[..., 0] - a[..., 1]
    return (a[..., 1] == a[..., 2]) & ((d == 140) | (d == 141))


def base_image(image_name):
    img8 = load_as_uint8(os.path.join(ORIGINAL_DIR, f"{image_name}.tif"),
                         **contrast_kwargs_for(image_name))
    x0, y0, x1, y1 = find_field_of_view(img8)
    return img8[y0:y1, x0:x1]


def region_table(image_name, m):
    """One row per connected crack region.

    Pixel units only -- no micron conversion. The scale bar is burned into the
    databar this pipeline deliberately crops off, so a µm/px factor is not
    recoverable from the data and inventing one would put a fabricated number in
    a results table. Multiply externally by the value from the microscope.
    """
    lab = measure.label(m, connectivity=2)
    rows = []
    for p in measure.regionprops(lab):
        y0, x0, y1, x1 = p.bbox
        minor = p.axis_minor_length if p.axis_minor_length > 0 else 0.5
        rows.append({
            "Region": int(p.label), "Area_px": int(p.area),
            "CentroidY_px": round(float(p.centroid[0]), 2),
            "CentroidX_px": round(float(p.centroid[1]), 2),
            "BBoxY0": int(y0), "BBoxX0": int(x0), "BBoxY1": int(y1), "BBoxX1": int(x1),
            "Length_px": round(float(p.axis_major_length), 2),
            "Width_px": round(float(p.axis_minor_length), 2),
            "AspectRatio": round(float(p.axis_major_length / minor), 3),
            "Eccentricity": round(float(p.eccentricity), 4),
            "Orientation_deg": round(float(np.degrees(p.orientation)), 2),
            "Solidity": round(float(p.solidity), 4),
            "Perimeter_px": round(float(p.perimeter), 2),
        })
    # Explicit columns so an image with no crack regions still writes a header.
    # pd.DataFrame([]) has no columns, so regions.csv came out as a single newline
    # -- 1 byte, no header -- which a reader treats as malformed rather than as
    # "zero regions found".
    _COLS = ["Region", "Area_px", "CentroidY_px", "CentroidX_px", "BBoxY0", "BBoxX0",
             "BBoxY1", "BBoxX1", "Length_px", "Width_px", "AspectRatio",
             "Eccentricity", "Orientation_deg", "Solidity", "Perimeter_px"]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=_COLS)
    if len(df):
        df = df.sort_values("Area_px", ascending=False).reset_index(drop=True)
    return df


def register(app, list_images):
    def _mask_or_404(name):
        m = crack_mask(name)
        if m is None:
            return None, (jsonify({"ok": False, "error": "no overlay yet -- process it first"}), 404)
        return m, None

    @app.route("/api/export/<image_name>/mask.png")
    def api_export_mask(image_name):
        m, err = _mask_or_404(image_name)
        if err:
            return err
        # crack BLACK on white, matching the TXM app's convention
        out = np.where(m, 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(out).save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png", as_attachment=True,
                         download_name=f"{image_name}_mask.png")

    @app.route("/api/export/<image_name>/overlay.png")
    def api_export_overlay(image_name):
        m, err = _mask_or_404(image_name)
        if err:
            return err
        img8 = base_image(image_name)
        if img8.shape != m.shape:
            return jsonify({"ok": False, "error": "overlay is stale -- reprocess"}), 409
        rgb = np.stack([img8] * 3, -1).astype(np.uint8)
        rgb[m] = (225, 25, 25)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png", as_attachment=True,
                         download_name=f"{image_name}_overlay.png")

    @app.route("/api/export/<image_name>/regions.csv")
    def api_export_regions(image_name):
        m, err = _mask_or_404(image_name)
        if err:
            return err
        df = region_table(image_name, m)
        buf = io.BytesIO(df.to_csv(index=False).encode())
        return send_file(buf, mimetype="text/csv", as_attachment=True,
                         download_name=f"{image_name}_regions.csv")

    def _prune_old_exports(keep_newest=3, max_age_s=6 * 3600):
        """Drop stale export zips before writing a new one.

        Each export is a full zip of every rendered overlay and CSV, so a researcher who
        exports a few times a day quietly loses gigabytes inside the project directory --
        the same volume the correction masks live on -- with no UI anywhere that lists or
        clears them, and no way to tell an old zip from the current one except by decoding
        its job hex.
        """
        try:
            zips = [os.path.join(EXPORT_DIR, f) for f in os.listdir(EXPORT_DIR)
                    if f.startswith("crack_export_") and f.endswith(".zip")]
        except OSError:
            return 0
        # The two rules are INDEPENDENT. The first version used
        # `if i < keep_newest and age < max_age_s: continue`, so the age rule overrode
        # keep_newest: come back the next morning and every zip is older than six hours, so
        # all of them are deleted including the newest -- the one the user was about to
        # download. keep_newest now wins unconditionally, and age only prunes the remainder.
        zips.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        now = time.time()
        removed = 0
        for i, z in enumerate(zips):
            if i < keep_newest:
                continue
            if (now - os.path.getmtime(z)) < max_age_s:
                continue
            try:
                os.remove(z)
                removed += 1
            except OSError:
                pass
        return removed

    @app.route("/api/export_all", methods=["POST"])
    def api_export_all():
        from app_endpoints import _new_job, _run_bg
        jid = _new_job("export_all")

        def work(report):
            names = [n for n in list_images() if crack_mask(n) is not None]
            os.makedirs(EXPORT_DIR, exist_ok=True)
            zpath = os.path.join(EXPORT_DIR, f"crack_export_{jid}.zip")
            summary = []
            with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
                for i, n in enumerate(names, 1):
                    report(stage="export", frac=i / max(len(names), 1),
                           note=f"{n} ({i}/{len(names)})")
                    m = crack_mask(n)
                    if m is None:
                        continue
                    b = io.BytesIO()
                    Image.fromarray(np.where(m, 0, 255).astype(np.uint8)).save(b, format="PNG")
                    z.writestr(f"masks/{n}_mask.png", b.getvalue())
                    try:
                        img8 = base_image(n)
                        if img8.shape == m.shape:
                            rgb = np.stack([img8] * 3, -1).astype(np.uint8)
                            rgb[m] = (225, 25, 25)
                            b = io.BytesIO()
                            Image.fromarray(rgb).save(b, format="PNG")
                            z.writestr(f"overlays/{n}_overlay.png", b.getvalue())
                    except Exception:
                        pass          # a missing source image must not kill the whole export
                    df = region_table(n, m)
                    z.writestr(f"regions/{n}_regions.csv", df.to_csv(index=False))
                    summary.append({
                        "SourceImage": n, "ImageH_px": int(m.shape[0]), "ImageW_px": int(m.shape[1]),
                        "CrackRegions": int(len(df)),
                        "CrackArea_px": int(m.sum()),
                        "CrackAreaPct": round(100.0 * m.sum() / m.size, 4),
                        "LargestRegion_px": int(df["Area_px"].max()) if len(df) else 0,
                        "MeanRegion_px": round(float(df["Area_px"].mean()), 1) if len(df) else 0.0,
                        "TotalLength_px": round(float(df["Length_px"].sum()), 1) if len(df) else 0.0,
                    })
                # Same for the cross-image summary: a zip in which nothing was
                # detected should still carry a readable summary.csv.
                sdf = (pd.DataFrame(summary) if summary else pd.DataFrame(
                    columns=["Image", "Regions", "CrackArea_px", "ImageArea_px",
                             "CrackAreaPct", "MeanRegion_px", "TotalLength_px"]))
                z.writestr("summary.csv", sdf.to_csv(index=False))
                z.writestr("README.txt",
                           "Crack detection export\n"
                           "======================\n"
                           "masks/    crack = BLACK on white\n"
                           "overlays/ detected crack burned into the source image in red\n"
                           "regions/  one row per connected crack region, per image\n"
                           "summary.csv  one row per image\n\n"
                           "All lengths and areas are in PIXELS. The scale bar lives in the\n"
                           "SEM databar, which this pipeline crops off before analysis, so no\n"
                           "micron factor is recoverable here -- multiply by the microscope's\n"
                           "own um/px value.\n\n"
                           "Human corrections are already included: they override the model\n"
                           "wherever they exist.\n")
            # Prune only AFTER the replacement zip is written and closed. Running it on the
            # way IN meant a failure anywhere in this function left the user with neither
            # the new export nor the old ones.
            _n_pruned = _prune_old_exports()

            return {"zip": os.path.basename(zpath), "images": len(names),
                    "regions": int(sum(s["CrackRegions"] for s in summary)),
                    "download": f"/api/export_all/{jid}/download"}

        _run_bg(jid, work)
        return jsonify({"ok": True, "job": jid})

    @app.route("/api/export_all/<job_id>/download")
    def api_export_all_download(job_id):
        p = os.path.join(EXPORT_DIR, f"crack_export_{job_id}.zip")
        if not os.path.exists(p):
            return jsonify({"ok": False, "error": "not ready or expired"}), 404
        return send_file(p, mimetype="application/zip", as_attachment=True,
                         download_name="crack_export.zip")

    return app
