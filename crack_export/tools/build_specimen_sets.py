"""Rebuild the specimen folders and the PDFs from MACHINE-ONLY masks.

What changed and why. The first build used the app's exported mask, which is the model's
output with the human's paint layered over it -- the app applies corrections after scoring
and the README promises they always win. That is right for the tool and wrong here: on
MAR_Amb_AS_ETD_0003 the human painted 49.2% of the frame and the machine found 4.7%, so the
"result" you scrolled past was about ten parts brush to one part detector, round brush
scallops and all.

Every mask in this rebuild is the pipeline re-run with corrections suppressed. No human
stroke reaches the page. Where a human HAS reviewed a frame, that work is not discarded --
it is in the separate three-state PDF, which is where it belongs, labelled as human.

Writes: per-frame files, one PDF per specimen, and one combined PDF with an outline.
"""
import csv, os, sys, glob, json, shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None
# Paths derive from this file, and outputs go under a directory this repo owns. An earlier
# draft of this tool hardcoded /Users/jiamingzhang/... and /tmp, which is the same defect
# three repos here were repaired for on 2026-09-24: an absolute path is an undeclared
# dependency on one machine's layout, and /tmp is swept. Override with SEMCRACK_DERIVED.
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(_HERE))
DERIVED = os.environ.get("SEMCRACK_DERIVED", os.path.join(REPO, "crack_export", "derived"))
os.makedirs(DERIVED, exist_ok=True)
sys.path.insert(0, f"{REPO}/interior_active_learning/code")
sys.path.insert(0, f"{REPO}/code")
from common import contrast_kwargs_for, PAINT_DIR          # noqa: E402
from detect_cracks import load_as_uint8, find_field_of_view  # noqa: E402
from aggregate import specimen_key                          # noqa: E402

MACH = os.path.join(DERIVED, "machine_masks")
OUT = os.path.expanduser("~/Desktop/SEM_sets_original_and_BW")
LONG, PANEL_W, GUT, CAP = 3000, 1500, 28, 62


def font(sz):
    for p in ("/System/Library/Fonts/SFNSMono.ttf",
              "/System/Library/Fonts/Supplemental/Menlo.ttc"):
        if os.path.exists(p):
            try: return ImageFont.truetype(p, sz)
            except Exception: pass
    return ImageFont.load_default()


def reviewed_share(stem):
    """Fraction of the frame a human has adjudicated either way. 0.0 when never opened."""
    p = os.path.join(PAINT_DIR, f"{stem}_correction_mask.png")
    if not os.path.exists(p):
        return 0.0
    a = np.array(Image.open(p))
    if a.ndim > 2: a = a[..., 0]
    return float(np.isin(a, (1, 2)).mean())


def main():
    stems = sorted(os.path.basename(p).replace("_machine.png", "")
                   for p in glob.glob(f"{MACH}/*_machine.png"))
    if not stems:
        sys.exit("no machine-only masks")
    print(f"  {len(stems)} frames", flush=True)

    for d in glob.glob(OUT + "/*"):
        shutil.rmtree(d) if os.path.isdir(d) else os.remove(d)
    os.makedirs(OUT, exist_ok=True)

    f_big, f_sm = font(26), font(19)
    groups, per_set_pages, rows = {}, {}, []
    for i, stem in enumerate(stems, 1):
        key = specimen_key(stem) or "unparsed"
        d = os.path.join(OUT, key); os.makedirs(d, exist_ok=True)
        img8 = load_as_uint8(f"{REPO}/original/{stem}.tif", **contrast_kwargs_for(stem))
        mask = np.array(Image.open(f"{MACH}/{stem}_machine.png").convert("L"))
        if img8.shape != mask.shape:
            x0, y0, x1, y1 = find_field_of_view(img8)
            img8 = img8[y0:y1, x0:x1]
        if img8.shape != mask.shape:
            print(f"    SKIP {stem}: {img8.shape} vs {mask.shape}", flush=True); continue

        rev = reviewed_share(stem)
        pct = 100 * float((mask < 128).mean())
        rows.append(dict(frame=stem, specimen=key, machine_pct=pct, reviewed_pct=100 * rev))

        sc = min(1.0, LONG / max(img8.shape[1], img8.shape[0]))
        sz = (max(1, int(img8.shape[1] * sc)), max(1, int(img8.shape[0] * sc)))
        Image.fromarray(img8).resize(sz, Image.LANCZOS).save(
            os.path.join(d, f"{stem}__1_original.jpg"), quality=90, optimize=True)
        Image.fromarray(mask).resize(sz, Image.NEAREST).save(
            os.path.join(d, f"{stem}__2_bw.png"), optimize=True)

        h = int(PANEL_W * img8.shape[0] / img8.shape[1])
        sheet = Image.new("RGB", (PANEL_W * 2 + GUT, h + CAP), (255, 255, 255))
        sheet.paste(Image.fromarray(img8).convert("RGB").resize((PANEL_W, h), Image.LANCZOS), (0, CAP))
        sheet.paste(Image.fromarray(mask).convert("RGB").resize((PANEL_W, h), Image.NEAREST), (PANEL_W + GUT, CAP))
        dr = ImageDraw.Draw(sheet)
        dr.text((4, 8), stem, fill=(0, 0, 0), font=f_big)
        dr.text((4, 38), "original", fill=(90, 90, 90), font=f_sm)
        note = f"MACHINE crack mask, no human paint  —  {pct:.1f}% of frame"
        note += (f"   ·  a human has adjudicated {100*rev:.1f}% of this frame" if rev > 0
                 else "   ·  no human has reviewed this frame")
        dr.text((PANEL_W + GUT + 4, 38), note, fill=(0, 90, 160), font=f_sm)
        per_set_pages.setdefault(key, []).append((stem, sheet))
        groups.setdefault(key, []).append(stem)
        if i % 20 == 0: print(f"  {i}/{len(stems)}", flush=True)

    for key, pages in sorted(per_set_pages.items()):
        ims = [p for _, p in pages]
        ims[0].save(os.path.join(OUT, f"{key}.pdf"), "PDF", resolution=150,
                    save_all=True, append_images=ims[1:])

    # combined, with an outline
    allp, marks = [], []
    cover = Image.new("RGB", (PANEL_W * 2 + GUT, 820), (255, 255, 255))
    dc = ImageDraw.Draw(cover)
    for j, (t, s, c) in enumerate([
            ("SEM frames by specimen", 62, (0, 0, 0)),
            (f"{len(rows)} frames in {len(per_set_pages)} specimens", 30, (40, 40, 40)),
            ("left: original.   right: the DETECTOR's own mask, black = crack.", 26, (40, 40, 40)),
            ("No human paint is in these masks. The pipeline was re-run with every", 24, (90, 90, 90)),
            ("correction suppressed, so nothing here is a brush stroke.", 24, (90, 90, 90)),
            ("Human review lives in THREE_STATE_machine_vs_human.pdf, labelled as human.", 24, (0, 90, 160))]):
        dc.text((110, 120 + j * 62), t, fill=c, font=font(s))
    allp.append(cover)
    for key, pages in sorted(per_set_pages.items()):
        div = Image.new("RGB", (PANEL_W * 2 + GUT, 520), (255, 255, 255))
        dd = ImageDraw.Draw(div)
        dd.text((110, 140), key, fill=(0, 0, 0), font=font(56))
        dd.text((110, 250), f"{len(pages)} frames", fill=(40, 40, 40), font=font(30))
        marks.append((key, len(allp), []))
        allp.append(div)
        for stem, im in pages:
            marks[-1][2].append((stem, len(allp)))
            allp.append(im)
    # Written without an outline here: pypdf is not in this app's venv and adding an
    # unpinned dependency to the venv that serves the app, to get bookmarks in a PDF, is a
    # bad trade. add_outline.py finishes the job under the sibling interpreter that has it.
    comb = os.path.join(OUT, "ALL_SETS_original_and_BW.pdf")
    allp[0].save(comb, "PDF", resolution=150, save_all=True, append_images=allp[1:])
    json.dump([{"title": k, "page": i,
                "kids": [{"title": s2, "page": p2} for s2, p2 in kids]}
               for k, i, kids in marks], open(os.path.join(DERIVED, "outline_marks.json"), "w"))

    json.dump(rows, open(os.path.join(DERIVED, "specimen_set_rows.json"), "w"))
    m = np.array([r["machine_pct"] for r in rows])
    rv = np.array([r["reviewed_pct"] for r in rows])
    print(f"\n  rebuilt {len(rows)} frames in {len(per_set_pages)} specimens")
    print(f"  machine crack: median {np.median(m):.2f}% of frame (range {m.min():.2f}-{m.max():.2f})")
    print(f"  frames with ANY human adjudication: {(rv>0).sum()}/{len(rv)}")
    print(f"  combined: {comb} ({len(allp)} pages, {os.path.getsize(comb)/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
