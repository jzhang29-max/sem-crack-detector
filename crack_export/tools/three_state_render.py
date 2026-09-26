"""Four panels per frame: the image, what the machine found, what a human adjudicated,
and what the shipped export throws away.

The current export is `np.where(mask, 0, 255)` -- binary. Crack is black and EVERYTHING
else is white, so "a human marked this as background" and "nobody ever looked here" become
the same pixel value. This repo's own audit (docs/UNLABELLED_PIXEL_AUDIT.md) is a six-tool
indictment of exactly that, and the app is a seventh instance of it.

Panels:
  1  original, cropped to the detected field of view
  2  MACHINE ONLY -- the pipeline with every human correction suppressed. No brush.
  3  HUMAN, three-state -- black crack, white adjudicated not-crack, GREY never reviewed
  4  what you get today -- the binary export, with the grey collapsed into white

Panel 4 exists to be compared with panel 3. The grey area is the size of the claim the
binary file makes on the reader's behalf.
"""
import os, sys, glob, json
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
from common import PAINT_DIR, contrast_kwargs_for          # noqa: E402
from detect_cracks import load_as_uint8, find_field_of_view  # noqa: E402

MACHINE = os.path.join(DERIVED, "machine_masks")
OUT = os.path.expanduser("~/Desktop/SEM_three_state")
PANEL = 1100
GUT, CAP = 22, 74
GREY = 150          # never reviewed


def font(sz):
    for p in ("/System/Library/Fonts/SFNSMono.ttf",
              "/System/Library/Fonts/Supplemental/Menlo.ttc"):
        if os.path.exists(p):
            try: return ImageFont.truetype(p, sz)
            except Exception: pass
    return ImageFont.load_default()


def main():
    os.makedirs(OUT, exist_ok=True)
    f_big, f_sm = font(24), font(17)
    names = sorted(os.path.basename(p).replace("_machine.png", "")
                   for p in glob.glob(f"{MACHINE}/*_machine.png"))
    if not names:
        sys.exit("no machine-only masks yet")
    rows, pages = [], []
    for i, n in enumerate(names, 1):
        cm = np.array(Image.open(os.path.join(PAINT_DIR, f"{n}_correction_mask.png")))
        if cm.ndim > 2: cm = cm[..., 0]
        mach = np.array(Image.open(f"{MACHINE}/{n}_machine.png").convert("L")) < 128
        if mach.shape != cm.shape:
            continue
        img8 = load_as_uint8(f"{REPO}/original/{n}.tif", **contrast_kwargs_for(n))
        if img8.shape != cm.shape:
            x0, y0, x1, y1 = find_field_of_view(img8)
            img8 = img8[y0:y1, x0:x1]
            if img8.shape != cm.shape:
                continue

        crack, notc, unrev = cm == 1, cm == 2, cm == 0
        three = np.full(cm.shape, GREY, np.uint8)
        three[crack] = 0; three[notc] = 255
        binary = np.where(crack, 0, 255).astype(np.uint8)      # what ships today
        machine_img = np.where(mach, 0, 255).astype(np.uint8)

        rows.append(dict(frame=n, pct_unreviewed=100*float(unrev.mean()),
                         pct_crack=100*float(crack.mean()), pct_notcrack=100*float(notc.mean()),
                         pct_machine=100*float(mach.mean()),
                         machine_in_reviewed=float(mach[~unrev].mean()) if (~unrev).any() else None,
                         machine_in_unreviewed=float(mach[unrev].mean()) if unrev.any() else None))

        h = int(PANEL * cm.shape[0] / cm.shape[1])
        panels = [Image.fromarray(img8).convert("RGB").resize((PANEL, h), Image.LANCZOS),
                  Image.fromarray(machine_img).convert("RGB").resize((PANEL, h), Image.NEAREST),
                  Image.fromarray(three).convert("RGB").resize((PANEL, h), Image.NEAREST),
                  Image.fromarray(binary).convert("RGB").resize((PANEL, h), Image.NEAREST)]
        W = PANEL * 4 + GUT * 3
        sheet = Image.new("RGB", (W, h + CAP), (255, 255, 255))
        for k, p in enumerate(panels):
            sheet.paste(p, (k * (PANEL + GUT), CAP))
        d = ImageDraw.Draw(sheet)
        d.text((4, 6), n, fill=(0, 0, 0), font=f_big)
        labs = [("original", (90, 90, 90)),
                (f"MACHINE only, no human paint   {100*mach.mean():.1f}% of frame", (0, 90, 160)),
                (f"HUMAN 3-state  black=crack {100*crack.mean():.1f}%   "
                 f"white=not-crack {100*notc.mean():.2f}%   GREY=never reviewed {100*unrev.mean():.1f}%", (0, 120, 0)),
                ("what the app exports today: grey collapsed into white", (170, 0, 0))]
        for k, (t, c) in enumerate(labs):
            d.text((k * (PANEL + GUT) + 4, 40), t, fill=c, font=f_sm)
        pages.append(sheet)
        print(f"  [{i}/{len(names)}] {n}", flush=True)

    json.dump(rows, open(os.path.join(DERIVED, "three_state_rows.json"), "w"))
    if pages:
        out = os.path.join(OUT, "THREE_STATE_machine_vs_human.pdf")
        pages[0].save(out, "PDF", resolution=150, save_all=True, append_images=pages[1:])
        print(f"\n  {out}  ({len(pages)} pages, {os.path.getsize(out)/1e6:.0f} MB)")
    u = np.array([r["pct_unreviewed"] for r in rows])
    print(f"  never-reviewed share of a frame: median {np.median(u):.1f}%  min {u.min():.1f}%  max {u.max():.1f}%")


if __name__ == "__main__":
    main()
