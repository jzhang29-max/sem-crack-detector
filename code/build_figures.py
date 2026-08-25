"""
Build the README figures, TXM-style: side-by-side baked into one PNG with the
labels burned in, on the app's own dark chrome.

The right panel comes from the SAVED paint template rather than a fresh detection
run, and the left panel is rebuilt from the source TIFF through the pipeline's own
load_as_uint8 + find_field_of_view. That guarantees the two panels are pixel
aligned -- the template is written after the field-of-view crop, so loading the
raw TIFF without that crop would offset every overlay by the databar height.

The overlay's alpha blend is exactly invertible, which is how the red and cyan
masks are recovered for the stats footer:
    crack    = img*0.45 + [255,0,0]*0.55   ->  r-g == 140 and r-b == 140
    rejected = img*0.55 + [0,204,255]*0.45 ->  g-r == 92  and b-r == 115
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "interior_active_learning", "code"))
sys.path.insert(0, os.path.join(ROOT, "code"))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None
PAINT = os.path.join(ROOT, "interior_active_learning", "paint")

BG = (15, 17, 20)          # matches the app's --bg
FG = (232, 236, 241)
DIM = (150, 158, 170)
RED_TXT = (255, 105, 92)
CYAN_TXT = (90, 200, 245)
GAP = 12
LABEL_H = 26
FOOT_H = 24


# Linux font candidates as well as the macOS ones. Without them every label on a Linux box
# fell through to ImageFont.load_default(), which IGNORES the requested size: measured on this
# repo's own labels, "Quality-Control" renders 173 px wide at truetype 26 and 70 px with the
# bare default -- inside a LABEL_H of 26, with no error raised, so figures came out with tiny
# unreadable text and nothing said why. load_default(size=) honours it (180 px) in modern
# Pillow, so that is the last resort rather than the sized-blind call.
_MAC_FONTS = ("/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "/System/Library/Fonts/Supplemental/Arial.ttf",
              "/System/Library/Fonts/Helvetica.ttc")
_LINUX_BOLD = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf")
_LINUX_REG = ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf")


def font(size, bold=False):
    mac = (_MAC_FONTS[0] if bold else _MAC_FONTS[1], _MAC_FONTS[2])
    for p in mac + (_LINUX_BOLD if bold else _LINUX_REG):
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    try:
        return ImageFont.load_default(size=size)
    except (TypeError, OSError):   # no size arg (Pillow < 10.1), or no font to load
        return ImageFont.load_default()


def overlay_masks(tpl_rgb):
    a = tpl_rgb.astype(np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    red = (np.abs(r - g - 140) <= 6) & (np.abs(r - b - 140) <= 6)
    cyan = (np.abs(g - r - 92) <= 8) & (np.abs(b - r - 115) <= 8)
    return red, cyan


def load_aligned(name):
    """The raw uint8 image, cropped exactly as the pipeline crops it."""
    from common import ORIGINAL_DIR, contrast_kwargs_for
    from detect_cracks import find_field_of_view, load_as_uint8
    img8 = load_as_uint8(os.path.join(ORIGINAL_DIR, f"{name}.tif"),
                         **contrast_kwargs_for(name))
    x0, y0, x1, y1 = find_field_of_view(img8)
    return img8[y0:y1, x0:x1]


def densest(mask, box_h, box_w, stride=100):
    h, w = mask.shape
    box_h, box_w = min(box_h, h), min(box_w, w)
    I = np.pad(mask.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    best = (-1, 0, 0)
    for y in range(0, max(h - box_h, 0) + 1, stride):
        for x in range(0, max(w - box_w, 0) + 1, stride):
            c = int(I[y + box_h, x + box_w] - I[y, x + box_w]
                    - I[y + box_h, x] + I[y, x])
            if c > best[0]:
                best = (c, y, x)
    return best[1], best[1] + box_h, best[2], best[2] + box_w


def compose(panels, panel_w, footer):
    """panels: list of (ndarray RGB, [(text, colour), ...])"""
    ims = []
    for arr, parts in panels:
        im = Image.fromarray(arr)
        h = max(1, round(im.height * panel_w / im.width))
        ims.append((im.resize((panel_w, h), Image.LANCZOS), parts))
    ph = max(im.height for im, _ in ims)
    W = panel_w * len(ims) + GAP * (len(ims) - 1)
    out = Image.new("RGB", (W, LABEL_H + ph + FOOT_H), BG)
    d = ImageDraw.Draw(out)
    f = font(14, bold=True)
    ff = font(12)
    for i, (im, parts) in enumerate(ims):
        x = i * (panel_w + GAP)
        out.paste(im, (x, LABEL_H))
        cx = x + 1
        for text, col in parts:
            d.text((cx, 6), text, font=f, fill=col)
            cx += d.textlength(text, font=f)
    d.text((1, LABEL_H + ph + 6), footer, font=ff, fill=DIM)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--out", required=True)
    ap.add_argument("--panel", type=int, default=694)
    ap.add_argument("--crop", default="auto")
    ap.add_argument("--aspect", type=float, default=1.38)
    ap.add_argument("--frac", type=float, default=0.30, help="crop height as a fraction of the image")
    ap.add_argument("--note", default="")
    ap.add_argument("--fullframe", action="store_true")
    # The right panel is NOT always "model output": on an image with a correction
    # mask the pipeline applies the human's verdicts, so the overlay shows the
    # reviewed result. Labelling that "model output" would overstate the model.
    ap.add_argument("--reviewed", action="store_true",
                    help="label the right panel as human-reviewed, not raw model output")
    a = ap.parse_args()

    tpl = np.asarray(Image.open(os.path.join(PAINT, f"{a.name}_paint_template.png")
                                ).convert("RGB"))
    img8 = load_aligned(a.name)
    if img8.shape != tpl.shape[:2]:
        raise SystemExit(f"MISALIGNED: raw {img8.shape} vs template {tpl.shape[:2]}")
    red, cyan = overlay_masks(tpl)
    h, w = red.shape
    print(f"  {a.name}: {w}x{h} = {h*w/1e6:.1f} MP, "
          f"crack {100*red.mean():.2f}%, rejected {100*cyan.mean():.2f}%")

    if a.fullframe:
        y0, y1, x0, x1 = 0, h, 0, w
    elif a.crop == "auto":
        bh = int(h * a.frac)
        y0, y1, x0, x1 = densest(red, bh, int(bh * a.aspect))
    else:
        y0, y1, x0, x1 = (int(v) for v in a.crop.split(","))
    print(f"  crop {y0}:{y1}, {x0}:{x1}  ({x1-x0}x{y1-y0})")

    sl = (slice(y0, y1), slice(x0, x1))
    left = np.stack([img8[sl]] * 3, -1)
    right = tpl[sl]
    if a.reviewed:
        parts_r = [("after review", FG), ("   red = crack", RED_TXT),
                   ("   cyan = marked not-crack", CYAN_TXT)]
    else:
        parts_r = [("model output, image never seen in training", FG),
                   ("   red = crack", RED_TXT),
                   ("   cyan = rejected", CYAN_TXT)]
    footer = (f"{a.name}.tif   {w}x{h} ({h*w/1e6:.1f} MP)   "
              f"{100*red.mean():.1f}% of area marked crack" + (f"   {a.note}" if a.note else ""))
    fig = compose([(left, [("SEM image, as acquired", FG)]), (right, parts_r)],
                  a.panel, footer)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.save(a.out, optimize=True)
    print(f"  wrote {a.out}  {fig.size[0]}x{fig.size[1]}  {os.path.getsize(a.out)/1024:.0f} KB")


if __name__ == "__main__":
    main()
