"""
Shared helper for numbering candidates directly on a full-image overlay
(rather than only in the per-item review-sheet thumbnails), so a specific
region -- crack, artifact, or proposed interior candidate -- can be
referenced by its Label number while looking at the whole image.
"""
from PIL import ImageDraw, ImageFont

_FONT = None


def _font(size=13):
    global _FONT
    if _FONT is None:
        _FONT = ImageFont.load_default(size=size)
    return _FONT


def draw_labels(pil_img, items, color=(0, 255, 0), font_size=13):
    """items: iterable of (x, y, label). Small text with a black outline
    (readable over both dark crack pixels and light background), offset
    slightly from the point so it doesn't sit directly on top of it."""
    draw = ImageDraw.Draw(pil_img)
    font = _font(font_size)
    for x, y, label in items:
        draw.text((x + 3, y + 3), str(int(label)), fill=color, font=font,
                   stroke_width=2, stroke_fill=(0, 0, 0))
    return pil_img
