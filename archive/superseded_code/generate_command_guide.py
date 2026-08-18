"""
A one-page visual command reference for the interior active-learning loop:
which script to run, in which order, to go from "just painted some
corrections" to "retrained model, fresh final results." Hand-authored SVG
(same toolkit as generate_scientific_diagram.py / generate_full_workflow_
diagram.py), rasterized to PNG via rsvg-convert.

Run:
    python3 generate_command_guide.py
Output:
    ../pipeline_diagram/command_guide.svg
    ../pipeline_diagram/command_guide.png
"""
import os
import subprocess
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "pipeline_diagram")
os.makedirs(OUT_DIR, exist_ok=True)

INK = "#1B242C"
SUBTEXT = "#5B6B76"
BG = "#F7F5F0"
CARD_BG = "#FFFFFF"
TERM_BG = "#20262B"
TERM_TEXT = "#D7E0E6"
TERM_COMMENT = "#7C8A93"

STEP_COLORS = ["#2A9D8F", "#5E9C6C", "#E9B44C", "#7A4C9E", "#E76F51"]


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class SVG:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.defs = []
        self.body = []

    def add_shadow_filter(self, fid, dx=0, dy=4, blur=8, opacity=0.14):
        self.defs.append(f'''
        <filter id="{fid}" x="-50%" y="-50%" width="200%" height="200%">
          <feDropShadow dx="{dx}" dy="{dy}" stdDeviation="{blur}" flood-color="{INK}" flood-opacity="{opacity}"/>
        </filter>''')

    def raw(self, s):
        self.body.append(s)

    def render(self):
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                f'viewBox="0 0 {self.w} {self.h}">'
                f'<defs>{"".join(self.defs)}</defs>{"".join(self.body)}</svg>')


def rounded_rect(x, y, w, h, r, fill, stroke=None, sw=0, filter_id=None, opacity=1.0):
    f = f'filter="url(#{filter_id})"' if filter_id else ""
    s = f'stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" ry="{r}" fill="{fill}" {s} {f} opacity="{opacity}"/>'


# icons authored in a -1..1 box, translated/scaled by caller
def icon_svg(kind, color):
    sw = 0.14
    if kind == "brush":
        return (f'<path d="M -0.55 0.6 L 0.15 -0.1" stroke="{color}" stroke-width="0.22" stroke-linecap="round"/>'
                f'<path d="M 0.1 -0.15 L 0.75 -0.8 L 0.9 -0.65 L 0.25 0" fill="{color}"/>'
                f'<circle cx="-0.58" cy="0.63" r="0.13" fill="{color}"/>')
    if kind == "model":
        parts = [f'<rect x="-0.32" y="-0.32" width="0.64" height="0.64" rx="0.08" fill="none" stroke="{color}" stroke-width="{sw}"/>']
        for dx, dy in [(-1, 0.5), (-1, -0.5), (1, 0.5), (1, -0.5), (0.5, 1), (-0.5, 1), (0.5, -1), (-0.5, -1)]:
            x0, y0 = dx * 0.32, dy * 0.32
            x1 = x0 + (0.18 if abs(dx) > abs(dy) else 0) * (1 if dx > 0 else -1)
            y1 = y0 + (0.18 if abs(dy) >= abs(dx) else 0) * (1 if dy > 0 else -1)
            parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="{color}" stroke-width="0.06"/>')
        return "".join(parts)
    if kind == "check":
        return (f'<circle cx="0" cy="0" r="0.82" fill="none" stroke="{color}" stroke-width="{sw}"/>'
                f'<polyline points="-0.36,0 -0.05,0.32 0.48,-0.34" fill="none" stroke="{color}" '
                f'stroke-width="0.19" stroke-linecap="round" stroke-linejoin="round"/>')
    if kind == "sheet":
        parts = [f'<rect x="-0.45" y="-0.55" width="0.9" height="1.1" rx="0.06" fill="none" stroke="{color}" stroke-width="{sw}"/>']
        for y in (-0.2, 0.05, 0.3):
            parts.append(f'<line x1="-0.28" y1="{y}" x2="0.28" y2="{y}" stroke="{color}" stroke-width="0.08"/>')
        return "".join(parts)
    if kind == "loop":
        return (f'<path d="M -0.6 -0.15 A 0.6 0.6 0 1 1 -0.6 0.2" fill="none" stroke="{color}" stroke-width="{sw}"/>'
                f'<polygon points="-0.6,0.2 -0.85,-0.05 -0.35,-0.05" fill="{color}"/>')
    return ""


def draw_terminal(svg, x, y, w, lines, top_pad=48, bottom_pad=22, line_h=25, font_size=15.5):
    h = top_pad + line_h * (len(lines) - 1) + bottom_pad
    svg.raw(rounded_rect(x, y, w, h, 10, TERM_BG))
    # traffic-light dots -- kept well clear of the first text line below
    # (confirmed visually: the original spacing let code text touch/overlap
    # the dots, e.g. the "p" in "python3" running into the green dot)
    for i, c in enumerate(["#EE6A5F", "#F5BD4F", "#61C454"]):
        svg.raw(f'<circle cx="{x+18+i*16}" cy="{y+18}" r="4.5" fill="{c}"/>')
    ty = y + top_pad
    for ln in lines:
        color = TERM_COMMENT if ln.strip().startswith("#") else TERM_TEXT
        svg.raw(f'<text x="{x+18}" y="{ty}" font-family="SF Mono, Menlo, monospace" '
                f'font-size="{font_size}" fill="{color}">{esc(ln)}</text>')
        ty += line_h
    return h


def wrap_text(svg, text, x, y, width_chars, line_h, font_size, color, weight="normal"):
    lines = textwrap.wrap(text, width=width_chars)
    for i, ln in enumerate(lines):
        svg.raw(f'<text x="{x}" y="{y + i*line_h}" font-family="Helvetica, Arial, sans-serif" '
                f'font-size="{font_size}" font-weight="{weight}" fill="{color}">{esc(ln)}</text>')
    return len(lines)


STEPS = [
    dict(
        icon="brush", title="Paint corrections",
        desc="Open the paint app, pick an image, and mark what the pipeline got wrong: "
             "red = crack, cyan = not-crack, magenta = erase entirely. Click Save & Ingest when done.",
        lines=[
            "python3 paint_server.py",
            "# then open http://127.0.0.1:8765 in a browser",
        ],
    ),
    dict(
        icon="model", title="Retrain the model",
        desc="Run this after painting a batch of new corrections, so the interior model "
             "actually learns from everything painted so far -- not just the picture, the model itself.",
        lines=[
            "python3 train_interior_model.py",
        ],
    ),
    dict(
        icon="check", title="Generate final results",
        desc="Applies the (now-updated) model to one image and saves the finished picture -- "
             "color overlay and black/white mask -- to both results/ and review/applied/.",
        lines=[
            "python3 apply_interior_model.py <image_name>",
            "",
            "# example:",
            "python3 apply_interior_model.py 260708_316_H_b2_front_CBS_002",
        ],
    ),
    dict(
        icon="sheet", title="(Optional) Batch review sheets",
        desc="For a bigger labeling push: export a printable contact-sheet batch of the "
             "hardest/most-uncertain candidates across all images, fill in TRUE/FALSE/SKIP, then ingest it.",
        lines=[
            "python3 active_learning_select.py",
            "# fill in the UserVerdict column on the exported sheet, then:",
            "python3 ingest_labels.py <round_number>",
        ],
    ),
    dict(
        icon="loop", title="Repeat",
        desc="Paint more corrections on the new results, retrain, regenerate. Each round the model's "
             "guesses get a little better, so there's less to fix next time.",
        lines=[
            "# back to step 1 -- open the paint app again",
            "python3 paint_server.py",
        ],
    ),
]

FILES = [
    ("paint/<image>_paint_template.png", "Read-only baseline you paint on top of -- regenerated fresh each round."),
    ("paint/<image>_painted.png", "Your actual paint layer, copied from the template and drawn on."),
    ("candidates/<image>_interior.csv", "Every candidate the model has ever proposed for this image, labeled or not."),
    ("models/interior_model.joblib", "The trained model -- rewritten each time you run train_interior_model.py."),
    ("results/ and review/applied/", "Final overlay + black/white mask, one pair per image."),
]

W = 980
HEADER_H = 130
STEP_GAP = 22
FOOTER_TITLE_H = 46
FOOTER_ROW_H = 34
CARD_PAD = 26
ICON_BOX = 56

step_heights = []
for s in STEPS:
    desc_lines = len(textwrap.wrap(s["desc"], width=78))
    term_h = 48 + 25 * (len(s["lines"]) - 1) + 22
    h = CARD_PAD + 34 + 8 + desc_lines * 21 + 16 + term_h + CARD_PAD
    step_heights.append(h)

H = HEADER_H + sum(step_heights) + STEP_GAP * (len(STEPS) - 1) + 40 + FOOTER_TITLE_H + FOOTER_ROW_H * len(FILES) + 50

svg = SVG(W, H)
svg.add_shadow_filter("cardShadow")
svg.raw(rounded_rect(0, 0, W, H, 0, BG))

# header
svg.raw(f'<text x="{W/2}" y="52" font-family="Helvetica, Arial, sans-serif" font-size="30" '
        f'font-weight="700" fill="{INK}" text-anchor="middle">Interior Active-Learning: Command Reference</text>')
svg.raw(f'<text x="{W/2}" y="82" font-family="Helvetica, Arial, sans-serif" font-size="15.5" '
        f'fill="{SUBTEXT}" text-anchor="middle">Run each script from interior_active_learning/code/ -- in this order</text>')

y = HEADER_H
card_x = 40
card_w = W - 80

for i, (s, h) in enumerate(zip(STEPS, step_heights)):
    color = STEP_COLORS[i % len(STEP_COLORS)]
    svg.raw(rounded_rect(card_x, y, card_w, h, 16, CARD_BG, filter_id="cardShadow"))
    svg.raw(rounded_rect(card_x, y, 8, h, 4, color))

    bx, by = card_x + CARD_PAD + 10, y + CARD_PAD + 8
    svg.raw(f'<circle cx="{bx+18}" cy="{by+18}" r="22" fill="{color}"/>')
    svg.raw(f'<text x="{bx+18}" y="{by+24}" font-family="Helvetica, Arial, sans-serif" font-size="18" '
            f'font-weight="700" fill="white" text-anchor="middle">{i+1}</text>')

    icon_cx, icon_cy = bx + 62, by + 18
    svg.raw(f'<g transform="translate({icon_cx},{icon_cy}) scale(18)">{icon_svg(s["icon"], color)}</g>')

    title_x = bx + 100
    svg.raw(f'<text x="{title_x}" y="{by+24}" font-family="Helvetica, Arial, sans-serif" font-size="19" '
            f'font-weight="700" fill="{INK}">{esc(s["title"])}</text>')

    desc_y = by + 52
    n_desc = wrap_text(svg, s["desc"], card_x + CARD_PAD, desc_y, 78, 21, 14.5, SUBTEXT)

    term_y = desc_y + n_desc * 21 - 5
    draw_terminal(svg, card_x + CARD_PAD, term_y, card_w - CARD_PAD * 2, s["lines"])

    y += h + STEP_GAP

y += 20
svg.raw(f'<text x="{card_x}" y="{y}" font-family="Helvetica, Arial, sans-serif" font-size="19" '
        f'font-weight="700" fill="{INK}">Where things live</text>')
y += 20
for path, note in FILES:
    svg.raw(rounded_rect(card_x, y, card_w, FOOTER_ROW_H - 6, 8, CARD_BG, filter_id="cardShadow"))
    svg.raw(f'<text x="{card_x+16}" y="{y+21}" font-family="SF Mono, Menlo, monospace" font-size="13.5" '
            f'font-weight="600" fill="{STEP_COLORS[2]}">{esc(path)}</text>')
    svg.raw(f'<text x="{card_x+380}" y="{y+21}" font-family="Helvetica, Arial, sans-serif" font-size="13" '
            f'fill="{SUBTEXT}">{esc(note)}</text>')
    y += FOOTER_ROW_H

svg_path = os.path.join(OUT_DIR, "command_guide.svg")
png_path = os.path.join(OUT_DIR, "command_guide.png")
with open(svg_path, "w") as f:
    f.write(svg.render())
subprocess.run(["rsvg-convert", "-o", png_path, "--width", str(W * 2), "--height", str(int(H) * 2), svg_path], check=True)
print(f"Wrote {svg_path}")
print(f"Wrote {png_path}")
