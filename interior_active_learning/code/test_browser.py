#!/usr/bin/env python3
"""Drive the real UI in a real browser and check WHICH FILE an edit lands in.

WHY THIS EXISTS. test_app.py drives the server over HTTP and passed 326 checks while a brush
stroke made straight after an upload was writing into a DIFFERENT image's correction mask. The
server was blameless -- it wrote to whatever name the client sent. The defect was in the served
JavaScript: loadImage drew an image without claiming it, and the only assignment to
currentImage was a change handler that does not fire when script sets sel.value, which is
exactly what the upload path did. Measured at the time: one stroke put 2,783 not-crack pixels
into an already hand-labelled frame that was not on screen, while the uploaded frame got none.

No HTTP-level test can see that, and neither can a DOM snapshot of the controls -- nothing
looks wrong until you perform the sequence. So this performs the sequence, and asserts on the
side effect (which mask file changed on disk) rather than on what the page says.

SAFETY. This never touches the repo's data. It points the server at empty scratch directories
via SEMCRACK_ORIGINAL_DIR and SEMCRACK_PAINT_DIR, and every image it works with is one it drew
itself, under the reserved apptest_ prefix. Even if the bug it guards against were fully
reintroduced, the only thing it could corrupt is a temp directory.

Run:  make test-browser        (or ./.venv/bin/python3 interior_active_learning/code/test_browser.py)
Needs: pip install playwright && playwright install chromium
"""

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image

CODE = os.path.dirname(os.path.abspath(__file__))
PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}" + (f"  -- {detail}" if detail else ""), flush=True)
    else:
        FAIL += 1
        print(f"  FAIL  {name}" + (f"  -- {detail}" if detail else ""), flush=True)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def synthetic_tif(path, w, h, seed):
    """A frame with real dark linear features, so the detector finds genuine candidates."""
    rng = np.random.default_rng(seed)
    img = rng.normal(150, 16, (h, w))
    yy, xx = np.mgrid[0:h, 0:w]
    for x0, y0, x1, y1, wid in [(int(w * .1), int(h * .2), int(w * .9), int(h * .6), 6),
                                (int(w * .2), int(h * .8), int(w * .8), int(h * .3), 5)]:
        t = np.clip(((xx - x0) * (x1 - x0) + (yy - y0) * (y1 - y0))
                    / max(1, ((x1 - x0) ** 2 + (y1 - y0) ** 2)), 0, 1)
        d = np.hypot(xx - (x0 + t * (x1 - x0)), yy - (y0 + t * (y1 - y0)))
        img -= 115 * np.exp(-(d / wid) ** 2)
    img = np.clip(img, 0, 255).astype(np.uint8)
    img[h - 40:, :] = 28                      # a databar, so find_field_of_view has work to do
    img[h - 30:h - 24, 20:160] = 235
    Image.fromarray(img).save(path)


def mask_counts(paint_dir, name):
    p = os.path.join(paint_dir, f"{name}_correction_mask.png")
    if not os.path.exists(p):
        return None
    a = np.array(Image.open(p))
    while a.ndim > 2:
        a = a[..., 0]
    return {v: int((a == v).sum()) for v in (1, 2, 3)}


def main():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("SKIP: playwright is not installed.\n"
              "      ./.venv/bin/python3 -m pip install playwright && "
              "./.venv/bin/python3 -m playwright install chromium")
        return 0

    tmp = tempfile.mkdtemp(prefix="sembrowser_")
    orig, paint = os.path.join(tmp, "original"), os.path.join(tmp, "paint")
    os.makedirs(orig); os.makedirs(paint)
    port = free_port()
    env = dict(os.environ, SEMCRACK_ORIGINAL_DIR=orig, SEMCRACK_PAINT_DIR=paint,
               PORT=str(port), OPEN="0")
    srv = subprocess.Popen([sys.executable, os.path.join(CODE, "paint_server.py")],
                           env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True)
    print(f"server on {port}, scratch dirs under {tmp}", flush=True)
    try:
        import urllib.request
        for _ in range(180):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2).read(1)
                break
            except Exception:
                if srv.poll() is not None:
                    print("server exited before serving:\n" + (srv.stdout.read() or ""))
                    return 1
                time.sleep(1)

        # Two frames of clearly different sizes, so canvas dimensions identify which is shown.
        # apptest_ is a reserved prefix the label ledger refuses, and A sorts before B so that
        # A is images[0] -- the stale target the original bug fell back to.
        a_name, b_name = "apptest_browser_a_first", "apptest_browser_b_second"
        synthetic_tif(os.path.join(tmp, "a.tif"), 900, 700, 3)
        synthetic_tif(os.path.join(tmp, "b.tif"), 1200, 800, 4)

        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"http://127.0.0.1:{port}/", wait_until="load")
            page.wait_for_timeout(1500)

            def drop(tif_path, upload_name):
                """Drag-and-drop the file, which is the path the bug lived in."""
                data = list(open(tif_path, "rb").read())
                page.evaluate(
                    """([bytes, fname]) => {
                        const f = new File([new Uint8Array(bytes)], fname,
                                           {type: 'image/tiff'});
                        const dt = new DataTransfer(); dt.items.add(f);
                        for (const t of ['dragenter', 'dragover', 'drop'])
                          document.dispatchEvent(new DragEvent(t, {dataTransfer: dt,
                                                 bubbles: true, cancelable: true}));
                    }""", [data, upload_name])

            def wait_for_open(name, timeout_ms=240000):
                """Wait for the DRAW, not just the state flip.

                currentImage is assigned by the dropdown's change handler before loadImage has
                drawn anything, so waiting on it alone races the render. canvasImage is set at
                the point the draw commits, so requiring both to equal `name` is the real
                "this image is now on screen and is what edits target" condition.
                """
                page.wait_for_function(
                    "n => typeof currentImage !== 'undefined' && currentImage === n"
                    " && typeof canvasImage !== 'undefined' && canvasImage === n",
                    arg=name, timeout=timeout_ms)

            drop(os.path.join(tmp, "a.tif"), a_name + ".tif")
            wait_for_open(a_name)
            check("the first uploaded image becomes the open image", True,
                  f"currentImage = {a_name}")

            # THE SEQUENCE THAT BROKE. Upload a second frame while the first is open. The app
            # renders the new one; the question is whether it also OWNS it.
            # Wait on what the USER can see -- the canvas taking the new frame's size -- not on
            # the state being right. Waiting on correctness would make a regression time out
            # after four minutes instead of failing in one line with the reason.
            drop(os.path.join(tmp, "b.tif"), b_name + ".tif")
            try:
                page.wait_for_function(
                    "() => document.getElementById('baseCanvas').width === 1200",
                    timeout=240000)
            except Exception as exc:
                check("the second upload renders at all", False, str(exc)[:160])

            state = page.evaluate("""() => {
                const c = document.getElementById('baseCanvas');
                return {current: currentImage,
                        canvas: c.width + 'x' + c.height,
                        header: document.getElementById('curName').textContent,
                        meta: document.getElementById('curMeta').textContent};
            }""")
            # Give the state a moment to catch up with the draw before judging it, so this
            # measures the app rather than the gap between two lines of JavaScript.
            try:
                page.wait_for_function(
                    "n => currentImage === n && canvasImage === n", arg=b_name, timeout=20000)
            except Exception:
                pass
            state = page.evaluate("""() => {
                const c = document.getElementById('baseCanvas');
                return {current: currentImage,
                        onCanvas: typeof canvasImage !== 'undefined' ? canvasImage : '(absent)',
                        canvas: c.width + 'x' + c.height,
                        header: document.getElementById('curName').textContent};
            }""")
            check("after a second upload, the app owns the image it is showing",
                  state["current"] == b_name and state["header"] == b_name
                  and state["canvas"].startswith("1200"),
                  f"currentImage={state['current']} canvasImage={state['onCanvas']} "
                  f"header={state['header']} canvas={state['canvas']}")

            before_a, before_b = mask_counts(paint, a_name), mask_counts(paint, b_name)

            # Paint, using the events the app actually binds (mousedown/mousemove/mouseup).
            page.evaluate("""() => {
                [...document.querySelectorAll('button')]
                  .find(b => b.textContent.trim() === 'Not crack').click();
                [...document.querySelectorAll('button')]
                  .find(b => b.textContent.trim() === 'Brush').click();
                const pc = document.getElementById('paintCanvas');
                const r = pc.getBoundingClientRect();
                const ev = (t, fx, fy) => pc.dispatchEvent(new MouseEvent(t, {
                    clientX: r.left + r.width * fx, clientY: r.top + r.height * fy,
                    bubbles: true, cancelable: true, buttons: 1, button: 0}));
                ev('mousedown', 0.25, 0.40);
                for (let i = 1; i <= 10; i++) ev('mousemove', 0.25 + i * 0.03, 0.40 + i * 0.01);
                window.dispatchEvent(new MouseEvent('mouseup', {bubbles: true}));
            }""")
            # marks autosave about a second after the stroke ends
            for _ in range(40):
                page.wait_for_timeout(500)
                if mask_counts(paint, b_name) != before_b:
                    break

            after_a, after_b = mask_counts(paint, a_name), mask_counts(paint, b_name)

            # THE CHECK THIS FILE EXISTS FOR.
            check("the stroke lands in the mask of the image on screen",
                  after_b is not None and after_b != before_b and after_b.get(2, 0) > 0,
                  f"{b_name}: {before_b} -> {after_b}")
            check("the stroke does NOT land in the previously-open image",
                  after_a == before_a,
                  f"{a_name}: {before_a} -> {after_a}"
                  + ("  <-- edits went to the wrong image" if after_a != before_a else ""))

            # A stroke while the canvas is stale must be REFUSED, not applied to either
            # image. Simulated by desynchronising the two variables the guard compares, which
            # is precisely the state an in-flight image switch produces.
            counts_before = (mask_counts(paint, a_name), mask_counts(paint, b_name))
            refused = page.evaluate("""() => {
                const real = currentImage;
                currentImage = 'apptest_browser_a_first';   // pretend a switch is in flight
                const pc = document.getElementById('paintCanvas');
                const r = pc.getBoundingClientRect();
                const ev = (t, fx, fy) => pc.dispatchEvent(new MouseEvent(t, {
                    clientX: r.left + r.width * fx, clientY: r.top + r.height * fy,
                    bubbles: true, cancelable: true, buttons: 1, button: 0}));
                ev('mousedown', 0.6, 0.6);
                for (let i = 1; i <= 6; i++) ev('mousemove', 0.6 + i * 0.02, 0.6);
                window.dispatchEvent(new MouseEvent('mouseup', {bubbles: true}));
                const msg = document.body.innerText;
                currentImage = real;
                return /wait for it to appear/i.test(msg);
            }""")
            page.wait_for_timeout(3000)
            check("an edit is refused while the canvas shows a different image",
                  refused and (mask_counts(paint, a_name), mask_counts(paint, b_name)) == counts_before,
                  "otherwise a stroke during an image switch writes old coordinates "
                  "into the new image's mask")

            # Undo is the other thing only a browser exercises.
            page.evaluate("""() => [...document.querySelectorAll('button')]
                                  .find(b => /Undo/.test(b.textContent)).click()""")
            page.wait_for_timeout(4000)
            undone = mask_counts(paint, b_name)
            # The button must revert the MASK, not merely repaint the canvas. It used to call
            # only the local canvas undo, so the stroke vanished on screen while the correction
            # stayed on disk -- the reviewer believed it was undone and the label survived.
            # "Reverted" means back to the pre-stroke state, and that state may legitimately
            # be "no mask file at all" (None) when the stroke was the only correction -- which
            # is the case here. A no-op undo still fails this: it would leave `undone` equal to
            # after_b, which is neither before_b nor a smaller count.
            check("the Undo button reverts the committed mask, not just the canvas",
                  undone == before_b or (undone or {}).get(2, 0) < after_b.get(2, 0),
                  f"not-crack px: {before_b} -> {after_b} -> {undone}")

            # THE GUARD MUST NOT BLOCK NORMAL WORK. editableNow() refuses edits while the
            # canvas and the edit target disagree; if those ever drift during ordinary use it
            # would silently make the app unpaintable. Switch back to the first image through
            # the sidebar -- the most common interaction there is -- and paint for real.
            page.evaluate("""n => {
                const row = [...document.querySelectorAll('.item')]
                    .find(e => e.textContent.includes(n));
                if (!row) throw new Error('no sidebar row for ' + n);
                row.click();
            }""", a_name)
            page.wait_for_function(
                "n => currentImage === n && canvasImage === n", arg=a_name, timeout=240000)
            a_before = mask_counts(paint, a_name)
            page.evaluate("""() => {
                [...document.querySelectorAll('button')]
                  .find(b => b.textContent.trim() === 'Add crack').click();
                const pc = document.getElementById('paintCanvas');
                const r = pc.getBoundingClientRect();
                const ev = (t, fx, fy) => pc.dispatchEvent(new MouseEvent(t, {
                    clientX: r.left + r.width * fx, clientY: r.top + r.height * fy,
                    bubbles: true, cancelable: true, buttons: 1, button: 0}));
                ev('mousedown', 0.35, 0.55);
                for (let i = 1; i <= 8; i++) ev('mousemove', 0.35 + i * 0.03, 0.55);
                window.dispatchEvent(new MouseEvent('mouseup', {bubbles: true}));
            }""")
            for _ in range(40):
                page.wait_for_timeout(500)
                if mask_counts(paint, a_name) != a_before:
                    break
            a_after = mask_counts(paint, a_name)
            check("painting still works after switching images through the sidebar",
                  a_after is not None and a_after != a_before and a_after.get(1, 0) > 0,
                  f"{a_name} crack px: {a_before} -> {a_after}"
                  + ("  <-- the stale-canvas guard is blocking normal work"
                     if a_after == a_before else ""))

            check("no uncaught JavaScript errors during the whole sequence",
                  not errors, "; ".join(errors[:3]))
            browser.close()
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=20)
        except subprocess.TimeoutExpired:
            srv.kill()
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{PASS} passed, {FAIL} failed, {PASS + FAIL} total")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
