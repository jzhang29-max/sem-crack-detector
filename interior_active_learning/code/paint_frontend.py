INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SEM Crack Detection</title>
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32' role='img' aria-label='SEM Crack Detection'%3E %3Cdefs%3E %3ClinearGradient id='field' x1='0' y1='0' x2='1' y2='1'%3E %3Cstop offset='0' stop-color='%239aa1aa'/%3E %3Cstop offset='1' stop-color='%234e545d'/%3E %3C/linearGradient%3E %3C/defs%3E %3Crect x='1' y='1' width='30' height='30' rx='7' fill='url(%23field)'/%3E %3Crect x='1' y='1' width='30' height='30' rx='7' fill='none' stroke='%2320242b' stroke-width='1.5'/%3E %3Cpath d='M5 8 L11 13 L9 17 L16 20 L14 24 L27 27' fill='none' stroke='%23ff2222' stroke-width='4.2' stroke-linecap='round' stroke-linejoin='round'/%3E %3Cpath d='M5 8 L11 13 L9 17 L16 20 L14 24 L27 27' fill='none' stroke='%23ff7a7a' stroke-width='1.4' stroke-linecap='round' stroke-linejoin='round' opacity='0.65'/%3E %3Ccircle cx='23' cy='10' r='3.1' fill='%2300ccff' opacity='0.92'/%3E %3C/svg%3E">
<style>
/* Ported from the sibling TXM app's current frontend (app/static/index.html) so
   the two tools are one family. Its three stated design rules are followed here
   too, and they happen to be the right ones for this app as well:
     1. Show the FOUR things the workflow needs (load, look, fix, retrain).
        Everything else is a tuning knob and lives behind Advanced.
     2. One primary action visible at a time. Retrain is the only filled button.
     3. Group by task with real whitespace, not by cramming into one flex row.
   Difference forced by this app's data model: "what you mark" (crack /
   not-crack / erase) and "how much you take" (brush vs whole region) are
   independent here, where TXM folds them into one tool list. They are two
   segmented controls rather than one, because collapsing them would misreport
   what a click is about to do. */
:root{
  --bg:#0f1114; --surface:#16191e; --surface2:#1c2027; --line:#252a32;
  --line2:#333a45; --ink:#eceef1; --ink2:#a8aeb8; --ink3:#6f7681;
  --brand:#4f8ef7; --ok:#2ea36b; --warn:#c9822a; --bad:#d64545;
  --r:8px; --shadow:0 1px 2px rgba(0,0,0,.35), 0 8px 24px rgba(0,0,0,.22);
}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--ink);display:flex;
  font:14px/1.55 ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  -webkit-font-smoothing:antialiased;overflow:hidden}
button{font:inherit;color:var(--ink);background:var(--surface2);cursor:pointer;
  border:1px solid var(--line2);border-radius:var(--r);padding:8px 13px;
  transition:background .12s,border-color .12s}
button:hover:not(:disabled){background:#232833;border-color:#3d4552}
button:active:not(:disabled){transform:translateY(1px)}
button:disabled{opacity:.4;cursor:default}
.primary{background:var(--brand);border-color:var(--brand);color:#fff;font-weight:600}
.primary:hover:not(:disabled){background:#5f9bfa;border-color:#5f9bfa}
.ghost{background:transparent;border-color:transparent;color:var(--ink2);padding:6px 9px}
.ghost:hover:not(:disabled){background:var(--surface2);color:var(--ink)}
.danger:hover:not(:disabled){border-color:var(--bad);color:#ff8080}

/* ---------------- sidebar ---------------- */
#side{width:296px;min-width:296px;background:var(--surface);
  border-right:1px solid var(--line);display:flex;flex-direction:column;overflow:hidden}
.brand{padding:16px 18px 14px;border-bottom:1px solid var(--line)}
.brand h1{margin:0;font-size:14px;font-weight:600;letter-spacing:-.01em}
.brand p{margin:3px 0 0;font-size:11.5px;color:var(--ink3)}
#dropTarget{margin:14px;padding:20px 14px;border:1.5px dashed var(--line2);border-radius:10px;
  text-align:center;color:var(--ink2);cursor:pointer;transition:.14s}
/* Once a corpus is loaded the 108px dashed box is redundant -- the full-window #dropZone
   already accepts drags -- so it collapses to one row and gives the list its height. */
#dropTarget.slim{margin:8px 14px;padding:6px 10px;border-radius:7px}
#dropTarget.slim span{display:none}
#dropTarget.slim b{font-size:11.5px;font-weight:400;color:var(--ink2)}
#dropTarget:hover{border-color:#44506b;background:#171c26}
#dropTarget.hot{border-color:var(--brand);background:#17233a;color:var(--ink)}
#dropTarget b{display:block;font-size:13px;font-weight:500;color:var(--ink)}
#dropTarget span{font-size:11px;color:var(--ink3)}
.lbl{padding:12px 18px 6px;font-size:10.5px;font-weight:600;letter-spacing:.07em;
  text-transform:uppercase;color:var(--ink3);display:flex;gap:8px;align-items:baseline}
#imgSearch{margin:0 14px 6px;padding:6px 9px;font:inherit;font-size:12px;color:var(--ink);
  background:var(--surface2);border:1px solid var(--line2);border-radius:6px}
#imgSearch::placeholder{color:var(--ink3)}
#imageList{flex:1;overflow-y:auto;padding-bottom:8px}
/* A 65-image worklist read through a 6-row porthole is not navigable. Fixed chrome
   (brand + drop target + label + filter + model card) took ~440px of a 296px column,
   leaving ~340px of list at ~53px per row. Rows are ~34px now and the chrome collapses,
   which puts roughly 17 on screen instead of 6. */
.item{display:flex;gap:8px;align-items:center;padding:5px 14px;cursor:pointer;
  border-left:2px solid transparent}
.item:hover{background:#1a1e25}
.item.sel{background:#1a2333;border-left-color:var(--brand)}
.item .th{width:26px;height:22px;flex:0 0 26px;border-radius:3px;object-fit:contain;
  background:#0b0d10;border:1px solid var(--line)}
.item .tx{min-width:0;flex:1}
.item .nm{display:block;font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
/* Shown on hover and on the selected row only. A second line on every row was the
   single biggest cost in list height, and the candidate counts are reference detail
   rather than something you scan down the list. */
.item .sub{display:none;font-size:10.5px;color:var(--ink3);margin-top:1px}
.item:hover .sub,.item.sel .sub{display:block}
.item .dot{width:6px;height:6px;border-radius:50%;background:var(--ok);flex:0 0 6px}
.item .dot.busy{background:var(--warn)}
.item .x{flex:0 0 18px;width:18px;height:18px;border:0;padding:0;border-radius:4px;
  background:transparent;color:var(--ink3);font-size:15px;line-height:1;opacity:0;
  transition:opacity .12s,background .12s,color .12s}
.item:hover .x,.item.sel .x{opacity:1}
.item .x:hover{background:#3a2326;color:#ff8080}
#modelcard{border-top:1px solid var(--line);padding:10px 14px;background:#13161b}
/* Collapsed by default: six rows of model metadata pinned to the bottom were competing
   with the worklist, and two of the values wrap to two lines at this width. */
#modelcard.collapsed .rows{display:none}
#modelcard .sum{font-size:11px;color:var(--ink2);cursor:pointer;display:flex;
  justify-content:space-between;gap:8px;align-items:baseline}
#modelcard .sum .caret{color:var(--ink3);font-size:10px}
#modelcard .k{font-size:10.5px;text-transform:uppercase;letter-spacing:.07em;color:var(--ink3)}
#modelcard .v{font-size:12px;margin-top:3px}
#modelcard .v b{font-weight:600}
/* The card's rows are built as <span>label</span><b>value</b>, with no layout rule
   until now -- so they rendered butted together as "TypeLogisticRegression".
   Note the two ids differ only in case: #modelcard is the section, #modelCard the
   value block inside it. Both exist; HTML ids are case-sensitive. */
#modelcard .v .row{display:flex;justify-content:space-between;gap:12px;line-height:1.6}
#modelcard .v .row span{color:var(--ink3)}
#modelcard .v .row b{text-align:right}
#mpick{width:100%;margin:6px 0 7px;padding:6px 8px;font:inherit;font-size:12px;
  color:var(--ink);background:var(--surface2);border:1px solid var(--line2);
  border-radius:6px;cursor:pointer}
#mpick:hover{border-color:#3d4552}

/* ---------------- main ---------------- */
#main{flex:1;display:flex;flex-direction:column;min-width:0}
#top{display:flex;align-items:center;gap:14px;padding:11px 18px;background:var(--surface);
  border-bottom:1px solid var(--line);flex-wrap:wrap}
.seg{display:inline-flex;background:var(--surface2);border:1px solid var(--line2);
  border-radius:var(--r);overflow:hidden;flex:0 0 auto}
.seg button{border:0;border-radius:0;background:transparent;padding:7px 13px;font-size:13px;
  color:var(--ink2);white-space:nowrap}
.seg button+button{border-left:1px solid var(--line2)}
.seg button.on,.seg button.selected{background:var(--brand);color:#fff;font-weight:500}
.seg button:hover:not(.on):not(.selected){background:#232833;color:var(--ink)}
.sp{flex:1}
.menu{position:relative}
.menu .pop{position:absolute;top:calc(100% + 7px);right:0;z-index:40;min-width:250px;
  background:var(--surface2);border:1px solid var(--line2);border-radius:10px;
  box-shadow:var(--shadow);padding:6px;display:none}
.menu.open .pop{display:block}
.pop button{display:block;width:100%;text-align:left;background:transparent;border:0;
  border-radius:6px;padding:8px 10px;font-size:13px}
.pop button:hover{background:#282e39}
.pop .hint{font-size:11px;color:var(--ink3);padding:6px 10px 8px;line-height:1.45}
.pop hr{border:0;border-top:1px solid var(--line);margin:5px 2px}
#adv{padding:0 18px;background:#12151a;border-bottom:1px solid var(--line);
  max-height:0;overflow:hidden;transition:max-height .18s ease,padding .18s ease}
#adv.open{max-height:60vh;overflow-y:auto;padding:13px 18px}
.advrow{display:flex;gap:26px;align-items:center;flex-wrap:wrap}
.fld{display:flex;gap:9px;align-items:center;font-size:12px;color:var(--ink2)}
.fld input[type=range]{width:118px;accent-color:var(--brand)}
.fld .num{font-variant-numeric:tabular-nums;color:var(--ink);min-width:38px}
.note{font-size:11px;color:var(--ink3);margin-top:9px;line-height:1.5;max-width:760px}
#prog{height:2px;background:var(--brand);width:0;transition:width .25s}
#canvasWrap{flex:1;overflow:auto;background:#0a0c0e;position:relative;min-height:0}
#canvasInner{position:relative;margin:0 auto}
#canvasWrap canvas{position:absolute;top:0;left:0;display:block;image-rendering:pixelated}
#paintCanvas{cursor:crosshair}
#foot{display:flex;gap:10px;align-items:center;padding:9px 18px;font-size:12px;
  color:var(--ink2);background:var(--surface);border-top:1px solid var(--line);min-height:36px}
#status{flex:1}
#status.error{color:#ff8080}
.tag{font-size:10.5px;padding:2px 7px;border-radius:20px;border:1px solid var(--line2);
  color:var(--ink3);white-space:nowrap}
.kbd{font:11px ui-monospace,monospace;background:#20242c;border:1px solid var(--line2);
  border-radius:4px;padding:1px 5px;color:var(--ink2)}
#dropZone{position:fixed;inset:0;z-index:900;display:none;background:rgba(10,12,14,.88);
  align-items:center;justify-content:center}
#dropZone.active{display:flex}
#dropInner{border:2px dashed var(--brand);border-radius:12px;padding:44px 62px;
  text-align:center;font-size:18px;background:#17233a}
#dropInner .hint{margin-top:8px;font-size:12px;color:var(--ink2)}
#jobBar{display:none}
#help{position:fixed;right:18px;bottom:48px;z-index:940;display:none;background:var(--surface2);
  border:1px solid var(--line2);border-radius:10px;padding:12px 14px;font-size:12px;
  color:var(--ink2);box-shadow:var(--shadow)}
#help.open{display:block}
#help td{padding:3px 8px 3px 0}
</style>
</head>
<body>

<aside id="side">
  <div class="brand">
    <h1>SEM Crack Detection</h1>
    <p>Load &middot; review &middot; correct &middot; retrain</p>
  </div>

  <div id="dropTarget">
    <b>Drop SEM images</b>
    <span>or click to browse &mdash; .tif .tiff .png .jpg</span>
  </div>

  <div class="lbl">Images <span id="ncount" style="color:var(--ink3)"></span></div>
  <input id="imgSearch" type="search" placeholder="Filter&hellip;" autocomplete="off">
  <div id="imageList"></div>
  <select id="imageSelect" style="display:none"></select>

  <div id="modelcard">
    <div class="sum" title="Show or hide the model details"><span id="modelSummary">model&hellip;</span><span class="caret">&#9656;</span></div>
    <div class="rows">
      <div class="k">Model</div>
      <select id="mpick" title="Switch models. Roll back by picking an earlier one."></select>
      <div class="v" id="modelCard">loading&hellip;</div>
    </div>
  </div>
</aside>

<main id="main">
  <div id="top">
    <div class="seg" id="tools">
      <button id="swatchRed" class="selected" title="Drag to mark crack the model missed">Add crack</button>
      <button id="swatchCyan" title="Drag to mark something the model wrongly called crack">Not crack</button>
      <button id="swatchErase" title="Drag to remove pixels from consideration entirely">Erase</button>
    </div>

    <div class="seg" id="scope">
      <button id="brushModeBtn" class="on" title="Take only the pixels your brush covers">Brush</button>
      <button id="bucketBtn" title="One click takes a whole connected region -- needed for large ones">Whole region</button>
    </div>

    <label class="fld" style="gap:7px"><input type="checkbox" id="showResult" checked> Show result</label>

    <div class="sp"></div>

    <button class="ghost" id="advToggle" title="Brush, zoom, re-apply, undo, reset">Advanced &#9662;</button>

    <div class="menu" id="expMenu">
      <button class="ghost" id="exportBtn">Export &#9662;</button>
      <div class="pop">
        <button id="dlMask">Black &amp; white mask</button>
        <button id="dlOverlay">Overlay image</button>
        <button id="dlCsv">Measurements (CSV)</button>
        <hr>
        <button id="dlAll"><b>Everything, all images (.zip)</b></button>
        <div class="hint">Exports what you see now, with your corrections applied. Pixel units.</div>
      </div>
    </div>

    <button class="primary" id="retrainBtn" title="Learn from every correction you have made. The new model is deployed only if it scores at least as well on held-out data.">Retrain</button>
  </div>

  <div id="adv">
    <div class="advrow">
      <div class="fld">Brush <input type="range" id="brushSize" min="2" max="120" value="18"><span class="num" id="brushSizeLabel">18px</span></div>
      <div class="fld">Zoom <input type="range" id="zoom" min="10" max="800" value="100"><span class="num" id="zoomLabel">100%</span>
        <button class="ghost" id="fitBtn">Fit</button></div>
      <button class="ghost" id="installSamBtn" style="display:none" title="Install PyTorch + transformers into this app's virtualenv, no terminal needed">Enable SAM (+6% accuracy)</button>
      <button class="ghost" id="setScaleBtn" title="Click the two ends of the burned-in scale bar, then type its label. Exported lengths become micrometres instead of pixels.">Set scale&hellip;</button>
      <span class="num" id="scaleState" style="margin-left:6px">uncalibrated</span>
      <div class="sp"></div>
      <button class="ghost" id="reapplyBtn" title="Re-render every image with the current model">Re-apply model</button>
      <button class="ghost" id="undoBtn">Undo <span class="kbd">&#8984;Z</span></button>
      <button class="ghost danger" id="clearBtn" title="Discard unsaved strokes on this image">Reset image</button>
      <button class="ghost" id="helpBtn">?</button>
    </div>
    <div class="note" id="advnote">SAM is switched off: this build runs the archived model on its own. Marks save themselves about a second after you stop drawing &mdash; there is no Save button. <span class="kbd">&#8984;Z</span> undoes strokes, and once those run out it undoes the last saved correction.</div>
  </div>

  <div id="prog"></div>

  <div id="canvasWrap">
    <div id="canvasInner">
      <canvas id="baseCanvas"></canvas>
      <canvas id="paintCanvas"></canvas>
    </div>
  </div>

  <div id="foot">
    <span id="status">Drop some images in to begin.</span>
    <span class="tag" id="curName">no image</span>
    <span class="tag" id="curMeta"></span>
    <span class="tag" id="saveState"></span>
    <button class="ghost" id="retryBtn" style="display:none">Retry save</button>
  </div>
</main>

<div id="help">
  <table>
    <tr><td><span class="kbd">&#8984;Z</span></td><td>Undo stroke, then last saved correction</td></tr>
    <tr><td><span class="kbd">1</span> <span class="kbd">2</span> <span class="kbd">3</span></td><td>Add crack / Not crack / Erase</td></tr>
    <tr><td><span class="kbd">F</span></td><td>Fit to window</td></tr>
    <tr><td colspan="2" style="padding-top:7px">Red = crack &middot; Cyan = not a crack.<br>
      Your corrections always override the model.</td></tr>
  </table>
</div>

<div id="dropZone"><div id="dropInner"><strong>Drop SEM images</strong>
  <div class="hint">the current model is applied automatically</div></div></div>

<div id="jobBar"><div id="jobLabel"></div><div id="jobTrack"><div id="jobFill"></div></div>
  <div id="jobNote"></div></div>

<script>
const RED = '#ff0000', CYAN = '#00ccff', ERASE = '#ff00ff';
const SWATCH_IDS = ['swatchRed', 'swatchCyan', 'swatchErase'];
let currentColor = RED;
let brushSize = 18;
let zoom = 1.0;
let nativeW = 0, nativeH = 0;
let drawing = false;
let lastX = 0, lastY = 0;
let undoStack = [];
let currentImage = null;
let tool = 'paint'; // 'paint' (brush) | 'bucket' (click-to-flip a whole region)

const baseCanvas = document.getElementById('baseCanvas');
const paintCanvas = document.getElementById('paintCanvas');
const baseCtx = baseCanvas.getContext('2d');
const paintCtx = paintCanvas.getContext('2d', { willReadFrequently: true });
const canvasInner = document.getElementById('canvasInner');
const canvasWrap = document.getElementById('canvasWrap');
const statusEl = document.getElementById('status');

function setStatus(msg, isError) {
  statusEl.textContent = msg;
  statusEl.className = isError ? 'error' : '';
}

async function loadImageList(keepCurrent) {
  // keepCurrent=true just refreshes the dropdown's candidate counts (e.g.
  // after an ingest) without navigating away from -- and re-fetching --
  // whatever image is currently open. Only the initial page load and an
  // explicit image switch should actually load a (possibly different) image.
  const res = await fetch('/api/images');
  const images = await res.json();
  const sel = document.getElementById('imageSelect');
  const previousSelection = keepCurrent ? currentImage : null;
  sel.innerHTML = '';
  needsDetect = new Set(images.filter(i => i.has_template === false).map(i => i.name));
  for (const info of images) {
    const opt = document.createElement('option');
    opt.value = info.name;
    // Show how many of the candidates are currently marked crack, not just
    // the total -- while correcting model proposals the useful question is
    // "how much is flagged here", and it also makes progress visible as you
    // work through an image.
    opt.textContent = info.name + '  (' + info.n_candidates + ' candidates'
      + (info.n_crack != null ? ', ' + info.n_crack + ' marked crack' : '') + ')';
    sel.appendChild(opt);
  }
  if (previousSelection) {
    sel.value = previousSelection;
    return;  // dropdown refreshed, but keep showing the same loaded image/canvas
  }
  if (images.length) {
    currentImage = images[0].name;
    sel.value = currentImage;
    await loadImage(currentImage);
  }
}

function applyZoomStyle() {
  const dispW = Math.round(nativeW * zoom);
  const dispH = Math.round(nativeH * zoom);
  for (const c of [baseCanvas, paintCanvas]) {
    c.style.width = dispW + 'px';
    c.style.height = dispH + 'px';
  }
  canvasInner.style.width = dispW + 'px';
  canvasInner.style.height = dispH + 'px';
}

function fitZoom() {
  const availW = canvasWrap.clientWidth - 40;
  const availH = canvasWrap.clientHeight - 40;
  zoom = Math.min(availW / nativeW, availH / nativeH, 1.0);
  document.getElementById('zoom').value = Math.round(zoom * 100);
  document.getElementById('zoomLabel').textContent = Math.round(zoom * 100) + '%';
  applyZoomStyle();
}

let loadRequestId = 0;
// Images that ship without an overlay (a fresh clone: every one of them). Filled
// from /api/images so loadImage can run detection as a progress-reporting job
// rather than blocking inside a single /api/template request for minutes.
let needsDetect = new Set();
// Images with a detection job already running, so a second click cannot start one.
let detectInFlight = new Set();
// Whether SAM is installed. There is no longer a checkbox: SAM is the detector's
// best-measured configuration (f1 0.776 vs 0.715) and it is a runtime stage, not a
// separate model, so if it is installed it is simply used. Leaving it to a tick box
// in Advanced meant the app quietly ran the weaker configuration and nothing on
// screen said which one produced the overlay you were looking at.
let samInstalled = false;
// SAM is intentionally OFF: the request is the archive model on its own, no
// added SAM stage. samInstalled still reports availability for the model card,
// but nothing runs SAM. Flip USE_SAM to re-enable it in one place.
const USE_SAM = false;

async function loadImage(name) {
  // Keep the scale readout honest: it is per image, so a stale value from the
  // previously opened frame would be worse than showing nothing.
  setTimeout(refreshScaleState, 0);
  // Guards against a slow load (e.g. the largest images need multiple
  // minutes to generate a fresh template) finishing AFTER a later call for
  // a different image has already started -- without this, the slow call
  // would still overwrite the canvas/status with the WRONG image's data
  // once it finally resolves, even though a newer selection has since won
  // (verified: switching away from the default first-loaded image while it
  // was still generating did exactly this).
  const myRequestId = ++loadRequestId;
  const stillCurrent = () => myRequestId === loadRequestId;

  setStatus('Loading ' + name + '...');
  undoStack = [];

  // Most images load in a couple seconds (already-cached template), but the
  // largest ones (~25 megapixels) can take a minute or more to generate the
  // FIRST time -- without this, that long a wait with no feedback beyond
  // "Loading..." looks identical to the page being broken/stuck.
  const slowLoadTimer = setTimeout(() => {
    if (stillCurrent()) setStatus('Still loading ' + name + '... large images can take a minute or two the first time.');
  }, 8000);

  // fetch() rather than a plain <img src=...> load -- only fetch() exposes
  // response headers, and the server uses X-Regenerated to say whether this
  // template was just freshly rebuilt against a newer retrained model, so
  // the picture can visibly change out from under you with an explanation
  // instead of silently (see the status message set below).
  let wasRegenerated = false;
  const img = new Image();

  // No overlay on disk yet: build it through the background job so the progress
  // bar moves, instead of letting /api/template do it inside the fetch below,
  // where the only feedback is the 8-second "still loading" message above.
  if (needsDetect.has(name)) {
    clearTimeout(slowLoadTimer);

    // One job per image. Clicking an unrendered image twice used to POST
    // /api/process twice and run two full pipelines over the same frame,
    // competing for CPU and each writing the same template.
    if (detectInFlight.has(name)) {
      setStatus('Already detecting ' + name + '\u2026');
      return;
    }
    detectInFlight.add(name);
    try {
      await processImage(name, USE_SAM);
      needsDetect.delete(name);
    } catch (e) {
      // Fall through to /api/template, which builds it the blocking way. A
      // failed job should not leave the image unopenable.
      setStatus('detection job failed (' + e.message + '); building directly...');
    } finally {
      detectInFlight.delete(name);
    }
    if (!stillCurrent()) return;
  }

  try {
    const templateRes = await fetch('/api/template/' + name + '?t=' + Date.now());
    if (!stillCurrent()) return;
    wasRegenerated = templateRes.headers.get('X-Regenerated') === 'true';
    const blob = await templateRes.blob();
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = reject;
      img.src = URL.createObjectURL(blob);
    });
  } finally {
    clearTimeout(slowLoadTimer);
  }
  if (!stillCurrent()) return;

  nativeW = img.naturalWidth;
  nativeH = img.naturalHeight;
  baseCanvas.width = nativeW;
  baseCanvas.height = nativeH;
  paintCanvas.width = nativeW;
  paintCanvas.height = nativeH;
  baseCtx.drawImage(img, 0, 0);
  paintCtx.clearRect(0, 0, nativeW, nativeH);

  fitZoom();

  // resume a previous painting session if one exists
  const layerRes = await fetch('/api/paintlayer/' + name + '?t=' + Date.now());
  if (!stillCurrent()) return;
  if (layerRes.status === 200) {
    const blob = await layerRes.blob();
    const layerImg = new Image();
    await new Promise((resolve) => {
      layerImg.onload = resolve;
      layerImg.src = URL.createObjectURL(blob);
    });
    if (!stillCurrent()) return;
    paintCtx.drawImage(layerImg, 0, 0);
    setStatus('Loaded ' + name + ' (resumed previous painting)' + (wasRegenerated ? ' -- refreshed with the newly retrained model' : ''));
  } else {
    setStatus('Loaded ' + name + (wasRegenerated ? ' (refreshed with the newly retrained model)' : ''));
  }
  resetUndoHistory();
}

// Undo stores only the rectangle a stroke actually touched.
//
// This previously snapshotted the WHOLE canvas on every stroke:
// getImageData(0,0,nativeW,nativeH) on a 6144x4096 image is 100 MB, kept 25
// deep, so an ordinary editing session could hold 2.5 GB of undo history. That
// was the main reason painting and image switching felt slow, and it made each
// getImageData call take seconds on the largest images.
//
// beforeCanvas holds the pre-stroke paint layer -- ONE full-size canvas
// (GPU-backed, far cheaper than ImageData) instead of one snapshot per undo
// step. On stroke end the touched rect is copied out of it, then it is
// resynced. Undo puts that rect back into both layers so they stay consistent.
let beforeCanvas = document.createElement('canvas');
let beforeCtx = beforeCanvas.getContext('2d', { willReadFrequently: true });
let undoBytes = 0;
const UNDO_BYTE_BUDGET = 150 * 1024 * 1024;   // ~150 MB, not 2.5 GB
let sMinX = 0, sMinY = 0, sMaxX = 0, sMaxY = 0, sTouched = false;

function resetUndoHistory() {
  undoStack = []; undoBytes = 0; sTouched = false;
  if (!nativeW || !nativeH) return;
  beforeCanvas.width = nativeW; beforeCanvas.height = nativeH;
  beforeCtx.clearRect(0, 0, nativeW, nativeH);
  beforeCtx.drawImage(paintCanvas, 0, 0);
}

function noteStrokePoint(x, y) {
  const pad = brushSize / 2 + 2;
  if (!sTouched) {
    sMinX = x - pad; sMinY = y - pad; sMaxX = x + pad; sMaxY = y + pad; sTouched = true;
  } else {
    sMinX = Math.min(sMinX, x - pad); sMinY = Math.min(sMinY, y - pad);
    sMaxX = Math.max(sMaxX, x + pad); sMaxY = Math.max(sMaxY, y + pad);
  }
}

function pushUndo(fullFrame) {
  if (!nativeW || !nativeH) return;
  if (beforeCanvas.width !== nativeW || beforeCanvas.height !== nativeH) resetUndoHistory();
  let x, y, w, h;
  if (fullFrame || !sTouched) {
    x = 0; y = 0; w = nativeW; h = nativeH;
  } else {
    x = Math.max(0, Math.floor(sMinX)); y = Math.max(0, Math.floor(sMinY));
    w = Math.min(nativeW, Math.ceil(sMaxX)) - x;
    h = Math.min(nativeH, Math.ceil(sMaxY)) - y;
  }
  sTouched = false;
  if (w <= 0 || h <= 0) return;
  let data;
  try { data = beforeCtx.getImageData(x, y, w, h); } catch (e) { return; }
  undoStack.push({ x, y, w, h, data });
  undoBytes += w * h * 4;
  while (undoStack.length > 1 && (undoBytes > UNDO_BYTE_BUDGET || undoStack.length > 60)) {
    const dropped = undoStack.shift();
    undoBytes -= dropped.w * dropped.h * 4;
  }
  beforeCtx.clearRect(x, y, w, h);
  beforeCtx.drawImage(paintCanvas, x, y, w, h, x, y, w, h);
}

function undo() {
  const e = undoStack.pop();
  if (!e) return false;
  undoBytes -= e.w * e.h * 4;
  paintCtx.clearRect(e.x, e.y, e.w, e.h);
  paintCtx.putImageData(e.data, e.x, e.y);
  beforeCtx.clearRect(e.x, e.y, e.w, e.h);
  beforeCtx.putImageData(e.data, e.x, e.y);
  return true;
}

function canvasCoords(e) {
  const rect = paintCanvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) / rect.width * nativeW;
  const y = (e.clientY - rect.top) / rect.height * nativeH;
  return [x, y];
}

function strokeAt(x, y) {
  paintCtx.lineCap = 'round';
  paintCtx.lineJoin = 'round';
  paintCtx.lineWidth = brushSize;
  paintCtx.globalCompositeOperation = 'source-over';
  paintCtx.strokeStyle = currentColor;
  paintCtx.fillStyle = currentColor;
  paintCtx.beginPath();
  paintCtx.moveTo(lastX, lastY);
  paintCtx.lineTo(x, y);
  paintCtx.stroke();
  // also draw a dot at the start so single clicks (no drag) still paint something
  paintCtx.beginPath();
  paintCtx.arc(x, y, brushSize / 2, 0, Math.PI * 2);
  paintCtx.fill();
  // remember what this stroke touched so undo can snapshot just that rect
  noteStrokePoint(lastX, lastY);
  noteStrokePoint(x, y);
  // and remember the geometry itself: the stroke is committed as points, not by
  // re-uploading the whole canvas (see commitStroke).
  strokePts.push([Math.round(x), Math.round(y)]);
}

// Points of the stroke in progress, in image coordinates.
let strokePts = [];

// Commit one stroke by sending its geometry. The old path uploaded the entire
// 25-megapixel paint layer as a PNG dataURL, had the server colour-match it against
// the template three times, re-render the overlay, write a 35.7 MB PNG, and then
// re-downloaded that overlay: measured 8.0 s per stroke on a 6144x4096 image, for a
// stroke touching a few thousand pixels. Sending {mode, points, radius} instead is
// ~0.1 s, and the canvas already shows the stroke so there is nothing to reload.
async function commitStroke() {
  const pts = strokePts;
  strokePts = [];
  if (!pts.length || !currentImage) return;
  const img = currentImage;
  const mode = currentColor === RED ? 'crack'
             : currentColor === CYAN ? 'not_crack'
             : 'erase';
  setSaveState('saving');
  try {
    const r = await fetch('/api/stroke/' + img, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({mode, points: pts, radius: Math.max(1, Math.round(brushSize / 2))}),
    });
    const d = await r.json();
    if (!d.ok) throw new Error(d.error || 'stroke failed');
    setSaveState('saved');
    loadImageList(true);
  } catch (e) {
    // Fall back to the whole-canvas path so a mark is never lost just because the
    // fast route failed.
    setStatus('fast save failed (' + e.message + '); falling back', true);
    markDirty();
  }
}

paintCanvas.addEventListener('mousedown', (e) => {
  // Calibration marks come first: while arming, a click measures the scale bar rather
  // than painting. Without this gate the first mark would also lay down a stroke on the
  // correction mask, so measuring the scale would silently edit the labels.
  if (calibArm) {
    const [cx] = canvasCoords(e);
    calibMarks.push(cx);
    setScaleState(`mark ${calibMarks.length} of 2 at x=${Math.round(cx)}`);
    if (calibMarks.length === 2) { calibArm = false; finishCalibration(); }
    return;
  }
  if (tool === 'bucket') {
    const [x, y] = canvasCoords(e);
    flipRegion(x, y);
    return;
  }
  drawing = true;
  strokePts = [];
  [lastX, lastY] = canvasCoords(e);
  strokeAt(lastX, lastY);
});

async function flipRegion(x, y) {
  const requestedImage = currentImage;
  // SET the clicked region to whatever colour is selected, rather than
  // flipping it to the opposite of whatever it currently is. Flipping
  // made "confirm this cyan region really is not a crack" impossible to
  // express (it would turn red instead), and set-semantics is also just
  // easier to reason about: pick red, click -> crack; pick cyan, click ->
  // not a crack; pick erase, click -> gone.
  const mode = (currentColor === ERASE) ? 'erase'
             : (currentColor === RED ? 'crack' : 'not_crack');
  setStatus(mode === 'erase' ? 'Erasing region...'
          : mode === 'crack' ? 'Marking region as crack...'
          : 'Marking region as NOT a crack...');
  try {
    const res = await fetch('/api/flip_region/' + requestedImage, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ x, y, mode }),
    });
    const result = await res.json();
    if (requestedImage !== currentImage) return;  // user switched images mid-request
    if (!result.ok) throw new Error(result.error || 'flip failed');
    setStatus(result.erased
      ? `Erased a ${result.area}px region.`
      : `Flipped a ${result.area}px region to ${result.newIsCrack ? 'crack (red)' : 'not-crack (cyan)'}.`);

    // the base template's colors changed server-side -- reload just that
    // layer, leaving the paint layer (any of the user's own unsaved
    // strokes) untouched
    const img = new Image();
    img.crossOrigin = 'anonymous';
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = reject;
      img.src = '/api/template/' + requestedImage + '?t=' + Date.now();
    });
    if (requestedImage !== currentImage) return;
    baseCtx.clearRect(0, 0, nativeW, nativeH);
    baseCtx.drawImage(img, 0, 0);
    loadImageList(true);
  } catch (err) {
    if (requestedImage === currentImage) setStatus('Error: ' + err.message, true);
  }
}
paintCanvas.addEventListener('mousemove', (e) => {
  if (!drawing) return;
  const [x, y] = canvasCoords(e);
  strokeAt(x, y);
  lastX = x; lastY = y;
});
window.addEventListener('mouseup', () => {
  if (drawing) { drawing = false; pushUndo(); commitStroke(); }
});
paintCanvas.addEventListener('mouseleave', () => {
  if (drawing) { drawing = false; pushUndo(); commitStroke(); }
});

function selectColor(id, color) {
  currentColor = color;
  // The colour (crack / not-crack / erase) and the scope (Brush vs Whole region)
  // are independent choices, and Whole region applies whichever colour is picked.
  // This used to call setBucketActive(false), so choosing a colour dropped you back
  // to Brush while the segmented control still showed "Whole region" selected --
  // the next click then painted a dab where the user expected a whole region.
  if (typeof syncScope === 'function') setTimeout(syncScope, 0);
  for (const otherId of SWATCH_IDS) {
    document.getElementById(otherId).classList.toggle('selected', otherId === id);
  }
}
document.getElementById('swatchRed').addEventListener('click', () => selectColor('swatchRed', RED));
document.getElementById('swatchCyan').addEventListener('click', () => selectColor('swatchCyan', CYAN));
document.getElementById('swatchErase').addEventListener('click', () => selectColor('swatchErase', ERASE));
function setBucketActive(active) {
  tool = active ? 'bucket' : 'paint';
  const btn = document.getElementById('bucketBtn');
  btn.classList.toggle('active', active);
  btn.textContent = 'Click-to-flip: ' + (active ? 'On' : 'Off');
  paintCanvas.style.cursor = active ? 'pointer' : 'crosshair';
}
document.getElementById('bucketBtn').addEventListener('click', () => {
  setBucketActive(tool !== 'bucket');
});
document.getElementById('brushSize').addEventListener('input', (e) => {
  brushSize = parseInt(e.target.value, 10);
  document.getElementById('brushSizeLabel').textContent = brushSize + 'px';
});
document.getElementById('zoom').addEventListener('input', (e) => {
  zoom = parseInt(e.target.value, 10) / 100;
  document.getElementById('zoomLabel').textContent = e.target.value + '%';
  applyZoomStyle();
});
document.getElementById('fitBtn').addEventListener('click', fitZoom);
document.getElementById('undoBtn').addEventListener('click', undo);

// Cmd-Z / Ctrl-Z for undo. The tool had no keyboard shortcuts at all, and
// reaching for the mouse after every misplaced stroke is the wrong ergonomics
// for something used across dozens of images.
//
// preventDefault matters: without it the browser's own undo fires as well and
// can revert the page's form state (the image dropdown, the brush slider),
// which looks like the app losing its place.
window.addEventListener('keydown', (e) => {
  const typing = /^(INPUT|TEXTAREA|SELECT)$/.test((e.target.tagName || '')) ||
                 e.target.isContentEditable;
  if (typing) return;                       // let a real text field keep its own undo
  const mod = e.metaKey || e.ctrlKey;
  const k = (e.key || '').toLowerCase();
  if (mod && k === 'z' && !e.shiftKey) {
    e.preventDefault();
    // Every stroke is committed to the server the moment it finishes, so the
    // authoritative revert is always the server's snapshot stack. The local canvas
    // undo only repaints -- it cannot unsay what the mask already records.
    //
    // This used to call markDirty() after a local undo, which queued the OLD
    // whole-canvas autosave: /api/save, then /api/ingest at ~7 s, then a 23 MB
    // overlay re-download. So pressing Cmd-Z right after a 24 ms stroke dragged the
    // whole 8-second path back into the session, which is exactly what "still saving
    // really slowly" was.
    undo();                       // repaint the canvas
    undoCommitted();              // and revert the committed mask
  } else if (!mod && k === 'f') {
    e.preventDefault();
    fitZoom();
  } else if (!mod && (k === '1' || k === '2' || k === '3')) {
    // 1/2/3 to pick crack / not-crack / erase without leaving the canvas
    e.preventDefault();
    const picks = [['swatchRed', RED], ['swatchCyan', CYAN], ['swatchErase', ERASE]];
    const [id, col] = picks[parseInt(k, 10) - 1];
    selectColor(id, col);
    setStatus(col === RED ? 'crack' : col === CYAN ? 'not-crack' : 'erase');
  }
});
document.getElementById('clearBtn').addEventListener('click', () => {
  if (confirm('Clear all painted strokes for this image?')) {
    pushUndo(true);
    paintCtx.clearRect(0, 0, nativeW, nativeH);
    // Persist it. Clearing only the canvas left <image>_painted.png untouched, so
    // the strokes came back on the next load with nothing said about it.
    markDirty();
  }
});
document.getElementById('imageSelect').addEventListener('change', async (e) => {
  // flush pending marks before leaving, so switching images cannot lose work
  if (savePending || saveInFlight) { setStatus('Saving before switching\u2026'); await commitNow(true); }
  currentImage = e.target.value;
  setSaveState('idle');
  loadImage(currentImage);
});

async function savePaint(img) {
  // The image is passed in, never read from currentImage here: this runs inside
  // commitNow after awaits, and the user can have switched images by then.
  img = img || currentImage;
  const dataURL = paintCanvas.toDataURL('image/png');
  const res = await fetch('/api/save/' + img, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataURL }),
  });
  const result = await res.json();
  if (!result.ok) throw new Error(result.error || 'save failed');
  return result;
}

// ---- autosave ----
// There is no Save button any more. Two buttons that both "saved" was a false
// choice: Save wrote the stroke layer, Save & Ingest additionally folded those
// strokes into the candidate set -- and only the second one actually recorded a
// verdict the model would ever learn from, so anyone pressing Save alone was
// quietly getting less than they expected.
//
// Marks are now committed automatically once drawing pauses. Debounced rather
// than per-stroke because ingest re-renders the overlay, and doing that between
// two quick strokes would fight the user.
let saveTimer = null, saveInFlight = false, savePending = false;
// Bumped on every mark. commitNow compares it across its awaits to detect strokes
// drawn while a commit was in flight -- those used to be wiped by the reload at the
// end of the commit and never sent anywhere.
let strokeSeq = 0;
// Resolves when the in-flight commit finishes, so commitNow(true) can actually
// wait for it instead of returning immediately and reporting a flush that did not
// happen.
let saveChain = null;
const AUTOSAVE_IDLE_MS = 1100;

function markDirty() {
  savePending = true;
  strokeSeq++;
  setSaveState('pending');
  clearTimeout(saveTimer);
  saveTimer = setTimeout(commitNow, AUTOSAVE_IDLE_MS);
}

function setSaveState(state) {
  const el = document.getElementById('saveState');
  if (!el) return;
  const map = {
    idle:    ['', ''],
    pending: ['Unsaved changes', 'var(--ink2)'],
    saving:  ['Saving\u2026', 'var(--ink2)'],
    saved:   ['All changes saved', '#7fd4a3'],
    error:   ['Save failed \u2014 use Retry', '#ff7b7b'],
  };
  const [txt, col] = map[state] || map.idle;
  el.textContent = txt;
  el.style.color = col;
  const retry = document.getElementById('retryBtn');
  if (retry) retry.style.display = (state === 'error') ? '' : 'none';
}

async function commitNow(silent, _depth) {
  clearTimeout(saveTimer);
  if (!currentImage || !savePending) return;
  if (saveInFlight) {
    // Wait for the in-flight commit, then flush what is still pending. This used
    // to return immediately, which made `await commitNow(true)` on an image switch
    // a no-op: the status line said "Saving before switching..." and nothing was
    // flushed, so the outgoing image's strokes never reached its correction mask.
    saveTimer = setTimeout(commitNow, 400);
    if (saveChain) { try { await saveChain; } catch (e) { /* reported already */ } }
    if ((_depth || 0) < 3 && savePending) return commitNow(silent, (_depth || 0) + 1);
    return;
  }
  // Pin the image and the stroke count for the whole commit. currentImage can
  // change while the awaits below are pending, and the ingest URL was built after
  // them -- so switching images mid-save ingested the NEW image, pushed an undo
  // snapshot onto it, and left the outgoing image's strokes uncommitted while the
  // badge read "All changes saved".
  const img = currentImage;
  const strokesAtStart = strokeSeq;
  saveInFlight = true; savePending = false;
  setSaveState('saving');
  let _release;
  saveChain = new Promise((r) => { _release = r; });
  try {
    await savePaint(img);
    const res = await fetch('/api/ingest/' + img, { method: 'POST' });
    const result = await res.json();
    if (!result.ok) throw new Error(result.error || 'ingest failed');
    // Keep pixels drawn while this commit was in flight. The reload below clears
    // the paint layer and repaints it from <image>_painted.png, which holds only
    // what toDataURL captured at the START of the commit -- so any stroke finished
    // in that window (it opens 1.1 s after a mouseup and lasts seconds) was wiped
    // off the canvas and had never been sent anywhere.
    // Preserve the Show-result choice across the reload: loadImage always draws
    // the overlay template, so unticking it was silently reverted by the next
    // autosave or region flip.
    const _showEl = document.getElementById('showResult');
    const _wasShowing = !_showEl || _showEl.checked;
    const live = (strokeSeq !== strokesAtStart)
      ? paintCanvas.toDataURL('image/png') : null;
    // reload so corrected/erased pixels show their real committed appearance
    // rather than the transient marker colour
    await loadImage(img);
    if (!_wasShowing && _showEl) {
      _showEl.checked = false;
      _showEl.dispatchEvent(new Event('change'));
    }
    if (live && currentImage === img) {
      await new Promise((r) => {
        const im = new Image();
        im.onload = () => { paintCtx.drawImage(im, 0, 0); r(); };
        im.onerror = r;
        im.src = live;
      });
      markDirty();                 // and commit them on the next cycle
    } else {
      setSaveState('saved');
    }
    if (!silent) setStatus(result.message || 'Saved.');
    loadImageList(true);
  } catch (err) {
    savePending = true;            // keep the work; let the user retry
    setSaveState('error');
    setStatus('Save failed: ' + err.message, true);
  } finally {
    saveInFlight = false;
    if (_release) _release();
    saveChain = null;
  }
}

// don't lose marks when switching images or closing the tab
const _origLoadImageForSave = loadImage;
window.addEventListener('beforeunload', (e) => {
  if (savePending || saveInFlight) { e.preventDefault(); e.returnValue = ''; }
});

document.getElementById('retryBtn').addEventListener('click', async () => {
  try {
    setStatus('Saving...');
    await savePaint();
    setStatus('Ingesting...');
    const res = await fetch('/api/ingest/' + currentImage, { method: 'POST' });
    const result = await res.json();
    if (!result.ok) throw new Error(result.error || 'ingest failed');
    // The server just committed this ingest -- corrected/erased pixels were
    // reset to their real final appearance and the template was rewritten
    // to match. Reload both canvas layers so the CURRENTLY OPEN session
    // reflects that immediately (without this, an erased stroke kept
    // showing its erase-marker color until you switched away and back,
    // even though the underlying data was already correct).
    await loadImage(currentImage);
    setStatus(result.message || (result.n_candidates + ' candidate(s) added.'));
    loadImageList(true);  // refresh candidate counts in the dropdown, stay on this image
  } catch (err) {
    setStatus('Error: ' + err.message, true);
  }
});

window.addEventListener('resize', () => {});

// ---------------------------------------------------------------- app layer
// Drag-and-drop upload, hybrid detection with progress, one-click retrain.
// Long jobs are polled rather than awaited: SAM takes ~3 min/image, which no
// browser should hold a request open for.

// Progress shows as a 3px strip under the toolbar plus the status line, which
// is how the sibling TXM app does it -- a boxed panel pinned over the bottom of
// the window covered the canvas exactly where the status line already lives.
function showJob(label) {
  document.getElementById('prog').style.width = '0%';
  document.getElementById('jobLabel').textContent = label;
  setStatus(label + '\u2026');
}
function hideJob() {
  document.getElementById('prog').style.width = '0%';
}

async function pollJob(jobId, label) {
  showJob(label);
  while (true) {
    await new Promise(r => setTimeout(r, 1200));
    let j;
    try { j = await (await fetch('/api/job/' + jobId)).json(); }
    catch (e) { continue; }                       // transient: keep polling
    if (!j.ok) { hideJob(); throw new Error(j.error || 'job vanished'); }
    const pct = Math.round((j.frac || 0) * 100);
    document.getElementById('prog').style.width = pct + '%';
    document.getElementById('jobFill').style.width = pct + '%';
    const detail = (j.stage || '') + (j.note ? ' \u2014 ' + j.note : '');
    document.getElementById('jobNote').textContent = detail;
    setStatus(label + ' \u2014 ' + pct + '%' + (detail ? '  (' + detail + ')' : ''));
    if (j.state === 'done') { hideJob(); return j.result; }
    if (j.state === 'error') { hideJob(); throw new Error(j.error || 'job failed'); }
  }
}

async function processImage(name, useSam) {
  const r = await (await fetch('/api/process/' + encodeURIComponent(name), {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({use_sam: useSam})
  })).json();
  if (!r.ok) throw new Error(r.error || 'could not start');
  return await pollJob(r.job, 'Detecting cracks in ' + name);
}

async function handleFiles(fileList) {
  const files = [...fileList];
  if (!files.length) return;
  const fd = new FormData();
  files.forEach(f => fd.append('files', f));
  setStatus('uploading ' + files.length + ' file(s)…');
  let up;
  try { up = await (await fetch('/api/upload', {method: 'POST', body: fd})).json(); }
  catch (e) { setStatus('upload failed: ' + e.message); return; }
  if (!up.ok) { setStatus('upload failed: ' + (up.error || 'unknown')); return; }
  if (up.failed && up.failed.length) {
    setStatus(up.failed.length + ' file(s) rejected: ' + up.failed.map(f => f.file).join(', '));
  }
  const useSam = USE_SAM;
  for (const name of up.added) {
    try {
      const res = await processImage(name, useSam);
      setStatus(name + ': ' + res.n_crack + ' crack regions of ' + res.n_candidates +
                (res.n_sam_regions ? ' (' + res.n_sam_regions + ' from SAM)' : ''));
    } catch (e) {
      setStatus('processing ' + name + ' failed: ' + e.message);
    }
  }
  await loadImageList(false);
  if (up.added.length) {
    const sel = document.getElementById('imageSelect');
    sel.value = up.added[up.added.length - 1];
    await loadImage(sel.value);
  }
}

const dz = document.getElementById('dropZone');
let dragDepth = 0;
window.addEventListener('dragenter', e => {
  if (!e.dataTransfer || ![...e.dataTransfer.types].includes('Files')) return;
  e.preventDefault(); dragDepth++; dz.classList.add('active');
});
window.addEventListener('dragover', e => {
  if (e.dataTransfer && [...e.dataTransfer.types].includes('Files')) e.preventDefault();
});
window.addEventListener('dragleave', e => {
  if (--dragDepth <= 0) { dragDepth = 0; dz.classList.remove('active'); }
});
window.addEventListener('drop', e => {
  if (!e.dataTransfer || !e.dataTransfer.files.length) return;
  e.preventDefault(); dragDepth = 0; dz.classList.remove('active');
  handleFiles(e.dataTransfer.files);
});
dz.addEventListener('click', () => {
  const inp = document.createElement('input');
  inp.type = 'file'; inp.multiple = true;
  inp.accept = '.tif,.tiff,.png,.jpg,.jpeg,.bmp';
  inp.onchange = () => handleFiles(inp.files);
  inp.click();
});

document.getElementById('retrainBtn').addEventListener('click', async () => {
  if (!confirm('Rebuild training data from every correction, retrain the model, and ' +
               're-render all images?\n\nThe new model is only deployed if it scores at ' +
               'least as well as the current one on held-out data.')) return;
  const btn = document.getElementById('retrainBtn');
  btn.disabled = true;
  try {
    const r = await (await fetch('/api/retrain', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({regenerate: true})
    })).json();
    if (!r.ok) throw new Error(r.error || 'could not start');
    const res = await pollJob(r.job, 'Retraining on your corrections');
    setStatus(res && res.reason ? res.reason : 'retrain finished');
    await refreshModelInfo();
    await loadImageList(true);
    if (currentImage) await loadImage(currentImage);
  } catch (e) {
    setStatus('retrain failed: ' + e.message);
  } finally {
    btn.disabled = false;
  }
});

async function refreshModelInfo() {
  try {
    const i = await (await fetch('/api/pipeline_info')).json();
    const m = i.model;
    samInstalled = !!i.sam_available;
  } catch (e) { /* non-fatal */ }
}

// ---- downloads ----
// Straight window.location for the single-image ones: they are streamed by the
// server as attachments, so the browser handles the save dialog and no blob
// juggling is needed. The all-images zip is a job, so it polls first.
function dl(path) {
  if (!currentImage) { setStatus('load an image first'); return; }
  window.location = '/api/export/' + encodeURIComponent(currentImage) + '/' + path;
}
document.getElementById('dlMask').addEventListener('click', () => dl('mask.png'));
document.getElementById('dlOverlay').addEventListener('click', () => dl('overlay.png'));
document.getElementById('dlCsv').addEventListener('click', () => dl('regions.csv'));
document.getElementById('dlAll').addEventListener('click', async () => {
  const btn = document.getElementById('dlAll');
  btn.disabled = true;
  try {
    const r = await (await fetch('/api/export_all', {method: 'POST'})).json();
    if (!r.ok) throw new Error(r.error || 'could not start');
    const res = await pollJob(r.job, 'Exporting every processed image');
    setStatus('exported ' + res.images + ' images, ' + res.regions + ' crack regions');
    window.location = res.download;
  } catch (e) {
    setStatus('export failed: ' + e.message);
  } finally { btn.disabled = false; }
});

// ================= redesigned-shell wiring =================
// The hidden <select id="imageSelect"> stays the single source of truth, so all
// existing logic (loadImage, ingest, template refresh) is untouched. The visual
// sidebar list is a view over it: clicking an item sets the select and fires the
// same 'change' event the old dropdown fired.

let _imgCache = [];

function renderImageList() {
  const q = (document.getElementById('imgSearch').value || '').toLowerCase();
  const box = document.getElementById('imageList');
  slimSidebar(_imgCache.length);
  box.innerHTML = '';
  let shown = 0;
  for (const info of _imgCache) {
    if (q && !info.name.toLowerCase().includes(q)) continue;
    shown++;
    const el = document.createElement('div');
    el.className = 'item' + (info.name === currentImage ? ' sel' : '');
    const ready = !!info.n_candidates;
    const dot = document.createElement('span');
    dot.className = 'dot' + (ready ? '' : ' busy');
    // Thumbnail rather than pointing an <img> at the template: these templates
    // are 6-33 MB, so 62 rows each decoding one full-resolution PNG down to a
    // 38px box would be well over a gigabyte of transfer to draw a sidebar.
    // /api/thumb is ~6 KB and shows the result, so a row is informative.
    let th;
    if (ready) {
      th = document.createElement('img');
      th.className = 'th'; th.loading = 'lazy';
      th.src = '/api/thumb/' + encodeURIComponent(info.name) + '?w=128';
      th.onerror = () => { th.style.visibility = 'hidden'; };
    } else {
      th = document.createElement('span'); th.className = 'th';
    }
    const tx = document.createElement('span');
    tx.className = 'tx';
    const nm = document.createElement('span');
    nm.className = 'nm'; nm.textContent = info.name; nm.title = info.name;
    const sub = document.createElement('span');
    sub.className = 'sub';
    sub.textContent = ready
      ? (info.n_crack != null ? info.n_crack + ' crack \u00b7 ' : '') + info.n_candidates + ' candidates'
      : 'not processed yet';
    tx.appendChild(nm); tx.appendChild(sub);
    const x = document.createElement('button');
    x.className = 'x'; x.textContent = '\u00d7';
    x.title = 'Remove ' + info.name + ' from the app (the file is moved, not deleted)';
    x.onclick = (ev) => { ev.stopPropagation(); removeImage(info.name); };
    el.appendChild(dot); el.appendChild(th); el.appendChild(tx); el.appendChild(x);
    el.addEventListener('click', () => {
      const sel = document.getElementById('imageSelect');
      sel.value = info.name;
      sel.dispatchEvent(new Event('change', {bubbles: true}));
    });
    if (info.name === currentImage) el.dataset.sel = '1';
    box.appendChild(el);
  }
  // With 62 images the selected one is usually outside the scroll viewport, so
  // the list looked like nothing was selected. Bring it into view -- 'nearest'
  // so it does not yank the list around when it is already visible.
  const cur = box.querySelector('.item[data-sel="1"]');
  if (cur) {
    // Explicit scrollTop rather than scrollIntoView: the latter silently did
    // nothing here, since the row is appended and measured in the same frame.
    // Defer one frame so layout has settled, then centre it only when it is
    // actually outside the viewport.
    requestAnimationFrame(() => {
      const top = cur.offsetTop, bot = top + cur.offsetHeight;
      if (top < box.scrollTop || bot > box.scrollTop + box.clientHeight) {
        box.scrollTop = Math.max(0, top - box.clientHeight / 2 + cur.offsetHeight / 2);
      }
    });
  }
  if (!shown) {
    const e = document.createElement('div');
    e.style.cssText = 'padding:10px;color:var(--ink3);font-size:11.5px';
    e.textContent = _imgCache.length ? 'No match.' : 'No images yet — drop some in.';
    box.appendChild(e);
  }
}
document.getElementById('imgSearch').addEventListener('input', renderImageList);

// loadImageList already fetches /api/images and fills the select; mirror that
// data into the sidebar by wrapping it rather than duplicating the fetch.
const _origLoadImageList = loadImageList;
loadImageList = async function (keepCurrent) {
  await _origLoadImageList(keepCurrent);
  try { _imgCache = await (await fetch('/api/images')).json(); } catch (e) { }
  renderImageList();
  updateHeader();
};

function updateHeader() {
  const info = _imgCache.find(i => i.name === currentImage);
  document.getElementById('curName').textContent = currentImage || 'No image selected';
  document.getElementById('curMeta').textContent = info
    ? ((info.n_crack != null ? info.n_crack + ' crack regions of ' : '') +
       (info.n_candidates || 0) + ' candidates')
    : '';
  renderImageList();
}
// keep the header and selection highlight in step with whatever loads
const _origLoadImage = loadImage;
loadImage = async function (name) {
  await _origLoadImage(name);
  updateHeader();
};

// ---- zoom as buttons; the slider stays hidden as the mechanism ----
const zoomEl = document.getElementById('zoom');
function bumpZoom(mult) {
  const cur = parseInt(zoomEl.value, 10) || 100;
  const next = Math.max(10, Math.min(800, Math.round(cur * mult)));
  zoomEl.value = next;
  zoomEl.dispatchEvent(new Event('input', {bubbles: true}));
}
// Null-safe binding. The zoom +/- buttons exist in some layouts and not others
// (the current one puts zoom in Advanced as a slider), and an unguarded
// getElementById(...).addEventListener on a missing element throws at load,
// which silently kills every line after it -- that is exactly what left the
// image list empty and the model picker unpopulated after the last restyle.
function on(id, ev, fn) {
  const el = document.getElementById(id);
  if (el) el.addEventListener(ev, fn);
  return !!el;
}
on('zoomIn', 'click', () => bumpZoom(1.25));
on('zoomOut', 'click', () => bumpZoom(0.8));

// ---- Brush / Whole-region reads as a mode, not a toggle labelled Off ----
const bucket = document.getElementById('bucketBtn');
function syncModeLabel() {
  const on = /whole/i.test(bucket.textContent) || bucket.classList.contains('on');
  bucket.textContent = on ? 'Whole region' : 'Brush';
  bucket.classList.toggle('on', on);
}
bucket.addEventListener('click', () => setTimeout(() => {
  // setBucketActive() flips `tool`; read that rather than guessing from the label
  const on = (typeof tool !== 'undefined') && tool === 'bucket';
  bucket.textContent = on ? 'Whole region' : 'Brush';
  bucket.classList.toggle('on', on);
}, 0));
syncModeLabel();

// ---- disclosure rows ----
document.getElementById('advToggle').addEventListener('click', () => {
  const a = document.getElementById('adv');
  a.classList.toggle('open');
  document.getElementById('advToggle').textContent =
    a.classList.contains('open') ? 'Options ▴' : 'Options ▾';
});
document.getElementById('helpBtn').addEventListener('click', () =>
  document.getElementById('help').classList.toggle('open'));

// ---- export menu ----
const expBtn = document.getElementById('exportBtn');
const expMenu = document.getElementById('expMenu');
expBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  const r = expBtn.getBoundingClientRect();
  expMenu.style.left = r.left + 'px';
  expMenu.style.top = (r.bottom + 5) + 'px';
  expMenu.classList.toggle('open');
});
document.addEventListener('click', (e) => {
  if (!expMenu.contains(e.target) && e.target !== expBtn) expMenu.classList.remove('open');
});
for (const id of ['dlMask', 'dlOverlay', 'dlCsv', 'dlAll']) {
  document.getElementById(id).addEventListener('click', () => expMenu.classList.remove('open'));
}

// ---- sidebar drop target mirrors the full-window drop zone ----
document.getElementById('dropTarget').addEventListener('click', () => {
  const inp = document.createElement('input');
  inp.type = 'file'; inp.multiple = true;
  inp.accept = '.tif,.tiff,.png,.jpg,.jpeg,.bmp';
  inp.onchange = () => handleFiles(inp.files);
  inp.click();
});

// ---- richer model card in the sidebar ----
const _origRefreshModelInfo = refreshModelInfo;
refreshModelInfo = async function () {
  await _origRefreshModelInfo();
  try {
    const i = await (await fetch('/api/pipeline_info')).json();
    const m = i.model, card = document.getElementById('modelCard');
    if (!m) {
      // Write the failure where it can be SEEN. #modelCard lives inside .rows, which is
      // collapsed by default now, so "No model loaded." was rendered into a hidden element:
      // the app looked normal while having no classifier at all. Set the summary too and
      // expand the card so the state is unmissable.
      card.textContent = 'No model loaded.';
      const s0 = document.getElementById('modelSummary');
      if (s0) { s0.textContent = 'NO MODEL LOADED'; s0.style.color = 'var(--bad)'; }
      const mc0 = document.getElementById('modelcard');
      if (mc0) mc0.classList.remove('collapsed');
      return;
    }
    card.innerHTML =
      '<div class="row"><span>Type</span><b>' + m.family + '</b></div>' +
      '<div class="row"><span>Threshold</span><b>' + m.threshold.toFixed(3) + '</b></div>' +
      // Only rendered when the bundle actually records them. The archived model
      // carries no n_train/n_images, and printing the .get() fallbacks rendered
      // "Trained on 0 regions / From 0 images", which reads as an untrained model.
      (m.n_train ? '<div class="row"><span>Trained on</span><b>' +
        m.n_train.toLocaleString() + ' regions</b></div>' : '') +
      (m.n_images ? '<div class="row"><span>From</span><b>' + m.n_images +
        ' images</b></div>' : '') +
      // Says which configuration produced what you are looking at. It used to read
      // only "available / not installed", so when the overlays silently reverted to
      // the no-SAM configuration nothing on screen indicated it.
      '<div class="row"><span>Detector</span><b style="color:' +
        (i.sam_available ? 'var(--ok)' : 'var(--ink3)') + '">' +
        'Pass 1 + Pass 2 (archive model, no SAM)' + '</b></div>' +
      '<div class="row"><span>model source</span><b>archive (CBS_Crack_Detection_All)</b></div>' +
      // Shown so a screenshot or a support question identifies the release. Exported
      // provenance records the same string, which is what ties a CSV to a version
      // rather than to a moving `main`.
      (i.version ? '<div class="row"><span>version</span><b>' + i.version + '</b></div>' : '');

      // ---- HOW WELL IT DOES, on screen ----
      // The card listed what the model IS and nothing about how well it works, so the only
      // performance figure anywhere in the UI was a fragment of the model-picker label. The
      // sibling TXM app leads its panel with a held-out score and a false-call rate; this is
      // the same idea with this project's numbers, and it states outright where a number
      // does not exist rather than leaving the row off.
      const P = i.performance || {};
      const g = P.grouped_cv, pxm = P.pixel;
      const perfRow = (label, val, tip) =>
        '<div class="row" title="' + String(tip).replace(/"/g, '&quot;') + '">' +
        '<span>' + label + '</span><b>' + val + '</b></div>';
      let perf = '';
      if (m.held_out_auc != null)
        perf += perfRow('held out', 'AUC ' + m.held_out_auc.toFixed(3),
          (m.held_out_kind || 'held out by specimen') +
          (m.held_out_image ? ', holding out ' + m.held_out_image : '') +
          (m.in_sample_auc != null ? '. In-sample for reference ' +
             m.in_sample_auc.toFixed(3) + '.' : '') +
          ' This is the bar a retrain has to clear before it is deployed.');
      if (g)
        perf += perfRow('grouped CV', 'AUC ' + g.auc.toFixed(3) + ' \u00b1' + g.auc_sd.toFixed(3),
          g.repeats + ' x StratifiedGroupKFold(' + g.repeats + ') over ' + g.n_regions +
          ' candidate regions (' + g.n_pos + ' crack / ' + g.n_neg + ' not) from ' +
          g.n_groups + ' images, grouped so train and test never share an image. ' +
          'Balanced accuracy ' + g.balacc.toFixed(3) + ' \u00b1' + g.balacc_sd.toFixed(3) +
          ', worst repeat ' + g.balacc_worst.toFixed(3) + '. Recall ' + g.recall.toFixed(3) +
          ', specificity ' + g.specificity.toFixed(3) + ', precision ' +
          g.precision.toFixed(3) + '. This answers "how will it do on an image it has not ' +
          'seen"; it scores the region LABEL, not the boundary.');
      if (pxm && pxm.f1 != null)
        perf += perfRow('pixel f1', pxm.f1.toFixed(3),
          'Pixel level on adjudicated pixels over ' + pxm.n_frames + ' frames carrying both ' +
          'a crack and a not-crack verdict: recall ' + pxm.recall.toFixed(3) + ', specificity ' +
          pxm.specificity.toFixed(3) + ', precision ' + pxm.precision.toFixed(3) + '. Lower ' +
          'than the region number because drawing a boundary is harder than labelling a ' +
          'region -- about ' + Math.round((1 - pxm.recall) * 100) + '% of crack pixels are ' +
          'still missed.');
      if (P.false_calls_reason)
        perf += perfRow('false calls', '\u2014', P.false_calls_reason);
      if (perf)
        card.innerHTML += '<div class="k" style="margin-top:9px">Performance</div>' +
                          '<div class="v">' + perf + '</div>';

      // The collapsed line has to carry enough to be useful on its own, otherwise collapsing
      // the card just hides information rather than compacting it. The held-out score belongs
      // here: the card is collapsed by default, so a performance number only inside it is a
      // number nobody sees.
      const _sum = document.getElementById('modelSummary');
      if (_sum) {
        // "held-out AUC 0.885" spelled out wrapped this line to two in a 296 px sidebar,
        // which is most of what collapsing the card was meant to save. The number stays; the
        // word moves to the tooltip.
        _sum.textContent = m.family + ' \u00b7 thr ' + Number(m.threshold).toFixed(3)
                           + (m.held_out_auc != null
                              ? ' \u00b7 AUC ' + m.held_out_auc.toFixed(3) : '')
                           + ' \u00b7 no SAM';
        _sum.title = (m.held_out_auc != null
          ? 'AUC ' + m.held_out_auc.toFixed(3) + ' is held out by specimen'
            + (m.held_out_image ? ' (' + m.held_out_image + ' left out)' : '')
            + ' -- the bar a retrain must clear. Click for the full breakdown.'
          : 'Click for the full model breakdown.');
      }
  } catch (e) { }
};

async function undoCommitted() {
  if (!currentImage) return;
  try {
    const r = await (await fetch('/api/undo_correction/' + encodeURIComponent(currentImage),
                                 {method: 'POST'})).json();
    if (!r.ok) { setStatus(r.error || 'Nothing left to undo.'); return; }
    const res = await pollJob(r.job, 'Undoing last saved correction');
    savePending = false; setSaveState('idle');
    await loadImage(currentImage);
    await loadImageList(true);
    setStatus((res && res.message ? res.message : 'Undone') +
              (res && res.undo_depth != null ? '  (' + res.undo_depth + ' more available)' : ''));
  } catch (err) {
    setStatus('Undo failed: ' + err.message, true);
  }
}

// ---- image count in the sidebar label ----
const _origRender = renderImageList;
renderImageList = function () {
  _origRender();
  const n = document.getElementById('ncount');
  if (n) n.textContent = _imgCache.length ? '(' + _imgCache.length + ')' : '';
};

// ---- remove an image (moved, not deleted) ----
async function removeImage(name) {
  if (!confirm('Remove ' + name + " from the app?\n\nThe file is MOVED to removed_images/, not deleted, and your corrections for it are kept.")) return;
  try {
    const r = await (await fetch('/api/remove/' + encodeURIComponent(name), {method: 'POST'})).json();
    if (!r.ok) throw new Error(r.error || 'failed');
    setStatus(name + ' removed — ' + (r.note || ''));
    if (currentImage === name) currentImage = null;
    await loadImageList(false);
  } catch (e) { setStatus('Remove failed: ' + e.message, true); }
}

// ---- scope segment: Brush vs Whole region ----
// bucketBtn already toggles `tool`; these two buttons are the visible state of
// that one variable, so the label can never disagree with what a click does.
function syncScope() {
  const isRegion = (typeof tool !== 'undefined') && tool === 'bucket';
  document.getElementById('bucketBtn').classList.toggle('on', isRegion);
  document.getElementById('brushModeBtn').classList.toggle('on', !isRegion);
}
document.getElementById('brushModeBtn').addEventListener('click', () => {
  if (typeof setBucketActive === 'function') setBucketActive(false);
  syncScope();
});
document.getElementById('bucketBtn').addEventListener('click', () => setTimeout(syncScope, 0));
syncScope();

// ---- Show result: swap the overlay for the plain image ----
// The base canvas normally holds the template (image + overlay burned in).
// Unticking draws /api/raw instead, so the microstructure can be inspected
// without the model's opinion on top of it.
const rawCache = {};
document.getElementById('showResult').addEventListener('change', async (e) => {
  if (!currentImage) return;
  const show = e.target.checked;
  try {
    if (show) { await loadImage(currentImage); return; }
    const img = rawCache[currentImage] || await new Promise((res, rej) => {
      const i = new Image();
      i.onload = () => res(i); i.onerror = rej;
      i.src = '/api/raw/' + encodeURIComponent(currentImage);
    });
    rawCache[currentImage] = img;
    baseCtx.clearRect(0, 0, nativeW, nativeH);
    baseCtx.drawImage(img, 0, 0, nativeW, nativeH);
    setStatus('Result hidden — showing the plain image');
  } catch (err) { setStatus('Could not load the plain image: ' + err.message, true); }
});

// ---- model picker / rollback ----
async function refreshModelPicker() {
  try {
    const r = await (await fetch('/api/models')).json();
    const sel = document.getElementById('mpick');
    sel.innerHTML = '';
    for (const m of (r.models || [])) {
      const o = document.createElement('option');
      o.value = m.file;
      const auc = m.held_out_auc != null ? '  held-out AUC ' + m.held_out_auc : '';
      o.textContent = (m.is_current ? '● ' : '   ') + m.file.replace('crack_classifier', 'model')
                      + '  · thr ' + (m.threshold != null ? m.threshold.toFixed(3) : '?') + auc;
      if (m.is_current) o.selected = true;
      sel.appendChild(o);
    }
  } catch (e) { /* non-fatal */ }
}
document.getElementById('mpick').addEventListener('change', async (e) => {
  const file = e.target.value;
  if (!confirm('Switch to ' + file + '?\n\nThe model currently in use is backed up first. Overlays stay stale until you press Re-apply model.')) {
    refreshModelPicker(); return;
  }
  try {
    const r = await (await fetch('/api/model/select', {method: 'POST',
      headers: {'Content-Type': 'application/json'}, body: JSON.stringify({file})})).json();
    if (!r.ok) throw new Error(r.error || 'failed');
    setStatus(r.message || 'model switched');
    await refreshModelInfo();
refreshModelPicker();
  } catch (err) { setStatus('Switch failed: ' + err.message, true); }
});

// ---- one-click SAM install, so the best detector needs no terminal ----
// This block used to sit INSIDE the model-picker change handler: the line above
// ended with a stray `await`, which made the listener registration its operand.
// So the Enable SAM button got no click handler on load, and the code that reveals
// it (and disables the Use SAM checkbox) never ran -- SAM could not be turned on
// from the UI at all.
// ---- physical-unit calibration -------------------------------------------------
// How precise the scale is, shown beside it. A calibration displayed as a bare number
// invites the reader to trust every digit; a 200 px bar marked to a pixel or two per end is
// good to about 1%, and saying so is also the only thing that makes marking a LONGER bar
// feel worth the extra second. Absent means not characterised -- never shown as 0%.
function scalePrecisionText(rec) {
  if (!rec) return '';
  const rel = rec.um_per_px_rel_sd;
  if (typeof rel !== 'number' || !(rel >= 0)) return '';
  const pct = 100 * rel;
  return ' \u00b1' + (pct < 1 ? pct.toFixed(2) : pct.toFixed(1)) + '%';
}
// Exported lengths were pixels, which is not a publishable quantity. Two clicks on the
// burned-in scale bar plus its printed label give um/px exactly; the span is measured
// from the marks rather than typed, so it does not inherit a hand-drawn line's aiming
// error the way ImageJ's Set Scale does.
let calibArm = false, calibMarks = [], calibImage = null, calibW = 0;

// ---- sidebar density ------------------------------------------------------------
// The list is the primary navigation for a 62-image corpus, so the chrome above and
// below it yields once there is a corpus to navigate.
function slimSidebar(nImages) {
  const dt = document.getElementById('dropTarget');
  if (dt) dt.classList.toggle('slim', nImages > 0);
  const mc = document.getElementById('modelcard');
  if (mc && !mc.dataset.wired) {
    mc.dataset.wired = '1';
    // Collapsed by default, EXCEPT when the URL asks for it. #model opens the card on load,
    // which is how docs/img/README.md captures the performance panel with headless Chrome --
    // app.png went stale for six days because its doc said "not scripted", and a panel that
    // can only be photographed by hand is a panel whose screenshot will rot the same way.
    if ((location.hash || '').toLowerCase() !== '#model') mc.classList.add('collapsed');
    const sum = mc.querySelector('.sum');
    if (sum) sum.addEventListener('click', () => {
      mc.classList.toggle('collapsed');
      const c = sum.querySelector('.caret');
      if (c) c.textContent = mc.classList.contains('collapsed') ? '\u25b8' : '\u25be';
    });
  }
}

// `good` picks the colour. NOTE the token is --ok, not --good: --good was never defined,
// so `color: var(--good)` was invalid at computed-value time and fell back to the inherited
// text colour. The effect was that a FAILED calibration went red (--bad exists) while a
// SUCCESSFUL one showed no colour at all -- the confirmation was missing and the failure
// was not, which is the wrong way round and is invisible unless you measure the DOM.
function setScaleState(txt, good) {
  const el = document.getElementById('scaleState');
  if (!el) return;
  el.textContent = txt;
  el.style.color = good === true ? 'var(--ok)' : (good === false ? 'var(--bad)' : '');
}

async function refreshScaleState() {
  if (!currentImage) return;
  try {
    const r = await (await fetch('/api/calibration/' + currentImage)).json();
    if (r.calibrated && r.record) {
      setScaleState(r.record.um_per_px.toPrecision(4) + scalePrecisionText(r.record) + ' \u00b5m/px (' + r.record.source + ')', true);
    } else {
      setScaleState('uncalibrated', null);
    }
  } catch (e) { }
}

async function finishCalibration() {
  const img = calibImage, imgW = calibW;   // pinned at arm time, not read at POST time
  if (!img) { setScaleState('cancelled'); return; }
  const span = Math.abs(calibMarks[1] - calibMarks[0]);
  const label = prompt('Scale bar label in micrometres (e.g. 400 for "400 \u00b5m").\n' +
                       'Marked span: ' + Math.round(span) + ' px', '');
  if (label === null || label.trim() === '') { setScaleState('cancelled'); return; }
  const um = parseFloat(label);
  if (!(um > 0)) { setScaleState('not a number', false); return; }
  // Offer the cross-check. HFW is printed in the same info panel, so the user can read
  // it off; supplying it makes the server refuse a pair that disagrees by >5% instead of
  // storing a calibration that would corrupt every exported length.
  const hfw = prompt('Optional cross-check \u2014 horizontal field width (HFW) in ' +
                     'micrometres, from the same info panel. Leave blank to skip.\n' +
                     'If the two disagree by more than 5% the calibration is refused.', '');
  const body = {mode: 'scale_bar', label_um: um,
                x1: calibMarks[0], x2: calibMarks[1]};
  if (hfw && parseFloat(hfw) > 0) {
    body.hfw_um = parseFloat(hfw);
    body.image_width_px = imgW;
  }
  setScaleState('checking\u2026');
  let res, j;
  try {
    res = await fetch('/api/calibration/' + img,
                      {method: 'POST', headers: {'Content-Type': 'application/json'},
                       body: JSON.stringify(body)});
    j = await res.json().catch(() => ({}));
  } catch (e) {
    // Without this the readout sat on "checking..." forever and the user had no idea
    // whether anything was stored.
    setScaleState('request failed', false);
    alert('Could not reach the server to store the calibration: ' + e.message);
    return;
  }
  if (res.status === 409) {
    // 409 covers every refusal the server makes, not just a bar-vs-HFW disagreement: a span
    // below the minimum comes back 409 too, and labelling that "readings disagree" sends the
    // user hunting for an HFW problem they do not have. Use the server's own reason.
    const disagree = /disagreement/.test(j.error || '');
    setScaleState(disagree ? 'refused \u2014 readings disagree' : 'refused', false);
    alert((j.error || 'the calibration was refused') +
          '\n\nNothing was stored.' +
          (disagree ? ' Re-mark the bar ends, or check the label.' : ''));
    return;
  }
  if (!j.ok) { setScaleState('failed', false); alert(j.error || 'calibration failed'); return; }
  setScaleState(j.record.um_per_px.toPrecision(4) + scalePrecisionText(j.record) + ' \u00b5m/px (scale_bar)', true);
}

document.getElementById('setScaleBtn').addEventListener('click', () => {
  if (!currentImage) { alert('Open an image first.'); return; }
  // Pin the image NOW. finishCalibration blocks on two prompt() dialogs, and currentImage
  // can change under it -- a click in the sidebar, or the arrow keys -- so reading
  // currentImage at POST time could store one image's scale bar as another image's
  // calibration, silently and with full provenance.
  calibArm = true; calibMarks = []; calibImage = currentImage; calibW = nativeW;
  setScaleState('click the LEFT end of the scale bar');
});

document.getElementById('installSamBtn').addEventListener('click', async () => {
  if (!confirm('Install PyTorch and transformers into this app\'s virtualenv?\n\n' +
               'About 2.5 GB and several minutes. It raises measured f1 from 0.715 to 0.776, ' +
               'at ~3 min per image instead of ~40 s.')) return;
  const b = document.getElementById('installSamBtn'); b.disabled = true;
  try {
    const r = await (await fetch('/api/install_sam', {method: 'POST'})).json();
    if (r.already) { setStatus(r.message); b.style.display = 'none'; return; }
    if (!r.ok) throw new Error(r.error || 'could not start');
    const res = await pollJob(r.job, 'Installing SAM');
    setStatus((res && res.message) || 'SAM installed');
    if (res && res.restart_needed) {
      setStatus((res.message || '') + ' Restart the app to load it.');
    }
    await refreshModelInfo();
  } catch (e) { setStatus('Install failed: ' + e.message, true); }
  finally { b.disabled = false; }
});

// surface the button only when SAM is actually missing
(async () => {
  try {
    const i = await (await fetch('/api/pipeline_info')).json();
    // Deliberately NOT offering the install. With USE_SAM=false a fresh clone reports
    // sam_available=false, so this used to reveal an Enable SAM button that downloads
    // ~2.5 GB of torch plus a ~2.4 GB checkpoint and then changes nothing, with no
    // message explaining why. Flip USE_SAM to revive both the stage and this offer.
    if (USE_SAM && !i.sam_available) {
      document.getElementById('installSamBtn').style.display = '';
    }
  } catch (e) { }
})();


// ---- re-apply the current model to every image ----
document.getElementById('reapplyBtn').addEventListener('click', async () => {
  // Ask about SAM explicitly and state the cost. Re-apply used to always run
  // pipeline-only, which silently discarded SAM regions from any image that had
  // them -- a downgrade from f1 0.776 to 0.715 with nothing on screen about it.
  const samOk = USE_SAM;
  let withSam = false;
  if (samOk) {
    withSam = confirm('Re-render every image.\n\n' +
      'OK = include SAM: best accuracy (f1 0.776), about 3 minutes per image.\n' +
      'Cancel = pipeline only: f1 0.715, about 40 seconds per image.\n\n' +
      'With ' + (_imgCache.length || 0) + ' images that is roughly ' +
      Math.round((_imgCache.length || 0) * 3) + ' min with SAM vs ' +
      Math.round((_imgCache.length || 0) * 40 / 60) + ' min without.');
  } else if (!confirm('Re-render every image with the current model (pipeline only)?')) {
    return;
  }
  const b = document.getElementById('reapplyBtn'); b.disabled = true;
  try {
    const r = await (await fetch('/api/reapply', {method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({use_sam: withSam})})).json();
    if (!r.ok) throw new Error(r.error || 'could not start');
    const res = await pollJob(r.job, 'Re-applying the model to every image');
    setStatus(res && res.message ? res.message : 're-applied');
    await loadImageList(true);
    if (currentImage) await loadImage(currentImage);
  } catch (e) { setStatus('Re-apply failed: ' + e.message, true); }
  finally { b.disabled = false; }
});

refreshModelPicker();
refreshModelInfo();
loadImageList();
</script>
</body>
</html>
"""
