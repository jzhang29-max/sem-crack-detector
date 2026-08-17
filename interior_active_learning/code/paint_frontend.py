INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SEM Crack Detection</title>
<style>
  /* ============================================================
     Visual language deliberately matched to the sibling TXM app
     (app/static/index.html) so the two tools feel like one family: same
     palette, type scale, 330px sidebar, full-width image rows with a blue
     left-edge marker on the selection, 10px letter-spaced group labels, and a
     3px progress strip rather than a boxed progress panel.

     What is NOT copied is TXM's control density. It puts brush, zoom,
     threshold and post-processing inline, which is the clutter this app was
     just asked to remove -- so brush size and the SAM toggle stay behind
     Options, and downloads stay in one menu.
     ============================================================ */
  :root { --bg:#14151a; --panel:#1c1e26; --line:#2c2f3a; --ink:#e8e8ea; --dim:#9a9ba4;
          --accent:#3b82f6; --ok:#22a06b; --warn:#d97706; --bad:#dc2626;
          --crack:#ff4d4d; --notcrack:#22ccff; --erase:#c77dff; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.5 -apple-system,
         BlinkMacSystemFont,"Segoe UI",sans-serif; height:100vh; display:flex; overflow:hidden; }
  #side { width:330px; min-width:330px; background:var(--panel); border-right:1px solid var(--line);
          display:flex; flex-direction:column; overflow:hidden; }
  #main { flex:1; display:flex; flex-direction:column; overflow:hidden; min-width:0; }
  h1 { font-size:15px; margin:0; padding:14px 16px; border-bottom:1px solid var(--line);
       font-weight:600; }
  .sec { padding:12px 16px; border-bottom:1px solid var(--line); }
  .sec h2 { font-size:11px; text-transform:uppercase; letter-spacing:.08em; color:var(--dim);
            margin:0 0 8px; font-weight:600; }
  button { background:#2a2d38; color:var(--ink); border:1px solid var(--line); border-radius:6px;
           padding:7px 11px; font:inherit; font-size:13px; cursor:pointer; }
  button:hover:not(:disabled) { background:#343845; }
  button:disabled { opacity:.45; cursor:default; }
  button.pri { background:var(--accent); border-color:var(--accent); color:#fff; }
  button.ok  { background:var(--ok); border-color:var(--ok); color:#fff; }
  button.on  { background:var(--bad); border-color:var(--bad); color:#fff; }

  #dropTarget { margin:12px 16px; padding:22px 12px; border:2px dashed var(--line);
                border-radius:10px; text-align:center; color:var(--dim); cursor:pointer; }
  #dropTarget:hover, #dropTarget.hot { border-color:var(--accent); color:var(--ink);
                                       background:#1a2436; }
  #dropTarget .sub { font-size:11px; }

  .kv { font-size:11px; color:var(--dim); line-height:1.7; }
  .kv b { color:var(--ink); font-weight:500; }
  .kv .row { display:flex; justify-content:space-between; gap:8px; }
  #retrainBtn { width:100%; margin-top:9px; font-weight:600; }

  #imgSearch { width:100%; padding:6px 9px; font:inherit; font-size:12px; color:var(--ink);
               background:var(--bg); border:1px solid var(--line); border-radius:6px; }
  #imgSearch::placeholder { color:var(--dim); }
  #imageList { flex:1; overflow-y:auto; }
  .item { padding:9px 16px; border-bottom:1px solid #23252e; cursor:pointer; font-size:12px; }
  .item:hover { background:#22242d; }
  .item.sel { background:#1e2a3f; border-left:3px solid var(--accent); padding-left:13px; }
  .item .nm { white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .item .mt { color:var(--dim); font-size:11px; margin-top:2px; }

  #bar { display:flex; gap:8px; align-items:center; padding:10px 14px;
         border-bottom:1px solid var(--line); flex-wrap:wrap; background:var(--panel); }
  .grp { font-size:10px; letter-spacing:.09em; color:var(--dim); font-weight:600; min-width:52px; }
  .vsep { width:1px; height:20px; background:var(--line); }
  label { font-size:12px; color:var(--dim); display:inline-flex; align-items:center; gap:6px; }
  input[type=range] { width:96px; accent-color:var(--accent); }

  .swatch { width:24px; height:24px; border-radius:50%; cursor:pointer; border:2px solid transparent;
            display:inline-flex; align-items:center; justify-content:center; font-size:13px; color:#fff; }
  .swatch.red { background:var(--crack); } .swatch.cyan { background:var(--notcrack); }
  .swatch.erase { background:var(--erase); }
  .swatch.selected { border-color:var(--ink); box-shadow:0 0 0 2px var(--panel); }

  #adv { display:none; align-items:center; gap:14px; padding:9px 14px;
         border-bottom:1px solid var(--line); background:var(--bg); flex-wrap:wrap; }
  #adv.open { display:flex; }

  #prog { height:3px; background:var(--accent); width:0; transition:width .2s; }
  #canvasWrap { flex:1; overflow:auto; background:#0e0f13; position:relative; min-height:0; }
  #canvasInner { position:relative; margin:20px; }
  canvas { position:absolute; top:0; left:0; display:block; image-rendering:pixelated; }
  #paintCanvas { cursor:crosshair; }
  #foot { padding:7px 14px; font-size:12px; color:var(--dim); border-top:1px solid var(--line);
          background:var(--panel); min-height:30px; display:flex; align-items:center; gap:12px; }
  #status { flex:1; } #status.error { color:var(--bad); }

  #dropZone { position:fixed; inset:0; z-index:900; display:none; background:rgba(14,15,19,.88);
              align-items:center; justify-content:center; }
  #dropZone.active { display:flex; }
  #dropInner { border:2px dashed var(--accent); border-radius:12px; padding:44px 62px;
               text-align:center; font-size:18px; background:#1a2436; }
  #dropInner .hint { margin-top:8px; font-size:12px; color:var(--dim); }
  #jobBar { display:none; }
  #expMenu { display:none; position:absolute; z-index:960; background:var(--panel);
             border:1px solid var(--line); border-radius:8px; padding:5px;
             box-shadow:0 12px 30px rgba(0,0,0,.55); min-width:210px; }
  #expMenu.open { display:block; }
  #expMenu button { display:block; width:100%; text-align:left; background:none; border:none;
                    padding:8px 10px; border-radius:5px; font-size:12.5px; }
  #expMenu button:hover { background:#22242d; }
  #expMenu .mdiv { height:1px; background:var(--line); margin:4px 2px; }
  #help { position:fixed; right:14px; bottom:42px; z-index:940; display:none; background:var(--panel);
          border:1px solid var(--line); border-radius:8px; padding:12px 14px; font-size:12px;
          color:var(--dim); box-shadow:0 12px 30px rgba(0,0,0,.55); }
  #help.open { display:block; }
  #help kbd { background:var(--bg); border:1px solid var(--line); border-radius:4px; padding:1px 5px;
              font-family:ui-monospace,monospace; font-size:11px; color:var(--ink); }
  #help td { padding:3px 8px 3px 0; }
</style>
</head>
<body>

<div id="side">
  <h1>SEM Crack Detection</h1>
  <div id="dropTarget">Drag SEM images here<br>
    <span class="sub">or click to choose &middot; .tif .tiff .png .jpg</span></div>

  <div class="sec">
    <h2>Model</h2>
    <div class="kv" id="modelCard">loading&hellip;</div>
    <button id="retrainBtn" class="ok" title="Rebuild training data from every correction, retrain, and re-render. The new model is deployed only if it scores at least as well on held-out data.">&#9635; Retrain on my corrections</button>
  </div>

  <div class="sec">
    <h2>Images</h2>
    <input id="imgSearch" type="search" placeholder="Filter&hellip;" autocomplete="off">
  </div>
  <div id="imageList"></div>
  <select id="imageSelect" style="display:none"></select>
</div>

<div id="main">
  <div id="bar" style="border-bottom:none;padding-bottom:0">
    <span class="grp">IMAGE</span>
    <span id="curName" style="font-size:13.5px;font-weight:600;color:var(--ink)">No image selected</span>
    <span id="curMeta" style="font-size:12px;color:var(--dim)"></span>
    <span style="flex:1"></span>
    <span id="saveState" style="font-size:12px;color:var(--dim)"></span>
    <button id="retryBtn" class="pri" style="display:none" title="A save failed -- your marks are still here.">Retry save</button>
    <button id="advToggle" style="background:none;border:none;color:var(--dim);font-size:12px">Options &#9662;</button>
    <button id="helpBtn" style="background:none;border:none;color:var(--dim)">?</button>
  </div>

  <div id="bar">
    <span class="grp">PAINT</span>
    <span class="swatch red selected" id="swatchRed" title="Crack (1)"></span>
    <span class="swatch cyan" id="swatchCyan" title="Not a crack (2)"></span>
    <span class="swatch erase" id="swatchErase" title="Remove from candidacy (3)">&times;</span>
    <button id="bucketBtn" title="Brush paints pixel by pixel. Whole region sets an entire connected region in one click.">Brush</button>
    <span class="vsep"></span>
    <span class="grp">VIEW</span>
    <button id="zoomOut">&minus;</button>
    <span id="zoomLabel" style="min-width:42px;text-align:center;font-size:12px;color:var(--dim);font-variant-numeric:tabular-nums">100%</span>
    <button id="zoomIn">+</button>
    <button id="fitBtn" title="Fit to window (F)">Fit</button>
    <span class="vsep"></span>
    <span class="grp">EDIT</span>
    <button id="undoBtn">Undo <span style="opacity:.6">&#8984;Z</span></button>
    <button id="clearBtn" title="Erase every unsaved stroke on this image.">Reset</button>
    <span class="vsep"></span>
    <button id="exportBtn" class="pri">Download &#9662;</button>
  </div>

  <div id="adv">
    <label>Brush <input type="range" id="brushSize" min="2" max="120" value="18">
      <span id="brushSizeLabel" style="font-variant-numeric:tabular-nums">18px</span></label>
    <label><input type="checkbox" id="useSam" checked> Use SAM on new images
      <span style="opacity:.6">(slower &middot; f1 0.776 vs 0.715)</span></label>
    <label style="display:none"><input type="range" id="zoom" min="10" max="800" value="100"></label>
  </div>

  <div id="prog"></div>

  <div id="canvasWrap">
    <div id="canvasInner">
      <canvas id="baseCanvas"></canvas>
      <canvas id="paintCanvas"></canvas>
    </div>
  </div>

  <div id="foot">
    <span id="status">Drag some images in to begin.</span>
    <span id="modelInfo" style="font-size:11px"></span>
  </div>
</div>

<div id="expMenu">
  <button id="dlMask">B&amp;W mask <span style="opacity:.55">&middot; crack = black</span></button>
  <button id="dlOverlay">Overlay <span style="opacity:.55">&middot; burned in</span></button>
  <button id="dlCsv">Region measurements <span style="opacity:.55">&middot; CSV</span></button>
  <div class="mdiv"></div>
  <button id="dlAll">All images &rarr; .zip</button>
</div>

<div id="help">
  <table>
    <tr><td><kbd>&#8984;Z</kbd> / <kbd>Ctrl Z</kbd></td><td>Undo</td></tr>
    <tr><td><kbd>1</kbd> <kbd>2</kbd> <kbd>3</kbd></td><td>Crack / not-crack / erase</td></tr>
    <tr><td><kbd>F</kbd></td><td>Fit to window</td></tr>
    <tr><td colspan="2" style="padding-top:7px">Red = crack &middot; Cyan = not a crack.<br>
      Marks save themselves. Corrections override the model.</td></tr>
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

async function loadImage(name) {
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
}

paintCanvas.addEventListener('mousedown', (e) => {
  if (tool === 'bucket') {
    const [x, y] = canvasCoords(e);
    flipRegion(x, y);
    return;
  }
  drawing = true;
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
  if (drawing) { drawing = false; pushUndo(); markDirty(); }
});
paintCanvas.addEventListener('mouseleave', () => {
  if (drawing) { drawing = false; pushUndo(); markDirty(); }
});

function selectColor(id, color) {
  currentColor = color;
  setBucketActive(false);
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
    undo();
    setStatus(undoStack.length > 1 ? 'undo' : 'nothing left to undo');
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
  }
});
document.getElementById('imageSelect').addEventListener('change', async (e) => {
  // flush pending marks before leaving, so switching images cannot lose work
  if (savePending || saveInFlight) { setStatus('Saving before switching\u2026'); await commitNow(true); }
  currentImage = e.target.value;
  setSaveState('idle');
  loadImage(currentImage);
});

async function savePaint() {
  const dataURL = paintCanvas.toDataURL('image/png');
  const res = await fetch('/api/save/' + currentImage, {
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
const AUTOSAVE_IDLE_MS = 1100;

function markDirty() {
  savePending = true;
  setSaveState('pending');
  clearTimeout(saveTimer);
  saveTimer = setTimeout(commitNow, AUTOSAVE_IDLE_MS);
}

function setSaveState(state) {
  const el = document.getElementById('saveState');
  if (!el) return;
  const map = {
    idle:    ['', ''],
    pending: ['Unsaved changes', 'var(--text-dim)'],
    saving:  ['Saving\u2026', 'var(--text-dim)'],
    saved:   ['All changes saved', '#7fd4a3'],
    error:   ['Save failed \u2014 use Retry', '#ff7b7b'],
  };
  const [txt, col] = map[state] || map.idle;
  el.textContent = txt;
  el.style.color = col;
  const retry = document.getElementById('retryBtn');
  if (retry) retry.style.display = (state === 'error') ? '' : 'none';
}

async function commitNow(silent) {
  clearTimeout(saveTimer);
  if (!currentImage || !savePending) return;
  if (saveInFlight) { saveTimer = setTimeout(commitNow, 400); return; }  // coalesce
  saveInFlight = true; savePending = false;
  setSaveState('saving');
  try {
    await savePaint();
    const res = await fetch('/api/ingest/' + currentImage, { method: 'POST' });
    const result = await res.json();
    if (!result.ok) throw new Error(result.error || 'ingest failed');
    // reload so corrected/erased pixels show their real committed appearance
    // rather than the transient marker colour
    await loadImage(currentImage);
    setSaveState('saved');
    if (!silent) setStatus(result.message || 'Saved.');
    loadImageList(true);
  } catch (err) {
    savePending = true;            // keep the work; let the user retry
    setSaveState('error');
    setStatus('Save failed: ' + err.message, true);
  } finally {
    saveInFlight = false;
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
  const useSam = document.getElementById('useSam').checked;
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
               're-render all images?\\n\\nThe new model is only deployed if it scores at ' +
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
    document.getElementById('modelInfo').textContent = m
      ? ('model: ' + m.family + '  ·  threshold ' + m.threshold.toFixed(3) +
         '  ·  trained on ' + m.n_train + ' reviewed regions from ' + m.n_images +
         ' images' + (i.sam_available ? '  ·  SAM available' : '  ·  SAM not installed'))
      : 'no model loaded';
    if (!i.sam_available) document.getElementById('useSam').checked = false;
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
  box.innerHTML = '';
  let shown = 0;
  for (const info of _imgCache) {
    if (q && !info.name.toLowerCase().includes(q)) continue;
    shown++;
    const el = document.createElement('div');
    el.className = 'item' + (info.name === currentImage ? ' sel' : '');
    const nm = document.createElement('div');
    nm.className = 'nm'; nm.textContent = info.name; nm.title = info.name;
    const mt = document.createElement('div');
    mt.className = 'mt';
    mt.textContent = (info.n_crack != null && info.n_candidates)
      ? info.n_crack + ' crack \u00b7 ' + info.n_candidates + ' candidates'
      : (info.n_candidates ? info.n_candidates + ' candidates' : 'not processed yet');
    el.appendChild(nm); el.appendChild(mt);
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
    e.style.cssText = 'padding:10px;color:var(--text-faint);font-size:11.5px';
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
document.getElementById('zoomIn').addEventListener('click', () => bumpZoom(1.25));
document.getElementById('zoomOut').addEventListener('click', () => bumpZoom(0.8));

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
    if (!m) { card.textContent = 'No model loaded.'; return; }
    card.innerHTML =
      '<div class="row"><span>Type</span><b>' + m.family + '</b></div>' +
      '<div class="row"><span>Threshold</span><b>' + m.threshold.toFixed(3) + '</b></div>' +
      '<div class="row"><span>Trained on</span><b>' + m.n_train.toLocaleString() +
        ' regions</b></div>' +
      '<div class="row"><span>From</span><b>' + m.n_images + ' images</b></div>' +
      '<div class="row"><span>SAM</span><b style="color:' +
        (i.sam_available ? 'var(--good)' : 'var(--text-faint)') + '">' +
        (i.sam_available ? 'available' : 'not installed') + '</b></div>';
  } catch (e) { }
};

refreshModelInfo();
loadImageList();
</script>
</body>
</html>
"""
