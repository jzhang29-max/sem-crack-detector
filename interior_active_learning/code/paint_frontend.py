INDEX_HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Crack Detector</title>
<style>
  /* ============================================================
     Layout: fixed sidebar + main column.
     The previous version put all 17 controls in one flat toolbar row, which
     read as a control panel rather than a tool. This groups them by WHEN you
     reach for them and demotes the fine-tuning (brush size, zoom slider, SAM
     toggle) behind a disclosure row, so the default view shows only what a
     first-time user needs: pick an image, mark, save.
     ============================================================ */
  :root {
    --bg:        #0f1216;
    --bg-panel:  #151a21;
    --bg-raised: #1c232c;
    --bg-hover:  #232c37;
    --line:      #262f3a;
    --line-soft: #1e252e;
    --text:      #e6edf5;
    --text-dim:  #8b9bad;
    --text-faint:#5f6e7f;
    --accent:    #4a9eff;
    --accent-dim:#2b6cb3;
    --crack:     #ff4d4d;
    --notcrack:  #22ccff;
    --erase:     #c77dff;
    --good:      #2ea86b;
    --radius:    7px;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body {
    font: 13px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif;
    background: var(--bg); color: var(--text);
    display: flex; overflow: hidden;
    -webkit-font-smoothing: antialiased;
  }
  button { font: inherit; color: var(--text); background: var(--bg-raised);
           border: 1px solid var(--line); border-radius: var(--radius);
           padding: 6px 11px; cursor: pointer; transition: background .12s, border-color .12s; }
  button:hover:not(:disabled) { background: var(--bg-hover); border-color: #33404e; }
  button:disabled { opacity: .42; cursor: default; }
  button.on { background: var(--accent-dim); border-color: var(--accent); color: #fff; }
  button.primary { background: var(--good); border-color: #35bd7b; color: #fff; font-weight: 550; }
  button.primary:hover:not(:disabled) { background: #35bd7b; }
  input[type=range] { accent-color: var(--accent); }

  /* ---------- sidebar ---------- */
  #side { width: 268px; flex: 0 0 268px; background: var(--bg-panel);
          border-right: 1px solid var(--line); display: flex; flex-direction: column;
          overflow: hidden; }
  #brand { padding: 15px 16px 13px; border-bottom: 1px solid var(--line-soft); }
  #brand h1 { margin: 0; font-size: 14.5px; font-weight: 600; letter-spacing: .2px; }
  #brand p { margin: 3px 0 0; font-size: 11.5px; color: var(--text-faint); }
  .sect { padding: 13px 16px; border-bottom: 1px solid var(--line-soft); }
  .sect h2 { margin: 0 0 9px; font-size: 10px; font-weight: 650; letter-spacing: .9px;
             text-transform: uppercase; color: var(--text-faint); }

  #dropTarget { border: 1.5px dashed #31404f; border-radius: var(--radius);
                padding: 17px 12px; text-align: center; cursor: pointer;
                transition: border-color .15s, background .15s; }
  #dropTarget:hover { border-color: var(--accent); background: rgba(74,158,255,.06); }
  #dropTarget .big { font-size: 12.5px; font-weight: 550; }
  #dropTarget .sub { font-size: 11px; color: var(--text-faint); margin-top: 3px; }

  #modelCard { font-size: 11.5px; color: var(--text-dim); line-height: 1.6; }
  #modelCard b { color: var(--text); font-weight: 550; }
  #modelCard .row { display: flex; justify-content: space-between; gap: 8px; }
  #retrainBtn { width: 100%; margin-top: 10px; }

  #listWrap { flex: 1 1 auto; overflow-y: auto; padding: 11px 10px 16px; }
  #listWrap h2 { padding: 0 6px; margin: 0 0 8px; font-size: 10px; font-weight: 650;
                 letter-spacing: .9px; text-transform: uppercase; color: var(--text-faint); }
  #imgSearch { width: 100%; margin: 0 0 8px; padding: 6px 9px; font: inherit;
               font-size: 12px; color: var(--text); background: var(--bg);
               border: 1px solid var(--line); border-radius: var(--radius); }
  #imgSearch::placeholder { color: var(--text-faint); }
  .item { padding: 7px 9px; border-radius: var(--radius); cursor: pointer;
          display: flex; align-items: baseline; gap: 7px; }
  .item:hover { background: var(--bg-raised); }
  .item.sel { background: var(--accent-dim); }
  .item .nm { flex: 1 1 auto; font-size: 12px; overflow: hidden;
              text-overflow: ellipsis; white-space: nowrap; }
  .item .ct { font-size: 10.5px; color: var(--text-faint); font-variant-numeric: tabular-nums; }
  .item.sel .ct { color: #cfe4ff; }
  .item .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--good);
               flex: 0 0 auto; }

  /* ---------- main ---------- */
  #main { flex: 1 1 auto; display: flex; flex-direction: column; min-width: 0; }
  #head { padding: 11px 16px; border-bottom: 1px solid var(--line);
          display: flex; align-items: center; gap: 12px; background: var(--bg-panel); }
  #curName { font-size: 13.5px; font-weight: 600; }
  #curMeta { font-size: 11.5px; color: var(--text-faint); }
  #head .spacer { flex: 1 1 auto; }

  #bar { display: flex; align-items: center; gap: 4px; padding: 8px 14px;
         border-bottom: 1px solid var(--line); background: var(--bg-panel);
         flex-wrap: wrap; }
  .grp { display: flex; align-items: center; gap: 4px; }
  .grp > .lbl { font-size: 9.5px; font-weight: 650; letter-spacing: .8px;
                text-transform: uppercase; color: var(--text-faint); margin-right: 3px; }
  .vsep { width: 1px; height: 20px; background: var(--line); margin: 0 7px; }

  .swatch { width: 26px; height: 26px; border-radius: 50%; cursor: pointer;
            border: 2px solid transparent; display: inline-flex; align-items: center;
            justify-content: center; font-size: 14px; color: #fff;
            transition: box-shadow .12s, transform .1s; }
  .swatch:hover { transform: scale(1.07); }
  .swatch.red    { background: var(--crack); }
  .swatch.cyan   { background: var(--notcrack); }
  .swatch.erase  { background: var(--erase); }
  .swatch.selected { box-shadow: 0 0 0 2px var(--bg-panel), 0 0 0 4px currentColor; }
  .swatch.red.selected    { color: var(--crack); }
  .swatch.cyan.selected   { color: var(--notcrack); }
  .swatch.erase.selected  { color: var(--erase); }

  #advToggle { font-size: 11.5px; color: var(--text-dim); background: none;
               border: none; padding: 4px 6px; }
  #advToggle:hover { color: var(--text); background: none; }
  #adv { display: none; align-items: center; gap: 14px; padding: 7px 16px;
         border-bottom: 1px solid var(--line); background: var(--bg);
         font-size: 11.5px; color: var(--text-dim); flex-wrap: wrap; }
  #adv.open { display: flex; }
  #adv label { display: flex; align-items: center; gap: 7px; }

  #canvasWrap { flex: 1 1 auto; min-height: 0; overflow: auto; background: #080a0d;
                display: flex; align-items: flex-start; justify-content: flex-start; }
  #canvasInner { position: relative; margin: 22px; flex: 0 0 auto; }
  canvas { position: absolute; top: 0; left: 0; image-rendering: pixelated; }
  #baseCanvas { z-index: 1; }
  #paintCanvas { z-index: 2; cursor: crosshair; }

  #foot { padding: 8px 16px; border-top: 1px solid var(--line); background: var(--bg-panel);
          display: flex; align-items: center; gap: 12px; font-size: 12px; min-height: 37px; }
  #status { color: #7fd4a3; flex: 1 1 auto; }
  #status.error { color: #ff7b7b; }
  #modelInfo { font-size: 10.5px; color: var(--text-faint); }

  /* ---------- overlays ---------- */
  #dropZone { position: fixed; inset: 0; z-index: 900; display: none;
              background: rgba(8,11,15,.86); backdrop-filter: blur(3px);
              align-items: center; justify-content: center; }
  #dropZone.active { display: flex; }
  #dropInner { border: 2.5px dashed var(--accent); border-radius: 16px;
               padding: 46px 66px; text-align: center; font-size: 19px;
               background: rgba(21,26,33,.92); }
  #dropInner .hint { margin-top: 9px; font-size: 12.5px; color: var(--text-dim); }

  #jobBar { display: none; position: fixed; left: 268px; right: 0; bottom: 0; z-index: 950;
            background: var(--bg-panel); border-top: 1px solid var(--accent-dim);
            padding: 9px 16px; }
  #jobLabel { font-size: 12.5px; font-weight: 550; }
  #jobTrack { height: 4px; background: #1e2833; border-radius: 2px; margin: 7px 0 4px;
              overflow: hidden; }
  #jobFill { height: 4px; width: 0%; background: linear-gradient(90deg,var(--accent-dim),var(--accent));
             border-radius: 2px; transition: width .3s ease; }
  #jobNote { color: var(--text-faint); font-size: 11px; }

  #expMenu { display: none; position: absolute; z-index: 960; background: var(--bg-raised);
             border: 1px solid var(--line); border-radius: var(--radius);
             padding: 5px; box-shadow: 0 10px 26px rgba(0,0,0,.5); min-width: 196px; }
  #expMenu.open { display: block; }
  #expMenu button { display: block; width: 100%; text-align: left; background: none;
                    border: none; padding: 7px 10px; border-radius: 5px; font-size: 12px; }
  #expMenu button:hover { background: var(--bg-hover); }
  #expMenu .mdiv { height: 1px; background: var(--line); margin: 4px 2px; }

  #help { position: fixed; right: 14px; bottom: 46px; z-index: 940; display: none;
          background: var(--bg-raised); border: 1px solid var(--line);
          border-radius: var(--radius); padding: 12px 14px; font-size: 11.5px;
          color: var(--text-dim); box-shadow: 0 10px 26px rgba(0,0,0,.5); }
  #help.open { display: block; }
  #help b { color: var(--text); }
  #help kbd { background: var(--bg); border: 1px solid var(--line); border-radius: 4px;
              padding: 1px 5px; font-family: ui-monospace, monospace; font-size: 10.5px;
              color: var(--text); }
  #help table { border-collapse: collapse; }
  #help td { padding: 3px 8px 3px 0; }
</style>
</head>
<body>

<aside id="side">
  <div id="brand">
    <h1>Crack Detector</h1>
    <p>Detect &middot; correct &middot; retrain</p>
  </div>

  <div class="sect">
    <h2>Add images</h2>
    <div id="dropTarget">
      <div class="big">Drop images here</div>
      <div class="sub">or click to browse &middot; TIF PNG JPG</div>
    </div>
  </div>

  <div class="sect">
    <h2>Model</h2>
    <div id="modelCard">loading&hellip;</div>
    <button id="retrainBtn" title="Rebuild training data from every correction you have made, retrain, and re-render all images. The new model is only deployed if it scores at least as well on held-out data.">Retrain &amp; re-overlay</button>
  </div>

  <div id="listWrap">
    <h2>Images</h2>
    <input id="imgSearch" type="search" placeholder="Filter&hellip;" autocomplete="off">
    <div id="imageList"></div>
    <!-- kept as the single source of truth for the existing logic; the visual
         list above just drives it, so no behaviour changed with the redesign -->
    <select id="imageSelect" style="display:none"></select>
  </div>
</aside>

<div id="main">
  <div id="head">
    <span id="curName">No image selected</span>
    <span id="curMeta"></span>
    <span class="spacer"></span>
    <button id="saveBtn" title="Save your strokes without re-running detection">Save</button>
    <button id="ingestBtn" class="primary" title="Save and fold your corrections into the model's candidate set">Save &amp; Ingest</button>
  </div>

  <div id="bar">
    <div class="grp">
      <span class="lbl">Mark</span>
      <span class="swatch red selected" id="swatchRed" title="Crack (1)"></span>
      <span class="swatch cyan" id="swatchCyan" title="Not a crack (2)"></span>
      <span class="swatch erase" id="swatchErase" title="Remove from candidacy (3)">&times;</span>
    </div>
    <div class="vsep"></div>
    <div class="grp">
      <span class="lbl">Mode</span>
      <button id="bucketBtn" title="Brush: paint pixel by pixel. Whole region: one click sets an entire connected region -- essential for large ones.">Brush</button>
    </div>
    <div class="vsep"></div>
    <div class="grp">
      <span class="lbl">View</span>
      <button id="zoomOut" title="Zoom out">&minus;</button>
      <span id="zoomLabel" style="min-width:44px;text-align:center;font-variant-numeric:tabular-nums;color:var(--text-dim)">100%</span>
      <button id="zoomIn" title="Zoom in">+</button>
      <button id="fitBtn" title="Fit to window (F)">Fit</button>
    </div>
    <div class="vsep"></div>
    <div class="grp">
      <span class="lbl">Edit</span>
      <button id="undoBtn" title="Undo (&#8984;Z)">Undo</button>
      <button id="clearBtn" title="Discard all unsaved strokes on this image">Clear</button>
    </div>
    <div class="vsep"></div>
    <div class="grp">
      <button id="exportBtn" title="Download masks, overlays and measurements">Export &#9662;</button>
    </div>
    <span class="spacer" style="flex:1 1 auto"></span>
    <button id="advToggle" title="Brush size, detection options">Options &#9662;</button>
    <button id="helpBtn" title="Keyboard shortcuts">?</button>
  </div>

  <div id="adv">
    <label>Brush
      <input type="range" id="brushSize" min="2" max="120" value="18">
      <span id="brushSizeLabel" style="min-width:38px;font-variant-numeric:tabular-nums">18px</span>
    </label>
    <label><input type="checkbox" id="useSam" checked>
      Use SAM on new images
      <span style="color:var(--text-faint)">(slower, f1 0.776 vs 0.715)</span>
    </label>
    <label style="display:none">
      <input type="range" id="zoom" min="10" max="800" value="100">
    </label>
  </div>

  <div id="canvasWrap">
    <div id="canvasInner">
      <canvas id="baseCanvas"></canvas>
      <canvas id="paintCanvas"></canvas>
    </div>
  </div>

  <div id="foot">
    <span id="status">Ready</span>
    <span id="modelInfo"></span>
  </div>
</div>

<div id="expMenu">
  <button id="dlMask">B&amp;W mask <span style="color:var(--text-faint)">&middot; crack = black</span></button>
  <button id="dlOverlay">Overlay <span style="color:var(--text-faint)">&middot; burned in</span></button>
  <button id="dlCsv">Region measurements <span style="color:var(--text-faint)">&middot; CSV</span></button>
  <div class="mdiv"></div>
  <button id="dlAll">All images &rarr; .zip</button>
</div>

<div id="help">
  <table>
    <tr><td><kbd>&#8984;Z</kbd> / <kbd>Ctrl Z</kbd></td><td>Undo</td></tr>
    <tr><td><kbd>1</kbd> <kbd>2</kbd> <kbd>3</kbd></td><td>Crack / not-crack / erase</td></tr>
    <tr><td><kbd>F</kbd></td><td>Fit to window</td></tr>
    <tr><td colspan="2" style="padding-top:7px;color:var(--text-faint)">
      Red = crack &middot; Cyan = not a crack<br>Corrections always override the model.</td></tr>
  </table>
</div>

<div id="dropZone">
  <div id="dropInner">
    <strong>Drop SEM images</strong>
    <div class="hint">the current model is applied automatically</div>
  </div>
</div>

<div id="jobBar">
  <div id="jobLabel">Working&hellip;</div>
  <div id="jobTrack"><div id="jobFill"></div></div>
  <div id="jobNote"></div>
</div>

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
  pushUndo();
}

function pushUndo() {
  undoStack.push(paintCtx.getImageData(0, 0, nativeW, nativeH));
  if (undoStack.length > 25) undoStack.shift();
}

function undo() {
  if (undoStack.length <= 1) return;
  undoStack.pop();
  const prev = undoStack[undoStack.length - 1];
  paintCtx.putImageData(prev, 0, 0);
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
  if (drawing) { drawing = false; pushUndo(); }
});
paintCanvas.addEventListener('mouseleave', () => {
  if (drawing) { drawing = false; pushUndo(); }
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
    paintCtx.clearRect(0, 0, nativeW, nativeH);
    pushUndo();
  }
});
document.getElementById('imageSelect').addEventListener('change', (e) => {
  currentImage = e.target.value;
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

document.getElementById('saveBtn').addEventListener('click', async () => {
  try {
    setStatus('Saving...');
    await savePaint();
    setStatus('Saved.');
  } catch (err) {
    setStatus('Error: ' + err.message, true);
  }
});

document.getElementById('ingestBtn').addEventListener('click', async () => {
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

function showJob(label) {
  document.getElementById('jobBar').style.display = 'block';
  document.getElementById('jobLabel').textContent = label;
  document.getElementById('jobFill').style.width = '0%';
  document.getElementById('jobNote').textContent = '';
}
function hideJob() { document.getElementById('jobBar').style.display = 'none'; }

async function pollJob(jobId, label) {
  showJob(label);
  while (true) {
    await new Promise(r => setTimeout(r, 1200));
    let j;
    try { j = await (await fetch('/api/job/' + jobId)).json(); }
    catch (e) { continue; }                       // transient: keep polling
    if (!j.ok) { hideJob(); throw new Error(j.error || 'job vanished'); }
    document.getElementById('jobFill').style.width = Math.round((j.frac || 0) * 100) + '%';
    document.getElementById('jobNote').textContent = (j.stage || '') + (j.note ? ' — ' + j.note : '');
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
    const nm = document.createElement('span');
    nm.className = 'nm'; nm.textContent = info.name; nm.title = info.name;
    const ct = document.createElement('span');
    ct.className = 'ct';
    ct.textContent = (info.n_crack != null && info.n_candidates)
      ? info.n_crack + '/' + info.n_candidates
      : (info.n_candidates ? info.n_candidates : '—');
    ct.title = 'crack regions / candidates';
    el.appendChild(nm); el.appendChild(ct);
    el.addEventListener('click', () => {
      const sel = document.getElementById('imageSelect');
      sel.value = info.name;
      sel.dispatchEvent(new Event('change', {bubbles: true}));
    });
    box.appendChild(el);
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
