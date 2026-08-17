# Crack Detection Pipeline — Complete Technical Reference

This document walks through **every step** of the system: what it does, the actual code, why that
approach was chosen over the alternatives, and — for every model used — what it is, why it was
picked, and the real numbers from every alternative that was tested against it.

The system has two halves:
- **Part 1 (Steps A–F):** the base, automated detection pipeline. Always Python.
- **Part 2 (Steps G–H):** the manual-correction and retraining loop. Python backend + JavaScript
  frontend for the paint app.

---

## PART 1 — Base Detection Pipeline

### Step A — Preprocessing (load + auto-crop)

**What it does:** Loads the raw 16-bit TIFF, rescales it to 8-bit, and auto-detects/removes any
burned-in instrument info bar or circular aperture vignette.

```python
def load_as_uint8(image_path, low_pct=1.0, high_pct=99.5):
    img = tifffile.imread(image_path)
    if img.ndim == 3:
        img = img[..., 0]
    lo, hi = np.percentile(img, [low_pct, high_pct])
    img8 = exposure.rescale_intensity(
        img.astype(np.float32), in_range=(lo, hi), out_range=(0, 255)
    ).astype(np.uint8)
    return img8
```

The crop-detection (`find_field_of_view`) works by finding a databar via a row-brightness/std-dev
drop-off, then — if there's a round vignette rather than a clean rectangle — computing the largest
square that fits inside the circular bright field of view:

```python
cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
radius = max(y1 - y0, x1 - x0) / 2
half_side = radius / math.sqrt(2) * (1 - shrink_frac)
```

**Why this step:** Raw exports often have UI chrome (scale bar, detector name, voltage) burned
into the image and/or a dark circular vignette around the edges. Both would otherwise get
misread by later steps as one giant "dark region" — i.e. a false crack the size of the image edge.

**Why a percentile stretch, not min/max:** A single hot pixel or a genuinely pure-black crack pixel
can dominate a min/max rescale and compress everything else into a narrow band. Clipping the
top/bottom ~1% before rescaling is far more robust to those outliers.

---

### Step B — Illumination Correction (background flattening)

```python
def flatten_background(img8, sigma=40, low_pct=0.5, high_pct=99.5):
    background = filters.gaussian(img8, sigma=sigma, preserve_range=True)
    flat = img8.astype(np.float32) - background
    lo, hi = np.percentile(flat, [low_pct, high_pct])
    flat = exposure.rescale_intensity(flat, in_range=(lo, hi), out_range=(0, 255)).astype(np.uint8)
    return flat
```

**What it does:** Blurs the image heavily (sigma=40px Gaussian) to estimate *just the lighting*,
then subtracts that out. This is a classic **high-pass filter**.

**Why:** Microscope illumination isn't perfectly even. Without this, one side of the frame being
naturally a bit darker gets misread as "this whole side is one giant crack."

**The trade-off this creates (important — this is *why* Step H exists):** A high-pass filter, by
definition, removes anything broad and gradual — including a genuinely wide, gradual damage zone
around a real crack. Confirmed by direct measurement on one image: a true void's raw pixel value
of 0–4 gets rescaled to ~122/255 (indistinguishable from generic gray) after this flattening. This
single fact is the reason the whole second candidate-generation system in Part 2 exists.

---

### Step C — Dark-Region Segmentation

```python
def segment_dark_regions(flat, denoise_sigma=1.0, mad_k=5.0, img8=None, absolute_dark_thresh=10):
    smooth = filters.gaussian(flat, sigma=denoise_sigma, preserve_range=True)
    otsu_thresh = filters.threshold_otsu(smooth)
    median = np.median(smooth)
    mad = np.median(np.abs(smooth - median))
    robust_thresh = median - mad_k * mad * 1.4826
    thresh = min(otsu_thresh, robust_thresh)
    relative_mask = smooth < thresh
    if img8 is None:
        return relative_mask
    absolute_mask = img8 < absolute_dark_thresh
    return relative_mask | absolute_mask
```

**What it does:** Marks every pixel darker than a threshold, using **two different thresholds
combined with OR**:
1. **Otsu's method** — automatically finds the threshold that best splits a *bimodal* histogram
   (assumes two clear peaks: background and dark stuff).
2. **A robust median/MAD outlier test** — `median - 5 × MAD × 1.4826` (the 1.4826 constant makes
   MAD comparable to a standard deviation for normal-ish data). Doesn't assume bimodality at all.
3. **An absolute darkness test** on the *un-flattened* image (`img8 < 10`) — catches wide voids
   that Step B's flattening has already washed out to mid-gray.

Whichever of Otsu or the robust threshold is *lower* gets used (`min(otsu_thresh, robust_thresh)`).

**Why combine three tests instead of one:** Otsu silently breaks on low-contrast/noisy captures
(confirmed: on this dataset's noisier ETD-detector images, Otsu's bimodality assumption fails and
it flags over 50% of the image as "dark"). The robust MAD-based threshold is the safety net for
those cases. The absolute test is the safety net for wide voids Step B's own math structurally
erases. None of the three alone is reliable across this whole dataset; together, each one's blind
spot is covered by one of the others.

---

### Step D — Feature Extraction

**Two computations feed into this:**

**1. Frangi vesselness filter** — repurposing a medical-imaging algorithm built for blood vessels:

```python
def compute_vesselness(flat, sigma_min=1, sigma_max=6):
    ves = filters.frangi(
        flat.astype(np.float64) / 255.0,
        sigmas=range(sigma_min, sigma_max, 1),
        black_ridges=True,
    )
    return ves / ves.max() if ves.max() > 0 else ves
```
This scores every pixel on "how much does this look like a thin curvilinear ridge" at several
thicknesses (sigma = 1 through 5px) simultaneously. High on cracks, low on round pores — exactly
the shape discrimination needed, computed from actual eigenvalues of the local Hessian matrix
(second-derivative curvature in two directions), not just brightness.

**2. Per-region feature vector** (via `skimage.measure.regionprops`):

```python
FEATURE_COLUMNS = [
    "LogArea", "Elongation", "Solidity", "Eccentricity", "Extent",
    "Circularity", "MeanDarkness", "MeanVesselness",
]
```
```python
minor = p.axis_minor_length if p.axis_minor_length > 0 else 0.5
elongation = p.axis_major_length / minor
perim = p.perimeter if p.perimeter > 0 else 1.0
circularity = min(4 * np.pi * p.area / (perim ** 2), 1.0)
```

**Why these 8 specifically:** Elongation, solidity (how filled-in vs. jagged the outline is),
eccentricity, extent, and circularity are all classic shape descriptors that separate "thin
winding crack" from "round blob" (pore/inclusion). LogArea (not raw area) compresses the huge
range between a tiny microcrack and a big void into something a linear model can weigh sensibly.
MeanDarkness and MeanVesselness bring in the two intensity-based signals from Steps B–C.

---

### Step E — ML Classification & Fragment Merging

#### E1: The classifier

```python
X = scaler.transform(df[FEATURE_COLUMNS].values)
proba = clf.predict_proba(X)[:, 1]
df["CrackProbability"] = proba
df["IsCrack"] = proba >= proba_threshold
```

**Model type, confirmed directly from the saved file (not assumed):**
```python
>>> joblib.load('models/crack_classifier.joblib')['clf']
sklearn.linear_model.LogisticRegression
```
**8 input features** = exactly the `FEATURE_COLUMNS` list above. **1 output** = probability the
region is a real crack (0.5 = the default cutoff).

**What logistic regression actually is:** It multiplies each of the 8 features by a learned weight,
sums them into one score, then squeezes that score through a sigmoid curve `1/(1+e^-z)` to get a
clean 0–1 probability. Training = repeatedly adjusting the 8 weights so predicted probabilities
match the human-confirmed labels as closely as possible (minimizing log-loss via gradient
descent).

**Why this model (see full comparison table below):** simple, doesn't need much labeled data,
produces a genuine probability (needed later for the hybrid rule in Step H), and its weights are
directly readable — you can point at "MeanVesselness: +2.19" and explain the decision.

#### E2: Fragment merging (Dijkstra + Minimum Spanning Tree)

One physical crack is often segmented into several disconnected pieces. This step reconnects them.

```python
from scipy.sparse.csgraph import minimum_spanning_tree
...
def _cheapest_path(flat, pA, pB, margin=15):
    """Dijkstra through a brightness-as-cost field (skimage's
    route_through_array) -- not the straight line."""
    cost = flat[r0:r1, c0:c1].astype(np.float64)
    indices, _ = route_through_array(cost, start, end, fully_connected=True)
    return indices + [r0, c0]
```
```python
# for every pair of crack fragments within max_gap_px of each other:
path = _cheapest_path(flat, tuple(pA), tuple(pB))
if np.percentile(flat[path[:, 0], path[:, 1]], 75) > max_bridge_darkness:
    continue  # a real chunk of even the darkest available route is bright, intact surface
dist_matrix[i, j] = d[k]
...
sparse = lil_matrix(np.where(np.isinf(dist_matrix), 0, dist_matrix))
mst = minimum_spanning_tree(sparse).toarray()  # <- real scipy MST call
```

**How this works, in the map analogy:** Dijkstra's algorithm is the same one GPS navigation uses
— find the cheapest route through a network, where "cheap" here means "stays as dark as
possible," not "shortest distance." Each pixel is a node; a bright pixel costs more to cross than
a dark one. This finds the real, possibly-curved dark path between two fragments, not a straight
line that might cut across intact bright surface.

Then, across *all* fragment pairs at once, scipy's `minimum_spanning_tree()` decides the minimum
set of connections needed to join every fragment of the same physical crack — so it doesn't draw
every possible pairwise bridge, just the necessary ones (same concept as connecting cities with
the fewest total miles of road).

**The darkness check uses the 75th percentile, not the mean**, specifically because a path with
dark endpoints but one bright gap in the middle can still *average* out dark — the percentile
check catches that a real chunk of the route is bright even when the mean would hide it.

---

### Step F — Output Generation

Writes the final black/white mask, the numbered red/cyan overlay, and a measurements CSV. Pure
file I/O — no modeling here, just packaging the result into forms a scientist can use directly.

---

## PART 1 comparison table — what was tried instead of Logistic Regression, and the results

An `--mode train` code path in `detect_cracks.py` exists using a `RandomForestClassifier` with
plain `cross_val_score` — worth flagging honestly: this looks like an earlier/alternate
experimental path, **not** what produced the actually-deployed `crack_classifier.joblib`
(confirmed: that file's `clf` is `LogisticRegression`, not a Random Forest). Don't present the
Random Forest path as "the current model's validation" — it isn't.

---

## PART 2 — Manual Correction & Retraining Loop

### Step G — Manual Correction (the paint app)

**Language: Python (Flask) backend + JavaScript/HTML/Canvas frontend.**

**How the two sides connect (plain HTTP, same as any website):**

```python
# paint_server.py
app = Flask(__name__)

@app.route("/api/template/<image_name>")
def api_template(image_name):
    ...
    return send_file(template_path, mimetype="image/png")

@app.route("/api/save/<image_name>", methods=["POST"])
def api_save(image_name):
    data = request.get_json()
    layer = Image.open(io.BytesIO(base64.b64decode(data["dataURL"].split(",",1)[1]))).convert("RGBA")
    ...

@app.route("/api/ingest/<image_name>", methods=["POST"])
def api_ingest(image_name):
    stage = get_stage(image_name)
    result = _ingest(image_name, stage=stage)
    return jsonify({"ok": True, **(result or {})})
```
```javascript
// paint_frontend.py -- this string IS the JavaScript that runs in your browser
async function savePaint() {
  const dataURL = paintCanvas.toDataURL('image/png');
  const res = await fetch('/api/save/' + currentImage, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ dataURL }),
  });
  ...
}
```

Flask starts a small web server on `127.0.0.1:8765` (your own machine only, nothing leaves it).
The browser's JavaScript is what you see and click; every time it needs something real (an image,
a save, a commit), it calls one of these `/api/...` addresses over plain HTTP, Python does the
actual file/model work, and sends back a small JSON answer the JavaScript displays.

**Candidate generation for the "harder" regions** (three purpose-built shape detectors, all in
`interior_candidates.py`):

```python
def concavity_candidates(crack_mask, close_radius=12, min_area=30, max_area=6000):
    """Morphological CLOSING (dilate then erode) with a small disk finds
    notches along a crack's own boundary -- bounded by close_radius so it
    can't run away across a whole bend the way a convex-hull approach did
    (confirmed: hull swallowed a wide swath on a curvy fragment)."""
    closed = morphology.binary_closing(sub, morphology.disk(close_radius))
    gap = closed & ~sub
```

```python
def interior_fill_candidates(labeled, df, img8, crack_mask, dist_from_crack, ...):
    """Flood-fills outward from a crack through 'loosely dark' pixels
    (calibrated per-image from that image's own clean-background median/MAD),
    capped at max_group_multiple x the crack's own confirmed area so a leaky
    threshold can't run away."""
    threshold = bg_median - darkness_k * bg_mad
    loose_dark = (img8 < threshold) & (dist_from_crack <= max_reach_px)
    loose_labeled = measure.label(loose_dark | crack_mask, connectivity=2)
```

**Why a paint app instead of editing a spreadsheet of numbers:** crack correction is inherently
spatial — pointing at exact pixels is far faster and more precise than describing a region by
coordinates or ID.

---

### Step H — Machine Learning (the SECOND, separate model)

Worth being explicit about: **this is a different model from Step E's classifier.** Step E judges
the *original* dark-region candidates. Step H judges a *different* population — the concavity,
bridge-corridor, and gradient-fade candidates Step B's own high-pass filter can't see at all.

#### Model + features

```python
INTERIOR_FEATURE_COLUMNS = [
    "LogArea", "Elongation", "Solidity", "Eccentricity", "Extent", "Circularity",
    "MeanRawBrightness", "MeanFlatBrightness", "MeanVesselness",
    "FracBoundaryTouchingCrack", "MeanDistToCrack",
]
```
Same model type (`LogisticRegression(class_weight="balanced")`) on **11 features** — 6 shape
features unchanged from Step E, brightness split into two versions (raw vs. flattened, so the
model can see the specific "dark in the raw picture but not after flattening" signature of a real
damage zone), plus two brand-new position features that only make sense once *some* cracks are
already confirmed: `FracBoundaryTouchingCrack` (what fraction of this candidate's own edge touches
a confirmed crack) and `MeanDistToCrack` (distance in pixels to the nearest one).

```python
def _region_features(mask_bool, flat, img8, vesselness, crack_mask, dil_crack, dist_to_crack):
    boundary = mask_bool[y0:y1, x0:x1] & ~morphology.binary_erosion(mask_bool[y0:y1, x0:x1])
    touching_crack = int(dil_crack[y0:y1, x0:x1][by, bx].sum())
    frac_touching_crack = touching_crack / max(1, len(by))
    mean_dist_to_crack = float(dist_to_crack[mask_bool].mean())
    return {
        "MeanRawBrightness": float(img8[mask_bool].mean()),
        "MeanFlatBrightness": float(flat[mask_bool].mean()),
        "MeanVesselness": float(vesselness[mask_bool].mean()),
        "FracBoundaryTouchingCrack": frac_touching_crack,
        "MeanDistToCrack": mean_dist_to_crack,
        ...
    }
```

**How your painting feeds this:** there is no special "was this hand-drawn" feature. When you
paint a new region, its exact shape *is* the `mask_bool` every one of the 11 numbers gets computed
from — identical treatment to an automatically-proposed candidate. Only the *label* (crack/not
crack) is special about painting — it's set directly from which color you used.

#### The hybrid rule — why interior_fill needs more than the raw model

With only 2 confirmed "no" examples vs. 39 confirmed "yes" for this specific candidate type, a
plain model accepts ~95% of everything (confirmed by direct measurement). The fix requires THREE
conditions simultaneously, not just the model's probability:

```python
def calibrate_interior_fill_rule(df_labeled, scaler, clf):
    """accept iff: ML probability >= floor
               AND MeanDistToCrack <= Nth percentile of the full candidate pool
               AND MeanFlatBrightness <= Mth percentile of the full candidate pool
    N/M/floor are grid-searched to maximize recall on known positives subject
    to rejecting BOTH known negatives under LEAVE-ONE-OUT-refit probability
    (not in-sample, which let an earlier version through that only worked
    because the model had memorized that exact negative)."""
    for N in range(5, 71, 5):
        for M in range(20, 91, 5):
            for floor in np.arange(0.10, 0.86, 0.05):
                if (neg_rule_ok & (loo_proba_neg >= floor)).any():
                    continue  # would accept a known negative -- reject this config
                recall_pos = (pos_rule_ok & (full_fit_proba[pos_mask] >= floor)).mean()
```

**Leave-one-out, explained:** with only 2 negative examples, refit the model excluding one
negative at a time (2 refits total) and check it's still correctly rejected using the fit that
never saw it. An ordinary 5-fold split would put most negatives in folds with zero negative
examples to test against — a meaningless "great" score. This is the honest substitute.

**Current calibrated rule (from the latest actual retrain):**
```
accept iff ML_proba >= 0.80  AND  MeanDistToCrack <= 12.90px (70th pct)  AND  MeanFlatBrightness <= 129.23 (85th pct)
recall on known positives: 61.5%   |   full candidate-pool acceptance: 43.4%
```

#### The feedback loop itself

```python
def get_stage(image_name, force=False):
    is_stale = cached is not None and current_model_mtime is not None and \
        (cached[1] is None or current_model_mtime > cached[1])
    if force or cached is None or is_stale:
        _stage_cache[image_name] = (run_enhanced_pipeline(image_name), current_model_mtime)
```
Whatever Step H currently accepts renders as plain red in the paint app (same color as any
confirmed crack — no third color). Correcting one doesn't just fix the picture; it logs a new,
correctly-typed training example. The paint app also auto-detects when its cached picture predates
the current model and regenerates before showing you anything, so retraining actually reaches what
you see without a manual refresh step.

---

## The full model-comparison record (13 real experiments, 3 rounds)

Every one of these was **actually run against the real data**, not estimated. All results below
are real printed numbers from those runs.

### Round 1 — first response to "the model over-predicts"

| Experiment | What it tried | Result |
|---|---|---|
| **per_type_models** | A separate model per candidate type instead of one pooled model | Didn't fix interior_fill's over-acceptance — the problem was data scarcity (2 negatives), not model architecture |
| **alternative_algorithms** | RandomForest, GradientBoosting, SVM (rbf) in place of LogisticRegression | All performed *worse* — with so little data, more flexible models overfit the tiny negative set instead of generalizing |
| **imbalance_and_calibration** | Aggressive class-reweighting + probability recalibration on the plain model | Improved but didn't solve it — recalibrating a model with almost no negative signal still has almost no negative signal |
| **feature_engineering** | Hand-built ratio/product/log feature combinations | No improvement — the 2 known negatives didn't separate from positives on any tried axis |
| **hybrid_rule_plus_ml** ✅ **WINNER** | ML floor + distance/brightness percentile gates, LOO-validated | Only approach that correctly rejected both known negatives under leave-one-out without degrading concavity/bridge_corridor. Adopted into production. |

### Round 2 — fine-tuning the winning hybrid rule

| Experiment | What it tried | Result |
|---|---|---|
| **finer_grid_pareto** | Finer-resolution grid search | Technical failure — didn't complete this round (successfully redone in round 3) |
| **third_gate_feature** | Add a 3rd gate (FracBoundaryTouchingCrack) | Looked better on paper (76.9% recall vs. 74.4%) — but my own independent margin analysis found it achieves this by moving one negative's rejection from a robust ML-floor margin onto a brand-new, never-stress-tested gate with only a 7% margin. **Rejected** — numerically better isn't the same as actually safer. |
| **stability_and_sensitivity** | Perturb feature values ±20–30%, see what breaks | Found production's own distance-gate margin was thinner (~7.5%) than assumed |
| **soft_score_alternative** | A single smooth weighted score instead of hard gates | Worse than the hybrid rule |

### Round 3 — deeper follow-up

| Experiment | What it tried | Result |
|---|---|---|
| **finer_grid_retry** | Redid round 2's failed finer grid (1% steps instead of 5%) | Found one config that dominates on raw numbers (76.9% recall) but confirmed it rests on the same thin ~7-8% margin already flagged as risky — not adopted |
| **margin_maximization** | Directly search for max worst-case safety margin instead of max recall | Found a genuinely safer alternative: `floor=0.63, dist≤11.04px, bright≤149.88` → margins roughly double (11.7%→15.9%, 7.8%→26.0%) at the cost of recall dropping 74.4%→71.8%. Offered as an option, not adopted automatically (a deliberate recall-vs-safety tradeoff, not a free win) |
| **simple_untried_algorithms** | GaussianNB, k-NN (k=3,5), single-feature decision stumps, 2-feature shallow tree | GaussianNB and k-NN both **memorized** the 2 negatives so completely that leave-one-out scored the excluded negative at 100% confidence *crack* — worse than doing nothing. Decision stumps mostly failed LOO too |
| **synthetic_negative_augmentation** | Gaussian-jitter synthetic negatives + a "far+bright" domain-prior synthetic negative | Jitter's apparent gain was a floor-relocation artifact with zero real robustness gain (confirmed via direct perturbation test). The physics-informed prior actively backfired since neither real negative is actually far-or-bright |

**Bottom line after all 13 experiments:** the hybrid rule (ML floor + distance gate + brightness
gate, leave-one-out calibrated) remains production. Nothing tested — across three independent
rounds — beat it without an honest trade-off. Getting meaningfully better from here needs more real
negative labels, not more algorithmic tricks.

---

## "Why not just use Google's (or any big pretrained) image segmentation model?"

A fair question, and worth having a real answer for rather than "we didn't think of it." Four
concrete reasons, in order of how much they actually drove the decision:

**1. There's no pretrained model for this specific domain, and general ones don't transfer.**
Models like DeepLab, Mask R-CNN, or SAM are trained on natural photos (or general medical scans)
— cars, people, organs. A grayscale SEM/TXM micrograph of annealed stainless steel looks nothing
like any of that, and a crack is not an "object" in the sense these models were built to find.
Without domain-specific fine-tuning, there's no real reason to expect one to know what a crack in
this material looks like versus a grain-boundary pit or an imaging artifact — which is exactly the
distinction this whole project is built around.

**2. Fine-tuning a deep model needs far more labeled data than exists here.** This project has on
the order of a couple hundred labeled examples total, accumulated by hand over many correction
rounds. Deep segmentation models typically need thousands of labeled examples to fine-tune
reliably on a new domain — trying to adapt one with this little data would very likely *overfit*,
which is the exact same failure mode already confirmed directly in this project when Random
Forest / Gradient Boosting / SVM (all more flexible than Logistic Regression) were tried on the
small interior_fill dataset and made things worse, not better.

**3. Interpretability and scientific trust matter for this use case.** This is feeding a materials
science measurement, not a consumer app — if a reviewer or advisor asks "why did the system call
this a crack," the honest answer with this pipeline is "here are the 8 (or 11) numbers and their
learned weights, and here's the leave-one-out test proving the model isn't just memorizing 2
examples." A large pretrained network's internal decision is much harder to explain or audit at
that level, which matters a lot when the output needs to be scientifically defensible.

**4. Domain-specific physical reasoning has to be built in either way.** Even a state-of-the-art
segmentation model wouldn't natively know that a crack can be split into disconnected fragments
that should be *counted as one*, or that a high-pass filter used for illumination correction
specifically erases a certain class of real damage — that logic (Dijkstra/MST bridging, the
three-part hybrid rule) had to be engineered regardless of which underlying model does the raw
pixel classification. Once you're building that much domain-specific pre/post-processing anyway,
a lot of the "why not just use a big model" advantage disappears.

**The honest, balanced version for a presentation:** this isn't "deep learning is wrong for this" —
it's a data-regime and interpretability tradeoff that matches what's actually available (~200
labeled examples, a need for explainable results) right now. If this dataset grows into the
thousands of labeled examples, fine-tuning a real segmentation model could well outperform this
pipeline eventually. The current design is the right tool for the data and trust requirements that
exist today, not a rejection of the alternative on principle.

---

## One-page summary table

| Step | Language | Model / Algorithm | Chosen because |
|---|---|---|---|
| A. Preprocess | Python | none (percentile stretch, geometry) | Robust to outlier pixels; auto-removes non-sample chrome |
| B. Illumination | Python | Gaussian blur high-pass filter | Removes lighting drift; trade-off is *why* Step H exists |
| C. Segmentation | Python | Otsu + robust MAD + absolute threshold, OR'd | Each covers the others' blind spot on this dataset |
| D. Features | Python | Frangi (Hessian-eigenvalue) vesselness filter | Same math as medical vessel-segmentation; scores "thin ridge-ness" directly |
| E. Classify+merge | Python | **Logistic Regression** (8 features) + Dijkstra + scipy MST | Simple model generalizes better on modest labeled data; MST avoids redundant bridges |
| F. Output | Python | none | File packaging |
| G. Manual correction | Python (Flask) + JavaScript (browser) | none (human input) | Spatial correction needs spatial interaction |
| H. Interior model | Python | **Logistic Regression** (11 features) + hand-designed hybrid rule, LOO-validated | Tested against 13 real alternatives across 3 rounds; none beat it without a real trade-off |
