# Crack analysis — index and status

Built 2026-08-26/27 from `crack_export/`. Read in this order. **Several results in here
are withdrawn; the banners say which.** Nothing below is a live claim unless marked so.

## Read first

| # | file | what it is | status |
|---|---|---|---|
| 1 | **`LABEL_GRANULARITY.md`** | 91 % of the 70 M hand-marked pixels are broad-brush region assertions (median stroke 59 px, max 413 px), not crack outlines | **live — most consequential** |
| 2 | **`CORRECTION_scale_and_magnification.md`** | µm/px *is* recoverable from the SEM databar; the corpus spans HFW 10.4 µm – 2.59 mm (249×) | **live** |
| 3 | **`VERDICT_2026.md`** | 2026 literature killed 5 of 6 novelty claims; one survivor (the paired-detector result). Its "one unoccupied methods gap" is **withdrawn** — see below; superset supervision is owned | **live — current verdict, minus the withdrawn gap** |
| 4 | `HEAD_TO_HEAD_VERDICT.md` | six competitors on a fixed rubric; where this method wins and loses | live, **except** the "full-frame 25 MP" advantage, retracted in `VERDICT_2026.md` §4 |
| 5 | `CRACK_ANALYSIS.md` | the original per-set analysis: censoring, two-population region structure, review audit, paired detector | live **except §4**, the `Cast ≫ AS ≈ HIP` process ranking, **withdrawn** |
| 6 | `sam3/CLEAN_RUN_RESULTS.md` | SAM 3 re-run leak-gated on **15 provably disjoint** tiles, with a verdict table of what died. This replaces the 16-tile run | **live** |
| 7 | `LINEARITY_AND_FRACTURE_MODE.md` | linearity, tortuosity confounds, transgranular/intergranular | live; **§7 corrects my own tortuosity estimator** (1.094 raw → 1.030 excess over an angle-matched null) |

## Superseded — kept for the record only

| file | superseded by | why |
|---|---|---|
| `NOVELTY_ASSESSMENT.md` | `VERDICT_2026.md` | searched 2023–2025 only; its surviving claims were later killed by 2026 work |
| `PUBLICATION_NOTES.md` | `VERDICT_2026.md` | recommended a dataset paper and a weak-supervision framing that both died |
| `sam3/SAM3_ON_SEM_CRACKS.md` | `sam3/CLEAN_RUN_RESULTS.md` | **⛔ retracted in full 2026-09-18** — the model input was the green channel of the annotated overlay, so the experiment measured its own label on 14 of 16 tiles. Every score in it is void, including the bimodal recall this index used to quote as live. It also did **not** correct `VERDICT_2026.md` §6: that section now records that the "correction" was itself the leaked measurement, so the original claim stands unrefuted. Kept unedited as the record; see `sam3/LEAK_POSTMORTEM.md` |
| `sam3/GAP_CONFOUND.md` | — | **⛔ retracted in full**, same leak. Kept unedited as the record |

## What is actually still standing

**One claim.** On identical fields at matched beam settings (10.00 kV, 1.6 nA, 6.0 mm WD,
matched HFW), changing only the SEM detector changes measured crack extent by **2.04×**
(7/8 pairs, p = 0.016) with *narrower* features (p = 0.039); +3.88 pp crack area, 8/8
pairs, p = 0.0078. No 2026 paper does this.

⛔ **p = 0.0078 is the exact test's floor at n = 8, not a measurement, and the 8 confirmed pairs are 3 specimens, not 8. Aggregating per specimen — which this repo instructs — gives exact p = 0.2500. The direction is unanimous (8/8 pairs, all 3 specimens, median +3.88 pp); the significance is withdrawn. See `VERDICT_2026.md` §S1(b).**

 Pair arithmetic, stated plainly: **16**
index-matched pairs, **15** share HFW exactly, **8** registration-confirmed carry the
p-values.

**Its biggest risk — unresolved.** That 2.04× is measured *by a detector trained on
91 %-broad-brush labels*. If the segmenter is contrast-sensitive and CBS renders cracks
with more contrast, it may be the segmenter's contrast response rather than crack extent.
Must be recomputed segmentation-free or recall-matched on fine hand-traced labels.

**~~One new finding, free to report — SAM 3's prompt brittleness.~~ WITHDRAWN 2026-09-18 —
it is owned, twice, at far better power.** The counts this paragraph used to quote (14/16 and
16/16) came from the leaked 16-tile run and are void regardless. The finding itself is not
ours: the mechanism is in Meta's own SAM 3 abstract (arXiv:2511.16719), and the synonym
instability is published at much larger n — CoCo-SAM3 (arXiv:2604.19648) and
arXiv:2604.17126, the latter over 263 COCO val2017 images with six prompts and a structure
analysis, against our four prompts over 15 tiles from 9 frames with no pre-registered synonym
list. See `sam3/PRIOR_ART_KILL.md`, which resolved every reference live against the arXiv API.
The leak-gated replacement numbers are in `sam3/CLEAN_RUN_RESULTS.md`.

**~~One unoccupied methods gap.~~ WITHDRAWN 2026-09-18 — it is owned.** Box supervision *is*
superset supervision, so the claim contradicted its own example list. Prior art, verified against source 2026-09-18: the formal setting is **Superset Label
Learning** (Liu & Dietterich, ICML 2014, PMLR 32:1629–1637); the segmentation machinery is
**box supervision with a tightness prior** (Kervadec et al., MIDL 2020, PMLR 121:365–381,
arXiv:2004.06816 — every line inside the annotated region must contain ≥1 foreground pixel,
which is exactly what a 59 px brush over a 3 px crack asserts); and the crack domain has
already done the over-inclusive case with a **shrink module** (*Unified weakly and
semi-supervised crack segmentation framework using limited coarse labels*, Eng. Appl. Artif.
Intell. 2024, 10.1016/j.engappai.2024.108497 — IoU 77.53%, +28.64 pp over
fully-supervised-on-coarse-labels). Full writeup:
`../../crack-depth-3d/docs/SUPERSET_CLAIM_CLOSED.md`. What
survives is the measurement (`LABEL_GRANULARITY.md`), not a gap.

**The blocker for everything else.** There is no pixel-precise ground truth. ~15 paired
fields hand-traced at ≤5 px (2–4 h each) unblocks the metrics, the circularity fix and any
benchmark.

## Every quoted statistic is machine-verified

`tools/verify_claims.py` recomputes **55 registered statistics** from their source
artefacts and fails if a document quotes a number the artefact no longer produces. Last
run: **55/55 pass**. In an unbuilt tree the claims whose regenerable artefacts are absent
report SKIP rather than FAIL, so a fresh clone does not look broken. Run it before sending
anything:

```bash
python tools/verify_claims.py    # 55 claims; exit 1 if any drifted
```

It exists because re-reading prose does not catch drift. A recall-threshold count that was
9/16 was published as 10/16 across six files and survived three manual review passes; this
would have caught it on the first run. A new claim is only covered once registered, so add
to `CLAIMS` when you add a number to a document.

## Corrections to numbers quoted earlier in this work

- **AUC is 0.714**, not 0.884. `pooled_auc = 0.7144 ± 0.0278` (grouped CV, 45 images);
  0.884 is `loio_auc_exhaustive_image` — leave-one-image-out on a *single* frame.
- `Cast ≫ AS ≈ HIP` — withdrawn; at matched HFW = 319 µm only 1–2 fields per route exist.
- ρ(area, width) > ρ(area, extent) — **tautological**; `A_A ≡ L_A · (A/L)` exactly
  (verified to 5.6 × 10⁻¹⁷).
- Tortuosity ≈ 1.09 — that is inside the estimator's own bias envelope; the corrected
  excess over an angle-matched null is **1.030**.
- "Full-frame 25 MP, no tiling" as an advantage — retracted (MegaSeg, Jan 2026, 67 MP).
- "No comparable public paired-detector dataset exists" — false (RODARE; Schmies 2023).
- **THE SAM 3 EXPERIMENT IS INVALID** — input was the green channel of the annotated overlay
  (the red burned in is (225,25,25), so green = 25 -- far below the 80 the threshold used,
  and constant); a bare threshold recovers the label at recall 1.000 on 14 of 16 tiles. All SAM 3 numbers withdrawn, including the "correction" they were used to make.
- ~~**"SAM-family models remain poor on thin low-contrast curvilinear structures" /
  "zero-shot: no"** (`VERDICT_2026.md` §6) — **disproven by measurement on this corpus.**
  SAM 3 reaches recall 0.969–1.000 on 10/16 tiles and IoU 0.59–0.74 where the label is complete.
  Section rewritten in place with the original text quoted.~~ **← this retraction is itself withdrawn**

## Data files

```
per_frame_metrics.csv / _with_review.csv   39 cols x 62 frames, + review coverage
per_set_metrics.csv                        mean/median/sd per set
review_coverage.csv                        hand-marked px per frame
label_granularity.csv                      brush stroke thickness per correction mask
scale_hfw.csv                              HFW um for 45 frames, read off the databars
paired_detector.csv                        CBS vs ETD on the same field
branches.csv                               33,471 skeleton branches x 22 cols
skeleton_frames.csv / holes.csv            per-frame skeleton, fractal dim, enclosed islands
straightline_null.json                     digitization-bias null vs angle
figures/fig1..fig12.png                    corpus-level figures
figures/per_set/*.png                      one standalone diagram per set (also in sets/)
crops/*.png                                native-resolution windows, incl. label_granularity.png
sam3/                                      SAM 3 run: results, figure, scripts, shims
```

## Reproduce

Everything resolves paths from its own location, so any working directory works. Nothing here
needs a GPU. Artefacts marked *regenerable* are gitignored; until they exist
`tools/verify_claims.py` reports the claims that need them as SKIP, not FAIL.

### A. Corpus analysis — pure CPU, project environment

```bash
python3 tools/split_sets.py          # sets/ (hard links, ~0 extra disk) -- run BEFORE per_set_diagram
python3 tools/analyse_sets.py        # per-frame + per-set metrics
python3 tools/review_coverage.py     # read-only over interior_active_learning/paint/
python3 tools/skeleton_metrics.py    # 62 frames, parallelisable by index range
python3 tools/holes.py
python3 tools/report_sets.py ; python3 tools/linearity_figs.py
python3 tools/csv_shape_figs.py ; python3 tools/per_set_diagram.py
python3 tools/paired_detector.py
python3 tools/verify_claims.py       # 55 claims; exit 1 on drift
```

### B. Benchmark round — `analysis/sam3/`

The order matters: registration before tiling, tiling before the leak gate, the gate before any
model run. `run_real.py` refuses to start if the gate fails.

```bash
cd analysis/sam3
python3 align_originals.py     # register label frames to the raw original/*.tif (9/9, ncc >= 0.99)
python3 make_tiles.py          # 15 PROVABLY DISJOINT tiles; grey from the original, never an overlay
python3 leak_check.py          # must print 0/N contaminated; keeps the known-bad input as a control
```

Model runs need **isolated** virtualenvs — installing either package into the project
environment downgrades numpy and breaks scipy/tifffile/opencv/zarr:

```bash
SAM3_SAVE_MASKS=1 SAM3_SAVE_SERD=1 python3 run_real.py   # ~28 min CPU; needs the sam3 venv
python3 run_omnicrack.py                                  # ~3 min + a 1.25 GB weight download
```

Scoring and analysis, all pure CPU in the project environment:

```bash
python3 analyse_clean_run.py   # per-prompt table, paired against the contaminated run
python3 permutation_test.py    # nulls for the presence-magnitude claim (20k draws, seeded)
python3 iou_ceiling.py         # THE number: a perfect 3 px trace scores IoU 0.1662 here
python3 methods_bench.py       # 5 methods x OIS / ODS / LOFO / nested LOFO  (~40 min)
python3 baseline_matched.py    # Otsu / ODS / OIS / information-free null at matched budgets
python3 tolerance_sweep.py     # clIoU_tau; the leader changes with tau
python3 corridor_metric.py     # superset-label scoring -- ALWAYS read its null panel
python3 trained_lofo.py        # supervised arms, leave-one-frame-out
python3 ensemble_lofo.py       # union / intersection / majority, nested
python3 omnicrack_eval.py      # the published SOTA, scored on our tiles
python3 serd_eval.py           # SAM 3's field read before the presence gate
python3 make_figures.py        # regenerates every figure from committed artefacts
```

### Read the results in this order

1. `sam3/POSITION_VS_2026.md` — where this stands, what is stronger, what is weaker
2. `sam3/SOTA_2026.md` — the verified 2026 literature and the model that is the one to beat
3. `sam3/PRIOR_ART_KILL.md` — why the SAM 3 finding is not a contribution
4. `sam3/LEAK_POSTMORTEM.md` — how the first run measured its own annotation
5. `LABEL_GRANULARITY.md` — why pixel IoU cannot rank methods on this corpus

