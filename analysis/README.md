# Crack analysis — index and status

Built 2026-08-26/27 from `crack_export/`. Read in this order. **Several results in here
are withdrawn; the banners say which.** Nothing below is a live claim unless marked so.

## Read first

| # | file | what it is | status |
|---|---|---|---|
| 1 | **`LABEL_GRANULARITY.md`** | 91 % of the 70 M hand-marked pixels are broad-brush region assertions (median stroke 59 px, max 413 px), not crack outlines | **live — most consequential** |
| 2 | **`CORRECTION_scale_and_magnification.md`** | µm/px *is* recoverable from the SEM databar; the corpus spans HFW 10.4 µm – 2.59 mm (249×) | **live** |
| 3 | **`VERDICT_2026.md`** | 2026 literature killed 5 of 6 novelty claims; one survivor; the one unoccupied gap | **live — current verdict** |
| 4 | `HEAD_TO_HEAD_VERDICT.md` | six competitors on a fixed rubric; where this method wins and loses | live, **except** the "full-frame 25 MP" advantage, retracted in `VERDICT_2026.md` §4 |
| 5 | `CRACK_ANALYSIS.md` | the original per-set analysis: censoring, two-population region structure, review audit, paired detector | live **except §4**, the `Cast ≫ AS ≈ HIP` process ranking, **withdrawn** |
| 6 | **`sam3/SAM3_ON_SEM_CRACKS.md`** | SAM 3 measured on 16 hand-labelled tiles: bimodal (recall 0.969–1.000 on 10/16, 0.000 on 6/16); only the prompt `crack` works | **live** — and it *corrects* `VERDICT_2026.md` §6 |
| 7 | `LINEARITY_AND_FRACTURE_MODE.md` | linearity, tortuosity confounds, transgranular/intergranular | live; **§7 corrects my own tortuosity estimator** (1.094 raw → 1.030 excess over an angle-matched null) |

## Superseded — kept for the record only

| file | superseded by | why |
|---|---|---|
| `NOVELTY_ASSESSMENT.md` | `VERDICT_2026.md` | searched 2023–2025 only; its surviving claims were later killed by 2026 work |
| `PUBLICATION_NOTES.md` | `VERDICT_2026.md` | recommended a dataset paper and a weak-supervision framing that both died |

## What is actually still standing

**One claim.** On identical fields at matched beam settings (10.00 kV, 1.6 nA, 6.0 mm WD,
matched HFW), changing only the SEM detector changes measured crack extent by **2.04×**
(7/8 pairs, p = 0.016) with *narrower* features (p = 0.039); +3.88 pp crack area, 8/8
pairs, p = 0.0078. No 2026 paper does this. Pair arithmetic, stated plainly: **16**
index-matched pairs, **15** share HFW exactly, **8** registration-confirmed carry the
p-values.

**Its biggest risk — unresolved.** That 2.04× is measured *by a detector trained on
91 %-broad-brush labels*. If the segmenter is contrast-sensitive and CBS renders cracks
with more contrast, it may be the segmenter's contrast response rather than crack extent.
Must be recomputed segmentation-free or recall-matched on fine hand-traced labels.

**One new finding, free to report.** SAM 3's concept prompt is brittle: zero instances on
14/16 tiles for "a crack in metal" and "thin dark line", and on **16/16 for "fracture"**.
Only the bare noun `crack` fires. Prompt wording is an unguessable researcher degree of
freedom, so any SAM-3 microscopy number published without a prompt ablation is reporting
luck with vocabulary. Costs nothing further to claim — the ablation is already run.

**One unoccupied methods gap.** Every 2026 weak-supervision method assumes the weak label
is a *subset* of the object. These labels are *supersets*. Nothing in the window owns that.

**The blocker for everything else.** There is no pixel-precise ground truth. ~15 paired
fields hand-traced at ≤5 px (2–4 h each) unblocks the metrics, the circularity fix and any
benchmark.

## Every quoted statistic is machine-verified

`tools/verify_claims.py` recomputes **33 registered statistics** from their source
artefacts and fails if a document quotes a number the artefact no longer produces. Last
run: **33/33 pass** (`CLAIM_VERIFICATION.txt`). Run it before sending anything:

```bash
python tools/verify_claims.py    # exit 1 if any claim drifted
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
  (red is (255,0,0), so green = 0); a bare threshold recovers the label at recall 1.000 on
  16/16. All SAM 3 numbers withdrawn, including the "correction" they were used to make.
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

```bash
python3 tools/split_sets.py          # sets/ (hard links, ~0 extra disk)
python3 tools/analyse_sets.py        # per-frame + per-set metrics
python3 tools/review_coverage.py     # read-only over paint/
python3 tools/skeleton_metrics.py    # 62 frames, parallelisable by index range
python3 tools/holes.py
python3 tools/report_sets.py ; python3 tools/linearity_figs.py
python3 tools/csv_shape_figs.py ; python3 tools/per_set_diagram.py
python3 tools/paired_detector.py
python3 tools/verify_claims.py     # 33/33 must pass
```

`cracktrace/` is a separate **prototype**, validated on synthetic data only — see its own
README.
