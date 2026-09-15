# SAM 3 on your SEM crack tiles — it ran, and the answer is bimodal

**Provenance, first.** `facebook/sam3` and `facebook/sam3.1` are gated-manual and your
account is **not authorised** (403 `GatedRepoError`, verified today). These weights are the
community mirror `1038lab/sam3` (`sam3.safetensors`, 1,465 tensors, header-validated),
used at your explicit direction. The model *code* is Meta's official `sam3` package.
**Nothing here is citable until official access is granted and the run is repeated.**

Setup: 16 tiles of 1024×1024 at native resolution, centred on hand-labelled crack, from
the 9 fine-stroke frames (median brush ≤25 px — the only labels in this corpus that
approximate an outline). CPU, ~65 s per tile encode. Four text prompts.

## 1. The headline is bimodal, not mediocre

| | tiles |
|---|---|
| **recall 0.969 – 1.000** — found essentially *all* the labelled crack | **10 / 16** |
| **recall = 0.000** — complete miss | **6 / 16** |

Median recall **0.979**, and **the gap is empty**: the lowest non-zero recall is 0.969 and
the next value down is 0.000. SAM 3 either gets these cracks essentially completely or does
not see them at all, and it gives you no signal about which.

## 2. My first aggregate number was wrong, and the figure shows why

I initially reported median union IoU 0.121 and would have concluded "SAM 3 is poor on
SEM cracks." That is an artefact of **label incompleteness**, not model failure.

Look at row 1 of `sam3_examples.png`: on `260622_316_H_b4_CBS_02__t0` the human labelled
**one** crack (0.49 % of the tile); the tile plainly contains many; SAM 3 returned 40
instances covering most of them. Recall 1.000, precision 0.083, IoU 0.083. The metric
punished it for finding cracks nobody annotated.

The correlation is clean — precision tracks how complete the label is:

| tile | label % of tile | recall | precision | IoU |
|---|---|---|---|---|
| `AS_24hr_BSE_Side_008__t0` | 10.16 | 0.984 | **0.746** | **0.737** |
| `AS_24hr_BSE_Side_008__t1` | 7.58 | 0.989 | **0.698** | **0.692** |
| `MAR_Amb_AS_CBS_0004__t0` | 4.82 | 0.987 | **0.598** | **0.593** |
| `MAR_Amb_HIP_ETD_0007__t0` | 2.27 | 1.000 | 0.386 | 0.386 |
| `MAR_Amb_Cast_ETD_0003__t0` | 0.50 | 1.000 | 0.213 | 0.213 |
| `260622_316_H_b4_CBS_02__t0` | 0.49 | 1.000 | 0.083 | 0.083 |
| `MAR_Amb_Cast_ETD_0003__t1` | 0.24 | 0.999 | 0.072 | 0.072 |

**Where your label is most complete, SAM 3 scores IoU 0.59–0.74** — comparable to the
62.34 % native-SAM3 crack IoU that CoRe-SAM3 (arXiv:2609.05816) reports on infrastructure
datasets. So SEM is *not* dramatically harder than concrete for SAM 3 when it engages.

This is the same trap as `LABEL_GRANULARITY.md` and your own "score the whole artifact,
not the labelled subset" rule. I walked into it and the picture caught me.

## 3. The prompt is brittle — the single most actionable finding

| prompt | tiles returning **zero** instances |
|---|---|
| `crack` | **2 / 16** |
| `a crack in metal` | 14 / 16 |
| `thin dark line` | 14 / 16 |
| `fracture` | **16 / 16** |

Only the bare noun `crack` works. "fracture" — a synonym any materials scientist would
use — returns *nothing at all* on every tile. Concept segmentation is real but the
concept vocabulary is narrow and unguessable, so prompt choice is a hidden researcher
degree of freedom. Anyone reporting SAM 3 numbers on microscopy without a prompt
ablation is reporting their luck with wording.

## 4. Union vs oracle: fragmentation

On `AS_24hr_BSE_Side_008__t0` the union of 54 instances scores IoU 0.737 while the single
best instance scores 0.265 — the field holds many separate cracks, so no one proposal
matches. Elsewhere the reverse: `260622_316_H_b4_CBS_02__t0` scores union 0.083, oracle
0.793 (clDice **0.992** — a near-perfect single crack buried among 40 proposals).

Which arm flatters depends on whether the field holds one crack or many. Report both.

## 5. What this means for you

- **SAM 3 is a credible baseline on your data** when it engages, and a fine-tuned
  adapter (CoRe-SAM3 needs 18.9 K trainable parameters) would very likely beat your
  logistic-regression detector on the tiles where it fires.
- **It is not trustworthy unsupervised**: 6/16 silent total misses, and nothing in the
  output distinguishes those from successes.
- **The pixel-precise validation set is still the blocker.** Every number above is
  bounded by label quality, and the most interesting cases are the ones where SAM 3
  found cracks your labels do not contain.
- **Get official access.** The result is reproducible in minutes once approved.

## Files

```
sam3_results.json      64 (tile, prompt) rows: union + oracle IoU/Dice/clDice/prec/rec
sam3_examples.png      4 tiles x [SEM | hand label | SAM3 union | SAM3 oracle-best]
tiles_meta.json        the 16 tiles, source frame, offsets, label fraction
make_tiles.py          tile extraction
prepare_checkpoint.py  regenerates sam3_original.pt from the cached safetensors (~5 s)
run_real.py            inference + scoring
shim/                  three compatibility shims, each documented in-file:
                         sam3_preload.py  - pre-empts sam3.model.edt (imports triton; CUDA-only)
                         cuda_redirect.py - position_encoding.py:55 hardcodes device="cuda";
                                            also neutralises pin_memory (pins to mps:0 here)
```

Reproduce, in order (verified from this directory today):

```bash
python make_tiles.py            # regenerates tiles/ — bit-identical to tiles_meta.json
python prepare_checkpoint.py    # rebuilds sam3_original.pt from the cached safetensors
PYTHONPATH=shim python run_real.py
```

`tiles/` and the 3.4 GB `.pt` are both regenerable and are not shipped; `make_tiles.py`
was re-run from here and reproduced the shipped metadata exactly.
Three further quirks needed handling and are noted in the code: the checkpoint is
bfloat16 while the model builds in fp32 (run under CPU bf16 autocast), and
`Sam3Processor.set_image` reads `shape[-2:]` for arrays, so an HWC numpy image is read as
3 px wide — pass PIL.
