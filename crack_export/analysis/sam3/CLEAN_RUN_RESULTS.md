# SAM 3 on SEM cracks: the leak-gated run, and why none of it is a contribution

*2026-09-18. Supersedes `SAM3_ON_SEM_CRACKS.md` (void — its input had the label written into it,
see `LEAK_POSTMORTEM.md`) and an earlier draft of this file that claimed a finding. The claim is
withdrawn: see `PRIOR_ART_KILL.md`. Weights are the community mirror `1038lab/sam3`;
`facebook/sam3` is still `gated: manual` (401 verified today), so nothing here is citable.*

Reproduce: `python align_originals.py && python make_tiles.py && python leak_check.py && SAM3_SAVE_MASKS=1 python run_real.py && python analyse_clean_run.py && python baseline_matched.py && python permutation_test.py && python make_figures.py`

**n = 15 tiles from 9 frames, provably disjoint.** An earlier 16-tile set contained
50.0%- and 56.6%-overlapping pairs from a bug in `make_tiles.py` (below).

---

## 0. The verdict first

| what I claimed | status |
|---|---|
| "the presence gate predicts emptiness on 64/64 pairs" | **tautology** — my wrapper read the tensors the library thresholds |
| "one scalar flips every instance simultaneously" | **false** — the threshold is per query over 200; survivors keep 1–62 |
| "a 112× presence span across four synonyms" | **one out-of-vocabulary noun**; 6.3× without `fracture`, 3.0× between the two real synonyms |
| "which words you type, not which micrograph" | **false** — tile is significant once conditioned on prompt |
| "an oracle threshold is a bar nobody stated; SAM 3 fails it" | **owned** (ODS/OIS, 2004/2011) and **rigged** (unmatched tuning); matched, p = 1.0000 |
| "per-tile IoU carries no model information" | **self-comparison**; SAM 3 beats the real null on 12/15 |
| "the contamination did not measurably change scores" | **underpowered**; another prompt gives p = 0.0312 |
| "an empty result is a presence-gate event" | **true only at my τ**; at the library default some of the 44 empties are decoder failures (the subcount is not reproducible — see §below) |
| the mechanism itself | **in Meta's abstract** (`arXiv:2511.16719`); synonym instability owned at 263 images (`arXiv:2604.17126`) |

What survives is a local measurement with honest caveats. Keep it as a reason not to trust
text-prompted SAM 3 on this material. Do not write it up.

---

## 1. Architecture, not measurement

`sam3_image_processor.py:195-200` — one global presence scalar `s_i` per (image, prompt)
rescales **all 200** per-query scores (`num_queries=200`, `model_builder.py:185`), then the
threshold is applied **per query**. Consequences, both definitional:

- whether the image returns *anything* is `s_i · max_j q_ij > τ`;
- the instance count is `#{j : q_ij · s_i > τ}` — there is no NMS, dedup or area filter after
  the gate.

So "the gate predicts emptiness on 64/64 pairs, 0 disagreements" **cannot fail**: the wrapper
reads the same tensors the library multiplies and thresholds with the same constant, and
`sigmoid(presence) > 0` makes `max_j(s·q_j) ≡ s·max_j q_j`. It is an instrumentation self-check
and is registered as one. Honest reference points for "did this pair return anything":
always-empty **39/60**, best prompt-identity-only rule **50/60** (83%). The implied comparison for any agreement figure is 83%, not chance.

**And "an empty result is a presence-gate event" is true only at the τ I picked.** Counting
empties whose *decoder* score `max_j q_ij` is itself below τ — a genuine query-score failure
rather than a gate event:

| τ | empty pairs | of which the decoder also failed |
|---|---|---|
| 0.3 | 39 | **0** |
| 0.4 | 42 | 0 |
| **0.5** (library default) | 44 | **6** |
| 0.6 | 46 | 17 |
| 0.7 | 50 | 34 |

The minimum `max_q` over all 60 pairs is 0.4434, which is the only reason nothing fails at
τ=0.3. At the library's own default the framing already breaks for some of the 44 empties.

> ⚠ **The subcount is not reproducible and the number has been removed, 2026-09-22.** The
> denominator checks out: 44 of the 60 (tile, prompt) pairs in `sam3_presence.json` are empty
> at τ = 0.5. The "6" does not. "Decoder failure" means empty for a reason other than the
> presence gate, and this document never says how that is decided; the obvious readings give
> 4 (presence alone ≥ τ) or 39 (max_q alone ≥ τ), and an independent audit got 5. Three
> readings, three answers, none of them 6. The qualitative point stands — at the library
> default some empties are not presence-gate events, which is enough to break the framing —
> but the count needs its definition written down before it can be quoted again.

And Meta published the mechanism: *"Recognition and localization are decoupled with a presence
head"* is in the abstract of `arXiv:2511.16719`. The synonym-instability half is owned too, at
far better power — CoCo-SAM3 (`arXiv:2604.19648`) names SAM 3 synonym inconsistency; and
`arXiv:2604.17126` measures prompt instability over 263 COCO images with six prompts.

## 2. The measurement, with its robustness

Median presence scalar, 15 disjoint tiles:

| prompt | median `s_i` | median `max_q` | fired (τ=0.3) |
|---|---|---|---|
| `crack` | 0.9062 | 0.8281 | 13/15 |
| `a crack in metal` | 0.3008 | 0.6914 | 7/15 |
| `thin dark line` | 0.1445 | 0.6562 | 1/15 |
| `fracture` | 0.0086 | 0.6328 | 0/15 |

Span 105.3×, and it survives permutation (20,000 shuffles: null median 3.16×, max 70.0× free;
3.48×/74.3× permuted within tile; **p < 1e-4** both ways). R² of logit(`s_i`) by prompt is
**0.8148** against a 4-level chance level of 0.041 (95th pct 0.130), p < 1e-4.

**But the span is one prompt, and it is not a synonym effect.** Drop-one-prompt:

| dropped | remaining span |
|---|---|
| `a crack in metal` | 105.30× |
| `thin dark line` | 105.30× |
| `crack` | 34.95× |
| **`fracture`** | **6.27×** |

Between the two phrases a materials scientist would actually use interchangeably — `crack` and
`a crack in metal` — it is **3.01×**. Meanwhile `max_q`, the decoder's best per-query score,
spans only **1.31×** across all four prompts: **the decoder proposes the same features whatever
word you type, and only the presence head collapses.** `fracture` reads as a bone/medical term
in a caption distribution. This is a vocabulary-grounding fact about one out-of-domain noun, not
"synonymous prompts give two orders of magnitude different answers".

**The ratio is also parameterisation-dependent** — the same two medians give 105.3× as a
probability ratio, **1114×** as an odds ratio, **7.02 nats** as a logit difference and **10.6×**
as a miss-probability ratio. Its size is set by how near the smaller median sits to zero.
Frame-clustered bootstrap over the 9 source frames: 95% CI **[77, 187]**. Quote no more than
two significant figures.

**"Which words, not which micrograph" is false.** Conditioned on prompt — correct for a fully
crossed design — image identity is significant, not chance: **F(14,42) = 3.94, p = 2.76e-4,
ω² = 0.078, partial R²(tile|prompt) = 0.568** against a 0.250 null, and **Kendall's W = 0.612
(p = 0.0019)** because the prompts rank tiles concordantly.

> These read F(15,45) = 4.14, p = 1.06e-4, ω² = 0.080, R² = 0.580, W = 0.627 until
> 2026-09-22. The degrees of freedom were the tell: this document is about **15** tiles × 4
> prompts, which gives df = (14, 42); (15, 45) is the **16**-tile design, i.e. the withdrawn
> contaminated set. Recomputed from the shipped `sam3_presence.json` on logit(presence), the
> transform this document uses elsewhere — raw presence gives F = 2.64 and log gives 2.85, so
> the transform has to be stated, and logit reproduces the published figure to three
> decimals. The conclusion is unchanged: image identity is significant conditioned on prompt. Both factors are real; the
prompt effect is roughly 39× larger per degree of freedom.

> **Do not quote raw R² across factors with different level counts.** A 15/16-level factor earns
> R² ≈ 0.23 by chance on this design; a 4-level factor earns ≈ 0.04. An earlier draft set 81.7%
> beside 10.6% as though the gap were the result. A later "correction" of mine — that tile sits
> below its own null — was also a misdiagnosis: it is below the *marginal* null only because the
> large prompt variance inflates the residual it is scored against. Compare each factor to its
> own null, and condition when the design is crossed.

## 3. Segmentation, at matched oracle budgets

This is the comparison the first draft got wrong. My baseline was tuned **per tile over 512
operating points chosen with the ground truth** while SAM 3 was **un-tuned at one fixed τ**. And
per-tile oracle thresholding is **OIS**, named in Arbeláez et al., TPAMI 33(5):898–916, 2011
(`10.1109/TPAMI.2010.161`), after Martin et al. 2004 (`10.1109/TPAMI.2004.1273918`), and standard
in crack segmentation (DeepCrack, `10.1109/TIP.2018.2878966`; OmniCrack30k,
`10.1109/CVPRW63382.2024.00392`). `baseline_matched.py`:

| arm | label access | median IoU | mean |
|---|---|---|---|
| Otsu, per tile | none | 0.0882 | 0.2495 |
| **SAM 3 union, `crack`, τ=0.3** | **none** | **0.1914** | **0.2546** |
| ODS — one threshold shared by all tiles | set-level (optimistic) | 0.2417 | 0.2614 |
| SAM 3 oracle instance | per tile (picks instance) | 0.3871 | 0.4129 |
| OIS — per-tile oracle threshold | per tile (512 points) | 0.4090 | 0.4024 |
| information-free constant-area | area only | 0.0000 | 0.0347 |

Paired Wilcoxon:

- **ODS vs SAM 3 union** (both un-tuned per tile): threshold wins 8/15, **p = 1.0000**
- **OIS vs SAM 3 oracle** (both per-tile oracle): threshold wins 9/15, **p = 0.8040**
- OIS vs SAM 3 union (**unmatched** — this is the comparison I published): 11/15, p = 0.0256

**At every matched budget the two are indistinguishable.** The only significant result is the
rigged one. Otsu — the sole rung with no label access at all — is beaten by SAM 3.

Other scores, `crack`, 15 tiles: median oracle IoU **0.3871**, median union recall **0.9691**,
firing **13/15** at τ=0.3. Frame-clustered bootstrap CIs on the earlier set were wide
([0.06, 0.39] for union IoU), so quote one decimal: union IoU ≈ 0.19, oracle ≈ 0.39, recall ≈ 0.97.

**Per-tile IoU does carry model information** — the opposite of what an earlier draft said. The
genuinely information-free baseline (predict the label's own area, arbitrarily placed) scores
median IoU **0.0000**, and SAM 3 beats it on **12/15** tiles. The earlier argument compared
`min(label_coverage/const, 1)` against label coverage — ρ = **+0.9989**, the same variable — and
reported the 0.001 difference as though it were two comparisons. It was a self-comparison, of
exactly the kind §1 of this document already names.

## 4. The contamination comparison was blind

> ⛔ **Recomputed 2026-09-22 on the shipped 15 tiles; the significance claim below did not
> survive.** Every figure in this section was from the 16-tile set. Paired on the 15 tiles
> common to both runs, with the **exact** signed-rank test rather than the normal
> approximation:
>
> | | contaminated | clean | discordant pairs | exact p |
> |---|---|---|---|---|
> | `crack`, union IoU | 0.0826 | **0.1914** (2.32×) | 12 of 15 | **0.8501** |
> | `crack`, precision | — | — | 12 of 15 | 0.6772 |
> | `a crack in metal`, IoU | 0.0000 | 0.0000 | 5 of 15 | **0.0625** |
> | `a crack in metal`, precision | 0.0000 | 0.0000 | 5 of 15 | **0.0625** |
>
> **p = 0.0312 is not obtainable here.** With 5 discordant pairs the smallest two-sided exact
> signed-rank p is 2/2⁵ = 0.0625, which is exactly what the test returns — it is the floor,
> not a result. (scipy's normal approximation gives 0.0431 at n = 5; that is an approximation
> artefact, and this is why the exact method has to be named.) So the sentence "the
> contamination *did* measurably change the scores" is **withdrawn**: at p = 0.0625 it does
> not clear 0.05, and the direction it points is unchanged only as a direction.
> Firing on `a crack in metal` moves **2/15 → 7/15**, not 2/16 → 8/16.

Earlier: "the leak did not measurably inflate the scores", from paired Wilcoxon p = 0.7869 (IoU)
and p = 0.8457 (recall) on `crack`. That is accepting a null from a test with a minimum
detectable effect of ΔIoU ≈ 0.084 — 44% of the clean median. It is demonstrably blind: on the
same pairing the median union IoU moves **0.0775 → 0.2039 (2.63×)** and the test still returns
p = 0.7869.

And on `a crack in metal` the identical paired design gives **p = 0.0312** on both IoU and
precision, with firing **2/16 → 8/16**. The contamination *did* measurably change the scores; I
had run the test on the one prompt where it did not show. Every point estimate favours the clean
arm (mean ΔIoU +0.0298, Δprecision +0.0441, Δrecall +0.1183), so the defensible statement is
directional and one-sided: **contamination did not inflate by more than ~0.08 IoU.**

"The burn-in suppressed alternative prompts" is 1 of 3: `thin dark line` moved the *other* way
(2/16 → 1/16) and `fracture` did not move (0/16). And the tile-level p = 0.0312 becomes
**p = 0.125** at frame level (6 flips, 4 frames) and after Bonferroni over 4 prompts. My
conclusion that the original brittleness finding was "overstated" is also wrong: the clean run
reproduces brittleness at the same statistic (`crack` 13/15 vs `a crack in metal` 7/15).

**An uncontrolled second difference.** My `to8()` percentile stretch does not reproduce the
overlay renderer on 6 frames — off-label MAE up to **19.1** grey levels. So "contaminated vs
clean" was never a single-variable contrast. `align_originals.py` was already reporting this as
ncc 0.9993 rather than 1.0000; I read it only as "registration certain" (true) and missed that it
also meant "the tone curve differs" (also true), because ncc is invariant to affine intensity
change. The correct control is to remove the burn-in *only* — inpaint the painted region from the
registered original — which is what a redo should do.

## 5. A real bug in the tile generator

`make_tiles.py` suppressed already-picked regions using the tile **origin** while `argmax`
returns the **centre** — an off-by-`S//2` = 512. `AS_24hr_BSE_Side_008` t0/t1 came out exactly
512 px apart: **50.0% overlap**; `260708…_001` t0/t1 at **56.6%**. The comment said "a second one
far from it". Two tiles sharing half their pixels are one observation reported as two — and those
two were exactly the tiles carrying the earlier draft's "SAM 3 wins outright" story.

Fixed twice: suppression moved into centre space, and then, because two disjoint 1024 tiles do
not *fit* on a 1490×1490 frame (the clip to `[0, H-S]` collapsed distinct centres to an 80%
overlap), an explicit rejection in origin space against every tile already taken. A frame that
cannot yield a disjoint second tile now contributes one. Result: **15 tiles, 0 overlapping
pairs** — three frames give one tile each.

## 6. Caveats that must travel with any of this

- **15 tiles from 9 frames**, ~9 of ~43 distinct fields, one material family. No crack-free
  negative control — the tile factor was built not to vary while the prompt factor was allowed to
  reach an out-of-vocabulary noun.
- **Community-mirror weights.** Not citable.
- **Labels are region assertions**, even in this fine-stroke subset (median brush 59 px against
  a ~3 px crack). Every IoU is indicative.
- **Frames are clipped at acquisition** (9.4% of pixels at exactly 0 on one; 69.0% at 65535 on
  another; 256 distinct values in a uint16 container on a third).
- **τ = 0.3 is a choice**; the library default is 0.5, and the firing counts move with it.
- **The synonym list was never pre-registered**, and the headline rests on one of its four items.
