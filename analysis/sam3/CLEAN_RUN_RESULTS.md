# SAM 3 on SEM cracks: the leak-gated run

*2026-09-18. Replaces `SAM3_ON_SEM_CRACKS.md`, which is void — its input had the label written
into it (`LEAK_POSTMORTEM.md`). Weights are the community mirror `1038lab/sam3`;
`facebook/sam3` is still gated (verified 401/gated:manual today), so **nothing here is citable**
until the run is repeated on official weights.*

Reproduce: `python align_originals.py && python make_tiles.py && python leak_check.py && SAM3_SAVE_MASKS=1 python run_real.py && python analyse_clean_run.py && python make_figures.py`

Inference is deterministic: an independent second run reproduced **union IoU on 64/64
(tile, prompt) pairs bit-identically**.

---

## 1. What is architecture, and what is measurement

This distinction is the whole of this document, because I got it wrong first.

**ARCHITECTURE — read from source, not measured, and therefore not a finding.**
`sam3_image_processor.py:195-200`:

```python
out_probs = out_logits.sigmoid()
presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
out_probs = (out_probs * presence_score).squeeze(-1)
keep = out_probs > self.confidence_threshold
```

One global presence scalar `s_i` per (image, prompt) rescales **all 200** per-query scores
(`num_queries=200`, `model_builder.py:185`), and the threshold is then applied **per query**.
Two consequences, both definitional:

- whether the image returns *anything* is decided by `s_i · max_j q_ij > τ`;
- the instance **count** is `#{j : q_ij · s_i > τ}` — there is no NMS, dedup or area filter
  after the gate (verified: no such call exists in the processor).

So a report that "the gate predicts emptiness on 64/64 pairs" proves nothing: the wrapper reads
the same tensors the library multiplies and thresholds with the same constant, and
`max_j(s·q_j) ≡ s·max_j q_j` because `sigmoid(presence) > 0`. **That figure is an
instrumentation self-check and is labelled as one below.** An earlier draft of this work
presented it as a 64-trial empirical result with an implied p-value. It was a tautology, and it
is the same error class as this project's area-fraction "identity" and its `forced/naive ≡
mean/median` tautology.

**MEASUREMENT — the part with content.** How large that scalar actually is, per prompt:

| prompt | median `s_i` | range | fired (τ=0.3) |
|---|---|---|---|
| `crack` | **0.9102** | 0.2158 – 0.9688 | 14/16 |
| `a crack in metal` | **0.3877** | 0.0317 – 0.8359 | 8/16 |
| `thin dark line` | **0.1157** | 0.0593 – 0.5195 | 1/16 |
| `fracture` | **0.0081** | 0.0004 – 0.0260 | 0/16 |

**A 112× span across four synonyms a materials scientist would use interchangeably**, and it
survives permutation: over 20,000 shuffles of the prompt labels the null span has median
**3.4×** and maximum **72.1×**, so **p = 0**. Permuting prompts *within* each tile — which
respects the fact that 16 tiles come from only 9 frames — gives null median 3.9×, max 65.3×,
again **p = 0**.

The variance is in the *prompt*, not the *image*: prompt identity explains **81.7%** of
logit(`s_i`) variance against a chance level of **0.038** for a 4-level factor (95th percentile
0.123, max 0.316 in 20,000 permutations), p = 0.

> **Compare each factor to its own null, not to each other.** Tile identity explains 10.6%, and
> an earlier draft set that beside 81.7% as though the gap were the result. That comparison is
> invalid: a 16-level factor earns R² ≈ **0.233** by chance on this design, so tile's 10.6% is
> *below* its own null (p = 0.98). The conclusion is if anything stronger — the image contributes
> no more than chance — but the two raw R² values are not comparable and must not be quoted
> side by side.

So the actionable statement is: *which words you type, not which micrograph you have, decides
whether SAM 3 reports anything at all.* `fracture` never comes within an order of magnitude of
any usable threshold.

**Honest baselines for the same question.** Predicting "did this pair return anything?":
always-empty gets **41/64 (64.1%)**, the best prompt-identity-only rule gets **53/64 (82.8%)**.
The gate rule gets 64/64 by construction. Quote 82.8% as the reference point, not 50%.

**τ is a choice.** τ = 0.3 here; the library default is 0.5 (`sam3_image_processor.py:17`). The
firing counts above move with τ. The presence *magnitudes* do not, which is why they are the
reportable quantity.

---

## 2. Segmentation quality, and the baseline it must beat

Prompt `crack`, all 16 tiles:

| metric | median |
|---|---|
| union IoU | **0.2039** |
| oracle IoU (best single returned instance, chosen *using* the label) | **0.3843** |
| union recall | **0.9585** |
| tiles returning ≥1 instance | **14/16** |

**The trivial baseline, which had never been stated.** An oracle-tuned single global grey
threshold — the best of 256 thresholds in both polarities, per tile — reaches **median IoU
0.3843** on the same tiles. It beats SAM 3's *union* everywhere and ties its *oracle* arm in
median. Per tile the two are unrelated (identical on 0/16; the threshold wins on 6/16, including
two tiles where SAM 3 returns nothing at all and thresholding still reaches IoU 0.60 and 0.44).
The median coincidence (0.384350 vs 0.384300) is coincidence and nothing more.

**Where SAM 3 does win, and where the score is meaningless** (`fig_qualitative.png`). On the
two tiles with the densest labels it beats the threshold outright — 0.728 vs 0.637 and 0.695 vs
0.668 — so it is not simply worse. On `MAR_Amb_AS_CBS_0001__t1` it returns **62** instances of
small dark features that look like genuine pits or micro-cracks and scores IoU **0.000**,
because the hand label on that tile marks one region at the frame edge and nothing else; the
threshold also collapses there (0.102). That tile is a label-disagreement case, not a model
failure, and it is why the aggregate medians should not be read as a verdict.

**Do not read per-tile IoU as model quality.** IoU here is dominated by how much of the tile the
label covers (Spearman ρ = +0.798, p = 0.0006, n = 14). A null model that predicts a *constant*
area fraction reproduces the precision ordering just as well (ρ = +0.757 against measured
precision, versus ρ = +0.758 for label coverage itself). The ranking is close to arithmetic, so
differences between tiles carry almost no information about the model. This is the same trap as
`../LABEL_GRANULARITY.md`: the labels are region assertions, median brush 59 px against a ~3 px
crack.

---

## 3. The contamination did not inflate the scores

Paired per (tile, prompt) against the void run, prompt `crack`:

- median ΔIoU **+0.0000**, median Δrecall **+0.0000**
- Wilcoxon on 13 non-zero IoU deltas: **p = 0.7869**; on 10 non-zero recall deltas: **p = 0.8457**

The burn-in was a dark curvilinear region where the crack was, so the model found a dark
curvilinear thing either way. **This does not rehabilitate the old numbers** — an input
containing the answer yields uninterpretable scores whether or not they happen to agree — but
it does mean the leak was not the reason SAM 3 scored poorly.

One thing the leak *did* change: it **suppressed alternative prompts**. `a crack in metal` fired
on **8/16** tiles clean versus **2/16** contaminated. So the original "only the bare noun
`crack` works" claim overstated the brittleness; the honest version is the 112× presence span
above.

---

## 4. Caveats that must travel with any of this

- **n = 16 tiles from 9 frames**, and those 9 frames are ~9 of ~43 distinct fields. Not 64
  independent trials; prompt identity dominates the variance anyway.
- **Community-mirror weights.** Not citable. `facebook/sam3` remains `gated: manual`.
- **The labels are not pixel-precise.** Only the fine-stroke subset (median brush ≤25 px) is
  used and even those are region assertions. Every IoU/Dice here is indicative.
- **Frames are clipped at acquisition** — one holds 9.4% of pixels at exactly 0, another 69.0%
  at 65535, one carries only 256 distinct values in a uint16 container. This inflates any
  threshold baseline honestly, and it is why threshold skill is not evidence of a leak.
- **τ = 0.3 ≠ library default 0.5.** Firing counts are τ-dependent.

## 5. What would make this publishable

The presence-magnitude measurement is the only thing here with a claim to novelty, and it has
**not yet been checked against prior art** — the literature sweep for it was killed by a session
limit before it ran. SAM 3 is recent, so assume 2026 work on presence heads and prompt
robustness exists until shown otherwise. Before writing anything: (a) repeat on official
weights, (b) sweep τ, (c) extend beyond four prompts with a pre-registered synonym list,
(d) search the prior art properly, (e) note that the falsifiable per-query count test is
*also* algebra, so there is no empirical claim to be had from the gate mechanism itself.
