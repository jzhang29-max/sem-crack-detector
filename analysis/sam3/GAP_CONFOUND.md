# The SAM 3 "empty gap"

> # ⛔ INVALID — built on a leaked experiment
> *2026-09-18.* The input was the green channel of the annotated overlay, so the label was
> visible to the model (`SAM3_ON_SEM_CRACKS.md`). A bare threshold scores recall 1.000 on
> 16/16. All three confound checks below are therefore moot. Worse, the gap is **forced**:
> `p_ij = q_ij · s_i` with one global presence scalar per image, so at fixed τ a single
> scalar flips every instance at once. My confound-2 note ("point mass at zero") had the
> right instinct and the wrong mechanism.
>
> Original follows as the record.
>
> ---

*2026-09-18. Tested before the wildcard agent reported, so its verdict can be read against this.*

## The observation

On 16 hand-labelled tiles, prompt `crack`: recall **0.969–1.000 on 10 tiles**, **exactly
0.000 on 6**, nothing between.

## Confound 1 — is it "fails on condition X" rather than all-or-nothing? NO.

| covariate | hit median | miss median | Mann–Whitney p |
|---|---|---|---|
| label % of tile | 1.58 | 1.67 | **0.958** |
| HFW (µm) | 336 | 1,164 | 0.546 |
| instances returned | 26 | 4 | 0.081 |

Detector is mixed on both sides (hits ETD 3 / CBS 5 / BSE 2; misses CBS 4 / ETD 2). And the
"it just returned nothing" story fails: **two of the six misses returned 56 and 7 instances**
— SAM found objects, none of which overlapped the label. One *hit* returned a single instance
and still scored 0.969.

So no measurable covariate explains the split. The pattern survives this check.

## Confound 2 — the gap is partly structural. THIS ONE BITES.

`recall = 0.000` is not a draw from a continuum; it is the **point mass** of the discrete
event "no pixel of the prediction intersects the label". A distribution with an atom at zero
is not evidence of bimodality. So the claim "bimodal with an empty gap" partly restates
"either the prediction intersects the label or it does not", which is a tautology about set
intersection.

**What is left after removing that is the non-trivial half:** when SAM 3 *does* intersect, it
covers **≥ 96.9 %** of the labelled region. Partial overlap essentially never happens. That is
the finding, if it is one.

## Confound 3 — it may be a property of the LABELS, not of SAM 3. UNTESTED, and decisive.

These labels are broad-brush region assertions (median stroke 59 px, max 413 px) over ~3 px
cracks — thick, contiguous blobs marking "the dark region". SAM 3 segments the dark region
too. **Two annotators of the same blob overlap almost totally or not at all**, so near-total
coverage may follow from the label geometry rather than from anything about the model.

This cannot be separated on the current data. It needs either pixel-precise labels (the
project's standing blocker) or replication on a public thin-structure dataset with thin
ground truth — DRIVE, CrackForest, DeepCrack, Massachusetts Roads. **Public replication also
removes the n = 16 objection permanently**, which is why it is the right next step regardless
of how the wildcard agent rules.

## Honest status

Confound 1 passed, confound 2 removes half the claim, confound 3 is unresolved and could
remove the rest. At n = 16 the emptiness of the gap is not established; it is a hypothesis
worth one afternoon on public data.
