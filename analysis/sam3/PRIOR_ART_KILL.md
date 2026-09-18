# The presence-scalar finding is owned, and the threshold baseline is a named protocol

*2026-09-18. Every reference below was resolved live against the arXiv API in this session and
quoted from the retrieved abstract, not from memory.*

## 1. Meta's own SAM 3 paper owns the mechanism — it is in the abstract

**Carion, Gustafson, Hu, Debnath, Hu et al. (38 authors, Meta Superintelligence Labs), "SAM 3:
Segment Anything with Concepts", arXiv:2511.16719, 2025-11-20.**

Retrieved abstract, verbatim:

> "**Recognition and localization are decoupled with a presence head, which boosts detection
> accuracy.**"

That is the whole of what I was calling a finding. The body goes further than my write-up did:
the presence token feeds an MLP head producing a single scalar **"crucially shared by all object
queries"**, and **"during inference, we use the product of the global presence score and the
object score as the total object score."**

Worse for the "empty output is a recognition event, not a segmentation failure" framing: that is
Meta's *evaluation protocol*, not an observation. Their headline metric factorises exactly that
way — `cgF1 = 100 · pmF1 · IL_MCC`, where `IL_MCC` is a binary image-level "is the object
present?" scored **without regard for mask quality** and `pmF1` measures localisation on
positives only. They also ablate presence supervision across four settings (Table 9) and
separately replace the presence score with an external verifier for **+7.2 cgF1**, explaining it
as "a better calibration of object scores."

So all three limbs — the scalar, the product, and the gate/segmentation factorisation — are
Meta's, published ten months before I measured them.

## 2. The read-out-as-diagnostic step was published twice within months

**Dabaja & Celik, "Promptable Concept Segmentation from Above: Evaluating SAM 3's Zero-Shot and
One-Shot Capabilities in Remote Sensing", arXiv:2607.09583, 2026-07-10.** Verbatim:

> "we introduce a structural adaptation of SAM 3 by **repurposing its decoupled binary presence
> head into a standalone zero-shot classifier**. Furthermore, by **systematically isolating
> textual and visual prompt modalities across five configurations**, we explicitly diagnose the
> alignment mechanics within the model's multimodal decoder."

That is the read-out *and* the prompt ablation, in an out-of-distribution domain, at larger n
than 15 tiles.

**SegEarth-OV3, arXiv:2512.08730, 2025-12-09.** Verbatim:

> "we **utilize the presence score from the presence head to filter out categories that do not
> exist in the scene**, reducing false positives caused by the vast vocabulary"

Three weeks after SAM 3's release, the presence score was already being used as a one-number
existence gate.

## 3. SAM 3 failing on thin low-contrast cracks is also published

**"Rapid-Deployment Crack Measurement Based on SAM3 Semantic-Edge Response Decoding",
arXiv:2607.12292, 2026-07-14.** Verbatim:

> "We identify an **output-interface mismatch in SAM3**: its prompt-conditioned semantic response
> preserves crack evidence that is **often suppressed or spatially distorted in the final
> candidate masks**. Across six [public crack datasets] …"

This is SAM 3, on cracks, with the finding that the final masks lose thin-crack evidence —
across six datasets rather than nine frames.

## 4. The "trivial baseline nobody stated" has been the standard protocol since 2004

I claimed an oracle-tuned single global threshold was a bar nobody had written down. It has a
name, two names in fact, and they are the default reporting pair in this literature:

- **Martin, Fowlkes & Malik, IEEE TPAMI 26(5):530–549, 2004**, DOI `10.1109/TPAMI.2004.1273918`
- **Arbeláez, Maire, Fowlkes & Malik, "Contour Detection and Hierarchical Image Segmentation",
  IEEE TPAMI 33(5):898–916, 2011**, DOI `10.1109/TPAMI.2010.161` — defines **ODS** (one
  oracle-tuned threshold for the whole dataset) and **OIS** (oracle-tuned threshold **per
  image, chosen using the label**).

My "best of 256 thresholds in both polarities, per tile" **is OIS**. And ODS/OIS are the standard
operating-point pair in crack segmentation specifically:

- **DeepCrack, IEEE TIP 28(3):1498–1512, 2019**, DOI `10.1109/TIP.2018.2878966`
- **OmniCrack30k, CVPRW 2024**, DOI `10.1109/CVPRW63382.2024.00392`

## 5. And the comparison I drew from it was rigged

Independently of prior art, the claim "SAM 3 fails to beat a trivial threshold" does not survive
a matched protocol. My baseline was tuned **per tile over 512 operating points chosen with the
ground truth**; SAM 3's union arm was **un-tuned at a single fixed τ**. Matched (see
`baseline_matched.py`, which reports all four rungs):

| arm | label access |
|---|---|
| Otsu, per tile | none |
| ODS — one threshold shared by all tiles | set-level |
| OIS — per-tile oracle over 512 points | per tile |
| SAM 3 union at fixed τ | none |

On the earlier 16-tile set the agent that caught this measured: ODS median IoU 0.2072 vs SAM 3
union 0.2039, paired Wilcoxon **p = 0.9799**, with **SAM 3 ahead on the mean** (0.2453 vs
0.2353); Otsu — the only rung with no label access at all — **loses** at median 0.0738; and even
OIS versus SAM 3's union reaches only **p = 0.1046** despite holding the oracle. There is no gap
to report.

## Bottom line

Nothing here is a contribution. The mechanism is in Meta's abstract, the read-out is published
twice over, SAM-3-is-poor-on-cracks is published on six datasets, the baseline is a 2004
protocol, and the one comparison that looked like a result is an artifact of unmatched tuning.

What remains is a **local measurement on this corpus** with honest caveats — see
`CLEAN_RUN_RESULTS.md`. It is worth keeping as an internal record and as a reason not to trust
text-prompted SAM 3 on this material. It is not worth writing up.
