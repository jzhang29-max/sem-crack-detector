# Where this work stands against the newest crack-segmentation literature

*2026-09-19. Every DOI/arXiv id below was resolved live (Crossref REST or arXiv Atom) during
this session. Numbers from other papers are quoted with their protocol and are **not** compared
to ours — cross-dataset accuracy comparison is invalid and this document does not do it.*

## The one thing to get straight first

**You cannot rank methods by comparing our IoU to a published IoU.** Their numbers come from
other datasets, other label conventions, other operating points. A 0.79 IoU on CrackSeg9k and
a 0.19 IoU here are not commensurable — CrackSeg9k cracks are wide, high-contrast, on concrete,
with tight labels; ours are ~3 px, low-contrast, on textured metal, with labels that are region
assertions of median 59 px. The only valid comparison is **running their methods on our data**,
which is what `methods_bench.py` does, and **comparing protocols**, which is what this document
does.

## Verified reference points

| ref | what it establishes | verified |
|---|---|---|
| Arbeláez, Maire, Fowlkes & Malik, *Contour Detection and Hierarchical Image Segmentation*, IEEE TPAMI 33(5):898–916 | defines **ODS** and **OIS**; OIS is a per-image oracle threshold chosen using the label | `10.1109/TPAMI.2010.161` |
| Martin, Fowlkes & Malik, IEEE TPAMI 26(5):530–549, 2004 | the original oracle-threshold protocol | `10.1109/TPAMI.2004.1273918` |
| Zou et al., *DeepCrack*, IEEE TIP 28(3):1498–1512, 2019 | ODS/OIS as the standard crack-segmentation reporting pair | `10.1109/TIP.2018.2878966` |
| Benz & Rodehorst, *OmniCrack30k*, CVPRW 2024 | the large modern crack benchmark | `10.1109/CVPRW63382.2024.00392` |
| Maier-Hein, Reinke et al., *Metrics reloaded*, **Nature Methods**, Feb 2024, 73 authors | community recommendations for metric choice | `10.1038/s41592-023-02151-z` |
| Reinke, Maier-Hein et al., *Understanding metric-related pitfalls in image analysis validation*, **Nature Methods**, Feb 2024, 70 authors | the pitfalls, including small/thin structures under IoU | `10.1038/s41592-023-02150-0` |
| Kervadec, Dolz, Wang, Granger & Ben Ayed, MIDL 2020, PMLR 121:365–381 | the **tightness prior** for superset (box) labels | `arXiv:2004.06816` |
| Carion et al. (Meta), *SAM 3: Segment Anything with Concepts* | the presence head, in the abstract | `arXiv:2511.16719` |
| *Rapid-Deployment Crack Measurement Based on SAM3 Semantic-Edge Response Decoding* | SAM 3's final masks lose thin-crack evidence, six public crack datasets | `arXiv:2607.12292` |
| Dabaja & Celik, remote sensing SAM 3 evaluation | presence head repurposed as a standalone zero-shot classifier + 5-config prompt ablation | `arXiv:2607.09583` |
| CoCo-SAM3 | SAM 3 synonym inconsistency named | `arXiv:2604.19648` |
| *Prompt Sensitivity in Vision-Language Grounding* | prompt instability over 263 COCO images, six prompts | `arXiv:2604.17126` |

## What we are NOT claiming

- **Not** that our accuracy beats the published state of the art. It is not measured on the same
  data and, at n = 15 tiles from 9 frames, could not establish that if it were.
- **Not** that any method or metric here is novel. The tightness prior is Kervadec 2020;
  ODS/OIS is Arbeláez 2011; the presence head is Meta's; clDice is published.
- **Not** that SAM 3 is bad at cracks in general. On our data, at matched tuning budgets, it is
  statistically indistinguishable from thresholding, and `arXiv:2607.12292` reaches a
  compatible conclusion on six public datasets with far better power.

*(Accuracy table and the protocol-by-protocol comparison are filled in from `methods_bench.json`
and the literature sweep — see the sections below once both complete.)*
