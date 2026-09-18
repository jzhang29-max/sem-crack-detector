#!/usr/bin/env python3
"""Compare the leak-gated SAM 3 run against the contaminated one, tile-paired.

The contaminated results are recovered from git (commit 3ae43eb, the retraction commit, which
is the last one holding the invalid sam3_results.json) so the comparison is paired per
(tile, prompt) rather than two independent medians.

It also prints a gate check, which is an INSTRUMENTATION SELF-CHECK AND NOT A RESULT. SAM 3
computes
    out_probs = sigmoid(pred_logits) * sigmoid(presence_logit_dec);  keep = out_probs > tau
(sam3_image_processor.py:195-200). The wrapper in run_real.py reads those same tensors and
applies the same constant, and sigmoid(presence) > 0 so max_j(s*q_j) = s*max_j q_j identically.
"n == 0 iff s*max_q <= tau" therefore CANNOT FAIL except through an instrumentation bug. There
is no NMS, dedup or area filter after the gate either, so the instance count is algebra too.

The scalar does NOT move all instances together: the threshold is per query over
num_queries=200 (model_builder.py:185) and survivors return 1 to 62 of them.

The EMPIRICAL content is the scalar's MAGNITUDE per prompt, printed at the end, together with
the information-free baselines that are the honest reference point for "did this pair return
anything" -- always-empty 41/64, best prompt-identity rule 53/64.
"""
import json, subprocess, sys
import numpy as np

CONTAM_COMMIT = "3ae43eb"

def load_contaminated():
    try:
        out = subprocess.run(["git", "show", f"{CONTAM_COMMIT}:analysis/sam3/sam3_results.json"],
                             cwd="../..", capture_output=True, text=True, check=True).stdout
        return json.loads(out)
    except Exception as e:
        print(f"  (could not recover contaminated run: {e})")
        return None

clean = json.load(open("sam3_results.json"))
contam = load_contaminated()
pres = json.load(open("sam3_presence.json"))
PROMPTS = ["crack", "a crack in metal", "thin dark line", "fracture"]

def key(r):
    return (r["tile"], r["prompt"])

def table(rows, label):
    print(f"\n{label}")
    print(f"  {'prompt':<18} {'n>0':>5} {'med n':>6} {'med union IoU':>14} {'med oracle IoU':>15} {'med union rec':>14}")
    for p in PROMPTS:
        rs = [r for r in rows if "error" not in r and r["prompt"] == p]
        if not rs:
            continue
        nz = [r for r in rs if r.get("n", 0) > 0]
        f = lambda k, sub: np.median([x.get(k, 0) for x in sub]) if sub else float("nan")
        print(f"  {p:<18} {len(nz):>2}/{len(rs):<2} {f('n',rs):>6.0f} "
              f"{f('union_IoU',rs):>14.4f} {f('oracle_IoU',rs):>15.4f} {f('union_rec',rs):>14.4f}")

table(clean, "LEAK-GATED RUN (raw originals)")
if contam:
    table(contam, "CONTAMINATED RUN (overlay green channel) -- INVALID, for comparison only")

# ---------------------------------------------------------------- paired per tile+prompt
if contam:
    cm = {key(r): r for r in contam if "error" not in r}
    print("\nPAIRED DELTA, crack prompt (clean - contaminated)")
    print(f"  {'tile':<36} {'contam IoU':>11} {'clean IoU':>10} {'delta':>8} {'contam rec':>11} {'clean rec':>10}")
    d_iou, d_rec = [], []
    for r in clean:
        if "error" in r or r["prompt"] != "crack":
            continue
        o = cm.get(key(r))
        if not o:
            continue
        di = r.get("union_IoU", 0) - o.get("union_IoU", 0)
        dr = r.get("union_rec", 0) - o.get("union_rec", 0)
        d_iou.append(di); d_rec.append(dr)
        print(f"  {r['tile'][:35]:<36} {o.get('union_IoU',0):>11.4f} {r.get('union_IoU',0):>10.4f} "
              f"{di:>+8.4f} {o.get('union_rec',0):>11.4f} {r.get('union_rec',0):>10.4f}")
    if d_iou:
        from scipy.stats import wilcoxon
        print(f"\n  median delta IoU {np.median(d_iou):+.4f}   median delta recall {np.median(d_rec):+.4f}")
        for name, d in (("IoU", d_iou), ("recall", d_rec)):
            nzd = [x for x in d if x != 0]
            if len(nzd) >= 6:
                st, p = wilcoxon(nzd)
                print(f"  Wilcoxon on {name}: n={len(nzd)} non-zero deltas, W={st:.1f}, p={p:.4g}")

# ---------------------------------------------------------------- the forced-gap test
print("\nGATE SELF-CHECK (an identity, not a result -- see this file's docstring)")
print(f"  {'tile|prompt':<52} {'presence':>9} {'max_q':>7} {'gated':>7} {'tau':>5} {'pred':>5} {'n':>4}")
agree = dis = 0
byp = {}
for r in clean:
    if "error" in r:
        continue
    k = f"{r['tile']}|{r['prompt']}"
    pr = pres.get(k)
    if not pr:
        continue
    predicted_nonempty = pr["survives"]
    actual_nonempty = r.get("n", 0) > 0
    ok = predicted_nonempty == actual_nonempty
    agree += ok; dis += (not ok)
    byp.setdefault(r["prompt"], []).append(pr["presence"])
    if not ok or r["prompt"] == "crack":
        print(f"  {k[:51]:<52} {pr['presence']:>9.4f} {pr['max_q']:>7.4f} {pr['gated_max']:>7.4f} "
              f"{pr['tau']:>5.2f} {str(predicted_nonempty):>5} {r.get('n',0):>4}"
              + ("" if ok else "   <-- DISAGREES"))
print(f"\n  wrapper reproduces the library gate on {agree}/{agree+dis} pairs; {dis} disagree")
print(f"  (this is algebra. Honest baselines for the same question: always-empty "
      f"{sum(1 for r in clean if 'error' not in r and r.get('n',0)==0)}/64, "
      f"best prompt-identity rule 53/64)")
print("\n  THE MEASUREMENT -- presence scalar by prompt (prompt explains 81.7% of its")
print("  logit variance, tile only 10.6%):")
for p in PROMPTS:
    v = byp.get(p)
    if v:
        v = np.array(v)
        print(f"    {p:<18} median {np.median(v):.4f}   range {v.min():.4f}-{v.max():.4f}   "
              f"n above tau-implied floor: {(v > 0).sum()}")
