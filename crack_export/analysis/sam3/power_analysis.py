#!/usr/bin/env python3
"""How many frames would it take to tell these methods apart?

Every comparison in POSITION_VS_2026.md is non-significant, and the write-up says "the corpus
is the binding constraint, not the model". That is an assertion until it has a number attached.
This script attaches one.

Method: bootstrap over the 9 SOURCE FRAMES (not the 15 tiles -- two tiles of one frame share a
specimen, an instrument setting and an operator). For a target frame count n, resample n frames
with replacement, take their tiles, and run the same paired Wilcoxon the main analysis uses.
Power at n is the fraction of 2,000 such resamples that reach p < 0.05.

This is a bootstrap power curve, not an analytic one, so it inherits the observed effect size
and its uncertainty -- which is the point: it answers "at the effect size we actually see, how
many frames would we need", not "how many for some effect we assume".

Caveat worth stating: resampling 9 frames to simulate 40 assumes the new frames look like these
nine. They will not exactly. Read the numbers as an order of magnitude.
"""
import os, json
import numpy as np
from scipy.stats import wilcoxon

SC = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(20260919)
NBOOT = 1000
NS = [9, 15, 25, 40, 60, 100, 150]


def paired_power(a, b, tiles, tframe, frames):
    """Power curve for the paired comparison of per-tile vectors a and b."""
    idx = {t: i for i, t in enumerate(tiles)}
    by = {}
    for t in tiles:
        by.setdefault(tframe[t], []).append(idx[t])
    out = {}
    for n in NS:
        hits = 0
        for _ in range(NBOOT):
            pick = rng.choice(frames, n, replace=True)
            ii = [i for f in pick for i in by[f]]
            d = [a[i] - b[i] for i in ii]
            nz = [x for x in d if x != 0]
            if len(nz) < 6:
                continue
            try:
                # asymptotic, not exact: scipy computes the exact signed-rank distribution
                # for small samples, which is combinatorial and made this run for over ten
                # minutes. The normal approximation is what the main analysis's n supports
                # anyway, and it is the conservative choice here.
                if wilcoxon(nz, method="asymptotic").pvalue < 0.05:
                    hits += 1
            except ValueError:
                pass
        out[n] = hits / NBOOT
    return out


def main():
    b = json.load(open(f"{SC}/methods_bench.json"))
    om = json.load(open(f"{SC}/omnicrack_eval.json"))
    tiles = b["_tiles"]; tframe = b["_frames"]
    frames = sorted(set(tframe.values()))
    res = {(r["tile"], r["prompt"]): r for r in json.load(open(f"{SC}/sam3_results.json"))
           if "error" not in r}
    sam_c = [res.get((t, "crack"), {}).get("union_clDice", 0.0) for t in tiles]

    pairs = [
        ("OmniCrack30k vs global threshold", om["clDice"]["per_tile"],
         b["global threshold|clDice"]["per_tile_LOFO"]),
        ("OmniCrack30k vs Meijering ridge", om["clDice"]["per_tile"],
         b["Meijering ridge|clDice"]["per_tile_LOFO"]),
        ("Meijering ridge vs SAM 3", b["Meijering ridge|clDice"]["per_tile_LOFO"], sam_c),
        ("global threshold vs SAM 3", b["global threshold|clDice"]["per_tile_LOFO"], sam_c),
    ]

    print(f"Bootstrap power to reach p < 0.05, clDice, {NBOOT} resamples over source frames")
    print(f"(observed corpus = 9 frames / 15 tiles)\n")
    print(f"  {'comparison':<36} {'obs delta':>10} " + " ".join(f"{'n='+str(n):>6}" for n in NS))
    out = {}
    for name, a, c in pairs:
        pw = paired_power(a, c, tiles, tframe, frames)
        d = float(np.median([x - y for x, y in zip(a, c)]))
        print(f"  {name:<36} {d:>+10.4f} " + " ".join(f"{pw[n]:>6.2f}" for n in NS))
        out[name] = {"median_delta": d, "power": {str(k): v for k, v in pw.items()}}

    print(f"\n  Frames needed for 80% power (interpolated):")
    for name, v in out.items():
        ns = [int(k) for k in v["power"]]
        ps = [v["power"][str(k)] for k in ns]
        need = next((n for n, p in zip(ns, ps) if p >= 0.80), None)
        print(f"    {name:<36} {'~' + str(need) + ' frames' if need else 'more than ' + str(max(ns)) + ' frames'}")
    json.dump(out, open(f"{SC}/power_analysis.json", "w"), indent=1)
    print("\nwrote power_analysis.json")


if __name__ == "__main__":
    main()
