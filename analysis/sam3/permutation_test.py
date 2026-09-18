#!/usr/bin/env python3
"""Permutation nulls for the presence-magnitude claim.

Two things this exists to stop.

1. A ratio of two medians over 16 non-independent tiles (they come from 9 source frames) is not
   self-evidently meaningful. So the 112x span is tested against shuffled prompt labels, both
   freely and WITHIN each tile -- the within-tile version respects the clustering.

2. R^2 by a 4-level factor and R^2 by a 16-level factor are NOT comparable. On this design a
   16-level factor earns R^2 ~ 0.233 by chance while a 4-level factor earns ~0.038. An earlier
   draft quoted "prompt 81.7% vs tile 10.6%" side by side as though the gap were the finding;
   in fact tile's 10.6% is BELOW its own null. Each factor is compared to its own null here.

Seeded, so the printed numbers are reproducible.
"""
import json, os
import numpy as np

SC = os.path.dirname(os.path.abspath(__file__))
PROMPTS = ["crack", "a crack in metal", "thin dark line", "fracture"]
NPERM = 20000
rng = np.random.default_rng(20260918)

pres = json.load(open(f"{SC}/sam3_presence.json"))
rows = [(k.split("|")[0], k.split("|", 1)[1], v["presence"]) for k, v in pres.items()]
tiles = sorted({t for t, _, _ in rows})
raw = np.array([s for _, _, s in rows])
lg = np.array([np.log(min(max(s, 1e-6), 1 - 1e-6) / (1 - min(max(s, 1e-6), 1 - 1e-6)))
               for s in raw])
tile_lab = [t for t, _, _ in rows]
prompt_lab = [p for _, p, _ in rows]


def r2(vals, groups):
    g = {}
    for v, k in zip(vals, groups):
        g.setdefault(k, []).append(v)
    sst = ((vals - vals.mean()) ** 2).sum()
    ssw = sum(((np.array(v) - np.mean(v)) ** 2).sum() for v in g.values())
    return 1 - ssw / sst


def span(labels):
    m = [np.median(raw[[i for i, l in enumerate(labels) if l == p]]) for p in PROMPTS]
    return max(m) / max(min(m), 1e-12)


def within_tile_perm():
    lab = list(prompt_lab)
    for t in tiles:
        idx = [i for i, x in enumerate(tile_lab) if x == t]
        for i, v in zip(idx, rng.permutation([lab[i] for i in idx])):
            lab[i] = v
    return lab


def report(name, obs, null, higher_is_extreme=True):
    p = (null >= obs).mean() if higher_is_extreme else (null <= obs).mean()
    print(f"  {name:<44} obs {obs:8.4f} | null med {np.median(null):7.4f} "
          f"95th {np.percentile(null, 95):7.4f} max {null.max():7.4f} | p = "
          + ("< 1e-4 (0 of %d)" % NPERM if p == 0 else f"{p:.4g}"))


if __name__ == "__main__":
    print(f"{NPERM} permutations, seed 20260918\n")
    report("R2 of logit(presence) by PROMPT (4 levels)", r2(lg, prompt_lab),
           np.array([r2(lg, list(rng.permutation(prompt_lab))) for _ in range(NPERM)]))
    report("R2 of logit(presence) by TILE (16 levels)", r2(lg, tile_lab),
           np.array([r2(lg, list(rng.permutation(tile_lab))) for _ in range(NPERM)]))
    print("    ^ a 16-level factor earns ~0.23 for free; tile sits BELOW its own null.\n"
          "      Never quote the two raw R2 values against each other.\n")
    report("presence span, free permutation", span(prompt_lab),
           np.array([span(list(rng.permutation(prompt_lab))) for _ in range(NPERM)]))
    report("presence span, permuted WITHIN tile", span(prompt_lab),
           np.array([span(within_tile_perm()) for _ in range(NPERM)]))
