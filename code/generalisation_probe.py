"""Does the detector work on micrographs it has never seen? No. This measures how badly.

Every number elsewhere in this repo comes from one corpus: 62 frames of one steel family,
imaged on one lab's instruments. That corpus is the whole evidence base, and nothing in it
tells you what happens on someone else's image. This script answers that separately, by
running the shipped detector on electron micrographs pulled from Wikimedia Commons.

    ./.venv/bin/python3 code/generalisation_probe.py

Needs a network connection: it fetches the images from Commons at run time rather than
committing them, because they are third-party CC BY / CC BY-SA works and this repo should not
redistribute them. Source page, licence and author for each are recorded in the output.

    docs/generalisation_probe.json      <- the result, with attribution per image

THERE IS NO GROUND TRUTH for these images, so this is not a benchmark and does not produce an
f1. What it produces is the fraction of each frame the detector claims as crack, plus an
overlay you can look at. That is enough, because the failures are not subtle:

  - a real SEM plan-view crack running across the frame: MISSED, 0.02% of frame claimed
  - a crack-free BSE frame of fly ash: 95.4% of the frame claimed as crack
  - two crack-free fracture surfaces: ~100 accepted regions each, all on topographic shadow
  - two other micrographs: zero candidates proposed, so no answer at all

For reference the same detector claims 5.35% of the frame on this project's own
260708_316_H_b2_front_CBS_002.

WHY IT FAILS, which matters more than that it fails. Pass 1 is a darkness threshold plus a
LogisticRegression over 8 morphology features, fitted on 16-bit ~27 MP frames of one material
at one contrast regime. None of that is invariant to contrast polarity, magnification, detector
mode, or material. Where the matrix is darker than the feature of interest the threshold
inverts its meaning; where the frame never clears the threshold there are no candidates at all.
So this is a detector for THIS corpus, and the measurement, review and provenance layers around
it are the part that travels.

Nothing here writes into original/ or interior_active_learning/: the images go to a temporary
directory and SEMCRACK_ORIGINAL_DIR points the pipeline at it.
"""
import json
import os
import re
import sys
import tempfile
import time
import traceback
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, ".."))
OUT = os.path.join(ROOT, "docs", "generalisation_probe.json")

#: Identify the script to Wikimedia, which asks that automated clients say who they are.
UA = "sem-crack-detector generalisation-probe (research evaluation; see repo README)"

#: Commons file titles, with what each one is. Chosen to be genuine electron micrographs --
#: an earlier version of this probe used photographs of walls and dried mud, which measure
#: nothing about an SEM detector.
WANT = {
    "File:SEM-image-showing-crack-propagation-of-TaCCNTsSiC.jpg":
        ("SEM_crack_propagation_TaC", "SEM, plan view, a crack propagating through a ceramic "
                                      "composite -- the closest analogue to this project"),
    "File:Back-Scattered Electron Micrograph of Coal Fly Ash.tif":
        ("SEM_flyash_BSE_NOcracks", "BSE-SEM, flat plan view, NO cracks (control)"),
    "File:Bruchfläche 500x.tif":
        ("SEM_bruchflaeche_500x", "SEM 500x, fracture surface of a metal, no crack"),
    "File:Bruchfläche 2000x.tif":
        ("SEM_bruchflaeche_2000x", "the same surface at 2000x"),
    "File:Ductile Fracture Surface 6061-T6 Al SEM.png":
        ("SEM_ductile_fracture_Al", "SEM, ductile fracture surface, 6061-T6 aluminium"),
    "File:Cubic ceramics.jpg":
        ("SEM_cubic_ceramics", "SEM, faceted ceramic grains"),
}


def _strip(html):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html or "")).strip()


def _resolve():
    """Ask Commons for the direct file URL, licence and author of each title."""
    q = ("https://commons.wikimedia.org/w/api.php?action=query&format=json&prop=imageinfo"
         "&iiprop=url|size|extmetadata"
         "&iiextmetadatafilter=LicenseShortName|Artist|LicenseUrl"
         "&titles=" + urllib.parse.quote("|".join(WANT)))
    req = urllib.request.Request(q, headers={"User-Agent": UA})
    pages = json.load(urllib.request.urlopen(req, timeout=90))["query"]["pages"]
    out = []
    for p in pages.values():
        ii = (p.get("imageinfo") or [{}])[0]
        if not ii.get("url"):
            print(f"  could not resolve {p['title']}", flush=True)
            continue
        em = ii.get("extmetadata") or {}
        key, note = WANT[p["title"]]
        out.append({"key": key, "commons_title": p["title"].replace("File:", ""),
                    "what_it_shows": note, "url": ii["url"],
                    "source_px": [ii.get("width"), ii.get("height")],
                    "license": _strip((em.get("LicenseShortName") or {}).get("value")),
                    "license_url": _strip((em.get("LicenseUrl") or {}).get("value")),
                    "author": _strip((em.get("Artist") or {}).get("value")),
                    "source_page": "https://commons.wikimedia.org/wiki/"
                                   + urllib.parse.quote(p["title"].replace(" ", "_"))})
    return out


def run():
    meta = _resolve()
    if not meta:
        print("nothing resolved -- is the network reachable?")
        return None
    tmp = tempfile.mkdtemp(prefix="semcrack_probe_")
    os.environ["SEMCRACK_ORIGINAL_DIR"] = tmp      # keeps this out of original/
    sys.path.insert(0, os.path.join(ROOT, "interior_active_learning", "code"))
    import unified_pipeline as up
    from interior_candidates import build_simple_overlay
    from skimage import measure as _m

    print(f"  detector: --sam2 {up.SAM2_MODE}   scratch: {tmp}\n", flush=True)
    for m in meta:
        req = urllib.request.Request(m["url"], headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=300) as r:
            blob = r.read()
        raw = os.path.join(tmp, m["key"] + ".bin")
        with open(raw, "wb") as fh:
            fh.write(blob)
        arr = np.asarray(Image.open(raw).convert("L"))
        Image.fromarray(arr).save(os.path.join(tmp, m["key"] + ".tif"))

        t0 = time.time()
        try:
            st = up.run_unified_pipeline(m["key"])
            lab, df = st["labeled"], st["df"]
            acc = df.loc[df["IsCrack"], "Label"].tolist()
            pred = np.isin(lab, acc)
            m.update({"candidates": int(len(df)), "accepted": int(len(acc)),
                      "flagged_px": int(pred.sum()),
                      "flagged_pct_of_frame": round(100.0 * pred.sum() / pred.size, 3),
                      "components": int(_m.label(pred, connectivity=2).max()),
                      "seconds": round(time.time() - t0, 1)})
            ov = os.path.join(tmp, f"{m['key']}_overlay.png")
            build_simple_overlay(st).save(ov)
            m["overlay_written_to"] = ov
            print(f"  {m['key']:28} {m['candidates']:5} cand {m['accepted']:5} acc "
                  f"{m['flagged_pct_of_frame']:8.3f}% of frame", flush=True)
        except Exception as e:
            m.update({"error": f"{type(e).__name__}: {e}",
                      "traceback": traceback.format_exc()[-700:]})
            print(f"  {m['key']:28} FAILED {type(e).__name__}: {e}", flush=True)

    payload = {
        "what_this_is": ("The shipped detector run on electron micrographs from Wikimedia "
                         "Commons, none from this project's corpus. No ground truth exists for "
                         "them, so this is a generalisation probe, not a benchmark."),
        "in_domain_reference": {"image": "260708_316_H_b2_front_CBS_002",
                                "flagged_pct_of_frame": 5.347,
                                "note": "same detector and code path, one of this project's "
                                        "own frames"},
        "detector": f"--sam2 {up.SAM2_MODE}",
        "overlays_written_to": tmp,
        "images": meta,
    }
    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\n  wrote {OUT}")
    print(f"  overlays are in {tmp} -- look at them, the numbers alone understate it")
    return payload


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        print("usage: generalisation_probe.py\n"
              "  no arguments. Needs a network connection: the micrographs are fetched from\n"
              "  Wikimedia Commons at run time rather than redistributed in this repo.")
        sys.exit(0)
    if len(sys.argv) > 1:
        print(f"unknown option {sys.argv[1]!r}. This script takes no arguments. Use --help.")
        sys.exit(2)
    run()
