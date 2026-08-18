#!/usr/bin/env bash
# Assemble the distributable repo.
#
#   ./make_package.sh [dest]        default: ~/Desktop/sem-crack-detector
#
# Why a separate repo rather than slimming this one: this project's git history is
# ~4.7 GB because paint templates and result images were committed early on.
# .gitignore cannot help -- it does not apply to already-tracked files -- and the
# only way to shrink the history is to rewrite it, which is destructive. So this
# copies out the parts a user needs into a fresh repo, and leaves this folder as
# the lab's full data archive. That archive repo is now archived on GitHub
# (read-only); THIS script's output is the one live repo.
#
# Included: all live code, the trained models, the human correction masks and
# label log (the irreplaceable part), the docs, the analysis/benchmark scripts
# behind the reported numbers, and THE SOURCE IMAGES, losslessly compressed.
# Excluded: derived outputs -- results, figures, paint templates, caches.
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
DEST="${1:-$HOME/Desktop/sem-crack-detector}"

echo "==> packaging from $SRC"
echo "    into           $DEST"

# Scripts superseded by something else in the live tree. They stay in the package
# under archive/superseded_code/ -- deleted would lose provenance, but left in
# code/ they actively mislead: there were FOUR scripts whose docstring says they
# render overlays, and only regenerate_templates.py is the one the app runs.
SUPERSEDED_CODE=(pipeline_stages.py generate_scientific_diagram.py batch_apply.py train_classifier.py)
SUPERSEDED_IAL=(build_fusion_cache.py eval_fusion.py render_final_overlays.py
                render_v3_overlays.py sam_union_overlays.py)

# Preserve .git AND original/ across rebuilds. .git: this used to `rm -rf "$DEST"`
# outright, destroying the history and the configured remote, so the next push
# silently tracked nothing. original/: the images are 1.2 GB compressed and
# re-deriving them every rebuild would mean re-reading 2.55 GB for no gain.
STASH="$(mktemp -d)"
[ -d "$DEST/.git" ]      && mv "$DEST/.git" "$STASH/git"           && echo "    (preserving git history and remote)"
[ -d "$DEST/original" ]  && mv "$DEST/original" "$STASH/original"  && echo "    (preserving packaged images)"
rm -rf "$DEST"
mkdir -p "$DEST"
[ -d "$STASH/git" ]      && mv "$STASH/git" "$DEST/.git"
[ -d "$STASH/original" ] && mv "$STASH/original" "$DEST/original"
rm -rf "$STASH"
mkdir -p "$DEST"/{code,interior_active_learning/code,interior_active_learning/paint,interior_active_learning/labels,models,original,training_data,figures,docs}
mkdir -p "$DEST"/archive/superseded_code

# --- code, minus the superseded scripts ---
for f in "$SRC"/code/*.py; do
  b="$(basename "$f")"; d="$DEST/code/$b"
  for s in "${SUPERSEDED_CODE[@]}"; do [ "$b" = "$s" ] && d="$DEST/archive/superseded_code/$b"; done
  cp "$f" "$d"
done
for f in "$SRC"/interior_active_learning/code/*.py; do
  b="$(basename "$f")"; d="$DEST/interior_active_learning/code/$b"
  for s in "${SUPERSEDED_IAL[@]}"; do [ "$b" = "$s" ] && d="$DEST/archive/superseded_code/$b"; done
  cp "$f" "$d"
done

# --- models ---
# TWO directories, and missing the second one silently degrades the app:
#   models/                          Pass 1 classifier (crack_classifier.joblib)
#   interior_active_learning/models/ Pass 2 model (unified_model.joblib)
# unified_pipeline._load_unified_bundle() returns None when that file is absent
# and Pass 2 is then skipped WITHOUT WARNING -- quietly costing 27% of all crack
# regions. An earlier version of this script copied only the first directory, so
# a fresh clone ran a weaker detector than the benchmarked one, silently.
#
# crack_classifier_v3_weighted.joblib and crack_classifier_PRE_V3_BACKUP.joblib
# are NOT copied: both were byte-identical to crack_classifier.joblib, so the
# model dropdown listed the same model three times under three names. The first
# is a build artifact that every retrain rewrites; the second is a stale backup
# whose job belongs to crack_classifier_PREV.joblib, which the retrain gate
# writes itself.
mkdir -p "$DEST"/interior_active_learning/models
for f in "$SRC"/models/*.joblib; do
  case "$(basename "$f")" in
    crack_classifier_v3_weighted.joblib|crack_classifier_PRE_V3_BACKUP.joblib) continue;;
  esac
  cp "$f" "$DEST"/models/
done
cp "$SRC"/models/*_metrics.json                   "$DEST"/models/ 2>/dev/null || true
cp "$SRC"/interior_active_learning/models/*.joblib \
   "$DEST"/interior_active_learning/models/ 2>/dev/null || true

# --- the irreplaceable part: human verdicts ---
cp "$SRC"/interior_active_learning/paint/*_correction_mask.png \
   "$DEST"/interior_active_learning/paint/ 2>/dev/null || true
cp "$SRC"/interior_active_learning/labels/*.csv \
   "$DEST"/interior_active_learning/labels/ 2>/dev/null || true
cp "$SRC"/interior_active_learning/paint/candidate_counts.json \
   "$DEST"/interior_active_learning/paint/ 2>/dev/null || true
cp "$SRC"/training_data/labeled_regions.csv       "$DEST"/training_data/ 2>/dev/null || true
cp "$SRC"/manual_corrections_ledger.csv           "$DEST"/ 2>/dev/null || true

# --- the source images, losslessly compressed ---
# The TIFFs are uncompressed uint16: 53.8 MB each, 2.55 GB for 62. zlib level 6
# takes them to 46% with BIT-IDENTICAL pixel data, so a clone is ~1.2 GB and the
# pipeline computes exactly the same numbers. Identity is asserted per image, not
# assumed: a silently-altered source image would corrupt every downstream result
# while looking fine. Output is byte-deterministic, so an unchanged image
# produces an unchanged file and git sees no churn across rebuilds.
echo "    compressing images (lossless, verified)"
SRC="$SRC" DEST="$DEST" python3 <<'IMGPY'
import hashlib, json, os, numpy as np, tifffile
from multiprocessing.pool import ThreadPool     # THREADS, not processes: this runs
# from a heredoc, so __main__ has no file for macOS's spawn-based Pool to
# re-import in the children, and it deadlocks silently -- observed 21 minutes at
# 0.1% CPU before being killed. tifffile's I/O and zlib both release the GIL, so
# threads parallelise this fine.

S = os.path.join(os.environ["SRC"], "original")
D = os.path.join(os.environ["DEST"], "original")
os.makedirs(D, exist_ok=True)
MAN = os.path.join(D, ".manifest.json")
man = {}
if os.path.exists(MAN):
    try:
        man = json.load(open(MAN))
    except Exception:
        man = {}

def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()

def one(f):
    """Compress if the source changed; otherwise verify the packaged copy cheaply.

    The first version re-read and array-compared every source AND packaged image
    on every rebuild: 5 GB of reads to conclude nothing had changed. The manifest
    records the source's size+mtime and the packaged file's md5, so an unchanged
    source with an intact copy costs one 25 MB hash instead of two full decodes --
    the same guarantee, seconds instead of minutes.
    """
    sp, dp = os.path.join(S, f), os.path.join(D, f)
    st = os.stat(sp)
    rec = man.get(f)
    if (rec and os.path.exists(dp)
            and rec.get("src_size") == st.st_size
            and rec.get("src_mtime_ns") == st.st_mtime_ns
            and rec.get("dst_md5") == md5(dp)):
        return f, rec, "cached"
    a = tifffile.imread(sp)
    tmp = dp + ".part"
    tifffile.imwrite(tmp, a, compression="zlib", compressionargs={"level": 6})
    # bit-identity asserted, never assumed: a silently-altered source image would
    # corrupt every downstream result while looking perfectly fine
    if not np.array_equal(a, tifffile.imread(tmp)):
        os.remove(tmp)
        raise RuntimeError(f"{f} did not round-trip losslessly; not written")
    os.replace(tmp, dp)
    return f, {"src_size": st.st_size, "src_mtime_ns": st.st_mtime_ns,
               "dst_md5": md5(dp), "dst_size": os.path.getsize(dp)}, "compressed"

if os.path.isdir(S):
    files = sorted(x for x in os.listdir(S) if x.lower().endswith((".tif", ".tiff")))
    # images deleted from the source must disappear here too, or every rebuild
    # resurrects one the user removed
    for x in os.listdir(D):
        if x.lower().endswith((".tif", ".tiff", ".part")) and x not in files:
            os.remove(os.path.join(D, x))
            print("      removed %s (gone from source)" % x)
    out, n_new = {}, 0
    with ThreadPool(4) as pool:
        for f, rec, note in pool.imap_unordered(one, files):
            out[f] = rec
            n_new += (note == "compressed")
    json.dump(out, open(MAN, "w"), indent=1, sort_keys=True)
    tot = sum(r["dst_size"] for r in out.values())
    print("      %d images, %.2f GB (%d compressed, %d unchanged and verified)"
          % (len(files), tot / 1e9, n_new, len(files) - n_new))
IMGPY

# --- docs and entry points ---
# run_app.sh is not copied: it was a one-line `exec ./run` shim, byte-identical in
# effect, so the repo shipped two ways to start the same app.
cp "$SRC"/requirements.txt "$SRC"/run "$SRC"/Makefile "$DEST"/
cp "$SRC"/MODEL_VALIDATION_BENCHMARK.md "$SRC"/PIPELINE_DEEP_DIVE.md "$SRC"/APP_COMPARISON.md "$DEST"/docs/ 2>/dev/null || true
# The app README's source of truth is PACKAGE_README.md HERE -- edit that, not the
# copy in the package, which every rebuild overwrites. It is written only as
# README.md; shipping it under both names meant two identical readmes and no way
# to tell which one to edit.
cp "$SRC"/PACKAGE_README.md "$DEST"/README.md

# The package tracks images; this project does not (its history is already too
# large). So the two .gitignore files genuinely differ and the app's is written
# here rather than copied.
cat > "$DEST"/.gitignore <<'GI'
# Derived outputs -- all regenerable from original/ plus the correction masks.
results/
figures/
fusion_cache/
removed_images/
training_data/*.png
training_data/*_cracks.csv
interior_active_learning/paint/*_paint_template.png
interior_active_learning/paint/*_painted.png
interior_active_learning/candidates/*_quicklook.png

# Build artifacts of a retrain: train_v3_weighted.py writes the candidate model,
# and the retrain gate writes the rollback copy. Committing them meant the model
# dropdown showed three names for one identical file.
models/crack_classifier_v3_weighted.joblib
models/crack_classifier_PREV.joblib
models/crack_classifier_PRE_V3_BACKUP.joblib

# original/ IS tracked here, unlike in the research project: without the images a
# clone can retrain from the shipped labels but cannot reproduce a single overlay.

.venv/
__pycache__/
*.pyc
.DS_Store
.upload_*
GI

# --- analysis provenance ---
# The archive/ scripts are the evidence behind the numbers in
# docs/MODEL_VALIDATION_BENCHMARK.md: the label-bias experiment that found 94% of
# negatives came from one image, the CV diagnosis, the SAM characterisation, the
# superseded models kept as counterexamples. Small, and the only record of HOW
# the reported results were arrived at.
cp -R "$SRC"/archive/. "$DEST"/archive/ 2>/dev/null || true
# The benchmark/experiment scripts that PRODUCED the figures and tables in the
# benchmark doc -- ROC curves, confusion matrices, learning curves, model
# comparisons, and the rejected alternatives (per-type models, synthetic
# negatives, margin maximisation). Without these the documented numbers have no
# reproducible source.
mkdir -p "$DEST"/interior_active_learning/code/experiments
cp "$SRC"/interior_active_learning/code/experiments/*.py \
   "$DEST"/interior_active_learning/code/experiments/ 2>/dev/null || true
# and the packager itself, so the single repo can rebuild its own distribution
cp "$SRC"/make_package.sh "$DEST"/ 2>/dev/null || true

# The diagram sources, but only the CURRENT architecture's. The pre-unified
# workflow SVG and command_guide.svg (a guide to commands this app no longer has,
# now that setup is one command) go to archive/ instead of docs/, where they read
# as current.
mkdir -p "$DEST"/docs/diagram "$DEST"/archive/superseded_diagrams
cp "$SRC"/pipeline_diagram/full_workflow_unified_*.svg "$DEST"/docs/diagram/ 2>/dev/null || true
cp "$SRC"/pipeline_diagram/command_guide.svg "$SRC"/pipeline_diagram/full_workflow_260708_*.svg \
   "$DEST"/archive/superseded_diagrams/ 2>/dev/null || true
chmod +x "$DEST"/run

# keep empty dirs in git so a fresh clone has somewhere to put things
for d in figures training_data interior_active_learning/paint original; do
  [ -z "$(ls -A "$DEST/$d" 2>/dev/null)" ] && touch "$DEST/$d/.gitkeep"
done
# original/ has images now, so a leftover .gitkeep there is just noise
[ -n "$(ls -A "$DEST"/original/*.tif 2>/dev/null)" ] && rm -f "$DEST"/original/.gitkeep

cd "$DEST"
[ -d .git ] || git init -q
git add -A
# Only commit if something actually changed, so a no-op rebuild does not create
# an empty commit.
if ! git diff --cached --quiet 2>/dev/null || [ -z "$(git rev-parse -q --verify HEAD 2>/dev/null)" ]; then
  git -c user.email=you@example.com -c user.name="packager" \
      commit -qm "Rebuild package from source project" || true
  echo "    committed changes"
else
  echo "    no changes to commit"
fi
if git remote get-url origin >/dev/null 2>&1; then
  echo "    remote: $(git remote get-url origin)  (push with: cd $DEST && git push)"
fi

SIZE=$(du -sh "$DEST" | awk '{print $1}')
echo
echo "==> package ready: $SIZE at $DEST"
echo "    tracked files: $(git ls-files | wc -l | tr -d ' ')"
echo
echo "    a new user runs:"
echo "      git clone <url> && cd sem-crack-detector && ./run"
