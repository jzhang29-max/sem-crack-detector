#!/usr/bin/env bash
# Assemble a small, cloneable repo for distribution.
#
#   ./make_package.sh [dest]        default: ~/Desktop/sem-crack-detector
#
# Why a separate repo rather than slimming this one: this project's git history
# is ~4.7 GB because paint templates and result images were committed early on.
# .gitignore cannot help -- it does not apply to already-tracked files -- and the
# only way to shrink the history is to rewrite it, which is destructive and would
# break anyone who already has a clone. So this copies out just the parts a new
# user needs, into a fresh repo of a few tens of MB, and leaves this folder as
# the lab's full data archive.
#
# Included: all code, the trained models, the human correction masks and label
# log (the one irreplaceable thing here), docs.
# Excluded: source images, paint templates, results, figures, caches -- every one
# of which is regenerable from the images plus the labels.
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
DEST="${1:-$HOME/Desktop/sem-crack-detector}"

echo "==> packaging from $SRC"
echo "    into           $DEST"

# Preserve an existing .git across rebuilds. This used to `rm -rf "$DEST"`
# outright, which silently destroyed the repo's history AND its configured
# remote -- so the next `git push` failed with no remote and the published copy
# silently stopped tracking rebuilds. Stash .git, wipe the working tree, restore.
STASHED_GIT=""
if [ -d "$DEST/.git" ]; then
  STASHED_GIT="$(mktemp -d)/git"
  mv "$DEST/.git" "$STASHED_GIT"
  echo "    (preserving existing git history and remote)"
fi
rm -rf "$DEST"
mkdir -p "$DEST"
if [ -n "$STASHED_GIT" ]; then
  mv "$STASHED_GIT" "$DEST/.git"
fi
mkdir -p "$DEST"/{code,interior_active_learning/code,interior_active_learning/paint,interior_active_learning/labels,models,original,training_data,figures,docs}

# --- code ---
cp "$SRC"/code/*.py                              "$DEST"/code/
cp "$SRC"/interior_active_learning/code/*.py      "$DEST"/interior_active_learning/code/

# --- models ---
# TWO directories, and missing the second one silently degrades the app:
#   models/                          Pass 1 classifier (crack_classifier.joblib)
#   interior_active_learning/models/ Pass 2 model (unified_model.joblib)
# unified_pipeline._load_unified_bundle() returns None when that file is absent
# and Pass 2 is then skipped WITHOUT WARNING -- which quietly costs 27% of all
# crack regions and drops measured f1 from 0.776 to 0.768. An earlier version of
# this script copied only the first directory; a fresh clone therefore ran a
# weaker detector than the one benchmarked, with nothing on screen to say so.
mkdir -p "$DEST"/interior_active_learning/models
cp "$SRC"/models/*.joblib                         "$DEST"/models/ 2>/dev/null || true
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

# --- docs and entry points ---
cp "$SRC"/requirements.txt "$SRC"/run "$SRC"/run_app.sh "$SRC"/Makefile "$SRC"/.gitignore "$DEST"/
cp "$SRC"/MODEL_VALIDATION_BENCHMARK.md "$SRC"/PIPELINE_DEEP_DIVE.md "$SRC"/APP_COMPARISON.md "$DEST"/docs/ 2>/dev/null || true
# The package README is a real source file, not something written into the
# destination by hand -- a previous rebuild deleted the hand-written one,
# because this script wipes the working tree.
cp "$SRC"/PACKAGE_README.md "$DEST"/README.md
cp "$SRC"/PACKAGE_README.md "$DEST"/PACKAGE_README.md   # the source, so it survives a rebuild

# --- analysis provenance ---
# The archive/ scripts are the evidence behind the numbers in
# docs/MODEL_VALIDATION_BENCHMARK.md: the label-bias experiment that found 94% of
# negatives came from one image, the CV diagnosis, the SAM characterisation, the
# superseded models kept as counterexamples. They are small (~200 KB) and are the
# only record of HOW the reported results were arrived at, so they ship here now
# that this is the single repo rather than living only in the research repo.
mkdir -p "$DEST"/archive
cp -R "$SRC"/archive/. "$DEST"/archive/ 2>/dev/null || true
# The benchmark/experiment scripts that PRODUCED the figures and tables in
# docs/MODEL_VALIDATION_BENCHMARK.md -- ROC curves, confusion matrices, learning
# curves, model comparisons, and the rejected alternatives (per-type models,
# synthetic negatives, margin maximisation). Without these the documented numbers
# have no reproducible source.
mkdir -p "$DEST"/interior_active_learning/code/experiments
cp "$SRC"/interior_active_learning/code/experiments/*.py \
   "$DEST"/interior_active_learning/code/experiments/ 2>/dev/null || true
# and the packager itself, so the single repo can rebuild its own distribution
cp "$SRC"/make_package.sh "$DEST"/ 2>/dev/null || true

# the diagram sources, but not the rendered PNGs (29 MB, regenerable)
mkdir -p "$DEST"/docs/diagram
cp "$SRC"/pipeline_diagram/*.svg "$DEST"/docs/diagram/ 2>/dev/null || true
chmod +x "$DEST"/run "$DEST"/run_app.sh

# keep empty dirs in git so a fresh clone has somewhere to put things
for d in original figures training_data interior_active_learning/paint; do
  [ -z "$(ls -A "$DEST/$d" 2>/dev/null)" ] && touch "$DEST/$d/.gitkeep"
done

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
echo "    to publish:"
echo "      cd $DEST"
echo "      gh repo create sem-crack-detector --private --source=. --push"
echo
echo "    a new user then runs:"
echo "      git clone <url> && cd sem-crack-detector && ./run"
