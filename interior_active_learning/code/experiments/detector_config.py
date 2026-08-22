"""Pin the detector configuration for an experiment, and stamp it into the result.

WHY THIS EXISTS
Commit 4d97602 made SAM 2 refinement the DEFAULT inside run_unified_pipeline. Every
experiment here calls run_unified_pipeline and none of them said which detector they wanted,
so overnight they all started measuring a different detector than the one their output is
labelled with:

  * naive_baselines.py's arm called "pipeline" -- and published in the README as the
    "deployed two-pass pipeline" -- would now be the REFINED detector
  * sam2_hybrid.py's "pipeline" baseline would be refined, and its "sam2_refine" arm would be
    refined TWICE, which makes the comparison it exists for meaningless
  * scoring_convention_bias, sparsity_sensitivity, oos_convention_gap and
    erased_state_sensitivity would all silently switch detector

None of their result JSONs recorded a detector, so nothing on disk would have revealed it.
That is the failure this project documents, committed against its own experiments, and the
fix is not to remember: it is to make an experiment state its detector or not run.

USE
    with pinned("off"):          # measure the bare two-pass detector
        ...
    with pinned("refine"):       # measure the shipped default
        ...
    payload["detector"] = stamp("off")
"""
import contextlib
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

import unified_pipeline as up

VALID = ("off", "refine", "hybrid")


@contextlib.contextmanager
def pinned(mode):
    """Force run_unified_pipeline's detector for the duration. Restores on exit.

    Explicit and required: there is no default, because a default is how this went wrong.
    """
    if mode not in VALID:
        raise ValueError(f"detector mode must be one of {VALID}, got {mode!r}")
    prev = up.SAM2_MODE
    up.SAM2_MODE = mode
    try:
        yield mode
    finally:
        up.SAM2_MODE = prev


def stamp(mode):
    """A record of which detector produced a result, for embedding in the result JSON.

    Includes the commit, so a figure can be tied to a tree rather than to a moving `main`.
    """
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=_HERE, capture_output=True, text=True,
                             timeout=10).stdout.strip() or None
    except Exception:
        sha = None
    out = {"detector_mode": mode, "git_commit": sha,
           "pipeline_default_at_runtime": up.SAM2_MODE,
           "note": ("detector_mode is what this experiment PINNED, not what the pipeline "
                    "defaults to. 'off' is the bare two-pass detector; 'refine' is the "
                    "shipped default since commit 4d97602, which applies SAM 2 to each "
                    "accepted candidate's boundary.")}
    if mode != "off":
        try:
            import sam2_refine as _sr
            out["sam2_model"] = _sr.DEFAULT_MODEL
        except Exception:
            pass
    return out


def out_for(base, detector):
    """One result file per detector configuration.

    The bare detector keeps the historical filename so every existing reference stays valid;
    any other configuration gets its own. Sharing one file would mean a re-measurement
    silently replacing the number it is supposed to be compared against -- which is the
    accident this whole audit was about.
    """
    return base if detector == "off" else base.replace(".json", f".sam2_{detector}.json")
