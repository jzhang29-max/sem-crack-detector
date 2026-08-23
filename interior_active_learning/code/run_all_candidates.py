"""Precompute Pass 2 (interior) candidates for every image in original/.

    ./.venv/bin/python3 interior_active_learning/code/run_all_candidates.py
    ./.venv/bin/python3 interior_active_learning/code/run_all_candidates.py --only NAME

Writes one candidates CSV per image into interior_active_learning/candidates/. Minutes per
frame on a large one, so it prints DONE lines as it goes and ALL_DONE at the end; a frame that
raises is reported as ERROR and does not stop the rest.

The work used to run at MODULE level, with no `if __name__` guard, so merely importing this
file -- which an import-checking sweep does -- started processing the whole corpus. It now
needs to be invoked deliberately.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import list_original_images
from interior_candidates import build_interior_candidates_for_image


def run(names=None):
    names = names or list_original_images()
    failed = 0
    for name in names:
        try:
            df, _ = build_interior_candidates_for_image(name)
            print(f"DONE {name}: {len(df)} candidates "
                  f"({(df['CandidateType']=='concavity').sum()} concavity, "
                  f"{(df['CandidateType']=='bridge_corridor').sum()} bridge_corridor, "
                  f"{(df['CandidateType']=='interior_fill').sum()} interior_fill)", flush=True)
        except Exception as e:
            failed += 1
            print(f"ERROR {name}: {e}", flush=True)
    print("ALL_DONE", flush=True)
    # A run where every frame failed used to exit 0 and read as success.
    return 1 if failed and failed == len(names) else 0


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__)
        print("usage: run_all_candidates.py [--only IMAGE_NAME]")
        sys.exit(0)
    only = None
    if argv:
        if argv[0] != "--only" or len(argv) < 2:
            print(f"unknown option {argv[0]!r}. Use --only IMAGE_NAME, or --help.")
            sys.exit(2)
        only = [argv[1]]
    sys.exit(run(only))
