"""
Shared loader for the UNIFIED model's pooled training dataset: every
labeled interior candidate (load_labeled_interior()) plus every original-
candidate correction from manual_corrections_ledger.csv, re-expressed in
the same 11-feature schema (original_ledger_unified_features.csv, built by
build_original_ledger_unified_features.py) -- together, every human-
verified label this project has, in one consistent feature space,
regardless of whether the candidate came from Step E (original darkness-
threshold region) or Step H (interior/concavity/bridge candidate).

Used by train_unified_model.py (the real retrain entry point) and every
benchmark_*_unified.py script, so training and evaluation always see
exactly the same pool.
"""
import os
import sys
from pathlib import Path

import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from train_interior_model import load_labeled_interior
from active_learning_select import INTERIOR_FEATURE_COLUMNS

ORIG_LEDGER_CSV = CODE_DIR.parent / "candidates" / "original_ledger_unified_features.csv"

# Both sources use these same column names; 'Label' is included because
# pd.concat requires matching columns across frames and interior_df carries
# it too (KeyError otherwise) -- see models/checkpoints/README.md note on
# this exact pooling step. 'CandidateType' is included so
# calibrate_interior_fill_rule() (which filters on CandidateType ==
# "interior_fill") still works correctly on the pooled frame -- the
# original-ledger side has no natural type of its own, so it's tagged
# "original" (never matches "interior_fill", which is exactly the point).
COMMON_COLS = INTERIOR_FEATURE_COLUMNS + ["IsCrack", "SourceImage", "Label", "CandidateType"]


def load_unified_pooled():
    interior_df = load_labeled_interior()
    orig_df = pd.read_csv(ORIG_LEDGER_CSV)
    orig_df["IsCrack"] = orig_df["IsCrack"].astype(bool)
    orig_df["CandidateType"] = "original"
    interior_df["IsCrack"] = interior_df["IsCrack"].astype(bool)
    pooled = pd.concat([interior_df[COMMON_COLS], orig_df[COMMON_COLS]], ignore_index=True)
    return pooled


if __name__ == "__main__":
    df = load_unified_pooled()
    n_pos, n_neg = int(df["IsCrack"].sum()), int((~df["IsCrack"]).sum())
    print(f"Pooled unified dataset: {len(df)} examples from {df['SourceImage'].nunique()} source images "
          f"({n_pos} True / {n_neg} False)")
