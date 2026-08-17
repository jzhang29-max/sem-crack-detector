"""
Read back a round's labeling template (after the user has filled in
UserVerdict = TRUE / FALSE / SKIP) and:
  1) save it as the permanent round_<N>_filled.csv record (so
     active_learning_select.py never shows the same candidate twice),
  2) write the verdict back into each interior candidate's own
     candidates/<image>_interior.csv (IsCrack column), so
     train_interior_model.py can just read straight from there.

Original-region verdicts are recorded in the round's filled.csv (so they
won't be re-selected) but are NOT written back into the main project's
results/*.csv or manual_corrections_ledger.csv -- this experiment stays
read-only with respect to production, per its whole point.
"""
import os
import sys
import glob
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CANDIDATES_DIR, LABELS_DIR

VALID = {"TRUE": True, "FALSE": False, "SKIP": None}


def ingest(round_num):
    template_path = os.path.join(LABELS_DIR, f"round_{round_num}_template.csv")
    if not os.path.exists(template_path):
        raise FileNotFoundError(template_path)
    df = pd.read_csv(template_path)

    verdicts = df["UserVerdict"].astype(str).str.strip().str.upper()
    unknown = ~verdicts.isin(list(VALID.keys()) + ["NAN", ""])
    if unknown.any():
        print("WARNING: unrecognized UserVerdict values (leaving these unlabeled):")
        print(df.loc[unknown, ["Label", "SourceImage", "UserVerdict"]].to_string(index=False))

    blank = verdicts.isin(["NAN", ""])
    if blank.any():
        print(f"NOTE: {blank.sum()} rows have no UserVerdict yet -- they'll be offered again next round.")

    df = df[~blank & ~unknown].copy()
    df["UserVerdictBool"] = verdicts[~blank & ~unknown].map(VALID)

    filled_path = os.path.join(LABELS_DIR, f"round_{round_num}_filled.csv")
    df.to_csv(filled_path, index=False)
    print(f"Saved {filled_path} ({len(df)} labeled rows: "
          f"{(df['UserVerdictBool']==True).sum()} True, "
          f"{(df['UserVerdictBool']==False).sum()} False, "
          f"{df['UserVerdictBool'].isna().sum()} Skip)")

    interior = df[(df["CandidateType"] != "original") & df["UserVerdictBool"].notna()]
    n_written = 0
    for name, g in interior.groupby("SourceImage"):
        csv_path = os.path.join(CANDIDATES_DIR, f"{name}_interior.csv")
        if not os.path.exists(csv_path):
            print(f"WARNING: no candidates CSV for {name}, skipping {len(g)} verdicts")
            continue
        cand = pd.read_csv(csv_path)
        # An all-empty-string IsCrack column round-trips through CSV as an
        # all-NaN float64 column, and pandas refuses to assign a bool into
        # a float column in place (LossySetitemError) -- cast to object
        # first so True/False can actually be written.
        cand["IsCrack"] = cand["IsCrack"].astype(object)
        label_to_verdict = dict(zip(g["Label"].astype(int), g["UserVerdictBool"]))
        mask = cand["Label"].astype(int).isin(label_to_verdict)
        cand.loc[mask, "IsCrack"] = cand.loc[mask, "Label"].astype(int).map(label_to_verdict)
        cand.to_csv(csv_path, index=False)
        n_written += int(mask.sum())
    print(f"Wrote {n_written} interior-candidate verdicts back into candidates/*_interior.csv")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("round", type=int)
    args = ap.parse_args()
    ingest(args.round)
