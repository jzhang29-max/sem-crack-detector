import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import list_original_images
from interior_candidates import build_interior_candidates_for_image

for name in list_original_images():
    try:
        df, _ = build_interior_candidates_for_image(name)
        print(f"DONE {name}: {len(df)} candidates "
              f"({(df['CandidateType']=='concavity').sum()} concavity, "
              f"{(df['CandidateType']=='bridge_corridor').sum()} bridge_corridor, "
              f"{(df['CandidateType']=='interior_fill').sum()} interior_fill)", flush=True)
    except Exception as e:
        print(f"ERROR {name}: {e}", flush=True)
print("ALL_DONE", flush=True)
