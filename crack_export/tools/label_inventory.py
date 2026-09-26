import os, sys, glob, json
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
sys.path.insert(0,'interior_active_learning/code')
from common import PAINT_DIR
rows=[]
paths=sorted(glob.glob(os.path.join(PAINT_DIR,"*_correction_mask.png")))
paths=[p for p in paths if not os.path.basename(p).startswith(("apptest","SELFTEST","MASKGUARD"))]
for i,p in enumerate(paths,1):
    n=os.path.basename(p).replace("_correction_mask.png","")
    a=np.array(Image.open(p))
    if a.ndim>2: a=a[...,0]
    tot=a.size
    c=np.bincount(a.ravel(), minlength=4)[:4]
    rows.append(dict(frame=n, px=int(tot),
                     unreviewed=int(c[0]), crack=int(c[1]),
                     not_crack=int(c[2]), erased=int(c[3])))
    if i%25==0: print(f"  {i}/{len(paths)}", flush=True)
os.makedirs(os.environ.get("SEMCRACK_DERIVED",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                 "crack_export", "derived")), exist_ok=True)
json.dump(rows, open(os.path.join(os.environ.get("SEMCRACK_DERIVED",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                 "crack_export", "derived")), "label_inventory.json"), "w"))
T=sum(r["px"] for r in rows)
print(f"\n  frames with a correction mask: {len(rows)}")
print(f"  total pixels: {T/1e6:.0f} MP")
for k in ("unreviewed","crack","not_crack","erased"):
    s=sum(r[k] for r in rows)
    nf=sum(1 for r in rows if r[k]>0)
    print(f"    {k:<12} {s/1e6:9.1f} MP  {100*s/T:6.2f}%   present on {nf}/{len(rows)} frames")
print()
print("  frames where painted CRACK exceeds 20% of the frame:")
big=sorted((r for r in rows if r["crack"]/r["px"]>0.20), key=lambda r:-r["crack"]/r["px"])
for r in big[:12]:
    print(f"    {100*r['crack']/r['px']:5.1f}%  {r['frame']}")
print(f"  ... {len(big)} such frames")
