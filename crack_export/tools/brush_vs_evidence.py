"""How much of the hand-painted crack is supported by the image, and how much is brush?

The test has to be label-free and segmenter-free or it is circular. So: inside the detected
field of view, take the median and MAD of the 8-bit display image -- robust statistics
dominated by matrix, because matrix is most of every frame -- and call a pixel DARK if it
sits below median - 3*MAD. That threshold never looks at the label or at the model.

Then ask, of the pixels a human painted as crack, what share are dark. A tight label tracks
the crack and scores high. A broad brush sweeps matrix in with the crack and scores low.
"""
import os, sys, glob, json
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
sys.path.insert(0,'interior_active_learning/code'); sys.path.insert(0,'code')
from common import PAINT_DIR, contrast_kwargs_for
from detect_cracks import load_as_uint8, find_field_of_view

out=[]
paths=sorted(glob.glob(os.path.join(PAINT_DIR,"*_correction_mask.png")))
paths=[p for p in paths if not os.path.basename(p).startswith(("apptest","SELFTEST","MASKGUARD"))]
for i,p in enumerate(paths,1):
    n=os.path.basename(p).replace("_correction_mask.png","")
    src=f"original/{n}.tif"
    if not os.path.exists(src): continue
    cm=np.array(Image.open(p))
    if cm.ndim>2: cm=cm[...,0]
    painted = cm==1
    if painted.sum()<5000: continue
    img8=load_as_uint8(src, **contrast_kwargs_for(n))
    if img8.shape!=cm.shape:
        x0,y0,x1,y1=find_field_of_view(img8)
        img8=img8[y0:y1,x0:x1]
        if img8.shape!=cm.shape: continue
    med=float(np.median(img8)); mad=float(np.median(np.abs(img8-med)))
    if mad<=0: continue
    thr = med - 3.0*mad
    dark = img8 < thr
    inside = painted
    frac_dark_inside = float(dark[inside].mean())
    frac_dark_frame  = float(dark.mean())
    # of everything the image calls dark, how much did the human paint?
    recall_of_dark = float(painted[dark].mean()) if dark.sum() else float("nan")
    out.append(dict(frame=n, painted_pct=100*float(inside.mean()),
                    dark_thr=thr, med=med, mad=mad,
                    pct_of_label_with_evidence=100*frac_dark_inside,
                    pct_frame_dark=100*frac_dark_frame,
                    pct_of_dark_that_is_labelled=100*recall_of_dark))
    if i%10==0: print(f"  {i}/{len(paths)}", flush=True)

os.makedirs(os.environ.get("SEMCRACK_DERIVED",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                 "crack_export", "derived")), exist_ok=True)
json.dump(out, open(os.path.join(os.environ.get("SEMCRACK_DERIVED",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                 "crack_export", "derived")), "brush_vs_evidence.json"), "w"))
ev=np.array([r["pct_of_label_with_evidence"] for r in out])
print(f"\n  frames measured: {len(out)}")
print(f"  share of hand-painted CRACK pixels that are actually dark:")
print(f"     median {np.median(ev):.1f}%   mean {ev.mean():.1f}%   range {ev.min():.1f}-{ev.max():.1f}%")
print(f"  -> the rest is brush: median {100-np.median(ev):.1f}% of the label has no darkness evidence")
print()
worst=sorted(out,key=lambda r:r["pct_of_label_with_evidence"])[:10]
print("  weakest labels (least evidence inside the paint):")
for r in worst:
    print(f"    {r['pct_of_label_with_evidence']:5.1f}% evidence   painted {r['painted_pct']:5.1f}% of frame   {r['frame']}")
