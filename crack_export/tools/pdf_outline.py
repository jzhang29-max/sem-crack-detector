"""Add the specimen/frame outline to the combined PDF. Run with an interpreter that has pypdf."""
import json, os, sys
from pypdf import PdfReader, PdfWriter
pdf = sys.argv[1]; marks = json.load(open(sys.argv[2]))
r, w = PdfReader(pdf), PdfWriter()
for p in r.pages: w.add_page(p)
for m in marks:
    parent = w.add_outline_item(m["title"], m["page"])
    for k in m["kids"]:
        w.add_outline_item(k["title"], k["page"], parent=parent)
tmp = pdf + ".tmp"
with open(tmp, "wb") as fh: w.write(fh)
os.replace(tmp, pdf)
print(f"  outline added: {len(marks)} specimens, {sum(len(m['kids']) for m in marks)} frames")
