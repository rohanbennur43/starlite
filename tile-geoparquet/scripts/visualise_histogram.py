# Re-run the visualization pipeline for histogram .npy files.
import os
import glob
import numpy as np
from PIL import Image, ImageOps
from datetime import datetime
from pathlib import Path
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Visualize histogram .npy files.")
parser.add_argument("hist_dir", help="Directory containing histogram .npy files")
args = parser.parse_args()

HIST_DIR = args.hist_dir  # use directory from command line argument
OUT_DIR = Path(HIST_DIR)
IMG_DIR = OUT_DIR / "_previews"
HTML_PATH = OUT_DIR / "index.html"

IMG_DIR.mkdir(parents=True, exist_ok=True)

npy_files = sorted(glob.glob(str(OUT_DIR / "*.npy")))
records = []

def to_png(npy_path: str) -> str:
    arr = np.load(npy_path)
    arr_log = np.log1p(arr.astype(np.float64))
    mn, mx = float(arr_log.min()), float(arr_log.max())
    if mx <= mn:
        scaled = np.zeros_like(arr_log, dtype=np.uint8)
    else:
        scaled = ((arr_log - mn) / (mx - mn) * 255.0).astype(np.uint8)
    img = Image.fromarray(scaled, mode="L")
    img = ImageOps.autocontrast(img)
    base = Path(npy_path).stem
    png_path = IMG_DIR / f"{base}.png"
    img.save(png_path)
    return str(png_path)

for f in npy_files:
    try:
        png = to_png(f)
        st = os.stat(f)
        with np.load(f, mmap_mode="r") as _arr:
            shape = str(_arr.shape)
            dtype = str(_arr.dtype)
        records.append({
            "file": os.path.basename(f),
            "preview_png": os.path.relpath(png, OUT_DIR),
            "shape": shape,
            "dtype": dtype,
            "size_bytes": st.st_size,
            "modified": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
        })
    except Exception as e:
        records.append({
            "file": os.path.basename(f),
            "preview_png": "",
            "shape": "ERR",
            "dtype": "ERR",
            "size_bytes": 0,
            "modified": "",
        })

df = pd.DataFrame(records)

if not df.empty:
    print(df.head())
# Write a simple HTML gallery
rows = []
rows.append("<!doctype html><meta charset='utf-8'><title>Histogram previews</title>")
rows.append("<style>body{font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif;padding:24px;}\
.grid{display:grid;grid-template-columns:repeat(auto-fill, minmax(320px,1fr));gap:16px;}\
.card{border:1px solid #ddd;border-radius:8px;padding:12px;}\
.card img{width:100%;height:auto;image-rendering:pixelated;}\
.meta{font-size:12px;color:#444;margin-top:8px;word-break:break-all;}\
a{color:#0645AD;text-decoration:none}</style>")
rows.append("<h1>Histogram previews</h1>")
rows.append("<div class='grid'>")
for _, r in df.iterrows():
    if not r["preview_png"]:
        continue
    file_name = r["file"]
    img_rel = r["preview_png"]
    meta = f"shape={r['shape']} dtype={r['dtype']} size={r['size_bytes']} bytes<br>modified={r['modified']}"
    rows.append(f"<div class='card'><a href='{img_rel}' target='_blank'><img src='{img_rel}' alt='{file_name}'></a>\
<div class='meta'><strong>{file_name}</strong><br>{meta}</div></div>")
rows.append("</div>")

with open(HTML_PATH, "w", encoding="utf-8") as fh:
    fh.write("\n".join(rows))

print(f"Created {len(df)} previews in {IMG_DIR}")
print(f"Open this page to browse: {HTML_PATH}")