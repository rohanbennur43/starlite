#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, json
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import pyarrow as pa
import pyarrow.parquet as pq
from shapely import from_wkb, from_wkt
import folium

def _is_bin(t: pa.DataType) -> bool: return pa.types.is_binary(t) or pa.types.is_large_binary(t)
def _is_str(t: pa.DataType) -> bool: return pa.types.is_string(t) or pa.types.is_large_string(t)
def _np(col: pa.ChunkedArray): return col.combine_chunks().to_numpy(zero_copy_only=False)

def _try_decode(arr, enc: str):
    try:
        geoms = from_wkt(arr) if enc == "WKT" else from_wkb(arr)
    except Exception as e:
        return [], 0, f"vectorized parse failed: {e}"
    good = 0
    out = []
    for g in geoms:
        if g is None or getattr(g, "is_empty", True):
            out.append(None)
        else:
            out.append(g); good += 1
    return out, good, None

def detect_geom_by_scan(tbl: pa.Table, debug=False):
    best = None  # (good, name, enc, geoms, note)
    for name in tbl.column_names:
        col = tbl[name]
        t = col.type
        encs = (["WKB"] if _is_bin(t) else []) + (["WKT"] if _is_str(t) else [])
        if not encs: 
            if debug: print(f"  - skip {name}: type {t}")
            continue
        arr = _np(col)
        for enc in encs:
            geoms, good, note = _try_decode(arr, enc)
            if debug:
                total = len(arr)
                print(f"  - try {name} as {enc}: parsed={good}/{total}" + (f" ({note})" if note else ""))
            if good > 0 and (best is None or good > best[0]):
                best = (good, name, enc, geoms, note)
    if best is None: 
        return None
    good, name, enc, geoms, _ = best
    meta = {
        "version": "1.0.0",
        "primary_column": name,
        "columns": {name: {"encoding": enc, "geometry_types": None, "crs": None, "edges": None}},
    }
    return name, enc, geoms, good, meta

def compute_bbox(geoms) -> Optional[Tuple[float, float, float, float]]:
    anyg = False
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    for g in geoms:
        if g is None: continue
        anyg = True
        x0,y0,x1,y1 = g.bounds
        if x0 < minx: minx = x0
        if y0 < miny: miny = y0
        if x1 > maxx: maxx = x1
        if y1 > maxy: maxy = y1
    return (minx, miny, maxx, maxy) if anyg else None

def gather(paths: List[Path]) -> List[Path]:
    out=[]
    for p in paths:
        if p.is_file() and p.suffix.lower()==".parquet": out.append(p)
        elif p.is_dir(): out.extend(sorted(p.rglob("*.parquet")))
    seen=set(); uniq=[]
    for f in out:
        if f not in seen: uniq.append(f); seen.add(f)
    return uniq

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_parquet_bboxes_scan.py <file_or_dir> [..] [--out OUT.html] [--debug]")
        sys.exit(1)
    args = sys.argv[1:]
    debug = False
    if "--debug" in args:
        debug = True
        args.remove("--debug")
    out_html = "parquet_bboxes.html"
    if "--out" in args:
        i = args.index("--out")
        if i == len(args)-1: sys.exit("ERROR: --out requires a filename")
        out_html = args[i+1]; del args[i:i+2]

    files = gather([Path(a) for a in args])
    if not files: sys.exit("No .parquet files found.")

    boxes = []
    gmin=[float("inf"), float("inf")]
    gmax=[float("-inf"), float("-inf")]

    for p in files:
        try:
            tbl = pq.read_table(p).combine_chunks()
            if debug:
                print(f"\n=== {p} ===")
                print(f"rows={tbl.num_rows}, cols={tbl.num_columns}")
                for n,c in zip(tbl.column_names, tbl.columns):
                    print(f"  schema: {n}: {c.type}")

            det = detect_geom_by_scan(tbl, debug=debug)
            if det is None:
                print(f"[{p.name}] NO GEOMETRY DETECTED (all parses zero).")
                continue

            geom_col, enc, geoms, good, meta = det
            total = tbl.num_rows
            if debug:
                print(f"-> detected geometry: column='{geom_col}', encoding={enc}, parsed={good}/{total}")

            bbox = compute_bbox(geoms)
            if bbox is None:
                # Detailed diagnostics: count None vs non-None and show first few items
                n_none = sum(1 for g in geoms if g is None)
                n_ok = good
                print(f"[{p.name}] BBOX: NONE (valid={n_ok}, empty/None={n_none}, total={len(geoms)})")
                # show a couple of raw cell samples to help
                col = tbl[geom_col].combine_chunks()
                arr = _np(col)
                for i in range(min(3, len(arr))):
                    sample = arr[i]
                    if isinstance(sample, (bytes, bytearray)):
                        print(f"  sample[{i}] = WKB bytes len={len(sample)}")
                    else:
                        txt = str(sample)
                        print(f"  sample[{i}] = {txt[:80]}{'...' if len(txt)>80 else ''}")
                continue

            xmin,ymin,xmax,ymax = bbox
            print(f"[{p.name}] BBOX = ({xmin}, {ymin}) → ({xmax}, {ymax})")

            gmin[0]=min(gmin[0], xmin); gmin[1]=min(gmin[1], ymin)
            gmax[0]=max(gmax[0], xmax); gmax[1]=max(gmax[1], ymax)
            boxes.append((p, bbox, meta, total))
        except Exception as e:
            print(f"[{p.name}] ERROR: {e}")

    if not boxes:
        sys.exit("No boxes to display (see logs above).")

    m = folium.Map(tiles="CartoDB positron")
    m.fit_bounds([[gmin[1], gmin[0]], [gmax[1], gmax[0]]])

    for p, (xmin, ymin, xmax, ymax), meta, total in boxes:
        popup = (
            f"{p}\nrows={total}\n"
            f"bbox=({xmin:.6f}, {ymin:.6f}) → ({xmax:.6f}, {ymax:.6f})\n"
            f"primary_column={meta['primary_column']}  "
            f"encoding={meta['columns'][meta['primary_column']]['encoding']}"
        )
        folium.Rectangle(
            bounds=[[ymin, xmin], [ymax, xmax]],
            fill=False,
            weight=3,
            tooltip=p.name,
            popup=popup,
        ).add_to(m)

    m.save(out_html)
    print(f"\nWrote {out_html} with {len(boxes)} rectangles.")

if __name__ == "__main__":
    main()