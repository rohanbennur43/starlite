#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
from pathlib import Path
from typing import Iterable, Tuple, List, Optional

import pyarrow.parquet as pq
import pyarrow as pa
from shapely import from_wkb, from_wkt
import folium


def detect_geom(schema: pa.Schema, default: str = "geometry") -> Tuple[str, str]:
    """Return (geometry_column_name, encoding[WKB|WKT])."""
    md = schema.metadata or {}
    geo = md.get(b"geo")
    if not geo:
        return default, "WKB"
    j = json.loads(geo.decode())
    col = j.get("primary_column") or default
    enc = j.get("columns", {}).get(col, {}).get("encoding", "WKB")
    return col, enc.upper()


def file_bounds(geoms_iter: Iterable) -> Optional[Tuple[float, float, float, float]]:
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    any_geom = False
    for g in geoms_iter:
        if g is None or g.is_empty:
            continue
        any_geom = True
        x0, y0, x1, y1 = g.bounds
        if x0 < minx: minx = x0
        if y0 < miny: miny = y0
        if x1 > maxx: maxx = x1
        if y1 > maxy: maxy = y1
    if not any_geom:
        return None
    return (minx, miny, maxx, maxy)


def iter_file_geoms(pf: pq.ParquetFile, geom_col: str, encoding: str):
    """Yield shapely geometries from a ParquetFile by row group."""
    for rg in range(pf.num_row_groups):
        col_tbl = pf.read_row_group(rg, columns=[geom_col]).combine_chunks()
        arr = col_tbl[geom_col].to_numpy(zero_copy_only=False)
        geoms = from_wkt(arr) if encoding == "WKT" else from_wkb(arr)
        for g in geoms:
            yield g


def gather_parquet_files(paths: List[Path]) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        if p.is_file() and p.suffix.lower() == ".parquet":
            out.append(p)
        elif p.is_dir():
            out.extend(sorted(p.rglob("*.parquet")))
    # de-dup while preserving order
    seen = set()
    unique = []
    for f in out:
        if f not in seen:
            unique.append(f)
            seen.add(f)
    return unique


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_parquet_bboxes.py <file_or_dir> [more files/dirs] [--out OUT.html]")
        sys.exit(1)

    # parse args
    args = sys.argv[1:]
    out_html = "parquet_bboxes.html"
    if "--out" in args:
        i = args.index("--out")
        try:
            out_html = args[i + 1]
        except IndexError:
            sys.exit("ERROR: --out requires a filename")
        del args[i:i+2]

    inputs = [Path(a) for a in args]
    files = gather_parquet_files(inputs)
    if not files:
        sys.exit("No .parquet files found in given paths.")

    boxes = []  # (path, (xmin,ymin,xmax,ymax), nrows)
    gmin = [float("inf"), float("inf")]
    gmax = [float("-inf"), float("-inf")]

    for p in files:
        try:
            pf = pq.ParquetFile(p)
            schema = pf.schema_arrow
            geom_col, enc = detect_geom(schema)
            if geom_col not in schema.names:
                print(f"Skip {p}: geometry column '{geom_col}' not found")
                continue

            b = file_bounds(iter_file_geoms(pf, geom_col, enc))
            if b is None:
                print(f"Skip {p}: no valid geometries")
                continue

            xmin, ymin, xmax, ymax = b
            gmin[0] = min(gmin[0], xmin)
            gmin[1] = min(gmin[1], ymin)
            gmax[0] = max(gmax[0], xmax)
            gmax[1] = max(gmax[1], ymax)
            nrows = pf.metadata.num_rows if pf.metadata is not None else None
            boxes.append((p, b, nrows))
        except Exception as e:
            print(f"Error reading {p}: {e}")

    if not boxes:
        sys.exit("No boxes to display.")

    # Build folium map
    m = folium.Map(tiles="CartoDB positron")
    m.fit_bounds([[gmin[1], gmin[0]], [gmax[1], gmax[0]]])

    for p, (xmin, ymin, xmax, ymax), nrows in boxes:
        folium.Rectangle(
            bounds=[[ymin, xmin], [ymax, xmax]],
            fill=False,
            weight=2,
            tooltip=p.name,
            popup=f"{p}\nrows={nrows}",
        ).add_to(m)

    m.save(out_html)
    print(f"Wrote {out_html} with {len(boxes)} rectangles.")


if __name__ == "__main__":
    main()