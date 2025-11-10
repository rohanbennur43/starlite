#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import csv
import math
from pathlib import Path
import folium

WORLD = {
    "minx": -180.0,
    "maxx": 180.0,
    "miny": -90.0,
    "maxy": 90.0,
}

def _to_float(s):
    try:
        v = float(s)
        return v
    except Exception:
        return None

def _clamp_or_fill(x, lo, hi):
    if x is None:
        return None
    if math.isnan(x):
        return None
    if math.isinf(x):
        return lo if x < 0 else hi
    return max(lo, min(hi, x))

def read_boxes(csv_path):
    boxes = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        required = {"pid", "minx", "miny", "maxx", "maxy"}
        if not required.issubset(r.fieldnames or []):
            raise ValueError(f"CSV must have columns: {sorted(required)}")

        for row in r:
            pid = row["pid"]
            xmin = _to_float(row["minx"])
            ymin = _to_float(row["miny"])
            xmax = _to_float(row["maxx"])
            ymax = _to_float(row["maxy"])

            xmin = _clamp_or_fill(xmin, WORLD["minx"], WORLD["maxx"])
            xmax = _clamp_or_fill(xmax, WORLD["minx"], WORLD["maxx"])
            ymin = _clamp_or_fill(ymin, WORLD["miny"], WORLD["maxy"])
            ymax = _clamp_or_fill(ymax, WORLD["miny"], WORLD["maxy"])

            if None in (xmin, ymin, xmax, ymax):
                continue
            if xmin > xmax or ymin > ymax:
                continue

            boxes.append((pid, xmin, ymin, xmax, ymax))
    return boxes

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_boxes.py <input_csv>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        sys.exit(f"File not found: {csv_path}")

    boxes = read_boxes(csv_path)
    if not boxes:
        sys.exit("No valid boxes to display.")

    gminx = min(b[1] for b in boxes)
    gminy = min(b[2] for b in boxes)
    gmaxx = max(b[3] for b in boxes)
    gmaxy = max(b[4] for b in boxes)

    m = folium.Map(tiles="CartoDB positron")
    m.fit_bounds([[gminy, gminx], [gmaxy, gmaxx]])

    for pid, xmin, ymin, xmax, ymax in boxes:
        folium.Rectangle(
            bounds=[[ymin, xmin], [ymax, xmax]],
            fill=False,
            weight=2,
            tooltip=f"pid={pid}",
            popup=f"pid={pid}\n({xmin:.6f}, {ymin:.6f}) → ({xmax:.6f}, {ymax:.6f})",
        ).add_to(m)

    out_html = "bounding_bboxes_from_csv.html"
    m.save(out_html)
    print(f"Wrote {out_html} with {len(boxes)} rectangles.")

if __name__ == "__main__":
    main()