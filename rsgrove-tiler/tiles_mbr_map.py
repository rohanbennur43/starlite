#!/usr/bin/env python3
import argparse, os, glob, gzip, json, warnings
from typing import List, Tuple, Optional

import folium
import geopandas as gpd
from shapely.geometry import box
import pyarrow.parquet as pq

# Optional: use Fiona for fast bounds/feature count on GeoJSONs
try:
    import fiona
    HAS_FIONA = True
except Exception:
    HAS_FIONA = False
    warnings.warn("Fiona not available; GeoJSON bounds will be computed via GeoPandas (slower).")

# -------------------------
# Helpers to list inputs
# -------------------------

GEOJSON_EXTS = (".geojson", ".json", ".geojsonl", ".jsonl", ".geojson.gz", ".json.gz", ".geojsonl.gz", ".jsonl.gz")
PARQUET_EXTS = (".parquet",)

def list_inputs(paths: List[str]) -> List[str]:
    files = []
    for p in paths:
        if os.path.isdir(p):
            # mixed directories allowed
            for ext in (*PARQUET_EXTS, *GEOJSON_EXTS):
                files += glob.glob(os.path.join(p, f"*{ext}"))
        else:
            # expand globs
            files += glob.glob(p)
    # De-dup + sort
    files = sorted(set(files))
    return files

# -------------------------
# Fast row/feature counts
# -------------------------

def row_count_parquet(path: str) -> int:
    try:
        return pq.read_metadata(path).num_rows
    except Exception:
        return -1

def feature_count_geojson(path: str) -> int:
    if not HAS_FIONA:
        return -1
    try:
        with fiona.open(path) as src:
            return len(src) if src is not None else -1
    except Exception:
        return -1

# -------------------------
# Bounds + CRS per file
# -------------------------

def bounds_crs_parquet(path: str) -> Tuple[Optional[Tuple[float,float,float,float]], str]:
    """
    Read minimal data needed from a GeoParquet to get total bounds.
    Falls back to geopandas.read_parquet and total_bounds.
    Returns (bounds, crs_str_or_None). Bounds in file CRS.
    """
    try:
        gdf = gpd.read_parquet(path)  # geometry-only slicing is not always safe; keep simple & robust
        if gdf.empty or gdf.geometry.isna().all():
            return None, str(gdf.crs) if gdf.crs else None
        b = tuple(map(float, gdf.total_bounds))  # (minx, miny, maxx, maxy)
        return b, (str(gdf.crs) if gdf.crs else None)
    except Exception:
        return None, None

def _open_text_maybe_gzip(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, "rt", encoding="utf-8")

def bounds_crs_geojson_via_fiona(path: str) -> Tuple[Optional[Tuple[float,float,float,float]], str]:
    try:
        with fiona.open(path) as src:
            # Fiona returns bounds in source CRS
            b = src.bounds  # (minx, miny, maxx, maxy)
            crs_str = None
            if src.crs_wkt:
                crs_str = src.crs_wkt
            elif src.crs:
                # dict like {'init': 'epsg:4326'} or modern PROJ dict
                crs_str = src.crs.get("init") or "EPSG:4326" if "epsg" in src.crs else None
            return (float(b[0]), float(b[1]), float(b[2]), float(b[3])), crs_str
    except Exception:
        return None, None

def bounds_crs_geojson_via_geopandas(path: str) -> Tuple[Optional[Tuple[float,float,float,float]], str]:
    try:
        gdf = gpd.read_file(path)
        if gdf.empty or gdf.geometry.isna().all():
            return None, str(gdf.crs) if gdf.crs else None
        b = tuple(map(float, gdf.total_bounds))
        return b, (str(gdf.crs) if gdf.crs else None)
    except Exception:
        # As a last resort, try streaming simple FeatureCollection bbox
        try:
            with _open_text_maybe_gzip(path) as f:
                head = f.read(2_000_000)  # small peek
            obj = json.loads(head)
            if isinstance(obj, dict) and "bbox" in obj and isinstance(obj["bbox"], (list, tuple)) and len(obj["bbox"]) >= 4:
                b = obj["bbox"]
                return (float(b[0]), float(b[1]), float(b[2]), float(b[3])), None
        except Exception:
            pass
        return None, None

def bounds_crs_any(path: str) -> Tuple[Optional[Tuple[float,float,float,float]], Optional[str], str]:
    """
    Returns (bounds, crs_str, kind) where kind is 'parquet' or 'geojson'.
    Bounds are in source CRS (may not be EPSG:4326).
    """
    lower = path.lower()
    if lower.endswith(PARQUET_EXTS):
        b, crs = bounds_crs_parquet(path)
        return b, crs, "parquet"
    elif lower.endswith(GEOJSON_EXTS):
        if HAS_FIONA:
            b, crs = bounds_crs_geojson_via_fiona(path)
        else:
            b, crs = bounds_crs_geojson_via_geopandas(path)
        return b, crs, "geojson"
    else:
        return None, None, "unknown"

# -------------------------
# Map building
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Interactive map of MBRs for GeoParquet and GeoJSON tiles (Leaflet/Folium).")
    ap.add_argument("--inputs", nargs="+", required=True, help="Files, globs, or directories (mixed types allowed)")
    ap.add_argument("--out", default="tiles_mbr_map.html", help="Output HTML file")
    ap.add_argument("--assume-crs", default="EPSG:4326", help="CRS to assume if missing (default EPSG:4326)")
    ap.add_argument("--opacity", type=float, default=0.15, help="Fill opacity of boxes (0–1)")
    ap.add_argument("--color", default="#e74c3c", help="Stroke color (hex)")
    ap.add_argument("--tiles", default="OpenStreetMap", help="Basemap (e.g., 'OpenStreetMap', 'CartoDB positron')")
    ap.add_argument("--labels", action="store_true", help="Show filename labels on rectangles")
    args = ap.parse_args()

    files = list_inputs(args.inputs)
    if not files:
        raise SystemExit("No matching files found.")

    # Collect rectangles (always stored in EPSG:4326 for the map)
    rects = []     # list of shapely boxes
    tooltips = []  # strings
    any_bounds = False

    for fp in files:
        b, crs_str, kind = bounds_crs_any(fp)
        if not b:
            print(f"[WARN] No bounds for {fp}")
            continue

        any_bounds = True
        minx, miny, maxx, maxy = b

        # Build a GeoDataFrame with one box to reproject to EPSG:4326 if needed
        g = gpd.GeoDataFrame({"file": [os.path.basename(fp)]},
                             geometry=[box(minx, miny, maxx, maxy)],
                             crs=(crs_str if crs_str else args.assume_crs))

        if str(g.crs) != "EPSG:4326":
            try:
                g = g.to_crs(4326)
            except Exception as e:
                print(f"[WARN] Reprojection failed for {fp} (CRS={g.crs}): {e}")
                # Fall back to assuming it's already lon/lat
                pass

        geom = g.geometry.iloc[0]
        rects.append(geom)

        # Tooltip
        if kind == "parquet":
            n = row_count_parquet(fp)
            tip = f"{os.path.basename(fp)} — {n:,} rows" if n >= 0 else os.path.basename(fp)
        else:
            n = feature_count_geojson(fp)
            tip = f"{os.path.basename(fp)} — {n:,} features" if n >= 0 else os.path.basename(fp)
        tooltips.append(tip)

        print(f"[OK] {fp}  bounds={tuple(map(float, g.total_bounds))}  kind={kind}  rows/features={n}")

    if not any_bounds or not rects:
        raise SystemExit("No geometries found to compute MBRs.")

    tiles_gdf = gpd.GeoDataFrame({"tooltip": tooltips}, geometry=rects, crs="EPSG:4326")

    # Create Leaflet map
    m = folium.Map(location=[20, 0], zoom_start=2, tiles=args.tiles)
    minx, miny, maxx, maxy = tiles_gdf.total_bounds
    m.fit_bounds([[miny, minx], [maxy, maxx]])

    for i, row in tiles_gdf.iterrows():
        geom = row.geometry
        coords = [(y, x) for x, y in geom.exterior.coords]
        label = row["tooltip"] if args.labels else None
        folium.Polygon(
            locations=coords,
            color=args.color,
            weight=2,
            fill=True,
            fill_color=args.color,
            fill_opacity=args.opacity,
            tooltip=label or row["tooltip"],
        ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(args.out)
    print(f"✔ Interactive map written to {args.out}")

if __name__ == "__main__":
    main()
