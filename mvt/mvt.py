from __future__ import annotations
import math
import argparse
from pathlib import Path
from collections import defaultdict
import logging

import numpy as np
import geopandas as gpd
from shapely.geometry import box, mapping
from shapely import make_valid, ops
from pyproj import Transformer
import mapbox_vector_tile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bucket_mvt")

# -------------------------------------------------------------------
# Web Mercator constants
# -------------------------------------------------------------------

WORLD_MINX = -20037508.342789244
WORLD_MINY = -20037508.342789244
WORLD_MAXX =  20037508.342789244
WORLD_MAXY =  20037508.342789244

WORLD_W = WORLD_MAXX - WORLD_MINX
WORLD_H = WORLD_MAXY - WORLD_MINY

EXTENT = 4096
TF_4326_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

# -------------------------------------------------------------------
# Histogram loader and prefix sum
# -------------------------------------------------------------------

def load_histogram(hist_path):
    logger.info(f"Loading histogram from {hist_path}")
    hist = np.load(hist_path)
    logger.info(f"Histogram shape: {hist.shape}")
    prefix = hist.cumsum(axis=0).cumsum(axis=1)
    logger.info(f"Computed prefix sum for histogram")
    return prefix

# -------------------------------------------------------------------
# Histogram tile density using prefix
# -------------------------------------------------------------------

# def hist_value_from_prefix(prefix: np.ndarray, z: int, x: int, y: int) -> float:
#     H, W = prefix.shape

#     minx, miny, maxx, maxy = mercator_tile_bounds(z, x, y)

#     j0 = int((minx - WORLD_MINX) / WORLD_W * W)
#     j1 = int((maxx - WORLD_MINX) / WORLD_W * W) - 1

#     i0 = int((WORLD_MAXY - maxy) / WORLD_H * H)
#     i1 = int((WORLD_MAXY - miny) / WORLD_H * H) - 1

#     j0 = max(0, min(j0, W - 1))
#     j1 = max(0, min(j1, W - 1))
#     i0 = max(0, min(i0, H - 1))
#     i1 = max(0, min(i1, H - 1))

#     if i0 > i1 or j0 > j1:
#         return 0.0

#     A = prefix[i1, j1]
#     B = prefix[i0 - 1, j1] if i0 > 0 else 0
#     C = prefix[i1, j0 - 1] if j0 > 0 else 0
#     D = prefix[i0 - 1, j0 - 1] if (i0 > 0 and j0 > 0) else 0

#     return A - B - C + D

def hist_value_from_prefix(prefix: np.ndarray, z: int, x: int, y: int) -> float:
    H, W = prefix.shape

    # Determine histogram zoom level
    # Assumes histogram is square and power of two
    hist_zoom = int(round(math.log2(W)))

    # ------------------------------------------------------------------
    # Case 1: exact match
    # ------------------------------------------------------------------
    if z == hist_zoom:
        # Direct lookup of bin (no prefix needed)
        if 0 <= y < H and 0 <= x < W:
            A = prefix[y, x]
            B = prefix[y - 1, x] if y > 0 else 0
            C = prefix[y, x - 1] if x > 0 else 0
            D = prefix[y - 1, x - 1] if (y > 0 and x > 0) else 0
            return A - B - C + D
        else:
            return 0.0

    # ------------------------------------------------------------------
    # Case 2: z < hist_zoom
    # Aggregate many histogram cells into one tile
    # ------------------------------------------------------------------
    if z < hist_zoom:
        scale = 2 ** (hist_zoom - z)
        # Each tile at z corresponds to a block of size scale by scale in histogram space
        j0 = x * scale
        j1 = (x + 1) * scale - 1
        i0 = y * scale
        i1 = (y + 1) * scale - 1

        # Clip to histogram bounds
        j0 = max(0, min(j0, W - 1))
        j1 = max(0, min(j1, W - 1))
        i0 = max(0, min(i0, H - 1))
        i1 = max(0, min(i1, H - 1))

        if i0 > i1 or j0 > j1:
            return 0.0

        A = prefix[i1, j1]
        B = prefix[i0 - 1, j1] if i0 > 0 else 0
        C = prefix[i1, j0 - 1] if j0 > 0 else 0
        D = prefix[i0 - 1, j0 - 1] if (i0 > 0 and j0 > 0) else 0
        return A - B - C + D

    # ------------------------------------------------------------------
    # Case 3: z > hist_zoom
    # Subdivide histogram bin evenly among children
    # ------------------------------------------------------------------
    # For each tile at z, find parent histogram bin
    scale = 2 ** (z - hist_zoom)
    parent_x = x // scale
    parent_y = y // scale

    # Check bounds
    if not (0 <= parent_x < W and 0 <= parent_y < H):
        return 0.0

    # Parent histogram bin value at (parent_y, parent_x)
    A = prefix[parent_y, parent_x]
    B = prefix[parent_y - 1, parent_x] if parent_y > 0 else 0
    C = prefix[parent_y, parent_x - 1] if parent_x > 0 else 0
    D = prefix[parent_y - 1, parent_x - 1] if (parent_y > 0 and parent_x > 0) else 0
    parent_val = A - B - C + D

    # Split evenly among the scale by scale children
    # This gives consistent density at higher zooms
    child_val = parent_val / (scale * scale)
    return child_val


# -------------------------------------------------------------------
# Tile math
# -------------------------------------------------------------------

def mercator_tile_bounds(z, x, y):
    n = 2 ** z
    tile_w = WORLD_W / n

    minx = WORLD_MINX + x * tile_w
    maxx = minx + tile_w

    maxy = WORLD_MAXY - y * tile_w
    miny = maxy - tile_w

    return minx, miny, maxx, maxy

def mercator_bounds_to_tile_range(z, minx, miny, maxx, maxy):
    n = 2 ** z
    tile_w = WORLD_W / n

    tx0 = int((minx - WORLD_MINX) // tile_w)
    tx1 = int((maxx - WORLD_MINX) // tile_w)
    ty0 = int((WORLD_MAXY - maxy) // tile_w)
    ty1 = int((WORLD_MAXY - miny) // tile_w)

    tx0 = max(tx0, 0)
    ty0 = max(ty0, 0)
    tx1 = min(tx1, n - 1)
    ty1 = min(ty1, n - 1)

    return tx0, ty0, tx1, ty1

def scale_to_tile(xx, yy, tb):
    minx, miny, maxx, maxy = tb
    xs = (xx - minx) / (maxx - minx) * EXTENT
    ys = (yy - miny) / (maxy - miny) * EXTENT
    return xs, ys

# -------------------------------------------------------------------
# Main bucket tiler
# -------------------------------------------------------------------

def explode_geom(g):
    if g.is_empty:
        return []
    if g.geom_type != "GeometryCollection":
        return [g]
    out = []
    for part in g.geoms:
        out.extend(explode_geom(part))
    return out

def generate_bucket_tiles(
    parquet_dir,
    hist_path,
    outdir,
    last_zoom,
    threshold
):
    zooms = list(range(0, last_zoom + 1))

    prefix = load_histogram(hist_path)

    buckets = {z: defaultdict(list) for z in zooms}
    nonempty_tiles = {z: set() for z in zooms}

    # Step 1 histogram gating
    logger.info(f"Step 1: Histogram gating with threshold={threshold}")
    for z in zooms:
        n = 2 ** z
        count = 0
        for x in range(n):
            for y in range(n):
                v = hist_value_from_prefix(prefix, z, x, y)

                if v >= threshold:
                    nonempty_tiles[z].add((x, y))
                    count += 1
        logger.info(f"  Zoom {z}: {count} non-empty tiles out of {n*n}")
    
    print("Size of nonempty_tiles per zoom level: ")
    for z in zooms:
        print(f"Zoom {z}: {len(nonempty_tiles[z])} tiles")
    # Step 2 assign geometries to buckets
    logger.info(f"Step 2: Reading geometries from {parquet_dir}")
    pf_list = list(Path(parquet_dir).rglob("*.parquet"))
    logger.info(f"Found {len(pf_list)} parquet files")
    for pf in pf_list:
        logger.debug(f"Processing {pf.name}")
        gdf = gpd.read_parquet(pf)
        logger.debug(f"Loaded {len(gdf)} geometries from {pf.name}")

        if gdf.crs is None:
            gdf = gdf.set_crs(4326)

        if gdf.crs.to_epsg() != 3857:
            gdf = gdf.to_crs(3857)

        gdf["geometry"] = gdf["geometry"].apply(make_valid)

        for geom in gdf.geometry:
            if geom.is_empty:
                continue

            minx, miny, maxx, maxy = geom.bounds

            for z in zooms:
                tx0, ty0, tx1, ty1 = mercator_bounds_to_tile_range(
                    z, minx, miny, maxx, maxy
                )

                for x in range(tx0, tx1 + 1):
                    for y in range(ty0, ty1 + 1):
                        if (x, y) in nonempty_tiles[z]:
                            buckets[z][(x, y)].append(geom)

    # Step 3 write tiles
    logger.info("Step 3: Writing MVT tiles")
    outdir = Path(outdir)
    total_tiles_written = 0


    for z in zooms:
        logger.info(f"Processing zoom level {z} ({len(buckets[z])} tiles)")
        for (x, y), geoms in buckets[z].items():
            logger.info(f"Tile z={z} x={x} y={y}: processing {len(geoms)} geometries")
            tb = mercator_tile_bounds(z, x, y)
            tile_poly = box(*tb)
            tol = 1000 / (2 ** z)

            features = []
            clipped_count = 0
            simplified_count = 0

            for g in geoms:
                clipped = g.intersection(tile_poly)
                if clipped.is_empty:
                    continue
                clipped_count += 1

                simp = clipped.simplify(tol, preserve_topology=True)
                if simp.is_empty:
                    continue
                simplified_count += 1

                def to_tile(xx, yy, zz=None):
                    return scale_to_tile(xx, yy, tb)
                for part in explode_geom(simp):
                    final_part = ops.transform(to_tile, part)
                    features.append({
                        "geometry": mapping(final_part),
                        "properties": {},
                    })

                # final_geom = ops.transform(to_tile, simp)

                # features.append({
                #     "geometry": mapping(final_geom),
                #     "properties": {},
                # })

            logger.debug(f"Tile z={z} x={x} y={y}: clipped={clipped_count}, simplified={simplified_count}, final_features={len(features)}")
            
            if not features:
                logger.debug(f"Tile z={z} x={x} y={y}: no features after processing, skipping")
                continue

            layer = {
                "name": "layer0",
                "features": features,
                "extent": EXTENT,
            }

            data = mapbox_vector_tile.encode([layer])

            tp = outdir / str(z) / str(x)
            tp.mkdir(parents=True, exist_ok=True)

            with open(tp / f"{y}.mvt", "wb") as f:
                f.write(data)
                total_tiles_written += 1

            logger.info(f"Wrote tile z={z} x={x} y={y} with {len(features)} features")
    
    logger.info(f"Complete! Wrote {total_tiles_written} MVT tiles to {outdir}")

# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bucket based MVT tile generator")

    parser.add_argument("--parquet_dir", required=True, help="Directory of GeoParquet tiles")
    parser.add_argument("--hist", required=True, help="Path to global histogram numpy file")
    parser.add_argument("--out", required=True, help="Output directory for MVT tiles")
    parser.add_argument("--zoom", type=int, default=7, help="Last zoom level to generate")
    parser.add_argument("--threshold", type=float, default=0, help="Histogram density threshold")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")

    args = parser.parse_args()
    
    # Configure logging level
    level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger.setLevel(level)
    
    logger.info(f"Starting MVT generation: parquet_dir={args.parquet_dir}, hist={args.hist}, out={args.out}, zoom={args.zoom}, threshold={args.threshold}")

    generate_bucket_tiles(
        parquet_dir=args.parquet_dir,
        hist_path=args.hist,
        outdir=args.out,
        last_zoom=args.zoom,
        threshold=args.threshold
    )



