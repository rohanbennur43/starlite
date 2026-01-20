# from __future__ import annotations
# import math
# import argparse
# from pathlib import Path
# from collections import defaultdict
# import logging

# import numpy as np
# import geopandas as gpd
# from shapely.geometry import box, mapping
# from shapely import make_valid, ops
# from pyproj import Transformer
# import mapbox_vector_tile

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("bucket_mvt")

# WORLD_MINX = -20037508.342789244
# WORLD_MINY = -20037508.342789244
# WORLD_MAXX = 20037508.342789244
# WORLD_MAXY = 20037508.342789244

# WORLD_W = WORLD_MAXX - WORLD_MINX
# WORLD_H = WORLD_MAXY - WORLD_MINY

# EXTENT = 4096
# TF_4326_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


# def load_histogram(hist_path):
#     logger.info(f"Loading histogram from {hist_path}")
#     hist = np.load(hist_path)
#     prefix = hist.cumsum(axis=0).cumsum(axis=1)
#     return prefix


# def hist_value_from_prefix(prefix: np.ndarray, z: int, x: int, y: int) -> float:
#     H, W = prefix.shape
#     hist_zoom = int(round(math.log2(W)))

#     if z == hist_zoom:
#         if 0 <= y < H and 0 <= x < W:
#             A = prefix[y, x]
#             B = prefix[y - 1, x] if y > 0 else 0
#             C = prefix[y, x - 1] if x > 0 else 0
#             D = prefix[y - 1, x - 1] if (y > 0 and x > 0) else 0
#             return A - B - C + D
#         return 0.0

#     if z < hist_zoom:
#         scale = 2 ** (hist_zoom - z)
#         j0 = x * scale
#         j1 = (x + 1) * scale - 1
#         i0 = y * scale
#         i1 = (y + 1) * scale - 1

#         j0 = max(0, min(j0, W - 1))
#         j1 = max(0, min(j1, W - 1))
#         i0 = max(0, min(i0, H - 1))
#         i1 = max(0, min(i1, H - 1))

#         if i0 > i1 or j0 > j1:
#             return 0.0

#         A = prefix[i1, j1]
#         B = prefix[i0 - 1, j1] if i0 > 0 else 0
#         C = prefix[i1, j0 - 1] if j0 > 0 else 0
#         D = prefix[i0 - 1, j0 - 1] if (i0 > 0 and j0 > 0) else 0
#         return A - B - C + D

#     scale = 2 ** (z - hist_zoom)
#     parent_x = x // scale
#     parent_y = y // scale

#     if not (0 <= parent_x < W and 0 <= parent_y < H):
#         return 0.0

#     A = prefix[parent_y, parent_x]
#     B = prefix[parent_y - 1, parent_x] if parent_y > 0 else 0
#     C = prefix[parent_y, parent_x - 1] if parent_x > 0 else 0
#     D = prefix[parent_y - 1, parent_x - 1] if (parent_y > 0 and parent_x > 0) else 0
#     parent_val = A - B - C + D

#     return parent_val / (scale * scale)


# def mercator_tile_bounds(z, x, y):
#     n = 2 ** z
#     tile_w = WORLD_W / n

#     minx = WORLD_MINX + x * tile_w
#     maxx = minx + tile_w

#     maxy = WORLD_MAXY - y * tile_w
#     miny = maxy - tile_w

#     return minx, miny, maxx, maxy


# def mercator_bounds_to_tile_range(z, minx, miny, maxx, maxy):
#     n = 2 ** z
#     tile_w = WORLD_W / n

#     tx0 = int((minx - WORLD_MINX) // tile_w)
#     tx1 = int((maxx - WORLD_MINX) // tile_w)
#     ty0 = int((WORLD_MAXY - maxy) // tile_w)
#     ty1 = int((WORLD_MAXY - miny) // tile_w)

#     tx0 = max(tx0, 0)
#     ty0 = max(ty0, 0)
#     tx1 = min(tx1, n - 1)
#     ty1 = min(ty1, n - 1)

#     return tx0, ty0, tx1, ty1


# def explode_geom(g):
#     if g.is_empty:
#         return []
#     if g.geom_type != "GeometryCollection":
#         return [g]
#     out = []
#     for part in g.geoms:
#         out.extend(explode_geom(part))
#     return out


# def generate_bucket_tiles(
#     parquet_dir,
#     hist_path,
#     outdir,
#     last_zoom,
#     threshold
# ):
#     zooms = list(range(0, last_zoom + 1))
#     prefix = load_histogram(hist_path)

#     buckets = {z: defaultdict(list) for z in zooms}
#     nonempty_tiles = {z: set() for z in zooms}

#     for z in zooms:
#         n = 2 ** z
#         for x in range(n):
#             for y in range(n):
#                 if hist_value_from_prefix(prefix, z, x, y) >= threshold:
#                     nonempty_tiles[z].add((x, y))

#     pf_list = list(Path(parquet_dir).rglob("*.parquet"))
#     for pf in pf_list:
#         gdf = gpd.read_parquet(pf)

#         if gdf.crs is None:
#             gdf = gdf.set_crs(4326)

#         if gdf.crs.to_epsg() != 3857:
#             gdf = gdf.to_crs(3857)

#         gdf["geometry"] = gdf["geometry"].apply(make_valid)

#         for geom in gdf.geometry:
#             if geom.is_empty:
#                 continue

#             minx, miny, maxx, maxy = geom.bounds

#             for z in zooms:
#                 tx0, ty0, tx1, ty1 = mercator_bounds_to_tile_range(
#                     z, minx, miny, maxx, maxy
#                 )

#                 for x in range(tx0, tx1 + 1):
#                     for y in range(ty0, ty1 + 1):
#                         if (x, y) in nonempty_tiles[z]:
#                             buckets[z][(x, y)].append(geom)

#     outdir = Path(outdir)
#     total_tiles_written = 0

#     for z in zooms:
#         for (x, y), geoms in buckets[z].items():
#             tb = mercator_tile_bounds(z, x, y)
#             tile_poly = box(*tb)

#             tol = (tb[2] - tb[0]) * 0.0005     # FIXED TOLERANCE

#             features = []

#             # FIXED TRANSFORM
#             def to_tile(xs, ys, zs=None):
#                 xs = np.asarray(xs)
#                 ys = np.asarray(ys)
#                 xx = (xs - tb[0]) / (tb[2] - tb[0]) * EXTENT
#                 yy = (ys - tb[1]) / (tb[3] - tb[1]) * EXTENT
#                 return xx, yy
                
#             def to_tile_single(x, y, z=None):
#                 xx = (x - tb[0]) / (tb[2] - tb[0]) * EXTENT
#                 yy = (y - tb[1]) / (tb[3] - tb[1]) * EXTENT
#                 return xx, yy

#             for g in geoms:
#                 clipped = g.intersection(tile_poly)
#                 if clipped.is_empty:
#                     continue

#                 simp = clipped.simplify(tol, preserve_topology=True)
#                 if simp.is_empty:
#                     continue
#                 simp = make_valid(simp)

#                 # Drop tiny scraps that appear on tile borders
#                 if simp.area < 1e-6:
#                     continue

#                 for part in explode_geom(simp):
#                     final_part = ops.transform(to_tile_single, part)
#                     features.append({
#                         "geometry": mapping(final_part),
#                         "properties": {},
#                     })

#             if not features:
#                 continue

#             layer = {
#                 "name": "layer0",
#                 "features": features,
#                 "extent": EXTENT,
#             }

#             data = mapbox_vector_tile.encode([layer])

#             tp = outdir / str(z) / str(x)
#             tp.mkdir(parents=True, exist_ok=True)
#             with open(tp / f"{y}.mvt", "wb") as f:
#                 f.write(data)
#                 total_tiles_written += 1

#     logger.info(f"Complete. Wrote {total_tiles_written} MVT tiles to {outdir}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Bucket based MVT tile generator")

#     parser.add_argument("--dir", required=True)
#     parser.add_argument("--zoom", type=int, default=7)
#     parser.add_argument("--threshold", type=float, default=0)
#     parser.add_argument("--log-level", default="INFO")

#     args = parser.parse_args()
#     level = getattr(logging, args.log_level.upper(), logging.INFO)
#     logger.setLevel(level)

#     generate_bucket_tiles(
#         parquet_dir=args.dir + "/parquet_tiles",
#         hist_path=args.dir + "/histograms/global.npy",
#         outdir=args.dir + "/mvt",
#         last_zoom=args.zoom,
#         threshold=args.threshold
#     )

# from __future__ import annotations
# import math
# import argparse
# from pathlib import Path
# from collections import defaultdict
# import logging

# import numpy as np
# import geopandas as gpd
# from shapely.geometry import box, mapping
# from shapely import make_valid, ops
# from shapely.ops import transform
# from pyproj import Transformer
# import mapbox_vector_tile

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("bucket_mvt")

# WORLD_MINX = -20037508.342789244
# WORLD_MINY = -20037508.342789244
# WORLD_MAXX = 20037508.342789244
# WORLD_MAXY = 20037508.342789244

# WORLD_W = WORLD_MAXX - WORLD_MINX
# WORLD_H = WORLD_MAXY - WORLD_MINY

# EXTENT = 4096
# TF_4326_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


# def load_histogram(hist_path):
#     logger.info(f"Loading histogram from {hist_path}")
#     hist = np.load(hist_path)
#     prefix = hist.cumsum(axis=0).cumsum(axis=1)
#     return prefix


# def hist_value_from_prefix(prefix: np.ndarray, z: int, x: int, y: int) -> float:
#     H, W = prefix.shape
#     hist_zoom = int(round(math.log2(W)))

#     if z == hist_zoom:
#         if 0 <= y < H and 0 <= x < W:
#             A = prefix[y, x]
#             B = prefix[y - 1, x] if y > 0 else 0
#             C = prefix[y, x - 1] if x > 0 else 0
#             D = prefix[y - 1, x - 1] if (y > 0 and x > 0) else 0
#             return A - B - C + D
#         return 0.0

#     if z < hist_zoom:
#         scale = 2 ** (hist_zoom - z)
#         j0 = x * scale
#         j1 = (x + 1) * scale - 1
#         i0 = y * scale
#         i1 = (y + 1) * scale - 1

#         j0 = max(0, min(j0, W - 1))
#         j1 = max(0, min(j1, W - 1))
#         i0 = max(0, min(i0, H - 1))
#         i1 = max(0, min(i1, H - 1))

#         if i0 > i1 or j0 > j1:
#             return 0.0

#         A = prefix[i1, j1]
#         B = prefix[i0 - 1, j1] if i0 > 0 else 0
#         C = prefix[i1, j0 - 1] if j0 > 0 else 0
#         D = prefix[i0 - 1, j0 - 1] if (i0 > 0 and j0 > 0) else 0
#         return A - B - C + D

#     scale = 2 ** (z - hist_zoom)
#     parent_x = x // scale
#     parent_y = y // scale

#     if not (0 <= parent_x < W and 0 <= parent_y < H):
#         return 0.0

#     A = prefix[parent_y, parent_x]
#     B = prefix[parent_y - 1, parent_x] if parent_y > 0 else 0
#     C = prefix[parent_y, parent_x - 1] if parent_x > 0 else 0
#     D = prefix[parent_y - 1, parent_x - 1] if (parent_y > 0 and parent_x > 0) else 0
#     parent_val = A - B - C + D

#     return parent_val / (scale * scale)


# def mercator_tile_bounds(z, x, y):
#     n = 2 ** z
#     tile_w = WORLD_W / n

#     minx = WORLD_MINX + x * tile_w
#     maxx = minx + tile_w

#     maxy = WORLD_MAXY - y * tile_w
#     miny = maxy - tile_w

#     return minx, miny, maxx, maxy


# def mercator_bounds_to_tile_range(z, minx, miny, maxx, maxy):
#     n = 2 ** z
#     tile_w = WORLD_W / n

#     tx0 = int((minx - WORLD_MINX) // tile_w)
#     tx1 = int((maxx - WORLD_MINX) // tile_w)
#     ty0 = int((WORLD_MAXY - maxy) // tile_w)
#     ty1 = int((WORLD_MAXY - miny) // tile_w)

#     tx0 = max(tx0, 0)
#     ty0 = max(ty0, 0)
#     tx1 = min(tx1, n - 1)
#     ty1 = min(ty1, n - 1)

#     return tx0, ty0, tx1, ty1


# def explode_geom(g):
#     if g.is_empty:
#         return []
#     if g.geom_type != "GeometryCollection":
#         return [g]
#     out = []
#     for part in g.geoms:
#         out.extend(explode_geom(part))
#     return out


# def generate_bucket_tiles(
#     parquet_dir,
#     hist_path,
#     outdir,
#     last_zoom,
#     threshold
# ):

#     zooms = list(range(0, last_zoom + 1))
#     prefix = load_histogram(hist_path)

#     buckets = {z: defaultdict(list) for z in zooms}
#     nonempty_tiles = {z: set() for z in zooms}

#     for z in zooms:
#         n = 2 ** z
#         for x in range(n):
#             for y in range(n):
#                 if hist_value_from_prefix(prefix, z, x, y) >= threshold:
#                     nonempty_tiles[z].add((x, y))

#     pf_list = list(Path(parquet_dir).rglob("*.parquet"))
#     for pf in pf_list:
#         gdf = gpd.read_parquet(pf)

#         if gdf.crs is None:
#             gdf = gdf.set_crs(4326)

#         if gdf.crs.to_epsg() != 3857:
#             gdf = gdf.to_crs(3857)

#         gdf["geometry"] = gdf["geometry"].apply(make_valid)

#         for geom in gdf.geometry:
#             if geom.is_empty:
#                 continue

#             minx, miny, maxx, maxy = geom.bounds

#             for z in zooms:
#                 tx0, ty0, tx1, ty1 = mercator_bounds_to_tile_range(
#                     z, minx, miny, maxx, maxy
#                 )

#                 for x in range(tx0, tx1 + 1):
#                     for y in range(ty0, ty1 + 1):
#                         if (x, y) in nonempty_tiles[z]:
#                             buckets[z][(x, y)].append(geom)

#     outdir = Path(outdir)
#     total_tiles_written = 0

#     for z in zooms:
#         for (x, y), geoms in buckets[z].items():
#             tb = mercator_tile_bounds(z, x, y)

#             # padded tile to avoid seam lines
#             # pad = (tb[2] - tb[0]) * 0.03
#             pad = (tb[2] - tb[0])* (512/EXTENT)  # FIXED PADDING
        
#             padded_poly = box(tb[0] - pad, tb[1] - pad, tb[2] + pad, tb[3] + pad)

#             # simple transform to local tile coordinates
#             def to_tile(xi, yi, zi=None):
#                 xt = (xi - tb[0]) / (tb[2] - tb[0]) * EXTENT
#                 yt = (yi - tb[1]) / (tb[3] - tb[1]) * EXTENT
#                 return xt, yt

#             features = []

#             # simplify before clipping since this preserves border alignment
#             simplify_tol = (tb[2] - tb[0]) * 0.0005

#             for g in geoms:
#                 if g.is_empty:
#                     continue

#                 g2 = g.simplify(simplify_tol, preserve_topology=True)
#                 if g2.is_empty:
#                     continue

#                 clipped = g2.intersection(padded_poly)
#                 if clipped.is_empty:
#                     continue

#                 clipped = make_valid(clipped)

#                 for part in explode_geom(clipped):
#                     final_part = transform(to_tile, part)
#                     features.append({
#                         "geometry": mapping(final_part),
#                         "properties": {},
#                     })

#             if not features:
#                 continue

#             layer = {
#                 "name": "layer0",
#                 "features": features,
#                 "extent": EXTENT,
#             }

#             data = mapbox_vector_tile.encode([layer])

#             tp = outdir / str(z) / str(x)
#             tp.mkdir(parents=True, exist_ok=True)
#             with open(tp / f"{y}.mvt", "wb") as f:
#                 f.write(data)
#                 total_tiles_written += 1

#     logger.info(f"Complete. Wrote {total_tiles_written} MVT tiles to {outdir}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Bucket based MVT tile generator")

#     parser.add_argument("--dir", required=True)
#     parser.add_argument("--zoom", type=int, default=7)
#     parser.add_argument("--threshold", type=float, default=0)
#     parser.add_argument("--log-level", default="INFO")

#     args = parser.parse_args()
#     level = getattr(logging, args.log_level.upper(), logging.INFO)
#     logger.setLevel(level)

#     generate_bucket_tiles(
#         parquet_dir=args.dir + "/parquet_tiles",
#         hist_path=args.dir + "/histograms/global.npy",
#         outdir=args.dir + "/mvt",
#         last_zoom=args.zoom,
#         threshold=args.threshold
#     )



from __future__ import annotations
import math
import argparse
from pathlib import Path
from collections import defaultdict
import logging

import numpy as np
import geopandas as gpd
from shapely.geometry import box, mapping, LineString, Polygon, MultiLineString, MultiPolygon
from shapely import make_valid, ops
from shapely.ops import transform
from pyproj import Transformer
import mapbox_vector_tile

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(relativeCreated).0fms] %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("bucket_mvt")

WORLD_MINX = -20037508.342789244
WORLD_MINY = -20037508.342789244
WORLD_MAXX = 20037508.342789244
WORLD_MAXY = 20037508.342789244

WORLD_W = WORLD_MAXX - WORLD_MINX
WORLD_H = WORLD_MAXY - WORLD_MINY

EXTENT = 4096
TF_4326_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def load_histogram(hist_path):
    logger.info(f"Loading histogram from {hist_path}")
    hist = np.load(hist_path)
    prefix = hist.cumsum(axis=0).cumsum(axis=1)
    return prefix


def hist_value_from_prefix(prefix: np.ndarray, z: int, x: int, y: int) -> float:
    H, W = prefix.shape
    hist_zoom = int(round(math.log2(W)))

    if z == hist_zoom:
        if 0 <= y < H and 0 <= x < W:
            A = prefix[y, x]
            B = prefix[y - 1, x] if y > 0 else 0
            C = prefix[y, x - 1] if x > 0 else 0
            D = prefix[y - 1, x - 1] if (y > 0 and x > 0) else 0
            return A - B - C + D
        return 0.0

    if z < hist_zoom:
        scale = 2 ** (hist_zoom - z)
        j0 = x * scale
        j1 = (x + 1) * scale - 1
        i0 = y * scale
        i1 = (y + 1) * scale - 1

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

    scale = 2 ** (z - hist_zoom)
    parent_x = x // scale
    parent_y = y // scale

    if not (0 <= parent_x < W and 0 <= parent_y < H):
        return 0.0

    A = prefix[parent_y, parent_x]
    B = prefix[parent_y - 1, parent_x] if parent_y > 0 else 0
    C = prefix[parent_y, parent_x - 1] if parent_x > 0 else 0
    D = prefix[parent_y - 1, parent_x - 1] if (parent_y > 0 and parent_x > 0) else 0
    parent_val = A - B - C + D

    return parent_val / (scale * scale)


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


def explode_geom(g):
    if g.is_empty:
        return []
    if g.geom_type != "GeometryCollection":
        return [g]
    out = []
    for part in g.geoms:
        out.extend(explode_geom(part))
    return out


# -------------------------------------------------------------------
# Sampling helper
# -------------------------------------------------------------------
MAX_POINTS = 5000

def sample_coords(coords):
    if len(coords) <= MAX_POINTS:
        return coords
    stride = max(1, len(coords) // MAX_POINTS)
    new = coords[::stride]
    if new[0] != new[-1]:
        new.append(new[0])
    return new

def sample_polygon(poly):
    ext = sample_coords(list(poly.exterior.coords))
    interiors = []
    for ring in poly.interiors:
        interiors.append(sample_coords(list(ring.coords)))
    new_poly = Polygon(ext, interiors)
    return make_valid(new_poly)

def flatten_polygons(parts):
    """Flatten nested MultiPolygon children into a simple list of Polygon objects."""
    out = []
    for p in parts:
        if isinstance(p, Polygon):
            out.append(p)
        elif isinstance(p, MultiPolygon):
            out.extend([child for child in p.geoms if isinstance(child, Polygon)])
        else:
            # Ignore weird entries that cannot become polygons
            continue
    return out

def sample_geometry(g):
    try:
        # LineString
        if isinstance(g, LineString):
            coords = sample_coords(list(g.coords))
            return LineString(coords)

        # Polygon
        if isinstance(g, Polygon):
            return sample_polygon(g)

        # MultiLineString
        if isinstance(g, MultiLineString):
            return MultiLineString([sample_geometry(p) for p in g.geoms])

        # MultiPolygon
        if isinstance(g, MultiPolygon):
            sampled_parts = [sample_polygon(p) for p in g.geoms]
            flat = flatten_polygons(sampled_parts)
            return MultiPolygon(flat)

        # GeometryCollection or other with geoms
        if hasattr(g, "geoms"):
            sampled = [sample_geometry(p) for p in g.geoms]
            return type(g)(sampled)

        return g

    except Exception as e:
        logger.warning(f"Sampling failed for geometry: {e}")
        return g


# -------------------------------------------------------------------

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

    for z in zooms:
        n = 2 ** z
        for x in range(n):
            for y in range(n):
                if hist_value_from_prefix(prefix, z, x, y) >= threshold:
                    nonempty_tiles[z].add((x, y))

    pf_list = list(Path(parquet_dir).rglob("*.parquet"))
    for pf in pf_list:
        gdf = gpd.read_parquet(pf)

        if gdf.crs is None:
            gdf = gdf.set_crs(4326)

        if gdf.crs.to_epsg() != 3857:
            gdf = gdf.to_crs(3857)

        gdf["geometry"] = gdf["geometry"].apply(make_valid)

        for geom in gdf.geometry:
            if geom.is_empty:
                continue

            # sampling step added
            geom = sample_geometry(geom)

            minx, miny, maxx, maxy = geom.bounds

            for z in zooms:
                tx0, ty0, tx1, ty1 = mercator_bounds_to_tile_range(
                    z, minx, miny, maxx, maxy
                )

                for x in range(tx0, tx1 + 1):
                    for y in range(ty0, ty1 + 1):
                        if (x, y) in nonempty_tiles[z]:
                            buckets[z][(x, y)].append(geom)

    outdir = Path(outdir)
    total_tiles_written = 0

    for z in zooms:
        for (x, y), geoms in buckets[z].items():
            tb = mercator_tile_bounds(z, x, y)

            pad = (tb[2] - tb[0]) * (256 / EXTENT)
            padded_poly = box(tb[0] - pad, tb[1] - pad, tb[2] + pad, tb[3] + pad)

            def to_tile(xi, yi, zi=None):
                xt = (xi - tb[0]) / (tb[2] - tb[0]) * EXTENT
                yt = (yi - tb[1]) / (tb[3] - tb[1]) * EXTENT
                return xt, yt

            features = []
            simplify_tol = (tb[2] - tb[0]) * 0.0005

            for g in geoms:
                if g.is_empty:
                    continue

                g2 = g.simplify(simplify_tol, preserve_topology=True)
                if g2.is_empty:
                    continue

                clipped = g2.intersection(padded_poly)
                if clipped.is_empty:
                    continue

                clipped = make_valid(clipped)

                for part in explode_geom(clipped):
                    final_part = transform(to_tile, part)
                    features.append({
                        "geometry": mapping(final_part),
                        "properties": {},
                    })

            if not features:
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

    logger.info(f"Complete. Wrote {total_tiles_written} MVT tiles to {outdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bucket based MVT tile generator")

    parser.add_argument("--dir", required=True)
    parser.add_argument("--zoom", type=int, default=7)
    parser.add_argument("--threshold", type=float, default=0)
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()
    level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger.setLevel(level)

    generate_bucket_tiles(
        parquet_dir=args.dir + "/parquet_tiles",
        hist_path=args.dir + "/histograms/global.npy",
        outdir=args.dir + "/mvt",
        last_zoom=args.zoom,
        threshold=args.threshold
    )
