# from __future__ import annotations
# from pathlib import Path
# from typing import Tuple
# import geopandas as gpd
# from shapely.geometry import box, mapping
# from shapely import ops
# import mapbox_vector_tile
# from tqdm import tqdm

# # =========================================================
# # CONFIGURATION
# # =========================================================
# OUTPUT_DIR = Path("ports_tiles_out")
# ZOOM_LEVELS = range(0, 8)  # Zoom levels 0–7
# EXTENT = 4096

# # Web Mercator world bounds
# WORLD_BOUNDS = (
#     -20037508.342789244,  # minx
#     -20037508.342789244,  # miny
#      20037508.342789244,  # maxx
#      20037508.342789244   # maxy
# )

# # =========================================================
# # TILE BOUNDS
# # =========================================================
# def mercator_tile_bounds(z: int, x: int, y: int) -> Tuple[float, float, float, float]:
#     """Return EPSG:3857 bounds for tile (z,x,y) with top-left origin."""
#     n = 2 ** z
#     tile_size = (WORLD_BOUNDS[2] - WORLD_BOUNDS[0]) / n

#     minx = WORLD_BOUNDS[0] + x * tile_size
#     maxx = WORLD_BOUNDS[0] + (x + 1) * tile_size
#     maxy = WORLD_BOUNDS[3] - y * tile_size
#     miny = WORLD_BOUNDS[3] - (y + 1) * tile_size
#     return (minx, miny, maxx, maxy)

# # =========================================================
# # TILE GENERATION
# # =========================================================
# def generate_tiles(gdf: gpd.GeoDataFrame):
#     print(f"Input GeoDataFrame: {len(gdf)} features")
#     print("Original CRS:", gdf.crs)

#     # Ensure CRS exists
#     if gdf.crs is None:
#         print("⚠️  No CRS found, assuming EPSG:4326")
#         gdf = gdf.set_crs(4326)

#     # Reproject to Web Mercator
#     print("Reprojecting to EPSG:3857 ...")
#     gdf = gdf.to_crs(3857)

#     # Clean empty geometries (no buffer!)
#     print("Cleaning geometries ...")
#     gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
#     print(f"After cleaning: {len(gdf)} valid geometries")

#     # Show geometry types
#     print("Geometry types:")
#     print(gdf.geom_type.value_counts())

#     # Print global data bounds
#     data_bounds = gdf.total_bounds
#     print(f"Data bounds (EPSG:3857): {data_bounds}")

#     # Generate tiles
#     for z in ZOOM_LEVELS:
#         print(f"\n=== Generating tiles for zoom {z} ===")
#         n = 2 ** z
#         tiles_written = 0

#         for x in tqdm(range(n), desc=f"Zoom {z}"):
#             for y in range(n):
#                 bounds = mercator_tile_bounds(z, x, y)
#                 tile_box = box(*bounds)

#                 subset = gdf[gdf.intersects(tile_box)]
#                 if subset.empty:
#                     continue

#                 subset = subset.copy()
#                 subset["geom_clip"] = subset.geometry.intersection(tile_box)

#                 features = []
#                 for idx, row in subset.iterrows():
#                     geom = row["geom_clip"]
#                     if geom.is_empty:
#                         continue

#                     # Scale coordinates into [0, EXTENT]
#                     def scale_to_tile(x, y, z=None):
#                         sx = (x - bounds[0]) / (bounds[2] - bounds[0]) * EXTENT
#                         sy = (y - bounds[1]) / (bounds[3] - bounds[1]) * EXTENT
#                         return sx, sy

#                     geom_scaled = ops.transform(scale_to_tile, geom)

#                     features.append({
#                         "geometry": mapping(geom_scaled),
#                         "properties": {
#                             k: str(v) for k, v in row.items()
#                             if k not in ["geometry", "geom_clip"]
#                         },
#                         "id": int(idx) if isinstance(idx, (int, float)) else None
#                     })

#                 if not features:
#                     continue

#                 layer = {
#                     "name": "layer0",
#                     "features": features,
#                     "extent": EXTENT
#                 }

#                 tile_data = mapbox_vector_tile.encode([layer])

#                 out_dir = OUTPUT_DIR / f"{z}/{x}"
#                 out_dir.mkdir(parents=True, exist_ok=True)
#                 with open(out_dir / f"{y}.mvt", "wb") as f:
#                     f.write(tile_data)
#                 tiles_written += 1

#         print(f"✅ Zoom {z}: wrote {tiles_written} non-empty tiles")

#     print("\n✅ Done! Vector tiles written to:", OUTPUT_DIR)

# # =========================================================
# # ENTRY POINT
# # =========================================================
# if __name__ == "__main__":
#     parquet_path = "/Users/rohanbennur/Documents/bigdata-project/repos/ucr-bigdatalab-starmap/original_datasets/ports/ports.parquet"
#     print(f"Reading {parquet_path} ...")
#     gdf = gpd.read_parquet(parquet_path)
#     generate_tiles(gdf)

from __future__ import annotations
from pathlib import Path
from typing import Tuple, List
import geopandas as gpd
from shapely.geometry import box, mapping
from shapely import ops
import mapbox_vector_tile
from tqdm import tqdm

# =========================================================
# CONFIGURATION
# =========================================================
OUTPUT_DIR = Path("roads_tiles_out")
ZOOM_LEVELS = range(0, 8)
EXTENT = 4096

WORLD_BOUNDS = (
    -20037508.342789244,  # minx
    -20037508.342789244,  # miny
     20037508.342789244,  # maxx
     20037508.342789244   # maxy
)

# =========================================================
# PYRAMID PARTITION FUNCTION
# =========================================================
def pyramid_partition(mbr: Tuple[float, float, float, float],
                      min_zoom: int,
                      max_zoom: int) -> List[Tuple[int, int, int]]:
    """Return list of (z, x, y) tiles overlapping geometry MBR."""
    xmin, ymin, xmax, ymax = mbr
    WORLD_MINX, WORLD_MINY, WORLD_MAXX, WORLD_MAXY = WORLD_BOUNDS
    WORLD_WIDTH = WORLD_MAXX - WORLD_MINX
    WORLD_HEIGHT = WORLD_MAXY - WORLD_MINY

    results = []
    for z in range(min_zoom, max_zoom + 1):
        n = 2 ** z
        tile_w = WORLD_WIDTH / n
        tile_h = WORLD_HEIGHT / n

        xmin_c = max(WORLD_MINX, xmin)
        xmax_c = min(WORLD_MAXX, xmax)
        ymin_c = max(WORLD_MINY, ymin)
        ymax_c = min(WORLD_MAXY, ymax)

        x1 = int((xmin_c - WORLD_MINX) / tile_w)
        x2 = int((xmax_c - WORLD_MINX) / tile_w)

        # ✅ FIX: flip Y index to top-left origin (Mapbox convention)
        y1 = int((WORLD_MAXY - ymax_c) / tile_h)
        y2 = int((WORLD_MAXY - ymin_c) / tile_h)

        for x in range(x1, x2 + 1):
            for y in range(y1, y2 + 1):
                results.append((z, x, y))
    return results

# =========================================================
# TILE BOUNDS (TOP-LEFT ORIGIN)
# =========================================================
def mercator_tile_bounds(z: int, x: int, y: int) -> Tuple[float, float, float, float]:
    """Return EPSG:3857 bounds for tile (z,x,y) with top-left origin."""
    n = 2 ** z
    tile_size = (WORLD_BOUNDS[2] - WORLD_BOUNDS[0]) / n
    minx = WORLD_BOUNDS[0] + x * tile_size
    maxx = WORLD_BOUNDS[0] + (x + 1) * tile_size
    maxy = WORLD_BOUNDS[3] - y * tile_size
    miny = WORLD_BOUNDS[3] - (y + 1) * tile_size
    return (minx, miny, maxx, maxy)

# =========================================================
# SCALING FUNCTION
# =========================================================
def scale_to_tile(x: float, y: float, bounds: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Scale coordinates into [0, EXTENT] without vertical flipping."""
    sx = (x - bounds[0]) / (bounds[2] - bounds[0]) * EXTENT
    sy = (y - bounds[1]) / (bounds[3] - bounds[1]) * EXTENT
    return sx, sy

# =========================================================
# TILE GENERATION
# =========================================================
def generate_tiles(gdf: gpd.GeoDataFrame):
    print(f"Input GeoDataFrame: {len(gdf)} features")
    print("Original CRS:", gdf.crs)

    if gdf.crs is None:
        print("⚠️  No CRS found, assuming EPSG:4326")
        gdf = gdf.set_crs(4326)

    print("Reprojecting to EPSG:3857 ...")
    gdf = gdf.to_crs(3857)

    print("Cleaning geometries ...")
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
    print(f"After cleaning: {len(gdf)} valid geometries")

    print("Geometry types:")
    print(gdf.geom_type.value_counts())
    print(f"Data bounds (EPSG:3857): {gdf.total_bounds}")

    # =========================================================
    # Partition geometries
    # =========================================================
    print("\n=== Partitioning geometries into tiles ===")
    tile_map: dict[Tuple[int, int, int], list] = {}
    for idx, geom in tqdm(list(enumerate(gdf.geometry)), desc="Assigning features"):
        xmin, ymin, xmax, ymax = geom.bounds
        tiles = pyramid_partition((xmin, ymin, xmax, ymax),
                                  min(ZOOM_LEVELS),
                                  max(ZOOM_LEVELS))
        for z, x, y in tiles:
            tile_map.setdefault((z, x, y), []).append(idx)
    print(f"Total tiles with data: {len(tile_map)}")

    # =========================================================
    # Generate MVTs
    # =========================================================
    for z in ZOOM_LEVELS:
        print(f"\n=== Generating tiles for zoom {z} ===")
        tiles_written = 0
        zoom_tiles = {k: v for k, v in tile_map.items() if k[0] == z}

        for (z, x, y), indices in tqdm(zoom_tiles.items(), desc=f"Zoom {z}"):
            bounds = mercator_tile_bounds(z, x, y)
            tile_box = box(*bounds)
            subset = gdf.iloc[indices]
            if subset.empty:
                continue

            subset = subset.copy()
            subset["geom_clip"] = subset.geometry.intersection(tile_box)

            features = []
            for idx, row in subset.iterrows():
                geom = row["geom_clip"]
                if geom.is_empty:
                    continue

                geom_scaled = ops.transform(lambda x, y, z=None: scale_to_tile(x, y, bounds), geom)
                features.append({
                    "geometry": mapping(geom_scaled),
                    "properties": {k: str(v) for k, v in row.items()
                                   if k not in ["geometry", "geom_clip"]},
                    "id": int(idx) if isinstance(idx, (int, float)) else None
                })

            if not features:
                continue

            layer = {"name": "layer0", "features": features, "extent": EXTENT}
            tile_data = mapbox_vector_tile.encode([layer])

            out_dir = OUTPUT_DIR / f"{z}/{x}"
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / f"{y}.mvt", "wb") as f:
                f.write(tile_data)
            tiles_written += 1

        print(f"✅ Zoom {z}: wrote {tiles_written} non-empty tiles")

    print("\n✅ Done! Vector tiles written to:", OUTPUT_DIR)

# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    parquet_path = "/Users/rohanbennur/Documents/bigdata-project/repos/ucr-bigdatalab-starmap/original_datasets/highways/roads.parquet"
    print(f"Reading {parquet_path} ...")
    gdf = gpd.read_parquet(parquet_path)
    generate_tiles(gdf)