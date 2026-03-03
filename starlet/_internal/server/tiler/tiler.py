import logging
from pathlib import Path
from time import perf_counter

from .tiler_bounds import TileBounds
from .parquet_index import ParquetIndex
from .mvt_encoder import MVTEncoder
from .tile_cache import TileCache
from shapely.geometry import mapping

logger = logging.getLogger(__name__)

from shapely.geometry import (
    Point,
    LineString,
    Polygon,
    MultiPoint,
    MultiLineString,
    MultiPolygon,
    GeometryCollection
)

def explode_collections(geom):
    if geom is None or geom.is_empty:
        return []

    if isinstance(geom, GeometryCollection):
        out = []
        for g in geom.geoms:
            out.extend(explode_collections(g))
        return out

    if isinstance(geom, (Polygon, MultiPolygon, LineString, MultiLineString, Point, MultiPoint)):
        return [geom]

    return []

class VectorTiler:
    """On-demand MVT tile server with a three-tier lookup.

    For each tile request the lookup order is:
      1. **Memory** — LRU cache (``TileCache``, default 256 entries)
      2. **Disk** — pre-generated ``.mvt`` files under ``<dataset>/mvt/z/x/y.mvt``
      3. **Generate** — reads intersecting GeoParquet tiles, clips/transforms
         geometries, and encodes a fresh MVT on the fly

    Generated tiles are persisted to disk and promoted into the memory cache
    so that subsequent requests are served from cache.
    """

    def __init__(self, dataset_root: str, memory_cache_size: int = 256) -> None:
        self.dataset_root = Path(dataset_root)
        self.parquet_dir = self.dataset_root / "parquet_tiles"
        self.mvt_dir = self.dataset_root / "mvt"
        self.index = ParquetIndex(self.parquet_dir)

        self.cache = TileCache(memory_cache_size)

    def tile_path(self, z: int, x: int, y: int) -> Path:
        return self.mvt_dir / str(z) / str(x) / f"{y}.mvt"

    def generate(self, z: int, x: int, y: int) -> bytes:
        t0 = perf_counter()
        bounds = TileBounds(z, x, y)
        encoder = MVTEncoder(bounds.bbox_3857, bounds.tile_poly_3857)

        try:
            intersecting = self.index.find_intersecting_files(bounds.bbox_4326)
        except Exception as e:
            logger.error("[TileGen] z=%d x=%d y=%d index error: %s", z, x, y, e)
            return encoder.empty_tile()

        if not intersecting:
            logger.debug("[TileGen] z=%d x=%d y=%d no intersecting files", z, x, y)
            return encoder.empty_tile()

        features = []

        for pf in intersecting:
            try:
                gdf = self.index.load_and_reproject(pf)
            except Exception as e:
                logger.error("[TileGen] z=%d x=%d y=%d load failed %s: %s", z, x, y, pf, e)
                continue

            try:
                clipped = encoder.clip_to_tile(gdf)
            except Exception as e:
                logger.error("[TileGen] z=%d x=%d y=%d clip failed %s: %s", z, x, y, pf, e)
                continue

            if clipped.empty:
                continue

            for idx, row in clipped.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue

                # extract attributes (all non geometry columns)
                attrs = {k: v for k, v in row.items() if k != "geometry" and v is not None}

                parts = explode_collections(geom)
                if not parts:
                    continue

                for part in parts:
                    if part.is_empty:
                        continue

                    try:
                        scaled = encoder.transform_geom(
                            part,
                            lambda xx, yy, zz=None:
                                TileBounds.scale_to_tile_coords(xx, yy, bounds.bbox_3857)
                        )
                    except Exception as e:
                        logger.error("[TileGen] z=%d x=%d y=%d transform failed: %s", z, x, y, e)
                        continue

                    if scaled.is_empty:
                        continue

                    features.append({
                        "geometry": mapping(scaled),
                        "properties": attrs
                    })

        if not features:
            return encoder.empty_tile()

        try:
            elapsed_ms = (perf_counter() - t0) * 1000
            logger.info("[TileGen] z=%d x=%d y=%d features=%d elapsed=%.1fms",
                        z, x, y, len(features), elapsed_ms)
            return encoder.encode(features)
        except Exception as e:
            logger.error("[TileGen] z=%d x=%d y=%d encode failed: %s", z, x, y, e)
            return encoder.empty_tile()

    def get_tile(self, z: int, x: int, y: int) -> bytes:
        key = (z, x, y)

        cached = self.cache.get(key)
        if cached is not None:
            logger.debug("[Cache] HIT memory z=%d x=%d y=%d", z, x, y)
            return cached

        path = self.tile_path(z, x, y)

        if path.exists():
            t0 = perf_counter()
            data = path.read_bytes()
            elapsed_ms = (perf_counter() - t0) * 1000
            logger.debug("[Cache] HIT disk z=%d x=%d y=%d elapsed=%.1fms", z, x, y, elapsed_ms)
            self.cache.put(key, data)
            return data

        logger.info("[Cache] MISS z=%d x=%d y=%d — generating", z, x, y)

        tile_bytes = self.generate(z, x, y)

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(tile_bytes)
        except Exception as e:
            logger.error("[Cache] disk write failed z=%d x=%d y=%d: %s", z, x, y, e)

        self.cache.put(key, tile_bytes)
        return tile_bytes
