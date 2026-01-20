import logging
from pathlib import Path

from .tiler_bounds import TileBounds
from .parquet_index import ParquetIndex
from .mvt_encoder import MVTEncoder
from .tile_cache import TileCache
from shapely.geometry import mapping

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
    def __init__(self, dataset_root, memory_cache_size=256):
        self.dataset_root = Path(dataset_root)
        self.parquet_dir = self.dataset_root / "parquet_tiles"
        self.mvt_dir = self.dataset_root / "mvt"
        self.index = ParquetIndex(self.parquet_dir)

        self.cache = TileCache(memory_cache_size)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(relativeCreated).0fms] [%(levelname)s] %(message)s"
        )

    def tile_path(self, z, x, y):
        return self.mvt_dir / str(z) / str(x) / f"{y}.mvt"

    def generate(self, z, x, y):
        bounds = TileBounds(z, x, y)
        encoder = MVTEncoder(bounds.bbox_3857, bounds.tile_poly_3857)

        try:
            intersecting = self.index.find_intersecting_files(bounds.bbox_4326)
        except Exception as e:
            logging.error(f"Error finding intersecting files: {e}")
            return encoder.empty_tile()

        if not intersecting:
            logging.info(f"No intersecting files found for {bounds}")
            return encoder.empty_tile()

        features = []

        for pf in intersecting:
            try:
                gdf = self.index.load_and_reproject(pf)
            except Exception as e:
                logging.error(f"Failed to load parquet {pf}: {e}")
                continue

            try:
                clipped = encoder.clip_to_tile(gdf)
            except Exception as e:
                logging.error(f"Clip failed on {pf}: {e}")
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
                    logging.error(f"Transform failed: {e}")
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
            logging.info(f"Encoding {len(features)} features for tile {bounds}")
            return encoder.encode(features)
        except Exception as e:
            logging.error(f"MVT encode failed: {e}")
            return encoder.empty_tile()

    def get_tile(self, z, x, y):
        key = (z, x, y)

        cached = self.cache.get(key)
        if cached is not None:
            logging.info(f"In memory cache hit for tile {z}/{x}/{y}")
            return cached

        logging.info(f"In memory cache miss for tile {z}/{x}/{y}")

        path = self.tile_path(z, x, y)

        if path.exists():
            logging.info(f"Serving tile from disk {path}")
            data = path.read_bytes()
            self.cache.put(key, data)
            return data

        logging.info(f"Tile not found on disk {path}, generating")

        tile_bytes = self.generate(z, x, y)

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(tile_bytes)
            logging.info(f"Tile written to disk {path}")
        except Exception as e:
            logging.error(f"Failed to write tile to disk {path}: {e}")

        self.cache.put(key, tile_bytes)
        return tile_bytes
