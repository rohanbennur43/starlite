from pathlib import Path
from .tiler_bounds import TileBounds
from .parquet_index import ParquetIndex
from .mvt_encoder import MVTEncoder
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
    def __init__(self, dataset_root):
        self.dataset_root = Path(dataset_root)
        self.parquet_dir = self.dataset_root / "parquet_tiles"
        self.mvt_dir = self.dataset_root / "mvt"
        self.index = ParquetIndex(self.parquet_dir)

    def tile_path(self, z, x, y):
        return self.mvt_dir / str(z) / str(x) / f"{y}.mvt"

    def generate(self, z, x, y):
        bounds = TileBounds(z, x, y)
        encoder = MVTEncoder(bounds.bbox_3857, bounds.tile_poly_3857)

        try:
            intersecting = self.index.find_intersecting_files(bounds.bbox_4326)
        except Exception as e:
            print("Error finding intersecting files", e)
            return encoder.empty_tile()

        if not intersecting:
            return encoder.empty_tile()

        features = []

        for pf in intersecting:
            try:
                gdf = self.index.load_and_reproject(pf)
            except Exception as e:
                print("Failed to load parquet", pf, e)
                continue

            try:
                clipped = encoder.clip_to_tile(gdf)
            except Exception as e:
                print("Clip failed on", pf, e)
                continue

            if clipped.empty:
                continue

            for geom in clipped.geometry:
                if geom is None or geom.is_empty:
                    continue

                # flatten geometry collections
                parts = explode_collections(geom)
                if not parts:
                    continue

                for part in parts:
                    if part.is_empty:
                        continue

                    try:
                        scaled = encoder.transform_geom(
                            part,
                            lambda xx, yy, zz=None: TileBounds.scale_to_tile_coords(xx, yy, bounds.bbox_3857)
                        )
                    except Exception as e:
                        print("Transform failed", e)
                        continue

                    if scaled.is_empty:
                        continue

                    features.append({
                        "geometry": mapping(scaled),
                        "properties": {}
                    })

        # return empty tile if nothing survived
        if not features:
            return encoder.empty_tile()

        # safe encode
        try:
            tile = encoder.encode(features)
            if tile is None:
                return encoder.empty_tile()
            return tile
        except Exception as e:
            print("MVT encode failed", e)
            return encoder.empty_tile()

                    
    def get_tile(self, z, x, y):
        path = self.tile_path(z, x, y)
        if path.exists():
            return path.read_bytes()

        tile_bytes = self.generate(z, x, y)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(tile_bytes)
        return tile_bytes
