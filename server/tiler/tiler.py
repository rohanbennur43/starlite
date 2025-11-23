from pathlib import Path
from .tiler_bounds import TileBounds
from .parquet_index import ParquetIndex
from .mvt_encoder import MVTEncoder
from shapely.geometry import mapping

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

        intersecting = self.index.find_intersecting_files(bounds.bbox_4326)
        if not intersecting:
            return encoder.empty_tile()

        features = []
        for pf in intersecting:
            gdf = self.index.load_and_reproject(pf)
            clipped = encoder.clip_to_tile(gdf)

            if clipped.empty:
                continue

            for geom in clipped.geometry:
                if geom.is_empty:
                    continue

                scaled = encoder.transform_geom(
                    geom,
                    lambda xx, yy, zz=None: TileBounds.scale_to_tile_coords(xx, yy, bounds.bbox_3857)
                )

                features.append({
                    "geometry": mapping(scaled),
                    "properties": {}
                })

        return encoder.encode(features)

    def get_tile(self, z, x, y):
        path = self.tile_path(z, x, y)
        if path.exists():
            return path.read_bytes()

        tile_bytes = self.generate(z, x, y)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(tile_bytes)
        return tile_bytes
