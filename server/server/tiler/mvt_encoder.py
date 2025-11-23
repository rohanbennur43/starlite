from shapely import ops
from shapely.geometry import mapping
import mapbox_vector_tile

class MVTEncoder:
    def __init__(self, bbox_3857, tile_poly_3857, extent=4096):
        self.bbox_3857 = bbox_3857
        self.tile_poly_3857 = tile_poly_3857
        self.extent = extent

    def clip_to_tile(self, gdf):
        return gdf.clip(self.tile_poly_3857)

    def transform_geom(self, geom, scale_func):
        return ops.transform(scale_func, geom)

    def encode(self, features):
        layer = {
            "name": "layer0",
            "features": features,
            "extent": self.extent
        }
        return mapbox_vector_tile.encode([layer])

    @staticmethod
    def empty_tile(extent=4096):
        layer = {"name": "layer0", "features": [], "extent": extent}
        return mapbox_vector_tile.encode([layer])
