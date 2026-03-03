"""Clip, transform, and encode geometries into Mapbox Vector Tile format."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import geopandas as gpd
import mapbox_vector_tile
from shapely import make_valid, ops


class MVTEncoder:
    """Clips a GeoDataFrame to a tile and encodes features to MVT."""

    def __init__(self, bbox_3857: Tuple[float, float, float, float], tile_poly_3857: Any, extent: int = 4096) -> None:
        self.bbox_3857 = bbox_3857
        self.tile_poly_3857 = tile_poly_3857
        self.extent = extent

    def clip_to_tile(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.apply(make_valid)
        gdf = gdf[~gdf.geometry.is_empty]
        return gdf.clip(self.tile_poly_3857)


    def transform_geom(self, geom: Any, scale_func: Callable) -> Any:
        return ops.transform(scale_func, geom)

    def encode(self, features: List[Dict[str, Any]]) -> bytes:
        
        layer = {
            "name": "layer0",
            "features": features,
            "extent": self.extent
        }
        return mapbox_vector_tile.encode([layer])

    @staticmethod
    def empty_tile(extent: int = 4096) -> bytes:
        layer = {"name": "layer0", "features": [], "extent": extent}
        return mapbox_vector_tile.encode([layer])
