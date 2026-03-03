"""Filename-based spatial index for GeoParquet tiles.

Parquet tiles are named ``tile_XXXXXX__minx_miny_maxx_maxy.parquet``.  The
bounding box is extracted from the filename to enable fast MBR intersection
filtering without reading file metadata.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import geopandas as gpd


class ParquetIndex:
    """Spatial index that parses bounding boxes from Parquet filenames."""

    def __init__(self, folder: Path) -> None:
        self.folder = Path(folder)

    @staticmethod
    def parse_parquet_bbox(fname: str) -> List[float]:
        parts = fname.replace(".parquet", "").split("__")[1].split("_")
        nums = []
        temp = []
        for p in parts:
            temp.append(p)
            if len(temp) == 2:
                nums.append(float(temp[0] + "." + temp[1]))
                temp = []
        return nums

    @staticmethod
    def intersects(a: Tuple[float, ...], b: Tuple[float, ...]) -> bool:
        aminlon, aminlat, amaxlon, amaxlat = a
        bminlon, bminlat, bmaxlon, bmaxlat = b
        return not (amaxlon < bminlon or aminlon > bmaxlon or amaxlat < bminlat or aminlat > bmaxlat)

    def find_intersecting_files(self, bbox_4326: Tuple[float, float, float, float]) -> List[Path]:
        result = []
        for pf in self.folder.glob("*.parquet"):
            try:
                pminlon, pminlat, pmaxlon, pmaxlat = self.parse_parquet_bbox(pf.name)
            except:
                continue

            if self.intersects((pminlon, pminlat, pmaxlon, pmaxlat), bbox_4326):
                result.append(pf)

        return result

    @staticmethod
    def load_and_reproject(path: Path) -> gpd.GeoDataFrame:
        gdf = gpd.read_parquet(path)
        if gdf.crs is None:
            gdf = gdf.set_crs(4326)
        if gdf.crs.to_epsg() != 3857:
            gdf = gdf.to_crs(3857)
        return gdf
