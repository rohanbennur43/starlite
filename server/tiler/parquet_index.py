import geopandas as gpd
from pathlib import Path

class ParquetIndex:
    def __init__(self, folder):
        self.folder = Path(folder)

    @staticmethod
    def parse_parquet_bbox(fname):
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
    def intersects(a, b):
        aminlon, aminlat, amaxlon, amaxlat = a
        bminlon, bminlat, bmaxlon, bmaxlat = b
        return not (amaxlon < bminlon or aminlon > bmaxlon or amaxlat < bminlat or aminlat > bmaxlat)

    def find_intersecting_files(self, bbox_4326):
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
    def load_and_reproject(path):
        gdf = gpd.read_parquet(path)
        if gdf.crs is None:
            gdf = gdf.set_crs(4326)
        if gdf.crs.to_epsg() != 3857:
            gdf = gdf.to_crs(3857)
        return gdf
