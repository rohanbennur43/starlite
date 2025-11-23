from shapely.geometry import box
from pyproj import Transformer

LIM = 20037508.342789244
EXTENT = 4096

tf_3857_to_4326 = Transformer.from_crs(3857, 4326, always_xy=True)
tf_4326_to_3857 = Transformer.from_crs(4326, 3857, always_xy=True)

class TileBounds:
    def __init__(self, z, x, y):
        self.z = z
        self.x = x
        self.y = y
        self.bbox_3857 = self.compute_mercator_bounds()
        self.bbox_4326 = self.compute_wgs84_bounds()
        self.tile_poly_3857 = box(*self.bbox_3857)

    def compute_mercator_bounds(self):
        n = 2 ** self.z
        tile_size = (2 * LIM) / n

        minx = -LIM + self.x * tile_size
        maxx = -LIM + (self.x + 1) * tile_size
        maxy = LIM - self.y * tile_size
        miny = LIM - (self.y + 1) * tile_size
        return minx, miny, maxx, maxy

    def compute_wgs84_bounds(self):
        minx, miny, maxx, maxy = self.bbox_3857
        lon1, lat1 = tf_3857_to_4326.transform(minx, miny)
        lon2, lat2 = tf_3857_to_4326.transform(maxx, maxy)
        return lon1, lat1, lon2, lat2

    @staticmethod
    def scale_to_tile_coords(xx, yy, bbox_3857):
        minx, miny, maxx, maxy = bbox_3857
        xs = (xx - minx) / (maxx - minx) * EXTENT
        ys = (yy - miny) / (maxy - miny) * EXTENT
        return xs, ys
