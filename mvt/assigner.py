import logging
from collections import defaultdict
from mvt.helpers import hist_value_from_prefix, mercator_bounds_to_tile_range
import random

logger = logging.getLogger(__name__)

MAX_GEOMS_PER_TILE = 5000   # choose your number


class TileAssigner:
    def __init__(self, zooms, prefix, threshold):
        logger.debug(f"Initializing TileAssigner: zooms={zooms}, threshold={threshold}")
        self.zooms = zooms
        self.prefix = prefix
        self.threshold = threshold
        self.nonempty = {z: set() for z in zooms}
        self.buckets = {z: defaultdict(list) for z in zooms}


    def _reservoir_insert(self, z, x, y, geom):
        bucket = self.buckets[z][(x, y)]
        k = MAX_GEOMS_PER_TILE
        n = len(bucket)

        if n < k:
            # still room
            bucket.append(geom)
            return

        # reservoir algorithm
        j = random.randint(0, n)
        if j < k:
            bucket[j] = geom

    def compute_nonempty(self):
        logger.debug("Computing nonempty tiles from histogram")
        for z in self.zooms:
            n = 2 ** z
            for x in range(n):
                for y in range(n):
                    if hist_value_from_prefix(self.prefix, z, x, y) >= self.threshold:
                        self.nonempty[z].add((x, y))
            logger.debug(f"Zoom {z}: {len(self.nonempty[z])} nonempty tiles")

    def assign_geometry(self, geom, attrs):
        logger.debug(f"Assigning geometry with bounds {geom.bounds}")
        minx, miny, maxx, maxy = geom.bounds

        for z in self.zooms:
            tx0, ty0, tx1, ty1 = mercator_bounds_to_tile_range(z, minx, miny, maxx, maxy)
            assigned = 0
            for x in range(tx0, tx1 + 1):
                for y in range(ty0, ty1 + 1):
                    if (x, y) in self.nonempty[z]:
                        # self.buckets[z][(x, y)].append(geom)
                        self._reservoir_insert(z, x, y, (geom, attrs))
                        assigned += 1
            if assigned > 0:
                logger.debug(f"Assigned geometry to {assigned} tiles at zoom {z}")
