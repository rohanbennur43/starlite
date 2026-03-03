"""Tile assignment with reservoir sampling for bounded-memory MVT generation."""
import logging
from collections import defaultdict
from .helpers import hist_value_from_prefix, mercator_bounds_to_tile_range
import random

logger = logging.getLogger(__name__)

MAX_GEOMS_PER_TILE = 25000   # choose your number


class TileAssigner:
    def __init__(self, zooms, prefix, threshold):
        logger.debug(f"Initializing TileAssigner: zooms={zooms}, threshold={threshold}")
        self.zooms = zooms
        self.prefix = prefix
        self.threshold = threshold
        self.nonempty = {z: set() for z in zooms}
        self.buckets = {z: defaultdict(list) for z in zooms}


    def _reservoir_insert(self, z, x, y, geom):
        """Insert *geom* into the tile bucket using reservoir sampling (Algorithm R).

        Keeps at most ``MAX_GEOMS_PER_TILE`` features per tile.  Once the
        bucket is full, each new geometry replaces an existing entry with
        probability ``k / (n + 1)``, giving every geometry an equal chance of
        being in the final sample regardless of input order.
        """
        bucket = self.buckets[z][(x, y)]
        k = MAX_GEOMS_PER_TILE
        n = len(bucket)

        if n < k:
            bucket.append(geom)
            return

        # Reservoir sampling: accept with probability k/(n+1)
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
                        self._reservoir_insert(z, x, y, (geom, attrs))
                        assigned += 1
            if assigned > 0:
                logger.debug(f"Assigned geometry to {assigned} tiles at zoom {z}")
