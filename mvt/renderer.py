import logging
import mapbox_vector_tile
from pathlib import Path
from shapely.geometry import box, mapping
from shapely import make_valid
from shapely.ops import transform

from mvt.helpers import EXTENT, mercator_tile_bounds, explode_geom

logger = logging.getLogger(__name__)


class TileRenderer:
    def __init__(self, outdir):
        logger.info(f"Initializing TileRenderer with outdir={outdir}")
        self.outdir = Path(outdir)

    def render(self, buckets):
        logger.info(f"Starting tile rendering for {len(buckets)} zoom levels")
        total = 0

        for z, tiles in buckets.items():
            logger.info(f"Rendering zoom {z}: {len(tiles)} tiles")

            for (x, y), geoms in tiles.items():
                tb = mercator_tile_bounds(z, x, y)

                pad = (tb[2] - tb[0]) * (256 / EXTENT)
                padded = box(tb[0] - pad, tb[1] - pad, tb[2] + pad, tb[3] + pad)

                def to_tile(xx, yy, zz=None):
                    xt = (xx - tb[0]) / (tb[2] - tb[0]) * EXTENT
                    yt = (yy - tb[1]) / (tb[3] - tb[1]) * EXTENT
                    return xt, yt

                simplify_tol = (tb[2] - tb[0]) * 0.0005
                features = []

                for (g, attrs) in geoms:
                    g2 = g.simplify(simplify_tol, preserve_topology=True)
                    if g2.is_empty:
                        continue
                    g2 = make_valid(g2)

                    try:
                        clipped = g2.intersection(padded)
                    except Exception as e:
                        logger.error(f"Error clipping geometry at z={z}, x={x}, y={y}: {e}")
                        g2 = g2.buffer(0)
                        clipped = g2.intersection(padded)

                    if clipped.is_empty:
                        continue

                    clipped = make_valid(clipped)
                    properties = {k: v for k, v in attrs.items() if v is not None}
                    for part in explode_geom(clipped):
                        transformed = transform(to_tile, part)
                        features.append({
                            "geometry": mapping(transformed),
                            "properties": properties
                        })

                if not features:
                    logger.debug(f"No features for tile z={z} x={x} y={y}, skipping")
                    continue

                # 6. encode tile
                layer = {"name": "layer0", "features": features, "extent": EXTENT}
                data = mapbox_vector_tile.encode([layer])

                # 7. write file
                tile_path = self.outdir / str(z) / str(x)
                tile_path.mkdir(parents=True, exist_ok=True)

                with open(tile_path / f"{y}.mvt", "wb") as f:
                    f.write(data)
                    total += 1
                    logger.debug(f"Wrote tile z={z} x={x} y={y} with {len(features)} features")

        logger.info(f"Rendering complete. Wrote {total} MVT tiles to {self.outdir}")
