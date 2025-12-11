import logging
from mvt.histogram import HistogramLoader
from mvt.streamer import GeometryStreamer
from mvt.assigner import TileAssigner
from mvt.renderer import TileRenderer

logger = logging.getLogger(__name__)


class BucketMVTGenerator:
    def __init__(self, parquet_dir, hist_path, outdir, last_zoom, threshold):
        logger.info(f"Initializing BucketMVTGenerator: parquet_dir={parquet_dir}, outdir={outdir}, last_zoom={last_zoom}, threshold={threshold}")
        self.parquet_dir = parquet_dir
        self.hist_path = hist_path
        self.outdir = outdir
        self.last_zoom = last_zoom
        self.threshold = threshold

    def run(self):
        logger.info("Starting MVT generation pipeline")
        prefix = HistogramLoader(self.hist_path).load()
        zooms = list(range(0, self.last_zoom + 1))
        logger.debug(f"Zooms to process: {zooms}")

        assigner = TileAssigner(zooms, prefix, self.threshold)
        logger.info("Computing nonempty tiles")
        assigner.compute_nonempty()
        total_nonempty = sum(len(v) for v in assigner.nonempty.values())
        logger.info(f"Found {total_nonempty} nonempty tiles across all zoom levels")

        logger.info(f"Streaming geometries from {self.parquet_dir}")
        geom_count = 0
        streamer = GeometryStreamer(self.parquet_dir)
        for geom, attrs in streamer.iter_geometries():
            assigner.assign_geometry(geom, attrs)
            geom_count += 1
            if geom_count % 10000 == 0:
                logger.debug(f"Processed {geom_count} geometries")
        logger.info(f"Assigned {geom_count} geometries to tiles")

        logger.info(f"Rendering tiles to {self.outdir}")
        TileRenderer(self.outdir).render(assigner.buckets)
        logger.info("Pipeline execution complete")
