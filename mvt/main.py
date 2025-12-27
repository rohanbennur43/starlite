import argparse
import logging
from time import perf_counter
from mvt.generator import BucketMVTGenerator

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir", required=True)
    parser.add_argument("--zoom", type=int, default=7)
    parser.add_argument("--threshold", type=float, default=0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger.info(f"Starting MVT generation with dir={args.dir}, zoom={args.zoom}, threshold={args.threshold}, log_level={args.log_level}")

    gen = BucketMVTGenerator(
        parquet_dir=args.dir + "/parquet_tiles",
        hist_path=args.dir + "/histograms/global.npy",
        outdir=args.dir + "/mvt",
        last_zoom=args.zoom,
        threshold=args.threshold
    )
    tiles_start = perf_counter()
    gen.run()
    tiles_end = perf_counter()
    logger.info("MVT generation completed successfully in %.2f seconds", tiles_end - tiles_start)
