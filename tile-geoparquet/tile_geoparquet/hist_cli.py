import argparse
import logging

from .hist_pyramid import build_histograms_for_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser(
        description="Build per-tile histogram pyramids from GeoParquet tiles with per-row-group partials."
    )
    ap.add_argument("--tiles-dir", required=True, help="Directory containing tile .parquet files.")
    ap.add_argument("--outdir", required=True, help="Where to write histogram tiles (.npy + .json per level).")
    ap.add_argument("--geom-col", default="geometry", help="Geometry column name (default: geometry).")
    ap.add_argument("--grid-size", type=int, default=4096, help="Base resolution N for NxN histogram (default 4096).")
    ap.add_argument("--levels", type=int, default=10, help="Number of pyramid levels to produce (default 10).")
    ap.add_argument("--out-crs", default="EPSG:3857", help="Target CRS (default EPSG:3857 Web Mercator).")
    ap.add_argument("--dtype", default="float64", choices=["float64", "float32"], help="Stored dtype (default float64).")

    # Parallelism & partials
    ap.add_argument("--hist-max-parallel", type=int, default=8, help="Max tiles processed concurrently (default 8).")
    ap.add_argument("--hist-rg-parallel", type=int, default=4, help="Row-group partials per tile in parallel (default 4).")
    ap.add_argument("--keep-partials", action="store_true", help="Persist per-row-group partial histograms (*.npy).")
    ap.add_argument("--partials-dir", default=None, help="Directory for partials (default: <outdir>/partials/<tile_id>/).")

    args = ap.parse_args()

    build_histograms_for_dir(
        tiles_dir=args.tiles_dir,
        outdir=args.outdir,
        geom_col=args.geom_col,
        grid_size=args.grid_size,
        levels=args.levels,
        out_crs=args.out_crs,
        dtype=args.dtype,
        hist_max_parallel=args.hist_max_parallel,
        hist_rg_parallel=args.hist_rg_parallel,
        keep_partials=args.keep_partials,
        partials_dir=args.partials_dir,
    )


if __name__ == "__main__":
    main()