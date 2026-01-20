# tile_geoparquet/hist_cli.py
import argparse
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(relativeCreated).0fms] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def debug_generate_pngs(outdir: str):
    """
    TEMPORARY DEBUG:
    After global.npy and global_prefix.npy are created, generate:
        global.png
        global_prefix.png
    This will be removed later.
    """

    outdir = Path(outdir)

    global_npy = outdir / "global.npy"
    prefix_npy = outdir / "global_prefix.npy"

    if not global_npy.exists():
        logger.warning("global.npy not found — skipping PNG generation.")
        return

    logger.info("DEBUG: generating PNGs for global histograms...")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---------------- GLOBAL.png ----------------
    total = np.load(global_npy, mmap_mode="r").astype(float)
    img = np.log1p(total)

    fig = plt.figure(figsize=(8, 8), dpi=512/8)
    ax = plt.axes([0, 0, 1, 1])
    ax.axis("off")
    # NOTE: origin='lower' so north is up in the debug image
    ax.imshow(img, origin="lower", cmap="magma")
    plt.savefig(outdir / "global.png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # ---------------- global_prefix.png ----------------
    if prefix_npy.exists():
        prefix = np.load(prefix_npy, mmap_mode="r").astype(float)
        img2 = np.log1p(prefix)

        fig = plt.figure(figsize=(8, 8), dpi=512/8)
        ax = plt.axes([0, 0, 1, 1])
        ax.axis("off")
        ax.imshow(img2, origin="lower", cmap="viridis")
        plt.savefig(outdir / "global_prefix.png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    logger.info("DEBUG: PNGs written (global.png, global_prefix.png)")


def main():
    from .hist_pyramid import build_histograms_for_dir

    ap = argparse.ArgumentParser(
        description="Generate per-tile histograms and global histogram (with prefix sum). "
                    "PNG generation included TEMPORARILY for debugging."
    )

    ap.add_argument("--tiles-dir", required=True, help="Directory containing parquet tiles.")
    ap.add_argument("--outdir", required=True, help="Directory for histogram output.")
    ap.add_argument("--geom-col", default="geometry")
    ap.add_argument("--grid-size", type=int, default=4096)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--hist-max-parallel", type=int, default=8)
    ap.add_argument("--hist-rg-parallel", type=int, default=4)

    args = ap.parse_args()

    # ---------------- Run per-tile + global builder ----------------
    build_histograms_for_dir(
        tiles_dir=args.tiles_dir,
        outdir=args.outdir,
        geom_col=args.geom_col,
        grid_size=args.grid_size,
        dtype=args.dtype,
        hist_max_parallel=args.hist_max_parallel,
        hist_rg_parallel=args.hist_rg_parallel,
    )

    # ---------------- TEMP: Debug PNG output ----------------
    debug_generate_pngs(args.outdir)


if __name__ == "__main__":
    main()


