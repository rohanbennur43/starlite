from __future__ import annotations
import argparse
import logging
from pathlib import Path
from time import perf_counter
import math

from .datasource import GeoParquetSource, GeoJSONSource, is_geojson_path
from .assigner import TileAssignerFromCSV, RSGroveAssigner
from .orchestrator import RoundOrchestrator
from .writer_pool import SortMode, SortKey
from .hist_pyramid import build_histograms_for_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(relativeCreated).0fms] %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def build_source(input_path: str, geom_col: str):
    if is_geojson_path(input_path):
        logger.info("Using GeoJSONSource for %s", input_path)
        return GeoJSONSource(input_path)
    else:
        logger.info("Using GeoParquetSource for %s", input_path)
        return GeoParquetSource(input_path)


def _parse_sort_mode(s: str) -> str:
    s = (s or "").strip().lower()
    if s in ("", "none"): return SortMode.NONE
    if s in ("columns", "cols", "column"): return SortMode.COLUMNS
    if s in ("z", "zorder", "z-order", "morton"): return SortMode.ZORDER
    if s in ("hilbert", "h"): return SortMode.HILBERT
    raise argparse.ArgumentTypeError(f"Unsupported --sort-mode: {s}")


def _parse_sort_keys(keys: str | None):
    # format examples:
    #   "colA,colB" (both ascending)
    #   "colA:asc,colB:desc"
    if not keys:
        return None
    out = []
    for tok in keys.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" in tok:
            name, order = tok.split(":", 1)
            out.append(SortKey(name.strip(), ascending=(order.strip().lower() != "desc")))
        else:
            out.append(SortKey(tok, True))
    return out


_SIZE_SUFFIXES = {
    "kb": 1024,
    "mb": 1024 ** 2,
    "gb": 1024 ** 3,
    "tb": 1024 ** 4,
}


def _parse_partition_size_bytes(raw: str) -> int:
    """
    Parse a human-friendly size string into bytes.

    Examples: "1073741824", "1gb", "512mb".
    """
    s = raw.strip().lower()
    if s.isdigit():
        return int(s)

    for suffix, mul in _SIZE_SUFFIXES.items():
        if s.endswith(suffix):
            num = s[: -len(suffix)].strip()
            try:
                return int(float(num) * mul)
            except ValueError:
                break

    raise argparse.ArgumentTypeError(f"Invalid partition size: {raw}")


def main():
    ap = argparse.ArgumentParser(
        description="GeoJSON/GeoParquet → tiled GeoParquet (round-based, bounded writers, single final writes)."
    )
    # Source
    ap.add_argument("--input", required=True, help="Path to input GeoJSON or GeoParquet.")
    ap.add_argument("--geom-col", default="geometry", help="Geometry column name (default: geometry).")

    # Output / run
    ap.add_argument("--outdir", required=True, help="Output directory for tiles.")
    ap.add_argument("--compression", default="zstd", help="Parquet compression codec (default: zstd).")
    ap.add_argument("--max-parallel-files", type=int, default=64,
                    help="Max files to write concurrently each round.")
    ap.add_argument("--sort-mode", type=_parse_sort_mode, default=SortMode.ZORDER,
                    help="none|columns|zorder|hilbert (hilbert currently = zorder).")
    ap.add_argument("--sort-keys", default=None,
                    help='Only for --sort-mode=columns. Example: "colA:asc,colB:desc".')
    ap.add_argument("--sfc-bits", type=int, default=16,
                    help="Bits per axis for Z-order/Hilbert key (typical: 16–20).")

    # Mode
    ap.add_argument("--index", help="CSV index with columns: id,minx,miny,maxx,maxy (legacy mode).")
    ap.add_argument("--num-tiles", type=int, help="Number of tiles to build via RSGrove (preferred).")
    ap.add_argument("--seed", type=int, default=42, help="Seed for RSGrove (if --num-tiles is used).")
    ap.add_argument("--partition-size", type=_parse_partition_size_bytes, default=1 << 30,
                    help="Target partition size (bytes). Accepts suffixes KB, MB, GB, TB. Default: 1GB.")

    # Sampling (RSGrove)
    ap.add_argument("--sample-ratio", type=float, default=1.0,
                    help="Bernoulli sampling probability for centroids (0<r<=1).")
    ap.add_argument("--sample-cap", type=int, default=None,
                    help="Reservoir sampling cap K (wins over ratio if provided).")

    args = ap.parse_args()

    source = build_source(args.input, geom_col=args.geom_col)
    input_size_bytes = Path(args.input).stat().st_size
    computed_partitions = max(1, math.ceil(input_size_bytes / args.partition_size))
    if args.num_tiles:
        target_partitions = int(args.num_tiles)
        logger.info(
            "Input size=%d bytes, partition size=%d bytes -> computed %d partitions (using --num-tiles=%d).",
            input_size_bytes, args.partition_size, computed_partitions, target_partitions,
        )
    else:
        target_partitions = computed_partitions
        logger.info(
            "Input size=%d bytes, partition size=%d bytes -> using %d partitions.",
            input_size_bytes, args.partition_size, target_partitions,
        )

    if args.index:
        logger.info("Using TileAssignerFromCSV with index=%s", args.index)
        assigner = TileAssignerFromCSV(args.index, geom_col=args.geom_col)
    else:
        logger.info(
            "Building RSGroveAssigner from source with num_tiles=%d seed=%d ratio=%.4f cap=%s",
            target_partitions, args.seed, args.sample_ratio, str(args.sample_cap),
        )
        assigner = RSGroveAssigner.from_source(
            tables=source.iter_tables(),  # streaming
            num_partitions=target_partitions,
            geom_col=args.geom_col,
            seed=args.seed,
            sample_ratio=args.sample_ratio,
            sample_cap=args.sample_cap,
        )


    orchestrator = RoundOrchestrator(
        source=source,
        assigner=assigner,
        outdir=args.outdir+"/parquet_tiles",
        max_parallel_files=args.max_parallel_files,
        compression=args.compression,
        sort_mode=args.sort_mode,
        sort_keys=_parse_sort_keys(args.sort_keys),
        sfc_bits=args.sfc_bits,
    )
    orchestrator.run()


    logger.info("Tiling complete. Starting histogram generation.")

    build_histograms_for_dir(
        tiles_dir=args.outdir + "/parquet_tiles",
        outdir=args.outdir + "/histograms",
        geom_col=args.geom_col,
        grid_size=4096,
        dtype="float64",
        hist_max_parallel=8,
        hist_rg_parallel=4,
    )

if __name__ == "__main__":
    main()
