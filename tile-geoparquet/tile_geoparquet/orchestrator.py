# tile_geoparquet/orchestrator.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Set, Dict
import logging

import pyarrow as pa
import pyarrow.parquet as pq

from .datasource import DataSource, GeoParquetSource
from .assigner import TileAssignerFromCSV
from .writer_pool import WriterPool
from .utils_large import ensure_large_types

logger = logging.getLogger(__name__)


class _OverflowWriter:
    """
    Streams rows that don't fit in this round into a single overflow Parquet file.

    - Defers opening ParquetWriter until the first batch arrives so we can
      adopt the *transformed* (large_*) schema produced by ensure_large_types().
    - Ensures parent directories exist before opening.
    """
    def __init__(self, path: Path, compression: str, geom_col: str):
        self.path = Path(path)
        self.compression = compression
        self.geom_col = geom_col

        self._pw: Optional[pq.ParquetWriter] = None
        self._rows = 0

        # Make sure the parent dir exists (fixes FileNotFoundError).
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write_batch(self, tbl: pa.Table) -> None:
        if tbl is None or tbl.num_rows == 0:
            return

        # Ensure large types up front to avoid "offset overflow" on large binary/list.
        tbl = tbl.combine_chunks()
        tbl = ensure_large_types(tbl, geom_col=self.geom_col)

        if self._pw is None:
            # Open the writer the moment we know the (possibly widened) schema.
            logger.debug("Opening overflow writer at %s", self.path)
            self._pw = pq.ParquetWriter(str(self.path), schema=tbl.schema, compression=self.compression)

        self._pw.write_table(tbl)
        self._rows += tbl.num_rows

    def close(self) -> int:
        if self._pw is not None:
            self._pw.close()
        return self._rows


class RoundOrchestrator:
    """
    Round-based tiler with bounded parallel writers:

    - At most (max_parallel_files - 1) tiles are included in the current round; any other
      tile rows are diverted to a single overflow file for this round.
    - Within the round, we buffer all rows for the selected tiles (no per-row I/O).
    - After all input is scanned, we flush the selected tiles once (WriterPool.flush_all).
    - If the overflow file has rows, it becomes the input for the next round; repeat.
    - Process ends when a round produces no overflow rows.

    Notes:
      * WriterPool is "flush-once": it concatenates, (optionally) sorts, recomputes bbox,
        patches GeoParquet metadata, and writes one Parquet per tile.
      * Overflow file is written progressively per batch to bound memory and to make
        RSS reflect actual spill behavior.
    """

    def __init__(
        self,
        source: DataSource,
        assigner: TileAssignerFromCSV,
        outdir: str,
        max_parallel_files: int,
        compression: str = "zstd",
        geom_col: str = "geometry",
        sort_mode: str = "zorder",
        sort_keys: Optional[str] = None,  # CLI passes raw string; WriterPool normalizes internally if needed
        sfc_bits: int = 16,
    ):
        self.source = source
        self.assigner = assigner
        self.outdir = outdir
        self.max_parallel_files = int(max_parallel_files)
        self.compression = compression
        self.geom_col = geom_col
        self.sort_mode = sort_mode
        self.sort_keys = sort_keys
        self.sfc_bits = int(sfc_bits)

        # Source schema is used for reference/logging; WriterPool/Overflow may widen types.
        self.src_schema: pa.Schema = source.schema()

    def _run_one_round(self, ds: DataSource, round_id: int) -> Optional[Path]:
        logger.info("Starting round %d", round_id)
        Path(self.outdir).mkdir(parents=True, exist_ok=True)

        # Prepare pool for this round. It buffers everything and writes once per tile.
        pool = WriterPool(
            outdir=self.outdir,
            compression=self.compression,
            geom_col=self.geom_col,
            max_parallel_files=self.max_parallel_files,  # used for concurrency during flush
            sort_mode=self.sort_mode,
            sort_keys=self.sort_keys,
            sfc_bits=self.sfc_bits,
        )

        # Overflow sink for tiles not admitted this round (bounded by cap below).
        overflow_path = Path(self.outdir) / f"_overflow_round_{round_id}.parquet"
        ow = _OverflowWriter(
            path=overflow_path,
            compression=self.compression,
            geom_col=self.geom_col,
        )

        # Admit at most (max_parallel_files - 1) tiles into this round; the "last slot" is reserved
        # conceptually for overflow (so we never exceed MPF) and to match your original semantics.
        # (Since WriterPool buffers in memory, "open tiles" here means: tiles whose rows are kept
        # for this round; all other tile rows spill to overflow.)
        open_tiles: Set[str] = set()
        cap = max(1, self.max_parallel_files - 1)

        # Stream the input: partition each batch to tiles; decide keep/spill.
        for batch_idx, batch in enumerate(ds.iter_tables()):
            parts: Dict[str, pa.Table] = self.assigner.partition_by_tile(batch)
            logger.debug(
                "Round %d: batch %d -> %d tiles (rows: %d)",
                round_id, batch_idx, len(parts), batch.num_rows
            )

            for tile_id, sub in parts.items():
                if tile_id in open_tiles or len(open_tiles) < cap:
                    if tile_id not in open_tiles:
                        logger.debug("Round %d: admitting tile %s", round_id, tile_id)
                        open_tiles.add(tile_id)
                    pool.append(tile_id, sub)
                else:
                    # Spill this tile's rows to overflow for next round.
                    ow.write_batch(sub)

        # Done scanning input for this round: flush current tiles once.
        if open_tiles:
            logger.info(
                "Round %d: flushing %d tiles (cap=%d, admitted=%d)",
                round_id, len(open_tiles), cap, len(open_tiles)
            )
        else:
            logger.info("Round %d: no tiles admitted; only overflow will be produced (if any).", round_id)

        pool.flush_all()

        # Close overflow and decide next input.
        overflow_rows = ow.close()
        if overflow_rows > 0:
            logger.info("Round %d: overflow file created at %s (rows=%d)",
                        round_id, overflow_path, overflow_rows)
            return overflow_path

        # No overflow rows — if an empty file was created for any reason, remove it.
        if overflow_path.exists():
            try:
                pf = pq.ParquetFile(str(overflow_path))
                if pf.metadata.num_rows == 0:
                    overflow_path.unlink(missing_ok=True)
            except Exception:
                # Best effort cleanup; ignore issues reading a partially-written file.
                pass
        return None

    def run(self) -> None:
        round_id = 0
        ds: DataSource = self.source

        while True:
            overflow_path = self._run_one_round(ds, round_id)
            if overflow_path is None:
                break

            logger.info(
                "Round %d produced overflow; continuing with overflow file %s",
                round_id, overflow_path
            )
            # Feed the overflow back as the next round's input.
            ds = GeoParquetSource(str(overflow_path))
            round_id += 1

        # Final cleanup: remove any zero-row overflow files that might remain.
        for p in Path(self.outdir).glob("_overflow_round_*.parquet"):
            try:
                pf = pq.ParquetFile(str(p))
                if pf.metadata.num_rows == 0:
                    p.unlink(missing_ok=True)
            except Exception:
                # Ignore errors on best-effort cleanup
                pass