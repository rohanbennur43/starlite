# tile_geoparquet/orchestrator.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Set, Dict, List
import logging

import pyarrow as pa
import pyarrow.parquet as pq
from time import perf_counter

from .datasource import DataSource, GeoParquetSource
from .assigner import TileAssignerFromCSV
from .writer_pool import WriterPool
from .utils_large import ensure_large_types

logger = logging.getLogger(__name__)

_TILE_COL = "geo_parquet_tile_num"


class _OverflowWriter:
    """
    Streams rows that don't fit in this round into a single overflow Parquet file.

    - Adds a persistent column 'geo_parquet_tile_num' to remember which tile each row belongs to.
    - Defers ParquetWriter opening until the first batch.
    - Ensures parent directories exist before opening.
    """
    def __init__(self, path: Path, compression: str, geom_col: str):
        self.path = Path(path)
        self.compression = compression
        self.geom_col = geom_col

        self._pw: Optional[pq.ParquetWriter] = None
        self._rows = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # def write_batch(self, tbl: pa.Table, tile_id: Optional[str] = None) -> None:
    #     if tbl is None or tbl.num_rows == 0:
    #         return

    #     tbl = tbl.combine_chunks()
    #     tbl = ensure_large_types(tbl, geom_col=self.geom_col)

    #     # Inject persistent tile column if available
    #     if tile_id is not None:
    #         try:
    #             tid_num = int(tile_id.split("_")[-1])
    #         except Exception:
    #             tid_num = -1
    #         if _TILE_COL not in tbl.column_names:
    #             col = pa.array([tid_num] * tbl.num_rows, type=pa.int32())
    #             tbl = tbl.append_column(_TILE_COL, col)

    #     if self._pw is None:
    #         logger.debug("Opening overflow writer at %s", self.path)
    #         self._pw = pq.ParquetWriter(str(self.path), schema=tbl.schema, compression=self.compression)

    #     self._pw.write_table(tbl)
    #     self._rows += tbl.num_rows

    def close(self) -> int:
        if self._pw is not None:
            self._pw.close()
        return self._rows

    def write_batch(self, tbl: pa.Table, tile_id: Optional[int] = None) -> None:
        if tbl is None or tbl.num_rows == 0:
            return

        tbl = tbl.combine_chunks()
        tbl = ensure_large_types(tbl, geom_col=self.geom_col)

        # Inject persistent tile column if available
        if tile_id is not None:
            try:
                tid_num = int(tile_id)
            except Exception:
                tid_num = -1
            if _TILE_COL not in tbl.column_names:
                col = pa.array([tid_num] * tbl.num_rows, type=pa.int32())
                tbl = tbl.append_column(_TILE_COL, col)
                logger.debug(
                    f"Added persistent tile column '{_TILE_COL}' for tile_id={tile_id} "
                    f"({tbl.num_rows} rows)"
                )
            else:
                logger.debug(
                    f"Column '{_TILE_COL}' already present in batch for tile_id={tile_id}"
                )

        if self._pw is None:
            logger.debug("Opening overflow writer at %s", self.path)
            self._pw = pq.ParquetWriter(str(self.path), schema=tbl.schema, compression=self.compression)

        self._pw.write_table(tbl)
        self._rows += tbl.num_rows


class RoundOrchestrator:
    """
    Round-based tiler with bounded parallel writers.
    Adds support for carrying forward per-row tile IDs across rounds using
    'geo_parquet_tile_num' column in overflow files.
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
        sort_keys: Optional[str] = None,
        sfc_bits: int = 16,
        records_per_round: int = 1000_000,
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
        self.src_schema: pa.Schema = source.schema()
        self.records_per_round = int(records_per_round)

    # ------------------------------------------------------------------

    @staticmethod
    def _tile_label(tile_id: int) -> str:
        return f"tile_{tile_id:06d}"

    def _group_by_tile_column(self, batch: pa.Table) -> Dict[int, pa.Table]:
        """Group a batch by the persistent tile column (geo_parquet_tile_num)."""
        col = batch[_TILE_COL]
        arr = col.to_numpy(zero_copy_only=False)
        groups: Dict[int, List[int]] = {}
        for i, v in enumerate(arr):
            if v is None:
                continue
            try:
                v_int = int(v)
            except Exception:
                continue
            groups.setdefault(v_int, []).append(i)

        out: Dict[int, pa.Table] = {}
        for tid, idxs in groups.items():
            out[tid] = batch.take(pa.array(idxs, type=pa.int32()))
        return out

    def _group_by_partition_ids(self, batch: pa.Table, partitions: pa.Table) -> Dict[int, pa.Table]:
        """Group a batch using the partition_id table returned by the assigner."""
        if partitions.num_rows != batch.num_rows:
            raise ValueError("Partition table must align with batch row count")
        if "partition_id" not in partitions.column_names:
            raise ValueError("Partition table missing 'partition_id' column")

        arr = partitions["partition_id"].to_numpy(zero_copy_only=False)
        groups: Dict[int, List[int]] = {}
        for i, v in enumerate(arr):
            if v is None:
                continue
            try:
                pid = int(v)
            except Exception:
                continue
            groups.setdefault(pid, []).append(i)

        out: Dict[int, pa.Table] = {}
        for pid, idxs in groups.items():
            out[pid] = batch.take(pa.array(idxs, type=pa.int32()))
        return out

    # ------------------------------------------------------------------

    def _run_one_round(
        self,
        ds: DataSource,
        round_id: int,
        records_per_round: Optional[int] = None,
    ) -> Optional[Path]:
        logger.info("Starting round %d", round_id)
        Path(self.outdir).mkdir(parents=True, exist_ok=True)

        pool = WriterPool(
            outdir=self.outdir,
            compression=self.compression,
            geom_col=self.geom_col,
            max_parallel_files=self.max_parallel_files,
            sort_mode=self.sort_mode,
            sort_keys=self.sort_keys,
            sfc_bits=self.sfc_bits,
        )

        overflow_path = Path(self.outdir) / f"_overflow_round_{round_id}.parquet"
        ow = _OverflowWriter(
            path=overflow_path,
            compression=self.compression,
            geom_col=self.geom_col,
        )

        open_tiles: Set[int] = set()
        cap = max(1, self.max_parallel_files - 1)
        batches: List[pa.Table] = []
        current_rows = 0
        records_limit = max(1, int(records_per_round or self.records_per_round))

        def process_accumulated(batch_id: int) -> None:
            nonlocal batches, current_rows
            if not batches:
                return

            combined = pa.concat_tables(batches, promote=True).combine_chunks()

            # Check for persistent tile ID column
            if _TILE_COL in combined.column_names:
                parts = self._group_by_tile_column(combined)
                logger.debug(
                    "Round %d: batches up to %d → reused cached tile IDs (%d tiles)",
                    round_id,
                    batch_id,
                    len(parts),
                )
            else:
                partition_table = self.assigner.partition_by_tile(combined)
                parts = self._group_by_partition_ids(combined, partition_table)
                logger.debug(
                    "Round %d: batches up to %d → assigned fresh (%d tiles)",
                    round_id,
                    batch_id,
                    len(parts),
                )

            for tile_id, sub in parts.items():
                if tile_id in open_tiles or len(open_tiles) < cap:
                    if tile_id not in open_tiles:
                        logger.debug("Round %d: admitting tile %s", round_id, self._tile_label(tile_id))
                        open_tiles.add(tile_id)
                    pool.append(tile_id, sub)
                else:
                    ow.write_batch(sub, tile_id=tile_id)

            batches = []
            current_rows = 0

        start_time = perf_counter()
        batch_idx = -1
        for batch_idx, batch in enumerate(ds.iter_tables()):
            batches.append(batch)
            current_rows += batch.num_rows
            if current_rows >= records_limit:
                process_accumulated(batch_idx)

        process_accumulated(batch_idx)
        end_time = perf_counter()
        logger.info(
            "Round %d: processed %d batches in %.2f seconds",
            round_id, batch_idx + 1, end_time - start_time)

        # Flush and handle overflow
        if open_tiles:
            logger.info(
                "Round %d: flushing %d tiles (cap=%d, admitted=%d)",
                round_id, len(open_tiles), cap, len(open_tiles)
            )
        else:
            logger.info("Round %d: no tiles admitted; only overflow will be produced (if any).", round_id)

        start_time = perf_counter()
        pool.flush_all()
        end_time = perf_counter()
        logger.info("Round %d: flushed all writers in %.2f seconds", round_id, end_time - start_time)

        overflow_rows = ow.close()
        if overflow_rows > 0:
            logger.info("Round %d: overflow file created at %s (rows=%d)",
                        round_id, overflow_path, overflow_rows)
            return overflow_path

        if overflow_path.exists():
            try:
                pf = pq.ParquetFile(str(overflow_path))
                if pf.metadata.num_rows == 0:
                    overflow_path.unlink(missing_ok=True)
            except Exception:
                pass
        return None

    # ------------------------------------------------------------------

    def run(self) -> None:
        round_id = 0
        ds: DataSource = self.source

        from time import perf_counter
        start_time = perf_counter()
        while True:
            overflow_path = self._run_one_round(ds, round_id, records_per_round=self.records_per_round)
            logger.info("Round %d finished in %.2f seconds", round_id, perf_counter() - start_time)
            if overflow_path is None:
                break

            logger.info(
                "Round %d produced overflow; continuing with overflow file %s",
                round_id, overflow_path
            )
            ds = GeoParquetSource(str(overflow_path))
            round_id += 1

        for p in Path(self.outdir).glob("_overflow_round_*.parquet"):
            try:
                pf = pq.ParquetFile(str(p))
                if pf.metadata.num_rows == 0:
                    p.unlink(missing_ok=True)
            except Exception:
                pass
