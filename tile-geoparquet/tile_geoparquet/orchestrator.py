from __future__ import annotations
from pathlib import Path
from typing import Optional, Set, Dict, List
import logging
import pyarrow as pa
import shutil
from concurrent.futures import ThreadPoolExecutor

from .datasource import DataSource
from .assigner import TileAssignerFromCSV
from .writer_pool import WriterPool
from .utils_large import ensure_large_types

logger = logging.getLogger(__name__)
_TILE_COL = "geo_parquet_tile_num"


# ---------------------------------------------------------------------------
# Overflow Writer → Arrow Stream
# ---------------------------------------------------------------------------
class _OverflowWriter:
    def __init__(self, out_dir: Path, compression: str, geom_col: str, max_rows_per_file: int = 50000):
        self.out_dir = Path(out_dir)
        self.compression = compression
        self.geom_col = geom_col
        self.max_rows_per_file = max_rows_per_file

        self._current_file = None
        self._writer: Optional[pa.ipc.RecordBatchStreamWriter] = None
        self._rows_in_file = 0
        self._total_rows = 0
        self._file_index = 0

        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _start_new_file(self, schema: pa.Schema):
        if self._writer:
            self._writer.close()
            self._current_file.close()
        file_path = self.out_dir / f"overflow_{self._file_index:03d}.arrowstream"
        self._file_index += 1
        self._current_file = open(file_path, "wb")
        self._writer = pa.ipc.new_stream(self._current_file, schema)
        self._rows_in_file = 0
        logger.debug(f"Started new overflow file: {file_path}")

    def write_batch(self, tbl: pa.Table, tile_id: Optional[str] = None) -> None:
        if tbl is None or tbl.num_rows == 0:
            return

        tbl = tbl.combine_chunks()
        tbl = ensure_large_types(tbl, geom_col=self.geom_col)

        if tile_id is not None:
            try:
                tid_num = int(tile_id.split("_")[-1])
            except Exception:
                tid_num = -1
            if _TILE_COL not in tbl.column_names:
                col = pa.array([tid_num] * tbl.num_rows, type=pa.int32())
                tbl = tbl.append_column(_TILE_COL, col)

        if self._writer is None or self._rows_in_file + tbl.num_rows > self.max_rows_per_file:
            self._start_new_file(tbl.schema)

        self._writer.write_table(tbl)
        self._rows_in_file += tbl.num_rows
        self._total_rows += tbl.num_rows

    def close(self) -> int:
        if self._writer:
            self._writer.close()
        if self._current_file:
            self._current_file.close()
        logger.info(f"Closed overflow writer: {self._file_index} files, {self._total_rows} total rows")
        return self._total_rows


# ---------------------------------------------------------------------------
# Round Orchestrator
# ---------------------------------------------------------------------------
class RoundOrchestrator:
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
        keep_overflows: bool = False,
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
        self.keep_overflows = keep_overflows

    # ------------------------------------------------------------------
    def _group_by_tile_column(self, batch: pa.Table) -> Dict[str, pa.Table]:
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

        out: Dict[str, pa.Table] = {}
        for tid, idxs in groups.items():
            out[f"tile_{tid:06d}"] = batch.take(pa.array(idxs, type=pa.int32()))
        return out

    # ------------------------------------------------------------------
    def _read_arrow_overflows(self, overflow_dir: Path, pool: WriterPool):
        files = sorted(overflow_dir.glob("overflow_*.arrowstream"))
        if not files:
            logger.info(f"No overflow files in {overflow_dir}")
            return

        logger.info(f"Reading {len(files)} overflow files in parallel from {overflow_dir}")

        def process_file(path: Path):
            with open(path, "rb") as f:
                reader = pa.ipc.open_stream(f)
                for batch in reader:
                    tbl = pa.Table.from_batches([batch])
                    # reassign leftover rows using intersection mode
                    parts = self.assigner.partition_by_tile(tbl, contains_only=False)
                    for tile_id, sub in parts.items():
                        pool.append(tile_id, sub)
            logger.debug(f"Finished {path}")

        with ThreadPoolExecutor(max_workers=min(8, len(files))) as ex:
            list(ex.map(process_file, files))

        if not self.keep_overflows:
            shutil.rmtree(overflow_dir, ignore_errors=True)
            logger.info(f"Deleted processed overflow directory {overflow_dir}")

    # ------------------------------------------------------------------
    def _run_one_round(self, ds: DataSource, round_id: int) -> Optional[Path]:
        logger.info(f"Starting round {round_id}")
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

        overflow_dir = Path(self.outdir) / "overflows" / f"round_{round_id}"
        overflow_dir.mkdir(parents=True, exist_ok=True)
        ow = _OverflowWriter(
            out_dir=overflow_dir,
            compression=self.compression,
            geom_col=self.geom_col,
        )

        open_tiles: Set[str] = set()
        cap = max(1, self.max_parallel_files - 1)

        for batch_idx, batch in enumerate(ds.iter_tables()):
            # First round = strict contains-only; next = intersection mode
            contains_only = (round_id == 0)
            parts = self.assigner.partition_by_tile(batch, contains_only=contains_only)
            logger.debug(f"Round {round_id}: batch {batch_idx} → assigned {len(parts)} tiles")

            for tile_id, sub in parts.items():
                if tile_id in open_tiles or len(open_tiles) < cap:
                    open_tiles.add(tile_id)
                    pool.append(tile_id, sub)
                else:
                    ow.write_batch(sub, tile_id=tile_id)

        pool.flush_all()
        overflow_rows = ow.close()
        if overflow_rows > 0:
            logger.info(f"Round {round_id}: overflow → {overflow_dir} ({overflow_rows} rows)")
            return overflow_dir

        if not any(overflow_dir.glob("*.arrowstream")):
            shutil.rmtree(overflow_dir, ignore_errors=True)
        return None

    # ------------------------------------------------------------------
    def run(self) -> None:
        round_id = 0
        ds: DataSource = self.source

        while True:
            overflow_dir = self._run_one_round(ds, round_id)
            if overflow_dir is None:
                break

            logger.info(f"Round {round_id} produced overflows → starting next round")

            pool = WriterPool(
                outdir=self.outdir,
                compression=self.compression,
                geom_col=self.geom_col,
                max_parallel_files=self.max_parallel_files,
                sort_mode=self.sort_mode,
                sort_keys=self.sort_keys,
                sfc_bits=self.sfc_bits,
            )
            self._read_arrow_overflows(overflow_dir, pool)
            pool.flush_all()
            round_id += 1

            if round_id > 10:
                logger.warning("Stopping after 10 rounds – likely degenerate geometries.")
                break

        logger.info("All rounds completed successfully.")