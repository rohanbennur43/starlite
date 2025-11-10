#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, logging, time, json
from contextlib import contextmanager
from queue import Queue
from threading import Thread
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import psutil

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from shapely import from_wkb, wkb
from shapely.geometry import Point, shape as shp_shape, mapping as shp_mapping

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("tile-writer")

# ---------------- Memory instrumentation ----------------
ARROW_POOL = pa.default_memory_pool()
def arrow_bytes():
    return ARROW_POOL.bytes_allocated()
def rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024*1024)
@contextmanager
def mem_scope(tag: str, extra: dict | None = None):
    b0, r0, t0 = arrow_bytes(), rss_mb(), time.time()
    yield
    payload = {
        "tag": tag,
        "arrow_MB": (arrow_bytes() - b0)/1e6,
        "rss_MB": (rss_mb() - r0),
        "sec": time.time() - t0,
    }
    if extra: payload.update(extra)
    logger.info("MEM " + ",".join(f"{k}={payload[k]}" for k in payload))

# ---------------- Tile helpers ----------------
def load_index(index_csv: str):
    df = pd.read_csv(index_csv)
    required = ["ID","File Name","xmin","ymin","xmax","ymax"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Index CSV missing columns: {missing}")
    mins = df[["xmin","ymin"]].to_numpy(float)
    maxs = df[["xmax","ymax"]].to_numpy(float)
    names = df["File Name"].astype(str).tolist()
    ids = df["ID"].tolist()
    return ids, names, mins, maxs, df

def point_in_rect(x, y, mins, maxs):
    ge_minx = x >= mins[:,0]
    lt_maxx = x <  maxs[:,0]
    ge_miny = y >= mins[:,1]
    lt_maxy = y <  maxs[:,1]
    hit = ge_minx & lt_maxx & ge_miny & lt_maxy
    idx = np.flatnonzero(hit)
    return int(idx[0]) if idx.size else -1

# ---------------- GeoJSON streaming ----------------
def geojson_batch_iterator(path: str, batch_size: int, properties_schema: List[str] | None = None):
    import fiona
    logger.info(f"Streaming GeoJSON from {path} ...")
    with fiona.open(path, "r") as src:
        try:
            logger.info(f"GeoJSON feature count reported: {len(src):,}")
        except Exception:
            logger.info("GeoJSON feature count unknown (will stream).")

        cols: List[str] = list(properties_schema) if properties_schema else []
        data: Dict[str, List] = {k: [] for k in cols}
        geom_wkb: List[bytes] = []
        xs: List[float] = []; ys: List[float] = []
        bxs: List[float] = []; bys: List[float] = []
        bxe: List[float] = []; bye: List[float] = []

        def flush_batch():
            if not geom_wkb:
                return None
            df_dict = {k: v for k, v in data.items()}
            df_dict["geometry"] = geom_wkb
            df = pd.DataFrame(df_dict)
            table = pa.Table.from_pandas(df, preserve_index=False)
            arr_xs = np.asarray(xs, dtype=float)
            arr_ys = np.asarray(ys, dtype=float)
            arr_bxs = np.asarray(bxs, dtype=float)
            arr_bys = np.asarray(bys, dtype=float)
            arr_bxe = np.asarray(bxe, dtype=float)
            arr_bye = np.asarray(bye, dtype=float)
            for k in data.keys(): data[k].clear()
            geom_wkb.clear()
            xs.clear(); ys.clear()
            bxs.clear(); bys.clear(); bxe.clear(); bye.clear()
            return table, arr_xs, arr_ys, arr_bxs, arr_bys, arr_bxe, arr_bye

        for feat in src:
            props = feat.get("properties") or {}
            geom_json = feat.get("geometry")
            if not geom_json:
                continue
            g = shp_shape(geom_json)
            if g.is_empty:
                continue

            for k in props.keys():
                if k not in data:
                    cols.append(k)
                    data[k] = [None] * (len(geom_wkb))
            for k in cols:
                data[k].append(props.get(k, None))

            geom_wkb.append(wkb.dumps(g))
            cx, cy = (g.x, g.y) if isinstance(g, Point) else (g.centroid.x, g.centroid.y)
            xs.append(cx); ys.append(cy)
            bx0, by0, bx1, by1 = g.bounds
            bxs.append(bx0); bys.append(by0); bxe.append(bx1); bye.append(by1)

            if len(geom_wkb) >= batch_size:
                out = flush_batch()
                if out is not None:
                    yield out
        out = flush_batch()
        if out is not None:
            yield out

# ---------------- Parquet writer factory ----------------
def make_parquet_writer_factory(args, out_path_for):
    def make_pq_writer(path: str, schema: pa.Schema):
        if getattr(pq, "WriterProperties", None) and hasattr(pq.WriterProperties, "builder"):
            wpb = pq.WriterProperties.builder()
            if args.compression != "none":
                wpb = wpb.compression(args.compression)
            wpb = wpb.write_statistics(True)
            if args.data_page_size:
                wpb = wpb.data_page_size(args.data_page_size)
            writer_props = wpb.build()
            return pq.ParquetWriter(path, schema=schema, version="2.6", writer_properties=writer_props)
        return pq.ParquetWriter(
            path, schema=schema, version="2.6",
            compression=(None if args.compression == "none" else args.compression),
            data_page_size=args.data_page_size,
            write_statistics=True
        )
    return make_pq_writer

# ---------------- Parallel writer worker ----------------
class WriterWorker(Thread):
    def __init__(self, wid: int, tasks: Queue, schema: pa.Schema,
                 out_path_for, make_parquet_writer,
                 row_group_size: int | None, tile_owner, out_format: str):
        super().__init__(daemon=True)
        self.wid = wid
        self.tasks = tasks
        self.schema = schema
        self.out_path_for = out_path_for
        self.make_parquet_writer = make_parquet_writer
        self.row_group_size = row_group_size
        self.tile_owner = tile_owner
        self.out_format = out_format
        self._pq_writers: Dict[int, pq.ParquetWriter] = {}
        self._gj_handles: Dict[int, Tuple[object, bool]] = {}

    def ensure_pq_writer(self, tile_id: int) -> pq.ParquetWriter:
        if self.tile_owner(tile_id) != self.wid:
            raise RuntimeError(f"Tile {tile_id} not owned by worker {self.wid}")
        w = self._pq_writers.get(tile_id)
        if w is None:
            path = self.out_path_for(tile_id)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            w = self.make_parquet_writer(path, self.schema)
            self._pq_writers[tile_id] = w
        return w

    def run(self):
        while True:
            item = self.tasks.get()
            if item is None:
                break
            tile_id, table, rows = item
            with mem_scope("write", {"worker": self.wid, "tile": tile_id, "rows": rows, "fmt": self.out_format}):
                w = self.ensure_pq_writer(tile_id)
                w.write_table(table, row_group_size=self.row_group_size)
            self.tasks.task_done()

    def close_all(self):
        for tile_id, w in self._pq_writers.items():
            try: w.close()
            except Exception as e:
                logger.warning(f"Close failed for tile {tile_id}: {e}")

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Split GeoParquet/GeoJSON into per-tile files (streaming, parallel, instrumented).")
    ap.add_argument("input", help="Path to GeoParquet (dataset/file) or GeoJSON file")
    ap.add_argument("index_csv", help="tiles_index.csv produced by tiler")
    ap.add_argument("out_dir", help="Output directory for per-tile files")

    ap.add_argument("--input-type", choices=["auto","parquet","geojson"], default="auto")
    ap.add_argument("--geometry", default="geometry")
    ap.add_argument("--mode", choices=["disjoint","covering"], default="disjoint")
    ap.add_argument("--batch-size", type=int, default=100_000)
    ap.add_argument("--format", choices=["parquet","geojson"], default="parquet")
    ap.add_argument("--suffix", default=None)
    ap.add_argument("--use-index-filenames", action="store_true")
    ap.add_argument("--row-group-size", type=int, default=None)
    ap.add_argument("--data-page-size", type=int, default=None)
    ap.add_argument("--compression", default="zstd", choices=["zstd","snappy","gzip","brotli","lz4","none"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--queue-size", type=int, default=8)
    ap.add_argument("--progress-interval", type=int, default=10)
    ap.add_argument("--read-only", action="store_true",
                    help="Do not write any output; only read/scan and log memory/throughput")

    args = ap.parse_args()

    in_lower = args.input.lower()
    if args.input_type == "auto":
        if in_lower.endswith(".parquet") or os.path.isdir(args.input):
            input_type = "parquet"
        elif in_lower.endswith(".geojson") or in_lower.endswith(".json"):
            input_type = "geojson"
        else:
            raise SystemExit("Cannot infer input type; use --input-type.")
    else:
        input_type = args.input_type

    if args.suffix is None:
        args.suffix = ".parquet" if args.format == "parquet" else ".geojson"

    logger.info(f"Input type: {input_type.upper()}; batch_size={args.batch_size}; mode={args.mode}; out format={args.format}")

    tile_ids, tile_names, mins, maxs, _ = load_index(args.index_csv)
    P = len(tile_ids)
    logger.info(f"Loaded {P} tiles from index")

    def out_path_for(i: int) -> str:
        if args.use_index_filenames:
            base = os.path.splitext(tile_names[i])[0]
            return os.path.join(args.out_dir, base + args.suffix)
        else:
            return os.path.join(args.out_dir, f"part-{tile_ids[i]:05d}{args.suffix}")

    if not args.read_only:
        os.makedirs(args.out_dir, exist_ok=True)

    make_parquet_writer = None if args.read_only else make_parquet_writer_factory(args, out_path_for)

    W = max(1, args.workers)
    def owner_of_tile(tile_id: int) -> int:
        return tile_id % W

    counts = np.zeros(P, dtype=np.int64)
    total = 0
    assigned = 0

    # ---------- PARQUET INPUT ----------
    if input_type == "parquet":
        dataset = ds.dataset(args.input, format="parquet")
        schema: pa.Schema = dataset.schema
        if args.geometry not in schema.names:
            raise SystemExit(f"Geometry column '{args.geometry}' not in dataset. Columns: {schema.names}")

        queues = None
        workers = None
        if not args.read_only:
            queues = [Queue(maxsize=args.queue_size) for _ in range(W)]
            workers = [WriterWorker(k, queues[k], schema, out_path_for, make_parquet_writer,
                                    args.row_group_size, owner_of_tile, args.format)
                       for k in range(W)]
            logger.info(f"Starting {W} writer workers")
            for w in workers: w.start()

        scanner = dataset.scanner(batch_size=args.batch_size)
        logger.info("Scanning Parquet ...")

        for b_idx, batch in enumerate(scanner.to_batches(), start=1):
            n = batch.num_rows
            with mem_scope("batch", {"batch": b_idx, "rows": n}):
                if n == 0:
                    continue
                total += n

                wkb_arr: pa.Array = batch.column(args.geometry)
                wkb_np = wkb_arr.to_numpy(zero_copy_only=False)
                geoms = from_wkb(wkb_np)
                valid_mask = np.array([g is not None and not g.is_empty for g in geoms], dtype=bool)
                if not np.any(valid_mask):
                    continue

                geoms = geoms[valid_mask]
                table = pa.Table.from_batches([batch.filter(pa.array(valid_mask))])

                if args.mode == "disjoint":
                    xs = np.array([g.x if isinstance(g, Point) else g.centroid.x for g in geoms], dtype=float)
                    ys = np.array([g.y if isinstance(g, Point) else g.centroid.y for g in geoms], dtype=float)
                    tile_idx = np.full(xs.shape[0], -1, dtype=int)
                    for j in range(xs.shape[0]):
                        tile_idx[j] = point_in_rect(xs[j], ys[j], mins, maxs)
                    for i in range(P):
                        sel = tile_idx == i
                        if not np.any(sel):
                            continue
                        c = int(sel.sum()); counts[i] += c; assigned += c
                        if not args.read_only:
                            taken = table.filter(pa.array(sel.tolist()))
                            queues[owner_of_tile(i)].put((i, taken, c))
                else:
                    bxs = np.array([g.bounds[0] for g in geoms], dtype=float)
                    bys = np.array([g.bounds[1] for g in geoms], dtype=float)
                    bxe = np.array([g.bounds[2] for g in geoms], dtype=float)
                    bye = np.array([g.bounds[3] for g in geoms], dtype=float)
                    for i in range(P):
                        sep = (maxs[i,0] <= bxs) | (bxe <= mins[i,0]) | (maxs[i,1] <= bys) | (bye <= mins[i,1])
                        sel = ~sep
                        if not np.any(sel):
                            continue
                        c = int(sel.sum()); counts[i] += c; assigned += c
                        if not args.read_only:
                            taken = table.filter(pa.array(sel.tolist()))
                            queues[owner_of_tile(i)].put((i, taken, c))

            if (b_idx % args.progress_interval) == 0:
                logger.info(f"Progress: batches={b_idx}, total_rows={total:,}, assigned_rows={assigned:,}, "
                            f"arrow_MB_now={arrow_bytes()/1e6:.1f}, rss_MB_now={rss_mb():.1f}")

        if not args.read_only and queues:
            logger.info("Draining writer queues ...")
            for q in queues: q.join()
            for q in queues: q.put(None)
            for w in workers: w.join()
            for w in workers: w.close_all()

    # ---------- GEOJSON INPUT ----------
    else:
        logger.info("Streaming GeoJSON ...")
        first = True
        queues = None
        workers = None
        b_idx = 0

        for (table, xs_arr, ys_arr, bxs, bys, bxe, bye) in geojson_batch_iterator(args.input, args.batch_size):
            b_idx += 1
            n = table.num_rows
            with mem_scope("batch", {"batch": b_idx, "rows": n}):
                if n == 0:
                    continue
                total += n

                if first:
                    schema = table.schema
                    if not args.read_only:
                        queues = [Queue(maxsize=args.queue_size) for _ in range(W)]
                        workers = [WriterWorker(k, queues[k], schema, out_path_for, make_parquet_writer,
                                                args.row_group_size, owner_of_tile, args.format)
                                   for k in range(W)]
                        logger.info(f"Starting {W} writer workers (schema discovered from GeoJSON batch 1)")
                        for w in workers: w.start()
                    first = False

                if args.mode == "disjoint":
                    tile_idx = np.full(xs_arr.shape[0], -1, dtype=int)
                    for j in range(xs_arr.shape[0]):
                        tile_idx[j] = point_in_rect(xs_arr[j], ys_arr[j], mins, maxs)
                    for i in range(P):
                        sel = tile_idx == i
                        if not np.any(sel):
                            continue
                        c = int(sel.sum()); counts[i] += c; assigned += c
                        if not args.read_only:
                            taken = table.filter(pa.array(sel.tolist()))
                            queues[owner_of_tile(i)].put((i, taken, c))
                else:
                    for i in range(P):
                        sep = (maxs[i,0] <= bxs) | (bxe <= mins[i,0]) | (maxs[i,1] <= bys) | (bye <= mins[i,1])
                        sel = ~sep
                        if not np.any(sel):
                            continue
                        c = int(sel.sum()); counts[i] += c; assigned += c
                        if not args.read_only:
                            taken = table.filter(pa.array(sel.tolist()))
                            queues[owner_of_tile(i)].put((i, taken, c))

            if (b_idx % args.progress_interval) == 0:
                logger.info(f"Progress: batches={b_idx}, total_rows={total:,}, assigned_rows={assigned:,}, "
                            f"arrow_MB_now={arrow_bytes()/1e6:.1f}, rss_MB_now={rss_mb():.1f}")

        if not first and not args.read_only and queues:
            logger.info("Draining writer queues ...")
            for q in queues: q.join()
            for q in queues: q.put(None)
            for w in workers: w.join()
            for w in workers: w.close_all()

    # ---------- Summary ----------
    if not args.read_only:
        for i in range(P):
            path = out_path_for(i)
            if os.path.exists(path):
                try:
                    sz_mb = os.path.getsize(path) / (1024*1024)
                    logger.info(f"Tile {i} → {path} ({counts[i]} rows, {sz_mb:.1f} MB)")
                except Exception:
                    logger.info(f"Tile {i} → {path} ({counts[i]} rows)")
    else:
        for i in range(P):
            if counts[i]:
                logger.info(f"(read-only) Tile {i}: {counts[i]} rows (no files written)")

    logger.info(f"Done. Total rows seen: {total:,}, rows {'written' if not args.read_only else 'accounted'}: {assigned:,} "
                f"(mode={args.mode}, tiles={P}, workers={W}, format={args.format}, read_only={args.read_only})")

if __name__ == "__main__":
    main()
