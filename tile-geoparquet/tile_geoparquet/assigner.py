from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Iterable
import logging
import numpy as np
import pandas as pd
import pyarrow as pa
from time import perf_counter

from shapely import from_wkb
from .RSGrove import RSGrovePartitioner, BeastOptions, EnvelopeNDLite
from .utils_large import ensure_large_types

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Legacy CSV assigner
# ---------------------------------------------------------------------------

class TileAssignerFromCSV:
    def __init__(self, index_csv_path: str, geom_col: str = "geometry"):
        import pandas as _pd
        df = _pd.read_csv(index_csv_path)
        required = {"id", "minx", "miny", "maxx", "maxy"}
        if not required.issubset(set(df.columns)):
            missing = required - set(df.columns)
            raise ValueError(f"Index CSV missing columns: {missing}")

        self.geom_col = geom_col
        self._bboxes = {
            str(r.id): (float(r.minx), float(r.miny), float(r.maxx), float(r.maxy))
            for r in df[["id", "minx", "miny", "maxx", "maxy"]].itertuples(index=False)
        }
        self._areas = {
            tid: (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            for tid, bbox in self._bboxes.items()
        }
        logger.info("TileAssignerFromCSV loaded %d tiles from %s", len(self._bboxes), index_csv_path)

    def tile_bbox(self, tile_id: str) -> Optional[Tuple[float, float, float, float]]:
        return self._bboxes.get(tile_id)

    def partition_by_tile(self, tbl: pa.Table) -> Dict[str, pa.Table]:
        if tbl.num_rows == 0:
            return {}
        if self.geom_col not in tbl.column_names:
            raise ValueError(f"Missing geometry column '{self.geom_col}'")

        t = tbl.combine_chunks()
        t = ensure_large_types(t, self.geom_col)
        geoms = from_wkb(t[self.geom_col].to_numpy(zero_copy_only=False))

        index_by_tile: Dict[str, List[int]] = {}
        for i, g in enumerate(geoms):
            if g is None or g.is_empty:
                continue
            gxmin, gymin, gxmax, gymax = g.bounds
            chosen = None
            chosen_area = float("inf")
            # legacy CSV mode stays "intersects"
            for tid, (xmin, ymin, xmax, ymax) in self._bboxes.items():
                if (gxmax >= xmin and gxmin <= xmax and gymax >= ymin and gymin <= ymax):
                    area = self._areas[tid]
                    if area < chosen_area:
                        chosen_area = area
                        chosen = tid
            if chosen is not None:
                index_by_tile.setdefault(chosen, []).append(i)

        out: Dict[str, pa.Table] = {}
        for tid, idxs in index_by_tile.items():
            out[tid] = t.take(pa.array(idxs, type=pa.int32()))
        return out


# ---------------------------------------------------------------------------
# RSGrove-based assigner (streaming sampling)
#   - Writes partition MBRs to rsgrove_partitions_debug.csv for verification
#   - CONTAINS-ONLY routing (inclusive eps): rows not fully contained are skipped
# ---------------------------------------------------------------------------

class RSGroveAssigner:
    def __init__(
        self,
        partitioner: RSGrovePartitioner,
        global_envelope: EnvelopeNDLite,
        geom_col: str = "geometry",
        boxes: Optional[List[Tuple[int, float, float, float, float]]] = None,
    ) -> None:
        self._part = partitioner
        self._env = global_envelope
        self._geom_col = geom_col
        self._boxes = boxes or []  # list of (pid, minx, miny, maxx, maxy)
        self._areas = {pid: (xmax - xmin) * (ymax - ymin) for pid, xmin, ymin, xmax, ymax in self._boxes}
        self._smallest_area_pid = min(self._areas, key=self._areas.get) if self._areas else None
        # Pre-sort partitions by xmin for plane-sweep filtering.
        self._boxes_by_xmin = sorted(self._boxes, key=lambda b: b[1])
        logger.info("RSGroveAssigner ready with %d partitions", self._part.numPartitions())

    @property
    def geom_col(self) -> str:
        return self._geom_col

    @classmethod
    def from_source(
        cls,
        tables: Iterable[pa.Table],
        num_partitions: int,
        geom_col: str = "geometry",
        seed: int = 42,
        options: Optional[BeastOptions] = None,
        sample_ratio: float = 1.0,
        sample_cap: Optional[int] = None,
    ) -> "RSGroveAssigner":
        """
        Build an RSGrovePartitioner from a streaming source with centroid sampling.
        """
        options = options or BeastOptions()
        # Ensure boxes don't expand to infinity: prevents overlapping tiles at domain edges
        options[RSGrovePartitioner.ExpandToInfinity] = False

        rng = np.random.default_rng(seed)
        mins = np.array([+np.inf, +np.inf], dtype=np.float64)
        maxs = np.array([-np.inf, -np.inf], dtype=np.float64)

        res_k = int(sample_cap) if sample_cap is not None else None
        X_s: List[float] = []
        Y_s: List[float] = []

        def reservoir_add(n_seen_local: int, x: float, y: float):
            if res_k is None:
                if rng.random() < sample_ratio:
                    X_s.append(x); Y_s.append(y)
                return
            if n_seen_local <= res_k:
                if len(X_s) < res_k:
                    X_s.append(x); Y_s.append(y)
                else:
                    j = rng.integers(0, n_seen_local)
                    if j < res_k:
                        X_s[j] = x; Y_s[j] = y
            else:
                j = rng.integers(0, n_seen_local)
                if j < res_k:
                    X_s[j] = x; Y_s[j] = y

        n_seen = 0
        n_batches = 0

        logger.info(
            "RSGroveAssigner.from_source: num_partitions=%d seed=%d sample_ratio=%.6f sample_cap=%s geom_col=%s",
            num_partitions, seed, sample_ratio, str(sample_cap), geom_col
        )

        for tb in tables:
            n_batches += 1
            t = tb.combine_chunks()
            if geom_col not in t.column_names or t.num_rows == 0:
                continue

            # upgrade to large_* early to avoid overflow in later takes/concats
            t = ensure_large_types(t, geom_col)
            geoms = from_wkb(t[geom_col].to_numpy(zero_copy_only=False))
            batch_start = n_seen

            for g in geoms:
                if g is None or g.is_empty:
                    continue
                minx, miny, maxx, maxy = g.bounds
                if minx < mins[0]: mins[0] = minx
                if miny < mins[1]: mins[1] = miny
                if maxx > maxs[0]: maxs[0] = maxx
                if maxy > maxs[1]: maxs[1] = maxy

                c = g.centroid
                reservoir_add(n_seen + 1, float(c.x), float(c.y))
                n_seen += 1

        if not X_s:
            raise ValueError("No geometries sampled to build RSGrove index. "
                             "Increase --sample-ratio or provide --sample-cap.")
        logger.info("Sampling complete: total_seen=%d, total_sampled=%d, batches=%d",
                    n_seen, len(X_s), n_batches)

        sample_points = np.stack(
            [np.asarray(X_s, dtype=np.float64), np.asarray(Y_s, dtype=np.float64)],
            axis=0
        )

        class _Summary2D:
            def __init__(self, mins, maxs):
                self._mins = np.asarray(mins, dtype=float)
                self._maxs = np.asarray(maxs, dtype=float)
            def getCoordinateDimension(self): return 2
            def getMinCoord(self, d): return float(self._mins[d])
            def getMaxCoord(self, d): return float(self._maxs[d])

        summary = _Summary2D(mins, maxs)

        part = RSGrovePartitioner()
        part.setup(options, True)  # disjoint
        part.construct(summary, sample_points, None, int(num_partitions))

        P = part.numPartitions()
        boxes: List[Tuple[int, float, float, float, float]] = []
        tmp = EnvelopeNDLite(np.zeros(2), np.zeros(2))
        for pid in range(P):
            part.getPartitionMBR(pid, tmp)
            boxes.append((pid, float(tmp.mins[0]), float(tmp.mins[1]), float(tmp.maxs[0]), float(tmp.maxs[1])))

        df = pd.DataFrame(boxes, columns=["pid", "minx", "miny", "maxx", "maxy"])
        debug_path = "rsgrove_partitions_debug.csv"
        try:
            df.to_csv(debug_path, index=False)
            logger.info("Wrote RSGrove partition MBRs to %s", debug_path)
        except Exception as e:
            logger.warning("Failed to write partition debug CSV: %s", e)

        env = EnvelopeNDLite(mins.copy(), maxs.copy())
        logger.info("Partitioner built: partitions=%d", part.numPartitions())
        return cls(part, env, geom_col=geom_col, boxes=boxes)

    def tile_bbox(self, tile_id: str) -> Optional[Tuple[float, float, float, float]]:
        try:
            pid = int(tile_id.split("_")[-1]) if tile_id.startswith("tile_") else int(tile_id)
        except Exception:
            return None
        env = EnvelopeNDLite(np.zeros(2), np.zeros(2))
        self._part.getPartitionMBR(pid, env)
        return (float(env.mins[0]), float(env.mins[1]), float(env.maxs[0]), float(env.maxs[1]))

    @staticmethod
    def _contains_inclusive(bbox: Tuple[float, float, float, float],
                            gminx: float, gminy: float, gmaxx: float, gmaxy: float,
                            eps: float = 1e-9) -> bool:
        xmin, ymin, xmax, ymax = bbox
        return (gminx >= xmin - eps) and (gminy >= ymin - eps) and (gmaxx <= xmax + eps) and (gmaxy <= ymax + eps)
    
    @staticmethod
    def _intersects(bbox: Tuple[float, float, float, float],
                    gminx: float, gminy: float, gmaxx: float, gmaxy: float) -> bool:
        xmin, ymin, xmax, ymax = bbox
        return not (gmaxx < xmin or gminx > xmax or gmaxy < ymin or gminy > ymax)

    @staticmethod
    def _expansion_area(bbox: Tuple[float, float, float, float],
                        gminx: float, gminy: float, gmaxx: float, gmaxy: float) -> float:
        xmin, ymin, xmax, ymax = bbox
        new_xmin = min(xmin, gminx)
        new_ymin = min(ymin, gminy)
        new_xmax = max(xmax, gmaxx)
        new_ymax = max(ymax, gmaxy)
        new_area = (new_xmax - new_xmin) * (new_ymax - new_ymin)
        old_area = (xmax - xmin) * (ymax - ymin)
        return new_area - old_area


    def partition_by_tile(self, tbl: pa.Table) -> pa.Table:
        """
        Center-first assignment with fallback expansion minimization:
          - Assign a row to the first partition whose MBR contains the row's centroid.
          - If none contains the centroid, assign to the partition that requires the
            least MBR expansion to encompass the centroid (point-based).
          - Returns a table of partition IDs aligned with the input order.
        """
        logger.info("[ASSIGNER] After ensure_large_types metadata: %s", tbl.schema.metadata)
        start_time = perf_counter()

        if tbl.num_rows == 0:
            return pa.table({"partition_id": pa.array([], type=pa.int32())})
        if self._geom_col not in tbl.column_names:
            raise ValueError(f"Missing geometry column '{self._geom_col}'")

        t = tbl.combine_chunks()
        t = ensure_large_types(t, self._geom_col)
        logger.info("[ASSIGNER] After ensure_large_types metadata: %s", t.schema.metadata)

        geoms = from_wkb(t[self._geom_col].to_numpy(zero_copy_only=False))

        # Pre-compute centroids and sort by x for plane sweep.
        geom_info: List[Tuple[float, float, int]] = []  # (cx, cy, idx)
        partition_ids: List[int] = [-1] * t.num_rows
        for i, g in enumerate(geoms):
            # Degenerate/empty geometries fall back to the smallest partition by area.
            if g is None or g.is_empty:
                fallback_pid = self._smallest_area_pid
                if fallback_pid is None:
                    raise ValueError(f"No partition found for geometry at index {i}")
                partition_ids[i] = int(fallback_pid)
                continue

            cx, cy = g.centroid.x, g.centroid.y
            geom_info.append((cx, cy, i))

        geom_info.sort(key=lambda x: x[0])  # sort by centroid x

        start_idx = 0
        end_idx = 0
        n_parts = len(self._boxes_by_xmin)

        if n_parts == 0:
            raise ValueError("No partitions available for assignment")

        for cx, cy, i in geom_info:
            # Expand active window: include partitions whose xmin is now in range (xmin <= cx).
            while end_idx < n_parts and self._boxes_by_xmin[end_idx][1] <= cx:
                end_idx += 1

            # Shrink active window: drop partitions whose xmax is left of the centroid (xmax < cx).
            while start_idx < end_idx and self._boxes_by_xmin[start_idx][3] < cx:
                start_idx += 1

            chosen_pid: int = -1

            # First pass: centroid containment among active partitions.
            for idx in range(start_idx, end_idx):
                pid, xmin, ymin, xmax, ymax = self._boxes_by_xmin[idx]
                if self._contains_inclusive((xmin, ymin, xmax, ymax), cx, cy, cx, cy):
                    chosen_pid = pid
                    break

            # Fallback: minimal expansion among active partitions; if none, fall back to smallest partition.
            if chosen_pid == -1:
                if start_idx < end_idx:
                    candidate_range = range(start_idx, end_idx)
                else:
                    candidate_range = range(n_parts)
                chosen_expansion = float("inf")
                for idx in candidate_range:
                    pid, xmin, ymin, xmax, ymax = self._boxes_by_xmin[idx]
                    expansion = self._expansion_area((xmin, ymin, xmax, ymax), cx, cy, cx, cy)
                    if expansion < chosen_expansion:
                        chosen_expansion = expansion
                        chosen_pid = pid

            if chosen_pid == -1:
                chosen_pid = int(self._smallest_area_pid) if self._smallest_area_pid is not None else -1
            assert chosen_pid != -1, f"No partition found for geometry at index {i}"
            partition_ids[i] = int(chosen_pid)

        if any(pid == -1 for pid in partition_ids):
            raise ValueError("Failed to assign partitions for all rows")

        out = pa.table({"partition_id": pa.array(partition_ids, type=pa.int32())})
        end_time = perf_counter()

        logger.info("partition_by_tile (contains-only): input_rows=%d, tiles=%d, finished in %.3f seconds, with a rate of %.3f rows/second",
                    t.num_rows, len(out), end_time - start_time, t.num_rows / (end_time - start_time) if end_time != start_time else 0)
        return out
