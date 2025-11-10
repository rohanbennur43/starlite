# from __future__ import annotations

# import json
# import logging
# import math
# import os
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Iterable, List, Optional, Tuple

# import numpy as np
# import pyarrow as pa
# import pyarrow.parquet as pq
# from shapely import from_wkb
# from shapely.geometry import (
#     Point,
#     LineString,
#     LinearRing,
#     Polygon,
#     MultiPoint,
#     MultiLineString,
#     MultiPolygon,
#     GeometryCollection,
# )
# from pyproj import Transformer

# logger = logging.getLogger(__name__)


# # ------------------------- Config -------------------------

# @dataclass
# class HistConfig:
#     grid_size: int = 4096          # base resolution N for NxN grid
#     levels: int = 10               # number of pyramid levels
#     out_crs: str = "EPSG:3857"     # Web Mercator
#     dtype: str = "float64"         # float64 for now (can switch to float32 later)
#     max_parallel_tiles: int = 8    # concurrent tiles
#     rg_parallel: int = 4           # concurrent row-group partials per *tile*
#     keep_partials: bool = False    # store per-RG partial .npy files
#     partials_dir: Optional[Path] = None  # where to store partials (default: outdir/partials)


# # ------------------------- Helpers -------------------------

# def _load_geo_metadata(schema: pa.Schema, geom_col: str) -> Tuple[Optional[dict], Optional[str]]:
#     meta = dict(schema.metadata or {})
#     raw = meta.get(b"geo")
#     if not raw:
#         return None, None
#     try:
#         j = json.loads(raw.decode("utf-8"))
#     except Exception:
#         return None, None
#     prim = j.get("primary_column") or geom_col
#     crs = j.get("columns", {}).get(prim, {}).get("crs")
#     return j, crs


# def _infer_source_crs(crs_from_meta: Optional[str]) -> str:
#     return crs_from_meta if crs_from_meta else "EPSG:4326"


# def _reproject_transformer(src_crs: str, dst_crs: str) -> Transformer:
#     return Transformer.from_crs(src_crs, dst_crs, always_xy=True)


# def _geometry_vertices_iter(g) -> Iterable[Tuple[float, float]]:
#     if g is None or g.is_empty:
#         return
#     if isinstance(g, Point):
#         yield (g.x, g.y)
#     elif isinstance(g, (LineString, LinearRing)):
#         for x, y in g.coords:
#             yield (x, y)
#     elif isinstance(g, Polygon):
#         for x, y in g.exterior.coords:
#             yield (x, y)
#         for ring in g.interiors:
#             for x, y in ring.coords:
#                 yield (x, y)
#     elif isinstance(g, (MultiPoint, MultiLineString, MultiPolygon, GeometryCollection)):
#         for sub in getattr(g, "geoms", []):
#             yield from _geometry_vertices_iter(sub)
#     else:
#         coords = getattr(g, "coords", None)
#         if coords is not None:
#             for x, y in coords:
#                 yield (x, y)


# def _downsample_2x2_sum(img: np.ndarray) -> np.ndarray:
#     h, w = img.shape
#     h2 = h // 2
#     w2 = w // 2
#     img = img[: 2 * h2, : 2 * w2]
#     return (img.reshape(h2, 2, w2, 2).sum(axis=(1, 3))).astype(img.dtype, copy=False)


# def _compute_tile_bbox_out_once(
#     pf: pq.ParquetFile,
#     geom_col: str,
#     transformer: Transformer,
#     out_crs: str,
# ) -> Tuple[float, float, float, float]:
#     # One linear pass to compute the bbox in *output* CRS
#     xmin = math.inf
#     ymin = math.inf
#     xmax = -math.inf
#     ymax = -math.inf

#     for rg in range(pf.metadata.num_row_groups):
#         t = pf.read_row_group(rg, columns=[geom_col]).combine_chunks()
#         geoms = from_wkb(t[geom_col].to_numpy(zero_copy_only=False))
#         for g in geoms:
#             if g is None or g.is_empty:
#                 continue
#             bxmin, bymin, bxmax, bymax = g.bounds
#             # project all 4 corners for tighter bounds (safer at extreme latitudes)
#             xs = [bxmin, bxmin, bxmax, bxmax]
#             ys = [bymin, bymax, bymin, bymax]
#             X, Y = transformer.transform(xs, ys)
#             gxmin = min(X)
#             gymin = min(Y)
#             gxmax = max(X)
#             gymax = max(Y)
#             if gxmin < xmin: xmin = gxmin
#             if gymin < ymin: ymin = gymin
#             if gxmax > xmax: xmax = gxmax
#             if gymax > ymax: ymax = gymax

#     # If we failed to compute finite bounds, fall back to a trivial bbox
#     if not (math.isfinite(xmin) and math.isfinite(ymin) and math.isfinite(xmax) and math.isfinite(ymax)):
#         xmin = ymin = 0.0
#         xmax = ymax = 1.0

#     # Optional clamp for Web Mercator
#     if isinstance(out_crs, str) and out_crs.upper() in ("EPSG:3857", "EPSG:900913"):
#         LIM = 20037508.342789244
#         xmin = max(xmin, -LIM); ymin = max(ymin, -LIM)
#         xmax = min(xmax,  LIM); ymax = min(ymax,  LIM)

#     logger.info("bbox_out(%s): xmin=%.3f ymin=%.3f xmax=%.3f ymax=%.3f", out_crs, xmin, ymin, xmax, ymax)
#     return (float(xmin), float(ymin), float(xmax), float(ymax))


# def _compute_tile_bbox_out_from_vertices(pf: pq.ParquetFile, geom_col: str, transformer: Transformer, out_crs: str) -> Tuple[float, float, float, float]:
#     xmin = math.inf; ymin = math.inf; xmax = -math.inf; ymax = -math.inf
#     for rg in range(pf.metadata.num_row_groups):
#         t = pf.read_row_group(rg, columns=[geom_col]).combine_chunks()
#         geoms = from_wkb(t[geom_col].to_numpy(zero_copy_only=False))
#         for g in geoms:
#             if g is None or g.is_empty:
#                 continue
#             xs = []; ys = []
#             for x, y in _geometry_vertices_iter(g):
#                 xs.append(x); ys.append(y)
#             if not xs:
#                 continue
#             X, Y = transformer.transform(xs, ys)
#             gxmin = float(np.min(X)); gymin = float(np.min(Y))
#             gxmax = float(np.max(X)); gymax = float(np.max(Y))
#             if gxmin < xmin: xmin = gxmin
#             if gymin < ymin: ymin = gymin
#             if gxmax > xmax: xmax = gxmax
#             if gymax > ymax: ymax = gymax
#     if not (math.isfinite(xmin) and math.isfinite(ymin) and math.isfinite(xmax) and math.isfinite(ymax)):
#         xmin = ymin = 0.0; xmax = ymax = 1.0
#     if isinstance(out_crs, str) and out_crs.upper() in ("EPSG:3857", "EPSG:900913"):
#         LIM = 20037508.342789244
#         xmin = max(xmin, -LIM); ymin = max(ymin, -LIM)
#         xmax = min(xmax,  LIM); ymax = min(ymax,  LIM)
#     logger.info("bbox_out(vertices %s): xmin=%.3f ymin=%.3f xmax=%.3f ymax=%.3f", out_crs, xmin, ymin, xmax, ymax)
#     return (float(xmin), float(ymin), float(xmax), float(ymax))


# def _accumulate_vertices_hist(
#     table: pa.Table,
#     geom_col: str,
#     bbox_out: Tuple[float, float, float, float],
#     transformer: Transformer,
#     n: int,
#     dtype: np.dtype
# ) -> np.ndarray:
#     hist = np.zeros((n, n), dtype=dtype)
#     geoms = from_wkb(table[geom_col].to_numpy(zero_copy_only=False))
#     minx, miny, maxx, maxy = bbox_out

#     for g in geoms:
#         if g is None or g.is_empty:
#             continue
#         xs: List[float] = []
#         ys: List[float] = []
#         for x, y in _geometry_vertices_iter(g):
#             xs.append(x); ys.append(y)
#         if not xs:
#             continue
#         X, Y = transformer.transform(xs, ys)
#         tx = (np.asarray(X) - minx) / (maxx - minx) if (maxx > minx) else np.zeros_like(X)
#         ty = (np.asarray(Y) - miny) / (maxy - miny) if (maxy > miny) else np.zeros_like(Y)
#         ix = np.floor(tx * n).astype(np.int64)
#         iy = np.floor(ty * n).astype(np.int64)
#         if ix.size == 0 or iy.size == 0:
#             continue
#         np.clip(ix, 0, n - 1, out=ix)
#         np.clip(iy, 0, n - 1, out=iy)
#         np.add.at(hist, (iy, ix), 1.0)
#     return hist


# def _write_hist_outputs(
#     base_hist: np.ndarray,
#     out_base: Path,
#     cfg: HistConfig,
#     tile_id: str,
#     bbox_out: Tuple[float, float, float, float],
#     geom_col: str
# ) -> None:
#     out_base.parent.mkdir(parents=True, exist_ok=True)

#     # L0
#     np.save(out_base.with_suffix(".npy"), base_hist)
#     meta0 = {
#         "tile_id": tile_id,
#         "level": 0,
#         "grid_size": int(cfg.grid_size),
#         "dtype": cfg.dtype,
#         "bbox_out_epsg": cfg.out_crs,
#         "bbox_out": [float(bbox_out[0]), float(bbox_out[1]), float(bbox_out[2]), float(bbox_out[3])],
#         "geom_col": geom_col,
#         "sum": float(base_hist.sum()),
#         "nonzero": int(np.count_nonzero(base_hist)),
#         "crs": cfg.out_crs,
#         "bbox": [float(bbox_out[0]), float(bbox_out[1]), float(bbox_out[2]), float(bbox_out[3])],
#     }
#     (out_base.with_suffix(".json")).write_text(json.dumps(meta0, indent=2))

#     # L1..L{levels-1} by 2x2 sum
#     img = base_hist
#     for level in range(1, cfg.levels):
#         img = _downsample_2x2_sum(img)
#         outp = out_base.parent / f"{out_base.stem}_L{level}.npy"
#         np.save(outp, img)
#         meta = {
#             "tile_id": tile_id,
#             "level": int(level),
#             "grid_size": int(img.shape[0]),
#             "dtype": cfg.dtype,
#             "bbox_out_epsg": cfg.out_crs,
#             "bbox_out": [float(bbox_out[0]), float(bbox_out[1]), float(bbox_out[2]), float(bbox_out[3])],
#             "geom_col": geom_col,
#             "sum": float(img.sum()),
#             "nonzero": int(np.count_nonzero(img)),
#             "derived_from": f"{out_base.stem}_L{level-1}.npy" if level > 0 else out_base.name,
#             "crs": cfg.out_crs,
#             "bbox": [float(bbox_out[0]), float(bbox_out[1]), float(bbox_out[2]), float(bbox_out[3])],
#         }
#         (outp.with_suffix(".json")).write_text(json.dumps(meta, indent=2))


# # ------------------------- Core: per-partition partials → reduce -------------------------

# def _compute_rg_partial(
#     pf: pq.ParquetFile,
#     rg_index: int,
#     geom_col: str,
#     bbox_out: Tuple[float, float, float, float],
#     transformer: Transformer,
#     n: int,
#     dtype: np.dtype,
#     partials_dir: Optional[Path],
#     tile_id: str
# ) -> np.ndarray:
#     t = pf.read_row_group(rg_index, columns=[geom_col]).combine_chunks()
#     hist = _accumulate_vertices_hist(t, geom_col, bbox_out, transformer, n, dtype)
#     if partials_dir is not None:
#         partials_dir.mkdir(parents=True, exist_ok=True)
#         np.save(partials_dir / f"{tile_id}_rg{rg_index:04d}.npy", hist)
#     return hist


# def _process_one_tile_with_partials(
#     parquet_path: Path,
#     outdir: Path,
#     cfg: HistConfig,
#     geom_col: str
# ) -> Tuple[str, Path]:
#     tile_id = parquet_path.stem
#     logger.info("Histogram(partials): tile %s", tile_id)

#     pf = pq.ParquetFile(str(parquet_path))
#     schema = pf.schema_arrow
#     _, crs_meta = _load_geo_metadata(schema, geom_col)
#     src_crs = _infer_source_crs(crs_meta)
#     transformer = _reproject_transformer(src_crs, cfg.out_crs)

#     # One pass to fix bbox_out (tight and stable across all RGs)
#     bbox_out = _compute_tile_bbox_out_once(pf, geom_col, transformer, cfg.out_crs)

#     # Per-RG partials in parallel (bounded by cfg.rg_parallel)
#     dtype = np.dtype(cfg.dtype)
#     base = np.zeros((cfg.grid_size, cfg.grid_size), dtype=dtype)
#     partials_dir = (cfg.partials_dir / tile_id) if (cfg.keep_partials and cfg.partials_dir) else (
#         (outdir / "partials" / tile_id) if cfg.keep_partials else None
#     )

#     num_rg = pf.metadata.num_row_groups
#     if num_rg == 0:
#         logger.warning("Tile %s has zero row groups; skipping.", tile_id)
#         return tile_id, outdir / f"{tile_id}_L0.npy"

#     with ThreadPoolExecutor(max_workers=max(1, int(cfg.rg_parallel))) as ex:
#         futs = [
#             ex.submit(
#                 _compute_rg_partial,
#                 pf,
#                 rg,
#                 geom_col,
#                 bbox_out,
#                 transformer,
#                 cfg.grid_size,
#                 dtype,
#                 partials_dir,
#                 tile_id,
#             )
#             for rg in range(num_rg)
#         ]
#         for f in as_completed(futs):
#             base += f.result()

#     if float(base.sum()) == 0.0:
#         logger.warning("Tile %s produced an empty histogram with bounds %s; retrying with vertex-based bbox.", tile_id, bbox_out)
#         bbox_out = _compute_tile_bbox_out_from_vertices(pf, geom_col, transformer, cfg.out_crs)
#         base = np.zeros((cfg.grid_size, cfg.grid_size), dtype=dtype)
#         with ThreadPoolExecutor(max_workers=max(1, int(cfg.rg_parallel))) as ex2:
#             futs2 = [
#                 ex2.submit(
#                     _compute_rg_partial,
#                     pf,
#                     rg,
#                     geom_col,
#                     bbox_out,
#                     transformer,
#                     cfg.grid_size,
#                     dtype,
#                     None,
#                     tile_id,
#                 ) for rg in range(num_rg)
#             ]
#             for f2 in as_completed(futs2):
#                 base += f2.result()
#         logger.info("Tile %s retry sum=%s", tile_id, float(base.sum()))

#     # Reduce done → write outputs (L0..)
#     out_base = outdir / f"{tile_id}_L0"
#     _write_hist_outputs(base, out_base, cfg, tile_id, bbox_out, geom_col)
#     logger.info("Histogram(partials): finished tile %s → %s", tile_id, out_base.with_suffix(".npy"))
#     return tile_id, out_base.with_suffix(".npy")


# # ------------------------- Public API -------------------------

# def build_histograms_for_dir(
#     tiles_dir: str,
#     outdir: str,
#     geom_col: str = "geometry",
#     grid_size: int = 4096,
#     levels: int = 10,
#     out_crs: str = "EPSG:3857",
#     dtype: str = "float64",
#     hist_max_parallel: int = 8,
#     hist_rg_parallel: int = 4,
#     keep_partials: bool = False,
#     partials_dir: Optional[str] = None,
# ) -> None:
#     """
#     For each *.parquet tile under `tiles_dir`, compute a per-row-group partial histogram,
#     reduce by summation to L0, then build a 2x2-sum pyramid.

#     Parameters
#     ----------
#     hist_max_parallel : int
#         Number of tiles processed concurrently.
#     hist_rg_parallel : int
#         Number of row groups processed concurrently *per tile*.
#     keep_partials : bool
#         If True, writes per-RG partial arrays to disk.
#     partials_dir : Optional[str]
#         Base directory to place partials (defaults to <outdir>/partials/<tile_id>/).
#     """
#     cfg = HistConfig(
#         grid_size=int(grid_size),
#         levels=int(levels),
#         out_crs=str(out_crs),
#         dtype=str(dtype),
#         max_parallel_tiles=int(hist_max_parallel),
#         rg_parallel=int(hist_rg_parallel),
#         keep_partials=bool(keep_partials),
#         partials_dir=(Path(partials_dir) if partials_dir else None),
#     )

#     tiles = sorted(Path(tiles_dir).rglob("*.parquet"))
#     logger.info("Found %d parquet tile(s) under %s; out CRS=%s", len(tiles), tiles_dir, cfg.out_crs)
#     if not tiles:
#         logger.warning("No parquet tiles found under %s", tiles_dir)
#         return

#     outdir_p = Path(outdir)
#     outdir_p.mkdir(parents=True, exist_ok=True)

#     with ThreadPoolExecutor(max_workers=max(1, int(cfg.max_parallel_tiles))) as ex:
#         futs = {
#             ex.submit(_process_one_tile_with_partials, p, outdir_p, cfg, geom_col): p.name
#             for p in tiles
#         }
#         for f in as_completed(futs):
#             name = futs[f]
#             try:
#                 _ = f.result()
#             except Exception as e:
#                 logger.exception("Failed processing tile %s: %s", name, str(e))
#                 raise


from __future__ import annotations

import json
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from shapely import from_wkb
from shapely.geometry import (
    Point,
    LineString,
    LinearRing,
    Polygon,
    MultiPoint,
    MultiLineString,
    MultiPolygon,
    GeometryCollection,
)
from pyproj import Transformer

logger = logging.getLogger(__name__)

# ------------------------- Config -------------------------

@dataclass
class HistConfig:
    grid_size: int = 4096
    levels: int = 10
    out_crs: str = "EPSG:3857"
    dtype: str = "float64"
    max_parallel_tiles: int = 8
    rg_parallel: int = 4
    keep_partials: bool = False
    partials_dir: Optional[Path] = None


# ------------------------- Helpers -------------------------

def _load_geo_metadata(schema: pa.Schema, geom_col: str) -> Tuple[Optional[dict], Optional[str]]:
    meta = dict(schema.metadata or {})
    raw = meta.get(b"geo")
    if not raw:
        return None, None
    try:
        j = json.loads(raw.decode("utf-8"))
    except Exception:
        return None, None
    prim = j.get("primary_column") or geom_col
    crs = j.get("columns", {}).get(prim, {}).get("crs")
    return j, crs


def _infer_source_crs(crs_from_meta: Optional[str]) -> str:
    return crs_from_meta if crs_from_meta else "EPSG:4326"


def _reproject_transformer(src_crs: str, dst_crs: str) -> Transformer:
    return Transformer.from_crs(src_crs, dst_crs, always_xy=True)


def _geometry_vertices_iter(g) -> Iterable[Tuple[float, float]]:
    if g is None or g.is_empty:
        return
    if isinstance(g, Point):
        yield (g.x, g.y)
    elif isinstance(g, (LineString, LinearRing)):
        for x, y in g.coords:
            yield (x, y)
    elif isinstance(g, Polygon):
        for x, y in g.exterior.coords:
            yield (x, y)
        for ring in g.interiors:
            for x, y in ring.coords:
                yield (x, y)
    elif isinstance(g, (MultiPoint, MultiLineString, MultiPolygon, GeometryCollection)):
        for sub in getattr(g, "geoms", []):
            yield from _geometry_vertices_iter(sub)
    else:
        coords = getattr(g, "coords", None)
        if coords is not None:
            for x, y in coords:
                yield (x, y)


def _downsample_2x2_sum(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    h2 = h // 2
    w2 = w // 2
    img = img[: 2 * h2, : 2 * w2]
    return (img.reshape(h2, 2, w2, 2).sum(axis=(1, 3))).astype(img.dtype, copy=False)


def _accumulate_vertices_hist(
    table: pa.Table,
    geom_col: str,
    bbox_out: Tuple[float, float, float, float],
    transformer: Transformer,
    n: int,
    dtype: np.dtype
) -> np.ndarray:
    hist = np.zeros((n, n), dtype=dtype)
    geoms = from_wkb(table[geom_col].to_numpy(zero_copy_only=False))
    minx, miny, maxx, maxy = bbox_out

    for g in geoms:
        if g is None or g.is_empty:
            continue
        xs: List[float] = []
        ys: List[float] = []
        for x, y in _geometry_vertices_iter(g):
            xs.append(x); ys.append(y)
        if not xs:
            continue
        X, Y = transformer.transform(xs, ys)
        tx = (np.asarray(X) - minx) / (maxx - minx) if (maxx > minx) else np.zeros_like(X)
        ty = (np.asarray(Y) - miny) / (maxy - miny) if (maxy > miny) else np.zeros_like(Y)
        ix = np.floor(tx * n).astype(np.int64)
        iy = np.floor(ty * n).astype(np.int64)
        if ix.size == 0 or iy.size == 0:
            continue
        np.clip(ix, 0, n - 1, out=ix)
        np.clip(iy, 0, n - 1, out=iy)
        np.add.at(hist, (iy, ix), 1.0)
    return hist


def _write_hist_outputs(
    base_hist: np.ndarray,
    out_base: Path,
    cfg: HistConfig,
    tile_id: str,
    bbox_out: Tuple[float, float, float, float],
    geom_col: str
) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)

    # Level 0
    np.save(out_base.with_suffix(".npy"), base_hist)
    meta0 = {
        "tile_id": tile_id,
        "level": 0,
        "grid_size": int(cfg.grid_size),
        "dtype": cfg.dtype,
        "bbox_out_epsg": cfg.out_crs,
        "bbox_out": [float(bbox_out[0]), float(bbox_out[1]), float(bbox_out[2]), float(bbox_out[3])],
        "geom_col": geom_col,
        "sum": float(base_hist.sum()),
        "nonzero": int(np.count_nonzero(base_hist)),
        "crs": cfg.out_crs,
    }
    (out_base.with_suffix(".json")).write_text(json.dumps(meta0, indent=2))

    # Downsample pyramid
    img = base_hist
    for level in range(1, cfg.levels):
        img = _downsample_2x2_sum(img)
        outp = out_base.parent / f"{out_base.stem}_L{level}.npy"
        np.save(outp, img)
        meta = {
            "tile_id": tile_id,
            "level": int(level),
            "grid_size": int(img.shape[0]),
            "dtype": cfg.dtype,
            "bbox_out_epsg": cfg.out_crs,
            "bbox_out": [float(bbox_out[0]), float(bbox_out[1]), float(bbox_out[2]), float(bbox_out[3])],
            "geom_col": geom_col,
            "sum": float(img.sum()),
            "nonzero": int(np.count_nonzero(img)),
            "derived_from": f"{out_base.stem}_L{level-1}.npy",
            "crs": cfg.out_crs,
        }
        (outp.with_suffix(".json")).write_text(json.dumps(meta, indent=2))


# ------------------------- Core -------------------------

def _compute_rg_partial(
    pf: pq.ParquetFile,
    rg_index: int,
    geom_col: str,
    bbox_out: Tuple[float, float, float, float],
    transformer: Transformer,
    n: int,
    dtype: np.dtype,
    partials_dir: Optional[Path],
    tile_id: str
) -> np.ndarray:
    t = pf.read_row_group(rg_index, columns=[geom_col]).combine_chunks()
    hist = _accumulate_vertices_hist(t, geom_col, bbox_out, transformer, n, dtype)
    if partials_dir is not None:
        partials_dir.mkdir(parents=True, exist_ok=True)
        np.save(partials_dir / f"{tile_id}_rg{rg_index:04d}.npy", hist)
    return hist


def _process_one_tile_with_partials(
    parquet_path: Path,
    outdir: Path,
    cfg: HistConfig,
    geom_col: str
) -> Tuple[str, Path]:
    tile_id = parquet_path.stem
    logger.info("Histogram(partials): tile %s", tile_id)

    pf = pq.ParquetFile(str(parquet_path))
    schema = pf.schema_arrow
    _, crs_meta = _load_geo_metadata(schema, geom_col)
    src_crs = _infer_source_crs(crs_meta)
    transformer = _reproject_transformer(src_crs, cfg.out_crs)

    # ✅ Use global EPSG:3857 extent for all tiles
    LIM = 20037508.342789244
    bbox_out = (-LIM, -LIM, LIM, LIM)

    dtype = np.dtype(cfg.dtype)
    base = np.zeros((cfg.grid_size, cfg.grid_size), dtype=dtype)
    partials_dir = (cfg.partials_dir / tile_id) if (cfg.keep_partials and cfg.partials_dir) else (
        (outdir / "partials" / tile_id) if cfg.keep_partials else None
    )

    num_rg = pf.metadata.num_row_groups
    if num_rg == 0:
        logger.warning("Tile %s has zero row groups; skipping.", tile_id)
        return tile_id, outdir / f"{tile_id}_L0.npy"

    with ThreadPoolExecutor(max_workers=max(1, int(cfg.rg_parallel))) as ex:
        futs = [
            ex.submit(
                _compute_rg_partial,
                pf,
                rg,
                geom_col,
                bbox_out,
                transformer,
                cfg.grid_size,
                dtype,
                partials_dir,
                tile_id,
            )
            for rg in range(num_rg)
        ]
        for f in as_completed(futs):
            base += f.result()

    out_base = outdir / f"{tile_id}_L0"
    _write_hist_outputs(base, out_base, cfg, tile_id, bbox_out, geom_col)
    logger.info("Histogram(partials): finished tile %s → %s", tile_id, out_base.with_suffix(".npy"))
    return tile_id, out_base.with_suffix(".npy")


# ------------------------- Public API -------------------------

def build_histograms_for_dir(
    tiles_dir: str,
    outdir: str,
    geom_col: str = "geometry",
    grid_size: int = 4096,
    levels: int = 10,
    out_crs: str = "EPSG:3857",
    dtype: str = "float64",
    hist_max_parallel: int = 8,
    hist_rg_parallel: int = 4,
    keep_partials: bool = False,
    partials_dir: Optional[str] = None,
) -> None:
    cfg = HistConfig(
        grid_size=int(grid_size),
        levels=int(levels),
        out_crs=str(out_crs),
        dtype=str(dtype),
        max_parallel_tiles=int(hist_max_parallel),
        rg_parallel=int(hist_rg_parallel),
        keep_partials=bool(keep_partials),
        partials_dir=(Path(partials_dir) if partials_dir else None),
    )

    tiles = sorted(Path(tiles_dir).rglob("*.parquet"))
    logger.info("Found %d parquet tile(s) under %s; out CRS=%s", len(tiles), tiles_dir, cfg.out_crs)
    if not tiles:
        logger.warning("No parquet tiles found under %s", tiles_dir)
        return

    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max(1, int(cfg.max_parallel_tiles))) as ex:
        futs = {
            ex.submit(_process_one_tile_with_partials, p, outdir_p, cfg, geom_col): p.name
            for p in tiles
        }
        for f in as_completed(futs):
            name = futs[f]
            try:
                _ = f.result()
            except Exception as e:
                logger.exception("Failed processing tile %s: %s", name, str(e))
                raise