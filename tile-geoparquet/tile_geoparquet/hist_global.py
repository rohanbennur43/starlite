# tile_geoparquet/hist_global.py
from __future__ import annotations

import os
import json
import math
import glob
import argparse
from pathlib import Path
from typing import Iterator, Tuple, Optional, List

import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
from shapely import from_wkb
from pyproj import Transformer

# ---- Global Web Mercator extent (EPSG:3857) covering ~whole world ----
WM_MINX = -20037508.342789244
WM_MAXX =  20037508.342789244
WM_MINY = -20037508.342789255
WM_MAXY =  20037508.342789255

def _iter_parquet_files(tiles_dir: str) -> Iterator[str]:
    for p in sorted(glob.glob(os.path.join(tiles_dir, "*.parquet"))):
        yield p

def _mk_transformer(in_crs: str, out_crs: str = "EPSG:3857") -> Transformer:
    # Always do x,y order in proj coords
    return Transformer.from_crs(in_crs, out_crs, always_xy=True)

def _cell_size(n: int) -> Tuple[float, float]:
    px = (WM_MAXX - WM_MINX) / n
    py = (WM_MAXY - WM_MINY) / n
    return px, py

def _xy_to_ij(x: np.ndarray, y: np.ndarray, n: int, px: float, py: float) -> Tuple[np.ndarray, np.ndarray]:
    # map mercator coords -> integer cell indices [0, n-1]
    ix = np.floor((x - WM_MINX) / px).astype(np.int64)
    iy = np.floor((WM_MAXY - y) / py).astype(np.int64)  # flip y so row 0 is north
    # clip
    np.clip(ix, 0, n - 1, out=ix)
    np.clip(iy, 0, n - 1, out=iy)
    return ix, iy

def _count_vertices_wkb_to_global(
    parquet_path: str,
    geom_col: str,
    transformer: Transformer,
    n: int,
    dtype: str = "uint64",
    rg_parallel: int = 1,
) -> np.ndarray:
    """
    Load a parquet tile in row-groups and accumulate vertex counts into a single n×n array.
    This uses polygon/line **vertices** (centroids NOT used). For area rasterization, swap strategy later.
    """
    arr = np.zeros((n, n), dtype=dtype)
    px, py = _cell_size(n)

    pf = pq.ParquetFile(parquet_path)
    rgs = list(range(pf.num_row_groups))

    def _iter_chunks() -> Iterator[np.ndarray]:
        for rg in rgs:
            batch = pf.read_row_group(rg, columns=[geom_col]).column(0)
            # Convert to Python bytes array once; avoid zero_copy to dodge memoryview pitfalls
            yield batch.to_numpy(zero_copy_only=False)

    for np_wkb in _iter_chunks():
        # Convert WKB -> shapely geometries
        geoms = from_wkb(np_wkb)
        xs: List[float] = []
        ys: List[float] = []

        for g in geoms:
            if g is None or g.is_empty:
                continue
            # We use vertices from the exterior + interiors (if polygon), otherwise coords
            try:
                if g.geom_type in ("Polygon", "MultiPolygon"):
                    if g.geom_type == "Polygon":
                        polys = [g]
                    else:
                        polys = list(g.geoms)
                    for poly in polys:
                        # exterior
                        ex = poly.exterior
                        if ex is not None:
                            cx, cy = ex.coords.xy
                            xs.extend(cx); ys.extend(cy)
                        # interiors
                        for ring in poly.interiors:
                            rx, ry = ring.coords.xy
                            xs.extend(rx); ys.extend(ry)
                else:
                    cx, cy = g.coords.xy  # LineString / LinearRing / Point / MultiPoint
                    xs.extend(cx); ys.extend(cy)
            except Exception:
                # Fallback: centroid as a minimal contribution if coords extraction fails
                c = g.centroid
                xs.append(float(c.x)); ys.append(float(c.y))

        if not xs:
            continue

        # Project to Web Mercator
        X, Y = transformer.transform(np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64))
        # Bin
        ix, iy = _xy_to_ij(X, Y, n, px, py)
        # Accumulate
        # Use np.add.at for repeated bins
        np.add.at(arr, (iy, ix), 1)

    return arr

def _save_partial_npy(out_path: str, arr: np.ndarray) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, arr, allow_pickle=False)

def _sum_partials_to_global(partial_dir: str, n: int, out_path: str, dtype: str = "uint64") -> None:
    parts = sorted(glob.glob(os.path.join(partial_dir, "*.npy")))
    if not parts:
        raise RuntimeError(f"No partial histograms found in {partial_dir}")

    total = np.zeros((n, n), dtype=dtype)
    for p in parts:
        a = np.load(p, mmap_mode="r")
        total += a  # safe: same dtype/shape
    np.save(out_path, total, allow_pickle=False)

def _downsample_2x(a: np.ndarray) -> np.ndarray:
    # reduce 2×2 blocks by sum → next coarser level
    h, w = a.shape
    h2, w2 = h // 2, w // 2
    a = a[: h2 * 2, : w2 * 2]
    return (
        a[0::2, 0::2] + a[0::2, 1::2] +
        a[1::2, 0::2] + a[1::2, 1::2]
    )

def _make_pyramid(global_npy: str, levels: int, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    base = np.load(global_npy, mmap_mode="r")
    cur = np.array(base, copy=True)
    np.save(os.path.join(outdir, f"level_0.npy"), cur, allow_pickle=False)
    for L in range(1, levels):
        cur = _downsample_2x(cur)
        np.save(os.path.join(outdir, f"level_{L}.npy"), cur, allow_pickle=False)

def _to_png(a: np.ndarray, out_png: str, cmap: str = "magma", vmin: Optional[float] = None, vmax: Optional[float] = None) -> None:
    # Render log1p density to an 8-bit PNG using matplotlib (no style)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig = plt.figure(figsize=(8, 8), dpi=512/8)  # results in 512×512 px
    ax = plt.axes([0, 0, 1, 1])
    ax.axis("off")

    img = np.log1p(a.astype(np.float64))
    if vmin is None:
        vmin = np.nanmin(img)
    if vmax is None:
        vmax = np.nanmax(img)

    ax.imshow(img, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def build_global_hist(
    tiles_dir: str,
    outdir: str,
    geom_col: str = "geometry",
    in_crs: str = "EPSG:4326",
    grid_size: int = 4096,
    partial_dir: Optional[str] = None,
    max_parallel_files: int = 0,  # not used here, kept for symmetry
    make_png: bool = True,
    pyramid_levels: int = 0,
    dtype: str = "uint64",
) -> str:
    """
    1) Iterate parquet tiles in tiles_dir
    2) For each tile, count vertices into the **global** 4096×4096 grid (partial .npy)
    3) Sum all partials -> global.npy
    4) Optional: build pyramid and top-level PNG
    Returns path to global.npy
    """
    if partial_dir is None:
        partial_dir = os.path.join(outdir, "_partials")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(partial_dir, exist_ok=True)

    transformer = _mk_transformer(in_crs, "EPSG:3857")
    n = int(grid_size)

    # 1 & 2: partials
    for idx, p in enumerate(_iter_parquet_files(tiles_dir), 1):
        arr = _count_vertices_wkb_to_global(
            parquet_path=p,
            geom_col=geom_col,
            transformer=transformer,
            n=n,
            dtype=dtype,
        )
        _save_partial_npy(os.path.join(partial_dir, f"partial_{idx:06d}.npy"), arr)

    # 3: sum
    global_npy = os.path.join(outdir, "global_level_0.npy")
    _sum_partials_to_global(partial_dir, n=n, out_path=global_npy, dtype=dtype)

    # 4: optional pyramid
    if pyramid_levels and pyramid_levels > 0:
        pyr_dir = os.path.join(outdir, "pyramid")
        _make_pyramid(global_npy, pyramid_levels, pyr_dir)

    # Optional quick PNG (level 0)
    if make_png:
        a0 = np.load(global_npy, mmap_mode="r")
        _to_png(a0, os.path.join(outdir, "global_level_0.png"))

    # Also write a tiny manifest so the viewer knows the georeference of the global PNG
    manifest = {
        "image": "global_level_0.png",
        "grid_size": n,
        "bbox_mercator": [WM_MINX, WM_MINY, WM_MAXX, WM_MAXY],
        "crs": "EPSG:3857",
        "scale": "log1p",
        "dtype": dtype,
    }
    with open(os.path.join(outdir, "global_manifest.json"), "w") as f:
        json.dump(manifest, f)

    return global_npy

def main():
    ap = argparse.ArgumentParser(description="Build a single global 4096×4096 histogram in EPSG:3857 by summing per-tile partials.")
    ap.add_argument("--tiles-dir", required=True, help="Directory of per-partition parquet tiles (*.parquet)")
    ap.add_argument("--outdir", required=True, help="Output directory (global .npy/.png + manifest)")
    ap.add_argument("--geom-col", default="geometry")
    ap.add_argument("--in-crs", default="EPSG:4326", help="Input CRS of geometries in parquet")
    ap.add_argument("--grid-size", type=int, default=4096)
    ap.add_argument("--dtype", default="uint64", choices=["uint64", "uint32"])
    ap.add_argument("--pyramid-levels", type=int, default=0, help="If >0, write 2x downsampled levels")
    ap.add_argument("--no-png", action="store_true", help="Skip writing quicklook PNG")
    args = ap.parse_args()

    build_global_hist(
        tiles_dir=args.tiles_dir,
        outdir=args.outdir,
        geom_col=args.geom_col,
        in_crs=args.in_crs,
        grid_size=args.grid_size,
        pyramid_levels=args.pyramid_levels,
        make_png=(not args.no_png),
        dtype=args.dtype,
    )

if __name__ == "__main__":
    main()