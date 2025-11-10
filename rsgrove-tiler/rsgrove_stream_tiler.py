#!/usr/bin/env python3
import argparse, logging, os, json
import numpy as np
import pandas as pd
from shapely import from_wkb
from shapely.geometry import Polygon
from RSGrove import RSGrovePartitioner, BeastOptions, EnvelopeNDLite

# Optional projection (only if --project-centroids is set)
try:
    from pyproj import CRS, Transformer
    HAVE_PYPROJ = True
except Exception:
    HAVE_PYPROJ = False

# ----------------------------- Logging --------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("RSGroveTilerStreaming")

# ----------------------------- Utilities ------------------------------

class Summary2D:
    def __init__(self, mins, maxs):
        self._mins = np.array(mins, float)
        self._maxs = np.array(maxs, float)
    def getCoordinateDimension(self): return 2
    def getMinCoord(self, d): return float(self._mins[d])
    def getMaxCoord(self, d): return float(self._maxs[d])

def _clip_inf_to_summary(mins, maxs, summary):
    smin = np.array([summary.getMinCoord(0), summary.getMinCoord(1)], float)
    smax = np.array([summary.getMaxCoord(0), summary.getMaxCoord(1)], float)
    mins = np.where(np.isneginf(mins), smin, mins)
    maxs = np.where(np.isposinf(maxs),  smax, maxs)
    return mins, maxs

def _maybe_build_transformers(source_crs_str, project_centroids):
    if not project_centroids:
        return None, None, None
    if not HAVE_PYPROJ:
        raise SystemExit("pyproj required for --project-centroids.")
    if not source_crs_str:
        raise SystemExit("--source-crs required when --project-centroids is used.")
    src = CRS.from_user_input(source_crs_str)
    web = CRS.from_epsg(3857)
    to_3857 = Transformer.from_crs(src, web, always_xy=True)
    to_src  = Transformer.from_crs(web, src, always_xy=True)
    return src, to_3857, to_src

def _valid_coord(cx, cy):
    # Optional sanity guard – avoids a few corrupted coords skewing bounds
    return np.isfinite(cx) and np.isfinite(cy) and (abs(cx) <= 1e7) and (abs(cy) <= 1e7)

# ---------------------- Streaming samplers ----------------------------

def stream_sample_geojson(path, sample_pct, source_crs, project_centroids):
    """
    True streaming GeoJSON using Fiona; reservoir-sample centroids.

    If feature count is known (len(src)), use a **fixed target** k = round(N * pct).
    Otherwise use a **growing target** k(t) = round(t * pct) as features stream in.
    """
    import fiona
    from shapely.geometry import shape

    logger.info(f"Streaming GeoJSON from {path} ...")
    _, t_to3857, t_tosrc = _maybe_build_transformers(source_crs, project_centroids)

    xs, ys = [], []
    xmin = ymin = float("inf")
    xmax = ymax = float("-inf")

    with fiona.open(path, "r") as src:
        fixed_target = None
        try:
            n_features = len(src)
            fixed_target = max(1, int(round(n_features * (sample_pct / 100.0))))
            logger.info(f"GeoJSON feature count: {n_features:,}. Fixed reservoir target: {fixed_target:,}")
        except Exception:
            logger.info("GeoJSON feature count unknown; using growing reservoir target.")

        # CRS guard
        if project_centroids and not source_crs and (src.crs_wkt or src.crs):
            raise SystemExit("--source-crs not provided. Supply EPSG/PROJ when using --project-centroids.")

        total_rows = 0
        growing_target = None  # only used when fixed_target is None

        for i, feat in enumerate(src):
            g = feat.get("geometry")
            if not g:
                continue
            geom = shape(g)
            if geom.is_empty:
                continue

            cx, cy = geom.centroid.x, geom.centroid.y
            if project_centroids:
                X, Y = t_to3857.transform(cx, cy)
                cx, cy = t_tosrc.transform(X, Y)

            if not _valid_coord(cx, cy):
                continue

            # update global bounds
            if cx < xmin: xmin = cx
            if cy < ymin: ymin = cy
            if cx > xmax: xmax = cx
            if cy > ymax: ymax = cy

            total_rows += 1

            if fixed_target is not None:
                # Standard reservoir with fixed k
                if len(xs) < fixed_target:
                    xs.append(cx); ys.append(cy)
                else:
                    j = np.random.randint(0, total_rows)
                    if j < fixed_target:
                        xs[j] = cx; ys[j] = cy
            else:
                # Growing target k(t) ~ t * pct
                desired = max(1, int(round(total_rows * (sample_pct / 100.0))))
                if growing_target is None:
                    growing_target = desired

                if len(xs) < desired:
                    xs.append(cx); ys.append(cy)
                    growing_target = desired
                else:
                    j = np.random.randint(0, total_rows)
                    if j < growing_target:
                        xs[j] = cx; ys[j] = cy

            if (i + 1) % 50_000 == 0:
                logger.info(f"Streamed {i + 1:,} features ... sampled so far: {len(xs):,} "
                            f"(target={fixed_target if fixed_target is not None else growing_target})")

    logger.info(f"Completed streaming {total_rows:,} features. Final sample size: {len(xs):,}.")
    return np.array(xs), np.array(ys), (xmin, ymin, xmax, ymax)

def stream_sample_geoparquet(path, sample_pct, batch_size, geometry_col, source_crs, project_centroids):
    """
    Stream GeoParquet using PyArrow; reservoir-sample centroids (fixed k if total rows known).
    """
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    logger.info(f"Opening GeoParquet dataset from {path} ...")
    dataset = ds.dataset(path, format="parquet")
    scanner = dataset.scanner(columns=[geometry_col], batch_size=batch_size)

    # Best-effort total row estimate
    total_est = 0
    for file_path in dataset.files:
        try:
            pf = pq.ParquetFile(file_path)
            md = pf.metadata
            if md is not None and md.num_rows is not None:
                total_est += int(md.num_rows)
        except Exception:
            pass
    if total_est > 0:
        logger.info(f"Estimated {total_est:,} total rows across Parquet files.")
    else:
        logger.info("Could not estimate total rows (missing metadata).")

    fixed_target = max(1, int(round(total_est * (sample_pct / 100.0)))) if total_est > 0 else None
    _, t_to3857, t_tosrc = _maybe_build_transformers(source_crs, project_centroids)

    xs, ys = [], []
    total_rows = 0
    xmin = ymin = float("inf")
    xmax = ymax = float("-inf")

    for b_idx, batch in enumerate(scanner.to_batches()):
        col_idx = batch.schema.get_field_index(geometry_col)
        if col_idx == -1:
            raise SystemExit(f"Column '{geometry_col}' not found in Parquet schema: {batch.schema}")

        wkb_np = batch.column(col_idx).to_numpy(zero_copy_only=False)
        geoms = from_wkb(wkb_np)

        for g in geoms:
            if g is None or g.is_empty:
                continue

            cx, cy = g.centroid.x, g.centroid.y
            if project_centroids:
                X, Y = t_to3857.transform(cx, cy)
                cx, cy = t_tosrc.transform(X, Y)

            if not _valid_coord(cx, cy):
                continue

            # bounds
            if cx < xmin: xmin = cx
            if cy < ymin: ymin = cy
            if cx > xmax: xmax = cx
            if cy > ymax: ymax = cy

            total_rows += 1

            k = fixed_target if fixed_target is not None else max(1, int(round(total_rows * (sample_pct / 100.0))))
            if len(xs) < k:
                xs.append(cx); ys.append(cy)
            else:
                j = np.random.randint(0, total_rows)
                if j < k:
                    xs[j] = cx; ys[j] = cy

        if (b_idx + 1) % 10 == 0:
            logger.info(f"Processed {b_idx + 1} batches (~{total_rows:,} rows) ... sampled: {len(xs):,} "
                        f"(target={fixed_target if fixed_target is not None else k})")

    logger.info(f"Finished sampling {len(xs):,} centroids from ~{total_rows:,} geometries.")
    return np.array(xs), np.array(ys), (xmin, ymin, xmax, ymax)

# ------------------------------ Main ---------------------------------

def main():
    ap = argparse.ArgumentParser(description="Streaming R*-Grove tiler (GeoJSON via Fiona; GeoParquet via PyArrow).")
    ap.add_argument("input", help="Path to GeoParquet file/dir or GeoJSON file")
    ap.add_argument("--input-type", choices=["auto", "parquet", "geojson"], default="auto")
    ap.add_argument("--geometry", default="geometry", help="Geometry column for GeoParquet (default: geometry)")
    ap.add_argument("--batch-size", type=int, default=100_000, help="PyArrow batch size for Parquet streaming")
    ap.add_argument("--sample-pct", type=float, required=True, help="Percent of rows to sample uniformly (e.g., 1.0)")
    ap.add_argument("--num-partitions", type=int, required=True, help="Number of R*-Grove partitions to build")
    ap.add_argument("--out", default="tiles.geojson", help="Output tiles GeoJSON; also writes *_index.csv")
    ap.add_argument("--expand-to-inf", action="store_true", help="Expand outer partitions to infinity")
    ap.add_argument("--project-centroids", action="store_true",
                    help="Project centroids to EPSG:3857 for computation, then return to source CRS")
    ap.add_argument("--source-crs", default=None,
                    help="Source CRS (e.g., 'EPSG:4326'). Required with --project-centroids.")
    args = ap.parse_args()

    if args.sample_pct <= 0:
        raise SystemExit("--sample-pct must be > 0")

    # Decide input type
    in_lower = args.input.lower()
    if args.input_type == "auto":
        if in_lower.endswith(".parquet") or os.path.isdir(args.input):
            input_type = "parquet"
        elif in_lower.endswith(".geojson") or in_lower.endswith(".json"):
            input_type = "geojson"
        else:
            raise SystemExit("Could not infer input type. Use --input-type explicitly.")
    else:
        input_type = args.input_type

    logger.info(f"Detected input type: {input_type.upper()}")
    logger.info(f"Sampling {args.sample_pct}% of features; target partitions: {args.num_partitions}")

    # Stream + sample + bounds
    if input_type == "parquet":
        xs, ys, (xmin, ymin, xmax, ymax) = stream_sample_geoparquet(
            args.input, args.sample_pct, args.batch_size, args.geometry,
            args.source_crs, args.project_centroids
        )
    else:
        xs, ys, (xmin, ymin, xmax, ymax) = stream_sample_geojson(
            args.input, args.sample_pct, args.source_crs, args.project_centroids
        )

    if len(xs) == 0:
        raise SystemExit("No geometries sampled (empty input or all empty geometries).")

    logger.info(f"Global bounds: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")
    summary = Summary2D([xmin, ymin], [xmax, ymax])

    # Configure and run R*-Grove
    logger.info(f"Setting up partitioner: disjoint=True")
    conf = BeastOptions({
        RSGrovePartitioner.MMRatio: 0.95,
        RSGrovePartitioner.MinSplitRatio: 0.0,
        RSGrovePartitioner.ExpandToInfinity: bool(args.expand_to_inf),
    })
    logger.info(f"Config: mMRatio=0.95, MinSplitRatio=0.0, ExpandToInf={bool(args.expand_to_inf)}")
    r = RSGrovePartitioner()
    r.setup(conf, disjoint=True)

    logger.info(f"Constructing {args.num_partitions} R*-Grove partitions ...")
    sample_coords = np.vstack([xs, ys])  # shape (2, N)
    r.construct(summary=summary, sample=sample_coords, histogram=None,
                numPartitions=args.num_partitions)
    logger.info(f"Constructed {r.numPartitions()} partitions successfully.")

    # Build tiles + index
    records, features = [], []
    for pid in range(r.numPartitions()):
        env = EnvelopeNDLite(np.zeros(2), np.zeros(2))
        r.getPartitionMBR(pid, env)
        mins, maxs = _clip_inf_to_summary(env.mins, env.maxs, summary)
        minx, miny = float(mins[0]), float(mins[1])
        maxx, maxy = float(maxs[0]), float(maxs[1])

        features.append({
            "type": "Feature",
            "properties": {"id": pid},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]]
            }
        })
        records.append({
            "ID": pid,
            "File Name": f"part-{pid:05d}.rtree",
            "Record Count": 0,
            "NonEmpty Count": 0,
            "NumPoints": 0,
            "Data Size": 0,
            "Sum_x": 0.0,
            "Sum_y": 0.0,
            "Geometry": f"POLYGON(({minx} {miny}, {maxx} {miny}, {maxx} {maxy}, {minx} {maxy}, {minx} {miny}))",
            "xmin": minx, "ymin": miny, "xmax": maxx, "ymax": maxy,
        })

    # Write outputs (tiles GeoJSON + index CSV)
    logger.info(f"Writing GeoJSON to {args.out} ...")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)

    index_path = os.path.splitext(args.out)[0] + "_index.csv"
    pd.DataFrame(records).to_csv(index_path, index=False)
    logger.info(f"Done. Wrote {args.out} and {index_path}")

if __name__ == "__main__":
    main()
