# streamer.py

import pyarrow as pa
import pyarrow.parquet as pq
from shapely import make_valid
from shapely.ops import transform as shapely_transform
import shapely.wkb as swkb
from pathlib import Path
from pyproj import Transformer
import logging

logger = logging.getLogger("bucket_mvt")


class GeometryStreamer:
    """
    Streams geometries from GeoParquet using PyArrow, row group by row group,
    exactly like your GeoParquetSource pattern.
    """

    def __init__(self, parquet_dir: str):
        self.parquet_dir = Path(parquet_dir)
        self.to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    def _decode_table(self, table):
        geom_col = table["geometry"].to_pylist()

        # Extract all columns except geometry
        attrs = {
            col: table[col].to_pylist()
            for col in table.column_names
            if col != "geometry"
        }

        for i, wkb in enumerate(geom_col):
            if wkb is None:
                continue

            geom = swkb.loads(wkb)
            geom = make_valid(geom)
            geom = shapely_transform(self.to_3857.transform, geom)

            if geom.is_empty:
                continue

            # Build attribute dict {column_name: value}
            row_attrs = {k: attrs[k][i] for k in attrs}

            yield geom, row_attrs


    def iter_geometries(self):
        """
        Main generator: iterate all parquet files, stream row groups,
        decode geometries, and yield shapely objects.
        """
        parquet_files = list(self.parquet_dir.rglob("*.parquet"))

        for pf in parquet_files:
            logger.info("Streaming GeoParquet file %s", pf)

            pf_obj = pq.ParquetFile(pf)
            num_row_groups = pf_obj.num_row_groups

            for rg in range(num_row_groups):
                table = pf_obj.read_row_group(rg)
                yield from self._decode_table(table)
