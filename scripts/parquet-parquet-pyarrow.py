#!/usr/bin/env python3
import pyarrow.parquet as pq
import sys

if len(sys.argv) != 3:
    print("Usage: python3 copy_simple.py input.parquet output.parquet")
    sys.exit(1)

src, dst = sys.argv[1], sys.argv[2]

# Just read and write — nothing else
table = pq.read_table(src)
pq.write_table(table, dst)

# Optional check: print whether geo metadata exists
in_meta = table.schema.metadata
print("Input has geo metadata:", bool(in_meta and b"geo" in in_meta))

out_pf = pq.ParquetFile(dst)
out_meta = out_pf.schema_arrow.metadata
print("Output has geo metadata:", bool(out_meta and b"geo" in out_meta))
