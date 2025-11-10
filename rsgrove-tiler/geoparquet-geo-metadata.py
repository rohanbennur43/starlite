import pyarrow.parquet as pq
pf = pq.ParquetFile("../original_datasets/out_ca.parquet")
print(pf.schema_arrow.metadata)
