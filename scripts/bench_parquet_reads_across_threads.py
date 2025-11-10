# #!/usr/bin/env python3
# import argparse, time, os, json
# import pyarrow as pa
# import pyarrow.parquet as pq
# import pyarrow.dataset as ds

# def fmt_bytes(n):
#     for unit in ["B","KB","MB","GB","TB"]:
#         if n < 1024 or unit == "TB": return f"{n:.1f} {unit}"
#         n /= 1024

# def bench_read_table(path, use_threads, columns=None):
#     t0 = time.perf_counter()
#     tbl = pq.read_table(path, columns=columns, use_threads=use_threads)
#     t1 = time.perf_counter()
#     secs = t1 - t0
#     rows = tbl.num_rows
#     nbytes = tbl.nbytes  # size of materialized table in memory
#     return {"api":"pq.read_table", "use_threads":use_threads, "secs":secs, "rows":rows, "tbl_bytes":nbytes}

# def bench_dataset_scan(path, use_threads, columns=None, batch_size=None):
#     dataset = ds.dataset(path, format="parquet")
#     scan_builder = dataset.scanner(columns=columns, use_threads=use_threads, batch_size=batch_size)
#     t0 = time.perf_counter()
#     rows = 0
#     nbytes = 0
#     for batch in scan_builder.to_batches():
#         rows += batch.num_rows
#         nbytes += sum(col.nbytes for col in batch.columns)
#     t1 = time.perf_counter()
#     secs = t1 - t0
#     return {"api":"ds.scanner", "use_threads":use_threads, "secs":secs, "rows":rows, "tbl_bytes":nbytes}

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("parquet_path")
#     ap.add_argument("--columns", nargs="*", default=None, help="Optional subset of columns")
#     ap.add_argument("--batch-size", type=int, default=None, help="Batch size for dataset scanner")
#     ap.add_argument("--reps", type=int, default=3, help="Repetitions per setting (best time reported)")
#     args = ap.parse_args()

#     path = args.parquet_path
#     if not os.path.exists(path):
#         raise SystemExit(f"File not found: {path}")

#     # quick warmup
#     try:
#         _ = pq.read_table(path, columns=args.columns, use_threads=True).slice(0,1)
#     except Exception:
#         pass

#     results = []

#     def best_of(fn, reps=3):
#         best = None
#         for _ in range(reps):
#             r = fn()
#             if best is None or r["secs"] < best["secs"]:
#                 best = r
#         return best

#     # Bench 1: pq.read_table use_threads=False/True
#     results.append(best_of(lambda: bench_read_table(path, use_threads=False, columns=args.columns), args.reps))
#     results.append(best_of(lambda: bench_read_table(path, use_threads=True,  columns=args.columns), args.reps))

#     # Bench 2: dataset scanner use_threads=False/True
#     results.append(best_of(lambda: bench_dataset_scan(path, use_threads=False, columns=args.columns, batch_size=args.batch_size), args.reps))
#     results.append(best_of(lambda: bench_dataset_scan(path, use_threads=True,  columns=args.columns, batch_size=args.batch_size), args.reps))

#     # Pretty print
#     print("\n=== Parquet read benchmark ===")
#     print(f"File: {path}")
#     if args.columns:
#         print(f"Columns: {args.columns}")
#     if args.batch_size:
#         print(f"Scanner batch_size: {args.batch_size}")
#     print(f"Repetitions: {args.reps} (best time shown)\n")

#     header = f"{'API':12} {'Threads':8} {'Time (s)':>10} {'Rows':>12} {'Rows/sec':>14} {'Materialized size':>18}"
#     print(header)
#     print("-"*len(header))
#     for r in results:
#         rows = max(r["rows"], 1)
#         rows_per_s = rows / r["secs"] if r["secs"] > 0 else float('inf')
#         print(f"{r['api']:12} {str(r['use_threads']):8} {r['secs']:10.3f} {r['rows']:12,d} {rows_per_s:14.0f} {fmt_bytes(r['tbl_bytes']):>18}")

#     # JSON dump for copy/paste into docs if you want
#     print("\nJSON:")
#     print(json.dumps(results, indent=2))

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
import argparse, time, pyarrow.parquet as pq, json, os

def fmt_bytes(n):
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024 or unit == "TB": return f"{n:.1f} {unit}"
        n /= 1024

def bench_read_table(path, use_threads):
    t0 = time.perf_counter()
    tbl = pq.read_table(path, use_threads=use_threads)
    t1 = time.perf_counter()
    secs = t1 - t0
    return {
        "use_threads": use_threads,
        "secs": secs,
        "rows": tbl.num_rows,
        "tbl_bytes": tbl.nbytes
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet_path")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    path = args.parquet_path

    results = []
    for flag in [False, True]:
        best = None
        for _ in range(args.reps):
            r = bench_read_table(path, flag)
            if best is None or r["secs"] < best["secs"]:
                best = r
        results.append(best)

    print(f"\n=== Benchmark: pyarrow.parquet.read_table ===\nFile: {path}\n")
    print(f"{'Threads':8} {'Time (s)':>10} {'Rows':>12} {'Rows/sec':>14} {'In-memory size':>18}")
    print("-" * 70)
    for r in results:
        rows_per_s = r["rows"] / r["secs"]
        print(f"{str(r['use_threads']):8} {r['secs'] * 35:10.3f} {r['rows']:12,d} {rows_per_s:14.0f} {fmt_bytes(r['tbl_bytes']):>18}")
    print("\nJSON:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()