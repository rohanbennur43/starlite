#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import datetime as dt
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Tuple

try:
    import psutil
except ImportError:
    print("psutil is required. Try: pip install psutil", file=sys.stderr)
    sys.exit(1)

def human_bytes(n: int) -> str:
    units = ["B","KB","MB","GB","TB","PB"]
    i = 0
    x = float(n)
    while x >= 1024.0 and i < len(units)-1:
        x /= 1024.0
        i += 1
    return f"{x:.1f} {units[i]}"

def parse_tiles(arg: str) -> List[int]:
    arg = arg.strip()
    if ":" in arg:
        a, b, *rest = arg.split(":")
        step = int(rest[0]) if rest else 1
        return list(range(int(a), int(b) + 1, step))
    return [int(x) for x in arg.split(",") if x.strip()]

def iter_mpf(start: int, end: int, step: int) -> Iterable[int]:
    v = start
    while v <= end:
        yield v
        v += step

def sum_tree_rss(proc: psutil.Process) -> int:
    try:
        rss = proc.memory_info().rss
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0
    try:
        for ch in proc.children(recursive=True):
            try:
                rss += ch.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return rss

def run_once(
    input_path: str,
    base_outdir: Path,
    num_tiles: int,
    mpf: int,
    geom_col: str,
    sort_mode: str,
    sample_ratio: float,
    seed: int,
    extra_args: List[str],
    py_exe: str = sys.executable,
) -> Tuple[int, float, int, Path, Path]:
    run_tag = f"tiles_{num_tiles}_mpf_{mpf}"
    run_outdir = base_outdir / f"out_{run_tag}"
    run_outdir.mkdir(parents=True, exist_ok=True)

    log_path = base_outdir / f"log_{run_tag}.txt"

    cmd = [
        py_exe, "-u", "-m", "tile_geoparquet.cli",
        "--input", input_path,
        "--outdir", str(run_outdir),
        "--num-tiles", str(num_tiles),
        "--max-parallel-files", str(mpf),
        "--geom-col", geom_col,
        "--sort-mode", sort_mode,
        "--sample-ratio", str(sample_ratio),
        "--seed", str(seed),
    ] + extra_args

    with open(log_path, "w", buffering=1) as lf:
        lf.write(f"# launched: {dt.datetime.now().isoformat()}\n")
        lf.write("# cmd: " + " ".join(shlex.quote(c) for c in cmd) + "\n\n")

    with open(log_path, "a", buffering=1) as lf:
        t0 = time.perf_counter()
        proc = subprocess.Popen(
            cmd,
            stdout=lf,
            stderr=lf,
            env=dict(os.environ, PYTHONUNBUFFERED="1"),
            start_new_session=True,
        )

        peak_rss = 0
        ps_proc = psutil.Process(proc.pid)

        try:
            while True:
                if proc.poll() is not None:
                    break
                rss = sum_tree_rss(ps_proc)
                if rss > peak_rss:
                    peak_rss = rss
                time.sleep(0.25)
        except KeyboardInterrupt:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            raise
        finally:
            try:
                rss = sum_tree_rss(ps_proc)
                if rss > peak_rss:
                    peak_rss = rss
            except Exception:
                pass

        wall = time.perf_counter() - t0
        exit_code = proc.returncode if proc.returncode is not None else -999

    return exit_code, wall, peak_rss, run_outdir, log_path

def write_row_csv(csv_path: Path, header: List[str], row: List[str]) -> None:
    # Append mode; create header iff file is new/empty
    need_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(header)
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())  # durability per-iteration

def main():
    ap = argparse.ArgumentParser(description="Sweep num_tiles and max_parallel_files; record time + peak RSS (appends per-iteration).")
    ap.add_argument("--input", required=True, help="Path to input GeoParquet/GeoJSON")
    ap.add_argument("--base-outdir", default="bench_out", help="Directory to store per-run outputs/logs")
    ap.add_argument("--geom-col", default="geometry")
    ap.add_argument("--sort-mode", default="zorder", choices=["none", "columns", "zorder", "hilbert"])
    ap.add_argument("--tiles", default="25,40,55,70,85,100", help="Comma list or range 'start:end:step'")
    ap.add_argument("--mpf-start", type=int, default=10)
    ap.add_argument("--mpf-end", type=int, default=50)
    ap.add_argument("--mpf-step", type=int, default=5)
    ap.add_argument("--sample-ratio", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv", default="bench_results.csv", help="Output CSV path (appended)")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, help="Extra args passed to tile_geoparquet.cli")
    args = ap.parse_args()

    tiles_list = parse_tiles(args.tiles)
    mpf_values = list(iter_mpf(args.mpf_start, args.mpf_end, args.mpf_step))
    base_outdir = Path(args.base_outdir)
    base_outdir.mkdir(parents=True, exist_ok=True)

    header = [
        "num_tiles",
        "max_parallel_files",
        "exit_code",
        "wall_time_sec",
        "peak_rss_bytes",
        "peak_rss_human",
        "outdir",
        "log_path",
        "timestamp",
    ]
    csv_path = Path(args.csv)

    total_runs = len(tiles_list) * len(mpf_values)
    run_idx = 0

    for nt in tiles_list:
        for mpf in mpf_values:
            run_idx += 1
            stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{run_idx}/{total_runs}] tiles={nt} mpf={mpf}  → running…", flush=True)

            try:
                code, wall, peak, outdir, logp = run_once(
                    input_path=args.input,
                    base_outdir=base_outdir,
                    num_tiles=nt,
                    mpf=mpf,
                    geom_col=args.geom_col,
                    sort_mode=args.sort_mode,
                    sample_ratio=args.sample_ratio,
                    seed=args.seed,
                    extra_args=(args.extra or []),
                )
            except KeyboardInterrupt:
                # Write a partial/failure row to keep trace (optional: comment out if undesired)
                write_row_csv(csv_path, header, [
                    nt, mpf, 130, f"{-1.0:.3f}", 0, human_bytes(0),
                    str(base_outdir), "INTERRUPTED", stamp,
                ])
                print("\nInterrupted. Partial results saved.", file=sys.stderr)
                return
            except Exception as e:
                # Record failure; continue sweep
                print(f"Run failed: tiles={nt}, mpf={mpf}: {e}", file=sys.stderr)
                code, wall, peak, outdir, logp = (111, -1.0, 0, base_outdir, csv_path)

            # Persist this row immediately (per-iteration)
            write_row_csv(csv_path, header, [
                nt,
                mpf,
                code,
                f"{wall:.3f}",
                peak,
                human_bytes(peak),
                str(outdir),
                str(logp),
                stamp,
            ])

    print(f"\nDone. Results → {csv_path}")

if __name__ == "__main__":
    main()