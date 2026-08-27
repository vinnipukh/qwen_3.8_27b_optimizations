#!/usr/bin/env python3
"""
benchmarks/tools/run_kernel_bench.py

Executes standalone HIP microbenchmarks and archives fingerprinted timing tables
into benchmarks/results/ using RunStore (KERN-01 criterion 2 & owner lock D4-00-3).
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

from benchmarks.lib.fingerprint import collect_manifest
from benchmarks.lib.store import RunStore


def run_benchmark(bin_path: str) -> list[dict]:
    """Runs the benchmark binary inside WSL (with DXG enabled) and parses JSON stdout."""
    env = os.environ.copy()
    env["HSA_ENABLE_DXG_DETECTION"] = "1"

    cmd = [bin_path]
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    if res.returncode != 0:
        sys.stderr.write(f"Benchmark run failed with code {res.returncode}:\n{res.stderr}\n")
        sys.exit(res.returncode)

    stdout = res.stdout.strip()
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as err:
        sys.stderr.write(f"Failed to parse JSON output: {err}\nOutput was:\n{stdout}\n")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run HIP kernel benchmark and archive to RunStore")
    parser.add_argument("--bin", type=str, default="kernels/build/demo_iq4xs_dequant/demo_bench", help="Path to bench binary")
    parser.add_argument("--op", type=str, default="demo_iq4xs_dequant", help="Op name identifier")
    parser.add_argument("--results-root", type=str, default="benchmarks/results", help="Results directory root")
    args = parser.parse_args()

    bin_path = Path(args.bin)
    if not bin_path.exists():
        sys.stderr.write(f"Benchmark binary not found at {bin_path}\n")
        sys.exit(1)

    print(f"[KernelBench] Executing benchmark binary: {bin_path}...")
    bench_data = run_benchmark(str(bin_path))
    print(f"[KernelBench] Collected {len(bench_data)} sweep data points.")

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"kernels_{args.op}_{ts}"
    store = RunStore.create(results_root=args.results_root, run_id=run_id, label=args.op)

    # 1. Write bench_sweep.json
    bench_sweep_path = store.run_dir / "bench_sweep.json"
    with open(bench_sweep_path, "w", encoding="utf-8") as f:
        json.dump(bench_data, f, indent=2)

    # 2. Append rows to rows.jsonl
    for row in bench_data:
        row_copy = dict(row)
        row_copy["timestamp_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        store.append_row(row_copy)

    # 3. Generate system manifest
    try:
        manifest = collect_manifest(
            run_dir=store.run_dir,
            backend_arm="HIP",
            telemetry_mode="absent",
            bin_path=str(bin_path),
            model_path="models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf",
            repo_root="."
        )
        with open(store.run_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    except Exception as e:
        print(f"[KernelBench Warning] Manifest generation partial fallback: {e}")
        fallback_manifest = {
            "run_id": run_id,
            "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "op": args.op,
            "gpu": "unknown",
            "rocm": "unknown",
            "commit": "unknown",
            "error": str(e)
        }
        with open(store.run_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(fallback_manifest, f, indent=2)

    # 4. Write checksums
    checksum_file = store.write_checksums()
    print(f"[KernelBench] Archived results to {store.run_dir}")
    print(f"[KernelBench] Checksums written to {checksum_file}")


if __name__ == "__main__":
    main()
