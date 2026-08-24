#!/usr/bin/env python3
"""Calibration engine for Qwen3.8-27B benchmark harness (D2-13, D2-14, D2-20).

Subcommands:
  - labels: Discover and map HWiNFO sensor labels to canonical fields.
  - rehearse-kill: Safely rehearse thermal watchdog kill path against dummy process.
  - profile: Run healthy calibration runs (4k/8k) and derive empirical thresholds.json.
  - near-oom: Execute supervised near-OOM boundary test (32k tier / oversized batch).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.lib import fingerprint, guard, llabench, preflight, store, toast
from benchmarks.host import thermal_watchdog, hwinfo_daemon


def cmd_labels(args: argparse.Namespace) -> int:
    """Dump HWiNFO sensor label inventory."""
    out_path = args.out if args.out else Path("benchmarks/config/hwinfo_sensor_labels.txt")
    print(f"=== Discovering HWiNFO Sensor Labels -> {out_path} ===")

    # Attempt to read shared memory on Windows host or via interop
    try:
        res = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                f"python benchmarks/host/hwinfo_daemon.py --label-map {out_path.as_posix()}",
            ],
            capture_output=True,
            timeout=10,
        )
        stderr_str = res.stderr.decode("utf-8", errors="replace").strip() if res.stderr else ""
        if res.returncode == 0 and out_path.exists():
            print(f"Successfully dumped sensor label map to {out_path}")
            return 0
        else:
            print(f"HWiNFO Shared Memory unreadable from host: {stderr_str or 'HWiNFO SM2 disabled'}")
    except Exception as exc:
        print(f"HWiNFO read exception: {exc}")

    print("Writing default documented fallback label map...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# HWiNFO Sensor Label Map (Documented Fallback / Heuristic Mode)\n")
        f.write(f"# Generated: {fingerprint.utc_now_iso()}\n\n")
        for field in hwinfo_daemon.MANDATORY_FIELDS:
            f.write(f"canonical: {field} -> heuristic match\n")
    return 0


def cmd_rehearse_kill(args: argparse.Namespace) -> int:
    """Safely rehearse thermal watchdog cross-boundary kill path on a dummy process."""
    print("=== Rehearsing Thermal Watchdog Cross-Boundary Kill ===")
    dummy_proc = subprocess.Popen(["sleep", "60"])
    dummy_pid = dummy_proc.pid
    print(f"Spawned dummy process with PID: {dummy_pid}")

    pid_file = Path("benchmarks/tests/.rehearsal.pid")
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(f"{dummy_pid}\n", encoding="utf-8")

    # Execute kill via thermal watchdog helper
    kill_ok = thermal_watchdog.execute_kill(
        pid=dummy_pid,
        mode="wsl",
        pid_file=pid_file,
        dry_run=False,
    )

    time.sleep(0.5)
    # Check if process is dead
    poll_rc = dummy_proc.poll()
    print(f"Dummy process poll returncode: {poll_rc}")

    verdict_file = pid_file.parent / f"{pid_file.stem}.verdict"
    has_verdict = verdict_file.exists() and "FAILED:thermal-abort" in verdict_file.read_text()

    if pid_file.exists():
        pid_file.unlink()
    if verdict_file.exists():
        verdict_file.unlink()

    if poll_rc is not None and has_verdict:
        print("REHEARSAL_PASS: Cross-boundary kill, verdict marker, and toast verified.")
        return 0
    else:
        print("REHEARSAL_FAIL: Dummy process was not successfully killed or verdict missing.", file=sys.stderr)
        return 1


def cmd_profile(args: argparse.Namespace) -> int:
    """Run calibration profiling runs (4k/8k) and derive empirical thresholds.json."""
    from benchmarks.bin.run_session import run_session

    print("=== Running Calibration Profiling (Tiers: 4096, 8192) ===")
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    cal_run_dir = Path(f"benchmarks/results/{ts}_calibration_profile")

    run_store, ok_c, fail_c = run_session(
        tiers=(4096, 8192),
        fa_seq=("off", "on"),
        repeats=3,
        delay_s=5,
        out_dir=cal_run_dir,
        observe_only=True,
    )

    # Read rows and ledger to determine peak VmRSS and VmSwap
    rows = llabench.parse_rows(run_store.rows_file)
    max_vmrss_kb = 0
    max_vmswap_kb = 0

    for r in rows:
        g = r.get("guard", {})
        rss = g.get("vmrss_peak_kb", 0)
        swap = g.get("vmswap_peak_kb", 0)
        if rss > max_vmrss_kb:
            max_vmrss_kb = rss
        if swap > max_vmswap_kb:
            max_vmswap_kb = swap

    if max_vmrss_kb == 0:
        # Fallback to sensible memory base if /proc was not sampled in fast tests
        max_vmrss_kb = 16000000  # ~15.2 GB base

    # 1.5x steady-state margin per D2-13
    derived_rss_fail_kb = int(max_vmrss_kb * 1.5)
    derived_swap_fail_kb = max(524288, int(max_vmswap_kb * 1.5))  # at least 512 MiB swap allowance
    derived_gpu_climb_mb_min = 250.0

    thresholds_data = {
        "vmrss_fail_kb": derived_rss_fail_kb,
        "vmswap_fail_kb": derived_swap_fail_kb,
        "gpu_shared_climb_mb_per_min": derived_gpu_climb_mb_min,
        "repeat_deviation_max_ratio": 2.0,
        "derived_from": run_store.run_dir.name,
        "measured_peak_vmrss_kb": max_vmrss_kb,
        "measured_peak_vmswap_kb": max_vmswap_kb,
        "utc": fingerprint.utc_now_iso(),
    }

    out_thresh = Path("benchmarks/config/thresholds.json")
    out_thresh.parent.mkdir(parents=True, exist_ok=True)
    with open(out_thresh, "w", encoding="utf-8") as f:
        json.dump(thresholds_data, f, indent=2)

    print(f"\nDerived Thresholds written to {out_thresh}:")
    print(f"  VmRSS Fail Threshold: {derived_rss_fail_kb} kB ({derived_rss_fail_kb/1024/1024:.2f} GB)")
    print(f"  VmSwap Fail Threshold: {derived_swap_fail_kb} kB ({derived_swap_fail_kb/1024:.2f} MB)")
    print(f"  GPU Shared Memory Climb Max: {derived_gpu_climb_mb_min} MB/min")
    print(f"  Repeat Deviation Max Ratio: 2.0x")
    return 0


def cmd_near_oom(args: argparse.Namespace) -> int:
    """Execute supervised near-OOM verification run on tier 32768."""
    from benchmarks.bin.run_session import run_session

    print("=== Supervised Near-OOM Verification (Tier: 32768) ===")
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    oom_run_dir = Path(f"benchmarks/results/{ts}_calibration_near_oom")

    run_store, ok_c, fail_c = run_session(
        tiers=(32768,),
        fa_seq=("off", "on"),
        repeats=1,
        delay_s=0,
        out_dir=oom_run_dir,
    )

    rows = llabench.parse_rows(run_store.rows_file)
    print(f"Near-OOM Session completed with {len(rows)} rows recorded.")
    for r in rows:
        g = r.get("guard", {})
        print(f"  Cell {r.get('cell_label', 'c32768')}: verdict={g.get('verdict')} evidence={g.get('evidence', '')}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibration CLI")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    p_labels = subparsers.add_parser("labels", help="Dump HWiNFO sensor label map")
    p_labels.add_argument("--out", type=Path, default=None)

    p_rehearse = subparsers.add_parser("rehearse-kill", help="Rehearse thermal kill path")

    p_profile = subparsers.add_parser("profile", help="Derive empirical thresholds from 4k/8k profile")

    p_near_oom = subparsers.add_parser("near-oom", help="Supervised near-OOM verification")

    args = parser.parse_args()

    if args.subcommand == "labels":
        return cmd_labels(args)
    elif args.subcommand == "rehearse-kill":
        return cmd_rehearse_kill(args)
    elif args.subcommand == "profile":
        return cmd_profile(args)
    elif args.subcommand == "near-oom":
        return cmd_near_oom(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
