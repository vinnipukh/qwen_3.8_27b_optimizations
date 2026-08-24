"""Thermal watchdog daemon with cross-boundary process kill (D2-20, BENCH-02).

Monitors GPU junction/hotspot temperature via HWiNFO shared memory. If temperature
reaches or exceeds threshold (default 95.0 °C), safely kills running benchmark process
via validated cross-boundary wsl.exe or native taskkill command.
NO software fan control is performed (record and abort only).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from benchmarks.lib import toast


def build_kill_command(pid: int | str, mode: str = "wsl") -> str:
    """Construct verified process kill command string with integer-validated PID."""
    # ASVS V5 validation: pid must be a valid integer
    try:
        pid_int = int(str(pid).strip())
        if pid_int <= 0:
            raise ValueError("PID must be positive")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid PID for kill command: {pid!r}") from exc

    if mode.lower() == "wsl":
        return f"wsl.exe -d Ubuntu-24.04 -u root -- sh -c 'kill -9 {pid_int}'"
    elif mode.lower() == "native":
        return f"taskkill /PID {pid_int} /F"
    else:
        raise ValueError(f"Unknown kill mode: {mode}")


def read_hotspot_temp_from_shmem() -> float | None:
    """Read current GPU hotspot/junction temperature from HWiNFO SM2 on Windows."""
    try:
        import mmap
        from benchmarks.host.hwinfo_daemon import (
            HWIS_SM2_MAP_NAME,
            parse_header,
            iter_readings,
            match_labels,
        )
        shm = mmap.mmap(-1, 256 * 1024, tagname=HWIS_SM2_MAP_NAME, access=mmap.ACCESS_READ)
        buf = shm.read(256 * 1024)
        shm.close()

        hdr = parse_header(buf)
        if not hdr["is_valid"] or hdr["is_dead"]:
            return None

        readings = iter_readings(buf, hdr)
        matched = match_labels(readings)
        return matched.get("temp_hotspot_c")
    except Exception:
        return None


def execute_kill(
    pid: int,
    mode: str,
    pid_file: Path,
    dry_run: bool = False,
    allow_terminate: bool = False,
) -> bool:
    """Execute kill command and write FAILED:thermal-abort verdict marker."""
    cmd_str = build_kill_command(pid, mode=mode)

    if dry_run:
        print(f"[DRY-RUN] Thermal kill triggered for PID {pid}: {cmd_str}")
        return True

    print(f"[CRITICAL] Thermal threshold exceeded! Executing: {cmd_str}", file=sys.stderr)
    try:
        res = subprocess.run(cmd_str, shell=True, capture_output=True, text=True, timeout=10)
        kill_ok = (res.returncode == 0)
    except Exception as exc:
        print(f"Error executing kill command: {exc}", file=sys.stderr)
        kill_ok = False

    # Marker file
    marker_file = pid_file.parent / f"{pid_file.stem}.verdict"
    try:
        with open(marker_file, "w", encoding="utf-8") as f_m:
            f_m.write("FAILED:thermal-abort\n")
    except OSError:
        pass

    # Toast alert
    toast.send(
        "Thermal Watchdog Abort",
        f"GPU Junction temp exceeded threshold! Terminated PID {pid} ({mode}).",
    )

    if not kill_ok and allow_terminate and mode == "wsl":
        print("[LAST RESORT] Terminating Ubuntu-24.04 WSL instance...", file=sys.stderr)
        subprocess.run(["wsl.exe", "--terminate", "Ubuntu-24.04"])

    return kill_ok


def run_watchdog(
    pid_file: Path,
    threshold_c: float = 95.0,
    poll_s: float = 2.0,
    kill_mode: str = "wsl",
    dry_run: bool = False,
    allow_terminate: bool = False,
) -> int:
    """Main watchdog polling loop."""
    print(f"Starting Thermal Watchdog: threshold={threshold_c}°C, mode={kill_mode}, poll={poll_s}s, pid_file={pid_file}")

    while True:
        if not pid_file.exists():
            time.sleep(poll_s)
            continue

        try:
            pid_content = pid_file.read_text(encoding="utf-8").strip()
            if not pid_content:
                time.sleep(poll_s)
                continue
            target_pid = int(pid_content)
        except (ValueError, OSError):
            time.sleep(poll_s)
            continue

        hotspot = read_hotspot_temp_from_shmem()
        if hotspot is not None:
            if hotspot >= threshold_c:
                execute_kill(
                    pid=target_pid,
                    mode=kill_mode,
                    pid_file=pid_file,
                    dry_run=dry_run,
                    allow_terminate=allow_terminate,
                )
                return 0

        time.sleep(poll_s)


def main() -> int:
    parser = argparse.ArgumentParser(description="Thermal Watchdog Service")
    parser.add_argument("--pid-file", type=Path, required=True, help="Path to PID file to monitor")
    parser.add_argument("--threshold-c", type=float, default=95.0, help="Junction temp kill threshold in °C")
    parser.add_argument("--poll-s", type=float, default=2.0, help="Poll interval in seconds")
    parser.add_argument("--kill-mode", choices=["wsl", "native"], default="wsl", help="Execution environment")
    parser.add_argument("--dry-run", action="store_true", help="Print kill command without executing")
    parser.add_argument("--allow-terminate", action="store_true", help="Allow wsl --terminate as last resort")
    args = parser.parse_args()

    if args.dry_run and not args.pid_file.exists():
        # Handle dry run when pid file doesn't exist yet
        print(f"[DRY-RUN READY] threshold={args.threshold_c}°C cmd={build_kill_command(4242, mode=args.kill_mode)}")
        return 0

    return run_watchdog(
        pid_file=args.pid_file,
        threshold_c=args.threshold_c,
        poll_s=args.poll_s,
        kill_mode=args.kill_mode,
        dry_run=args.dry_run,
        allow_terminate=args.allow_terminate,
    )


if __name__ == "__main__":
    sys.exit(main())
