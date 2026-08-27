#!/usr/bin/env python3
"""Session orchestrator for Qwen3.8-27B benchmark harness (D2-15, D2-16, BENCH-01, BENCH-03).

Executes fingerprinted, guarded, pre-flighted, and checksummed benchmark sessions
across ascending context tiers with continuous telemetry and crash resilience.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.lib import fingerprint, guard, llabench, preflight, store, toast

LOCK_FILE = "benchmarks/results/.session.lock"
FROZEN_FREE_MIB_ANCHOR = 18245.0  # Phase-1 startup log anchor


def acquire_session_lock(lock_path: str = LOCK_FILE) -> Any:
    """Acquire non-blocking flock on lock file. Exit 5 if locked."""
    p = Path(lock_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    f = open(p, "w")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        f.write(f"pid={os.getpid()}\nutc={datetime.datetime.now(datetime.timezone.utc).isoformat()}\n")
        f.flush()
        return f
    except (BlockingIOError, OSError):
        print(f"Error: Another benchmark session holds lock {lock_path}", file=sys.stderr)
        sys.exit(5)


def release_session_lock(lock_file: Any) -> None:
    """Release flock and close lock file descriptor."""
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()
    except Exception:
        pass


def run_session(
    tiers: tuple[int, ...] = (4096, 8192, 16384, 32768),
    fa_seq: tuple[str, ...] = ("off", "on"),
    repeats: int = 5,
    delay_s: int = 30,
    threads: int | None = None,
    bin_path: str = llabench.BIN_PATH,
    model_path: str = llabench.MODEL_PATH,
    out_dir: Path | None = None,
    smoke: bool = False,
    observe_only: bool = False,
) -> tuple[store.RunStore, int, int]:
    """Execute complete benchmark session."""
    lock = acquire_session_lock()

    if smoke:
        tiers = (1024,)
        fa_seq = ("off",)
        repeats = 1
        delay_s = 0

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    label = "smoke_hip" if smoke else "baseline_hip"
    run_store = store.RunStore.create(run_id=f"{ts}_{label}") if out_dir is None else store.RunStore(out_dir)

    print(f"=== Starting Benchmark Session: {run_store.run_dir.name} ===")
    print(f"Tiers: {tiers}, fa_seq: {fa_seq}, repeats: {repeats}, delay: {delay_s}s")

    # Load thresholds (from calibration if available)
    thresholds = guard.Thresholds.from_json()
    if thresholds is None or observe_only:
        print("Notice: thresholds.json absent or observe_only mode -> running guard in observe-only mode")
        observe_only = True

    # 1. Collect fingerprint manifest
    telemetry_mode = "absent"
    manifest = fingerprint.collect_manifest(
        run_dir=run_store.run_dir,
        backend_arm="HIP",
        telemetry_mode=telemetry_mode,
        bin_path=bin_path,
        model_path=model_path,
    )

    # Spawn host-side telemetry/watchdog if available
    pid_file = run_store.run_dir / "run" / "current.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)

    tier_plans = llabench.enumerate_tiers(
        tiers=tiers,
        fa_seq=fa_seq,
        repeats=repeats,
        delay_s=delay_s,
        threads=threads,
        bin_path=bin_path,
        model_path=model_path,
    )

    ok_count = 0
    failed_count = 0

    try:
        for plan in tier_plans:
            tier = plan.tier
            print(f"\n--- Context Tier: {tier} ---")

            # Pre-flight check
            needed_mib = preflight.estimate_needed_mib(tier)
            # Use frozen DXG free VRAM anchor or probe
            pf_verdict = preflight.check(needed_mib, FROZEN_FREE_MIB_ANCHOR)
            print(f"Pre-flight verdict: {pf_verdict.verdict} ({pf_verdict.evidence})")

            # If tier is 32768 and preflight failed -> write expected failure rows without allocating
            if tier >= 32768 and pf_verdict.verdict != "PASS":
                print(f"Tier {tier} exceeded pre-flight allocation limit. Publishing {preflight.VERDICT_PREFLIGHT} rows.")
                for cell in plan.expected_cells:
                    failed_row = {
                        "backend_arm": "HIP",
                        "run_id": run_store.run_dir.name,
                        "n_prompt": cell.get("n_prompt", tier),
                        "n_gen": cell.get("n_gen", 0),
                        "flash_attn": cell.get("flash_attn", 0),
                        "type": cell.get("type", "pp"),
                        "avg_ts": 0.0,
                        "stddev_ts": 0.0,
                        "samples_ts": [],
                        "guard": {
                            "verdict": preflight.VERDICT_PREFLIGHT,
                            "evidence": pf_verdict.evidence,
                            "flags": pf_verdict.flags,
                        },
                        "telemetry_slice": None,
                        "timestamp": fingerprint.utc_now_iso(),
                    }
                    run_store.append_row(failed_row)
                    failed_count += 1
                toast.send("Pre-flight OOM Prevented", f"Tier {tier} skipped to prevent hard crash.")
                continue

            # Run tier invocation
            raw_stdout = run_store.run_dir / f"rows.raw.c{tier}.jsonl"
            stderr_log = run_store.logs_dir / f"bench_c{tier}.txt"

            # Background thread to monitor RSS during execution
            stop_guard_event = threading.Event()
            rss_result: list[guard.RssProfile] = []

            def guard_thread_fn(target_pid: int):
                prof = guard.poll_proc(target_pid, stop_guard_event, interval_s=1.0)
                rss_result.append(prof)

            # Spawn child with setsid detachment
            bin_dir = str(Path(bin_path).parent)
            env = dict(os.environ)
            env["HSA_ENABLE_DXG_DETECTION"] = "1"
            curr_ld = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{bin_dir}:{curr_ld}" if curr_ld else bin_dir

            f_out = open(raw_stdout, "wb")
            f_err = open(stderr_log, "wb")

            proc = subprocess.Popen(
                plan.argv,
                stdout=f_out,
                stderr=f_err,
                env=env,
                start_new_session=True,
            )

            # Write PID file for thermal watchdog and watchers
            pid_file.write_text(f"{proc.pid}\n", encoding="utf-8")

            g_thread = threading.Thread(target=guard_thread_fn, args=(proc.pid,))
            g_thread.start()

            rc = proc.wait()
            stop_guard_event.set()
            g_thread.join()

            f_out.close()
            f_err.close()

            if pid_file.exists():
                try:
                    pid_file.unlink()
                except OSError:
                    pass

            rss_prof = rss_result[0] if rss_result else guard.RssProfile(pid=proc.pid)

            # Parse stderr buffer sizes for ledger
            stderr_content = stderr_log.read_text(encoding="utf-8", errors="replace") if stderr_log.exists() else ""
            buffers = preflight.parse_buffer_lines(stderr_content)

            if rc != 0:
                print(f"Error: Tier {tier} process failed with exit code {rc}", file=sys.stderr)
                # Check for thermal abort marker
                thermal_marker = run_store.run_dir / "run" / "current.verdict"
                verdict_str = guard.VERDICT_THERMAL if (thermal_marker.exists() and "thermal" in thermal_marker.read_text()) else f"FAILED:exit_{rc}"
                for cell in plan.expected_cells:
                    fail_row = {
                        "backend_arm": "HIP",
                        "run_id": run_store.run_dir.name,
                        "n_prompt": cell.get("n_prompt", tier),
                        "n_gen": cell.get("n_gen", 0),
                        "flash_attn": cell.get("flash_attn", 0),
                        "type": cell.get("type", "pp"),
                        "avg_ts": 0.0,
                        "stddev_ts": 0.0,
                        "samples_ts": [],
                        "guard": {
                            "verdict": verdict_str,
                            "vmrss_peak_kb": rss_prof.vmrss_peak_kb,
                            "vmswap_peak_kb": rss_prof.vmswap_peak_kb,
                        },
                        "telemetry_slice": None,
                        "timestamp": fingerprint.utc_now_iso(),
                    }
                    run_store.append_row(fail_row)
                    failed_count += 1
                toast.send("Tier Execution Failed", f"Tier {tier} exited with code {rc}")
                continue

            # Parse rows
            rows = llabench.parse_rows(raw_stdout)
            try:
                llabench.assert_tier_rows(rows, plan)
            except llabench.MatrixContaminationError as exc:
                print(f"CRITICAL: Integrity violation in tier {tier}: {exc}", file=sys.stderr)
                # Fail-fast: record contaminated rows with failed verdict
                for r in rows:
                    r["backend_arm"] = "HIP"
                    r["run_id"] = run_store.run_dir.name
                    r["guard"] = {"verdict": "FAILED:matrix-contamination", "error": str(exc)}
                    run_store.append_row(r)
                    failed_count += 1
                continue

            # Evaluate guard against rows
            for r in rows:
                rep_samples = r.get("samples_ts", [])
                g_verdict = guard.evaluate(
                    rss_profile=rss_prof,
                    repeat_means=rep_samples,
                    thresholds=thresholds,
                    observe_only=observe_only,
                )

                cell_label = f"c{tier}_{('pp' if r.get('n_gen')==0 else 'tg')}_fa_{('on' if r.get('flash_attn') in (1,True,'on') else 'off')}"

                r["backend_arm"] = "HIP"
                r["run_id"] = run_store.run_dir.name
                r["tier"] = tier
                r["cell_label"] = cell_label
                r["guard"] = {
                    "verdict": g_verdict.verdict,
                    "signals": g_verdict.signals,
                    "flagged_for_review": g_verdict.flagged_for_review,
                    "vmrss_peak_kb": rss_prof.vmrss_peak_kb,
                    "vmswap_peak_kb": rss_prof.vmswap_peak_kb,
                }
                r["telemetry_slice"] = f"telemetry/c{tier}.csv"

                run_store.append_row(r)

                # Record VRAM ledger entry
                preflight.record_ledger(
                    run_store=run_store,
                    cell_label=cell_label,
                    buffers=buffers,
                    guard_peaks={"vmrss_peak_kb": rss_prof.vmrss_peak_kb, "vmswap_peak_kb": rss_prof.vmswap_peak_kb},
                    dxg_line=f"(20421 MiB, {FROZEN_FREE_MIB_ANCHOR} MiB free)",
                )

                if g_verdict.verdict == guard.VERDICT_OK:
                    ok_count += 1
                    print(f"  [{cell_label}] {r.get('avg_ts', 0):.2f} tok/s ±{r.get('stddev_ts', 0):.2f} -> {g_verdict.verdict}")
                else:
                    failed_count += 1
                    print(f"  [{cell_label}] {r.get('avg_ts', 0):.2f} tok/s -> {g_verdict.verdict} (tripped: {g_verdict.signals.get('tripped')})")
                    toast.send("Guard Trip Alert", f"Cell {cell_label} marked {g_verdict.verdict}")

    finally:
        # Close session
        chk_file = run_store.write_checksums()
        index_entry = run_store.index_entry(ok_count=ok_count, failed_count=failed_count, backend_arm="HIP")
        index_file = Path("benchmarks/results/index.jsonl")
        with open(index_file, "a", encoding="utf-8") as f_idx:
            f_idx.write(json.dumps(index_entry) + "\n")

        release_session_lock(lock)

    print(f"\n=== Session Closed: {ok_count} OK / {failed_count} FAILED ===")
    print(f"Checksums: {chk_file}")
    toast.send_summary(ok_count, failed_count)

    return run_store, ok_count, failed_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Benchmark Session")
    parser.add_argument("--tiers", type=int, nargs="+", default=[4096, 8192, 16384, 32768])
    parser.add_argument("--fa-seq", type=str, nargs="+", default=["off", "on"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--delay", type=int, default=30)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--bin", type=str, default=llabench.BIN_PATH)
    parser.add_argument("--model", type=str, default=llabench.MODEL_PATH)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run tiny smoke session")
    parser.add_argument("--observe-only", action="store_true", help="Force observe-only guard mode")
    args = parser.parse_args()

    store_res, ok_c, fail_c = run_session(
        tiers=tuple(args.tiers),
        fa_seq=tuple(args.fa_seq),
        repeats=args.repeats,
        delay_s=args.delay,
        threads=args.threads,
        bin_path=args.bin,
        model_path=args.model,
        out_dir=args.out_dir,
        smoke=args.smoke,
        observe_only=args.observe_only,
    )
    return 0 if (ok_c > 0 and fail_c == 0) or args.smoke else (0 if ok_c > 0 else 1)


if __name__ == "__main__":
    sys.exit(main())
