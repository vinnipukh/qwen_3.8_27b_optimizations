#!/usr/bin/env python3
"""Native Windows Vulkan benchmark session driver (D2-02, D2-03, BENCH-04).

Executes Vulkan benchmark matrix against native Windows build at pinned commit bb4caa75,
reusing the locked wave-1 harness contracts (llabench, fingerprint, store, guard, toast).
"""
from __future__ import annotations

import argparse
import datetime
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

DEFAULT_VULKAN_BIN = r"E:\vulkan-arm\llama.cpp\build\bin\Release\llama-bench.exe"
DEFAULT_VULKAN_MODEL = r"\\wsl.localhost\Ubuntu-24.04\root\models\Qwen3.8-27B-Uncensored-IQ4_XS.gguf"


def run_vulkan_session(
    tiers: tuple[int, ...] = (4096, 8192, 16384, 32768),
    fa_seq: tuple[str, ...] = ("off", "on"),
    repeats: int = 5,
    delay_s: int = 30,
    threads: int | None = None,
    bin_path: str = DEFAULT_VULKAN_BIN,
    model_path: str = DEFAULT_VULKAN_MODEL,
    out_dir: Path | None = None,
    smoke: bool = False,
) -> tuple[store.RunStore, int, int]:
    """Execute guarded, fingerprinted native Vulkan benchmark session."""
    if smoke:
        tiers = (1024,)
        fa_seq = ("off",)
        repeats = 1
        delay_s = 0

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    label = "smoke_vulkan" if smoke else "baseline_vulkan"
    run_store = store.RunStore.create(run_id=f"{ts}_{label}") if out_dir is None else store.RunStore(out_dir)

    print(f"=== Starting Native Vulkan Benchmark Session: {run_store.run_dir.name} ===")
    print(f"Binary: {bin_path}")
    print(f"Model: {model_path}")
    print(f"Tiers: {tiers}, fa_seq: {fa_seq}, repeats: {repeats}")

    if not Path(bin_path).exists():
        print(f"Notice: Vulkan binary not found at {bin_path}. Please build via benchmarks/vulkan/build-vulkan-arm.ps1", file=sys.stderr)
        return run_store, 0, 0

    # Load thresholds (observe-only if absent)
    thresholds = guard.Thresholds.from_json()
    observe_only = (thresholds is None)

    # 1. Collect manifest (Vulkan arm degradation)
    telemetry_mode = "absent"
    manifest = fingerprint.collect_manifest(
        run_dir=run_store.run_dir,
        backend_arm="Vulkan",
        telemetry_mode=telemetry_mode,
        bin_path=bin_path,
        model_path=model_path,
    )

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
            print(f"\n--- Vulkan Context Tier: {tier} ---")

            raw_stdout = run_store.run_dir / f"rows.raw.c{tier}.jsonl"
            stderr_log = run_store.logs_dir / f"bench_c{tier}.txt"

            stop_guard_event = threading.Event()
            rss_result: list[guard.RssProfile] = []

            def guard_thread_fn(target_pid: int):
                prof = guard.poll_proc(target_pid, stop_guard_event, interval_s=1.0)
                rss_result.append(prof)

            f_out = open(raw_stdout, "wb")
            f_err = open(stderr_log, "wb")

            proc = subprocess.Popen(
                plan.argv,
                stdout=f_out,
                stderr=f_err,
            )

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

            if rc != 0:
                print(f"Error: Vulkan tier {tier} failed with exit code {rc}", file=sys.stderr)
                for cell in plan.expected_cells:
                    fail_row = {
                        "backend_arm": "Vulkan",
                        "run_id": run_store.run_dir.name,
                        "n_prompt": cell.get("n_prompt", tier),
                        "n_gen": cell.get("n_gen", 0),
                        "flash_attn": cell.get("flash_attn", 0),
                        "type": cell.get("type", "pp"),
                        "avg_ts": 0.0,
                        "stddev_ts": 0.0,
                        "samples_ts": [],
                        "guard": {
                            "verdict": f"FAILED:exit_{rc}",
                            "vmrss_peak_kb": rss_prof.vmrss_peak_kb,
                            "vmswap_peak_kb": rss_prof.vmswap_peak_kb,
                        },
                        "telemetry_slice": None,
                        "timestamp": fingerprint.utc_now_iso(),
                    }
                    run_store.append_row(fail_row)
                    failed_count += 1
                continue

            rows = llabench.parse_rows(raw_stdout)
            try:
                llabench.assert_tier_rows(rows, plan)
            except llabench.MatrixContaminationError as exc:
                print(f"Contamination error in Vulkan tier {tier}: {exc}", file=sys.stderr)
                for r in rows:
                    r["backend_arm"] = "Vulkan"
                    r["run_id"] = run_store.run_dir.name
                    r["guard"] = {"verdict": "FAILED:matrix-contamination", "error": str(exc)}
                    run_store.append_row(r)
                    failed_count += 1
                continue

            for r in rows:
                rep_samples = r.get("samples_ts", [])
                g_verdict = guard.evaluate(
                    rss_profile=rss_prof,
                    repeat_means=rep_samples,
                    thresholds=thresholds,
                    observe_only=observe_only,
                )
                cell_label = f"c{tier}_{('pp' if r.get('n_gen')==0 else 'tg')}_fa_{('on' if r.get('flash_attn') in (1,True,'on') else 'off')}"

                r["backend_arm"] = "Vulkan"
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
                r["telemetry_slice"] = f"telemetry/vulkan_c{tier}.csv"

                run_store.append_row(r)

                if g_verdict.verdict == guard.VERDICT_OK:
                    ok_count += 1
                    print(f"  [Vulkan {cell_label}] {r.get('avg_ts', 0):.2f} tok/s ±{r.get('stddev_ts', 0):.2f} -> {g_verdict.verdict}")
                else:
                    failed_count += 1
                    print(f"  [Vulkan {cell_label}] {r.get('avg_ts', 0):.2f} tok/s -> {g_verdict.verdict}")

    finally:
        chk_file = run_store.write_checksums()
        index_entry = run_store.index_entry(ok_count=ok_count, failed_count=failed_count, backend_arm="Vulkan")
        index_file = Path("benchmarks/results/index.jsonl")
        with open(index_file, "a", encoding="utf-8") as f_idx:
            f_idx.write(json.dumps(index_entry) + "\n")

    print(f"\n=== Vulkan Session Closed: {ok_count} OK / {failed_count} FAILED ===")
    print(f"Checksums: {chk_file}")
    toast.send_summary(ok_count, failed_count)
    return run_store, ok_count, failed_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Vulkan Benchmark Session")
    parser.add_argument("--tiers", type=int, nargs="+", default=[4096, 8192, 16384, 32768])
    parser.add_argument("--fa-seq", type=str, nargs="+", default=["off", "on"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--delay", type=int, default=30)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--bin", type=str, default=DEFAULT_VULKAN_BIN)
    parser.add_argument("--model", type=str, default=DEFAULT_VULKAN_MODEL)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run tiny smoke session")
    args = parser.parse_args()

    run_vulkan_session(
        tiers=tuple(args.tiers),
        fa_seq=tuple(args.fa_seq),
        repeats=args.repeats,
        delay_s=args.delay,
        threads=args.threads,
        bin_path=args.bin,
        model_path=args.model,
        out_dir=args.out_dir,
        smoke=args.smoke,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
