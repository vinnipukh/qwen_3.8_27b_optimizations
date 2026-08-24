#!/usr/bin/env python3
"""Layer-2 greedy prompt runner (D2-05 layer 2).

Executes deterministic prompt corpus through pinned llama-cli with --temp 0,
--single-turn, --simple-io, and --load-mode none. Enriched rows are fsynced
per prompt execution into rows.jsonl.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

DEFAULT_BIN = "/root/llama.cpp/build-ci/bin/llama-cli"
DEFAULT_MODEL = "/root/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf"
DEFAULT_CORPUS_DIR = "benchmarks/prompts"
DEFAULT_GEN = 128
DEFAULT_TIER = 4096


def sha256_file(path: str | Path) -> str:
    """Compute sha256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def run_prompt(
    corpus_file: Path,
    tier: int,
    gen: int,
    bin_path: Path,
    model_path: Path,
    logs_dir: Path,
    pid_file: Path,
) -> dict[str, Any]:
    """Execute a single prompt through llama-cli with setsid detachment."""
    stem = corpus_file.stem
    out_path = logs_dir / f"{stem}.out"
    err_path = logs_dir / f"{stem}.err"
    prompt_hash = sha256_file(corpus_file)

    argv = [
        str(bin_path),
        "-m", str(model_path),
        "-c", str(tier),
        "--temp", "0",
        "--single-turn",
        "--simple-io",
        "--load-mode", "none",
        "-ngl", "99",
        "-f", str(corpus_file),
        "-n", str(gen),
    ]

    bin_dir = str(bin_path.parent)
    env = dict(os.environ)
    curr_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{bin_dir}:{curr_ld}" if curr_ld else bin_dir

    start_t = time.perf_counter()
    start_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

    with open(out_path, "wb") as f_out, open(err_path, "wb") as f_err:
        proc = subprocess.Popen(
            argv,
            stdout=f_out,
            stderr=f_err,
            env=env,
            start_new_session=True,  # setsid detachment
        )
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        with open(pid_file, "w", encoding="utf-8") as f_pid:
            f_pid.write(f"{proc.pid}\n")

        rc = proc.wait()
        f_out.flush()
        f_err.flush()

    if pid_file.exists():
        try:
            pid_file.unlink()
        except OSError:
            pass

    wall_time = time.perf_counter() - start_t
    output_len = out_path.stat().st_size if out_path.exists() else 0

    return {
        "prompt": corpus_file.name,
        "prompt_sha256": prompt_hash,
        "tier": tier,
        "n_gen": gen,
        "returncode": rc,
        "wall_time_s": round(wall_time, 4),
        "output_bytes": output_len,
        "stdout_file": str(out_path),
        "stderr_file": str(err_path),
        "timestamp": start_utc,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Layer-2 Greedy Prompt Runner")
    parser.add_argument("--corpus-dir", type=Path, default=Path(DEFAULT_CORPUS_DIR))
    parser.add_argument("--tier", type=int, default=DEFAULT_TIER)
    parser.add_argument("--bin", type=Path, default=Path(DEFAULT_BIN))
    parser.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--gen", type=int, default=DEFAULT_GEN)
    parser.add_argument("--smoke", action="store_true", help="Run only shortest prompt for smoke verification")
    args = parser.parse_args()

    # Determine run directory
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir if args.out_dir else Path(f"benchmarks/results/{ts}_layer2_prompts")
    logs_dir = run_dir / "logs"
    telemetry_dir = run_dir / "telemetry"
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    telemetry_dir.mkdir(parents=True, exist_ok=True)

    # Write manifest placeholder
    manifest_file = run_dir / "manifest.json"
    if not manifest_file.exists():
        with open(manifest_file, "w", encoding="utf-8") as f_m:
            json.dump({
                "layer": 2,
                "tier": args.tier,
                "bin": str(args.bin),
                "model": str(args.model),
                "timestamp": ts,
            }, f_m, indent=2)

    pid_file = run_dir / "run" / "current.pid"
    rows_file = run_dir / "rows.jsonl"

    corpus_files = sorted(args.corpus_dir.glob("*.txt"))
    if not corpus_files:
        print(f"Error: No *.txt prompt files found in {args.corpus_dir}", file=sys.stderr)
        return 1

    if args.smoke:
        # Run only the shortest prompt
        shortest = min(corpus_files, key=lambda p: p.stat().st_size)
        corpus_files = [shortest]

    print(f"Starting Layer-2 prompt runner: {len(corpus_files)} prompts, tier={args.tier}, gen={args.gen}")
    all_ok = True

    with open(rows_file, "a", encoding="utf-8") as f_rows:
        for p_file in corpus_files:
            print(f"  Executing prompt: {p_file.name} ... ", end="", flush=True)
            row = run_prompt(
                corpus_file=p_file,
                tier=args.tier,
                gen=args.gen,
                bin_path=args.bin,
                model_path=args.model,
                logs_dir=logs_dir,
                pid_file=pid_file,
            )
            f_rows.write(json.dumps(row) + "\n")
            f_rows.flush()
            os.fsync(f_rows.fileno())

            if row["returncode"] == 0:
                print(f"OK ({row['wall_time_s']}s, {row['output_bytes']} bytes)")
            else:
                print(f"FAILED (rc={row['returncode']}, {row['wall_time_s']}s)")
                all_ok = False

    if args.smoke and all_ok:
        sentinel = Path("benchmarks/prompts/.smoke_out_check")
        with open(sentinel, "w", encoding="utf-8") as f_s:
            f_s.write(f"run_dir={run_dir}\ntier={args.tier}\nstatus=OK\n")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
