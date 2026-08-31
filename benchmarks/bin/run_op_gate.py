#!/usr/bin/env python3
"""Op-Level Correctness Gate runner (QUAL-01, Plan 03-01).

Executes `test-backend-ops test -b ROCm0 --output csv` under WSL2 ROCm0,
asserts zero errors across all operations, verifies core hybrid architecture ops,
and emits structured JSON gate results.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

DEFAULT_BIN = "/root/llama.cpp/build-ci/bin/test-backend-ops"
DEFAULT_BACKEND = "ROCm0"
DEFAULT_OUT_JSON = "benchmarks/results/phase3/op_gate.json"

CORE_HYBRID_OPS = [
    "GATED_DELTA_NET",
    "SOLVE_TRI",
    "SSM_CONV",
    "SSM_SCAN",
    "FLASH_ATTN_EXT",
    "MUL_MAT",
]


def run_op_gate(
    backend: str = DEFAULT_BACKEND,
    bin_path: str = DEFAULT_BIN,
    ops: list[str] | None = None,
    out_json: Path | str = DEFAULT_OUT_JSON,
    strict_core_ops: list[str] | None = None,
    mock_csv: str | None = None,
    mock_exit_code: int = 0,
) -> dict[str, Any]:
    """Execute test-backend-ops and parse CSV output with strict validation."""
    if strict_core_ops is None:
        strict_core_ops = CORE_HYBRID_OPS

    start_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    raw_csv = ""
    stderr_out = ""
    exit_code = 0

    if mock_csv is not None:
        raw_csv = mock_csv
        exit_code = mock_exit_code
    else:
        # Build command
        cmd: list[str] = [bin_path, "test", "-b", backend, "--output", "csv"]
        if ops:
            cmd.extend(["-o", ",".join(ops)])

        env = os.environ.copy()
        env["HSA_ENABLE_DXG_DETECTION"] = "1"
        bin_dir = str(Path(bin_path).parent)
        existing_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{bin_dir}:{existing_ld}" if existing_ld else bin_dir

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        raw_csv = proc.stdout
        stderr_out = proc.stderr
        exit_code = proc.returncode

    # Filter non-CSV lines (e.g. debug/warmup messages) from raw_csv
    clean_csv_lines: list[str] = []
    header_found = False
    for line in raw_csv.splitlines():
        line_str = line.strip()
        if not line_str:
            continue
        if '"backend_name","op_name"' in line_str:
            header_found = True
            clean_csv_lines.append(line_str)
        elif header_found and line_str.startswith('"'):
            clean_csv_lines.append(line_str)

    clean_csv = "\n".join(clean_csv_lines)
    reader = csv.DictReader(io.StringIO(clean_csv))
    rows = list(reader)

    total_cases = len(rows)
    supported_cases = 0
    unsupported_cases = 0
    error_cases = 0
    errors: list[dict[str, Any]] = []

    op_summary: dict[str, dict[str, int]] = {}

    for row in rows:
        op = row.get("op_name", "UNKNOWN")
        if op not in op_summary:
            op_summary[op] = {
                "total": 0,
                "supported": 0,
                "unsupported": 0,
                "errors": 0,
            }
        op_summary[op]["total"] += 1

        supported_val = row.get("supported", "0")
        err_msg = row.get("error_message", "").strip()

        if supported_val == "1":
            supported_cases += 1
            op_summary[op]["supported"] += 1
        else:
            unsupported_cases += 1
            op_summary[op]["unsupported"] += 1

        if err_msg and err_msg != "not supported":
            error_cases += 1
            op_summary[op]["errors"] += 1
            errors.append({
                "op": op,
                "params": row.get("op_params", ""),
                "error": err_msg,
            })

    # Validate core operations
    core_status: dict[str, dict[str, Any]] = {}
    core_all_pass = True

    # Only enforce strict core ops check if ops filter is None or contains them
    for cop in strict_core_ops:
        if ops is not None and cop not in ops:
            continue
        cop_stat = op_summary.get(cop)
        if not cop_stat or cop_stat["supported"] == 0 or cop_stat["errors"] > 0:
            core_all_pass = False
            core_status[cop] = {
                "status": "FAIL",
                "total": cop_stat["total"] if cop_stat else 0,
                "supported": cop_stat["supported"] if cop_stat else 0,
                "errors": cop_stat["errors"] if cop_stat else 0,
            }
        else:
            core_status[cop] = {
                "status": "PASS",
                "total": cop_stat["total"],
                "supported": cop_stat["supported"],
                "errors": cop_stat["errors"],
            }

    overall_pass = (exit_code == 0) and (error_cases == 0) and core_all_pass and (total_cases > 0)
    status_str = "PASS" if overall_pass else "FAIL"

    end_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

    result: dict[str, Any] = {
        "timestamp_start": start_utc,
        "timestamp_end": end_utc,
        "status": status_str,
        "exit_code": exit_code,
        "backend": backend,
        "bin_path": bin_path,
        "total_cases": total_cases,
        "supported_cases": supported_cases,
        "unsupported_cases": unsupported_cases,
        "error_cases": error_cases,
        "unique_ops_tested": len(op_summary),
        "core_ops_status": core_status,
        "errors": errors[:50],  # cap recorded errors
        "op_summary": op_summary,
    }

    out_p = Path(out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Op-Level Correctness Gate runner (QUAL-01) N=10 rigour")
    parser.add_argument("--backend", default=DEFAULT_BACKEND, help="Backend to test (default: ROCm0)")
    parser.add_argument("--bin-path", default=DEFAULT_BIN, help="Path to test-backend-ops binary")
    parser.add_argument("--ops", default=None, help="Comma-separated op list (default: all)")
    parser.add_argument("--out-json", default=DEFAULT_OUT_JSON, help="Path for JSON output")
    parser.add_argument("--runs", type=int, default=1, help="Repeats N=10 for REQ-STAT-07 (QUAL-01 0 errors N=10)")
    # allow --runs 10 via help string with %% escape handled
    args = parser.parse_args()

    ops_list = [o.strip() for o in args.ops.split(",") if o.strip()] if args.ops else None

    # N=10 rigour: loop test-backend-ops --runs times, assert 0 errors each run (REQ-STAT-07)
    if args.runs == 10:
        print(f"=== QUAL-01: Running Op-Level Correctness Gate N=10 (0 errors) on {args.backend} ===")
        print(f"REQ-STAT-07: run_op_gate.py --runs 10 requires 0 errors in each of 10 repeats")
    else:
        print(f"=== QUAL-01: Running Op-Level Correctness Gate on {args.backend} ===")
    all_results = []
    for run_idx in range(args.runs):
        if args.runs > 1:
            print(f"[QUAL-01] run {run_idx+1}/{args.runs}")
        res = run_op_gate(
            backend=args.backend,
            bin_path=args.bin_path,
            ops=ops_list,
            out_json=args.out_json if args.runs == 1 else str(Path(args.out_json).with_suffix(f".run{run_idx+1}.json")),
        )
        all_results.append(res)
        if res["error_cases"] != 0:
            print(f"[QUAL-01] FAIL run {run_idx+1}: {res['error_cases']} errors", file=sys.stderr)
    # Aggregate verdict: all 10 runs must be 0 errors
    if args.runs > 1:
        total_errors = sum(r["error_cases"] for r in all_results)
        print(f"\n[QUAL-01 N={args.runs}] total_errors across {args.runs} runs: {total_errors} (require 0)")
        # Write aggregated JSON
        agg = {"runs": args.runs, "total_errors": total_errors, "per_run": all_results, "gate": "QUAL-01 N=10 0 errors"}
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2)
        print(f"Aggregated N={args.runs} results saved to: {args.out_json}")
        return 0 if total_errors == 0 and all(r["status"] == "PASS" for r in all_results) else 1

    res = all_results[0]
    print(f"Total test cases: {res['total_cases']}")
    print(f"Supported (passed): {res['supported_cases']}")
    print(f"Unsupported: {res['unsupported_cases']}")
    print(f"Errors: {res['error_cases']}")
    print(f"Unique ops: {res['unique_ops_tested']}")
    print("\nCore Hybrid Architecture Ops:")
    for cop, cstat in res["core_ops_status"].items():
        print(f"  - {cop:16s}: {cstat['status']} (supported: {cstat['supported']}, errors: {cstat['errors']})")

    print(f"\nGate Verdict: {res['status']}")
    print(f"Results saved to: {args.out_json}")

    return 0 if res["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
