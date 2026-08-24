#!/usr/bin/env python3
"""Model-Level Quality Gate runner (QUAL-02, Plan 03-02).

Evaluates:
1. WikiText-2 perplexity within +/- 1.0% tolerance of stock baseline reference.
2. Exact-match greedy canary verification across the 6 deterministic prompt corpus files.
Emits structured JSON results to `benchmarks/results/phase3/model_gate.json`.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

DEFAULT_MODEL = "/root/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf"
DEFAULT_PPL_BIN = "/root/llama.cpp/build-ci/bin/llama-perplexity"
DEFAULT_CLI_BIN = "/root/llama.cpp/build-ci/bin/llama-cli"
DEFAULT_WIKI_DATA = "benchmarks/data/wiki.test.raw"
DEFAULT_PROMPTS_DIR = "benchmarks/prompts"
DEFAULT_GOLDEN_PATH = "benchmarks/golden/stock_baseline_golden.json"
DEFAULT_OUT_JSON = "benchmarks/results/phase3/model_gate.json"

# Stock baseline reference PPL on full wiki.test.raw (145 chunks, ctx 2048)
STOCK_PPL_REFERENCE = 6.4271
PPL_TOLERANCE_PCT = 1.0  # +/- 1.0%


def sha256_file(path: str | Path) -> str:
    """Compute sha256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def extract_generation(raw_stdout: str) -> str:
    """Extract generated text from llama-cli simple-io output."""
    match = re.search(r"\[Start thinking\]\s*\n\n(.*?)(?=\n\n\[ Prompt:|\n\nExiting|\Z)", raw_stdout, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw_stdout.strip()


def run_perplexity(
    model_path: str = DEFAULT_MODEL,
    ppl_bin: str = DEFAULT_PPL_BIN,
    data_path: str = DEFAULT_WIKI_DATA,
    chunks: int | None = None,
    ctx_size: int = 2048,
    mock_output: str | None = None,
    mock_returncode: int = 0,
) -> dict[str, Any]:
    """Run llama-perplexity and parse the final PPL estimate."""
    if mock_output is not None:
        raw_out = mock_output
        rc = mock_returncode
    else:
        cmd = [
            ppl_bin,
            "-m", model_path,
            "-f", str(Path(data_path).resolve()),
            "-c", str(ctx_size),
            "-ngl", "99",
            "--load-mode", "none",
        ]
        if chunks is not None and chunks > 0:
            cmd.extend(["--chunks", str(chunks)])

        env = os.environ.copy()
        env["HSA_ENABLE_DXG_DETECTION"] = "1"
        bin_dir = str(Path(ppl_bin).parent)
        existing_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{bin_dir}:{existing_ld}" if existing_ld else bin_dir

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        raw_out = proc.stdout
        rc = proc.returncode

    # Parse PPL from output: "Final estimate: PPL = 6.4271 +/- 0.04103"
    match = re.search(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)\s*\+/-\s*([0-9.]+)", raw_out)
    if not match or rc != 0:
        return {
            "status": "FAIL",
            "returncode": rc,
            "ppl": None,
            "ppl_err": None,
            "raw_output": raw_out[-1000:],
        }

    ppl = float(match.group(1))
    ppl_err = float(match.group(2))
    return {
        "status": "PASS",
        "returncode": rc,
        "ppl": ppl,
        "ppl_err": ppl_err,
        "chunks": chunks if chunks else 145,
    }


def run_prompt_canaries(
    model_path: str = DEFAULT_MODEL,
    cli_bin: str = DEFAULT_CLI_BIN,
    prompts_dir: str = DEFAULT_PROMPTS_DIR,
    n_tokens: int = 32,
    ctx_size: int = 4096,
    mock_results: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run greedy decode across prompt corpus files and return extracted text."""
    p_dir = Path(prompts_dir)
    prompt_files = sorted(p_dir.glob("*.txt"))
    canaries: dict[str, dict[str, Any]] = {}

    env = os.environ.copy()
    env["HSA_ENABLE_DXG_DETECTION"] = "1"
    bin_dir = str(Path(cli_bin).parent)
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{bin_dir}:{existing_ld}" if existing_ld else bin_dir

    for pf in prompt_files:
        p_sha = sha256_file(pf)
        if mock_results is not None:
            gen_text = mock_results.get(pf.name, "")
            rc = 0
        else:
            cmd = [
                cli_bin,
                "-m", model_path,
                "--temp", "0",
                "--single-turn",
                "--simple-io",
                "--load-mode", "none",
                "-c", str(ctx_size),
                "-ngl", "99",
                "-f", str(pf.resolve()),
                "-n", str(n_tokens),
                "--log-disable",
            ]
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            rc = proc.returncode
            gen_text = extract_generation(proc.stdout)

        canaries[pf.name] = {
            "prompt_sha256": p_sha,
            "generated_text": gen_text,
            "output_sha256": hashlib.sha256(gen_text.encode("utf-8")).hexdigest(),
            "returncode": rc,
        }

    return canaries


def record_golden(
    model_path: str = DEFAULT_MODEL,
    ppl_bin: str = DEFAULT_PPL_BIN,
    cli_bin: str = DEFAULT_CLI_BIN,
    data_path: str = DEFAULT_WIKI_DATA,
    prompts_dir: str = DEFAULT_PROMPTS_DIR,
    golden_path: str = DEFAULT_GOLDEN_PATH,
    chunks: int | None = None,
) -> dict[str, Any]:
    """Capture and persist stock baseline golden metrics."""
    print("Recording golden perplexity baseline...")
    ppl_res = run_perplexity(
        model_path=model_path,
        ppl_bin=ppl_bin,
        data_path=data_path,
        chunks=chunks,
    )
    if ppl_res["status"] != "PASS" or ppl_res["ppl"] is None:
        raise RuntimeError(f"Failed to record baseline perplexity: {ppl_res}")

    print(f"Perplexity: {ppl_res['ppl']} +/- {ppl_res['ppl_err']}")
    print("Recording golden prompt canaries...")
    canaries = run_prompt_canaries(
        model_path=model_path,
        cli_bin=cli_bin,
        prompts_dir=prompts_dir,
    )

    golden_data = {
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_path": model_path,
        "dataset_path": data_path,
        "dataset_sha256": sha256_file(data_path),
        "perplexity": {
            "reference_ppl": ppl_res["ppl"],
            "ppl_stddev": ppl_res["ppl_err"],
            "chunks": ppl_res.get("chunks", 145),
            "allowed_tolerance_pct": PPL_TOLERANCE_PCT,
            "min_allowed_ppl": round(ppl_res["ppl"] * (1.0 - PPL_TOLERANCE_PCT / 100.0), 4),
            "max_allowed_ppl": round(ppl_res["ppl"] * (1.0 + PPL_TOLERANCE_PCT / 100.0), 4),
        },
        "canaries": canaries,
    }

    g_path = Path(golden_path)
    g_path.parent.mkdir(parents=True, exist_ok=True)
    with open(g_path, "w", encoding="utf-8") as f:
        json.dump(golden_data, f, indent=2)

    print(f"Golden baseline saved to: {golden_path}")
    return golden_data


def evaluate_model_gate(
    model_path: str = DEFAULT_MODEL,
    ppl_bin: str = DEFAULT_PPL_BIN,
    cli_bin: str = DEFAULT_CLI_BIN,
    data_path: str = DEFAULT_WIKI_DATA,
    prompts_dir: str = DEFAULT_PROMPTS_DIR,
    golden_path: str = DEFAULT_GOLDEN_PATH,
    out_json: str = DEFAULT_OUT_JSON,
    chunks: int | None = None,
    mock_ppl: float | None = None,
    mock_canaries: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Execute model gate validation against golden store."""
    g_path = Path(golden_path)
    if not g_path.exists():
        raise FileNotFoundError(f"Golden baseline file {golden_path} does not exist. Run with --record-golden first.")

    with open(g_path, "r", encoding="utf-8") as f:
        golden = json.load(f)

    start_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    ref_ppl = golden["perplexity"]["reference_ppl"]
    min_ppl = golden["perplexity"]["min_allowed_ppl"]
    max_ppl = golden["perplexity"]["max_allowed_ppl"]

    # 1. Perplexity check
    if mock_ppl is not None:
        ppl_val = mock_ppl
        ppl_status = "PASS" if (min_ppl <= ppl_val <= max_ppl) else "FAIL"
        ppl_res = {
            "status": ppl_status,
            "ppl": ppl_val,
            "ppl_err": 0.04,
            "chunks": chunks if chunks else 145,
        }
    else:
        ppl_res = run_perplexity(
            model_path=model_path,
            ppl_bin=ppl_bin,
            data_path=data_path,
            chunks=chunks,
        )
        if ppl_res["status"] == "PASS" and ppl_res["ppl"] is not None:
            ppl_val = ppl_res["ppl"]
            ppl_status = "PASS" if (min_ppl <= ppl_val <= max_ppl) else "FAIL"
            ppl_res["gate_status"] = ppl_status
        else:
            ppl_status = "FAIL"
            ppl_res["gate_status"] = "FAIL"

    # 2. Canaries check
    canaries_res = run_prompt_canaries(
        model_path=model_path,
        cli_bin=cli_bin,
        prompts_dir=prompts_dir,
        mock_results=mock_canaries,
    )

    golden_canaries = golden.get("canaries", {})
    canary_checks: dict[str, dict[str, Any]] = {}
    canaries_all_pass = True

    for name, golden_c in golden_canaries.items():
        curr_c = canaries_res.get(name)
        if not curr_c:
            canaries_all_pass = False
            canary_checks[name] = {"status": "FAIL", "reason": "missing from test run"}
            continue

        match = (curr_c["output_sha256"] == golden_c["output_sha256"]) and (curr_c["returncode"] == 0)
        if not match:
            canaries_all_pass = False
            canary_checks[name] = {
                "status": "FAIL",
                "expected_sha256": golden_c["output_sha256"],
                "actual_sha256": curr_c["output_sha256"],
                "generated_text": curr_c["generated_text"][:100],
            }
        else:
            canary_checks[name] = {
                "status": "PASS",
                "output_sha256": curr_c["output_sha256"],
            }

    overall_pass = (ppl_status == "PASS") and canaries_all_pass
    end_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

    gate_result: dict[str, Any] = {
        "timestamp_start": start_utc,
        "timestamp_end": end_utc,
        "status": "PASS" if overall_pass else "FAIL",
        "perplexity": {
            "status": ppl_status,
            "measured_ppl": ppl_res.get("ppl"),
            "measured_err": ppl_res.get("ppl_err"),
            "reference_ppl": ref_ppl,
            "allowed_range": [min_ppl, max_ppl],
            "delta_pct": round(((ppl_res.get("ppl", 0) - ref_ppl) / ref_ppl) * 100.0, 3) if ppl_res.get("ppl") else None,
        },
        "canaries": {
            "status": "PASS" if canaries_all_pass else "FAIL",
            "total": len(golden_canaries),
            "passed": sum(1 for c in canary_checks.values() if c["status"] == "PASS"),
            "checks": canary_checks,
        },
    }

    out_p = Path(out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(gate_result, f, indent=2)

    return gate_result


def main() -> int:
    parser = argparse.ArgumentParser(description="Model-Level Quality Gate runner (QUAL-02)")
    parser.add_argument("--record-golden", action="store_true", help="Record stock baseline golden values")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model GGUF path")
    parser.add_argument("--ppl-bin", default=DEFAULT_PPL_BIN, help="Path to llama-perplexity")
    parser.add_argument("--cli-bin", default=DEFAULT_CLI_BIN, help="Path to llama-cli")
    parser.add_argument("--data", default=DEFAULT_WIKI_DATA, help="Path to wiki.test.raw")
    parser.add_argument("--prompts", default=DEFAULT_PROMPTS_DIR, help="Path to prompts directory")
    parser.add_argument("--golden", default=DEFAULT_GOLDEN_PATH, help="Path to golden json store")
    parser.add_argument("--out-json", default=DEFAULT_OUT_JSON, help="Path for JSON output")
    parser.add_argument("--chunks", type=int, default=None, help="Number of chunks for perplexity (default: full)")
    args = parser.parse_args()

    if args.record_golden:
        record_golden(
            model_path=args.model,
            ppl_bin=args.ppl_bin,
            cli_bin=args.cli_bin,
            data_path=args.data,
            prompts_dir=args.prompts,
            golden_path=args.golden,
            chunks=args.chunks,
        )
        return 0

    print("=== QUAL-02: Running Model-Level Quality Gate ===")
    res = evaluate_model_gate(
        model_path=args.model,
        ppl_bin=args.ppl_bin,
        cli_bin=args.cli_bin,
        data_path=args.data,
        prompts_dir=args.prompts,
        golden_path=args.golden,
        out_json=args.out_json,
        chunks=args.chunks,
    )

    print(f"Perplexity Gate : {res['perplexity']['status']} (measured: {res['perplexity']['measured_ppl']}, allowed: {res['perplexity']['allowed_range']})")
    print(f"Canary Gate     : {res['canaries']['status']} ({res['canaries']['passed']}/{res['canaries']['total']} exact matches)")
    print(f"\nOverall Verdict : {res['status']}")
    print(f"Results saved to: {args.out_json}")

    return 0 if res["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
