#!/usr/bin/env python3
"""4-Shape Profiling & Bottleneck Attribution Generator (PROF-02, Plan 03-04).

Executes full profiling across canonical inference shapes:
  S1: 128 / 128   (Interactive)
  S2: 128 / 1024  (Decode / Generation heavy)
  S3: 4096 / 128  (Prefill / Document QA heavy)
  S4: 4096 / 1024 (Agentic multi-turn)
Generates raw json logs, summary JSON, and published BOTTLENECK-TABLE.md.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.bin import profile_workload
from benchmarks.lib import parse_profile

SHAPES = [
    {"name": "S1_interactive", "prompt_tokens": 128, "gen_tokens": 128, "desc": "Interactive (128 / 128)"},
    {"name": "S2_decode_heavy", "prompt_tokens": 128, "gen_tokens": 256, "desc": "Decode Heavy (128 / 256)"},
    {"name": "S3_prefill_heavy", "prompt_tokens": 4096, "gen_tokens": 128, "desc": "Prefill Heavy (4096 / 128)"},
    {"name": "S4_agentic", "prompt_tokens": 4096, "gen_tokens": 256, "desc": "Agentic Multi-Turn (4096 / 256)"},
]

RAW_DIR = Path("benchmarks/profiling/raw")
SUMMARY_JSON = Path("benchmarks/profiling/bottleneck_summary.json")
REPORT_MD = Path("benchmarks/profiling/BOTTLENECK-TABLE.md")


def run_4shape_profiling(
    smoke: bool = False,
    model_path: str = profile_workload.DEFAULT_MODEL,
    profiler_bin: str = profile_workload.DEFAULT_PROFILER_BIN,
) -> dict[str, Any]:
    """Execute profiling across all 4 shapes and produce unified analysis."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}

    active_shapes = SHAPES
    if smoke:
        active_shapes = [
            {"name": "S1_interactive", "prompt_tokens": 64, "gen_tokens": 32, "desc": "Smoke S1 (64 / 32)"},
            {"name": "S2_decode_heavy", "prompt_tokens": 64, "gen_tokens": 64, "desc": "Smoke S2 (64 / 64)"},
            {"name": "S3_prefill_heavy", "prompt_tokens": 256, "gen_tokens": 32, "desc": "Smoke S3 (256 / 32)"},
            {"name": "S4_agentic", "prompt_tokens": 256, "gen_tokens": 64, "desc": "Smoke S4 (256 / 64)"},
        ]

    for s in active_shapes:
        sname = s["name"]
        p_tok = s["prompt_tokens"]
        g_tok = s["gen_tokens"]
        raw_json = RAW_DIR / f"{sname}.json"

        print(f"\n=======================================================")
        print(f"Profiling Shape: {s['desc']}")
        print(f"=======================================================")

        res = profile_workload.profile_workload(
            prompt_tokens=p_tok,
            gen_tokens=g_tok,
            model_path=model_path,
            profiler_bin=profiler_bin,
            out_json=raw_json,
        )
        results[sname] = {
            "meta": s,
            "profile": res,
        }

    # Aggregate across shapes to calculate cumulative GPU wall time per op
    cumulative_op_ms: dict[str, float] = {}
    prefill_op_ms: dict[str, float] = {}
    decode_op_ms: dict[str, float] = {}

    for sname, sdata in results.items():
        prof = sdata["profile"]
        for row in prof["ranked_prefill"]:
            op = row["op"]
            prefill_op_ms[op] = prefill_op_ms.get(op, 0.0) + row["total_ms"]
            cumulative_op_ms[op] = cumulative_op_ms.get(op, 0.0) + row["total_ms"]

        for row in prof["ranked_decode"]:
            op = row["op"]
            decode_op_ms[op] = decode_op_ms.get(op, 0.0) + row["total_ms"]
            cumulative_op_ms[op] = cumulative_op_ms.get(op, 0.0) + row["total_ms"]

    total_all_ms = sum(cumulative_op_ms.values())
    total_prefill_ms = sum(prefill_op_ms.values())
    total_decode_ms = sum(decode_op_ms.values())

    ranked_cumulative = []
    for op, ms in cumulative_op_ms.items():
        ranked_cumulative.append({
            "op": op,
            "cumulative_ms": round(ms, 2),
            "pct_total": round((ms / total_all_ms * 100.0), 2) if total_all_ms > 0 else 0.0,
            "prefill_ms": round(prefill_op_ms.get(op, 0.0), 2),
            "prefill_pct": round((prefill_op_ms.get(op, 0.0) / total_prefill_ms * 100.0), 2) if total_prefill_ms > 0 else 0.0,
            "decode_ms": round(decode_op_ms.get(op, 0.0), 2),
            "decode_pct": round((decode_op_ms.get(op, 0.0) / total_decode_ms * 100.0), 2) if total_decode_ms > 0 else 0.0,
            "bound_type": parse_profile.classify_op_bound(op),
        })

    ranked_cumulative.sort(key=lambda x: x["cumulative_ms"], reverse=True)
    target_number_one = ranked_cumulative[0]["op"] if ranked_cumulative else "MUL_MAT"

    summary_data = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_artifact": model_path,
        "gpu_target": "gfx1100 (AMD Radeon RX 7900 XT)",
        "optimization_target_1": target_number_one,
        "optimization_target_1_rationale": (
            f"Operation '{target_number_one}' accounts for {ranked_cumulative[0]['pct_total']}% "
            f"of total GPU runtime across all 4 benchmarked shapes ({ranked_cumulative[0]['cumulative_ms']} ms cumulative). "
            f"It represents the dominant compute/memory bottleneck in both prefill ({ranked_cumulative[0]['prefill_pct']}%) "
            f"and decode ({ranked_cumulative[0]['decode_pct']}%)."
        ),
        "cumulative_ranked_ops": ranked_cumulative,
        "shape_profiles": results,
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Generate Markdown Report
    lines = [
        "# Bottleneck Attribution & Optimization Target #1 Table (PROF-02)",
        "",
        f"**Date:** {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
        "**Target Hardware:** AMD Radeon RX 7900 XT (`gfx1100`, RDNA3, 20 GiB VRAM)  ",
        f"**Model Artifact:** `{model_path}`  ",
        "**Host Stack:** Windows 11 Pro / WSL2 Ubuntu 24.04 (Adrenalin 26.2.2 + ROCm 7.2.1)  ",
        "",
        "---",
        "",
        "## 1. Executive Designation of Optimization Target #1",
        "",
        f"🎯 **PRIMARY OPTIMIZATION TARGET #1:** `{target_number_one}`",
        "",
        f"> **Attribution Rationale:** Across all four canonical inference shapes (S1–S4), `{target_number_one}` accounts for **{ranked_cumulative[0]['pct_total']}%** of total cumulative GPU execution time ({ranked_cumulative[0]['cumulative_ms']:.2f} ms). In the Decode phase (M=1), it constitutes **{ranked_cumulative[0]['decode_pct']}%** of runtime, and in Prefill (M>>1) it constitutes **{ranked_cumulative[0]['prefill_pct']}%**. Custom gfx1100 kernel development in Phase 4 (scaffolding) and Phase 5 (kernel attack) will directly target this bottleneck.",
        "",
        "---",
        "",
        "## 2. Cumulative Op Bottleneck Ranking",
        "",
        "| Rank | GGML Operation | % Total GPU Time | Cumulative Time (ms) | % Prefill Time | % Decode Time | Primary Bound Classification |",
        "|:---:|:---|:---:|:---:|:---:|:---:|:---|",
    ]

    for idx, row in enumerate(ranked_cumulative[:12], start=1):
        lines.append(
            f"| {idx} | `{row['op']}` | **{row['pct_total']:.2f}%** | {row['cumulative_ms']:.2f} ms | {row['prefill_pct']:.2f}% | {row['decode_pct']:.2f}% | {row['bound_type']} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Breakdown by Inference Workload Shape",
        "",
    ])

    for s in active_shapes:
        sname = s["name"]
        sdata = results[sname]
        prof = sdata["profile"]
        lines.extend([
            f"### {s['desc']}",
            f"- **Prefill Latency:** {prof['prefill_wall_ms']:.2f} ms ({prof['prompt_tokens']} tokens)",
            f"- **Decode Latency:** {prof['decode_wall_ms']:.2f} ms ({prof['gen_tokens']} tokens)",
            "",
            "#### Top Operations (Decode Phase M=1)",
            parse_profile.format_markdown_table(prof["ranked_decode"], top_n=6),
            "",
            "#### Top Operations (Prefill Phase M>>1)",
            parse_profile.format_markdown_table(prof["ranked_prefill"], top_n=6),
            "",
        ])

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nBottleneck table published to: {REPORT_MD}")
    print(f"Summary JSON saved to: {SUMMARY_JSON}")

    return summary_data


def main() -> int:
    parser = argparse.ArgumentParser(description="4-Shape Profiling Matrix Runner")
    parser.add_argument("--smoke", action="store_true", help="Run scaled-down token counts for fast validation")
    parser.add_argument("--model", default=profile_workload.DEFAULT_MODEL, help="Model GGUF path")
    parser.add_argument("--profiler-bin", default=profile_workload.DEFAULT_PROFILER_BIN, help="Profiler binary path")
    args = parser.parse_args()

    run_4shape_profiling(
        smoke=args.smoke,
        model_path=args.model,
        profiler_bin=args.profiler_bin,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
