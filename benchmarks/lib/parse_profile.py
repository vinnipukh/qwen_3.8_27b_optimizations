"""Profile Parser and Op Attribution Engine (Plan 03-03, Plan 03-04).

Aggregates per-node and per-op execution records into structured profiling tables,
separating Prefill (M >> 1) and Decode (M = 1) phases, and classifying bound types.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

# Canonical op classification heuristics
OP_BOUND_CLASSIFICATION = {
    "MUL_MAT": "Memory Bandwidth / Dequant Bound",
    "GATED_DELTA_NET": "Compute / Register Bound",
    "SOLVE_TRI": "Compute / Latency Bound",
    "SSM_CONV": "Memory / Register Bound",
    "SSM_SCAN": "Compute Bound",
    "FLASH_ATTN_EXT": "Compute / Memory Bandwidth Bound",
    "RMS_NORM": "Memory Bandwidth Bound",
    "MUL": "Memory Bandwidth Bound",
    "ADD": "Memory Bandwidth Bound",
    "GET_ROWS": "Memory Bandwidth / Gathering Bound",
    "CPY": "Memory Bandwidth Bound",
    "SILU": "Compute / Elementwise Bound",
    "SWIGLU": "Compute / Elementwise Bound",
    "SIGMOID": "Compute / Elementwise Bound",
    "ROPE": "Memory Bandwidth / Compute Bound",
    "SCALE": "Memory Bandwidth Bound",
    "L2_NORM": "Memory Bandwidth Bound",
}


def classify_op_bound(op_name: str) -> str:
    """Return dominant performance bound type for a given GGML operation."""
    return OP_BOUND_CLASSIFICATION.get(op_name, "Memory / Compute Bound")


def parse_profile_json(profile_path: str | Path) -> dict[str, Any]:
    """Parse raw JSON emitted by eval_profiler."""
    with open(profile_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prefill_summary = data.get("prefill_summary", {})
    decode_summary = data.get("decode_summary", {})
    overall_summary = data.get("overall_summary", {})

    total_prefill_us = data.get("total_prefill_gpu_us", 0.0)
    total_decode_us = data.get("total_decode_gpu_us", 0.0)
    total_overall_us = data.get("total_overall_gpu_us", 0.0)

    # Sort op lists
    def to_ranked_list(summary_dict: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        for op, stats in summary_dict.items():
            rows.append({
                "op": op,
                "pct": stats.get("pct", 0.0),
                "total_ms": round(stats.get("total_us", 0.0) / 1000.0, 3),
                "count": stats.get("count", 0),
                "avg_us": round(stats.get("avg_us", 0.0), 2),
                "bound_type": classify_op_bound(op),
            })
        rows.sort(key=lambda x: x["pct"], reverse=True)
        return rows

    return {
        "prompt_tokens": data.get("prompt_tokens", 0),
        "gen_tokens": data.get("gen_tokens", 0),
        "prefill_wall_ms": data.get("prefill_wall_ms", 0.0),
        "decode_wall_ms": data.get("decode_wall_ms", 0.0),
        "total_prefill_ms": round(total_prefill_us / 1000.0, 3),
        "total_decode_ms": round(total_decode_us / 1000.0, 3),
        "total_overall_ms": round(total_overall_us / 1000.0, 3),
        "ranked_prefill": to_ranked_list(prefill_summary),
        "ranked_decode": to_ranked_list(decode_summary),
        "ranked_overall": to_ranked_list(overall_summary),
    }


def format_markdown_table(ranked_ops: Sequence[dict[str, Any]], top_n: int = 10) -> str:
    """Render markdown table for a list of ranked operations."""
    lines = [
        "| Rank | GGML Operation | % Runtime | Total Time (ms) | Invocation Count | Avg Latency (μs) | Bound Classification |",
        "|:---:|:---|:---:|:---:|:---:|:---:|:---|",
    ]
    for idx, row in enumerate(ranked_ops[:top_n], start=1):
        lines.append(
            f"| {idx} | `{row['op']}` | {row['pct']:.2f}% | {row['total_ms']:.2f} ms | {row['count']} | {row['avg_us']:.1f} μs | {row['bound_type']} |"
        )
    return "\n".join(lines)
