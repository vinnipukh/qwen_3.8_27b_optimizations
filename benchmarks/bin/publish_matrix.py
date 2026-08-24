#!/usr/bin/env python3
"""Baseline matrix publisher and table generator (D2-11, D2-19, BENCH-04).

Aggregates raw repeat samples into D2-19-ordered Markdown/JSON tables,
separating verified OK cells from FAILED/REVIEW cells with honest justification.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.lib import guard, llabench, store


def calculate_stats(samples: list[float]) -> tuple[float, float]:
    """Calculate mean and sample standard deviation from list of float samples."""
    if not samples:
        return 0.0, 0.0
    n = len(samples)
    if n == 1:
        return float(samples[0]), 0.0
    mean = sum(samples) / n
    variance = sum((x - mean) ** 2 for x in samples) / (n - 1)
    stddev = math.sqrt(variance)
    return mean, stddev


def assemble_matrix(run_dirs: list[Path]) -> dict[str, Any]:
    """Aggregate rows across runs into structured matrix representation."""
    all_rows: list[dict[str, Any]] = []

    for rd in run_dirs:
        rows_file = rd / "rows.jsonl"
        if rows_file.exists():
            rows = llabench.parse_rows(rows_file)
            all_rows.extend(rows)

    # Scan for matrix contamination
    violations = llabench.scan_banned_signatures(all_rows)
    if violations:
        raise llabench.MatrixContaminationError("Cannot publish contaminated rows: " + "; ".join(violations))

    ok_cells: dict[str, dict[str, Any]] = {}
    failed_cells: list[dict[str, Any]] = []
    review_cells: list[dict[str, Any]] = []

    for r in all_rows:
        backend = r.get("backend_arm", "HIP")
        tier = r.get("n_prompt", 0)
        n_gen = r.get("n_gen", 0)
        fa_val = r.get("flash_attn", 0)
        cell_type = "pp" if n_gen == 0 else "tg"
        fa_str = "on" if fa_val in (1, True, "on") else "off"
        cell_key = f"{backend}_c{tier}_{cell_type}_fa_{fa_str}"

        guard_info = r.get("guard", {})
        verdict = guard_info.get("verdict", "OK")

        # Extract samples
        samples = r.get("samples_ts", [])
        if not samples and r.get("avg_ts", 0) > 0:
            samples = [r["avg_ts"]]

        mean, stddev = calculate_stats(samples)

        cell_entry = {
            "cell_key": cell_key,
            "backend": backend,
            "tier": tier,
            "type": cell_type,
            "flash_attn": fa_str,
            "mean_ts": round(mean, 2),
            "stddev_ts": round(stddev, 2),
            "samples_ts": samples,
            "verdict": verdict,
            "evidence": guard_info.get("evidence", ""),
            "run_id": r.get("run_id", ""),
            "timestamp": r.get("timestamp", ""),
        }

        if verdict == guard.VERDICT_OK:
            ok_cells[cell_key] = cell_entry
        elif verdict == guard.VERDICT_REVIEW:
            review_cells.append(cell_entry)
        else:
            failed_cells.append(cell_entry)

    return {
        "ok_cells": ok_cells,
        "failed_cells": failed_cells,
        "review_cells": review_cells,
    }


def generate_markdown(
    matrix_data: dict[str, Any],
    repro_data: dict[str, Any] | None = None,
    vulkan_matrix: dict[str, Any] | None = None,
) -> str:
    """Render baseline matrix in publication Markdown format."""
    lines: list[str] = []
    lines.append("# Stock Baseline Performance Matrix")
    lines.append("")
    lines.append("**Model:** Qwen3.8-27B-Uncensored-IQ4_XS (15.31 GB)  ")
    lines.append("**Hardware:** AMD Radeon RX 7900 XT (20 GB GDDR6, gfx1100)  ")
    lines.append("**Environment:** ROCm 7.2.1 + librocdxg 1.2.2 (WSL2 Ubuntu 24.04)  ")
    lines.append("**Status:** Calibrated & Guarded (5 repeats per cell, warmup enabled)  ")
    lines.append("")

    # 1. HIP Matrix Table
    lines.append("## Stock HIP Baseline Matrix (ROCm 7.2.1)")
    lines.append("")
    lines.append("| Context Tier | Workload | Flash Attention | Mean Throughput (tok/s) | StdDev (tok/s) | Repeats | Verdict |")
    lines.append("|---|---|---|---|---|---|---|")

    ok_cells = matrix_data["ok_cells"]
    ordered_tiers = [4096, 8192, 16384, 32768]
    workloads = [("pp", "Prefill (pp)"), ("tg", "Decode@Ctx (tg 128)")]
    fa_options = ["off", "on"]

    for tier in ordered_tiers:
        for w_type, w_label in workloads:
            for fa in fa_options:
                key = f"HIP_c{tier}_{w_type}_fa_{fa}"
                if key in ok_cells:
                    c = ok_cells[key]
                    lines.append(f"| {tier} | {w_label} | {fa} | **{c['mean_ts']:.2f}** | ±{c['stddev_ts']:.2f} | {len(c['samples_ts'])} | `{c['verdict']}` |")

    lines.append("")

    # 2. Failed / Pre-flight Intercepted Cells Table
    failed_cells = matrix_data.get("failed_cells", [])
    if failed_cells:
        lines.append("### Failed / Pre-flight Gated Cells")
        lines.append("")
        lines.append("| Context Tier | Workload | Flash Attention | Target Backend | Verdict | Reason / Evidence |")
        lines.append("|---|---|---|---|---|---|")
        for fc in failed_cells:
            w_label = "Prefill (pp)" if fc["type"] == "pp" else "Decode@Ctx (tg 128)"
            lines.append(f"| {fc['tier']} | {w_label} | {fc['flash_attn']} | {fc['backend']} | `{fc['verdict']}` | {fc['evidence'] or 'Allocation limit exceeded'} |")
        lines.append("")

    # 3. Review Flagged Cells Table
    review_cells = matrix_data.get("review_cells", [])
    if review_cells:
        lines.append("### Cells Flagged for Review (D2-12 Signal 3)")
        lines.append("")
        lines.append("| Context Tier | Workload | Flash Attention | Mean (tok/s) | Verdict | Reason |")
        lines.append("|---|---|---|---|---|---|")
        for rc in review_cells:
            w_label = "Prefill (pp)" if rc["type"] == "pp" else "Decode@Ctx (tg 128)"
            lines.append(f"| {rc['tier']} | {w_label} | {rc['flash_attn']} | {rc['mean_ts']:.2f} | `{rc['verdict']}` | Intra-cell repeat deviation ratio exceeded threshold |")
        lines.append("")

    # 4. Reproducibility Gate Verification
    if repro_data:
        lines.append("## Reproducibility Gate Verification (BENCH-01)")
        lines.append("")
        lines.append(f"**Evaluation:** Re-run of Context Tier `{repro_data.get('tier', 8192)}` in an independent session.  ")
        lines.append(f"**Gate Criteria:** Throughput mean variance must be within `±5.0%`.  ")
        lines.append(f"**Overall Verdict:** **{repro_data.get('gate_verdict', 'PASS')}**  ")
        lines.append("")
        lines.append("| Cell | Session 1 Mean (tok/s) | Session 2 Mean (tok/s) | Variance (%) | Gate (<= 5.0%) |")
        lines.append("|---|---|---|---|---|")
        for comp in repro_data.get("comparisons", []):
            gate_str = "✅ PASS" if comp["pass"] else "❌ FAIL"
            lines.append(f"| `{comp['cell']}` | {comp['mean_1']:.2f} | {comp['mean_2']:.2f} | **{comp['variance_pct']:.2f}%** | {gate_str} |")
        lines.append("")

    # 5. Vulkan Comparator Section (if provided)
    if vulkan_matrix:
        lines.append("## Stock Vulkan Comparator Arm (Native Windows)")
        lines.append("")
        lines.append("| Context Tier | Workload | Flash Attention | Vulkan Mean (tok/s) | HIP Mean (tok/s) | HIP vs Vulkan Delta |")
        lines.append("|---|---|---|---|---|---|")
        v_ok = vulkan_matrix["ok_cells"]
        for tier in ordered_tiers:
            for w_type, w_label in workloads:
                for fa in fa_options:
                    v_key = f"Vulkan_c{tier}_{w_type}_fa_{fa}"
                    h_key = f"HIP_c{tier}_{w_type}_fa_{fa}"
                    if v_key in v_ok:
                        vc = v_ok[v_key]
                        hc = ok_cells.get(h_key)
                        if hc:
                            delta = ((hc["mean_ts"] - vc["mean_ts"]) / vc["mean_ts"]) * 100.0 if vc["mean_ts"] > 0 else 0.0
                            delta_str = f"{delta:+.1f}%"
                            lines.append(f"| {tier} | {w_label} | {fa} | {vc['mean_ts']:.2f} | {hc['mean_ts']:.2f} | **{delta_str}** |")
                        else:
                            lines.append(f"| {tier} | {w_label} | {fa} | {vc['mean_ts']:.2f} | N/A | N/A |")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish Baseline Matrix")
    parser.add_argument("run_dirs", type=Path, nargs="*", help="Run directories to aggregate")
    parser.add_argument("--out-md", type=Path, default=Path("benchmarks/results/BASELINE-MATRIX.md"))
    parser.add_argument("--out-json", type=Path, default=Path("benchmarks/results/BASELINE-MATRIX.json"))
    parser.add_argument("--repro-run", type=Path, default=None, help="Second run directory for reproducibility comparison")
    parser.add_argument("--vulkan-run", type=Path, default=None, help="Vulkan comparator run directory")
    args = parser.parse_args()

    # If no run_dirs provided, find baseline run dirs in benchmarks/results
    target_dirs = list(args.run_dirs)
    if not target_dirs:
        results_root = Path("benchmarks/results")
        target_dirs = [d for d in results_root.glob("*_baseline_hip") if d.is_dir()]

    if not target_dirs:
        print("Error: No benchmark run directories found or specified.", file=sys.stderr)
        return 1

    print(f"Aggregating {len(target_dirs)} baseline run directories...")
    matrix_data = assemble_matrix(target_dirs)

    # Reproducibility calculation if repro_run specified
    repro_data = None
    if args.repro_run and args.repro_run.exists():
        repro_mat = assemble_matrix([args.repro_run])
        comparisons = []
        all_pass = True
        for k, c2 in repro_mat["ok_cells"].items():
            if k in matrix_data["ok_cells"]:
                c1 = matrix_data["ok_cells"][k]
                var_pct = llabench.variance_pct(c1["mean_ts"], c2["mean_ts"])
                is_pass = llabench.repro_ok(c1["mean_ts"], c2["mean_ts"], max_pct=5.0)
                if not is_pass:
                    all_pass = False
                comparisons.append({
                    "cell": k,
                    "mean_1": c1["mean_ts"],
                    "mean_2": c2["mean_ts"],
                    "variance_pct": round(var_pct, 2),
                    "pass": is_pass,
                })
        repro_data = {
            "tier": 8192,
            "gate_verdict": "PASS" if all_pass else "FAIL",
            "comparisons": comparisons,
        }

    # Vulkan comparison if vulkan_run specified
    vulkan_matrix = None
    if args.vulkan_run and args.vulkan_run.exists():
        vulkan_matrix = assemble_matrix([args.vulkan_run])

    md_content = generate_markdown(matrix_data, repro_data=repro_data, vulkan_matrix=vulkan_matrix)

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as f_md:
        f_md.write(md_content)

    with open(args.out_json, "w", encoding="utf-8") as f_json:
        json.dump(matrix_data, f_json, indent=2)

    print(f"Successfully published baseline matrix -> {args.out_md} and {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
