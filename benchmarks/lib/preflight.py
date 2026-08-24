"""VRAM pre-flight allocation gate and per-cell ledger recorder (D2-18, BENCH-03).

Computes expected VRAM consumption before allocating heavy tiers against
measured DXG free VRAM anchor to prevent silent allocation crashes.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from benchmarks.lib.guard import VERDICT_PREFLIGHT
from benchmarks.lib.store import RunStore

# Model and architecture constants (Qwen3.8-27B hybrid: 16 attention layers)
KV_BYTES_PER_TOKEN_F16 = 64 * 1024  # 64 KiB per token in f16 for 16 full-attention layers
WEIGHTS_BYTES = 15309039008  # models/README.md sha256-of-record artifact size
WEIGHTS_MIB = WEIGHTS_BYTES / (1024 * 1024)  # ~14599.8 MiB
DEFAULT_COMPUTE_MIB = 1536.0  # Conservative fallback compute buffer


@dataclass
class PreflightVerdict:
    """Pre-flight allocation check result."""
    verdict: str
    evidence: str
    flags: dict[str, Any]


def parse_free_mib(text: str) -> tuple[int, int]:
    """Parse total and free MiB from llama.cpp startup log text.

    Matches patterns like:
      - (20421 MiB, 18245 MiB free)
      - Total VRAM: 20421 MiB ... free: 18245 MiB
    """
    # Pattern 1: "(20421 MiB, 18245 MiB free)"
    m1 = re.search(r"\((\d+)\s*MiB,\s*(\d+)\s*MiB free\)", text, re.IGNORECASE)
    if m1:
        return int(m1.group(1)), int(m1.group(2))

    # Pattern 2: "Total VRAM: (\d+) MiB"
    m_total = re.search(r"Total VRAM:\s*(\d+)\s*MiB", text, re.IGNORECASE)
    m_free = re.search(r"free:\s*(\d+)\s*MiB", text, re.IGNORECASE)
    if m_total and m_free:
        return int(m_total.group(1)), int(m_free.group(2))
    elif m_total:
        total = int(m_total.group(1))
        return total, total

    return 0, 0


def parse_buffer_lines(stderr_text: str) -> dict[str, float]:
    """Extract KV and compute buffer allocations from verbose logs."""
    buffers: dict[str, float] = {}

    kv_match = re.search(r"kv cache size\s*=\s*([\d\.]+)\s*MiB", stderr_text, re.IGNORECASE)
    if not kv_match:
        kv_match = re.search(r"llama_kv_cache_init:.*size\s*=\s*([\d\.]+)\s*MiB", stderr_text, re.IGNORECASE)
    if kv_match:
        buffers["kv_mib"] = float(kv_match.group(1))

    compute_match = re.search(r"compute buffer size\s*=\s*([\d\.]+)\s*MiB", stderr_text, re.IGNORECASE)
    if compute_match:
        buffers["compute_mib"] = float(compute_match.group(1))

    return buffers


def estimate_needed_mib(tier: int, compute_mib_observed: float | None = None) -> float:
    """Calculate total VRAM needed for model weights + KV cache at tier + compute buffer."""
    kv_mib = (tier * KV_BYTES_PER_TOKEN_F16) / (1024 * 1024)
    compute_mib = compute_mib_observed if compute_mib_observed is not None else DEFAULT_COMPUTE_MIB
    return WEIGHTS_MIB + kv_mib + compute_mib


def check(needed_mib: float, free_mib: float, margin: float = 1.05) -> PreflightVerdict:
    """Evaluate whether estimated VRAM fits within free VRAM with safety margin."""
    budget_needed = needed_mib * margin
    flags = {
        "needed_mib": round(needed_mib, 2),
        "budget_with_margin_mib": round(budget_needed, 2),
        "free_mib": round(free_mib, 2),
        "margin": margin,
    }

    if budget_needed <= free_mib:
        return PreflightVerdict(
            verdict="PASS",
            evidence=f"Estimated needed {needed_mib:.1f} MiB (with margin {budget_needed:.1f} MiB) <= available {free_mib:.1f} MiB free",
            flags=flags,
        )
    else:
        return PreflightVerdict(
            verdict=VERDICT_PREFLIGHT,
            evidence=f"Estimated needed {needed_mib:.1f} MiB (with margin {budget_needed:.1f} MiB) exceeds available {free_mib:.1f} MiB free",
            flags=flags,
        )


def record_ledger(
    run_store: RunStore,
    cell_label: str,
    buffers: dict[str, float],
    guard_peaks: dict[str, Any],
    dxg_line: str,
) -> dict[str, Any]:
    """Append a VRAM ledger entry to the run store."""
    entry = {
        "cell": cell_label,
        "buffers": buffers,
        "guard_peaks": guard_peaks,
        "dxg_line": dxg_line,
    }
    run_store.append_ledger_row(entry)
    return entry
