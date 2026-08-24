"""Synthetic llama-bench jsonl row generator for unit tests and fixtures."""
from __future__ import annotations

import json
from typing import Any


def make_jsonl_row(
    n_prompt: int = 4096,
    n_gen: int = 0,
    flash_attn: int = 0,
    avg_ts: float = 110.0,
    stddev_ts: float = 1.0,
    samples_ts: list[float] | None = None,
    build_commit: str = "bb4caa7",
    backend: str = "ROCm",
    **overrides: Any,
) -> dict[str, Any]:
    """Create a dictionary matching llama-bench jsonl row schema."""
    if samples_ts is None:
        samples_ts = [avg_ts] * 5
    avg_ns = int(1e9 / (avg_ts / (n_prompt if n_gen == 0 else n_gen))) if avg_ts > 0 else 0
    stddev_ns = 0

    row: dict[str, Any] = {
        "build_commit": build_commit,
        "build_number": 1,
        "cpu_info": "AMD Ryzen 7 5700X 8-Core Processor",
        "gpu_info": "AMD Radeon RX 7900 XT",
        "backends": backend,
        "model_filename": "/root/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf",
        "model_type": "qwen35 27B IQ4_XS - 4.25 bpw",
        "model_size": 15298043904,
        "model_n_params": 27320697856,
        "n_batch": 2048,
        "n_ubatch": 512,
        "n_threads": 8,
        "cpu_mask": "0x0",
        "cpu_strict": False,
        "poll": 50,
        "type_k": "f16",
        "type_v": "f16",
        "n_gpu_layers": 99,
        "n_cpu_moe": 0,
        "split_mode": "none",
        "main_gpu": 0,
        "no_kv_offload": False,
        "flash_attn": flash_attn,
        "devices": "auto",
        "tensor_split": "0.00",
        "tensor_buft_overrides": "none",
        "load_mode": "auto",
        "embeddings": False,
        "no_op_offload": 0,
        "no_host": False,
        "fit_target": 0,
        "fit_min_ctx": 0,
        "n_prompt": n_prompt,
        "n_gen": n_gen,
        "n_depth": 0,
        "test_time": "2026-08-23T16:00:00Z",
        "avg_ns": avg_ns,
        "stddev_ns": stddev_ns,
        "avg_ts": avg_ts,
        "stddev_ts": stddev_ts,
        "samples_ns": [avg_ns] * len(samples_ts),
        "samples_ts": samples_ts,
    }
    row.update(overrides)
    return row


def make_tier_rows(
    tier: int = 4096,
    gen: int = 128,
    fa_seq: tuple[int, ...] = (0, 1),
    pp_ts: float = 111.5,
    tg_ts: float = 33.5,
) -> list[dict[str, Any]]:
    """Generate 2 * len(fa_seq) rows representing a clean tier execution."""
    rows: list[dict[str, Any]] = []
    for fa in fa_seq:
        # Prompt processing (pp) row
        rows.append(make_jsonl_row(n_prompt=tier, n_gen=0, flash_attn=fa, avg_ts=pp_ts))
        # Text generation (tg) row
        rows.append(make_jsonl_row(n_prompt=tier, n_gen=gen, flash_attn=fa, avg_ts=tg_ts))
    return rows


def write_jsonl_file(path: str, rows: list[dict[str, Any]]) -> None:
    """Write list of row dicts as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
