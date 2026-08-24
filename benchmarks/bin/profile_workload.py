#!/usr/bin/env python3
"""Workload Profiling CLI Runner (PROF-01, PROF-02, Plan 03-03, Plan 03-04).

Executes `eval_profiler` across configured inference shapes, separating Prefill
and Decode phases, aggregates op statistics, and outputs structured profile reports.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

# Ensure repository root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.lib import parse_profile

DEFAULT_PROFILER_BIN = "benchmarks/bin/eval_profiler"
DEFAULT_MODEL = "/root/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf"
DEFAULT_OUT_DIR = "benchmarks/profiling/raw"


def generate_synthetic_prompt(n_tokens: int) -> str:
    """Generate deterministic text matching approximately desired token count."""
    words_per_token = 0.75
    target_words = int(n_tokens * words_per_token)
    base_phrase = "The rapid evolution of high performance GPU computing drives modern deep learning acceleration. "
    repetitions = (target_words // len(base_phrase.split())) + 1
    full_text = (base_phrase * repetitions)
    # Return bounded slice
    words = full_text.split()[:target_words]
    return " ".join(words)


def profile_workload(
    prompt_tokens: int = 128,
    gen_tokens: int = 128,
    prompt_file: str | None = None,
    prompt_text: str | None = None,
    model_path: str = DEFAULT_MODEL,
    profiler_bin: str = DEFAULT_PROFILER_BIN,
    out_json: Path | str | None = None,
    disable_graphs: bool = False,
    ctx_size: int | None = None,
) -> dict[str, Any]:
    """Execute profiler and return parsed profile summary."""
    p_bin = Path(profiler_bin).resolve()
    if not p_bin.exists():
        raise FileNotFoundError(f"Profiler binary {p_bin} not found. Build it with g++ first.")

    if ctx_size is None:
        ctx_size = max(4096, prompt_tokens + gen_tokens + 512)

    batch_size = max(2048, prompt_tokens)

    # Determine prompt source
    temp_prompt_file: Path | None = None
    if prompt_file:
        p_arg = ["-f", str(Path(prompt_file).resolve())]
    elif prompt_text:
        p_arg = ["-p", prompt_text]
    else:
        # Generate synthetic prompt
        syn_text = generate_synthetic_prompt(prompt_tokens)
        temp_prompt_file = Path(f"/tmp/syn_prompt_{prompt_tokens}.txt")
        with open(temp_prompt_file, "w", encoding="utf-8") as f:
            f.write(syn_text)
        p_arg = ["-f", str(temp_prompt_file)]

    raw_json_out = Path(out_json) if out_json else Path(f"/tmp/prof_{prompt_tokens}_{gen_tokens}.json")
    raw_json_out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(p_bin),
        "-m", model_path,
        *p_arg,
        "-n", str(gen_tokens),
        "-c", str(ctx_size),
        "-b", str(batch_size),
        "-ub", str(min(512, batch_size)),
        "-ngl", "99",
        "--load-mode", "none",
        "--out-json", str(raw_json_out),
    ]

    env = os.environ.copy()
    env["HSA_ENABLE_DXG_DETECTION"] = "1"
    env["LD_LIBRARY_PATH"] = f"/root/llama.cpp/build-ci/bin:{env.get('LD_LIBRARY_PATH', '')}"
    if disable_graphs:
        env["GGML_CUDA_DISABLE_GRAPHS"] = "1"

    print(f"Executing profiler: Prompt Tokens ~{prompt_tokens}, Gen Tokens = {gen_tokens}, Graphs = {'OFF' if disable_graphs else 'ON'}")
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    if temp_prompt_file and temp_prompt_file.exists():
        try:
            temp_prompt_file.unlink()
        except OSError:
            pass

    if proc.returncode != 0:
        print(f"Profiler error (code {proc.returncode}):\n{proc.stderr}\n{proc.stdout}", file=sys.stderr)
        raise RuntimeError(f"Profiler failed with exit code {proc.returncode}")

    if not raw_json_out.exists():
        raise FileNotFoundError(f"Expected profiler output file {raw_json_out} was not generated.")

    parsed = parse_profile.parse_profile_json(raw_json_out)
    parsed["disable_graphs"] = disable_graphs
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Workload Profiler CLI Runner")
    parser.add_argument("--prompt-tokens", type=int, default=128, help="Prompt tokens count")
    parser.add_argument("--gen-tokens", type=int, default=128, help="Generated tokens count")
    parser.add_argument("--prompt-file", default=None, help="Prompt file path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="GGUF model path")
    parser.add_argument("--profiler-bin", default=DEFAULT_PROFILER_BIN, help="Path to eval_profiler binary")
    parser.add_argument("--out-json", default=None, help="Output JSON path")
    parser.add_argument("--disable-graphs", action="store_true", help="Disable HIP graphs via GGML_CUDA_DISABLE_GRAPHS=1")
    args = parser.parse_args()

    res = profile_workload(
        prompt_tokens=args.prompt_tokens,
        gen_tokens=args.gen_tokens,
        prompt_file=args.prompt_file,
        model_path=args.model,
        profiler_bin=args.profiler_bin,
        out_json=args.out_json,
        disable_graphs=args.disable_graphs,
    )

    print("\n=== Profiling Summary ===")
    print(f"Prompt Tokens  : {res['prompt_tokens']}")
    print(f"Gen Tokens     : {res['gen_tokens']}")
    print(f"Prefill Time   : {res['prefill_wall_ms']:.2f} ms")
    print(f"Decode Time    : {res['decode_wall_ms']:.2f} ms")
    print("\nTop Decode Operations:")
    print(parse_profile.format_markdown_table(res["ranked_decode"], top_n=5))

    return 0


if __name__ == "__main__":
    sys.exit(main())
