"""llama-bench wrapper enforcing explicit cell definitions and matrix integrity.

Eliminates upstream default-cell contamination (BENCH-01) by explicitly
zeroing default vectors and explicitly enumerating pure prefill (-p C)
and decode-at-context (-pg C,128) cells.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from dataclasses import dataclass
from typing import Any, Sequence

PIN_COMMIT = "bb4caa7540188872173c44d161602d9271386413"
BIN_PATH = "/root/llama.cpp/build-ci/bin/llama-bench"
MODEL_PATH = "/root/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf"
TIERS: tuple[int, ...] = (4096, 8192, 16384, 32768)


class MatrixContaminationError(Exception):
    """Raised when benchmark output violates expected cell matrix constraints."""
    pass


def get_physical_cores() -> int:
    """Detect number of physical CPU cores on Linux, fallback to os.cpu_count()."""
    try:
        out = subprocess.check_output(["lscpu", "-p=CORE"], text=True, stderr=subprocess.DEVNULL)
        cores = {line.strip() for line in out.splitlines() if line and not line.startswith("#")}
        if cores:
            return len(cores)
    except Exception:
        pass
    count = os.cpu_count()
    return count if count else 8


def build_argv(
    prompt_tokens: int,
    gen_pairs: list[tuple[int, int]],
    fa_seq: tuple[str, ...] = ("off", "on"),
    repeats: int = 5,
    delay_s: int = 30,
    threads: int | None = None,
    bin_path: str = BIN_PATH,
    model: str = MODEL_PATH,
    warmup: bool = True,
) -> list[str]:
    """Construct llama-bench argv with explicit zero-default overrides.

    Order:
      1. -m <model>
      2. -p <prompt_tokens>
      3. -n 0 (explicitly zeroes default gen vector)
      4. -pg <p,g> for each pair
      5. -fa <off,on>
      6. -r <repeats>
      7. --no-warmup (if warmup is False; omitted when warmup is True per D2-07)
      8. --delay <delay_s> (NEVER -D)
      9. -ngl 99
      10. -sm none
      11. -t <threads>
      12. -o jsonl
      13. -oe jsonl
      14. -v
      15. --progress
    """
    if threads is None:
        threads = get_physical_cores()

    argv: list[str] = [
        bin_path,
        "-m", str(model),
        "-p", str(prompt_tokens),
        "-n", "0",
    ]

    for p, g in gen_pairs:
        argv.extend(["-pg", f"{p},{g}"])

    argv.extend(["-fa", ",".join(fa_seq)])
    argv.extend(["-r", str(repeats)])

    if not warmup:
        argv.append("--no-warmup")

    argv.extend(["--delay", str(delay_s)])
    argv.extend(["-ngl", "99"])
    argv.extend(["-sm", "none"])
    argv.extend(["-t", str(threads)])
    argv.extend(["-o", "jsonl"])
    argv.extend(["-oe", "jsonl"])
    argv.extend(["-v"])
    argv.append("--progress")

    return argv


def run_invocation(
    argv: list[str],
    stdout_path: str | Path,
    stderr_path: str | Path,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> int:
    """Execute binary via subprocess, writing stdout and stderr to disk with LD_LIBRARY_PATH.

    Fail-fast: caller inspects returncode, no internal retries.
    """
    stdout_p = Path(stdout_path)
    stderr_p = Path(stderr_path)
    stdout_p.parent.mkdir(parents=True, exist_ok=True)
    stderr_p.parent.mkdir(parents=True, exist_ok=True)

    run_env = dict(os.environ if env is None else env)
    bin_dir = str(Path(argv[0]).parent)
    current_ld = run_env.get("LD_LIBRARY_PATH", "")
    run_env["LD_LIBRARY_PATH"] = f"{bin_dir}:{current_ld}" if current_ld else bin_dir

    with open(stdout_p, "wb") as f_out, open(stderr_p, "wb") as f_err:
        proc = subprocess.run(
            argv,
            stdout=f_out,
            stderr=f_err,
            cwd=cwd,
            env=run_env,
        )
        f_out.flush()
        f_err.flush()

    return proc.returncode


def parse_rows(path: str | Path) -> list[dict[str, Any]]:
    """Parse JSONL rows from file, ignoring blank lines."""
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                row = json.loads(line_str)
                rows.append(row)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON on line {line_no} of {path}: {exc}") from exc
    return rows


def scan_banned_signatures(rows: Sequence[dict[str, Any]]) -> list[str]:
    """Scan rows for default-contamination or unexpected shape signatures."""
    violations: list[str] = []
    for idx, r in enumerate(rows):
        n_prompt = r.get("n_prompt")
        n_gen = r.get("n_gen")
        # Upstream default contamination:
        # 1. Generation row with n_prompt == 512 (default prompt vector)
        # 2. Generation row with empty context n_prompt == 0 and default n_gen == 128
        if n_prompt == 512 and n_gen != 0:
            violations.append(f"Row {idx}: Contaminated by upstream default prompt 512 with n_gen={n_gen}")
        if n_prompt == 0 and n_gen == 128:
            violations.append(f"Row {idx}: Banned empty-context generation row n_prompt=0, n_gen=128")
    return violations


def assert_cell_integrity(
    rows: list[dict[str, Any]],
    expected_cells: list[dict[str, Any]],
) -> None:
    """Mechanically assert that rows match expected cells and contain no default contamination."""
    violations = scan_banned_signatures(rows)
    if violations:
        raise MatrixContaminationError("Found banned default signatures: " + "; ".join(violations))

    if len(rows) != len(expected_cells):
        raise MatrixContaminationError(
            f"Row count mismatch: got {len(rows)} rows, expected {len(expected_cells)}"
        )

    for idx, (actual, expected) in enumerate(zip(rows, expected_cells)):
        # Determine cell type
        actual_type = "pp" if actual.get("n_gen") == 0 else "tg"
        expected_type = expected.get("type", "pp" if expected.get("n_gen", 0) == 0 else "tg")

        if actual_type != expected_type:
            raise MatrixContaminationError(
                f"Row {idx} type mismatch: expected {expected_type}, got {actual_type}"
            )

        if "n_prompt" in expected and actual.get("n_prompt") != expected["n_prompt"]:
            raise MatrixContaminationError(
                f"Row {idx} n_prompt mismatch: expected {expected['n_prompt']}, got {actual.get('n_prompt')}"
            )

        if "n_gen" in expected and actual.get("n_gen") != expected["n_gen"]:
            raise MatrixContaminationError(
                f"Row {idx} n_gen mismatch: expected {expected['n_gen']}, got {actual.get('n_gen')}"
            )

        if "flash_attn" in expected:
            exp_fa = expected["flash_attn"]
            # Map string or int/bool representation
            if isinstance(exp_fa, str):
                exp_fa_val = 1 if exp_fa == "on" else 0
            else:
                exp_fa_val = 1 if exp_fa else 0

            act_fa = actual.get("flash_attn")
            act_fa_val = 1 if act_fa in (1, True, "on") else 0

            if act_fa_val != exp_fa_val:
                raise MatrixContaminationError(
                    f"Row {idx} flash_attn mismatch: expected {exp_fa_val}, got {act_fa_val}"
                )


@dataclass(frozen=True)
class TierPlan:
    tier: int
    argv: list[str]
    expected_cells: list[dict[str, Any]]
    fa_seq: tuple[str, ...]


def enumerate_tiers(
    tiers: tuple[int, ...] = TIERS,
    fa_seq: tuple[str, ...] = ("off", "on"),
    repeats: int = 5,
    delay_s: int = 30,
    threads: int | None = None,
    bin_path: str = BIN_PATH,
    model: str = MODEL_PATH,
    model_path: str | None = None,
    warmup: bool = True,
) -> list[TierPlan]:
    """Generate TierPlans in ascending context tier order (D2-19).

    Within each tier:
      fa_seq order (default 'off' then 'on').
      For each fa: pure prefill (-p C) then decode-at-context (-pg C,128).
    """
    target_model = model_path if model_path is not None else model
    plans: list[TierPlan] = []
    for tier in sorted(tiers):
        argv = build_argv(
            prompt_tokens=tier,
            gen_pairs=[(tier, 128)],
            fa_seq=fa_seq,
            repeats=repeats,
            delay_s=delay_s,
            threads=threads,
            bin_path=bin_path,
            model=target_model,
            warmup=warmup,
        )

        expected_cells: list[dict[str, Any]] = []
        for fa in fa_seq:
            fa_val = 1 if fa == "on" else 0
            expected_cells.append({
                "type": "pp",
                "n_prompt": tier,
                "n_gen": 0,
                "flash_attn": fa_val,
            })
            expected_cells.append({
                "type": "tg",
                "n_prompt": tier,
                "n_gen": 128,
                "flash_attn": fa_val,
            })

        plans.append(
            TierPlan(
                tier=tier,
                argv=argv,
                expected_cells=expected_cells,
                fa_seq=fa_seq,
            )
        )
    return plans


def assert_tier_rows(rows: list[dict[str, Any]], plan: TierPlan) -> None:
    """Validate rows from a tier execution against its TierPlan."""
    assert_cell_integrity(rows, plan.expected_cells)


def variance_pct(mean_a: float, mean_b: float) -> float:
    """Calculate absolute percentage difference relative to mean_a.

    If mean_a is 0, compares against mean_b or returns 0 if both are 0.
    """
    if mean_a == 0.0 and mean_b == 0.0:
        return 0.0
    ref = mean_a if mean_a != 0.0 else mean_b
    return abs(mean_b - mean_a) / ref * 100.0


def repro_ok(mean_a: float, mean_b: float, max_pct: float = 5.0) -> bool:
    """Check if two throughput measurements reproduce within max_pct tolerance."""
    return variance_pct(mean_a, mean_b) <= max_pct
