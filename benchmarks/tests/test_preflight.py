"""Unit tests for VRAM preflight checks and ledger records."""
import json
import pytest
from benchmarks.lib import preflight
from benchmarks.lib.guard import VERDICT_PREFLIGHT
from benchmarks.lib.store import RunStore


FROZEN_FREE_MIB = 18245.0


def test_tier_preflight_arithmetic():
    # Tier 4096
    needed_4k = preflight.estimate_needed_mib(4096)
    v_4k = preflight.check(needed_4k, FROZEN_FREE_MIB)
    assert v_4k.verdict == "PASS"

    # Tier 8192
    needed_8k = preflight.estimate_needed_mib(8192)
    v_8k = preflight.check(needed_8k, FROZEN_FREE_MIB)
    assert v_8k.verdict == "PASS"

    # Tier 16384
    needed_16k = preflight.estimate_needed_mib(16384)
    v_16k = preflight.check(needed_16k, FROZEN_FREE_MIB)
    assert v_16k.verdict == "PASS"

    # Tier 32768 -> Must fail preflight against 18245 MiB
    needed_32k = preflight.estimate_needed_mib(32768)
    v_32k = preflight.check(needed_32k, FROZEN_FREE_MIB)
    assert v_32k.verdict == VERDICT_PREFLIGHT
    assert "exceeds available" in v_32k.evidence


def test_parse_free_mib():
    sample_log = "ggml_cuda_init: (20421 MiB, 18245 MiB free) detected on ROCm0"
    total, free = preflight.parse_free_mib(sample_log)
    assert total == 20421
    assert free == 18245


def test_parse_buffer_lines():
    sample_stderr = """
    llama_kv_cache_init: kv cache size = 256.00 MiB
    llama_model_load: compute buffer size = 1024.50 MiB
    """
    bufs = preflight.parse_buffer_lines(sample_stderr)
    assert bufs.get("kv_mib") == 256.0
    assert bufs.get("compute_mib") == 1024.5


def test_record_ledger_roundtrip(tmp_path):
    run_dir = tmp_path / "ledger_run"
    store = RunStore(run_dir)

    entry = preflight.record_ledger(
        run_store=store,
        cell_label="c4096_tg_fa_off",
        buffers={"kv_mib": 256.0, "compute_mib": 1024.0},
        guard_peaks={"vmrss_peak_kb": 15500000, "vmswap_peak_kb": 0},
        dxg_line="(20421 MiB, 18245 MiB free)",
    )

    assert store.ledger_file.exists()
    lines = store.ledger_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    loaded = json.loads(lines[0])
    assert loaded["cell"] == "c4096_tg_fa_off"
    assert loaded["buffers"]["kv_mib"] == 256.0
    assert loaded["guard_peaks"]["vmrss_peak_kb"] == 15500000
