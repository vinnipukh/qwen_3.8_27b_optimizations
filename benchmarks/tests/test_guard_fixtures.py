"""Unit tests for three-signal guard against synthetic trace fixtures."""
import pytest
from benchmarks.lib import guard
from benchmarks.tests.fixtures.gen_rss_trace import (
    make_healthy_profile,
    make_spiked_rss_profile,
    make_swap_growing_profile,
    make_shared_gpu_trace,
    make_repeat_samples,
)


@pytest.fixture
def sample_thresholds():
    return guard.Thresholds(
        vmrss_fail_kb=2000000,
        vmswap_fail_kb=500000,
        gpu_shared_climb_mb_per_min=200.0,
        repeat_deviation_max_ratio=2.0,
    )


def test_guard_healthy_trace(sample_thresholds):
    rss = make_healthy_profile(base_rss_kb=1000000)
    shm = [50.0, 52.0, 51.0]
    repeats = make_repeat_samples(ratio=1.1)

    verdict = guard.evaluate(
        rss_profile=rss,
        gpu_shared_series=shm,
        repeat_means=repeats,
        thresholds=sample_thresholds,
    )
    assert verdict.verdict == guard.VERDICT_OK
    assert not verdict.flagged_for_review
    assert len(verdict.signals["tripped"]) == 0


def test_guard_spiked_rss_triggers_spill(sample_thresholds):
    rss = make_spiked_rss_profile(base_rss_kb=1000000, spike_rss_kb=3000000)
    verdict = guard.evaluate(rss_profile=rss, thresholds=sample_thresholds)

    assert verdict.verdict == guard.VERDICT_SPILL
    assert "signal1_vmrss_spill" in verdict.signals["tripped"]


def test_guard_swap_growth_triggers_spill(sample_thresholds):
    rss = make_swap_growing_profile(base_rss_kb=1000000, spike_swap_kb=1000000)
    verdict = guard.evaluate(rss_profile=rss, thresholds=sample_thresholds)

    assert verdict.verdict == guard.VERDICT_SPILL
    assert "signal1_vmswap_spill" in verdict.signals["tripped"]


def test_guard_gpu_shared_climb_triggers_spill(sample_thresholds):
    shm_trace = make_shared_gpu_trace(climb_mb=300.0)  # Threshold is 200.0
    verdict = guard.evaluate(gpu_shared_series=shm_trace, thresholds=sample_thresholds)

    assert verdict.verdict == guard.VERDICT_SPILL
    assert "signal2_gpu_shared_climb" in verdict.signals["tripped"]


def test_guard_repeat_deviation_triggers_review(sample_thresholds):
    repeats = make_repeat_samples(ratio=2.5)  # Ratio exceeds 2.0
    verdict = guard.evaluate(repeat_means=repeats, thresholds=sample_thresholds)

    assert verdict.verdict == guard.VERDICT_REVIEW
    assert verdict.flagged_for_review is True
    assert "signal3_repeat_deviation" in verdict.signals["tripped"]


def test_guard_observe_only_without_thresholds():
    rss = make_spiked_rss_profile(base_rss_kb=1000000, spike_rss_kb=100000000)
    shm = make_shared_gpu_trace(climb_mb=5000.0)

    verdict = guard.evaluate(
        rss_profile=rss,
        gpu_shared_series=shm,
        thresholds=None,  # No thresholds -> observe only
    )
    assert verdict.verdict == guard.VERDICT_OK
    assert verdict.signals["observe_only"] is True


def test_no_hardcoded_fail_thresholds_in_guard_source():
    from pathlib import Path
    import re

    code = Path("benchmarks/lib/guard.py").read_text(encoding="utf-8")
    # Search for large literal integers in guard.py that could be hardcoded memory limits (> 100000)
    large_literals = re.findall(r"\b\d{6,}\b", code)
    assert len(large_literals) == 0, f"Found hardcoded memory constants in guard.py: {large_literals}"
