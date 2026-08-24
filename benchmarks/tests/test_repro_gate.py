"""Unit tests for reproducibility gate and variance calculations."""
import pytest
from benchmarks.lib import llabench
from benchmarks.tests.fixtures.gen_llabench_jsonl import make_tier_rows


def test_variance_pct_calculations():
    # Identical values -> 0.0%
    assert llabench.variance_pct(100.0, 100.0) == 0.0
    assert llabench.variance_pct(0.0, 0.0) == 0.0

    # +4.9% change
    assert pytest.approx(llabench.variance_pct(100.0, 104.9), 1e-5) == 4.9
    assert llabench.repro_ok(100.0, 104.9, max_pct=5.0) is True

    # -4.9% change
    assert pytest.approx(llabench.variance_pct(100.0, 95.1), 1e-5) == 4.9
    assert llabench.repro_ok(100.0, 95.1, max_pct=5.0) is True

    # +5.1% change -> fails 5.0% threshold
    assert pytest.approx(llabench.variance_pct(100.0, 105.1), 1e-5) == 5.1
    assert llabench.repro_ok(100.0, 105.1, max_pct=5.0) is False

    # -5.1% change -> fails 5.0% threshold
    assert pytest.approx(llabench.variance_pct(100.0, 94.9), 1e-5) == 5.1
    assert llabench.repro_ok(100.0, 94.9, max_pct=5.0) is False


def test_session_tier_reproducibility_comparison():
    # Simulate two sessions running tier 4096
    session1_rows = make_tier_rows(tier=4096, pp_ts=111.5, tg_ts=33.5)
    # Session 2 has 2% variance on pp and 3% variance on tg (within 5%)
    session2_rows = make_tier_rows(tier=4096, pp_ts=113.7, tg_ts=34.5)

    for r1, r2 in zip(session1_rows, session2_rows):
        assert llabench.repro_ok(r1["avg_ts"], r2["avg_ts"], max_pct=5.0)

    # Session 3 has 8% variance on pp (fails gate)
    session3_rows = make_tier_rows(tier=4096, pp_ts=121.0, tg_ts=33.5)
    pp_r1 = session1_rows[0]["avg_ts"]
    pp_r3 = session3_rows[0]["avg_ts"]
    assert llabench.repro_ok(pp_r1, pp_r3, max_pct=5.0) is False
