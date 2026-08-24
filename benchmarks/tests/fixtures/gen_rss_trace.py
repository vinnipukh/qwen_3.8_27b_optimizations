"""Synthetic RSS and telemetry trace fixture generators for guard testing."""
from __future__ import annotations

import time
from benchmarks.lib.guard import RssProfile


def make_healthy_profile(base_rss_kb: int = 1000000) -> RssProfile:
    """Flat healthy memory series."""
    profile = RssProfile(pid=1001)
    for i in range(10):
        profile.vmrss_series_kb.append(base_rss_kb + (i * 100))
        profile.vmswap_series_kb.append(0)
        profile.timestamps.append(time.time() + i)
    return profile


def make_spiked_rss_profile(base_rss_kb: int = 1000000, spike_rss_kb: int = 5000000) -> RssProfile:
    """Memory series that crosses fail threshold."""
    profile = make_healthy_profile(base_rss_kb)
    profile.vmrss_series_kb.extend([spike_rss_kb, spike_rss_kb + 1000])
    profile.vmswap_series_kb.extend([0, 0])
    profile.timestamps.extend([time.time() + 10, time.time() + 11])
    return profile


def make_swap_growing_profile(base_rss_kb: int = 1000000, spike_swap_kb: int = 2000000) -> RssProfile:
    """Memory series where swap spikes."""
    profile = make_healthy_profile(base_rss_kb)
    profile.vmrss_series_kb.extend([base_rss_kb, base_rss_kb])
    profile.vmswap_series_kb.extend([spike_swap_kb // 2, spike_swap_kb])
    profile.timestamps.extend([time.time() + 10, time.time() + 11])
    return profile


def make_shared_gpu_trace(climb_mb: float = 500.0, steps: int = 10) -> list[float]:
    """Shared GPU memory climb series."""
    return [100.0 + (i * (climb_mb / steps)) for i in range(steps)]


def make_repeat_samples(ratio: float = 1.3, base: float = 100.0) -> list[float]:
    """List of repetition throughput values with a specific ratio spread."""
    return [base, base * 1.1, base * ratio]
