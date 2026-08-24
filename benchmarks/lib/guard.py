"""Three-signal VRAM spill guard and RSS poller (D2-12, D2-13, D2-14, BENCH-03).

Detects silent VRAM overcommit and performance collapse using:
1. Guest /proc/<pid>/status VmRSS and VmSwap monitoring.
2. Windows shared-GPU-memory steady climb detection.
3. Intra-cell repeat throughput deviation (>2.0x spread).
All fail thresholds are loaded from calibration config; when absent, operates in observe-only mode.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Sequence

# Locked verdict vocabulary
VERDICT_OK = "OK"
VERDICT_SPILL = "FAILED:suspected-spill"
VERDICT_REVIEW = "REVIEW:repeat-deviation"
VERDICT_THERMAL = "FAILED:thermal-abort"
VERDICT_PREFLIGHT = "FAILED:preflight-oom"

DEFAULT_THRESHOLDS_PATH = "benchmarks/config/thresholds.json"


@dataclass
class Thresholds:
    """Empirically-derived guard thresholds loaded from calibration output."""
    vmrss_fail_kb: int
    vmswap_fail_kb: int
    gpu_shared_climb_mb_per_min: float
    repeat_deviation_max_ratio: float = 2.0

    @classmethod
    def from_json(cls, path: str | Path = DEFAULT_THRESHOLDS_PATH) -> Thresholds | None:
        """Load thresholds from JSON file; returns None if file is absent."""
        p = Path(path)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return cls(
                vmrss_fail_kb=int(data["vmrss_fail_kb"]),
                vmswap_fail_kb=int(data["vmswap_fail_kb"]),
                gpu_shared_climb_mb_per_min=float(data["gpu_shared_climb_mb_per_min"]),
                repeat_deviation_max_ratio=float(data.get("repeat_deviation_max_ratio", 2.0)),
            )
        except Exception:
            return None

    def to_json(self, path: str | Path = DEFAULT_THRESHOLDS_PATH) -> None:
        """Save thresholds to JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "vmrss_fail_kb": self.vmrss_fail_kb,
            "vmswap_fail_kb": self.vmswap_fail_kb,
            "gpu_shared_climb_mb_per_min": self.gpu_shared_climb_mb_per_min,
            "repeat_deviation_max_ratio": self.repeat_deviation_max_ratio,
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


@dataclass
class RssProfile:
    """Time-series and peak memory metrics captured during a run."""
    pid: int
    vmrss_series_kb: list[int] = field(default_factory=list)
    vmswap_series_kb: list[int] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)

    @property
    def vmrss_peak_kb(self) -> int:
        return max(self.vmrss_series_kb) if self.vmrss_series_kb else 0

    @property
    def vmswap_peak_kb(self) -> int:
        return max(self.vmswap_series_kb) if self.vmswap_series_kb else 0


@dataclass
class GuardVerdict:
    """Structured verdict produced by the three-signal detector."""
    verdict: str
    signals: dict[str, Any]
    flagged_for_review: bool


def _read_proc_status_kb(status_text: str, field_name: str) -> int:
    """Parse integer KB value from /proc/<pid>/status field."""
    match = re.search(rf"^{field_name}:\s*(\d+)\s*kB", status_text, re.MULTILINE)
    if match:
        return int(match.group(1))
    return 0


def _poll_proc_windows(pid: int, stop_event: threading.Event, interval_s: float = 1.0) -> RssProfile:
    """Poll Windows process memory via GetProcessMemoryInfo."""
    profile = RssProfile(pid=pid)
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        PROCESS_QUERY_INFORMATION = 0x0400
        PROCESS_VM_READ = 0x0010
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
        if not handle:
            return profile

        psapi = ctypes.windll.psapi
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)

        try:
            while not stop_event.is_set():
                if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                    rss_kb = int(counters.WorkingSetSize // 1024)
                    swap_kb = int(counters.PagefileUsage // 1024)
                    profile.vmrss_series_kb.append(rss_kb)
                    profile.vmswap_series_kb.append(swap_kb)
                    profile.timestamps.append(time.time())
                else:
                    break
                stop_event.wait(interval_s)
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    except Exception:
        pass
    return profile


def poll_proc(
    pid: int,
    stop_event: threading.Event,
    interval_s: float = 1.0,
) -> RssProfile:
    """Poll /proc/<pid>/status for memory usage at interval_s until stop_event is set."""
    if sys.platform == "win32":
        return _poll_proc_windows(pid, stop_event, interval_s)

    profile = RssProfile(pid=pid)
    proc_status = Path(f"/proc/{pid}/status")

    while not stop_event.is_set():
        if not proc_status.exists():
            break
        try:
            text = proc_status.read_text(encoding="utf-8")
            rss = _read_proc_status_kb(text, "VmRSS")
            swap = _read_proc_status_kb(text, "VmSwap")
            profile.vmrss_series_kb.append(rss)
            profile.vmswap_series_kb.append(swap)
            profile.timestamps.append(time.time())
        except OSError:
            break
        stop_event.wait(interval_s)

    return profile


def evaluate(
    rss_profile: RssProfile | None = None,
    gpu_shared_series: Sequence[float] | None = None,
    repeat_means: Sequence[float] | None = None,
    thresholds: Thresholds | None = None,
    observe_only: bool = False,
) -> GuardVerdict:
    """Evaluate signals against thresholds, producing locked house verdict."""
    tripped_signals: list[str] = []
    signals_evidence: dict[str, Any] = {}
    flagged_for_review = False

    if thresholds is None:
        observe_only = True

    # Signal 1: VmRSS / VmSwap spikes
    if rss_profile:
        signals_evidence["vmrss_peak_kb"] = rss_profile.vmrss_peak_kb
        signals_evidence["vmswap_peak_kb"] = rss_profile.vmswap_peak_kb
        if thresholds:
            if rss_profile.vmrss_peak_kb > thresholds.vmrss_fail_kb:
                tripped_signals.append("signal1_vmrss_spill")
            if rss_profile.vmswap_peak_kb > thresholds.vmswap_fail_kb:
                tripped_signals.append("signal1_vmswap_spill")

    # Signal 2: Windows shared-GPU-memory climb
    if gpu_shared_series and len(gpu_shared_series) >= 2:
        start_val = gpu_shared_series[0]
        end_val = gpu_shared_series[-1]
        climb = end_val - start_val
        signals_evidence["gpu_shared_climb_mb"] = climb
        if thresholds and thresholds.gpu_shared_climb_mb_per_min > 0:
            if climb > thresholds.gpu_shared_climb_mb_per_min:
                tripped_signals.append("signal2_gpu_shared_climb")

    # Signal 3: Intra-cell repeat throughput deviation (>2.0x spread)
    if repeat_means and len(repeat_means) >= 2:
        min_v = min(repeat_means)
        max_v = max(repeat_means)
        ratio = (max_v / min_v) if min_v > 0 else 1.0
        signals_evidence["repeat_deviation_ratio"] = round(ratio, 3)
        max_ratio = thresholds.repeat_deviation_max_ratio if thresholds else 2.0
        if ratio > max_ratio:
            tripped_signals.append("signal3_repeat_deviation")
            flagged_for_review = True

    # Compute verdict
    spill_tripped = any(s in ("signal1_vmrss_spill", "signal1_vmswap_spill", "signal2_gpu_shared_climb") for s in tripped_signals)

    if spill_tripped and not observe_only:
        verdict = VERDICT_SPILL
    elif flagged_for_review:
        verdict = VERDICT_REVIEW
    else:
        verdict = VERDICT_OK

    signals_evidence["tripped"] = tripped_signals
    signals_evidence["observe_only"] = observe_only

    return GuardVerdict(
        verdict=verdict,
        signals=signals_evidence,
        flagged_for_review=flagged_for_review,
    )
