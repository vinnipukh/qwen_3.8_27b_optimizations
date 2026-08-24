"""Synthetic HWiNFO SM2 shared memory buffer fixture generator."""
from __future__ import annotations

import struct
from benchmarks.host.hwinfo_daemon import (
    HEADER_FMT,
    HEADER_SIZE,
    READING_FMT,
    READING_SIZE,
    HWIS_SIG,
    DEAD_SIG,
)


def _encode_fixed(s: str, length: int) -> bytes:
    b = s.encode("utf-8")
    return b.ljust(length, b"\x00")[:length]


def make_sm2_snapshot(
    sig: bytes = HWIS_SIG,
    poll_time: int = 1724428800,
    poll_period_ms: int = 2000,
    sensors: list[dict] | None = None,
) -> bytes:
    """Construct byte buffer representing HWiNFO SM2 memory block."""
    if sensors is None:
        sensors = [
            {"type": 1, "id": 0x1000, "label": "GPU Clock", "unit": "MHz", "val": 2400.0},
            {"type": 1, "id": 0x1001, "label": "Memory Clock", "unit": "MHz", "val": 2500.0},
            {"type": 2, "id": 0x1002, "label": "GPU Temperature", "unit": "°C", "val": 62.0},
            {"type": 2, "id": 0x1003, "label": "GPU Temperature (Hot Spot)", "unit": "°C", "val": 78.0},
            {"type": 3, "id": 0x1004, "label": "GPU Board Power", "unit": "W", "val": 285.0},
            {"type": 4, "id": 0x1005, "label": "GPU Fan", "unit": "%", "val": 45.0},
            {"type": 5, "id": 0x1006, "label": "GPU Utilization", "unit": "%", "val": 99.0},
            {"type": 6, "id": 0x1007, "label": "GPU Memory Allocated", "unit": "MB", "val": 16400.0},
            {"type": 6, "id": 0x1008, "label": "Shared GPU Memory", "unit": "MB", "val": 120.0},
        ]

    num_readings = len(sensors)
    offset_sensors = HEADER_SIZE
    size_sensors = 64
    count_sensors = 1
    offset_readings = offset_sensors + (count_sensors * size_sensors)
    size_readings = READING_SIZE

    header_bytes = struct.pack(
        HEADER_FMT,
        sig,
        2,  # version
        1,  # revision
        poll_time,
        offset_sensors,
        size_sensors,
        count_sensors,
        offset_readings,
        size_readings,
        num_readings,
        poll_period_ms,
    )

    # Pad up to offset_readings
    pad_len = offset_readings - len(header_bytes)
    buf = bytearray(header_bytes + (b"\x00" * pad_len))

    for idx, s in enumerate(sensors):
        t_type = s.get("type", 0)
        sensor_idx = 0
        r_id = s.get("id", idx)
        label_orig = _encode_fixed(s.get("label", ""), 128)
        label_user = _encode_fixed(s.get("label_user", ""), 128)
        unit = _encode_fixed(s.get("unit", ""), 16)
        val = float(s.get("val", 0.0))
        dmin = float(s.get("min", val))
        dmax = float(s.get("max", val))
        davg = float(s.get("avg", val))

        r_bytes = struct.pack(
            READING_FMT,
            t_type,
            sensor_idx,
            r_id,
            label_orig,
            label_user,
            unit,
            val,
            dmin,
            dmax,
            davg,
        )
        buf.extend(r_bytes)

    return bytes(buf)


def make_dead_snapshot() -> bytes:
    """Construct buffer with DEAD signature."""
    return make_sm2_snapshot(sig=DEAD_SIG)
