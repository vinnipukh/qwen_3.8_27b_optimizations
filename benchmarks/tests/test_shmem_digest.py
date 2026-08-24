"""Unit tests for HWiNFO shared memory decoder and manual CSV fallback."""
import pytest
from benchmarks.host import hwinfo_daemon
from benchmarks.tests.fixtures.gen_shmem_snapshot import make_sm2_snapshot, make_dead_snapshot


def test_parse_valid_sm2_snapshot():
    buf = make_sm2_snapshot()
    header = hwinfo_daemon.parse_header(buf)

    assert header["is_valid"] is True
    assert header["is_dead"] is False
    assert header["count_readings"] == 9

    readings = hwinfo_daemon.iter_readings(buf, header)
    assert len(readings) == 9

    matched = hwinfo_daemon.match_labels(readings)
    assert matched["gpu_core_clock_mhz"] == 2400.0
    assert matched["gpu_mem_clock_mhz"] == 2500.0
    assert matched["temp_edge_c"] == 62.0
    assert matched["temp_hotspot_c"] == 78.0
    assert matched["power_board_w"] == 285.0
    assert matched["fan_pct"] == 45.0
    assert matched["gpu_util_pct"] == 99.0
    assert matched["vram_used_mb"] == 16400.0
    assert matched["shared_gpu_memory_mb"] == 120.0


def test_parse_dead_sm2_snapshot():
    buf = make_dead_snapshot()
    header = hwinfo_daemon.parse_header(buf)

    assert header["is_dead"] is True
    assert header["is_valid"] is False

    readings = hwinfo_daemon.iter_readings(buf, header)
    assert len(readings) == 0


def test_parse_manual_csv_iso8859_1(tmp_path):
    csv_file = tmp_path / "hwinfo_manual.csv"
    # Create ISO-8859-1 content with special degree symbol \xb0
    header_line = "Date,Time,GPU Temperature [°C],GPU Temperature (Hot Spot) [°C],GPU Memory Allocated [MB]\n"
    data_line = "23.8.2026,16:00:00,60.5,75.2,15200.0\n"
    csv_content = (header_line + data_line).encode("iso-8859-1")
    csv_file.write_bytes(csv_content)

    rows = hwinfo_daemon.parse_manual_csv(csv_file)
    assert len(rows) == 1
    row = rows[0]

    assert row["timestamp"] == "23.8.2026 16:00:00"
    assert row["temp_edge_c"] == 60.5
    assert row["temp_hotspot_c"] == 75.2
    assert row["vram_used_mb"] == 15200.0
    # Missing columns must be None, NOT 0.0
    assert row["gpu_core_clock_mhz"] is None
    assert row["power_board_w"] is None
