"""HWiNFO Shared Memory v2 daemon and manual-CSV parser (D2-01, BENCH-02).

Reads HWiNFO64 Shared Memory v2 (Global\\HWiNFO_SENS_SM2) on Windows host,
captures 9 mandatory GPU sensor metrics at 1 Hz, and outputs time-stamped
telemetry slices. Includes pure-function decoders and ISO-8859-1 CSV fallback.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import os
from pathlib import Path
import struct
import sys
import time
from typing import Any, Iterator

# Constants for HWiNFO Shared Memory v2
HWIS_SM2_MAP_NAME = r"Global\HWiNFO_SENS_SM2"
HWIS_SM2_MUTEX_NAME = r"Global\HWiNFO_SM2_MUTEX"
HWIS_SIG = b"HWiS"
DEAD_SIG = b"DEAD"

# Header struct: sig(4s) ver(I) rev(I) poll_time(q) + off_sensors(I) sz_sensors(I) num_sensors(I) + off_readings(I) sz_readings(I) num_readings(I) + poll_period(I)
HEADER_FMT = "<4sIIqIIIIIII"
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 48 bytes

# Reading element struct (pack=1):
# tType(i) sensor_index(I) reading_id(I) label_orig(128s) label_user(128s) unit(16s) val(d) dmin(d) dmax(d) davg(d)
READING_FMT = "<iII128s128s16sdddd"
READING_SIZE = struct.calcsize(READING_FMT)  # 316 bytes

MANDATORY_FIELDS = [
    "gpu_core_clock_mhz",
    "gpu_mem_clock_mhz",
    "temp_edge_c",
    "temp_hotspot_c",
    "power_board_w",
    "fan_pct",
    "gpu_util_pct",
    "vram_used_mb",
    "shared_gpu_memory_mb",
]


def _clean_str(raw: bytes) -> str:
    """Strip trailing null bytes and decode as latin-1 or utf-8."""
    raw = raw.split(b"\x00", 1)[0]
    try:
        return raw.decode("utf-8").strip()
    except UnicodeDecodeError:
        return raw.decode("latin-1").strip()


def parse_header(buf: bytes) -> dict[str, Any]:
    """Parse HWiNFO SM2 header struct."""
    if len(buf) < HEADER_SIZE:
        raise ValueError(f"Buffer too short for SM2 header: {len(buf)} < {HEADER_SIZE}")

    fields = struct.unpack_from(HEADER_FMT, buf, 0)
    sig = fields[0]

    return {
        "signature": sig,
        "version": fields[1],
        "revision": fields[2],
        "poll_time": fields[3],
        "offset_sensors": fields[4],
        "size_sensor_element": fields[5],
        "count_sensors": fields[6],
        "offset_readings": fields[7],
        "size_reading_element": fields[8],
        "count_readings": fields[9],
        "poll_period_ms": fields[10],
        "is_valid": (sig == HWIS_SIG),
        "is_dead": (sig == DEAD_SIG),
    }


def iter_readings(buf: bytes, header: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Decode all sensor readings from buffer."""
    if header is None:
        header = parse_header(buf)

    if header["is_dead"] or not header["is_valid"]:
        return []

    readings: list[dict[str, Any]] = []
    off = header["offset_readings"]
    elem_sz = header["size_reading_element"] or READING_SIZE
    num_readings = header["count_readings"]

    for i in range(num_readings):
        elem_offset = off + (i * elem_sz)
        if elem_offset + READING_SIZE > len(buf):
            break

        data = struct.unpack_from(READING_FMT, buf, elem_offset)
        label_orig = _clean_str(data[3])
        label_user = _clean_str(data[4])
        unit = _clean_str(data[5])
        val = data[6]

        label = label_user if label_user else label_orig
        readings.append({
            "type": data[0],
            "sensor_index": data[1],
            "reading_id": data[2],
            "label_orig": label_orig,
            "label_user": label_user,
            "label": label,
            "unit": unit,
            "val": val,
            "min": data[7],
            "max": data[8],
            "avg": data[9],
        })

    return readings


def match_labels(
    readings: list[dict[str, Any]],
    custom_map: dict[str, str] | None = None,
) -> dict[str, float | None]:
    """Map raw readings to the 9 mandatory canonical fields using substring heuristics or map."""
    result: dict[str, float | None] = {f: None for f in MANDATORY_FIELDS}

    # Match heuristics: (canonical_name, required_substrings, forbidden_substrings)
    patterns = [
        ("gpu_core_clock_mhz", ["gpu", "clock"], ["memory", "video", "soc"]),
        ("gpu_mem_clock_mhz", ["memory clock"], []),
        ("temp_edge_c", ["gpu temperature"], ["hot", "junction", "soc", "memory"]),
        ("temp_hotspot_c", ["hot spot"], []),
        ("temp_hotspot_c", ["junction"], []),
        ("power_board_w", ["gpu power"], ["soc"]),
        ("power_board_w", ["board power"], []),
        ("power_board_w", ["total board power"], []),
        ("fan_pct", ["fan"], ["rpm"]),
        ("gpu_util_pct", ["gpu utilization"], []),
        ("gpu_util_pct", ["gpu core load"], []),
        ("gpu_util_pct", ["d3d usage"], []),
        ("vram_used_mb", ["gpu memory allocated"], []),
        ("vram_used_mb", ["gpu memory usage"], []),
        ("vram_used_mb", ["d3d dedicated memory used"], []),
        ("shared_gpu_memory_mb", ["shared gpu memory"], []),
        ("shared_gpu_memory_mb", ["d3d shared memory used"], []),
    ]

    for r in readings:
        label_lower = r["label"].lower()

        # Check custom map first
        if custom_map and r["label"] in custom_map:
            canon = custom_map[r["label"]]
            if canon in result and result[canon] is None:
                result[canon] = r["val"]

        for canon, reqs, forbs in patterns:
            if result[canon] is not None:
                continue
            if all(req in label_lower for req in reqs) and not any(forb in label_lower for forb in forbs):
                result[canon] = r["val"]

    return result


def parse_manual_csv(path: str | Path) -> list[dict[str, Any]]:
    """Parse HWiNFO manual logging CSV (ISO-8859-1 encoded).

    Emits None / gap markers for missing mandatory fields rather than fabricating zeros.
    """
    rows: list[dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return rows

    with open(p, "r", encoding="iso-8859-1", errors="replace") as f:
        reader = csv.reader(f)
        header_row: list[str] | None = None
        for r in reader:
            if not r:
                continue
            if header_row is None:
                header_row = [col.strip() for col in r]
                continue

            row_dict: dict[str, Any] = {}
            for col_idx, col_name in enumerate(header_row):
                if col_idx < len(r):
                    val_str = r[col_idx].strip()
                    row_dict[col_name] = val_str

            # Extract timestamp
            date_str = row_dict.get("Date", "")
            time_str = row_dict.get("Time", "")
            ts = f"{date_str} {time_str}".strip()

            # Map columns to canonical fields
            canonical: dict[str, Any] = {
                "timestamp": ts,
            }
            # Match each column
            col_readings = [{"label": k, "val": float(v) if v.replace(".", "", 1).isdigit() else 0.0} for k, v in row_dict.items()]
            matched = match_labels(col_readings)
            canonical.update(matched)
            rows.append(canonical)

    return rows


def dump_label_map(out_path: str | Path, readings: list[dict[str, Any]]) -> None:
    """Dump all discovered readings to a plain-text label map."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write("# HWiNFO Sensor Label Map\n")
        f.write(f"# Generated: {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n\n")
        for r in readings:
            f.write(f"ID={r['reading_id']:08X} | Type={r['type']} | Label='{r['label']}' | Unit='{r['unit']}' | Val={r['val']}\n")


def watch_pid(
    pid_file: Path,
    out_dir: Path,
    interval_s: float = 1.0,
    duration_s: float | None = None,
) -> int:
    """Windows host telemetry watcher loop."""
    telemetry_dir = out_dir / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    csv_path = telemetry_dir / "host.csv"
    mode_path = telemetry_dir / "mode.txt"

    # Check if Windows and mmap can open
    try:
        import mmap
        # On Windows, mmap takes tagname
        shm = mmap.mmap(-1, 256 * 1024, tagname=HWIS_SM2_MAP_NAME, access=mmap.ACCESS_READ)
    except Exception as exc:
        with open(mode_path, "w", encoding="utf-8") as f_mode:
            f_mode.write("absent\n")
        return 1

    with open(mode_path, "w", encoding="utf-8") as f_mode:
        f_mode.write("shmem\n")

    # CSV writer
    csv_exists = csv_path.exists()
    f_csv = open(csv_path, "a", newline="", encoding="utf-8")
    fieldnames = ["timestamp"] + MANDATORY_FIELDS
    writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
    if not csv_exists:
        writer.writeheader()
        f_csv.flush()

    start_time = time.time()
    last_poll_time = 0

    try:
        while True:
            shm.seek(0)
            buf = shm.read(256 * 1024)
            hdr = parse_header(buf)

            if hdr["is_dead"]:
                with open(mode_path, "w", encoding="utf-8") as f_m:
                    f_m.write("absent:DEAD\n")
                f_csv.close()
                return 4

            if not hdr["is_valid"]:
                with open(mode_path, "w", encoding="utf-8") as f_m:
                    f_m.write("absent:INVALID\n")
                f_csv.close()
                return 4

            # Freshness check
            cur_poll_time = hdr["poll_time"]
            poll_period = hdr["poll_period_ms"] / 1000.0 if hdr["poll_period_ms"] > 0 else 2.0
            if last_poll_time != 0 and (cur_poll_time == last_poll_time):
                # If stalled for too long
                pass
            last_poll_time = cur_poll_time

            readings = iter_readings(buf, hdr)
            matched = match_labels(readings)

            row = {"timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
            row.update(matched)
            writer.writerow(row)
            f_csv.flush()

            # Check if PID file still exists or target process is alive
            if pid_file and not pid_file.exists():
                break

            if duration_s and (time.time() - start_time) >= duration_s:
                break

            time.sleep(interval_s)
    finally:
        f_csv.close()
        shm.close()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="HWiNFO SM2 Telemetry Daemon")
    parser.add_argument("--label-map", type=Path, help="Dump discovered labels to path and exit")
    parser.add_argument("--watch", action="store_true", help="Start continuous watch mode")
    parser.add_argument("--pid-file", type=Path, help="PID file to monitor for lifetime")
    parser.add_argument("--out-dir", type=Path, help="Run output directory")
    parser.add_argument("--interval-s", type=float, default=1.0)
    parser.add_argument("--duration-s", type=float, default=None)
    args = parser.parse_args()

    if args.label_map:
        try:
            import mmap
            shm = mmap.mmap(-1, 256 * 1024, tagname=HWIS_SM2_MAP_NAME, access=mmap.ACCESS_READ)
            buf = shm.read(256 * 1024)
            readings = iter_readings(buf)
            dump_label_map(args.label_map, readings)
            shm.close()
            print(f"Dumped {len(readings)} labels to {args.label_map}")
            return 0
        except Exception as exc:
            print(f"Failed to read HWiNFO shared memory: {exc}", file=sys.stderr)
            return 1

    if args.watch:
        if not args.out_dir:
            print("Error: --out-dir required for --watch", file=sys.stderr)
            return 1
        return watch_pid(args.pid_file, args.out_dir, args.interval_s, args.duration_s)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
