import json
import pytest
from pathlib import Path
from benchmarks.lib import parse_profile


MOCK_RAW_PROFILE = {
    "prompt_tokens": 128,
    "gen_tokens": 128,
    "prefill_wall_ms": 250.0,
    "decode_wall_ms": 4000.0,
    "total_prefill_gpu_us": 200000.0,
    "total_decode_gpu_us": 3800000.0,
    "total_overall_gpu_us": 4000000.0,
    "prefill_summary": {
        "MUL_MAT": {"pct": 52.0, "total_us": 104000.0, "count": 497, "avg_us": 209.25},
        "GATED_DELTA_NET": {"pct": 10.0, "total_us": 20000.0, "count": 48, "avg_us": 416.6},
        "RMS_NORM": {"pct": 8.0, "total_us": 16000.0, "count": 209, "avg_us": 76.5},
    },
    "decode_summary": {
        "MUL_MAT": {"pct": 32.0, "total_us": 1216000.0, "count": 63616, "avg_us": 19.1},
        "MUL": {"pct": 14.0, "total_us": 532000.0, "count": 41088, "avg_us": 12.9},
        "RMS_NORM": {"pct": 10.0, "total_us": 380000.0, "count": 26752, "avg_us": 14.2},
    },
    "overall_summary": {
        "MUL_MAT": {"pct": 33.0, "total_us": 1320000.0, "count": 64113, "avg_us": 20.5},
        "MUL": {"pct": 13.5, "total_us": 540000.0, "count": 41409, "avg_us": 13.0},
    },
}


def test_parse_profile_json(tmp_path: Path):
    prof_file = tmp_path / "mock_prof.json"
    with open(prof_file, "w") as f:
        json.dump(MOCK_RAW_PROFILE, f)

    parsed = parse_profile.parse_profile_json(prof_file)
    assert parsed["prompt_tokens"] == 128
    assert parsed["gen_tokens"] == 128
    assert len(parsed["ranked_prefill"]) == 3
    assert parsed["ranked_prefill"][0]["op"] == "MUL_MAT"
    assert parsed["ranked_prefill"][0]["bound_type"] == "Memory Bandwidth / Dequant Bound"
    assert parsed["ranked_decode"][0]["op"] == "MUL_MAT"


def test_format_markdown_table():
    sample_ranked = [
        {"op": "MUL_MAT", "pct": 31.12, "total_ms": 88033.03, "count": 2000, "avg_us": 200.0, "bound_type": "Memory Bandwidth / Dequant Bound"},
        {"op": "GATED_DELTA_NET", "pct": 4.55, "total_ms": 6376.70, "count": 336, "avg_us": 892.1, "bound_type": "Compute / Register Bound"},
    ]
    md = parse_profile.format_markdown_table(sample_ranked)
    assert "MUL_MAT" in md
    assert "31.12%" in md
    assert "GATED_DELTA_NET" in md


def test_saved_bottleneck_table_and_summary():
    summary_file = Path("benchmarks/profiling/bottleneck_summary.json")
    report_file = Path("benchmarks/profiling/BOTTLENECK-TABLE.md")
    dispatch_file = Path("benchmarks/profiling/dispatch_overhead_report.md")

    assert summary_file.exists(), "bottleneck_summary.json must exist"
    assert report_file.exists(), "BOTTLENECK-TABLE.md must exist"
    assert dispatch_file.exists(), "dispatch_overhead_report.md must exist"

    with open(summary_file, "r") as f:
        sdata = json.load(f)

    assert "optimization_target_1" in sdata
    assert sdata["optimization_target_1"] == "MUL_MAT"
    assert len(sdata["shape_profiles"]) == 4

    with open(report_file, "r") as f:
        report_text = f.read()

    assert "PRIMARY OPTIMIZATION TARGET #1" in report_text
    assert "MUL_MAT" in report_text
    assert "S1_interactive" in str(sdata["shape_profiles"].keys())
