"""Unit tests for fingerprinting and manifest generation."""
import json
from pathlib import Path
import pytest
from benchmarks.lib import fingerprint


def test_sha256_file(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("hello world", encoding="utf-8")
    expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    assert fingerprint.sha256_file(f) == expected


def test_skew_check():
    skew = fingerprint.skew_check()
    assert "guest_utc" in skew
    assert "host_utc" in skew
    assert "delta_seconds" in skew
    assert isinstance(skew["delta_seconds"], (int, float))
    assert skew["delta_seconds"] < 10.0


def test_collect_manifest_completeness(tmp_path):
    run_dir = tmp_path / "run_test"
    manifest = fingerprint.collect_manifest(
        run_dir=run_dir,
        backend_arm="HIP",
        telemetry_mode="shmem",
        supersedes=None,
    )

    manifest_file = run_dir / "manifest.json"
    assert manifest_file.exists()
    loaded = json.loads(manifest_file.read_text(encoding="utf-8"))

    # Required D2-10 fields
    required_keys = [
        "harness_git_rev",
        "llamacpp_commit",
        "llamacpp_pin_tag",
        "llamacpp_build_flags",
        "llamacpp_compiler",
        "binary_sha256",
        "baseline_binary_sha256",
        "model_sha256",
        "model_hf_revision",
        "backend_arm",
        "rocm_version",
        "librocdxg_version",
        "guest_kernel",
        "wsl_kernel",
        "windows_build",
        "adrenalin_driver_version",
        "wslconfig_sha256",
        "telemetry_mode",
        "utc_start",
        "utc_end",
        "skew_check",
        "tune_state",
        "supersedes",
    ]

    for k in required_keys:
        assert k in loaded, f"Missing key in manifest: {k}"
        if k != "supersedes":  # supersedes can be None
            assert loaded[k] is not None, f"Key {k} is None"
            assert str(loaded[k]) != "", f"Key {k} is empty"


def test_manifest_vulkan_native_degradation(tmp_path):
    run_dir = tmp_path / "run_vulkan"
    manifest = fingerprint.collect_manifest(
        run_dir=run_dir,
        backend_arm="Vulkan",
        telemetry_mode="absent",
    )
    # If run on Windows or tested with Vulkan arm
    assert manifest["backend_arm"] == "Vulkan"
    assert manifest["telemetry_mode"] == "absent"


def test_build_kill_command_interop_string():
    from benchmarks.host.thermal_watchdog import build_kill_command

    cmd = build_kill_command(4242, mode="wsl")
    assert cmd == "wsl.exe -d Ubuntu-24.04 -u root -- sh -c 'kill -9 4242'"

    cmd_native = build_kill_command(4242, mode="native")
    assert cmd_native == "taskkill /PID 4242 /F"

    # Reject non-integer PID
    with pytest.raises(ValueError, match="Invalid PID"):
        build_kill_command("4242; rm -rf /", mode="wsl")


def test_build_toast_xml_escaping():
    from benchmarks.lib.toast import build_toast_xml

    title = "Test & Alert <Critical>"
    body = "Junction temp > 95°C & load == 100%"
    xml_str = build_toast_xml(title, body)

    assert "Test &amp; Alert &lt;Critical&gt;" in xml_str
    assert "Junction temp &gt; 95°C &amp; load == 100%" in xml_str


def test_send_summary_formatting():
    from benchmarks.lib.toast import send_summary
    # We test body format via build_toast_xml
    from benchmarks.lib.toast import build_toast_xml

    xml_str = build_toast_xml("Benchmark Session Complete", "16 OK / 0 FAILED")
    assert "<text>16 OK / 0 FAILED</text>" in xml_str

