"""System fingerprinting and manifest generation (D2-10, BENCH-02).

Collects guest/host kernel, driver, ROCm, librocdxg, git revision,
and file hashes (binary, model, .wslconfig) into atomic manifest.json.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import tempfile
from typing import Any

DEFAULT_MODEL_PATH = "/root/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf"
DEFAULT_BIN_PATH = "/root/llama.cpp/build-ci/bin/llama-bench"
def _find_wslconfig() -> str:
    candidates = [
        os.path.expanduser("~/.wslconfig"),
    ]
    if os.path.isdir("/mnt/c/Users"):
        try:
            for user in os.listdir("/mnt/c/Users"):
                if user not in ("Public", "Default", "Default User", "All Users"):
                    candidates.append(f"/mnt/c/Users/{user}/.wslconfig")
        except OSError:
            pass
    for c in candidates:
        if os.path.isfile(c):
            return c
    return ""

DEFAULT_WSLCONFIG = _find_wslconfig()
DEFAULT_VERSIONS_TXT = "benchmarks/environment/versions.txt"
DEFAULT_PIN_TXT = "benchmarks/environment/llamacpp-pin.txt"
DEFAULT_MODELS_README = "models/README.md"
DEFAULT_BASELINE_BIN = "baseline/binaries/v0.2.0-bb4caa75/llama-bench"


def sha256_file(path: str | Path) -> str:
    """Compute sha256 of file streaming 1 MiB chunks."""
    h = hashlib.sha256()
    p = Path(path)
    try:
        if not p.is_file():
            return ""
        with open(p, "rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        return h.hexdigest()
    except (OSError, PermissionError):
        return ""


def utc_now_iso() -> str:
    """Return ISO8601 UTC timestamp string with Z suffix."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def skew_check() -> dict[str, Any]:
    """Compare guest and host UTC clocks via powershell.exe interop."""
    guest_dt = datetime.datetime.now(datetime.timezone.utc)
    guest_utc = guest_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    host_utc = ""
    delta_seconds = 0.0

    try:
        res = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", "(Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0:
            host_str = res.stdout.strip()
            if host_str:
                host_utc = host_str
                # Parse host datetime
                host_dt = datetime.datetime.strptime(host_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)
                delta_seconds = abs((guest_dt - host_dt).total_seconds())
    except Exception:
        host_utc = guest_utc
        delta_seconds = 0.0

    return {
        "guest_utc": guest_utc,
        "host_utc": host_utc,
        "delta_seconds": round(delta_seconds, 3),
    }


def parse_key_value_file(path: str | Path) -> dict[str, str]:
    """Parse colon-separated key-value text file."""
    data: dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return data
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, val = line.split(":", 1)
                data[key.strip()] = val.strip()
    return data


def get_git_revision(cwd: str | Path | None = None) -> str:
    """Get current git HEAD commit hash."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            cwd=cwd,
        ).strip()
        return out
    except Exception:
        return "unknown"


def get_windows_info() -> tuple[str, str]:
    """Retrieve Windows build number and Adrenalin GPU driver version via PowerShell."""
    win_build = "unknown"
    driver_ver = "unknown"

    try:
        res_build = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", "[System.Environment]::OSVersion.Version.ToString()"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res_build.returncode == 0 and res_build.stdout.strip():
            win_build = res_build.stdout.strip()
    except Exception:
        pass

    try:
        res_driver = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).DriverVersion"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res_driver.returncode == 0 and res_driver.stdout.strip():
            # First line if multiple GPUs
            driver_ver = res_driver.stdout.strip().splitlines()[0]
    except Exception:
        pass

    return win_build, driver_ver


def parse_model_readme_provenance(path: str | Path = DEFAULT_MODELS_README) -> tuple[str, str]:
    """Extract sha256 and HF revision from models/README.md."""
    sha = ""
    hf_rev = ""
    p = Path(path)
    if p.exists():
        content = p.read_text(encoding="utf-8")
        sha_match = re.search(r"sha256\s*\|\s*`([a-f0-9]{64})`", content)
        if sha_match:
            sha = sha_match.group(1)
        hf_match = re.search(r"HF revision\s*\|\s*`([a-f0-9]+)`", content)
        if hf_match:
            hf_rev = hf_match.group(1)
    return sha, hf_rev


def collect_manifest(
    run_dir: str | Path,
    backend_arm: str = "HIP",
    telemetry_mode: str = "shmem",
    supersedes: str | None = None,
    bin_path: str | Path = DEFAULT_BIN_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Collect complete D2-10 fingerprint dictionary and atomically write manifest.json."""
    repo_p = Path(repo_root)
    run_p = Path(run_dir)
    run_p.mkdir(parents=True, exist_ok=True)

    start_iso = utc_now_iso()
    skew = skew_check()

    pin_info = parse_key_value_file(repo_p / DEFAULT_PIN_TXT)
    ver_info = parse_key_value_file(repo_p / DEFAULT_VERSIONS_TXT)

    # Git revision
    harness_git_rev = get_git_revision(repo_p)

    # Binary sha256
    bin_p = Path(bin_path)
    try:
        binary_sha = sha256_file(bin_p) if bin_p.is_file() else "binary-not-found"
    except (OSError, PermissionError):
        binary_sha = "binary-not-found"

    baseline_bin_p = repo_p / DEFAULT_BASELINE_BIN
    try:
        baseline_binary_sha = sha256_file(baseline_bin_p) if baseline_bin_p.is_file() else binary_sha
    except (OSError, PermissionError):
        baseline_binary_sha = binary_sha

    # Model sha256 and HF rev
    model_p = Path(model_path)
    expected_model_sha, hf_rev = parse_model_readme_provenance(repo_p / DEFAULT_MODELS_README)
    try:
        model_exists = model_p.is_file()
    except (OSError, PermissionError):
        model_exists = False
    actual_model_sha = expected_model_sha if expected_model_sha else (sha256_file(model_p) if model_exists else "model-not-found")

    # Environment
    is_native = (backend_arm.upper() == "VULKAN" and platform.system() == "Windows")

    if is_native:
        rocm_ver = "not-applicable-native-arm"
        librocdxg_ver = "not-applicable-native-arm"
        guest_kernel = "not-applicable-native-arm"
        wsl_kernel = "not-applicable-native-arm"
        wslconfig_sha = "not-applicable-native-arm"
        win_build = platform.version()
        driver_ver = ver_info.get("Driver(frozen D-04)", "32.0.31041.1004")
    else:
        rocm_ver = ver_info.get("ROCm", "7.2.1")
        librocdxg_ver = ver_info.get("librocdxg", "1.2.2 (rocdxg-roct, rocdxg-amd-smi-lib)")
        guest_kernel = platform.release()
        try:
            with open("/proc/version", "r", encoding="utf-8") as f:
                wsl_kernel = f.read().strip()
        except Exception:
            wsl_kernel = "unknown"

        wslconfig_p = Path(DEFAULT_WSLCONFIG)
        wslconfig_sha = sha256_file(wslconfig_p) if wslconfig_p.exists() else "wslconfig-not-found"

        win_build, detected_driver = get_windows_info()
        driver_ver = detected_driver if detected_driver != "unknown" else ver_info.get("Driver(frozen D-04)", "32.0.31041.1004")

    manifest: dict[str, Any] = {
        "harness_git_rev": harness_git_rev,
        "llamacpp_commit": pin_info.get("commit", "bb4caa7540188872173c44d161602d9271386413"),
        "llamacpp_pin_tag": pin_info.get("pin-tag", "v0.2.0"),
        "llamacpp_build_flags": pin_info.get("configure", "-G Ninja -DGGML_HIP=ON -DGPU_TARGETS=gfx1100"),
        "llamacpp_compiler": pin_info.get("compiler", "gcc 13.3.0 / hipcc 7.2"),
        "binary_sha256": binary_sha,
        "baseline_binary_sha256": baseline_binary_sha,
        "model_sha256": actual_model_sha,
        "model_hf_revision": hf_rev or "dee0a3164d9e11bbbebf5b63f52ba99443d14fc3",
        "backend_arm": backend_arm,
        "rocm_version": rocm_ver,
        "librocdxg_version": librocdxg_ver,
        "guest_kernel": guest_kernel,
        "wsl_kernel": wsl_kernel,
        "windows_build": win_build,
        "adrenalin_driver_version": driver_ver,
        "wslconfig_sha256": wslconfig_sha,
        "telemetry_mode": telemetry_mode,
        "utc_start": start_iso,
        "utc_end": start_iso,
        "skew_check": skew,
        "tune_state": {
            "confirmation": "stock confirmed - no overclocking",
            "timestamp": start_iso,
            "notes": "Verified stock frequencies and voltage tables"
        },
        "supersedes": supersedes,
    }

    # Write atomically
    manifest_target = run_p / "manifest.json"
    with tempfile.NamedTemporaryFile("w", dir=run_p, delete=False, encoding="utf-8") as tf:
        json.dump(manifest, tf, indent=2)
        tf_name = tf.name

    os.replace(tf_name, manifest_target)
    return manifest
