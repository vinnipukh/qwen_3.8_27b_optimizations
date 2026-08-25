# Technology Stack

**Analysis Date:** 2026-08-25

## Languages

**Primary:**
- Python 3 (3.11+ era syntax: `X | None` unions, `from __future__ import annotations`) — benchmark harness, host daemons, offline tooling (`benchmarks/bin/`, `benchmarks/lib/`, `benchmarks/host/`, `tools/`)
- HIP C++ (C++17 / `CMAKE_HIP_STANDARD 17`) — custom GPU kernels (`kernels/common/`, `kernels/template/`, `kernels/demo_iq4xs_dequant/`, `kernels/matmul_iq4xs/`)

**Secondary:**
- Bash — gate scripts (`scripts/check_no_ggml.sh`, `benchmarks/tests/smoke_matrix.sh`, `benchmarks/tests/vulkan_gate.sh`)
- PowerShell (invoked from Python) — Windows toast notifications and Vulkan arm builds (`benchmarks/lib/toast.py`, `benchmarks/vulkan/build-vulkan-arm.ps1`)
- C++ (host-side, non-HIP) — CPU reference oracles and comparators (`kernels/*/ref_cpu.cpp`, `test_compare.cpp`, `bench_sweep.cpp`)

## Runtime

**Environment:**
- Windows 11 host + WSL2 guest (Ubuntu 24.04, root-only) — all GPU work runs in the guest
- ROCm 7.2.1 pinned in the guest (HIP 7.2.53211-e1a6bc5663, gcc 13.3.0), accessed through `/dev/dxg` passthrough via librocdxg 1.2.2
- `HSA_ENABLE_DXG_DETECTION=1` is required (persisted at `/etc/profile.d/rocdxg.sh`) for ROCr to enumerate the gfx1100 GPU over the WSL2 DXG path
- Target GPU: AMD Radeon RX 7900 XT (gfx1100), driver 32.0.31041.1004 / Adrenalin 26.10.41 — frozen
- Host `.wslconfig` must set `[wsl2] memory=28GB` (lower values cause DXG ENOMEM during VRAM allocation)

**Package Manager:**
- No Python packaging manifest (no `requirements.txt` / `pyproject.toml`). Dependencies installed via apt/pip directly into the WSL2 guest: `python3-pytest`, `numpy`, `gguf-py`
- CMake 3.21+ + Ninja drive all native builds; no lockfiles

## Frameworks

**Core:**
- HIP / ROCm 7.2.1 — GPU kernel platform (`kernels/CMakeLists.txt` pins `CMAKE_HIP_ARCHITECTURES=gfx1100` and passes `--offload-arch=gfx1100`)
- llama.cpp pinned v0.2.0 @ `bb4caa7540188872173c44d161602d9271386413` — inference engine under test; built `-DGGML_HIP=ON -DGPU_TARGETS=gfx1100 -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF`; source lives guest-side at `/root/llama.cpp` (DrvFs breaks git lock-files)
- gguf-py (`gguf.constants`, `gguf.quants`) — GGUF parsing/dequantization oracle in fixture tooling (`tools/dump_gguf_fixtures.py`, `tools/dump_matmul_fixtures.py`)

**Testing:**
- pytest — 55 unit/regression tests under `benchmarks/tests/` (`pytest.ini` present); run as `PYTHONPATH=. python3 -m pytest benchmarks/tests/ -q`

**Build/Dev:**
- CMake ≥ 3.21 (`cmake_minimum_required(VERSION 3.21)` in `kernels/CMakeLists.txt`) with Ninja generator
- HIP compiler toolchain from `/opt/rocm` (`find_package(hip REQUIRED CONFIG PATHS /opt/rocm/...)`)
- Quilt-style patching of upstream llama.cpp (`patches/phase5_mul_mat_custom.patch`)

## Key Dependencies

**Critical:**
- numpy — tensor fixtures, dequantization math, `.npz` fixture I/O (`tools/dump_gguf_fixtures.py`, `tools/dump_matmul_fixtures.py`, `benchmarks/tests/`)
- gguf-py — reads real IQ4_XS tensors out of the model GGUF; zero llama.cpp runtime dependency
- HWiNFO64 Shared Memory v2 — Windows-host GPU telemetry bridge read by `benchmarks/host/hwinfo_daemon.py` (`Global\HWiNFO_SENS_SM2` map)
- rocprof v3 — kernel-level profiling output parsed by `benchmarks/lib/parse_profile.py`; raw dumps land in `.rocprofv3/` and `benchmarks/profiling/raw/`

**Infrastructure:**
- llama-cli / llama-bench / llama-perplexity / test-backend-ops (stock binaries archived in `baseline/binaries/v0.2.0-bb4caa75/`, gitignored) — measurement subjects invoked via `subprocess`
- Vulkan comparator arm — optional second backend build scripts (`benchmarks/vulkan/run_session_vulkan.py`, `build-vulkan-arm.ps1`)

**Standard library only (no third-party deps):** `benchmarks/lib/guard.py`, `store.py`, `preflight.py`, `fingerprint.py`, `llabench.py`, `toast.py`, `benchmarks/bin/*.py` use only stdlib (`subprocess`, `fcntl`, `json`, `hashlib`, `threading`, etc.)

## Configuration

**Environment:**
- No `.env` files detected; configuration is code + docs driven
- Required env vars set per-subprocess in Python: `HSA_ENABLE_DXG_DETECTION=1`, `LD_LIBRARY_PATH=/root/llama.cpp/build-ci/bin:$LD_LIBRARY_PATH` (see `benchmarks/bin/profile_workload.py`, `tools/ask_model.py`)
- Optional toggle: `GGML_CUDA_DISABLE_GRAPHS=1` to disable graph capture during profiling
- Guard thresholds loaded from JSON config: `benchmarks/config/thresholds.json` (consumed by `benchmarks/lib/guard.py::Thresholds.from_json`; absent file ⇒ observe-only mode)
- Hardware sensor label map: `benchmarks/config/hwinfo_sensor_labels.txt`

**Build:**
- `kernels/CMakeLists.txt` — top-level standalone HIP build; hard-pins gfx1100, exports `compile_commands.json`; conditionally includes `demo_iq4xs_dequant/` and `matmul_iq4xs/` subdirs
- Per-kernel `CMakeLists.txt` under each `kernels/<kernel>/` directory
- llama.cpp build flags documented in `docs/CONFIGURATION.md` and README ("Hardware & software requirements")
- Isolation gate: `scripts/check_no_ggml.sh` fails any build tree that lets `#include <ggml|llama>` leak into `kernels/` (KERN-01 requirement)

## Platform Requirements

**Development:**
- WSL2 Ubuntu 24.04 (root-only), ROCm 7.2.1 toolchain at `/opt/rocm`
- Guest-side source/model paths are mandatory: `/root/llama.cpp`, `/root/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf` (mmap over `/mnt/*` stalls; DrvFs breaks git locks)
- Frozen environment snapshot: `E:\wsl-snapshots\ubuntu-2404-rocm721-phase1.tar`
- Model artifact: `JonathanColetti/Qwen3.8-27B-Uncensored-IQ4_XS.gguf`, 15.31 GB, sha256-verified (provenance in `models/README.md`)

**Production:**
- N/A — research/measurement repository, not a deployable service. "Production" is the frozen benchmark rig itself (Windows host + WSL2 guest + RX 7900 XT)

---

*Stack analysis: 2026-08-25*
