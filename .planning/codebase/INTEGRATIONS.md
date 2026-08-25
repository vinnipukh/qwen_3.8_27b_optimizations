# External Integrations

**Analysis Date:** 2026-08-25

This is an offline GPU-kernel research repository. It has **no network-facing services, no
databases, and no auth providers**. "Integrations" here means local process invocation,
host/guest telemetry bridges, and pinned external artifacts.

## APIs & External Services

**None (no cloud/network APIs).**
- llama.cpp is built with `-DLLAMA_CURL=OFF`; no HTTP calls anywhere in the harness.

**Local inference engine (process integration, not API):**
- llama.cpp v0.2.0 @ `bb4caa7540188872173c44d161602d9271386413`
  - Invoked as: `/root/llama.cpp/build-ci/bin/llama-cli`, `llama-bench`, `llama-perplexity`, `test-backend-ops` via `subprocess` (`benchmarks/lib/llabench.py`, `benchmarks/bin/profile_workload.py`, `tools/ask_model.py`)
  - Contract: JSON output via `--out-json`, exit-code + output-line parsing; headless runs require `setsid --single-turn --simple-io --load-mode none`

## Data Storage

**Databases:**
- None. All persistence is append-only files on disk.

**File Storage:**
- Local filesystem only, two roots:
  - Repo-side results journals: `benchmarks/results/<run_id>/` containing `rows.jsonl`, `manifest.json`, `CHECKSUMS.sha256`, plus `logs/`, `run/`, `telemetry/` subdirs (written by `benchmarks/lib/store.py`)
  - Guest-side artifacts (outside repo): model at `/root/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf`, binaries at `/root/llama.cpp/build-ci/bin/`
- Fixture store: `kernels/fixtures/*.npz|*.bin|manifest.json|manifest_matmul.json` — extracted GGUF tensors and synthetic IQ4_XS fixtures

**Caching:**
- None (only Python `__pycache__/`)

## Authentication & Identity

**Auth Provider:** None. Single-user root-only WSL2 guest; no auth anywhere.
**Session concurrency control:** `fcntl.flock` non-blocking lock at `benchmarks/results/.session.lock` (`benchmarks/bin/run_session.py`) — exits with code 5 if another session holds it.

## Monitoring & Observability

**Error Tracking:**
- None (no Sentry etc.). Failures surface through verdict vocabulary in the harness (`OK`, `FAILED:suspected-spill`, `REVIEW:repeat-deviation`, `FAILED:thermal-abort`, `FAILED:preflight-oom` — locked constants in `benchmarks/lib/guard.py`)

**Logs / Telemetry:**
- Windows toast notifications sent from WSL via PowerShell interop (`benchmarks/lib/toast.py` → `powershell.exe -File ...` with `Windows.UI.Notifications.ToastNotificationManager`)
- GPU telemetry: HWiNFO64 Shared Memory v2 bridge on the host, read by `benchmarks/host/hwinfo_daemon.py` (`Global\HWiNFO_SENS_SM2` map, sensor labels mapped via `benchmarks/config/hwinfo_sensor_labels.txt`)
- Thermal watchdog: `benchmarks/host/thermal_watchdog.py` kills runaway `wsl.exe` processes at 95 °C
- VRAM spill guard: polls guest `/proc/<pid>/status` VmRSS/VmSwap + shared-GPU-memory climb (`benchmarks/lib/guard.py`)
- Kernel profiling: rocprof v3 dumps parsed by `benchmarks/lib/parse_profile.py`; raw output under `.rocprofv3/` and `benchmarks/profiling/raw/`
- Environment fingerprints archived in `benchmarks/environment/` (`versions.txt`, `hipconfig.txt`, `rocminfo.txt`, `vram-probe.txt`, `startup-log.txt`)

## CI/CD & Deployment

**Hosting:**
- None. Runs only on the frozen local rig (Windows 11 + WSL2 Ubuntu 24.04 + RX 7900 XT).

**CI Pipeline:**
- None detected (no GitHub Actions / GitLab CI config). Gates are run manually:
  - Op gate: `benchmarks/bin/run_op_gate.py` (21,093 backend ops must pass)
  - Model gate: `benchmarks/bin/run_model_gate.py` (perplexity PPL 6.4271 ± 0.04103 + 6 golden prompt canaries in `benchmarks/golden/`)
  - Isolation gate: `scripts/check_no_ggml.sh` (zero ggml/llama includes in `kernels/`)
  - Test suite: 55 pytest tests in `benchmarks/tests/`

## Environment Configuration

**Required env vars (set per-subprocess inside scripts):**
- `HSA_ENABLE_DXG_DETECTION=1` — mandatory for ROCr over WSL2 DXG path (persisted in `/etc/profile.d/rocdxg.sh`)
- `LD_LIBRARY_PATH=/root/llama.cpp/build-ci/bin:$LD_LIBRARY_PATH` — loads pinned HIP GGML shared objects (`tools/ask_model.py`, `benchmarks/bin/profile_workload.py`)
- Optional: `GGML_CUDA_DISABLE_GRAPHS=1` — disable graph capture during profiling runs

**Host-level configuration (not env vars):**
- `.wslconfig`: `[wsl2] memory=28GB`, `swap=16GB` — required or VRAM allocation fails with DXG ENOMEM
- Guard thresholds file: `benchmarks/config/thresholds.json` (loaded by `benchmarks/lib/guard.py`)

**Secrets location:**
- No `.env` files, credential files, or token stores detected in the repo; nothing to rotate. Model artifact integrity is enforced by sha256 recorded in `models/README.md`.

## Webhooks & Callbacks

**Incoming:** None.
**Outgoing:** None (Windows toast notifications are local OS interop, not network callbacks).

## Pinned External Artifacts (provenance chain)

- llama.cpp source pin: `bb4caa7540188872173c44d161602d9271386413` — constant `PIN_COMMIT` in `benchmarks/lib/llabench.py`; archived binaries in `baseline/binaries/v0.2.0-bb4caa75/`
- Model: `JonathanColetti/Qwen3.8-27B-Uncensored-IQ4_XS.gguf` (15.31 GB, sha256-verified) — provenance in `models/README.md`
- ROCm 7.2.1 / HIP 7.2.53211-e1a6bc5663 / librocdxg 1.2.2 — fingerprinted in `benchmarks/environment/`
- Upstream patches: quilt-style overlay at `patches/phase5_mul_mat_custom.patch`

---

*Integration audit: 2026-08-25*
