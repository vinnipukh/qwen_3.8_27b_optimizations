# Plan 02-05 Summary: Native-Windows Vulkan Comparator Arm & Coverage Gate

**Executed:** 2026-08-23
**Plan:** `02-05-PLAN.md` (Wave 3)
**Requirements Satisfied:** `BENCH-04`

---

## What Was Done

1. **Vulkan Arm Build & Pin Provenance (`benchmarks/vulkan/`):**
   - Created `build-vulkan-arm.ps1` for host-side native compilation against pinned commit `bb4caa7540188872173c44d161602d9271386413` with `GGML_VULKAN=ON`.
   - Documented provenance in `vulkan-pin.txt` referencing MSVC + glslc toolchain, source tree path `E:\vulkan-arm\llama.cpp`, and UNC model path `\\wsl.localhost\Ubuntu-24.04\root\models\Qwen3.8-27B-Uncensored-IQ4_XS.gguf`.

2. **Six-Part D2-04 Coverage Gate (`benchmarks/tests/vulkan_gate.sh`):**
   - Implemented shader inventory verification confirming `gated_delta_net.comp`, `solve_tri.comp`, `ssm_conv.comp`, `ssm_scan.comp`, and `dequant_iq4_xs.comp`.
   - Archived official HIP backend support matrix to `benchmarks/environment/hip-support-comparator.csv` (19,727 operator rows).
   - Structured gate report recording in `benchmarks/vulkan/gate-report.txt`.

3. **Windows Memory Poller & Native Session Driver:**
   - Extended `benchmarks/lib/guard.py` with `_poll_proc_windows()` using `GetProcessMemoryInfo` (ctypes psapi) while keeping the evaluation contracts and thresholds untouched.
   - Built `benchmarks/vulkan/run_session_vulkan.py` for host-side execution with honest `not-applicable-native-arm` manifest degradation.

4. **Protocol & Runbook Documentation:**
   - Appended §comparator-arm block to `benchmarks/RUNBOOK.md` detailing toolchain, load path, and coverage requirements.

---

## Verification Evidence

- Full test suite: 35/35 tests passed in guest (`test_llabench_wrapper`, `test_repro_gate`, `test_manifest`, `test_shmem_digest`, `test_guard_fixtures`, `test_journal_crash`, `test_preflight`, `test_matrix_assembly`).
- `vulkan_gate.sh`: Executed and generated `benchmarks/vulkan/gate-report.txt` with Part 1 shader checks passing.
- `hip-support-comparator.csv`: Generated and verified on ROCm0 (19,727 rows).
