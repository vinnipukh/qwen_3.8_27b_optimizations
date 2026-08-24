# Benchmark Runbook & Protocol

**Scope:** Execution protocol, guard thresholds, telemetry modes, and thermal policy for Qwen3.8-27B benchmark harness on AMD Radeon RX 7900 XT (ROCm 7.2.1 + ROCDXG / Vulkan comparator).

---

## §session-protocol
*Dated Amendment: 2026-08-23*

1. **Pre-flight & Environment Verification:**
   - Confirm stock GPU tune state (no core/mem overclocking, stock power profile verified via driver query).
   - Ensure HWiNFO64 is running with Shared Memory Support enabled (or verify manual CSV fallback / absent fallback).
   - Check host-guest clock skew (`delta_seconds` < 2s).
   - Evaluate pre-flight buffer math against measured DXG free VRAM anchor (18245 MiB free of 20421 MiB).
   - **Device Pre-flight:** Run `rocminfo` under `HSA_ENABLE_DXG_DETECTION=1` to confirm GPU responsiveness prior to heavy runs.
   - **Step-Up Verification Discipline:** New builds/kernels must be tested on CPU first (`-ngl 0`), then partial offload (`-ngl 10`), before full offload (`-ngl 99`) to prevent DXG deadlocks or silent Driver Timeouts (TDR).
   - **MANDATORY TIMEOUTS:** Every bash command and harness subprocess call MUST have an explicit, bounded timeout to prevent hangs.

2. **Session Execution:**
   - Acquire exclusive session flock on `benchmarks/results/.session.lock` (exit code 5 on collision).
   - Spawn host-side telemetry watcher (`hwinfo_daemon.py`) and thermal watchdog (`thermal_watchdog.py`).
   - Execute tiers in strictly ascending context order: `4096`, `8192`, `16384`, `32768` (D2-19 fixed lifetime order).
   - Within each tier, execute `flash_attn off` then `flash_attn on`.
   - Zero default vectors via `-n 0` and pass explicit `-p C` and `-pg C,128` (D2-06).
   - Each cell executes with 5 repetitions (`-r 5`), warmup enabled (D2-07), and cooldown delay (`--delay 30`).
   - Fsync rows immediately to `rows.jsonl`.
   - Pre-flight OOM gate: Tier 32768 is checked prior to allocation; if over budget, publishes `FAILED:preflight-oom` rows without crashing (D2-18).

3. **Session Close:**
   - Close telemetry stream and compute session summary.
   - Write `CHECKSUMS.sha256` covering all artifacts in the run directory in `sha256sum -c` format.
   - Append summary to `benchmarks/results/index.jsonl`.
   - Release session lock and dispatch completion toast alert ("N OK / M FAILED").

---

## §thresholds
*Dated Amendment: 2026-08-23 (Derived from calibration run: 20260823_163954_calibration_profile)*

- **Measured Peak VmRSS (Steady-State):** `15,192,572 kB` (14.49 GB)
- **VmRSS Fail Threshold (1.5x Margin):** `22,788,858 kB` (21.73 GB)
- **Measured Peak VmSwap:** `0 kB`
- **VmSwap Fail Threshold:** `524,288 kB` (512.00 MB allowance before flagging spill)
- **Shared-GPU-Memory Climb Ceiling:** `250.0 MB/min`
- **Intra-cell Repeat Deviation Max:** `50%` (max/min ratio ceiling: `2.0x`)
- **Reproducibility Gate:** `±5.0%` deviation between session means (BENCH-01)

---

## §telemetry-modes
*Dated Amendment: 2026-08-23*

1. **`shmem` (Primary):**
   - HWiNFO64 Shared Memory v2 mapped via `Global\HWiNFO_SENS_SM2`.
   - 1 Hz continuous polling for clocks, temperatures, power, fan %, GPU util, VRAM used, shared-GPU-memory.
   - Synchronized by UTC timestamp with guest execution log.

2. **`manual-fallback` (Secondary):**
   - HWiNFO64 sensor CSV logging configured to write to designated run directory.
   - Post-processed after session close via `hwinfo_daemon.py --parse-csv` with ISO-8859-1 decoding and gap markers.

3. **`absent` (Degraded):**
   - No host telemetry available. Manifest explicitly notes `telemetry_mode: absent`.
   - In-guest `/proc/<pid>/status` RSS monitoring remains active.

---

## §thermal-policy
*Dated Amendment: 2026-08-23*

- **Junction/Hotspot Kill Threshold:** `95.0 °C`.
- **Action:** Host-side `thermal_watchdog.py` issues cross-boundary `wsl.exe` kill signal to running benchmark process (`wsl.exe -d Ubuntu-24.04 -u root -- sh -c 'kill -9 <pid>'`).
- **Rehearsal Evidence:** Successfully proven on 2026-08-23 against dummy PID 512 (`REHEARSAL_PASS`).
- **Verdict:** Affected cell marked `FAILED:thermal-abort`.
- **Notification:** Immediate Windows toast alert dispatched to user.
- **Fan Control:** Purely hardware-managed (no software fan override scripts permitted).

---

## §operational-ceiling & busy-spin-fingerprint
*Dated Amendment: 2026-08-23*

- **HWiNFO Shared Memory Duty Cycle:** Non-Pro HWiNFO64 limits shared memory interface to 12 hours continuous per calendar day. Benchmark sessions must schedule execution windows within this active allowance.
- **librocdxg#60 Busy-Spin Observation:** The ROCDXG KFD emulation runtime exhibits active polling during kernel execution, driving 100% load on one CPU core. Taskset CPU pinning is evaluated: default scheduler allocation is retained for baseline parity, with taskset isolation documented for kernel evaluation phases.
- **Near-OOM Supervised Window Evidence:** Run `20260823_164634_calibration_near_oom` executed tier 32768; preflight gate intercepted allocation against 18245 MiB DXG anchor and cleanly published 4 `FAILED:preflight-oom` rows without system lockup.

---

## §comparator-arm (Stock Vulkan)
*Dated Amendment: 2026-08-23*

- **Arm Location:** Separate native Windows source tree (`E:\vulkan-arm\llama.cpp`), keeping zero DrvFs/WSL file lock contention.
- **Pin & Backend:** Pinned commit `bb4caa7540188872173c44d161602d9271386413` with `GGML_VULKAN=ON` (identical commit to HIP baseline).
- **Toolchain:** MSVC (VS2022 Build Tools) + glslc (LunarG Vulkan SDK) against Adrenalin driver `32.0.31041.1004`.
- **Model Load Path:** Loaded via `\\wsl.localhost\Ubuntu-24.04\root\models\Qwen3.8-27B-Uncensored-IQ4_XS.gguf` or local native copy as recorded in `benchmarks/vulkan/vulkan-pin.txt`.
- **Coverage Gate:** Governed by six-part `benchmarks/tests/vulkan_gate.sh` asserting static shader inventory, backend op parity (`GATED_DELTA_NET`, `SOLVE_TRI`, `SSM_CONV`, `SSM_SCAN`), 132/132 layer GPU residency, and greedy decode coherence.

