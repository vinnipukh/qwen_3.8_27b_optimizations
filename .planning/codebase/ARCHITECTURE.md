<!-- refreshed: 2026-08-25 -->
# Architecture

**Analysis Date:** 2026-08-25 (Updated Phase 6 / v1.0.0-gfx1100)

## System Overview

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                    Windows Host (driver + telemetry)                     │
│   Adrenalin driver 26.2.2, HWiNFO64 SM2 bridge, thermal watchdog        │
│   `[benchmarks/host/hwinfo_daemon.py]` `[benchmarks/host/thermal_watchdog.py]` │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │ /dev/dxg passthrough (librocdxg 1.2.2)
┌───────────────────────────────▼──────────────────────────────────────────┐
│              WSL2 Guest: Ubuntu 24.04 + ROCm 7.2.1 (pinned)              │
│                                                                          │
│  ┌────────────────────────────┐   ┌───────────────────────────────────┐ │
│  │ Benchmark Harness (Python) │   │ Kernel Playground (HIP/C++17)     │ │
│  │ `[benchmarks/bin/*]`       │   │ `[kernels/*]`                     │ │
│  │  run_session.py (orchestr.)│   │  ref_cpu → impl.hip →             │ │
│  │  run_op_gate.py            │   │  test_compare → bench_sweep       │ │
│  │  run_model_gate.py         │   │  zero llama.cpp headers           │ │
│  │  profile_matrix.py         │   └──────────────┬────────────────────┘ │
│  │ `[benchmarks/lib/*]`       │                  │                      │
│  │  llabench/guard/store/     │                  │                      │
│  │  preflight/fingerprint     │                  │                      │
│  └──────────┬─────────────────┘                  │                      │
│             │ subprocess                          │                      │
│  ┌──────────▼─────────────────────────────────────▼──────────────────┐ │
│  │ llama.cpp v0.2.0 @ bb4caa75 + Quilt Patch Integration             │ │
│  │ `patches/0001-gfx1100-mul-mat-custom.patch` (switch-gated)        │ │
│  │ `build-stock`  : -DGGML_CUDA_ENABLE_CUSTOM_GFX1100=OFF            │ │
│  │ `build-custom` : -DGGML_CUDA_ENABLE_CUSTOM_GFX1100=ON             │ │
│  └──────────────────────────────┬────────────────────────────────────┘ │
└─────────────────────────────────┼──────────────────────────────────────┘
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  AMD RX 7900 XT (gfx1100, RDNA3) — model fully resident (-ngl 99)        │
│  Results archived append-only to `[benchmarks/results/<timestamp>*/]`    │
└──────────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| Session orchestrator | Fingerprinted, guarded, pre-flighted benchmark sessions across ascending context tiers | `benchmarks/bin/run_session.py` |
| llama-bench wrapper | Explicit cell matrix construction (pure prefill `-p C`, decode `-pg C,128`); rejects default-cell contamination | `benchmarks/lib/llabench.py` |
| Guard | Three-signal VRAM spill/RSS/throughput-deviation detection; verdict vocabulary (`OK`, `FAILED:suspected-spill`, ...) | `benchmarks/lib/guard.py` |
| Pre-flight gate | VRAM allocation math vs measured DXG free anchor before heavy tiers; emits `FAILED:preflight-oom` without crashing | `benchmarks/lib/preflight.py` |
| Run store | Crash-resilient append-only result journaling: fsynced `rows.jsonl`, `CHECKSUMS.sha256`, `meta.json` | `benchmarks/lib/store.py` |
| Fingerprinting | System/binary/model sha256 manifests for reproducibility | `benchmarks/lib/fingerprint.py` |
| Telemetry daemons | HWiNFO SM2 sensor capture at 1 Hz; thermal kill at 95°C | `benchmarks/host/hwinfo_daemon.py`, `benchmarks/host/thermal_watchdog.py` |
| Quality gates | Op-level correctness gate (test-backend-ops) and model-level PPL + canary gate | `benchmarks/bin/run_op_gate.py`, `benchmarks/bin/run_model_gate.py` |
| Profiling | rocprof-based bottleneck attribution across 4 canonical shapes | `benchmarks/bin/profile_matrix.py`, `benchmarks/bin/profile_workload.py`, `benchmarks/lib/parse_profile.py` |
| Kernel playground | Standalone gfx1100 HIP kernels with CPU-reference oracle quartet; zero ggml/llama headers | `kernels/CMakeLists.txt`, `scripts/check_no_ggml.sh` |
| Fixture extraction | GGUF tensor dumpers producing binary/npz fixtures with manifests | `tools/dump_gguf_fixtures.py`, `tools/dump_matmul_fixtures.py` |
| Baseline archive | Frozen stock binaries (v0.2.0 @ bb4caa75), never rebuilt or overwritten | `baseline/binaries/v0.2.0-bb4caa75/` |
| Quilt patch integration | Switch-gated in-tree patch (`GGML_CUDA_ENABLE_CUSTOM_GFX1100`) applying over pristine `bb4caa75` | `patches/0001-gfx1100-mul-mat-custom.patch` |

### Phase 7 High-Yield Variant Map (re-scoped)

`kernels/matmul_iq4xs/` now ships as a raced set (re-scoped Phase 7 high-yield architecture):

- `kernels/matmul_iq4xs/impl_gemv_dp4a_gfx1100.hip` — LDS `[32][33]` padded (`+33`, `+3%` via CK Tile `+33`) vs XOR preshuffle `x'=(y%(64/8))⊕x` 0-overhead; `__launch_bounds__(256,4)` + `amdgpu_flat_work_group_size(256,256)` → `≤64 VGPR`, `16 waves/SIMD` (see `output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md` LDS `32×4B` 8-phase `ds_write_b128` `lane0~7…56~63`)
- `kernels/matmul_iq4xs/impl_gemm_wmma_stream.hip` — `64×32 P=2 [2][32][33]` double-buffer vs `64×64 P=4 XOR [4][32][32]` quad-buffer; `MARLIN P=4` hides `GMEM→LDS` while WMMA runs; `gemm_optimization` `T=64 →64×` reuse vs naive `2K` (see `output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md` `gemm_optimization T=64`)
- `kernels/matmul_iq4xs/impl_gemm_lut_iq4xs.hip` — LUT variant `μ=4` (16-entry half, `d*(ls-32)` baked offline, `32B/LUT`) vs inline `d*(ls-32)*kvalues_iq4nl` dequant
- `tools/swizzle_iq4xs.py` — offline `16×64` swizzle to `128B` cache lines + LUT bake (host Python, not shipped)
- `amd_matrix_instruction_calculator` oracle — `python matrix_calculator.py -a gfx1100 -i wmma_f32_16x16x16_f16 -d -R --csv` predicts `A/B 8 VGPR fp16 / D 8 VGPR wave32` (`OPSEL`) → `VGPR ≤64` before commit
- `bench --runs 10` — every `bench_*` (`bench_gemv_dp4a`, `bench_gemm_wmma`, `bench_real_stock`) emits `median/mean/stddev/p95/speedup_median` (`N=10`, `REQ-STAT-07`); racing via `race.py --repeats 10` interleaved `A,B,A,B…` (adelj88 pattern) to kill thermal bias

## Data Flow — Phase 7 High-Yield Pipeline

`Q8_1` quant (`quantize_row_q8_1_coop`, `amax/127` → `half2 ds`) → LDS `[2..4][32][33]` double-buffer `P=2` (today `sB[2][32][33]` stride-33) / `P=4` (`sB[4][32][32]` XOR quad-buffer) with `__builtin_amdgcn_sched_barrier(0x0080)` (DS) / `0x0008` (WMMA) pinning `GMEM→VGPR→LDS→VGPR→WMMA` 4-stage overlap → `B-stationary` weight frag `8 VGPR` (`v16f16`, `b_frag`) in VGPRs + activation streamed via LDS → `b128` `float4`/`ulong2` `16B` coalesced (`32 thr×4B→8×16B` via `__builtin_amdgcn_global_load_b128`, `SWDEV-556587`) + offline `16×64` swizzle to `128B` lines → WMMA `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` (`_OPSEL` false for low half, `wave32` replicates `0–15→16–31`, `1024 ops/CU/clock`) — LDS banking `32×4B` 8-phase `ds_write_b128` (`0~7…56~63` conflict-free iff consecutive) and `ds_read_b128` `4-way→0` via `+33` (`+3%`) vs XOR `x'=(y%(64/8))⊕x` `0%` per CK Tile `lds_bank_conflicts.html`; tiling `T=64 →64×` reuse (`loads/output = K·(1/M+1/N)`, `2K/T`) per CK Tile `gemm_optimization.html` (see `output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md` LDS 32×4B 8-phase, `gemm_optimization T=64`).

## Pattern Overview

**Overall:** Measurement-first optimization harness with a gated kernel-development pipeline and non-destructive quilt patch overlays.

**Key Characteristics:**
- **Frozen baseline discipline:** stock binaries are archived once (`baseline/binaries/v0.2.0-bb4caa75/`) and every A/B comparison runs against that frozen reference; upstream is pinned at commit `bb4caa75`.
- **Gates before integration:** every candidate kernel must pass numerical correctness gates against a CPU reference oracle (`cosine = 1.0`) before in-tree integration.
- **Append-only evidence:** all benchmark results land as timestamped, checksummed run journals under `benchmarks/results/`; failures are published exactly like wins.
- **Hard isolation boundary:** `kernels/` contains zero llama.cpp/ggml includes, enforced by `scripts/check_no_ggml.sh`; only the vendored quant block header `kernels/common/block_iq4_xs.h` is shared.
- **Verdict vocabulary:** guard/preflight components share a locked string vocabulary defined in `benchmarks/lib/guard.py` (`OK`, `FAILED:suspected-spill`, `FAILED:preflight-oom`, `REVIEW:repeat-deviation`, `FAILED:thermal-abort`).

## Layers

**Host Telemetry Layer (Windows):**
- Purpose: GPU sensor capture and thermal protection outside the guest.
- Location: `benchmarks/host/`
- Contains: HWiNFO SM2 memory-mapped reader daemon, manual CSV fallback decoder, cross-boundary process-kill watchdog.
- Depends on: Windows shared memory (`Global\HWiNFO_SENS_SM2`), `wsl.exe` for kills.
- Used by: session orchestrator (`benchmarks/bin/run_session.py` spawns both daemons).

**Harness Orchestration Layer (Python CLIs):**
- Purpose: end-to-end guarded sessions, gates, profiling, matrix publication.
- Location: `benchmarks/bin/`
- Contains: argparse-driven entry points; each CLI imports from `benchmarks.lib`.
- Depends on: harness libraries, pinned llama.cpp binaries at `/root/llama.cpp/build-ci/bin/`, model at `/root/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf`.
- Used by: operator/agent per `benchmarks/RUNBOOK.md`.

**Harness Library Layer (reusable modules):**
- Purpose: single-responsibility modules with pure-function cores (testable without GPU).
- Location: `benchmarks/lib/`
- Contains: `llabench.py` (argv construction/wrapper), `guard.py`, `preflight.py`, `store.py`, `fingerprint.py`, `parse_profile.py`, `toast.py`.
- Depends on: stdlib only plus thresholds config `benchmarks/config/thresholds.json`.
- Used by: everything in `benchmarks/bin/`, `benchmarks/vulkan/run_session_vulkan.py`, and `benchmarks/tools/run_kernel_bench.py`.

**Kernel Playground Layer (standalone HIP/C++):**
- Purpose: develop custom gfx1100 kernels completely decoupled from llama.cpp.
- Location: `kernels/`
- Contains: shared headers (`kernels/common/`), op quartets (`ref_cpu.cpp`, `impl*.hip`, `test_compare.cpp`, `bench_sweep.cpp`) in `kernels/template/`, `kernels/demo_iq4xs_dequant/`, `kernels/matmul_iq4xs/`; fixtures in `kernels/fixtures/` with `manifest_dequant.json` / `manifest_matmul.json`.
- Depends on: HIP runtime only (`hip::device`); vendored `block_iq4_xs.h` (136-byte IQ4_XS layout).
- Used by: Phase 5/6 integration path via quilt patches (`patches/0001-gfx1100-mul-mat-custom.patch`).

**In-Tree Integration Layer (llama.cpp Overlay):**
- Purpose: hook custom gfx1100 kernels into GGML CUDA/HIP execution graph behind compile switch.
- Location: `ggml/src/ggml-cuda/custom_gfx1100/` (`gemv_iq4xs.cuh`, `gemm_iq4xs.cuh`)
- Intercepts: `mmvq.cu` (for $M=1$ decode) and `mmq.cu` (for $M \ge 16$ prefill).
- Switch: `GGML_CUDA_ENABLE_CUSTOM_GFX1100` (default OFF).

## Roadmap summary (6 phases)

| Phase | Focus | Status |
|---|---|---|
| 1 | Environment validation & stock baseline | done — ROCm 7.2.1 cleared, 132/132 GPU layers verified |
| 2 | Benchmark harness & baseline matrix | done — 16-cell baseline published, guard & preflight active |
| 3 | Correctness gates & bottleneck profiling | done — op-gate 21,093/0, PPL 6.4271, bottleneck `MUL_MAT` 31.12% |
| 4 | Kernel playground scaffold | done — standalone gfx1100 playground, zero llama headers, demo `dequant_iq4_xs` passing GREEN/RED |
| 5 | First custom kernel (bottleneck attack) | done — custom gfx1100 GEMV (2.05x) + WMMA GEMM (6.7x) beat stock, cosine 1.0 |
| 6 | Integration, full validation & publication | done — Winners behind switch, baseline preserved, published v1.0.0-gfx1100 |
