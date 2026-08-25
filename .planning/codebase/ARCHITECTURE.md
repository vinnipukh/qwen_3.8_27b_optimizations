<!-- refreshed: 2026-08-25 -->
# Architecture

**Analysis Date:** 2026-08-25

## System Overview

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                    Windows Host (driver + telemetry)                     │
│   Adrenalin driver, HWiNFO64 SM2 bridge, thermal watchdog                │
│   `[benchmarks/host/hwinfo_daemon.py]` `[benchmarks/host/thermal_watchdog.py]` │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │ /dev/dxg passthrough (librocdxg)
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
│             │ subprocess                          │ hipLaunchKernelGGL   │
│  ┌──────────▼─────────────────────────────────────▼──────────────────┐ │
│  │ llama.cpp v0.2.0 @ bb4caa75 (pinned, GGML_HIP=ON, gfx1100)        │ │
│  │ guest ext4 `/root/llama.cpp/build-ci/bin/` — never rebuilt        │ │
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

## Pattern Overview

**Overall:** Measurement-first optimization harness with a gated kernel-development pipeline.

**Key Characteristics:**
- **Frozen baseline discipline:** stock binaries are archived once (`baseline/binaries/v0.2.0-bb4caa75/`) and every A/B comparison runs against that frozen reference; upstream is pinned at commit `bb4caa75`.
- **Gates before integration:** every candidate kernel must pass a numerical correctness gate against a CPU reference oracle before any performance claim is accepted.
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
- Contains: shared headers (`kernels/common/`), op quartets (`ref_cpu.cpp`, `impl*.hip`, `test_compare.cpp`, `bench_sweep.cpp`) in `kernels/template/`, `kernels/demo_iq4xs_dequant/`, `kernels/matmul_iq4xs/`; fixtures in `kernels/fixtures/` with `manifest.json` / `manifest_matmul.json`.
- Depends on: HIP runtime only (`hip::device`); vendored `block_iq4_xs.h` (136-byte IQ4_XS layout).
- Used by: Phase 5/6 integration path via quilt patches (`patches/phase5_mul_mat_custom.patch`).

**Offline Tooling Layer:**
- Purpose: fixture extraction and one-off probes, run offline without GPU.
- Location: `tools/`, `scripts/`, `freetoken-rocm-probe/`
- Contains: GGUF dumpers (`gguf-py` + numpy based), isolation-gate shell script, bandwidth probe sources.
- Depends on: `gguf-py`, numpy; probe tools are self-contained.

## Data Flow

### Primary Request Path (guarded benchmark session)

1. Operator invokes orchestrator with tier list (`benchmarks/bin/run_session.py`)
2. Exclusive flock acquired on `benchmarks/results/.session.lock` (exit code 5 on collision)
3. System fingerprint collected into atomic `manifest.json` (`benchmarks/lib/fingerprint.py`)
4. Per-tier VRAM pre-flight computed against DXG free anchor; over-budget tiers publish `FAILED:preflight-oom` rows instead of crashing (`benchmarks/lib/preflight.py`)
5. Host daemons spawned: HWiNFO telemetry + thermal watchdog (`benchmarks/host/hwinfo_daemon.py`, `benchmarks/host/thermal_watchdog.py`)
6. Cells executed ascending: tier → flash-attn off/on; explicit argv built by `build_argv()` (`benchmarks/lib/llabench.py`); each cell repeated 5× with cooldown
7. During each cell, three-signal guard polls RSS/swap/shared-memory/repeat-deviation and assigns verdicts (`benchmarks/lib/guard.py`)
8. Each row fsynced immediately to `<run_dir>/rows.jsonl` (`benchmarks/lib/store.py`)
9. Session close writes summary, `CHECKSUMS.sha256`, and index entry (`benchmarks/results/index.jsonl`)

### Kernel Candidate Path

1. Extract real/synthetic tensors into fixtures with manifests (`tools/dump_gguf_fixtures.py`, `tools/dump_matmul_fixtures.py` → `kernels/fixtures/`)
2. Implement CPU reference oracle (`kernels/matmul_iq4xs/ref_cpu.cpp`)
3. Implement HIP kernel (`kernels/matmul_iq4xs/impl_gemv_gfx1100.hip`, `impl_gemm_wmma.hip`)
4. Numerical compare against oracle within tight bounds (max_abs < 1e-5, cosine ≥ 0.99999) (`kernels/matmul_iq4xs/test_gemv_compare.cpp`, `test_gemm_compare.cpp`)
5. Microbenchmark vs stock comparator (`bench_gemm.cpp`, `bench_gemv.cpp`, `stock_hip_comparator.hip`)
6. Archive timing tables via RunStore (`benchmarks/tools/run_kernel_bench.py`)
7. On win: wire into upstream dispatch behind compile flag via quilt patch gated OFF by default (`patches/phase5_mul_mat_custom.patch`, flag `GGML_CUDA_ENABLE_CUSTOM_GFX1100`)

### Correctness Gate Path

1. Op-level: `test-backend-ops test -b ROCm0 --output csv`, assert zero errors + core ops present (`benchmarks/bin/run_op_gate.py`)
2. Model-level: WikiText-2 perplexity ±1% of stock baseline + greedy canary exact-match over 6 prompts (`benchmarks/bin/run_model_gate.py`, corpus in `benchmarks/prompts/`, reference in `benchmarks/golden/stock_baseline_golden.json`)

**State Management:**
- No shared mutable state between runs: every session creates a fresh timestamped directory under `benchmarks/results/`.
- Cross-process exclusion via `fcntl.flock` on `benchmarks/results/.session.lock`.
- Thresholds are externalized config, not code: `benchmarks/config/thresholds.json`; guard operates observe-only when absent.

## Key Abstractions

**Op Quartet (kernel playground contract):**
- Purpose: every candidate op ships four artifacts — CPU reference, GPU implementation, comparator, sweep bench.
- Examples: `kernels/template/` (skeleton), `kernels/demo_iq4xs_dequant/`, `kernels/matmul_iq4xs/`
- Pattern: convention-over-configuration; identical file names per op directory, wired through per-op `CMakeLists.txt` auto-detected by parent build.

**RunStore (append-only journal):**
- Purpose: crash-resilient, verifiable record of every benchmark run.
- Examples: `benchmarks/lib/store.py`
- Pattern: write-ahead row journal (`rows.jsonl`), post-hoc checksum manifest, supersede metadata rather than mutation.

**Verdict Vocabulary:**
- Purpose: machine-readable pass/fail classification across guard, preflight, and reporting.
- Examples: `benchmarks/lib/guard.py` constants (`VERDICT_OK`, etc.)
- Pattern: locked string constants imported by consumers — never re-declared inline.

**Fixture Manifest:**
- Purpose: provenance-tracked test tensors (sha256, source commit, ROCm version).
- Examples: `kernels/fixtures/manifest.json`, `kernels/fixtures/manifest_matmul.json`
- Pattern: JSON array of entries with raw hash + path fields, generated by dumpers in `tools/`.

## Entry Points

**Benchmark session:**
- Location: `benchmarks/bin/run_session.py`
- Triggers: manual invocation per `benchmarks/RUNBOOK.md` (WSL2 guest, e.g. `python3 benchmarks/bin/run_session.py --tiers 4096 8192 16384 32768 --repeats 5 --delay 10`)
- Responsibilities: lock, fingerprint, preflight, orchestrate cells, archive results.

**Quality gates:**
- Location: `benchmarks/bin/run_op_gate.py`, `benchmarks/bin/run_model_gate.py`
- Triggers: phase validation before/after any kernel change.
- Responsibilities: structured JSON verdicts to `benchmarks/results/phase3/*.json`.

**Profiling:**
- Location: `benchmarks/bin/profile_workload.py`, `benchmarks/bin/profile_matrix.py`
- Triggers: bottleneck attribution runs across shapes S1–S4.
- Responsibilities: rocprof execution + published `BOTTLENECK-TABLE.md`.

**Kernel build:**
- Location: `kernels/CMakeLists.txt`
- Triggers: `cmake -S kernels -B kernels/build -G Ninja -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_BUILD_TYPE=Release && cmake --build kernels/build`
- Responsibilities: standalone HIP playground build; auto-includes op subdirectories when their `CMakeLists.txt` exists.

**Isolation gate:**
- Location: `scripts/check_no_ggml.sh`
- Triggers: pre-commit / CI check for kernel-playground independence.
- Responsibilities: fail if any `#include` of ggml/llama appears under `kernels/`.

## Architectural Constraints

- **Pinned environment:** ROCm 7.2.1, llama.cpp v0.2.0 @ `bb4caa75`, driver frozen; no silent updates allowed (`README.md`, `benchmarks/environment/versions.txt`).
- **Source tree location:** llama.cpp must live on guest ext4 `/root/llama.cpp` — DrvFs (`/mnt/*`) breaks git lock-files and stalls mmap.
- **Memory floor:** `.wslconfig` `memory=28GB` required; lower values cause DXG ENOMEM during VRAM allocation.
- **Single-threaded sessions:** one exclusive flock; concurrent sessions exit with code 5 (`benchmarks/bin/run_session.py`).
- **Ascending tier order:** context tiers always execute 4096 → 8192 → 16384 → 32768 (locked decision D2-19).
- **Stock baseline immutability:** baseline binaries never rebuilt; custom paths enter only behind default-OFF compile flags (`patches/phase5_mul_mat_custom.patch`).
- **Header isolation:** no ggml/llama headers inside `kernels/` (enforced by `scripts/check_no_ggml.sh`); only the vendored `block_iq4_xs.h` block layout is copied.
- **Wave size templating:** kernels are templated on `WarpSize` (32/64) with `__launch_bounds__(256, 4)`; Wave32 is the gfx1100 target (`kernels/common/hip_helpers.h`, `kernels/matmul_iq4xs/impl_gemv_gfx1100.hip`).

## Anti-Patterns

### Default-cell contamination

**What happens:** relying on llama-bench defaults silently injects extra benchmark cells.
**Why it's wrong:** contaminates the cell matrix, making A/B comparisons invalid (documented bug BENCH-01).
**Do this instead:** explicitly zero defaults (`-n 0`) and enumerate exact cells via `-p C` / `-pg C,128` — see `build_argv()` in `benchmarks/lib/llabench.py`.

### Optimizing before measuring

**What happens:** writing a "faster" kernel without a frozen baseline comparison.
**Why it's wrong:** claims become unverifiable; regressions hide.
**Do this instead:** always bench against archived stock binaries in `baseline/binaries/v0.2.0-bb4caa75/`; measure prefill (M≫1) and decode (M≈1) separately.

### Mutating published results

**What happens:** editing rows in an existing run directory after publication.
**Why it's wrong:** breaks checksum verification and audit trail.
**Do this instead:** create a new timestamped run via `RunStore` and use supersede metadata (`benchmarks/lib/store.py`).

### Unbounded GPU commands

**What happens:** running inference/benchmark scripts without timeouts.
**Why it's wrong:** dead-PTY hangs on `llama-cli` stall the whole pipeline (`AGENTS.md` rule 1).
**Do this instead:** every bash/GPU invocation carries an explicit bounded timeout; headless `llama-cli` requires `setsid --simple-io --single-turn --load-mode none`.

## Error Handling

**Strategy:** fail loudly at gates, never crash mid-session; classify failures with the locked verdict vocabulary.

**Patterns:**
- `HIP_CHECK` macro aborts with file/line on any HIP error (`kernels/common/hip_helpers.h`)
- Custom exceptions for protocol violations: `MatrixContaminationError` (`benchmarks/lib/llabench.py`)
- Pre-flight interception returns structured verdict rows instead of raising (`benchmarks/lib/preflight.py`)
- Deliberate-bug mutants prove gate discrimination: `impl_broken.hip` must fail >10× worse than correct impl (`kernels/demo_iq4xs_dequant/impl_broken.hip`)

## Cross-Cutting Concerns

**Logging:** per-run directories with `logs/` and `telemetry/` subdirectories; append-only `rows.jsonl` with immediate fsync (`benchmarks/lib/store.py`).

**Validation:** three tiers — numerical compare vs CPU oracle (per-op), quality gates (op-level + model-level PPL/canaries), and runtime guards (VRAM/RSS/thermal). Thresholds externalized in `benchmarks/config/thresholds.json`.

**Authentication:** none — local single-user research environment; no network services in the hot path (`LLAMA_CURL=OFF` builds).

---

*Architecture analysis: 2026-08-25*
