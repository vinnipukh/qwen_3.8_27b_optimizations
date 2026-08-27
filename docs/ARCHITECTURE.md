<!-- generated-by: gsd-doc-writer -->

# Architecture

Qwen3.8-27B (IQ4_XS) inference optimization on an AMD Radeon RX 7900 XT (`gfx1100`) via llama.cpp HIP
under WSL2 + ROCm 7.2.1. Goal: custom HIP kernels that beat stock on at least one workload,
with correctness gates enforced before any integration.

## Repository layout

```
.
├── baseline/
│   └── binaries/v0.2.0-bb4caa75/   # stock pinned binaries (llama-cli, llama-bench,
│                                   #   llama-perplexity, test-backend-ops); gitignored
├── benchmarks/
│   ├── bin/                        # Orchestrator CLIs (run_session, run_prompts, calibrate, publish_matrix)
│   ├── config/                     # Empirical thresholds (thresholds.json) and label maps
│   ├── environment/                # Environment fingerprints: versions.txt, hipconfig.txt, rocminfo.txt,
│   │                               #   hip-support-comparator.csv, startup-log.txt, vram-probe.txt
│   ├── host/                       # Host-side daemons: hwinfo_daemon.py, thermal_watchdog.py
│   ├── lib/                        # Core harness libraries: llabench.py, fingerprint.py, guard.py,
│   │                               #   store.py, preflight.py, toast.py
│   ├── prompts/                    # Deterministic 6-prompt corpus (short/long x code/prose)
│   ├── results/                    # Append-only run journals (rows.jsonl, manifest.json, CHECKSUMS.sha256)
│   │                               #   + kernels_mul_mat_iq4xs* (3 runs: GEMV/GEMM vs stock)
│   ├── profiling/                  # KERNEL-BENCH-DIFF.md — Phase 5 GEMV/GEMM vs stock diff (prefill/decode)
│   ├── tests/                      # Pytest suite (55 tests) + fixtures + smoke/gate shell scripts
│   ├── vulkan/                     # Native Vulkan comparator arm build scripts and coverage gate report
│   └── RUNBOOK.md                  # Binding session protocol, guard thresholds, and thermal policy
├── models/                         # GGUF artifact (gitignored) + README.md provenance
├── kernels/                        # Standalone gfx1100 HIP kernel playground (zero llama.cpp headers)
│   ├── common/                     # Shared headers: block_iq4_xs.h (vendored 136B), hip_helpers.h, bench.h
│   ├── template/                   # Op quartet skeleton (ref_cpu, impl.hip, test_compare, bench_sweep)
│   ├── fixtures/                   # Model-extracted & synthetic IQ4_XS tensor fixtures + manifest.json
│   │                               #   + matmul_* (32 fixtures) via dump_matmul_fixtures.py (manifest_matmul.json)
│   ├── demo_iq4xs_dequant/         # Worked example: CPU oracle, GPU kernel, mutant, comparator, sweep
│   ├── matmul_iq4xs/               # Phase 5 MUL_MAT: ref_cpu.h/cpp, stock_hip_comparator.hip,
│   │                               #   impl_gemv_gfx1100.hip, impl_gemm_wmma.hip, test_*/bench_*.cpp, CMakeLists
│   └── CMakeLists.txt              # Top-level standalone HIP build (CMAKE_HIP_ARCHITECTURES=gfx1100)
├── tools/                          # Offline tools (dump_gguf_fixtures.py, dump_matmul_fixtures.py)
├── patches/                        # Quilt patches over pinned upstream (0001-gfx1100-mul-mat-custom.patch)
├── scripts/                        # Isolation and verification scripts (check_no_ggml.sh)
├── src/                            # placeholder — custom kernels land in kernels/, not src/
├── logs/                           # run logs
├── freetoken-rocm-probe/           # early ROCm probe tooling
└── .planning/                      # ROADMAP.md, REQUIREMENTS.md, PROJECT.md, STATE.md,
                                    #   phases/01-*/, phases/02-*/, reference/ROADMAP-original.md
```

## Execution environment

```
Windows host
│   AMD Adrenalin driver (WSL2 support), .wslconfig memory=28GB (REQUIRED)
│   HWiNFO64 Shared Memory v2 telemetry bridge (Global\HWiNFO_SENS_SM2)
│   Thermal watchdog service (cross-boundary wsl.exe process kill @ 95°C)
▼
WSL2 guest (Ubuntu 24.04, root-only)
│   /dev/dxg passthrough via librocdxg v1.2.2
│   HSA_ENABLE_DXG_DETECTION=1  (persisted in /etc/profile.d/rocdxg.sh)
▼
ROCm 7.2.1 (pinned; HIP 7.2.53211-e1a6bc5663, gcc 13.3.0)
▼
llama.cpp @ v0.2.0 (bb4caa75), built -DGGML_HIP=ON -DGPU_TARGETS=gfx1100
│   -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
│   source tree lives on guest ext4: /root/llama.cpp (DrvFs git-lock issue)
▼
gfx1100 GPU: model fully resident (15.31 GB IQ4_XS from /root/models/, zero CPU fallback)
```

Key constraints:

| Constraint | Reason |
|---|---|
| `.wslconfig` `memory=28GB` | DXG ENOMEM during VRAM allocation at lower values (`dxgkio_create_allocation: -12`) |
| Source tree on guest ext4 | git file-lock operations fail on DrvFs (`/mnt/e`) |
| Model copy at `/root/models/` | mmap reads over `/mnt/e` stall |
| Headless runs: `setsid --simple-io --single-turn --load-mode none` | `llama-cli` hangs in `n_tty_write` on dead PTY otherwise |
| Pre-flight VRAM Gate | Allocations > 18.25 GB free VRAM intercepted to prevent silent memory thrashing or DXG panic |

## Roadmap summary (7 phases)

Phases 1–4 produce measurement and validation infrastructure; Phase 5 attacked `MUL_MAT` against naive scalar references; Phase 6 delivered integration and release `v1.0.0-gfx1100`; Phase 7 optimizes against real upstream DP4A and WMMA tensor pipelines. See `.planning/ROADMAP.md`.

| Phase | Focus | Status |
|---|---|---|
| 1 | Environment validation & stock baseline | done — ROCm 7.2.1 cleared, 132/132 GPU layers verified |
| 2 | Benchmark harness & baseline matrix | done — 16-cell baseline published, guard & preflight active |
| 3 | Correctness gates & bottleneck profiling | done — op-gate 21,093/0, PPL 6.4271, bottleneck `MUL_MAT` 31.12% |
| 4 | Kernel playground scaffold | done — standalone gfx1100 playground, zero llama headers, demo `dequant_iq4_xs` passing GREEN/RED |
| 5 | First custom kernel (bottleneck attack) | done — custom gfx1100 GEMV (2.05x) + WMMA GEMM (6.7x) beat naive stock, cosine 1.0 |
| 6 | Integration, full validation & publication | done — Winners behind switch, baseline preserved, published v1.0.0-gfx1100 |
| 7 | Hybrid DP4A & WMMA Matrix Core Optimization | planned — Fuse Q8_1 integer quantization with cooperative Wave32 DP4A and RDNA3 WMMA matrix cores to beat real stock end-to-end |

Binding methodology rules: benchmark before optimize; one change at a time; keep the stock
baseline forever; prefill (M≫1) and decode (M≈1) measured separately; publish failures too.

## Kernel playground pipeline (Phase 4 — delivered)

Each candidate kernel runs through a four-stage standalone pipeline outside llama.cpp:

```
ref_cpu          impl_gfx1100.hip       test_compare           bench_sweep
CPU reference -> HIP implementation -> numerical compare    -> microbenchmark sweep
(golden output)  (gfx1100 target)     (correctness gate vs    (prefill M≫1 and
                                      ref, tolerance-bounded) decode M≈1, vs stock)
```

A kernel advances only if it passes `test_compare` and wins in `bench_sweep`; failures are
recorded like successes. Phase 4 delivered: standalone `kernels/` build (`CMAKE_HIP_ARCHITECTURES=gfx1100`,
zero llama headers, vendored `block_iq4_xs.h` 136B), fixture dumper (`tools/dump_gguf_fixtures.py`
via `gguf-py` + synthetic edge cases), and worked example `kernels/demo_iq4xs_dequant/` traversing the
quartet with tight gate max_abs 1e-5 / mean 1e-6 / cosine 0.99999 and ≥10× broken discrimination (315.91 GB/s wave32).
(owner locks D4-00-1..5). Wave32 and wave64 variants are templated (`template<int WarpSize>`) and benched
separately. See `.planning/phases/04-kernel-playground-scaffold/04-CONTEXT.md` and `04-01..03-PLAN.md`.

## Integration strategy

Winning kernels are integrated as **quilt patches over the pinned upstream** commit
(v0.2.0 @ bb4caa75), each gated behind ON/OFF build/runtime flags. The stock baseline
binaries are never rebuilt or overwritten, so every A/B comparison runs against a frozen
reference. Patches carry their correctness-gate evidence in the commit message.

## Original methodology plan

The pre-GSD 18-phase methodology plan is preserved verbatim at
`.planning/reference/ROADMAP-original.md`; every retained element maps into the 6-phase
structure via the Merge Map in `.planning/ROADMAP.md`.
