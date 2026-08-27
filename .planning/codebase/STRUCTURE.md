<!-- refreshed: 2026-08-25 -->
# Repository Structure

**Analysis Date:** 2026-08-25 (Updated Phase 6 / v1.0.0-gfx1100)

## Directory Tree

```
qwen_3.8_27b_optimizations/
├── .gitignore
├── AGENTS.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
├── NOTICE
├── README.md
├── skills-lock.json
│
├── .planning/
│   ├── PROJECT.md                  # Project scope, core value, principles
│   ├── REQUIREMENTS.md             # Functional requirements (ENV, BENCH, QUAL, PROF, KERN, INTEG, PUB)
│   ├── ROADMAP.md                  # 6-phase roadmap, methodology rules, merge map
│   ├── STATE.md                    # Current phase status and deliverables
│   ├── codebase/                   # Codebase map (ARCHITECTURE, STRUCTURE, STACK, etc.)
│   ├── phases/                     # Context, research, plans, and summaries for Phases 1–6
│   │   ├── 01-environment-validation-stock-baseline/
│   │   ├── 02-benchmark-harness-baseline-matrix/
│   │   ├── 03-correctness-gates-bottleneck-profiling/
│   │   ├── 04-kernel-playground-scaffold/
│   │   ├── 05-first-custom-kernel-bottleneck-attack/
│   │   └── 06-integration-full-validation-publication/
│   └── reference/                  # Original roadmap and RDNA3 GPU kernel resources
│
├── baseline/
│   └── binaries/
│       └── v0.2.0-bb4caa75/        # Pinned stock binaries (llama-cli, llama-bench, llama-perplexity, test-backend-ops)
│
├── benchmarks/
│   ├── RUNBOOK.md                  # Binding benchmark runbook & session protocol
│   ├── bin/                        # Python orchestration CLIs
│   │   ├── calibrate.py            # HWiNFO sensor label discovery & near-OOM watchdog
│   │   ├── profile_matrix.py       # 4-shape bottleneck matrix profiling orchestrator
│   │   ├── profile_workload.py     # Single workload rocprof/eval_profiler runner
│   │   ├── publish_matrix.py       # Baseline matrix aggregator
│   │   ├── run_model_gate.py       # QUAL-02 WikiText-2 PPL and prompt canary gate
│   │   ├── run_op_gate.py          # QUAL-01 test-backend-ops correctness gate
│   │   ├── run_prompts.py          # Layer-2 deterministic prompt corpus runner
│   │   └── run_session.py          # Full session orchestrator with preflight & RSS guard
│   ├── config/
│   │   ├── thresholds.json         # Guard thresholds (RSS, throughput variance, VRAM)
│   │   └── labels.json             # Sensor label maps
│   ├── data/
│   │   └── wiki.test.raw           # WikiText-2 test set for PPL gate (QUAL-02)
│   ├── environment/                # System environment fingerprints & logs
│   │   ├── versions.txt            # ROCm, driver, librocdxg, kernel, and snapshot hashes
│   │   ├── rocminfo.txt            # rocminfo device enumeration
│   │   ├── hipconfig.txt           # hipconfig --full output
│   │   └── startup-log.txt         # llama.cpp startup log with VRAM buffer breakdown
│   ├── golden/
│   │   └── stock_baseline_golden.json # Golden perplexity (6.4271) and canary SHA256 hashes
│   ├── host/                       # Host-side (Windows) background daemons
│   │   ├── hwinfo_daemon.py        # HWiNFO64 Shared Memory v2 1Hz poller
│   │   └── thermal_watchdog.py     # 95°C thermal kill watchdog
│   ├── lib/                        # Core harness libraries
│   │   ├── fingerprint.py          # D2-10 manifest collection & system hashes
│   │   ├── guard.py                # 3-signal VRAM spill & throughput guard
│   │   ├── llabench.py             # llama-bench argv construction & row parser
│   │   ├── parse_profile.py        # rocprof trace parser & kernel aggregator
│   │   ├── preflight.py            # VRAM allocation preflight check
│   │   ├── store.py                # RunStore append-only result journaler
│   │   └── toast.py                # Cross-platform toast notifications
│   ├── profiling/                  # Profiling reports
│   │   ├── BOTTLENECK-TABLE.md     # Phase 3 ranked bottleneck table (MUL_MAT #1)
│   │   ├── KERNEL-BENCH-DIFF.md    # Phase 5/6 microbenchmark diff report vs stock
│   │   └── dispatch_overhead_report.md # HIP Graphs evaluation report (+19% decode)
│   ├── prompts/                    # Deterministic 6-prompt evaluation corpus
│   ├── results/                    # Append-only run journals (rows.jsonl, CHECKSUMS.sha256)
│   ├── tests/                      # Pytest suite (55 unit and integration tests)
│   ├── tools/
│   │   └── run_kernel_bench.py     # Standalone HIP benchmark RunStore archiver
│   └── vulkan/                     # Vulkan comparator arm build and runner scripts
│
├── docs/                           # Architecture, development, configuration, testing, and publication guides
│   ├── ARCHITECTURE.md
│   ├── CONFIGURATION.md
│   ├── DEVELOPMENT.md
│   ├── GETTING-STARTED.md
│   ├── PUBLICATION.md              # Phase 6 publication package
│   ├── QWEN-GRAPH.md               # Graph-aware tensor projection breakdown
│   └── TESTING.md                  # 7-level testing hierarchy
│
├── kernels/                        # Standalone gfx1100 HIP kernel playground (zero llama.cpp headers)
│   ├── CMakeLists.txt              # Standalone HIP build targeting gfx1100
│   ├── common/                     # Common headers & testing harness
│   │   ├── bench.h                 # hipEvent_t timing & statistical sampler
│   │   ├── block_iq4_xs.h          # Vendored 136-byte IQ4_XS struct & kvalues table
│   │   ├── hip_helpers.h           # HIP error checking & launch macros
│   │   └── matmul_test_util.h      # Shared metrics (cosine/max_rel) & weight generators
│   ├── fixtures/                   # Extracted and synthetic tensor fixtures
│   │   ├── manifest_dequant.json   # Dequant op fixtures manifest
│   │   └── manifest_matmul.json    # Matmul 32-shape fixtures manifest
│   ├── demo_iq4xs_dequant/         # Demo op quartet (correct, mutant, tests, benchmarks)
│   └── matmul_iq4xs/               # Target #1 MUL_MAT kernels
│       ├── ref_cpu.h/cpp           # FP64 double-accumulate CPU reference oracle
│       ├── stock_hip_comparator.hip# Naive scalar HIP comparator baseline
│       ├── impl_gemv_gfx1100.hip   # Custom Wave32 GEMV decode kernel (M=1)
│       ├── impl_gemm_wmma.hip      # Custom Wave32 GEMM prefill kernel (TILE_M=16 + WMMA)
│       ├── test_stock_compare.cpp  # Baseline comparator test binary
│       ├── test_gemv_compare.cpp   # Custom GEMV correctness test binary
│       ├── test_gemm_compare.cpp   # Custom GEMM correctness test binary
│       ├── bench_gemv.cpp          # Dedicated GEMV decode microbenchmark
│       ├── bench_gemm.cpp          # Dedicated GEMM prefill microbenchmark
│       └── bench_matmul.cpp        # Unified 32-shape microbenchmark
│
├── patches/
│   └── 0001-gfx1100-mul-mat-custom.patch # Quilt patch adding custom kernels behind switch
│
├── scripts/
│   └── check_no_ggml.sh            # Isolation check ensuring kernels/ has 0 llama headers
│
└── tools/                          # Offline tooling
    ├── dump_gguf_fixtures.py       # Extracts raw quant blocks from GGUF files
    ├── dump_matmul_fixtures.py     # Extracts 32 canonical matmul tensor fixtures
    ├── gemv_iq4xs.cuh              # In-tree GEMV header for quilt patch
    └── gemm_iq4xs.cuh              # In-tree GEMM header for quilt patch
```
