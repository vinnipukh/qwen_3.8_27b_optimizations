# Codebase Structure

**Analysis Date:** 2026-08-25

## Directory Layout

```
qwen_3.8_27b_optimizations/
├── .agents/skills/             # Agent skills (magpie-kernel-evaluator, rocm-doctor)
├── .planning/                  # GSD planning: ROADMAP.md, REQUIREMENTS.md, PROJECT.md,
│   │                           #   STATE.md, phases/01..06-*/, reference/, research/
│   └── codebase/               # Codebase map documents (this directory)
├── AGENTS.md                   # Agent operational rules (timeouts, pre-flight, step-up)
├── README.md                   # Project overview, status matrix, quick links
├── CONTRIBUTING.md
├── benchmarks/                 # Python benchmark harness (the measurement system)
│   ├── bin/                    # Orchestrator CLIs (run_session, run_op_gate, profile_matrix, ...)
│   ├── config/                 # thresholds.json, HWiNFO sensor label maps
│   ├── data/                   # wiki.test.raw (perplexity corpus)
│   ├── environment/            # Frozen env fingerprints (versions.txt, rocminfo.txt, vram-probe.txt)
│   ├── golden/                 # stock_baseline_golden.json (PPL + canary references)
│   ├── host/                   # Windows-side daemons (hwinfo_daemon.py, thermal_watchdog.py)
│   ├── lib/                    # Core harness modules (llabench, guard, store, preflight, fingerprint, ...)
│   ├── prompts/                # 6 deterministic canary prompt files
│   ├── profiling/              # Published bottleneck reports (BOTTLENECK-TABLE.md), raw profiles
│   ├── results/                # Append-only run journals (timestamped dirs) + BASELINE-MATRIX.md
│   ├── tests/                  # Pytest suite (55 tests) + fixtures + gate shell scripts
│   ├── tools/                  # run_kernel_bench.py (HIP microbench archival)
│   ├── vulkan/                 # Vulkan comparator arm driver
│   └── RUNBOOK.md              # Binding session protocol and guard thresholds
├── baseline/
│   └── binaries/v0.2.0-bb4caa75/  # Frozen stock llama.cpp binaries (gitignored artifacts)
├── docs/                       # Generated project docs (ARCHITECTURE, GETTING-STARTED,
│                               #   CONFIGURATION, DEVELOPMENT, TESTING, PUBLICATION, QWEN-GRAPH)
├── freetoken-rocm-probe/       # Standalone bandwidth probe tooling (C++ src, Zig toolchain, bin/)
├── kernels/                    # Standalone gfx1100 HIP kernel playground (zero llama.cpp headers)
│   ├── common/                 # Shared headers: block_iq4_xs.h, hip_helpers.h, bench.h
│   ├── template/               # Op quartet skeleton (ref_cpu, impl.hip, test_compare, bench_sweep)
│   ├── demo_iq4xs_dequant/     # Worked example op incl. deliberate-bug mutant
│   ├── matmul_iq4xs/           # Phase 5 MUL_MAT attack (GEMV + WMMA GEMM vs stock comparator)
│   ├── fixtures/               # IQ4_XS tensor fixtures (.bin/.npz) + manifest.json / manifest_matmul.json
│   ├── results/                # Kernel microbenchmark run archives
│   ├── build/                  # CMake build dir (Ninja, generated — do not commit changes here)
│   └── CMakeLists.txt          # Top-level HIP build (gfx1100 only)
├── logs/                       # Misc run logs
├── models/                     # GGUF model artifact (gitignored) + README.md provenance
├── patches/                    # Quilt patches over pinned upstream (phase5_mul_mat_custom.patch)
├── scripts/                    # check_no_ggml.sh isolation gate
├── scrape_out/                 # Web-scrape output (research artifacts)
├── src/                        # Empty placeholder — custom kernels go in kernels/, not src/
├── tools/                      # Offline fixture dumpers (dump_gguf_fixtures.py, dump_matmul_fixtures.py, ask_model.py)
├── arhan-masaustu/             # Local telemetry database (16050_results.db)
├── .rocprofv3/                 # rocprof output artifacts
└── test_wmma-*.{bc,s,o,...}    # Stray compiler artifacts at repo root (generated debris)
```

## Directory Purposes

**`benchmarks/bin/`:**
- Purpose: operator-facing orchestration CLIs.
- Contains: argparse entry points; thin orchestration only, logic lives in `benchmarks/lib/`.
- Key files: `run_session.py`, `run_op_gate.py`, `run_model_gate.py`, `profile_matrix.py`, `profile_workload.py`, `publish_matrix.py`, `calibrate.py`, `run_prompts.py`.

**`benchmarks/lib/`:**
- Purpose: reusable single-responsibility harness modules with pure-function cores.
- Contains: `llabench.py`, `guard.py`, `preflight.py`, `store.py`, `fingerprint.py`, `parse_profile.py`, `toast.py`.
- Key files: all of the above; import as `from benchmarks.lib import ...` (repo root is put on `sys.path` by the CLIs).

**`kernels/`:**
- Purpose: standalone HIP kernel development decoupled from llama.cpp.
- Contains: one subdirectory per candidate op following the quartet convention (`ref_cpu.cpp` → `impl*.hip` → `test_*_compare.cpp` → `bench_*.cpp`) plus shared headers and fixtures.
- Key files: `kernels/CMakeLists.txt`, `kernels/common/block_iq4_xs.h` (vendored IQ4_XS layout), `kernels/matmul_iq4xs/impl_gemv_gfx1100.hip`, `kernels/matmul_iq4xs/impl_gemm_wmma.hip`.

**`tools/`:**
- Purpose: offline fixture extraction and utilities; no GPU required.
- Contains: GGUF tensor dumpers using `gguf-py` + numpy.
- Key files: `tools/dump_gguf_fixtures.py`, `tools/dump_matmul_fixtures.py`.

**`benchmarks/results/`:**
- Purpose: append-only evidence store — every session gets a timestamped directory with `rows.jsonl`, `meta.json`, `logs/`, `telemetry/`, `CHECKSUMS.sha256`.
- Generated: Yes (runtime). Committed: published summary docs are committed (e.g., `BASELINE-MATRIX.md`); raw run payloads largely gitignored.

**`.planning/`:**
- Purpose: GSD workflow state — roadmap, requirements, per-phase plans/summaries/research.
- Key files: `.planning/ROADMAP.md`, `.planning/STATE.md`, `.planning/phases/05-first-custom-kernel-bottleneck-attack/`.

## Key File Locations

**Entry Points:**
- `benchmarks/bin/run_session.py`: full guarded benchmark session
- `benchmarks/bin/run_op_gate.py`: op-level correctness gate
- `benchmarks/bin/run_model_gate.py`: PPL + canary quality gate
- `benchmarks/bin/profile_workload.py` / `profile_matrix.py`: bottleneck profiling
- `benchmarks/tools/run_kernel_bench.py`: standalone HIP microbench archival
- `benchmarks/vulkan/run_session_vulkan.py`: native Windows Vulkan comparator sessions

**Configuration:**
- `benchmarks/config/thresholds.json`: guard/preflight thresholds (observe-only if absent)
- `benchmarks/environment/*`: frozen environment fingerprints and pin records (`llamacpp-pin.txt`)
- `benchmarks/golden/stock_baseline_golden.json`: golden PPL + canary references
- `.wslconfig` (on Windows host, outside repo): memory=28GB requirement documented in `README.md`

**Core Logic:**
- `benchmarks/lib/llabench.py`: llama-bench argv construction and wrapper
- `benchmarks/lib/guard.py`: verdict vocabulary + three-signal spill detection
- `benchmarks/lib/store.py`: RunStore append-only journaling
- `kernels/matmul_iq4xs/impl_gemv_gfx1100.hip` / `impl_gemm_wmma.hip`: custom kernels
- `patches/phase5_mul_mat_custom.patch`: gated integration into upstream dispatch

**Testing:**
- `benchmarks/tests/`: pytest suite (55 unit/regression tests), `pytest.ini`
- `benchmarks/tests/fixtures/`: synthetic input generators for wrapper tests
- `benchmarks/tests/smoke_matrix.sh`, `vulkan_gate.sh`: shell-level smoke gates

## Naming Conventions

**Files:**
- Python modules: `snake_case.py` (`llabench.py`, `run_session.py`)
- CLIs prefixed by action: `run_*`, `profile_*`, `publish_*`, `calibrate` (`benchmarks/bin/`)
- Test files: `test_<unit>.py` (`benchmarks/tests/test_guard_fixtures.py`)
- HIP implementations: `impl_<variant>.hip` (`impl_gemv_gfx1100.hip`, `impl_gemm_wmma.hip`)
- Comparators/tests/benches in kernel ops: `ref_cpu.cpp`, `test_*_compare.cpp`, `bench_*.cpp`
- Fixtures: `<kind>_<tensor>_<shape>_<M>.{bin,npz}` plus `manifest*.json`

**Directories:**
- Run outputs: timestamped `<source>_<YYYYMMDD_HHMMSS><suffix>` (`benchmarks/results/kernels_mul_mat_iq4xs_gemv_20260825_164649/`)
- Phase plans: zero-padded NN-slug (`.planning/phases/04-kernel-playground-scaffold/`)
- Kernel op directories named after the op (`demo_iq4xs_dequant/`, `matmul_iq4xs/`)

**Documents:**
- UPPERCASE.md for top-level/project docs (`README.md`, `RUNBOOK.md`, `BASELINE-MATRIX.md`, `BOTTLENECK-TABLE.md`)

## Where to Add New Code

**New benchmark capability:**
- Library module first: `benchmarks/lib/<module>.py` (pure-function core, dataclass results, stdlib only)
- CLI wrapper: `benchmarks/bin/<verb>_<thing>.py` importing from `benchmarks.lib`
- Tests: `benchmarks/tests/test_<module>.py`; register any thresholds in `benchmarks/config/thresholds.json`

**New custom kernel op:**
- Create `kernels/<op_name>/` following the quartet: `ref_cpu.cpp`, `impl_<target>.hip`, `test_<op>_compare.cpp`, `bench_<op>.cpp`, own `CMakeLists.txt` (parent auto-adds it — see `kernels/CMakeLists.txt` conditional `add_subdirectory`)
- Reuse shared headers via `kernels_common` interface target (`kernels/common/`)
- Never include ggml/llama headers — enforced by `scripts/check_no_ggml.sh`
- Extract fixtures via `tools/dump_matmul_fixtures.py` pattern; record them in a manifest under `kernels/fixtures/`
- Archive microbench results through `benchmarks/tools/run_kernel_bench.py` (uses `RunStore`)

**Integration change to upstream behavior:**
- Quilt patch in `patches/` against pinned commit `bb4caa75`, gated behind default-OFF compile flag (pattern: `patches/phase5_mul_mat_custom.patch`, flag `GGML_CUDA_ENABLE_CUSTOM_GFX1100`)
- Stock baseline binaries in `baseline/binaries/v0.2.0-bb4caa75/` must never be rebuilt or overwritten

**Utilities:**
- Offline/model-free helpers → `tools/`
- Repo hygiene/verification shell scripts → `scripts/`
- One-off probes → separate self-contained subtree like `freetoken-rocm-probe/`

**Do NOT add code to:** `src/` (intentionally empty placeholder), `kernels/build/` (generated), `benchmarks/results/` (runtime evidence).

## Special Directories

**`baseline/binaries/v0.2.0-bb4caa75/`:**
- Purpose: immutable stock binary archive — the A/B reference for every comparison
- Generated: No (archived once). Committed: No (gitignored artifact)

**`models/`:**
- Purpose: sha256-verified GGUF artifact + provenance README
- Generated: No. Committed: only `models/README.md`; the 15.31 GB GGUF is gitignored

**`kernels/build/`, `__pycache__/`, `.rocprofv3/`:**
- Purpose: generated build/profiling artifacts
- Generated: Yes. Committed: No (gitignored)

**`scrape_out/`, `.planning/research/deep-research/raw/scrapes/`:**
- Purpose: research capture (web scrapes, raw research dumps)
- Generated: Yes (tooling output). Committed: partially

**Repo root stray artifacts (`test_wmma-*.{bc,s,hipi,out}`):**
- Compiler debris from WMMA experiments at repo root — safe to delete, should be gitignored

---

*Structure analysis: 2026-08-25*
