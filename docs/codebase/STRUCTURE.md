# Codebase Structure

**Analysis Date:** 2026-08-30 (Phase 7 replan closure; llama.cpp overlay tracked as quilt patch, not directory)

## Directory Layout

```
qwen_3.8_27b_optimizations/
├── .agents/skills/               # Project skills (magpie-kernel-evaluator, rocm-doctor)
├── .gitattributes                # *.patch / *.hip / *.cuh eol=lf (+ core.autocrlf=false note)
├── .gitignore                    # llama.cpp/, kernels/build/, models/*.gguf, baseline/binaries/, *.bc/.hipi
├── docs/                    # GSD planning state (phases 01–08, codebase map, research)
│   ├── PROJECT.md / REQUIREMENTS.md / ROADMAP.md / STATE.md / config.json
│   ├── codebase/                 # This map (ARCHITECTURE, STRUCTURE, STACK, CONVENTIONS, TESTING, CONCERNS, INTEGRATIONS)
│   ├── phases/                   # 01-… 08-… phase dirs (07 = current hybrid DP4A+WMMA)
│   ├── reference/                # Original 18-phase roadmap + RDNA3 GPU resources
│   └── research/                 # Deep-research reports and model/stack decisions
├── AGENTS.md / CHANGELOG.md / CONTRIBUTING.md / LICENSE / NOTICE / README.md / skills-lock.json
├── baseline/binaries/v0.2.0-bb4caa75/   # FROZEN stock binaries (gitignored, never rebuilt)
├── benchmarks/                   # Python benchmark harness (target of Phase 8 py-prune 40→0)
│   ├── RUNBOOK.md                # Binding session protocol
│   ├── bin/                      # Orchestrator CLIs (run_session, run_op_gate, run_model_gate, …)
│   ├── config/                   # thresholds.json, labels.json
│   ├── data/                     # wiki.test.raw (QUAL-02 corpus)
│   ├── environment/              # rocminfo/hipconfig/version fingerprints
│   ├── golden/                   # stock_baseline_golden.json (PPL 6.4271, canaries)
│   ├── host/                     # Windows daemons: hwinfo_daemon.py, thermal_watchdog.py
│   ├── lib/                      # store/guard/preflight/llabench/fingerprint/parse_profile/toast.py
│   ├── profiling/                # KERNEL-BENCH-DIFF.md, BOTTLENECK-TABLE.md, dispatch_overhead_report.md
│   ├── prompts/                  # Deterministic 6-prompt corpus
│   ├── results/                  # Append-only run journals + phase7/ (race.py, N=10 JSONs)
│   ├── tests/                    # Pytest suite (55 tests) + fixtures + smoke gates
│   ├── tools/                    # run_kernel_bench.py, eval_profiler.cpp
│   └── vulkan/                   # Native Vulkan comparator arm (build-vulkan-arm.ps1)
├── build_windows.bat             # REQ-WIN-07 native gfx1100 build (:8000 smoke)
├── docs/                         # ARCHITECTURE (freshest 2026-08-30) + DEVELOPMENT/PUBLICATION/TESTING/…
├── docs/research/freetoken-probe/         # Early ROCm probe tooling (tokio/zig probes)
├── kernels/                      # STANDALONE gfx1100 HIP playground (zero llama.cpp headers)
│   ├── CMakeLists.txt            # find_package(hip … PATHS "$ENV{HIP_PATH}/…")
│   ├── build/                    # CMake/Ninja artifacts (gitignored)
│   ├── common/                   # block_iq4_xs.h (vendored 136B), hip_helpers.h (hard HIP_CHECK), bench.h, matmul_test_util.h
│   ├── template/                 # Op quartet skeleton
│   ├── demo_iq4xs_dequant/       # Worked example quartet + mutant
│   ├── matmul_iq4xs/             # TARGET #1 MUL_MAT kernels (GEMV + GEMM families)
│   └── fixtures/                 # Binary/npz tensor fixtures + manifest_dequant/matmul.json (fixture bins gitignored)
├── llama.cpp/                    # Pinned upstream bb4caa75 checkout — GITIGNORED (tracked via patch only)
│   └── ggml/src/ggml-cuda/custom_gfx1100/   # Overlay headers (gemv/gemm_iq4xs.cuh, empty.cuh)
├── logs/                         # build.log, gate.log (gitignored)
├── models/                       # Qwen GGUF (gitignored, provenance in README.md) — /root/models on guest
├── output/                       # Working notes + deep-research (quilt_note, race_note, P8_VARIANT_COMPILE_PROPOSAL, …)
├── patches/                      # QUILT: 0001-gfx1100-mul-mat-custom.patch (356 lines) — THE tracked overlay artifact
├── scripts/check_no_ggml.sh      # Kernel playground isolation gate
├── src/                          # Empty placeholder (kernels land in kernels/, not src/)
└── tools/                        # dump_gguf_fixtures.py, dump_matmul_fixtures.py, swizzle_iq4xs.py (offline-only)
```

## Directory Purposes

**`docs/`:**
- Purpose: GSD planning state — roadmap, requirements, per-phase plans/summaries, deep research
- Contains: `phases/07-hybrid-dp4a-wmma-kernel-optimization/` holds the current replan plans (`07-01-PLAN.md` Windows toolchain, `07-02-PLAN.md` variant race, `07-03-PLAN.md` N=10/15 rigour; `07-04` plan/summaries deleted in the replan), `07-CONTEXT.md`, `07-RESEARCH.md`, `07-VERIFICATION.md`; `phases/08-refactor-windows-native/` retained as the REQ-WIN-07 landing execution (no new phase number)
- Key files: `docs/STATE.md` (current status), `docs/ROADMAP.md` (7-phase roadmap), `docs/REQUIREMENTS.md` (REQ-WIN-07/REQ-PERF-07/REQ-STAT-07)

**`benchmarks/`:**
- Purpose: measurement arm — guarded A/B sessions, correctness gates, profiling, run journals, variant racing
- Contains: argparse CLIs in `bin/` importing pure-function modules from `lib/`; append-only journals under `results/<ts>_<label>/`; N=10 race harness under `results/phase7/` (`race.py`, `rows.jsonl`, `llama_bench_{stock,custom}_4tier_N10.json`, `hwinfo.log`)
- Key files: `bin/run_session.py`, `lib/store.py`, `lib/guard.py`, `results/phase7/race.py`

**`kernels/`:**
- Purpose: standalone gfx1100 HIP playground — develop and numerically gate kernels before any integration; hard-isolated from ggml headers (`scripts/check_no_ggml.sh`)
- Contains: per-op quartet directories (`ref_cpu.cpp` oracle, `impl*.hip` kernel, `test_*_compare.cpp` gate, `bench_*.cpp` N=10 sweep); shared headers in `common/`; fixtures in `fixtures/`
- Key files: `matmul_iq4xs/CMakeLists.txt` (7 variant OBJECT libs), `matmul_iq4xs/real_stock_dp4a_comparator.hip`, `matmul_iq4xs/impl_gemv_dp4a_gfx1100.hip`, `matmul_iq4xs/impl_gemm_wmma_stream.hip`

**`llama.cpp/` (gitignored):**
- Purpose: pinned upstream v0.2.0 `bb4caa75` checkout; never mutated directly — overlay applied via quilt patch regeneration (`git -C llama.cpp diff bb4caa75`)
- Contains: `ggml/src/ggml-cuda/custom_gfx1100/` with vendored `gemv_iq4xs.cuh` (120 lines), `gemm_iq4xs.cuh` (95 lines), `empty.cuh` (OFF stub), `README.md`
- Key files (dispatch intercepts): `ggml/src/ggml-cuda/mmq.cu:114`, `ggml/src/ggml-cuda/mmvq.cu:1280`, `ggml/CMakeLists.txt:221` (switch option)
- Note: ignored by this repo's git — the tracked capture of the overlay is `patches/0001-gfx1100-mul-mat-custom.patch`

**`patches/`:**
- Purpose: quilt overlay — the only tracked representation of in-tree change
- Key files: `0001-gfx1100-mul-mat-custom.patch` (356 lines / 276 insertions, 8 files: ggml/CMakeLists.txt option + custom_gfx1100/* + mmq.cu/mmvq.cu intercepts + ggml-hip CMake)

**`tools/`:**
- Purpose: offline host tooling (Python + headless .cuh sources for the patch)
- Contains: `dump_gguf_fixtures.py`, `dump_matmul_fixtures.py` (fixture extraction), `swizzle_iq4xs.py` (offline 16×64 swizzle + LUT bake, host-only), `ask_model.py`, `gemm_iq4xs.cuh` / `gemv_iq4xs.cuh` (pre-patch in-tree header sources)

**`output/`:**
- Purpose: working notes and deep-research evidence, honest FAIL tables
- Key files: `docs/research/P8_VARIANT_COMPILE_PROPOSAL.md` (variant OBJECT proposal), `docs/research/notes/quilt_note.md` (2026-08-30 attestation), `docs/research/notes/race_note.md` (race harness + honest results), `docs/research/deep-research/1000t-s-at-8k-gfx1100.md`, `docs/research/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md`, `docs/research/technical-synthesis-gfx1100-wmma-vs-dp4a.md`

**`docs/research/freetoken-probe/`:**
- Purpose: early-stage ROCm probe tooling (scaffold-era, superseded by the Phase 4 playground)

## Key File Locations

**Entry Points:**
- `benchmarks/bin/run_session.py`: full guarded A/B session orchestrator
- `benchmarks/results/phase7/race.py`: interleaved variant race (`--repeats 10` exact, `--tiers 512,1024,2048,4096,8192`)
- `kernels/build/matmul_iq4xs/bench_gemm_wmma` (built): per-variant N=10 GEMM microbench, `--variant <name>`
- `kernels/build/matmul_iq4xs/bench_gemv_dp4a` (built): per-variant N=10 GEMV microbench
- `build_windows.bat`: Windows-native gfx1100 build + `:8000` server smoke (REQ-WIN-07, requires HIP SDK)

**Configuration:**
- `kernels/CMakeLists.txt`: HIP discovery via `$ENV{HIP_PATH}` + `/opt/rocm` fallback, gfx1100 arch
- `kernels/matmul_iq4xs/CMakeLists.txt`: variant OBJECT library matrix + executables
- `llama.cpp/ggml/CMakeLists.txt:221` (via patch): `GGML_CUDA_ENABLE_CUSTOM_GFX1100` OFF/ON switch
- `benchmarks/config/thresholds.json`: guard thresholds
- `.gitattributes`: `*.patch eol=lf` (quilt hygiene)
- `.wslconfig` (host, not in repo): `memory=28GB` requirement

**Core Logic:**
- `kernels/matmul_iq4xs/real_stock_dp4a_comparator.hip`: honest DP4A baseline (quantize + vec_dot, 552 lines)
- `kernels/matmul_iq4xs/impl_gemv_dp4a_gfx1100.hip`: coop Wave32 GEMV decode (341 lines) + `gemv_variant_xor.cuh` (38 lines)
- `kernels/matmul_iq4xs/impl_gemm_wmma_stream.hip`: 5-variant streaming WMMA GEMM (364 lines, weak tiled helper `:353`)
- `kernels/matmul_iq4xs/impl_gemm_lut_iq4xs.hip`: LUT μ=4 GEMM (157 lines, secondary)
- `llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/{gemv,gemm}_iq4xs.cuh`: vendored in-tree dispatch
- `kernels/common/block_iq4_xs.h`: vendored 136-byte IQ4_XS layout

**Testing:**
- `kernels/matmul_iq4xs/test_gemv_dp4a_compare.cpp`, `test_gemm_wmma_compare.cpp`, `test_real_stock_compare.cpp`, `test_stock_compare.cpp`, `test_gemv_compare.cpp`, `test_gemm_compare.cpp`: kernel correctness gates (cosine ≥ 0.999 vs FP64 oracle)
- `benchmarks/tests/`: 55 pytest unit/integration tests + `smoke_matrix.sh`, `vulkan_gate.sh`
- `benchmarks/bin/run_op_gate.py` (QUAL-01), `benchmarks/bin/run_model_gate.py` (QUAL-02)

**Bench tools (llama.cpp upstream `benches/`):**
- `llama.cpp/benches/{dgx-spark,mac-m2-ultra,nemotron}/`: upstream vendor bench reports (HTML/JSON/md) included in the pinned checkout; not part of this repo's measurement arm (which uses `llama-bench` directly)

## Naming Conventions

**Files:**
- Kernel impls: `impl_<op>_<technique>[_<arch>].hip` — e.g., `impl_gemv_dp4a_gfx1100.hip`, `impl_gemm_wmma_stream.hip`, `impl_gemm_lut_iq4xs.hip`
- CPU oracles: `ref_cpu.h` / `ref_cpu.cpp` (per op dir)
- Comparators: `<role>_hip_comparator.hip` — `stock_hip_comparator.hip` (naive scalar), `real_stock_dp4a_comparator.hip` (production DP4A)
- Variant add-ons: `<base>_variant_<trick>.cuh` — `gemv_variant_xor.cuh`
- Test gates: `test_<kernel>_compare.cpp` — `test_gemv_dp4a_compare.cpp`, `test_gemm_wmma_compare.cpp`
- Benches: `bench_<kernel>.cpp` — `bench_gemm_wmma.cpp`, `bench_gemv_dp4a.cpp`, `bench_real_stock.cpp`
- Hardware evidence: `bench_<kernel>.hardware.json` (N=10, committed); root-level `bench_*.bare.json` are transient homedir artifacts (untracked)
- CMake OBJECT libraries: `matmul_<path>_hip` with variant suffix — `matmul_gemv_dp4a_{,xor}_hip`, `matmul_gemm_wmma_{stream,p4_xor,64x64,lut}_hip`
- Variant compile definitions: `GEMV_XOR`, `GEMM_P4_XOR`, `TILE_64x64`
- Quilt patches: `NNNN-<slug>-<topic>.patch` — `0001-gfx1100-mul-mat-custom.patch`
- Phase dirs: `NN-<slug>/` with `NN-0M-<KIND>.md` files (`07-02-PLAN.md`, `07-VERIFICATION.md`)

**Directories:**
- Op-scoped playground dirs: `kernels/<op>/` containing the quartet (`kernels/matmul_iq4xs/`, `kernels/demo_iq4xs_dequant/`, `kernels/template/`)
- Harness split: `benchmarks/bin/` (commands) vs `benchmarks/lib/` (reusable modules)

**Functions/Symbols:**
- Kernel entry: `gemv_iq4xs_dp4a_coop_kernel<WARP_SIZE>` / `gemm_iq4xs_wmma_stream_kernel_cuh`; exports `gemv_iq4xs_dp4a_gpu`, `gemm_iq4xs_wmma_stream_gpu_cuh` per variant
- Dispatch: `custom_{gemv,gemm}_iq4xs_can_handle` + `custom_{gemv,gemm}_iq4xs_dispatch`
- Bench JSON fields: `variant`, `tile`, `P`, `banking`, `speedup_median`, `speedup_mean_minus_1sigma`, `winner` (`PASS`/`FAIL`/`SKIPPED`)

## Where to Add New Code

**New kernel variant (same algorithm, different tile/P/banking):**
- Implementation: `kernels/matmul_iq4xs/impl_gemm_wmma_stream.hip` via `#ifndef TILE_M/TILE_N/P/USE_XOR` guards + a per-variant OBJECT in `kernels/matmul_iq4xs/CMakeLists.txt` with `target_compile_definitions` (pattern: `impl_gemm_wmma_64x32_p4_xor.hip` include-wrapper per `docs/research/P8_VARIANT_COMPILE_PROPOSAL.md`)
- Bench wiring: add `--variant <name>` dispatch in `kernels/matmul_iq4xs/bench_gemm_wmma.cpp` (real object, no synthetic jitter)
- Verify: `nm -D` distinct symbols; `hipcc --offload-arch=gfx1100 -Rpass-analysis | grep VGPR` ≤ 64; `llvm-objdump | grep v_wmma`; bench `--runs 10 --json --variant <name>`

**New op playground (e.g., attention/KV):**
- Copy `kernels/template/` quartet: `ref_cpu.cpp` + `impl.hip` + `test_compare.cpp` + `bench_sweep.cpp`, register in `kernels/CMakeLists.txt`; keep zero ggml includes (`scripts/check_no_ggml.sh`)

**New fixture:**
- `tools/dump_matmul_fixtures.py` (model tensors) or synthetic generation; register in `kernels/fixtures/manifest_matmul.json`

**In-tree integration (when a kernel wins):**
1. Vendor the kernel into `llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/gemv_iq4xs.cuh` / `gemm_iq4xs.cuh` with provenance header
2. Ensure real `can_handle` shape gates (never `return false` stubs)
3. Regenerate: `git -C llama.cpp diff bb4caa75 > ../patches/0001-gfx1100-mul-mat-custom.patch` (`core.autocrlf=false`; `git apply --check` PASS)
4. Build both: `cmake -S llama.cpp -B build-stock -D...=OFF` and `-B build-custom -DGGML_CUDA_ENABLE_CUSTOM_GFX1100=ON`; qualify via gates before claiming perf

**New harness CLI/utility:**
- `benchmarks/bin/<name>.py` + logic in `benchmarks/lib/<name>.py` + tests `benchmarks/tests/test_<name>.py` (note Phase 8 prunes these; new harness code should justify its existence or live in `kernels/` C++)

**New race tier/variant set:**
- Edit `benchmarks/results/phase7/race.py` `VARIANTS`/`TIERS` (keep `--repeats 10` assertion, `A,B,A,B` interleave, 8192 preflight SKIPPED logic)

## Special Directories

**`llama.cpp/`:**
- Purpose: pinned upstream checkout + overlay headers
- Generated: No (upstream source)
- Committed: **No — gitignored** (`/llama.cpp` in `.gitignore`); overlay tracked only via `patches/0001-gfx1100-mul-mat-custom.patch`

**`kernels/build/`:**
- Purpose: CMake/Ninja output (bench/test binaries, object archives)
- Generated: Yes
- Committed: No (gitignored)

**`baseline/binaries/`:**
- Purpose: frozen stock binaries (`v0.2.0-bb4caa75/` with llama-cli/llama-bench/llama-perplexity/test-backend-ops)
- Generated: No (archived once, never rebuilt)
- Committed: No (gitignored; provenance in `models/README.md`)

**`models/`:**
- Purpose: model artifact location (host `/root/models/` guest copy; repo copy for provenance-reading only)
- Generated: No
- Committed: No (`*.gguf` gitignored)

**`benchmarks/results/`:**
- Purpose: append-only evidence (rows.jsonl, CHECKSUMS.sha256, manifest.json per run; phase7/ race harness + N=10 JSONs)
- Generated: Yes (runtime)
- Committed: Yes — evidence is intentionally versioned (never edited after write)

**`kernels/fixtures/`:**
- Purpose: tensor fixtures for correctness gates (32 matmul shapes + synthetic edge cases)
- Generated: Partially (dump tools) / manually curated (synthetics)
- Committed: Manifest JSONs only; fixture binaries gitignored (`kernels/fixtures/*` with `!manifest*.json` negation), regenerated via `tools/dump_*_fixtures.py` + synthetic scripts

**`output/`:**
- Purpose: working notes, deep-research evidence, honest FAIL tables, proposals
- Generated: No (analysis artifacts)
- Committed: Yes

**Root scratch artifacts:** `.playwright-cli/`, `.rocprofv3/`, `scrape_out/`, `temp_readme.txt`, `fix-p10-thermal.md`, `fix-p6-perf.md`, `impl_gemv_dp4a_gfx1100-*` (`.bc`/`.hipi`/`.out`) — exploration/intermediate outputs; `*.bc`/`*.hipi`/`*.hipfb` gitignored.

---

*Structure analysis: 2026-08-30*