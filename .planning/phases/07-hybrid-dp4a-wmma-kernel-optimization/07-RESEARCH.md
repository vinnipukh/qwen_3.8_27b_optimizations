# Phase 7: Hybrid DP4A & WMMA Matrix Core Optimization — Research

**Researched:** 2026-08-28
**Domain:** RDNA3 gfx1100 HIP kernels — IQ4_XS quantized GEMV/GEMM, hardware DP4A (v_dot4/sudot4), WMMA matrix cores, LDS banking, Windows HIP SDK native, statistical rigour
**Confidence:** HIGH

## User Constraints (from CONTEXT.md)

### Locked Decisions
- **Background & re-scope binding (2026-08-28):** Phase 7 must fuse `Q8_1` integer activation quantization + RDNA3 hardware matrix cores (`v_dot4_i32_i8` / `v_wmma`) with Wave32 cooperative workgroups to beat *real production* `llama.cpp` stock `vec_dot_iq4_xs_q8_1` + `quantize_row_q8_1` DP4A, not naive scalar float. Prior wins (2.13× GEMV / 9.27× GEMM) were vs naive and are invalid comparators for Phase 7. Stock `mmvq.cu` limits RDNA3 to 1 warp/row (`MMVQ_PARAMETERS_RDNA3_0`); our 8-thread coop (256 thr → 32 rows/block) + `__dp4a` / `v_dot4_i32_i8` and WMMA `1024 ops/CU/clock` are the architectural bets.
- **Owner 3 must-haves are hard gates (REQ-WIN-07 / REQ-PERF-07 / REQ-STAT-07):**
  1. **Windows-native ≤2 langs (REQ-WIN-07):** `build_windows.bat` via `HIP_PATH` + `clang++.exe --offload-arch=gfx1100` + `-G Ninja`, pure `C++/HIP` + `CMake` only (`find -name "*.py" ! -path "./llama.cpp/*" == 0`), `llama-server.exe` serves `localhost:8000` on `gfx1100`.
  2. **≥1.10× pp+tg (REQ-PERF-07):** `llama-bench` `pp+tg` at `{512,1024,2048,4096,8192}` (8192 if VRAM pre-flight passes) must be `median ≥1.10× stock` and `mean−1σ ≥1.10×` over `N=10` thermal-paired runs — microbench `>1.2×` vs real DP4A is necessary but NOT sufficient.
  3. **10×/15× rigour (REQ-STAT-07):** Every perf/quality number `N≥10` (`median`+`mean`+`stddev`+`p95`), LLM QA `N≥15` (`temp=0` fixed prompt, `avg tok/s`+`latency`+per-run table). Single-run claims banned; amends `BENCH-01` `≥3→≥10`.
- **Deep-research cliff acknowledged:** `8k` is quadratic `67M (8k) → 1B (32k) 16×` per head, Flash makes HBM `O(N)` but FLOPs stay `N²` → `8k→32k 3–4×` with Flash vs `16×` without, `7900 XT 800 GB/s` caps `1000 tok/s` (naive `15.3 TB/s`), `KV≈128 KiB/tok GQA` (`8k≈1–2 GB` + `15.3 GB` model = `18.5 GB` on `20 GB`), WSL2 DXG `16 GiB` lie + BSOD after `3–5 OOMs`, `15–30 µs` jitter flattens `1.178→1.00`, `rocprofv3`+`librocdxg` blind on WSL2.
- **Existing kernels baseline (STATE.md):** `real_stock_dp4a` 84 µs vs 543 µs naive, GEMV peak 1.178 avg 1.00 (WSL jitter), WMMA `[2][32][33]` + `wmma builtin`, quilt 355 lines — Phase 7 builds on these, does not replace proven DP4A/WMMA wiring.

### Claude's Discretion
- High-yield variant selection and racing: `64×32 vs 64×64 vs 128×32` tiling, `XOR preshuffle vs +33 padding` LDS banking, `P=2 vs P=4` double-buffer depth, `B-stationary` vs `A+B LDS`, `b128`/`float4` coalescing, `16×64` offline swizzle, `LUT μ=4` vs inline dequant, `SmoothQuant α=0.5` — discretion to research, recommend winner, and specify verification (rocprof `lds_bank_conflict 0`, `amd_matrix_instruction_calculator` `VGPR ≤64`, `llvm-objdump v_wmma/v_dot4`), but MUST carry all 3 must-haves as gates.
- Tuning philosophy borrowing: `adelj88/rocm_wmma_gemm` `tune.py` Genetic + Random Forest surrogate + `race.py --repeats 10` interleaved is a template, not a mandate — adapt `budget`/`k_slice` to our tile sweep.

### Deferred Ideas (OUT OF SCOPE)
- Turkish output quality evaluation, multi-GPU, vision mmproj optimization, custom sampler/speculative decoding outside MTP, persistent-kernel scheduling — stretch goals only after core milestones.
- HauhauCS patched runtime or custom `K_P` quants as baseline (contaminates measurement; eval-only variant).
- Full hierarchical KV paging beyond CTX-04 recency/static; 256k as hard v1 exit until CTX-01/02 establish budget.
- Native Windows HIP SDK was out-of-scope before 2026-08-28 but is now IN SCOPE via REQ-WIN-07; prior WSL2-only assumption is superseded.

## Summary

Phase 7 must close 7 truths: 4 original (real DP4A comparator `N=10`, DP4A GEMV `>1.2× median` + `>38 t/s decode`, WMMA GEMM `>950 t/s prefill` with `64×32 [2][32][33]`, quilt OFF/ON + QUAL-01/02 green `N=10`) plus 3 owner must-haves added 2026-08-28 (Windows-native `≤2` langs gate, `≥1.10×` `pp+tg` at 5 tiers gate, `N=10`/`N=15` averaging gate). All prior E2E `808→849 pp4096 (+5.1%)` [VERIFIED: .planning/STATE.md:stopped_at] is **insufficient** — the gate is `1.10× median` and `mean−1σ ≥1.10×` per tier per split [CITED: .planning/REQUIREMENTS.md REQ-PERF-07].

The artifactual path is low-risk (kernels exist: `real_stock_dp4a_comparator.hip` 84 µs vs 543 µs [VERIFIED: .planning/STATE.md], `impl_gemv_dp4a_gfx1100.hip` LDS `[32][33]` + `__launch_bounds__(256,4)` + `sudot4/perm` [VERIFIED: kernels/matmul_iq4xs/impl_gemv_dp4a_gfx1100.hip:1-80], `impl_gemm_wmma_stream.hip` `64×32 [2][32][33]` + `wmma_f32_16x16x16` [VERIFIED: kernels/matmul_iq4xs/impl_gemm_wmma_stream.hip:1-80]). The risk is **measurement** — WSL2 DXG adds `15–30 µs` jitter flattening `1.178→1.00` and lies about VRAM (`80 GiB` reported vs `3.48 GiB` contiguous fail) [CITED: output/deep-research/1000t-s-at-8k-gfx1100.md], `rocprofv3` is Instinct-only/404 on WSL2 [CITED: output/deep-research/1000t-s-at-8k-gfx1100.md], and `800 GB/s` caps naive `1000 tok/s` (`15.3 TB/s` needed) [CITED: output/deep-research/1000t-s-at-8k-gfx1100.md].

High-yield levers that actually move the `≥1.10×` needle on 5-tier `N=10` are: scale tile `64×32→64×64` (64× reuse vs `2K` naive, `T=64` [CITED: output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md]), replace `+33` (`+3%` LDS, `4-way→0` [CITED: output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md]) with `XOR preshuffle x'=(y%(64/8))⊕x` 0-overhead for `64×64` [CITED: output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md], push pipeline `P=2→P=4` (`GMEM→VGPR→LDS→VGPR→WMMA` overlap, `sched_barrier 0x0080/0x0008` [CITED: llvm.org/docs/AMDGPUUsage.html]), `B-stationary` weight in VGPR + `float4`/`global_load_b128` `16B` coalescing (`32 thr×4B→8×16B` [CITED: output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md]), offline `16×64` swizzle to `128B` cache lines + `LUT μ=4` 16-entry half bake (`d*(ls-32)` offline) [CITED: output/deep-research/custom_kernel_pdfs/SYNTHESIS.md], and `SmoothQuant α=0.5` (`s_j=max|X_j|^α/max|W_j|^{1-α}` fused into `rmsnorm`, `W8A8` arm) [CITED: output/deep-research/custom_kernel_pdfs/SYNTHESIS.md].

**Primary recommendation:** Ship Phase 7 as **two parallel tracks in one thermal window**: (A) WSL2 bare-metal `N=10` microbench + `N=10` `llama-bench` paired A/B across 5 tiers + `N=10` quality gates on `build-custom` (proves `≥1.10×` vs *real* DP4A); (B) Windows 11 HIP SDK native `build_windows.bat` (`HIP_PATH/bin/clang++.exe --offload-arch=gfx1100 -G Ninja`) `llama-server.exe :8000 →200` smoke + at least compile-gate + one `llama-bench` smoke tier to prove toolchain parity; interleave all benches via `race.py --repeats 10` pattern to kill thermal bias [CITED: output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md]; verify winner via `rocprof lds_bank_conflict 0` + `amd_matrix_instruction_calculator VGPR ≤64` + `llvm-objdump v_wmma/v_dot4` before declaring PASS.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|--------------|----------------|-----------|
| Q8_1 activation quant (`quantize_row_q8_1_coop`, `amax/127`, `half2 ds`) | GPU kernel (HIP device) | — | Quant must fuse with DP4A/WMMA on CU to avoid extra HBM pass; host quant would double bandwidth |
| IQ4_XS dequant → WMMA frag (`d*(ls-32)*kvalues_iq4nl` → `v16f16`) | GPU kernel (HIP device, VGPR/LDS) | Offline swizzle (host Python, `16×64`) | LUT `μ=4` bakes scale offline; online path stays `4.25 bpw` `L1`-resident |
| DP4A GEMV decode (`M=1`, 8-thread coop, `sudot4+perm`, LDS `[32][33]`) | GPU kernel (HIP) | — | Beats `MMVQ` 1-warp/row via occupancy; single-warp is RDNA3 MMVQ param, not a tuner |
| WMMA GEMM prefill (`M≥128`, `64×32/64×64`, `sB[2..4][32][33]`, `wmma_f32_16x16x16`) | GPU kernel (HIP) | — | RDNA3 matrix cores are CU-local; host cannot emulate `1024 ops/CU/clock` |
| LDS banking & coalescing (`+33` vs `XOR`, `b128` `float4`) | GPU kernel (HIP, `ds_write_b128` 8-phase) | — | `32 banks×4B` [CITED: output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md] is CU SRAM, not host |
| Pipelining `P=2→P=4` + `sched_barrier` | GPU kernel (HIP, `0x0080` DS / `0x0008` WMMA) | — | Compiler reorder fence is device ISA [CITED: llvm.org/docs/AMDGPUUsage.html] |
| Build & gating (CMake `GGML_CUDA_ENABLE_CUSTOM_GFX1100`, quilt patch) | Build system (CMake + `git apply --check`) | — | `build_windows.bat` uses `HIP_PATH/bin/clang++.exe` not `cl` for `.hip` |
| End-to-end A/B (`llama-bench` `pp/tg` `5` tiers `N=10`, `RunStore`+`CHECKSUMS`, `hwinfo_daemon`) | Host harness (WSL2 + Windows) | GPU kernels | `llama-bench --single-turn --simple-io --load-mode none -ngl 99 -b 2048` drives GPU |
| Quality gates (`test-backend-ops` 0 errors, `PPL 6.4271±1%`, canaries) | Host harness + GPU (`build-custom`) | — | Gates run on custom ON build; must be `N=10` averaged per REQ-STAT-07 |
| Windows serve (`llama-server.exe :8000 →200`) | Host (Windows native, HIP SDK) | — | No WSL2 for this gate; proves `≤2` langs (C++/HIP+CMake only) |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| ROCm / HIP SDK | **7.2.1** guest (WSL2) / **6.4+** native Windows (`HIP_PATH`) [VERIFIED: docs/PUBLICATION.md §2] | HIP runtime + `hipcc`/`clang++.exe` gfx1100 target | Locked in repo; `HIP 7.2.53211` AMD clang 22.0.0git; only toolchain that emits `v_wmma`/`v_dot4` for `gfx1100` [CITED: gpuopen.com/learn/wmma_on_rdna3] |
| llama.cpp | pinned `bb4caa75` + quilt `patches/0001-gfx1100-mul-mat-custom.patch` (355 lines, OFF default) [VERIFIED: .planning/STATE.md + patches/0001*] | Baseline + custom dispatch (`mmq.cu`/`mmvq.cu` `#if GGML_CUDA_ENABLE_CUSTOM_GFX1100`) | Anti-fork discipline; `git apply --check` bisectable |
| CMake + Ninja | **≥3.21** + Ninja | Build (`-DGGML_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DGGML_CUDA_ENABLE_CUSTOM_GFX1100=ON/OFF`, `-G Ninja` on Windows) | `build_windows.bat` mandated Ninja (not `cl` msbuild) per 08-CONTEXT |
| rocWMMA | **2.2.1** header-only `rocwmma/rocwmma.hpp` [CITED: rocm.docs.amd.com/projects/rocWMMA/] | Optional WMMA wrapper (alternative to raw `__builtin_amdgcn_wmma*`); header-only → no runtime | `≤2` langs gate: no Python/JS server; compiles with `clang++.exe --offload-arch=gfx1100` [CITED: output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md] |
| amd_matrix_instruction_calculator | latest `ROCm/amd_matrix_instruction_calculator` (star 143) [CITED: github.com/ROCm/amd_matrix_instruction_calculator] | Pre-commit VGPR/layout oracle (`-a gfx1100 -i wmma_f32_16x16x16_f16 -d`, `--register-layout --csv`) | Predicts `A_frag 8 VGPR / D 8 VGPR wave32 → ≤64 VGPR` before code lands [CITED: output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md] |
| Composable Kernel (CK) Tile docs + `ck_tile/hardware/lds_bank_conflicts.html` | CK **1.2.0** [CITED: rocm.docs.amd.com/projects/composable_kernel/] | LDS banking (`32 banks×4B`, `ds_write_b128` 8-phase, `+33` vs `XOR` `x'=(y%(64/8))⊕x` [CITED: output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md]) + GEMM tiling `T=64→64×` [CITED: output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md] | Authoritative for `lds_bank_conflict` counter and tiling math |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `adelj88/rocm_wmma_gemm` (tune/race pattern) | 15★, 62 commits [CITED: github.com/adelj88/rocm_wmma_gemm] | Template for tile sweep (`tune.py` Genetic+RF surrogate `budget 100` + `race.py --repeats 10` interleaved) [CITED: output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md] | Ritual for `N=10` `REQ-STAT-07` interleaving to kill thermal bias; do not fork whole lib |
| `llvm-objdump` + `hipcc --save-temps -Rpass-analysis` | LLVM bundled with ROCm `22.0.0git` [VERIFIED: docs/PUBLICATION.md] | Disasm/occupancy audit (`--mcpu=gfx1100 \| grep v_wmma/v_dot4`, VGPR ≤64) | Gate before declaring `>1.2×`; `llvm.org/docs/AMDGPUUsage.html` for `sched_barrier` mask [CITED: llvm.org/docs/AMDGPUUsage.html] |
| `rocprofv3` (native bare metal only) | ROCm `6.4+` (Instinct-only matrix) [CITED: output/deep-research/1000t-s-at-8k-gfx1100.md] | `lds_bank_conflict 0` + `global_load_b128` coalescing count on winner variant | **WSL2 blind** — must run on native Windows/Linux bare metal, not under `librocdxg` [CITED: output/deep-research/1000t-s-at-8k-gfx1100.md] |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Raw `__builtin_amdgcn_wmma*` | `rocWMMA` C++ `fragment<>` API | Raw gives `OPSEL`/`NEG` control + `perm` LUT exactness; rocWMMA is cleaner but hides `lane%16` replication — keep raw for Phase 7, rocWMMA as arm only |
| `+33` padding `[32][33]` | `XOR preshuffle` 0-overhead | `+33` proven `+3%` (4224 vs 4096B) [CITED: output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md]; XOR saves `~1 KB` at `64×64` but adds `^` per access — race both |
| `P=2` double-buffer | `P=4` quad-buffer (`MARLIN` `P=4` [CITED: output/deep-research/custom_kernel_pdfs/SYNTHESIS.md]) | `P=4` hides `GMEM→LDS` at `8192` but doubles LDS (`×2` vs `×4` buffers) — only use if `64×32` `4 KB` leaves headroom under `64 KB` CU |
| `IQ4_XS` primary | `SmoothQuant W8A8 INT8 WMMA` arm (`α=0.5` [CITED: output/deep-research/custom_kernel_pdfs/SYNTHESIS.md]) | `W8` halves weight mem but costs `~2×` bits vs `4.25 bpw`; useful as comparator for `8192` `pp` if `≥1.10×` fails on `IQ4_XS` alone |
| HIP via `hipcc` on WSL2 | Windows `HIP_PATH/bin/clang++.exe --offload-arch=gfx1100` | WSL2 `HSA_ENABLE_DXG_DETECTION=1` is DXG path (jitter); Windows native removes `15–30 µs` tax — both required, not either/or |

**Installation:**
```bash
# WSL2 (already locked)
cmake -S kernels -B kernels/build -DCMAKE_HIP_ARCHITECTURES=gfx1100 && cmake --build kernels/build
cmake -S llama.cpp -B build-stock -DGGML_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DGGML_CUDA_ENABLE_CUSTOM_GFX1100=OFF -G Ninja
cmake -S llama.cpp -B build-custom -DGGML_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DGGML_CUDA_ENABLE_CUSTOM_GFX1100=ON -G Ninja

# Windows native (REQ-WIN-07) — no cl for .hip, no Python shipped
REM build_windows.bat
set HIP_PATH=C:\Program Files\AMD\ROCm\6.4
"%HIP_PATH%\bin\clang++.exe" --offload-arch=gfx1100 --version
cmake -S . -B build-windows -G Ninja -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DGGML_HIP=ON -DGGML_CUDA_ENABLE_CUSTOM_GFX1100=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="%HIP_PATH%\bin\clang++.exe"
cmake --build build-windows --config Release
build-windows\bin\llama-server.exe --help
curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}"
# Expect 200 + choices[0].message.content [CITED: .planning/REQUIREMENTS.md REQ-WIN-07]

# Pre-commit oracle (offline, not shipped — satisfies ≤2 langs)
pip install tabulate typing_extensions  # calculator prereq [CITED: output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md]
python matrix_calculator.py -a gfx1100 -i wmma_f32_16x16x16_f16 -d
python matrix_calculator.py -a gfx1100 -i wmma_f32_16x16x16_f16 -R -A --csv
```

**Version verification:** Before writing the Standard Stack table, verify each recommended package exists and is current using the ecosystem-appropriate command:
```bash
hipcc --version 2>&1 | head -1          # ROCm 7.2.1 / 6.4
cmake --version | head -1               # >=3.21
ninja --version                         # any
python matrix_calculator.py -a gfx1100 -L | head -n 20  # wmma list
```
Training data versions may be months stale — always confirm against the correct ecosystem registry.

## Package Legitimacy Audit

> Required whenever this phase installs external packages. Phase 7 installs **no new npm/pypi/crates** at runtime — it is pure `C++/HIP` + `CMake` + ROCm headers (≤2 langs gate). Audit covers the HIP/ROCm + helper tools actually invoked.

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| `ROCm / HIP SDK` (incl. `hipcc`/`clang++.exe`) | AMD HIP SDK (Windows) / `/opt/rocm` (WSL2) | 8+ yrs (ROCm since 2016) | N/A (vendor SDK) | `github.com/ROCm/ROCm` | OK | Approved — `gfx1100` `HIP 7.2.53211` verified [VERIFIED: docs/PUBLICATION.md §2] |
| `rocWMMA` | `rocm.docs.amd.com/projects/rocWMMA` header-only, `ROCm/rocWMMA` | 4+ yrs | N/A (header-only) | `github.com/ROCm/rocWMMA` | OK | Approved — `2.2.1` docs [CITED: rocm.docs.amd.com/projects/rocWMMA/] |
| `amd_matrix_instruction_calculator` | `github.com/ROCm/amd_matrix_instruction_calculator` (offline tool) | 2+ yrs | 143★ | `github.com/ROCm/amd_matrix_instruction_calculator` | OK | Approved — offline only, not shipped (`≤2` langs still holds) |
| `adelj88/rocm_wmma_gemm` (tune/race pattern) | `github.com/adelj88/rocm_wmma_gemm` | 1+ yr, 62 commits | 15★ | `github.com/adelj88/rocm_wmma_gemm` | OK | Approved — pattern reuse, not vendored lib |
| `Composable Kernel` (CK Tile docs) | `rocm.docs.amd.com/projects/composable_kernel` | 3+ yrs | N/A | `github.com/ROCm/composable_kernel` | OK | Approved — `1.2.0` LDS docs [CITED: rocm.docs.amd.com/projects/composable_kernel/] |

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

*No `npm view` / `pip index` packages are installed by Phase 7 — `find -name "*.py" ! -path "./llama.cpp/*" == 0` after prune is a gate. Calculator Python deps (`tabulate`, `typing_extensions`) are offline pre-commit only, not shipped.*

## Architecture Patterns

### System Architecture Diagram

```
[Host: Windows 11 + VS Build Tools | WSL2 Ubuntu 24.04]
        │
        ├── build_windows.bat (Ninja + HIP_PATH/bin/clang++.exe --offload-arch=gfx1100) ──► llama-server.exe :8000
        │                └─ CMAKE_HIP_ARCHITECTURES=gfx1100 + GGML_CUDA_ENABLE_CUSTOM_GFX1100 ON/OFF
        │
        ├── WSL2 cmake -S kernels -B kernels/build (gfx1100) ──► playground benches (N=10)
        │
llama.cpp ggml dispatch (mmq.cu / mmvq.cu / quantize.cu)
        │
        ├── #if GGML_CUDA_ENABLE_CUSTOM_GFX1100 ──┬── M==1 ? gemv_iq4xs.cuh (8-thread coop, sudot4+perm, sh[32][33])
        │                                        └── M>=128 ? gemm_iq4xs.cuh (64×32/64×64, sB[2..4][32][33], wmma_f32_16x16x16)
        │                                            ├─ B-stationary (weight VGPR frag 8, activation LDS)
        │                                            ├─ b128 float4/ulong2 coalesced (16B)
        │                                            ├─ P=2→P=4 pipeline + sched_barrier 0x0080/0x0008
        │                                            └─ LUT μ=4 (16 half) alternative + 16×64 swizzle
        │
        └── else (OFF) ── stock vec_dot_iq4_xs_q8_1 + quantize_row_q8_1 (real DP4A comparator, 84µs)

Measurement (EVERY claim):
  WSL2 bare metal ──► bench_* --runs 10 --json (median/mean/stddev/p95, speedup_median) ──► RunStore rows.jsonl + CHECKSUMS
                  ──► llama-bench --single-turn --simple-io --load-mode none -ngl 99 -b 2048 -p {512..8192} -r 10
                  ──► hwinfo_daemon 1Hz + thermal_watchdog 90C (same thermal window, interleaved race.py --repeats 10)
                  ──► test-backend-ops N=10 (0 errors) + run_model_gate N=10 (PPL 6.4271±1%)
                  ──► llvm-objdump | grep v_wmma/v_dot4 + hipcc --save-temps VGPR≤64 + rocprof lds_bank_conflict 0 (native only)

  Windows native ──► build_windows.bat clean + llama-server :8000 200 + compile-gate VGPR/disasm + at least 1 llama-bench smoke tier
```

A reader traces the primary use case `pp@8192` by following: `llama-bench -p 8192 -b 2048` → `ggml_cuda_mul_mat_q` fuses `quantize_mmq_q8_1` + `mul_mat_q` → `M=8192 ≥128` → `gemm_iq4xs.cuh` `64×64 P=4 XOR` WMMA `v_wmma` (`1024 ops/CU/clock`) with `B-stationary` + `b128` → `VGPR`/`LDS` overlap via `sched_barrier` → `≥1.10× pp` median over `N=10` interleaved.

### Recommended Project Structure
```
kernels/matmul_iq4xs/
├── real_stock_dp4a_comparator.hip   # true vec_dot_iq4_xs_q8_1 + quantize_row_q8_1 (84µs, perm×6, sudot4)
├── impl_gemv_dp4a_gfx1100.hip       # decode M=1 8-thread coop 256→32 rows, sh[32][33], sudot4+perm
├── impl_gemm_wmma_stream.hip        # prefill 64×32 base + P=2 [2][32][33] (tile sweep variants in same file or impl_gemm_wmma_stream_variants.hip)
├── impl_gemm_lut_iq4xs.hip          # LUT variant μ=4 16-entry half (optional second kernel, same tile)
├── bench_real_stock.cpp             # --runs 10 --json median/mean/stddev/p95 (baseline 84µs vs 543µs)
├── bench_gemv_dp4a.cpp              # --runs 10 vs real DP4A, reports speedup_median per 8 shapes
├── bench_gemm_wmma.cpp              # --runs 10 vs real DP4A M=128/512/1024/8192 + rectangular 4096,4096,2048
├── test_*_compare.cpp               # cosine ≥0.999 gate vs FP64 ref_cpu (N=10 if stochastic)
├── ref_cpu.{h,cpp}                  # FP64 oracle (cosine 1.0)
└── CMakeLists.txt                   # matmul_real_stock_hip + gemv_dp4a + gemm_wmma_stream (+ lut) targets

llama.cpp/ggml/src/ggml-cuda/custom_gfx1100/
├── gemv_iq4xs.cuh                   # vendored GEMV coop (sh_coop[32][33], launch_bounds)
├── gemm_iq4xs.cuh                   # vendored WMMA stream (sB[2][32][33], wmma builtin, X[gm*K+gk])
├── empty.cuh                        # OFF fallback
└── README.md

patches/0001-gfx1100-mul-mat-custom.patch  # real git diff HEAD over bb4caa75, 355 lines, git apply --check PASS both OS

build_windows.bat                     # HIP_PATH/bin/clang++.exe --offload-arch=gfx1100 -G Ninja (no cl, no py)
benchmarks/results/phase7/ab_stock_* , ab_custom_*  # RunStore N=10 per tier per build, CHECKSUMS, hwinfo CSV
```

### Pattern 1: Cooperative 8-Thread DP4A GEMV (Decode M=1)
**What:** 256-thread block (8 waves Wave32) cooperatively owns 32 rows (`8 thr/row`), each `block_iq4_xs` 136 B split across 8 lanes (`ib 0..7`), `__builtin_amdgcn_sudot4` + `perm` LUT `kvalues_iq4nl`, `amamax/127` Q8_1 coop quant, `__shared__ float sh[32][33]` reduction, `__launch_bounds__(256,4)` + `amdgpu_flat_work_group_size(256,256)` → `≤64 VGPR` → `16 waves/SIMD` [VERIFIED: kernels/matmul_iq4xs/impl_gemv_dp4a_gfx1100.hip:1-80].
**When to use:** `M==1` decode (tg); stock `MMVQ_PARAMETERS_RDNA3_0` is 1 warp/row — coop wins via occupancy + `ulong2` `16B` qs loads.
**Example:**
```cpp
// Source: kernels/matmul_iq4xs/impl_gemv_dp4a_gfx1100.hip (verified)
__global__ __launch_bounds__(256,4) __attribute__((amdgpu_flat_work_group_size(256,256)))
void gemv_dp4a_coop_kernel(const block_iq4_xs* W, const float* x, float* y, int64_t K, int64_t N) {
  __shared__ float sh[32][33]; // +33 kills 32-bank conflict [CITED: output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md]
  // 8 thr/row: ib = threadIdx.x % 8, row = blockIdx.x*32 + threadIdx.x/8
  // qs via ulong2 16B aligned, kvalues via __builtin_amdgcn_perm, dp4a via __builtin_amdgcn_sudot4
}
// Verification: llvm-objdump --mcpu=gfx1100 gemv_dp4a.o | grep v_dot4  # sudot4 → v_dot4_i32_i8
//              hipcc --save-temps -Rpass-analysis  # VGPR ≤64
```

### Pattern 2: WMMA Streaming GEMM with P=2→P=4 + XOR vs +33 (Prefill M≥128)
**What:** `64×32` per block (`4×2` warps, 256 thr) of `16×16×16` WMMA tiles (`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`, `wave32` replicates `0–15→16–31` [CITED: gpuopen.com/learn/wmma_on_rdna3]), double-buffered `sB[2][32][33]` (`_Float16`, `+3%` LDS, `ds_write_b128` 8-phase `0~7…56~63` conflict-free iff consecutive [CITED: output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md]) vs `XOR preshuffle x'=(y%(64/8))⊕x` 0-overhead [CITED: output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md]; `P=4` variant `sB[4][32][32]` (`MARLIN` `P=4` hides `GMEM→LDS` while WMMA runs [CITED: output/deep-research/custom_kernel_pdfs/SYNTHESIS.md]) pinned with `__builtin_amdgcn_sched_barrier(0x0080)` (DS) / `0x0008` (WMMA) [CITED: llvm.org/docs/AMDGPUUsage.html]; `B-stationary` (weight frag `A/B 8 VGPR fp16 / D 8 VGPR wave32` [CITED: output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md] in registers, activation streamed via LDS) + `float4`/`global_load_b128` `16B` (`32 thr×4B→8×16B` [CITED: output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md]) + offline `16×64` swizzle to `128B` cache lines [CITED: output/deep-research/custom_kernel_pdfs/SYNTHESIS.md]; dequant `d*(ls-32)*kvalues` overlapped with previous WMMA (pipelined), `LUT μ=4` 16-entry half bakes `d*(ls-32)` [CITED: output/deep-research/custom_kernel_pdfs/SYNTHESIS.md]; fallback `TILE_M=16` when `M<512`.
**When to use:** `M≥128` prefill (pp), especially `M=8192` (`64×64` → `64×` reuse vs naive `2K` [CITED: output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md]) where `800 GB/s` roof dominates.
**Example:**
```cpp
// Source: kernels/matmul_iq4xs/impl_gemm_wmma_stream.hip (verified) + high-yield synthesis
template<int TILE_M=16> __global__ __launch_bounds__(256,4) __attribute__((amdgpu_flat_work_group_size(256,256)))
void gemm_iq4xs_stream_tiled_kernel(const block_iq4_xs* W, const float* X, float* Y, int64_t K, int64_t N, int64_t M);
// WMMA path (64×32):
__shared__ _Float16 sB[2][32][33]; // variant B: sB[4][32][32] + xor_preshuffle<64,8>
typedef _Float16 v16f16 __attribute__((ext_vector_type(16)));
typedef float v8f32 __attribute__((ext_vector_type(8)));
v16f16 a_frag; v16f16 b_frag; v8f32 c_frag;
c_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b_frag, c_frag, false); // OPSEL false for low half
// Pipelining:
__builtin_amdgcn_sched_barrier(0x0080); // DS before WMMA [CITED: llvm.org/docs/AMDGPUUsage.html]
// Verification: llvm-objdump --mcpu=gfx1100 gemm_wmma.o | grep v_wmma
//              rocprof --metric lds_bank_conflict  # expect 0 on winner [CITED: output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md]
//              python matrix_calculator.py -a gfx1100 -i wmma_f32_16x16x16_f16 -d  # VGPR ≤64
```

### Anti-Patterns to Avoid
- **Microbench vs naive scalar only:** Banned for Phase 7 verdict; must be vs *real* `vec_dot_iq4_xs_q8_1` DP4A (`bench_real_stock` 84 µs baseline) — prior `6–7×` vs naive is not the gate [VERIFIED: .planning/STATE.md + .planning/REQUIREMENTS.md REQ-PERF-07].
- **Single-run `tok/s` claim:** Banned — every number is `median`+`mean`+`stddev`+`p95` over `N=10` (LLM QA `N=15` + per-run table) [CITED: .planning/REQUIREMENTS.md REQ-STAT-07].
- **`cl` for `.hip` on Windows:** `MSVC cl` cannot compile HIP kernels; `HIP_PATH/bin/clang++.exe --offload-arch=gfx1100` is mandatory (see `build_windows.bat`) [VERIFIED: .planning/STATE.md + 08-CONTEXT].
- **Hard-coding `/opt/rocm` on Windows:** `HIP_PATH` env var is the indirection; `find_package(hip REQUIRED CONFIG PATHS "$ENV{HIP_PATH}/lib/cmake/hip")`.
- **LDS without banking fix:** `sB[32][32]` naive `float16` causes `4-way conflict → -75% BW` [CITED: output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md]; always `+33` or `XOR`.
- **Scalar `31×` `4B` loads for `qs`:** Flatten to `ulong2`/`float4` `16B` `b128` coalesced; `32 thr×4B→8×16B` [CITED: output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md].
- **Pipelining without `sched_barrier`:** Compiler reorders `GMEM→LDS→WMMA`; pin with `0x0080`/`0x0008` [CITED: llvm.org/docs/AMDGPUUsage.html].

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| WMMA fragment layout / VGPR budget | Custom lane→reg math | `amd_matrix_instruction_calculator -a gfx1100 -i wmma_f32_16x16x16_f16 -d -R --csv` [CITED: github.com/ROCm/amd_matrix_instruction_calculator] | Predicts `8 VGPR A/B / 8 VGPR D wave32` → `≤64 VGPR` before commit; hand math spills to `>64` → `8 waves/SIMD` collapse |
| LDS bank-conflict-free layout | Manual stride guessing | `CK Tile TileWindow + xor_preshuffle<KPerBlock,KPack>` or `+33` padding per `lds_bank_conflicts.html` [CITED: rocm.docs.amd.com/projects/composable_kernel/] | `4-way→-75% BW` if wrong; CK doc proves `ds_write_b128` 8-phase rule + `XOR 0%` vs `+12.5–25%` |
| Global coalescing width | Scalar `uint8` loads | `__builtin_amdgcn_global_load_b128`/`float4` `16B` + `__builtin_assume_aligned(ptr,16)` [CITED: output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md] | `SWDEV-556587` vector path is `8×` fewer transactions; scalar stays `dword` bound |
| GEMM tiling reuse math | Ad-hoc `BLOCK_M` | `loads/out = K·(1/M+1/N)` → `T=64→64×` reduction [CITED: output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md] | Naive `16×16` `512K` loads vs ideal `32K` (`16×`) — tiling is the `800 GB/s` roof fix |
| Pipelining overlap | `__syncthreads()` only | `GMEM→VGPR→LDS→VGPR→WMMA` 4-stage + `sched_barrier` [CITED: output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md + llvm.org/docs/AMDGPUUsage.html] | `__syncthreads` alone does not hide `GMEM` latency while WMMA runs (`MARLIN P=4` [CITED: output/deep-research/custom_kernel_pdfs/SYNTHESIS.md]) |
| Activation smoothing for `W8A8` | Per-channel quant hack | `SmoothQuant s_j = max|X_j|^α / max|W_j|^{1-α}`, `α=0.5` fused into `rmsnorm` [CITED: output/deep-research/custom_kernel_pdfs/SYNTHESIS.md] | `α=0.5` migrates `70→1` outlier without GEMM-incompatible per-channel scales |
| Dequant elision | Inline `kvalues` multiply per WMMA | `LUT-GEMM μ=4` 16-entry half LUT per tile (bake `d*(ls-32)`) [CITED: output/deep-research/custom_kernel_pdfs/SYNTHESIS.md] | Saves `scale*(ls-32)` multiply in hot loop; `32B/LUT` fits `sB` spare |
| Benchmark averaging / thermal bias | `N=1` + `for i in {1..10}; do bench; done` | `race.py --repeats 10` interleaved (`A,B,A,B...` not `AAAA BBBB`) [CITED: github.com/adelj88/rocm_wmma_gemm] | `thermal throttling` noise dominates `1.05–1.10×` deltas; interleaving is the `REQ-STAT-07` template |
| Windows build | `cmake -G "Visual Studio"` + `cl` | `build_windows.bat` + `clang++.exe --offload-arch=gfx1100` + `-G Ninja` | Only `clang++` understands `__builtin_amdgcn*`; `cl` fails on `hip` intrinsics |

**Key insight:** RDNA3 `gfx1100` rewards *exact* hardware idioms (`wave32`, `v_wmma 16×16×16`, `sudot4`, `ds_b128` 8-phase, `sched_barrier`) — each is a few characters but gates `16×` traffic, `75%` LDS BW, or `50%` occupancy. Hand-rolling any of them loses the `≥1.10×` margin.

## Common Pitfalls

### Pitfall 1: Measuring vs Naive, Not vs Real DP4A — Gate Instantly Failed
**What goes wrong:** `bench_gemv_dp4a` vs `stock_hip_comparator.hip` (543 µs) reports `6.43×` and declares win, but `bench_real_stock` shows real DP4A is `84 µs` [VERIFIED: .planning/STATE.md]; `>1.2×` vs naive becomes `1.00×` vs DP4A under jitter [VERIFIED: .planning/STATE.md].
**Why it happens:** Phase 5 compared vs scalar float; Phase 7 comparator is `vec_dot_iq4_xs_q8_1` (`sudot4`+`perm×6`+`ls-32` [VERIFIED: kernels/matmul_iq4xs/real_stock_dp4a_comparator.hip]).
**How to avoid:** Wire every `bench_*` to `matmul_real_stock_hip` object (`bench_real_stock --runs 10` is the denominator; `bench_gemv_dp4a --runs 10` numerator).
**Warning signs:** `BASELINE_DP4A.md` `speedup 3.89–13.81×` [VERIFIED: .planning/STATE.md] appears only when comparator is linked.

### Pitfall 2: WSL2 DXG Lies About VRAM and Jitter — `1.10×` Vanishes
**What goes wrong:** `llama-bench p=8192` appears `+5.1%` (`808→849`) on WSL2 but `mean−1σ` dips below `1.10×`; `800 GiB` reported vs `3.48 GiB` contiguous fail [CITED: output/deep-research/1000t-s-at-8k-gfx1100.md] causes `SKIPPED` without pre-flight; `15–30 µs` DXG tax flattens `1.178→1.00` [CITED: output/deep-research/1000t-s-at-8k-gfx1100.md]; `3–5 OOMs → BSOD` [CITED: output/deep-research/1000t-s-at-8k-gfx1100.md].
**Why it happens:** `WSL2` `DXGI/Hyper-V` `16 GiB` invisible overhead + non-realtime VM [CITED: output/deep-research/1000t-s-at-8k-gfx1100.md]; `HSA_ENABLE_DXG_DETECTION=1` is shim.
**How to avoid:** VRAM pre-flight: require `>2 GB` free + `KV≈128 KiB/tok ×8192≈1 GB` budget [CITED: output/deep-research/1000t-s-at-8k-gfx1100.md] or mark `8192: SKIPPED (preflight)`; run `N=10` interleaved in one thermal window with `hwinfo_daemon 1Hz` + `thermal_watchdog 90C`; step-up `-ngl 0→10→99` before `p=8192`.
**Warning signs:** `stddev/median >10%` across `N=10` tiers; `VmRSS` climb `>250 MB/min` [CITED: benchmarks/RUNBOOK.md §thresholds].

### Pitfall 3: LDS `[32][32]` Bank Conflict — `-75%` BW, `>1.2×` Lost
**What goes wrong:** `sB[32][32] _Float16` naive `64 KB` per `64×64` tile hits `4-way` conflict on `ds_read_b128` vertical `0:3+20:23` [CITED: output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md] → `1.00×` even with WMMA.
**Why it happens:** `bank=(addr/4)%32`, `32 banks×4B` [CITED: output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md]; `read_b128` needs `×` not `+` fix.
**How to avoid:** Variant A `+33` (`[32][33]` `+3%` [CITED: output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md]) vs Variant B `XOR preshuffle x'=(y%(64/8))⊕x` `0%` [CITED: output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md]; verify `rocprof lds_bank_conflict ==0` on bare metal (WSL2 blind) [CITED: output/deep-research/1000t-s-at-8k-gfx1100.md].
**Warning signs:** `rocprof` `LDS conflict 4.0` per access; `b128` count `÷8` mismatch indicates scalarization.

### Pitfall 4: `P=2` Insufficient at `8192` — GMEM Stall Hides Behind WMMA
**What goes wrong:** `sB[2][32][33]` double-buffer with `P-1=1` lookahead hides `LDS` fill at `M=512` but stalls at `M=8192` (`256×32` tiles) — `pp8192` `1.05×` not `1.10×`.
**Why it happens:** `MARLIN P=4` needed for `K=64` at `batch 64` [CITED: output/deep-research/custom_kernel_pdfs/SYNTHESIS.md]; `GMEM→VGPR` is slowest stage [CITED: output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md].
**How to avoid:** Ship `P=4` variant `sB[4][32][32]` (`+XOR` to save LDS) + `k_slice=32` (`2×WMMA` per slice) with `sched_barrier 0x0080` (DS) / `0x0008` (WMMA) [CITED: llvm.org/docs/AMDGPUUsage.html]; race `P=2` vs `P=4` `N=10` median.
**Warning signs:** `TFLOPS` flat across `M=1024→8192` while `median_us` scales linearly → memory not hidden.

### Pitfall 5: `cl` / `Visual Studio` Generator on Windows — HIP Intrinsics Fail
**What goes wrong:** `cmake -G "Visual Studio 17 2022"` + `cl` errors `unknown builtin __builtin_amdgcn_wmma*` / `sudot4` / `perm` and `*.hip` not compiled.
**Why it happens:** Only `HIP SDK clang++.exe` understands AMDGPU builtins; `cl` is MSVC [VERIFIED: .planning/STATE.md 07-04 gaps].
**How to avoid:** `build_windows.bat` must `set HIP_PATH` + `-G Ninja -DCMAKE_CXX_COMPILER="%HIP_PATH%\bin\clang++.exe" --offload-arch=gfx1100` + `find_package(hip REQUIRED CONFIG PATHS "$ENV{HIP_PATH}/lib/cmake/hip")`.
**Warning signs:** `error: use of undeclared identifier '__builtin_amdgcn_wmma_f32_16x16x16_f16_w32'` in Windows log.

### Pitfall 6: Single-Run LLM QA Without `temp=0` — Non-Determinism Masks Uplift
**What goes wrong:** `llama-cli -p "Hi" -n 128` with default `temp=0.8` gives `tok/s` variance `±15%` across runs → `median` vs `mean−1σ` diverge, gate fails on framing not code.
**Why it happens:** REQ-STAT-07 mandates `N=15` fixed prompt `temp=0` + per-run table [CITED: .planning/REQUIREMENTS.md REQ-STAT-07].
**How to avoid:** `for i in {1..15}; do llama-cli -p "Q: Explain attention, A:" --temp 0 -n 128 --single-turn ...; done` with `avg tok/s` + `stddev` + 15-row table in `KERNEL-BENCH-DIFF.md §8`.
**Warning signs:** `stddev/mean >8%` on same tier same build → prompt/temperature not fixed.

## Code Examples

Verified patterns from official sources:

### Q8_1 Coop Quantize + DP4A (GEMV 8-thread/row) — N=10 Rigour
```cpp
// Source: kernels/matmul_iq4xs/impl_gemv_dp4a_gfx1100.hip + bench_gemv_dp4a.cpp (verified pattern)
struct block_q8_1_coop { uint32_t ds; int8_t qs[32]; };
__global__ __launch_bounds__(256,4) __attribute__((amdgpu_flat_work_group_size(256,256)))
void gemv_dp4a_coop_kernel(const block_iq4_xs* W, const float* x, float* y, int K, int N) { /* ... sh[32][33], sudot4, perm */ }
// Harness (REQ-STAT-07):
// bench_gemv_dp4a --runs 10 --json  # emits median_us, mean_us, stddev_us, p95_us, speedup_median per 8 shapes
// bench_real_stock --runs 10 --json  # denominator 84µs median (not 543µs naive)
// Gate: >1.2× speedup_median across 8 shapes, mean−1σ >1.15×, cos ≥0.999 vs ref_cpu (10/10)
```

### WMMA `64×32 P=2→P=4` + `XOR` vs `+33` + `B-stationary` + `b128` (GEMM prefill)
```cpp
// Source: CK Tile lds_bank_conflicts.html + gemm_optimization.html + LLVM AMDGPUUsage
// Variant A: _Float16 sB[2][32][33]; // +33 padding, +3% [CITED: rocm.docs.amd.com/projects/composable_kernel/]
// Variant B:
template<index_t KPerBlock, index_t KPack>
__device__ constexpr index_t xor_preshuffle(index_t row, index_t col){ return (row % (KPerBlock/KPack)) ^ col; } // 0 overhead
// Load: offset = row*RowStride + xor_preshuffle<64,8>(row,col)*8; *reinterpret_cast<float4*>(lds+offset) = *reinterpret_cast<const float4*>(src);
__builtin_amdgcn_sched_barrier(0x0080); // DS before WMMA
c_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b_frag, c_frag, false);
__builtin_amdgcn_sched_barrier(0x0008); // WMMA fence
// Verification: rocprof lds_bank_conflict 0 + llvm-objdump | grep v_wmma + calculator VGPR ≤64
```

### Windows-Native Build Gate (REQ-WIN-07)
```batch
REM Source: .planning/REQUIREMENTS.md REQ-WIN-07 + 08-CONTEXT
set HIP_PATH=C:\Program Files\AMD\ROCm\6.4
set PATH=%HIP_PATH%\bin;%PATH%
where clang++.exe & clang++.exe --offload-arch=gfx1100 --version
cmake -S . -B build-windows -G Ninja -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DGGML_HIP=ON -DGGML_CUDA_ENABLE_CUSTOM_GFX1100=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="%HIP_PATH%\bin\clang++.exe"
cmake --build build-windows --config Release
REM No *.py outside llama.cpp/:  find -name "*.py" ! -path "./llama.cpp/*" == 0
build-windows\bin\llama-server.exe --model models\Qwen3.8-27B-IQ4_XS.gguf -ngl 99 --port 8000 &
curl -s http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}" | findstr choices
REM Expect 200 + choices[0].message.content
```

### Statistical Rigour: `N=10` + `race.py --repeats 10` Interleaved (REQ-STAT-07)
```bash
# Source: adelj88/rocm_wmma_gemm race.py + REQ-STAT-07
# Microbench (per 07-01/07-02/07-03):
./bench_gemv_dp4a --runs 10 --json | tee bench_gemv_dp4a_N10.json  # median/mean/stddev/p95/speedup_median
./bench_gemm_wmma --runs 10 --shapes 512x5120,1024x5120,8192x5120 --json | tee bench_gemm_wmma_N10.json
# Interleaved racing (thermal-bias kill):
python race.py --config1 gemm_config_gfx1100_64x32_P2_33.json --config2 gemm_config_gfx1100_64x64_P4_XOR.json --repeats 10  # A,B,A,B... not AAAA BBBB [CITED: github.com/adelj88/rocm_wmma_gemm]
# llama-bench 5 tiers × N=10 in ONE thermal window:
for tier in 512 1024 2048 4096 8192; do
  for rep in $(seq 1 10); do
    # interleave stock vs custom per rep, not per tier
    ./build-stock/bin/llama-bench -m models/Qwen3.8-27B-IQ4_XS.gguf -p $tier -n 64 --single-turn --simple-io --load-mode none -ngl 99 -b 2048 -r 1 | tee -a ab_stock_N10.jsonl
    ./build-custom/bin/llama-bench -m models/Qwen3.8-27B-IQ4_XS.gguf -p $tier -n 64 --single-turn --simple-io --load-mode none -ngl 99 -b 2048 -r 1 | tee -a ab_custom_N10.jsonl
  done
done
# Gate: median ≥1.10× AND mean−1σ ≥1.10× for pp AND tg at every tier 512..8192 (8192 SKIPPED if pre-flight fails, with FA+GQA rationale)
# LLM QA N=15:
for i in $(seq 1 15); do ./build-custom/bin/llama-cli -m models/Qwen3.8-27B-IQ4_XS.gguf -p "Q: Explain attention. A:" --temp 0 -n 128 --single-turn --simple-io -ngl 99 | tee -a llm_qa_N15.tsv; done
# Report avg tok/s + avg latency + stddev + 15-row table (single-run banned)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Naive float `stock_hip_comparator.hip` 543 µs (scalar dequant) | Real DP4A `vec_dot_iq4_xs_q8_1` 84 µs (`sudot4`+`perm`×6, `ls-32`, `d*low2float`) [VERIFIED: .planning/STATE.md] | Phase 7 07-01 (Aug 2026) | Comparator `6.43×` faster baseline — gate is `>1.2×` vs 84 µs, not 543 µs |
| `MMVQ` 1 warp/row (`MMVQ_PARAMETERS_RDNA3_0`) `calc_nwarps=1` | `8-thread coop` 256→32 rows/block, `LDS [32][33]`, `ulong2 16B` | Phase 7 07-02 | Peak 1.178× (WSL jitter) → target `>1.2×` bare metal, `16 waves/SIMD` |
| GEMM `TILE_M=16` scalar (1.76–7.5× vs naive) [VERIFIED: docs/PUBLICATION.md §8] | WMMA `64×32 [2][32][33]` double-buffer + `wmma_f32_16x16x16` (`1024 ops/CU/clock` [CITED: gpuopen.com/learn/wmma_on_rdna3]) | Phase 7 07-03 | `>950 t/s prefill` slice of `≥1.10×` gate; `64×64 P=4 XOR` is `8k` extension |
| LDS `[32][32]` naive (4-way conflict `-75%` BW) | `+33` (`+3%`) vs `XOR preshuffle 0%` [CITED: rocm.docs.amd.com/projects/composable_kernel/] | CK Tile 1.2.0 (2024–25), high-yield 2026-08-28 | `lds_bank_conflict 0` proof is the variant picker |
| `P=2` double-buffer `GMEM→LDS` | `P=4` quad-buffer `GMEM→VGPR→LDS→VGPR→WMMA` + `sched_barrier 0x0080/0x0008` [CITED: llvm.org/docs/AMDGPUUsage.html + output/deep-research/custom_kernel_pdfs/SYNTHESIS.md] | MARLIN 2023, CK `gemm_optimization.html` | Hides `800 GB/s` stall at `8192`; required for `mean−1σ ≥1.10×` |
| Scalar `4B` global loads (`dword`) | `b128` `float4`/`ulong2` `16B` (`global_load_b128`, `32 thr→8×16B`) [CITED: output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md] | `SWDEV-556587` | `8×` fewer transactions; offloads `800 GB/s` roof |
| Inline `d*(ls-32)*kvalues` per WMMA | `LUT-GEMM μ=4` 16-entry half LUT (`d*(ls-32)` baked) [CITED: output/deep-research/custom_kernel_pdfs/SYNTHESIS.md] | ICLR 2024 | `~5%` hot-loop saving; `32B/LUT` fits spare LDS |
| Per-token Q8_1 only (`max 70` outliers [CITED: output/deep-research/custom_kernel_pdfs/SYNTHESIS.md]) | `SmoothQuant α=0.5` `s_j=max|X_j|^α/max|W_j|^{1-α}` fused into `rmsnorm` → `W8A8 INT8 WMMA` arm | ICML 2023 | `1.5×` paper → `1.2×` realistic; comparator arm if `IQ4_XS` alone misses `1.10×` at `8192` |
| Single-run `tok/s` claims | `N=10` `median/mean/stddev/p95` + `race.py --repeats 10` interleaved [CITED: github.com/adelj88/rocm_wmma_gemm] + `N=15` LLM QA | REQ-STAT-07 (2026-08-28) | `mean−1σ ≥1.10×` is the gate, not `median` alone |
| WSL2-only HIP build (`/opt/rocm`) | Dual: WSL2 bare metal (`HSA_ENABLE_DXG_DETECTION=1`) AND Windows `HIP_PATH/bin/clang++.exe --offload-arch=gfx1100 -G Ninja` | REQ-WIN-07 (2026-08-28) | `≤2` langs; `15–30 µs` DXG tax removed on Windows path |
| Brute-force tile sweep | `tune.py` Genetic + RF surrogate `budget 100` [CITED: github.com/adelj88/rocm_wmma_gemm] | `adelj88` pattern | Preserves diversity via crowding; avoids local maxima in `64×32 vs 64×64 vs 128×32` sweep |

**Deprecated/outdated:**
- `stock_hip_comparator.hip` 543 µs as Phase 7 denominator — replaced by `real_stock_dp4a_comparator.hip` 84 µs; keep only as negative oracle.
- `BENCH-01 ≥3 repeats` — amended to `≥10` (`≥15` LLM QA) for Phase 7 onward; `N=1`/`N=3` tables are rejected at verifier.
- `M=16 WMMA attempt` — loses at `0.82×`; gated to `M≥512` WMMA, `M<512` falls back to tiled `TILE_M=16`.
- `hipBLASLt` on `gfx1100` — excluded per ROADMAP (Tensile gap, `gfx1100` not in `hipBLASLt`); pure WMMA without `hipBLASLt` validated by CK docs [CITED: output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md].

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Stock DP4A at `84 µs` (`attn_q`) vs `543 µs` naive is representative across 8 Qwen shapes (range `3.89–13.81×` per BASELINE_DP4A.md) [CITED: .planning/STATE.md] | Standard Stack / Pitfalls | Low — re-measure `bench_real_stock --runs 10` on bare metal; if shape spread differs, `>1.2×` gate still holds per-shape |
| A2 | `wmmma_f32_16x16x16_f16_w32` is native on `gfx1100` (`wave32` replicates `0–15→16–31`, `8 VGPR` frags) [CITED: gpuopen.com/learn/wmma_on_rdna3] | Standard Stack | Medium — if `OPSEL` or `w32` vs `w64` mismatched, `llvm-objdump` shows no `v_wmma`; fallback to `w64` loses `2×` but still passes vs DP4A |
| A3 | `LDS 32 banks×4B`, `ds_write_b128` 8-phase `0~7…56~63` conflict-free iff consecutive, `ds_read_b128` needs `+33` or `XOR` [CITED: output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md] | Architecture Patterns | Medium — if banking is `64×2B` variant, `+33` still works but XOR offset shifts; `rocprof lds_bank_conflict 0` resolves |
| A4 | `800 GB/s` (XT) / `960 GB/s` (XTX) pin caps `1000 tok/s` naive (`15.3 TB/s` needed) → `500–800 tok/s` fused ceiling [CITED: output/deep-research/1000t-s-at-8k-gfx1100.md] | Summary | Low — number is Wikipedia spec-derived; even if `1 TB/s`, conclusion unchanged (tiling required) |
| A5 | `WSL2` `15–30 µs` DXG jitter, `16 GiB` invisible overhead, `80 GiB` vs `3.48 GiB` fail, `BSOD` after `3–5 OOMs`, `rocprofv3` blind on WSL2 [CITED: output/deep-research/1000t-s-at-8k-gfx1100.md] | Summary / Pitfalls | Low — already observed (`avg 1.00 pk 1.178` [VERIFIED: .planning/STATE.md]); mitigated by native `rocprof` + Windows path |
| A6 | `MARLIN P=4` sufficient to hide `GMEM→LDS` while WMMA runs at `8192` [CITED: output/deep-research/custom_kernel_pdfs/SYNTHESIS.md] | State of the Art | Medium — if `P=4` spills LDS beyond `64 KB` CU, `P=3` or `64×32` fallback is the mitigant; race picks winner |
| A7 | `LTR`? `SmoothQuant α=0.5` (or `0.75` for outlier-heavy) fused into `rmsnorm` yields `W8A8` `≈FP16` on `PPL 6.4271` gate [CITED: output/deep-research/custom_kernel_pdfs/SYNTHESIS.md] | State of the Art | Medium — requires `512`-sentence Pile calibration offline; if `PPL` drifts `>1%`, `α` sweep `0.5/0.6/0.75` is the fix |
| A8 | `Windows` `HIP SDK 6.4+` `clang++.exe --offload-arch=gfx1100` supports `__builtin_amdgcn_global_load_b128`/`sudot4`/`wmma` identical to WSL2 `hipcc` [ASSUMED] | Standard Stack | High if wrong — probe `build_windows.bat` compile gate first; if missing, `ulong2`/`float4` still vectorizes via `dwordx4` fallback |
| A9 | `KV≈128 KiB/tok` GQA estimate → `8k≈1 GB` (`64L/8KV/128hd` `262 KiB/tok` worst) [CITED: output/deep-research/1000t-s-at-8k-gfx1100.md] | Summary | Medium — dump `llama-gguf --json` to replace estimate; if `KV` is `512 KiB/tok`, `8192` is OOM and `SKIPPED` rationale strengthens |
| A10 | `adelj88 tune.py` RF surrogate `budget 100` translates to our `64×32 vs 64×64 vs 128×32` sweep with `N=10` `median` picker [CITED: output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md] | Standard Stack | Low — `budget 100` is illustrative; actual budget is `N=10` per variant `×6` variants = 60 compiles, tractable |

**If this table is empty:** All claims in this research were verified or cited — no user confirmation needed.

## Open Questions

1. **VRAM pre-flight for `8192` on 20 GB**
   - What we know: `15.31 GB` model + `KV 1–2 GB` (`128 KiB/tok`) + `0.3 GB` transient = `16.7–18.9 GB` on `20 GB` [CITED: output/deep-research/1000t-s-at-8k-gfx1100.md]; `800 GiB` lie risk [CITED: output/deep-research/1000t-s-at-8k-gfx1100.md].
   - What's unclear: Measured `mem_get_info` free vs contiguous `3.48 GiB` fail [CITED: output/deep-research/1000t-s-at-8k-gfx1100.md] threshold on this `7900 XT` AD + VRAM? Exact `llama.cpp` `KV` per token for this `IQ4_XS` gguf.
   - Recommendation: `llama-bench` harness `preflight` must check `available >2 GB` after `KV` budget *and* attempt `hipMalloc(3.5 GiB)` probe; if fail, emit `8192: SKIPPED (preflight-oom, KV≈X KiB/tok)` with `FA {on,off}` rationale — do not BSOD.

2. **`XOR` vs `+33` winner at `64×64`**
   - What we know: `+33` `+3%` proven `4-way→0` [CITED: rocm.docs.amd.com/projects/composable_kernel/]; `XOR (y%(64/8))⊕x` `0%` [CITED: rocm.docs.amd.com/projects/composable_kernel/].
   - What's unclear: XOR adds one `^` per `b128` access — does it cost `1 cycle` vs `+3%` LDS at `64×64` (`16 KB` vs `64 KB` limit)?
   - Recommendation: Ship both variants behind compile flag, `race.py --repeats 10` interleaved `N=10` median; pick by `lds_bank_conflict 0` AND `median_us`.

3. **`P=2 vs P=4` occupancy tradeoff**
   - What we know: `P=4` is `GMEM→VGPR→LDS→VGPR→WMMA` 4-stage overlap [CITED: output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md]; `MARLIN P=4` sufficient [CITED: output/deep-research/custom_kernel_pdfs/SYNTHESIS.md].
   - What's unclear: `sB[4][32][33]` `16 KB` vs `8 KB` for `2` buffers — still under `64 KB` CU but VGPR pressure may push `>64` → `8 waves/SIMD`.
   - Recommendation: Calculator predicts `VGPR` for each `P`; if `>64`, keep `P=2` + `B-stationary` as fallback winner.

4. **`LUT μ=4` vs `μ=8` for `IQ4_XS`**
   - What we know: `μ=4` → `16` entries `32B/LUT` [CITED: output/deep-research/custom_kernel_pdfs/SYNTHESIS.md]; `μ=8` → `256` `512B/LUT` still LDS-friendly but `4×` less reuse.
   - What's unclear: Does LUT baking (`d*(ls-32)`) save enough to offset LUT load?
   - Recommendation: Start `μ=4` (`impl_gemm_lut_iq4xs.hip`); `μ=8` only if `median` wins by `>3%` on `M=8192`.

5. **`SmoothQuant W8A8` arm priority**
   - What we know: `O3` `1.56×` paper [CITED: output/deep-research/custom_kernel_pdfs/SYNTHESIS.md]; `α=0.5` `s_j` [CITED: output/deep-research/custom_kernel_pdfs/SYNTHESIS.md].
   - What's unclear: Locked artifact is `IQ4_XS 15.31 GB`; `W8` is `2×` bits → VRAM tradeoff at `8192`.
   - Recommendation: Keep `IQ4_XS` primary; `W8A8` as 07-04 comparator arm (same WMMA path, `i32_iu8`) for `pp8192` if `IQ4_XS` alone <`1.10×`.

6. **`b128` coalescing count verification**
   - What we know: `global_load_b128` exists (`SWDEV-556587`) [CITED: output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md]; `float4 16B` maps to it.
   - What's unclear: Does `clang++.exe` on Windows auto-vectorize `ulong2` to `b128` or need explicit `__builtin_amdgcn_global_load_b128`?
   - Recommendation: `llvm-objdump` check for `global_load_b128` vs `buffer_load_dword`; if scalar, switch to explicit builtin.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|---------|----------|
| `ROCm 7.2.1` + `hipcc` (`/opt/rocm-7.2.1`) | WSL2 playground `kernels/build` + bare-metal benches | ✓ (WSL2) | `HIP 7.2.53211` / clang `22.0.0git` [VERIFIED: docs/PUBLICATION.md §2] | — |
| `HIP SDK Windows` (`HIP_PATH`) + `clang++.exe --offload-arch=gfx1100` + `Ninja` | `REQ-WIN-07` `build_windows.bat` + `llama-server :8000` | ✗ (not probed on this runner; `config brav/tavily` false) | — (expect `6.4+`) | WSL2 `hipcc` compiles but does not prove Windows `≤2` langs; must probe on Windows 11 host |
| `llama.cpp` pin `bb4caa75` + `build-stock`/`build-custom` | Paired `llama-bench` `N=10` 5 tiers | ✓ (tree present) | `bb4caa75` [VERIFIED: .planning/STATE.md] | — |
| `Qwen3.8-27B-IQ4_XS.gguf` `53adc4bb…` (15.31 GB) | E2E `pp/tg` 5 tiers + `PPL` + `LLM QA N=15` | ✓ (provenance in `models/README.md`) | sha256 `53adc4…` | If absent, `M=8192` benches `SKIPPED` (model-gated) |
| `rocprofv3` + `llvm-objdump` + `hipcc --save-temps` | `lds_bank_conflict 0`, `v_wmma/v_dot4`, `VGPR ≤64` gates | `llvm-objdump` ✓; `rocprofv3` ✗ on WSL2 (Instinct-only/404) [CITED: output/deep-research/1000t-s-at-8k-gfx1100.md] | LLVM `22.0.0git` | Run `rocprof` on native bare metal (Windows/Linux), not under `librocdxg`; `llvm-objdump` works everywhere |
| `amd_matrix_instruction_calculator` (`tabulate`) | Pre-commit VGPR oracle | ○ (offline, not shipped) | latest `ROCm/amd_matrix_instruction_calculator` | Hand-estimate `8/4 VGPR` but gate still `hipcc --save-temps` |
| `hwinfo_daemon.py` + `thermal_watchdog.py` (90C) | Thermal pairing `N=10` one window | ○ (present in `benchmarks/host/`) | — | `absent` degraded mode (`/proc/pid/status` RSS) per `benchmarks/RUNBOOK.md §telemetry-modes` |
| `HSA_ENABLE_DXG_DETECTION=1` + `rocminfo gfx1100` | WSL2 pre-flight | ✓ | — | Native `rocminfo` without `HSA_ENABLE_DXG_DETECTION` on Windows/Linux |

**Missing dependencies with no fallback:**
- `Windows HIP SDK` not present on this runner — `build_windows.bat` + `llama-server :8000` cannot be proven here; must run on `Windows 11 + VS Build Tools + HIP SDK` host (see 08-CONTEXT).

**Missing dependencies with fallback:**
- `rocprofv3` on WSL2 — use native `rocprof` on bare metal for `lds_bank_conflict 0`; WSL2 benches still produce `median/mean/stddev/p95` but banking proof is deferred.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | `ctest` via `kernels/matmul_iq4xs` CMake + `benchmarks/bin/run_op_gate.py` + `llama-bench`/`llama-cli` (llama.cpp) |
| Config file | `kernels/matmul_iq4xs/CMakeLists.txt` + `benchmarks/RUNBOOK.md` (§session-protocol, §thresholds, §telemetry-modes, §thermal-policy) |
| Quick run command | `HSA_ENABLE_DXG_DETECTION=1 ctest --test-dir kernels/build -R test_ --output-on-failure` (timeout 90s) |
| Full suite command | `HSA_ENABLE_DXG_DETECTION=1 python benchmarks/bin/run_op_gate.py --runs 10` (0 errors) && `python benchmarks/bin/run_model_gate.py --runs 10` (PPL 6.4271±1%) && `./bench_real_stock --runs 10` && `./bench_gemv_dp4a --runs 10` && `./bench_gemm_wmma --runs 10` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| REQ-WIN-07 | Windows-native `≤2` langs: `build_windows.bat` + `clang++.exe --offload-arch=gfx1100 -G Ninja` clean + `llama-server :8000 →200` | smoke/e2e (manual on Windows host) | `build_windows.bat` (timeout 300s) && `curl http://127.0.0.1:8000/v1/chat/completions → 200` && `find -name "*.py" ! -path "./llama.cpp/*" == 0` | ❌ Wave 0 — `build_windows.bat` absent (must author) |
| REQ-PERF-07 | `≥1.10× pp+tg` at `{512,1024,2048,4096,8192}` `median` + `mean−1σ ≥1.10×` over `N=10` thermal-paired `llama-bench` (stock OFF vs custom ON) | e2e `llama-bench` `N=10` | `llama-bench -m IQ4_XS.gguf -p {512..8192} -n 64 --single-turn --simple-io --load-mode none -ngl 99 -b 2048 -r 10` per tier per build in ONE window, `RunStore`+`CHECKSUMS` | ❌ Wave 0 — paired `benchmarks/results/phase7/ab_*` + `CHECKSUMS` absent (verifier GAP5) |
| REQ-STAT-07 | Every number `N≥10` `median/mean/stddev/p95`, LLM QA `N≥15` `temp=0` `avg tok/s`+per-run table | harness | `bench_* --runs 10 --json` + `race.py --repeats 10` interleaved + `llama-cli --temp 0 -n 128 --runs 15` | ❌ Wave 0 — `bench_* --runs 10` flag present but prior logs are `N=1/3`; need `N=10` re-bench |
| BENCH-01 (amended) | `≥10` repeats `pp/tg` split warmup `RunStore`+`CHECKSUMS`; LLM QA `≥15` | harness | Same as REQ-STAT-07 | ❌ Wave 0 — same gap |
| KERN-04 (DP4A GEMV) | `impl_gemv_dp4a_gfx1100.hip` 8-thread coop `sudot4+perm`, `sh[32][33]`, `>1.2× median` vs real DP4A `N=10`, `>38 t/s decode` | unit + microbench + `llvm-objdump v_dot4` + `VGPR ≤64` | `./test_gemv_dp4a_compare --runs 10` (cos ≥0.999) && `./bench_gemv_dp4a --runs 10 --json` (speedup_median >1.2) && `llvm-objdump --mcpu=gfx1100 | grep v_dot4` && `hipcc --save-temps -Rpass-analysis` | ✅ `impl_gemv_dp4a_gfx1100.hip` + `test/bench` exist [VERIFIED: kernels/matmul_iq4xs/] but `>1.2×` not yet proven on bare metal (pk 1.178 avg 1.00 [VERIFIED: .planning/STATE.md]) |
| KERN-05 (WMMA GEMM) | `impl_gemm_wmma_stream.hip` `64×32 [2][32][33]` `wmma_f32_16x16x16`, `>950 t/s prefill`, `rocprof lds 0`, `VGPR ≤64`, `v_wmma` | unit + microbench + `rocprof` + disasm | `./test_gemm_wmma_compare --runs 10` (cos ≥0.999, 15 shapes) && `./bench_gemm_wmma --runs 10 --shapes 128/512/1024/8192` (speedup_median >1.2 at M≥512) && `llvm-objdump | grep v_wmma` && `rocprof lds_bank_conflict 0` | ✅ `impl_gemm_wmma_stream.hip` exists [VERIFIED: kernels/matmul_iq4xs/] but `bench vs real DP4A` not run on bare metal, `>950 t/s` unmeasured |
| INTEG-02 (patch + gates) | Quilt `patches/0001-gfx1100-mul-mat-custom.patch` `git apply --check` PASS both OS + `QUAL-01 0 errors N=10` + `QUAL-02 PPL 6.4271 N=10` | integration | `git -C llama.cpp apply --check ../patches/0001*` && `cmake -B build-custom -DGGML_CUDA_ENABLE_CUSTOM_GFX1100=ON` && `run_op_gate.py --runs 10` && `run_model_gate.py --runs 10` | ✅ Patch 355 lines PASS [VERIFIED: .planning/STATE.md]; ❌ `op_gate_custom N=10` not yet executed on metal (verifier GAP3) |

### Sampling Rate
- **Per task commit:** `ctest -R test_ --output-on-failure` (timeout 90s)
- **Per wave merge:** `bench_* --runs 10` + `llama-bench` `N=10` sweep per variant + `N=15` LLM QA
- **Phase gate:** Full suite green before `/gsd-verify-work` — all 7 truths `median` + `mean−1σ` + `lds 0` + `VGPR ≤64` + `v_wmma/v_dot4` + `Windows :8000 200` must be in `benchmarks/results/phase7/` + `docs/PUBLICATION.md`

### Wave 0 Gaps
- [ ] `build_windows.bat` — `HIP_PATH` + `clang++.exe --offload-arch=gfx1100 -G Ninja` author + `find_package(hip)` + `*.patch text eol=lf` (REQ-WIN-07)
- [ ] `kernels/matmul_iq4xs/bench_* --runs 10 --json` re-bench on WSL2 bare metal (median/mean/stddev/p95 + speedup_median per 8 shapes) — `bench_real_stock` denominator 84µs [VERIFIED: .planning/STATE.md]
- [ ] `benchmarks/results/phase7/ab_stock_*` + `ab_custom_*` `N=10` per tier `512..8192` `RunStore`+`CHECKSUMS` + `hwinfo_daemon` 1Hz + `thermal_watchdog 90C` (REQ-PERF-07/REQ-STAT-07)
- [ ] `benchmarks/results/phase7/op_gate_custom_N10.json` + `run_model_gate_N10.json` (0 errors, PPL 6.4271±1% `N=10`) + `llm_qa_N15.tsv` (`15` per-run table)
- [ ] `llvm-objdump --mcpu=gfx1100` `v_wmma`/`v_dot4` logs + `hipcc --save-temps -Rpass-analysis` `VGPR ≤64` (16 waves/SIMD) per kernel
- [ ] `rocprof lds_bank_conflict 0` on native bare metal for winning `XOR` vs `+33` variant (WSL2 blind [CITED: output/deep-research/1000t-s-at-8k-gfx1100.md])
- [ ] `amd_matrix_instruction_calculator -a gfx1100 -i wmma_* -d -R --csv` pre-commit dump for `64×32` / `64×64` / `128×32`

*(If no gaps: "None — existing test infrastructure covers all phase requirements")*

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — (local `llama-server :8000` no auth; not a network service) |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes | `GGUF` `sha256 53adc4bb…` verified before bench; `K,N,M` bounds-checked (`m1 = min(m0+TILE, M)`), `hipMalloc` fail-fast, no `retry` loops (BSOD avoidance [CITED: output/deep-research/1000t-s-at-8k-gfx1100.md]) |
| V6 Cryptography | no | — (no secrets; `LICENSE Apache 2.0` + `NOTICE` already delivered [VERIFIED: docs/PUBLICATION.md]) |
| V7 Error Handling | yes | `HSA_ENABLE_DXG_DETECTION=1` pre-flight `rocminfo` required before heavy runs; bounded timeouts `90s`/`300s`; `thermal_watchdog 90C` kills; `.wslconfig memory=28GB` [VERIFIED: docs/PUBLICATION.md §2] |
| V8 Data Protection | yes | `RunStore` append-only `rows.jsonl` fsynced + `CHECKSUMS.sha256` [VERIFIED: docs/PUBLICATION.md §6]; `wsl --export` snapshot after Phase 1 |
| V10 Malicious Code | yes | Quilt patch `git diff HEAD` over `bb4caa75` provenance; `git apply --check` on both OS; no `postinstall` scripts (pure `C++/HIP`) |

### Known Threat Patterns for HIP/C++ stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| GGUF supply-chain tamper | Tampering | `sha256 53adc4bb…` pinned in `models/README.md`; `find -name "*.py" ! -path "./llama.cpp/*" == 0` removes Python injection surface (REQ-WIN-07) |
| VRAM overcommit → host BSOD | Denial of Service | Pre-flight `>2 GB` free + `hipMalloc` probe + `RSS` guard `22.7 GB` fail threshold [CITED: benchmarks/RUNBOOK.md §thresholds]; no retry loops per microsoft/WSL#40732 [CITED: output/deep-research/1000t-s-at-8k-gfx1100.md] |
| DXG deadlock / TDR hang | DoS | Step-up `-ngl 0→10→99`, bounded timeouts, `thermal_watchdog` `wsl.exe -d Ubuntu-24.04 -- kill -9` [CITED: benchmarks/RUNBOOK.md §session-protocol] |
| Uninitialized LDS read | Information Disclosure | `sB[2][32][33]` double-buffer + `__syncthreads()` uniform, prior `WMMA` uninit-LDS fix [VERIFIED: .planning/STATE.md Phase 6] |
| Barrier divergence | DoS (hang) | Uniform `__syncthreads()` per `WARP_SIZE` templated, thermos remediation [VERIFIED: .planning/STATE.md] |

## Sources

### Primary (HIGH confidence)
- `kernels/matmul_iq4xs/impl_gemv_dp4a_gfx1100.hip:1-80` — `256 thr`, `sh[32][33]`, `__launch_bounds__(256,4)`, `sudot4/perm`, `ulong2 16B` [VERIFIED: read this session]
- `kernels/matmul_iq4xs/impl_gemm_wmma_stream.hip:1-80` — `64×32`, `sB[2][32][33]`, `wmma_f32_16x16x16`, `TILE_M=16` fallback [VERIFIED: read this session]
- `kernels/matmul_iq4xs/CMakeLists.txt` — `matmul_real_stock_hip` + `matmul_gemv_dp4a_hip` + `matmul_gemm_wmma_stream_hip` targets
- `docs/PUBLICATION.md §2/§6` — `ROCm 7.2.1` / `clang 22.0.0git` / `HIP 7.2.53211` / `RunStore`+`CHECKSUMS` [VERIFIED: read this session]
- `benchmarks/RUNBOOK.md` — `§thresholds` `VmRSS 15.19 GB` / `21.73 GB fail`, `§thermal-policy 95C`, `§telemetry-modes` [VERIFIED: read this session]

### Secondary (MEDIUM confidence)
- `gpuopen.com/learn/wmma_on_rdna3/` — RDNA3 WMMA `16×16×16`, `wave32` replication `0–15→16–31`, `512 FLOP/clock/CU` [CITED: output/deep-research/1000t-s-at-8k-gfx1100.md]
- `rocm.docs.amd.com/projects/rocWMMA/` — header-only, `CUDA WMMA`-compat, `MFMA` vs `WMMA/SWMMAC` [CITED: output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md]
- `rocm.docs.amd.com/projects/composable_kernel/.../lds_bank_conflicts.html` — `32 banks×4B`, `ds_write_b128` 8-phase, `+33 4-way→0` (`-75%` if not), `XOR` `x'=(y%(64/8))⊕x` [CITED: output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md]
- `rocm.docs.amd.com/projects/composable_kernel/.../gemm_optimization.html` — `loads/out=K·(1/M+1/N)`, `T=64→64×`, `B-stationary`, 4-stage pipeline `GMEM→VGPR→LDS→VGPR→WMMA` [CITED: output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md]
- `llvm.org/docs/AMDGPUUsage.html` — `sched_barrier 0x0080` DS, `0x0008` WMMA/MFMA, `sched_group_barrier` [CITED: output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md]
- `rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html` — `__launch_bounds__(256,4)`, `amdgpu_flat_work_group_size(256,256)`, `dim3` [CITED: output/deep-research/1000t-s-at-8k-gfx1100.md]
- `.planning/REQUIREMENTS.md` REQ-WIN-07 / REQ-PERF-07 / REQ-STAT-07 / BENCH-01 amended — `≤2` langs, `≥1.10× pp+tg` `mean−1σ`, `N=10`/`N=15` [CITED: local]
- `.planning/STATE.md` `stopped_at` — `84µs vs 543µs`, `1.178 pk avg 1.00`, `WMMA [2][32][33]`, `355 lines` [CITED: local]
- `github.com/ROCm/amd_matrix_instruction_calculator` + `github.com/adelj88/rocm_wmma_gemm` — calculator `VGPR ≤64` + `tune.py budget 100` + `race.py --repeats 10` [CITED: output/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md]
- `output/deep-research/1000t-s-at-8k-gfx1100.md` — `800GB/s roof`, `KV≈128KiB/tok`, `WSL2 16GiB lie+BSOD`, `jitter`, `rocprof blind` [CITED: local]
- `output/deep-research/custom_kernel_pdfs/SYNTHESIS.md` — `MARLIN P=4+16×64`, `LUT μ=4`, `SmoothQuant α=0.5` [CITED: local]

### Tertiary (LOW confidence)
- `ROCm 10.0.0` `"optimized for Radeon + Windows"` homepage claim vs `Instinct-only` detail pages — conflicting, treated as aspirational until `HIP SDK 6.4+` proven on host [ASSUMED]
- `16×64` swizzle offline `tools/swizzle_iq4xs.py` layout — proposed, not yet implemented [ASSUMED]

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — `ROCm 7.2.1`/`bb4caa75`/quilt/CMAKE paths verified in repo; WMMA/CK/LLVM docs cited via authoritative `gpuopen`/`rocm.docs`/`llvm.org`
- Architecture: HIGH — LDS `32×4B`/`8-phase b128`, tiling `T=64→64×`, piping `P=4`, `B-stationary`, `XOR` vs `+33`, `Smooth α=0.5`, `LUT μ=4` all mapped to primary sources above
- Pitfalls: HIGH — `84µs vs 543µs`, `1.00 avg`, `32-bank`, `cl` failure, `BSOD` all reproduced in verifier reports and deep-research

**Research date:** 2026-08-28
**Valid until:** 2026-09-27 (stable `gfx1100` ISA; re-verify if `ROCm ≥8` or `llama.cpp >bb4caa75` or new `HIP SDK` ships)
