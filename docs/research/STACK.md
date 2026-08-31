# Stack Research — GPU LLM Inference Optimization (AMD RDNA3 / gfx1100)

**Domain:** HIP/ROCm kernel-level inference optimization on RX 7900 XT (gfx1100, 20 GB) under WSL2
**Researched:** 2026-08-21
**Confidence:** HIGH (core stack verified against official AMD ROCm docs + first-party HuggingFace API data; WSL2 profiling items individually graded below)

> **Scope note:** "Qwen3.8-27B" does not exist (LOCKED scoping decision). Canonical candidates are **Qwen3-32B (dense)** vs **Qwen3-30B-A3B (MoE)**; final pick PENDING-USER. This file prescribes the stack for both and gives the VRAM-fit logic needed to decide. One material correction for PROJECT.md/ROADMAP: official Qwen3-32B **Q4_K_M is 19.76 GB** (not "~16–17 GB" as assumed in planning docs) — it does **not** fit fully on a 20 GB card. See Model Artifacts section.

---

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended | Conf. |
|------------|---------|---------|-----------------|-------|
| Windows driver: AMD Software Adrenalin Edition **for WSL2** | **26.2.2** | Host-side GPU driver exposing DXCore (`/dev/dxg`) to WSL2 | This exact release introduces production support for ROCDXG/librocdxg — AMD's open-source WSL compute path (announced in the official ROCm-on-Radeon WSL how-to). It replaces the deprecated `roc4wsl` packaging. No version pairing needed with guest ROCm. | HIGH |
| ROCm (guest, WSL2) | **7.2.1** (ROCDXG path) | HIP compiler, runtime, rocBLAS, rocprofiler-sdk inside Ubuntu WSL2 | ROCm 7.2.1 is the release paired with production ROCDXG support. ROCDXG needs the ROCm ≥7.1 runtime feature; 7.2.1 is the documented, supported combination for Radeon GPUs incl. RDNA3. Install per the **ROCm/librocdxg GitHub Quickstart** (user-mode only — no RSL/dkms/kernel modules in the guest). | HIGH |
| WSL2 guest OS | **Ubuntu 24.04 LTS** (22.04 also supported) | Dev/build/profiling environment | Officially supported guests for ROCm-on-WSL (AMD recommends 22.04 historically; both 22.04 & 24.04 listed "Yes" on WSL2-Linux-Kernel 5.15). Use 24.04 for fresher clang/gcc/cmake unless a doc says otherwise. Kernel comes from Microsoft's WSL2 kernel 5.15 line — do not swap to a custom kernel; `/dev/kfd` will never appear, `/dev/dxg` is the interface. | HIGH |
| llama.cpp (ggml-org) | master, pinned commit at execution | Reference runtime + patch target | Explicit decision (locked): build on llama.cpp, replace hot paths incrementally. Current HIP backend uses CMake-native HIP support; `GPU_TARGETS` is the canonical arch flag (legacy `AMDGPU_TARGETS` is auto-forwarded to it). AMD itself publishes a llama.cpp-on-ROCm install guide (docs-26.02). Pin a commit and record it with every benchmark result. | HIGH |
| HIP backend flags | `-DGGML_HIP=ON -DGPU_TARGETS=gfx1100` | Compile all ggml HIP kernels for exactly this silicon | Single-target builds cut compile time massively vs default multi-arch and guarantee you're profiling the ISA you'll hand-tune. gfx1100 is an officially documented example target in llama.cpp's build docs. Do **not** set `HSA_OVERRIDE_GFX_VERSION` — gfx1100 needs no override. | HIGH |

### Supporting Libraries

| Library | Version | Purpose | When to Use | Conf. |
|---------|---------|---------|-------------|-------|
| rocBLAS (+ hipBLAS) | ships with ROCm 7.2.x | BLAS fallback used by ggml HIP backend | Present automatically; the comparison point when answering "does a custom kernel beat the vendor GEMM?" (original Phase 6 question). | HIGH |
| rocWMMA | 2.x (headers) | Warp-matrix primitives for RDNA3 wave32 | Only via llama.cpp's optional `-DGGML_HIP_ROCWMMA_FATTN=ON` FlashAttention path, and as reference material for your own WMMA usage in custom kernels. **Treat as an A/B benchmark flag, never an unconditional default**: huge PP gains reported on 7900 XTX, but a −41% long-context prefill regression was filed on gfx1151 and compile failures exist against certain ROCm versions (e.g. 6.4.4). Benchmark ON vs OFF per workload. | MEDIUM |
| rocprofiler-sdk (rocprofv3) | ships with ROCm 7.2.x | Kernel tracing + dispatch/hardware-counter profiling | The profiler of record for Phase 3. WSL2 compute+profiling support for the gfx11 family landed in rocprofiler-sdk over `/dev/dxg` (rocm-systems PR #7016): agent enumeration, counter registration, kernel launch tracing and dispatch profiling work without `/dev/kfd`. Validate on day 1 (Phase 1 gate) — see WSL2 Verdict. | MEDIUM |
| librocdxg | latest from GitHub Quickstart | The WSL translation layer itself | Installed once during environment bring-up; loosely coupled to ROCm/driver versions thereafter. | HIGH |
| Ninja, ccache | latest | Build orchestration / rebuild speed | Kernel-dev means dozens of full rebuilds; `-G Ninja` matches the roadmap's example and ccache saves minutes per cycle. | HIGH |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `rocminfo`, `hipconfig`, `hipcc`/amdclang++ | Environment validation + kernel playground compiles | `rocminfo \| grep gfx` must show `gfx1100`; record `hipconfig --full` output into `benchmarks/environment/` every result set. |
| `llama-bench`, `llama-cli`, `llama-server`, `llama-perf` (built with the tree) | Baseline throughput/latency harnesses | llama.cpp prints its own VRAM breakdown (weights / KV buffer / compute buffer) at startup — this becomes your **primary VRAM accounting tool under WSL2**, because rocm-smi does not work there (see below). |
| rocprofv3 CLI | Phase 3 profiling: `--hip-trace`, `--kernel-trace`, dispatch counters | Runs the target app unmodified; outputs per-kernel timings/counters. Confirm gfx1100 counter collection under WSL2 in Phase 1 before committing to it. |
| Windows-side: Task Manager GPU view / Adrenalin performance overlay | VRAM + utilization cross-check | Because guest-side `rocm-smi`/`radeontop` don't function under WSL2, host-side observation is the honest second source for peak VRAM. |
| git + benchmark CSV/JSON discipline | Bisectability | Roadmap rule: record compiler/ROCm/driver/commit versions with every run. |

---

## Installation

```bash
# ── Windows host (once) ─────────────────────────────────────────────
#   1. Install "AMD Software: Adrenalin Edition 26.2.2 for WSL2" driver
#   2. wsl --install -d Ubuntu-24.04   (WSL2, default kernel 5.15 line)

# ── Inside WSL2 guest (Ubuntu 22.04/24.04) ──────────────────────────
#   3. ROCm 7.2.x via the ROCm/librocdxg GitHub Quickstart
#      (user-mode library; NO dkms, NO amdgpu-dkms, NO kernel driver in guest)
#      https://github.com/ROCm/librocdxg  → Quickstart
#   4. Verify the compute path:
rocminfo | grep -E "Name|Marketing"     # expect: gfx1100 agent
hipconfig --full
/opt/rocm/bin/rocprofv3 --version       # profiler present

# ── llama.cpp gfx1100 baseline build ────────────────────────────────
sudo apt install -y git cmake ninja-build ccache build-essential
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git rev-parse HEAD > ../LLAMA_COMMIT.txt        # pin for reproducibility

HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
cmake -S . -B build -G Ninja \
  -DGGML_HIP=ON \
  -DGPU_TARGETS=gfx1100 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# Optional A/B variant (bench separately, do NOT make the default):
#   -DGGML_HIP_ROCWMMA_FATTN=ON     # rocWMMA FA path, RDNA3 wave32
# Legacy alias (older docs/scripts): -DAMDGPU_TARGETS=gfx1100 == GPU_TARGETS

# ── Smoke test ──────────────────────────────────────────────────────
./build/bin/llama-bench -m <model>.gguf -p 512 -n 128 -ngl 99
```

---

## Model Artifacts & VRAM-Fit Analysis (20 GB RX 7900 XT)

Exact sizes pulled from the HuggingFace API (first-party, HIGH confidence).

### Candidate A — Qwen3-32B (dense; 64 layers, 64Q/8KV heads, head_dim 128 → **f16 KV = 256 KiB/token**)

| Repo / File | Quant | Size (GB) | Fits 20 GB full-GPU? |
|-------------|-------|-----------|----------------------|
| `Qwen/Qwen3-32B-GGUF` → `Qwen3-32B-Q4_K_M.gguf` | Q4_K_M | **19.76** | ❌ **No.** Weights alone exceed any sane budget once KV+buffers (~1–1.5 GB) are added; would force layer/CPU-expert offload — wrong shape for a GPU-kernel project. |
| `bartowski/Qwen_Qwen3-32B-GGUF` → `Qwen_Qwen3-32B-Q4_K_S.gguf` | Q4_K_S | **18.77** | ⚠️ Marginal: ≈20.3 GB at 4 K f16 KV. Only viable with tiny contexts or quantized KV; uncomfortable daily driver. |
| `bartowski/Qwen_Qwen3-32B-GGUF` → `Qwen_Qwen3-32B-IQ4_XS.gguf` | IQ4_XS | **17.69** | ✅ Borderline-yes: ≈19.7 GB at 4 K f16 / ≈19.2 GB at 8 K with Q8_0 KV. Working quant *if* dense-32B is chosen. |
| (mradermacher equivalent) `Qwen3-32B-Q3_K_M` | Q3_K_M | ~15.97 | ✅ Comfortable at 8 K f16 (~19 GB), 16 K with Q8 KV. Quality step down. |
| `Qwen/Qwen3-32B-GGUF` → `Qwen3-32B-Q5_K_M.gguf` | Q5_K_M | **23.21** | ❌ Never fits. Drop Q5_K_M from the baseline matrix for this card (original roadmap said "if VRAM permits" — it does not permit). |

### Candidate B — Qwen3-30B-A3B (MoE, 128 experts/8 active, 30.5 B total / 3.3 B active; 48 layers, 32Q/4KV heads, head_dim 128 → **f16 KV = 96 KiB/token**)

| Repo / File | Quant | Size (GB) | Fits 20 GB full-GPU? |
|-------------|-------|-----------|----------------------|
| `Qwen/Qwen3-30B-A3B-GGUF` → `Qwen3-30B-A3B-Q4_K_M.gguf` | Q4_K_M | **18.56** | ✅ At 4 K f16 (~19.6 GB) yes; 8 K f16 is ~20.3 GB → use `--cache-type-k/v q8_0` at 8 K (~20.0 GB, tight). |
| `unsloth/Qwen3-30B-A3B-GGUF` → `Qwen3-30B-A3B-Q4_K_S.gguf` | Q4_K_S | **17.46** | ✅ ~19.2 GB at 8 K f16; ~20.0 GB at 16 K f16. Good working quant. |
| `unsloth/Qwen3-30B-A3B-GGUF` → `Qwen3-30B-A3B-IQ4_XS.gguf` | IQ4_XS | **16.38** | ✅ ~18.9 GB at 16 K f16; ~18.9 GB at 32 K with Q8 KV. Best context headroom at ≈Q4 quality tier. |
| `unsloth/Qwen3-30B-A3B-GGUF` → `Qwen3-30B-A3B-UD-Q4_K_XL.gguf` | UD-Q4_K_XL (dynamic) | ~17.5 | ✅ Unsloth dynamic mix; alternative to Q4_K_S. Prefer plain Q4_K_S/IQ4_XS for kernel work — uniform quants keep dequant-kernel comparisons clean. |
| `Qwen/Qwen3-30B-A3B-GGUF` → `Qwen3-30B-A3B-Q5_K_M.gguf` | Q5_K_M | **21.73** | ❌ Does not fit full-GPU. Same conclusion as above for the matrix. |

### Recommendation logic (final pick PENDING-USER)

- **If raw quality per token matters most and short context is acceptable** → dense Qwen3-32B at **IQ4_XS** (17.69 GB) with Q8 KV beyond 4 K. Accept ~25–35 tok/s decode class (bandwidth-bound over ~800 GB/s reading ~17 GB/token — estimate, verify with llama-bench).
- **If iteration speed, longer contexts, and headroom for KV-cache experiments matter** (they do for Phases 8–9 of the roadmap) → **Qwen3-30B-A3B at Q4_K_S (17.46 GB) primary, IQ4_XS (16.38 GB) for long-context runs**. Decode is bounded by 3.3 B active params, so benchmarks also run several times faster — materially better for the benchmark-everything methodology.
- **Kernel-project angle:** dense makes the bottleneck analysis purer (every token streams all weights; matmul dominates cleanly). MoE adds routing/gather kernels but the expert FFN GEMMs still dominate runtime, so the custom fused dequant+matmul thesis holds for both. Downloading **both** families' working quants (~70 GB disk) is cheap and lets Milestone 2's matrix cover both architectures — recommended regardless of the final pick.

---

## WSL2 Verdict (what works today vs what needs native Linux)

### Works today under WSL2 (Adrenalin 26.2.2 + ROCm 7.2.1 + librocdxg)

| Capability | Status | Confidence |
|------------|--------|------------|
| HIP compute, hipcc/amdclang compilation, llama.cpp inference | ✅ Production-supported path | HIGH |
| `rocminfo` (shows gfx1100 agent), `hipconfig` | ✅ | HIGH |
| `rocprofv3` kernel/dispatch tracing + counter registration (rocprofiler-sdk over `/dev/dxg`, gfx11 family) | ✅ landed upstream (PR #7016); **validate in Phase 1 gate** | MEDIUM |
| `hipEvent`-based microbenchmark timers (kernel playground pipeline) | ✅ independent of WSL specifics | HIGH |
| llama.cpp built-in VRAM/perf logging | ✅ | HIGH |

### Broken or absent under WSL2 → drives contingency triggers

| Capability | Status | Workaround / Implication |
|------------|--------|--------------------------|
| `rocm-smi` | ❌ Not supported (UKI limitation, per AMD docs) — no GPU util/temp/active-process queries | Use llama.cpp startup VRAM breakdown + Windows Task Manager/Adrenalin overlay on the host |
| `radeontop` / anything needing `/dev/kfd` or `amdgpu` module | ❌ `/dev/kfd` doesn't exist in WSL2 | Same workaround |
| Radeon GPU Profiler (RGP) | ❌ Remains a Windows-host-side tool | Not required for rocprofv3-style dispatch profiling |
| Omniperf / ROCm Compute Profiler full hardware-counter sweeps | ⚠️ Unvalidated under WSL2 (rides the same rocprofiler-sdk counter path that PR #7016 enabled) | Treat as a Phase 1/3 experiment, not an assumption; if counters prove unusable → **native-Linux contingency trigger** |
| Free-VRAM parity | ⚠️ Community report: DXG path showed ~3 GB less free VRAM than native Windows on a 7900 XTX | On a 20 GB card treat ~1.5–3 GB as possibly reserved; measure actuals in Phase 1 before finalizing quant choice. Also explains why the fit table above prefers ≤17.5 GB weights |
| MIGraphX, mGPU | ❌ Explicitly unsupported on WSL | Irrelevant to the llama.cpp path anyway |

**Contingency rule (matches locked scoping):** stay on WSL2 as primary. If Phase 1 finds rocprofv3 cannot produce usable kernel timelines/counters for gfx1100, or free VRAM is crushed enough to invalidate the chosen quant, fall back to native Linux (dual-boot or Hyper-V GPU-P VM) **before any kernel work begins**. Everything above (build flags, artifacts, methodology) transfers unchanged.

---

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| ROCm 7.2.1 + librocdxg on WSL2 | Legacy roc4wsl / preview WSL drivers (≤ ROCm 7.2) | Never for new setup — superseded by ROCDXG; kept only for legacy reference by AMD |
| ROCm 7.2.1 + librocdxg on WSL2 | Native Linux dual-boot / Hyper-V GPU-P VM | Only if WSL2 profiling or VRAM budget fails the Phase 1 gate (contingency, per locked scope) |
| llama.cpp source build | Prebuilt llama.cpp HIP binaries / winget package | Never for this project — you need to patch ggml kernels and rebuild continuously |
| `GPU_TARGETS=gfx1100` single-target build | Multi-arch or `GGML_NATIVE`-style generic build | Multi-target only if you later publish binaries for other RDNA3 cards; always benchmark single-target |
| Stock FlashAttention path | `-DGGML_HIP_ROCWMMA_FATTN=ON` | As a permanent switch only after your own A/B numbers justify it per workload (known regressions off-7900) |
| rocBLAS comparison point | hipBLASLt | hipBLASLt *is* gfx110x-capable and PyTorch now prefers it there, but llama.cpp's ggml backend doesn't route through it; consider only as an extra Phase 6 "vendor GEMM" data point |
| Plain Q4_K_S / IQ4_XS artifacts | Unsloth UD dynamic quants (UD-Q4_K_XL) | UD mixes are great for end-user quality/size tradeoffs, but mixed quant formats muddy dequant-kernel microbenchmarks — use uniform quants for kernel work |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| **Native Windows HIP SDK llama.cpp build** | Explicitly out of scope (locked); Linux-only profiling tools; open compile-error reports against ROCm 7.x on Windows | WSL2 + ROCm 7.2.1 guest |
| `HSA_OVERRIDE_GFX_VERSION=11.0.0` env hack | Masking-layer meant for unsupported gfx11xx APUs (e.g. 780M gfx1103); gfx1100 is natively supported and overriding hides real errors | Nothing — remove it if present |
| Docker-in-WSL2 ROCm images | Stacks a container boundary over an already-paravirtualized `/dev/dxg` path; `/dev/kfd` device passthrough used by ROCm containers doesn't exist here; doubles the debugging surface | Direct guest-native install per librocdxg Quickstart |
| Q4_K_M (dense 32B) or any Q5_K_M as the on-card quant | 19.76 / 23.21 / 21.73 GB — none leave room for KV+buffers on 20 GB | Dense→IQ4_XS/Q4_K_S; MoE→Q4_K_S/IQ4_XS; Q3_K_M when max context needed |
| vLLM / SGLang / PyTorch serving stacks | Wrong abstraction level: you're replacing ggml HIP kernels, not configuring a serving engine; none target custom-HIP-kernel iteration | llama.cpp as reference runtime (locked decision) |
| `AMDGPU_TARGETS=` in new CMake code/docs | Legacy alias; silently forwarded, confusing in scripts | `GPU_TARGETS=gfx1100` |
| Custom-kernel-first development without stock baseline | Violates core project rule #1/#3; ROCm 7.x + WSL2 is new enough that stock behavior must be pinned first | Stock llama.cpp HIP build as the permanent baseline, custom work behind flags |

## Stack Patterns by Variant

**If Qwen3-32B (dense) is picked:**
- Primary artifacts: bartowski `IQ4_XS` (17.69 GB) + `Q4_K_S` (18.77 GB) for the quality tier; `Q3_K_M` for long-context runs.
- Plan for tight VRAM: Q8_0 KV cache (`--cache-type-k q8_0 --cache-type-v q8_0`) beyond 4–8 K context; expect VRAM-reduction wins (custom kernel → smaller scratch) to convert directly into context length.
- Benchmark iteration will be slower (~25–35 tok/s decode class) — budget wall-clock accordingly.

**If Qwen3-30B-A3B (MoE) is picked:**
- Primary artifacts: official `Q4_K_M` (18.56 GB) for fidelity anchor, unsloth `Q4_K_S` (17.46 GB)/`IQ4_XS` (16.38 GB) as working quants.
- Real headroom exists for Phase 8–9 attention/KV experiments at 16–32 K context.
- Custom-kernel targets skew toward expert-FFN GEMMs and MoE routing overhead; keep prefill(M≫1)/decode(M≈1) split analysis — MoE amplifies the difference.
- Faster decode → faster benchmark loops → more experiments per day.

**Either way:**
- Pin llama.cpp commit, ROCm 7.2.1, Adrenalin 26.2.2, WSL kernel version into `benchmarks/environment/` at Milestone 1.
- Run Phase 1 gates in order: (1) `rocminfo` sees gfx1100, (2) llama-bench runs full-GPU, (3) rocprofv3 produces a kernel timeline, (4) actual free VRAM measured → then finalize the quant/model pick.

## Version Compatibility

| Component | Compatible With | Notes |
|-----------|-----------------|-------|
| Adrenalin 26.2.2 (Win) | ROCm 7.2.1 guest + librocdxg current | Documented production pair; librocdxg decouples future updates |
| ROCm 7.2.1 | Ubuntu 22.04 / 24.04, WSL2 kernel 5.15 | Both distros "Yes" in AMD WSL matrix |
| llama.cpp master | ROCm 6.4+ / 7.x HIP | CMake-native HIP path (PR #5966 era onward); pin commit regardless |
| `GGML_HIP_ROCWMMA_FATTN=ON` | rocWMMA 2.x headers; RDNA3+ | Compile failures reported vs ROCm 6.4.4; perf regressions reported on gfx1151 — A/B only |
| rocBLAS / hipBLAS (bundled) | gfx1100 ✓ | Automatic |
| hipBLASLt | gfx110x ✓ (tensilelite) | Optional Phase 6 comparison only |
| `rocm-smi` | ✗ under WSL2 (any version) | UKI limitation; use host-side monitoring |

## Sources

- AMD ROCm on Radeon — WSL How-To / ROCDXG production announcement (Adrenalin 26.2.2 + ROCm 7.2.1): https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/install/installrad/wsl/howto_wsl.html — **HIGH**
- AMD ROCm WSL compatibility matrices (Ubuntu 22.04/24.04, WSL2 kernel 5.15): https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html — **HIGH**
- ROCm WSL limitations (rocm-smi unsupported, UKI): https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-6.4.2/docs/limitations.html — **HIGH**
- rocprofiler-sdk WSL2 gfx11-family compute+profiling over `/dev/dxg`: https://github.com/ROCm/rocm-systems/pull/7016 — **MEDIUM** (upstream PR; validate on hardware)
- rocprofv3 usage docs: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html — **HIGH**
- llama.cpp build docs (HIP flags, gfx1100 example): https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md and `ggml/src/ggml-hip/CMakeLists.txt` — **HIGH**
- llama.cpp rocWMMA FA flag + RDNA3 perf discussion (#15021), gfx1151 regression (#24437), 6.4.4 compile failure (#19580): GitHub ggml-org/llama.cpp — **MEDIUM**
- AMD llama.cpp-on-ROCm install guide: https://rocm.docs.amd.com/projects/llama-cpp/en/docs-26.02/install/llama-cpp-install.html — **HIGH**
- hipBLASLt hw requirements (gfx110x): https://github.com/rocm/hipblaslt ; rocWMMA supported archs (gfx1100/01/02 wave32): https://rocm.docs.amd.com/projects/rocWMMA/en/latest/api-reference/api-reference-guide.html — **HIGH**
- Exact GGUF sizes: HuggingFace API blobs for `Qwen/Qwen3-32B-GGUF`, `Qwen/Qwen3-30B-A3B-GGUF`, `unsloth/Qwen3-30B-A3B-GGUF`, `bartowski/Qwen_Qwen3-32B-GGUF` (queried 2026-08-21) — **HIGH**
- KV/token math: `config.json` of `Qwen/Qwen3-32B` (64L/8KV/128) and `Qwen/Qwen3-30B-A3B` (48L/4KV/128) — **HIGH**
- WSL2 DXG free-VRAM deficit report (~3 GB on 7900 XTX, ROCm 7.2.4-era): aiweekly.co alert — **LOW/MEDIUM** (single secondary source; re-measure in Phase 1)

---
*Stack research for: gfx1100-tuned Qwen3 inference optimization (RX 7900 XT / WSL2)*
*Researched: 2026-08-21*
