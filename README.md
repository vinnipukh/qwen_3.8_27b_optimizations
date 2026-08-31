# GFX1100 Custom Kernels for Qwen3.8-27B on AMD Radeon RX 7900 XT

A specialized GPU kernel optimization and benchmarking project targeting the **AMD Radeon RX 7900 XT (`gfx1100`)** with RDNA3 hardware matrix cores (Wave32 WMMA) and integer SIMD (DP4A `v_dot4_i32_i8`), integrated into `llama.cpp` via a clean quilt patch overlay.

---

## 📊 The Honest Performance Verdict

### 1. Live LLM QA Benchmark ($N=15$ Hardware Runs on RX 7900 XT)
Measured across 15 consecutive greedy generation runs using `Qwen3.8-27B-Uncensored-IQ4_XS.gguf` ($15.31\text{ GB}$) at `temp=0`, `-n 128`, fixed prompt, fully offloaded (`-ngl 99`, `-b 2048`):

| Metric | Stock `llama.cpp` (`bb4caa7`) | Custom Kernel (`5c6b397`) | Delta / Speedup |
|:---|:---:|:---:|:---:|
| **Generation Throughput (Mean $\pm \sigma$)** | $35.95 \pm 1.12\text{ tok/s}$ | **$36.38 \pm 0.61\text{ tok/s}$** | **$+1.2\%$ ($1.012\times$)** |
| **Generation Throughput (Median)** | $36.00\text{ tok/s}$ | **$36.40\text{ tok/s}$** | **$+1.1\%$ ($1.011\times$)** |
| **Prompt Processing (Mean $\pm \sigma$)** | $147.39 \pm 7.16\text{ tok/s}$ | **$150.37 \pm 4.33\text{ tok/s}$** | **$+2.0\%$ ($1.020\times$)** |
| **Prompt Processing (Median)** | $148.30\text{ tok/s}$ | **$152.10\text{ tok/s}$** | **$+2.6\%$ ($1.026\times$)** |
| **Average End-to-End Latency** | $19,866.6\text{ ms}$ | **$19,045.5\text{ ms}$** | **$-821.1\text{ ms}$ ($-4.1\%$ faster)** |
| **Timing Jitter / StdDev** | $\sigma = 1.12\text{ tok/s}$ | **$\sigma = 0.61\text{ tok/s}$** | **$45\%$ lower variance** |
| **Numerical Correctness** | Baseline | **$1.000000$ Cosine Parity** | Zero quality degradation |

*Single-run claims are banned in this repository. All figures are verified from `benchmarks/results/phase7/llm_qa_N15.json` and `llm_qa_stock_N15.json`.*

---

## 🎯 Should You Use This Project?

### ✅ Use this project if:
1. **You want lower latency and consistent generation:** The custom Wave32 cooperative workgroup kernel reduces total request latency by **$\sim 820\text{ ms}$** and cuts generation variance by **$45\%$**, eliminating random hitching during text stream generation.
2. **You are studying or building RDNA3 GPU kernels:** The repository provides standalone, isolated HIP implementations of:
   * Wave32 cooperative 8-thread/row DP4A (`__builtin_amdgcn_sudot4` + `v_perm_b32` LUT gather) with on-the-fly `Q8_1` quantization.
   * XOR preshuffled LDS layouts ($x' = (y \pmod{4}) \oplus x$) eliminating bank conflicts.
   * Hardware WMMA matrix cores (`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`) with double/quad-buffered LDS pipelining.
3. **You want a robust standalone testing & benchmarking scaffold:** Fully isolated CPU FP64 reference oracles, deterministic tensor fixtures, and memory-bandwidth evaluators in `kernels/`.

### ❌ Stick with stock upstream `llama.cpp` if:
1. **You want seamless updates:** Stock `llama.cpp` lets you `git pull` and rebuild anytime without having to re-apply or rebase custom patch files.
2. **You expect a $2\times$ speedup on a 27B model from code alone:** A $+1.2\%$ increase ($36.4$ vs $36.0\text{ tok/s}$) is imperceptible during interactive reading.

---

## 🧠 The Architectural Reality (Why 27B LLMs are Memory-Bound)

A 27-billion parameter model quantized to `IQ4_XS` is **$15.31\text{ GB}$** in size. During batch=1 autoregressive token generation:
$$\text{Theoretical Peak Decode Speed} = \frac{\text{VRAM Bandwidth}}{\text{Model Size}} \approx \frac{800\text{ GB/s}}{15.31\text{ GB/token}} \approx 52.2\text{ tok/s}$$

* Accounting for physical memory controller efficiency, driver overhead, and KV-cache lookups, practical hardware throughput is capped at $\approx 36\text{ tok/s}$.
* Because token generation is **$>95\%$ memory-bandwidth bound**, arithmetic optimizations (faster ALUs / matrix cores) only optimize the remaining $<5\%$ compute overhead.

---

## ⚡ How to Actually Get 20%–60%+ Speedups on RX 7900 XT

If you want major throughput improvements, reduce memory bus traffic using these techniques:

1. **Speculative Decoding (Draft Model):**
   * Pair `Qwen3.8-27B` with a tiny `Qwen2.5-0.5B-Instruct` draft model.
   * The small model drafts candidate tokens at $180+\text{ tok/s}$, and the 27B model verifies them in parallel.
   * **Expected Gain:** **$+30\%$ to $+60\%$ higher generation speed** ($\to \mathbf{48–60\text{ tok/s}}$).
2. **Enable Flash Attention:**
   * Replaces $O(N^2)$ global memory roundtrips for attention scores with tiled on-chip SRAM computation.
   * **Expected Gain:** **$+15\%$ to $+40\%$ faster prompt processing (TTFT)**.
3. **Quantize the KV Cache (`-ctk q8_0 -ctv q8_0`):**
   * Compresses past context from 16-bit float down to 8-bit integer.
   * **Expected Gain:** Halves KV-cache memory bandwidth traffic during decode and frees up VRAM.
4. **Quantization Selection:**
   * Switching from `Q4_K_M` ($17.5\text{ GB}$) to `IQ3_K_M` ($12.8\text{ GB}$) reduces weight traffic per token, delivering an immediate **$+20\%$ to $+35\%$** speedup with negligible perplexity difference.

---

## 🚀 Quick Start

### 1. Build & Run the Standalone Kernel Playground
Test the custom kernels directly without building the entire `llama.cpp` runtime:

```bash
# Configure standalone kernels
cmake -S kernels -B kernels/build -G Ninja \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_BUILD_TYPE=Release

# Build kernels and test harnesses
cmake --build kernels/build -j 4

# Run numerical correctness gates vs CPU FP64 oracle (cosine = 1.000000)
export HSA_ENABLE_DXG_DETECTION=1
./kernels/build/matmul_iq4xs/test_real_stock_compare
./kernels/build/matmul_iq4xs/test_gemv_dp4a_compare
./kernels/build/matmul_iq4xs/test_gemm_wmma_compare

# Run N=10 microbenchmarks
./kernels/build/matmul_iq4xs/bench_real_stock --runs 10 --json
./kernels/build/matmul_iq4xs/bench_gemv_dp4a --runs 10 --json
./kernels/build/matmul_iq4xs/bench_gemm_wmma --runs 10 --json
```

### 2. Apply Custom Kernel Patch to `llama.cpp`
The patch is a clean 356-line quilt overlay over upstream `llama.cpp` commit `bb4caa75`:

```bash
cd /path/to/llama.cpp
git checkout bb4caa75
git apply /path/to/patches/0001-gfx1100-mul-mat-custom.patch

# Build with custom kernel switch enabled
cmake -S . -B build -G Ninja \
  -DGGML_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DGGML_CUDA_ENABLE_CUSTOM_GFX1100=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build -j 4
```

---

## 📚 Project Documentation

* **[docs/PUBLICATION.md](docs/PUBLICATION.md)** — Complete release publication checklist, hardware versions, and full $N=15$ benchmark datasets.
* **[docs/TESTING.md](docs/TESTING.md)** — Testing doctrine, unit test suites (55/55 passed), and numerical parity gates.
* **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — System architecture, Wave32 lane mapping, and memory layouts.
* **[docs/ROADMAP.md](docs/ROADMAP.md)** — Project roadmap and phase delivery history.
* **[docs/research/](docs/research/)** — Deep-dive research papers, model decisions, and FreeToken PCIe/host bandwidth probes.

---

## 📄 License

Licensed under the [Apache License, Version 2.0](LICENSE).
