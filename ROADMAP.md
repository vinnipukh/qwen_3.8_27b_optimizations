# Qwen3.8-27B on RX 7900 XT — ROCm / HIP Optimization Roadmap

> Goal: build and benchmark a **GPU-specific inference stack for the RX 7900 XT (RDNA3 / gfx1100)** around Qwen3.8-27B, starting from existing llama.cpp/ROCm support and gradually replacing hot paths with custom HIP kernels.

## 0. Scope and success criteria

### Primary goal

Create a reproducible project that answers:

> “How much faster and/or more memory-efficient can Qwen3.8-27B inference become on a 20 GB RX 7900 XT when the kernels, memory layout, KV cache, and runtime are tuned specifically for gfx1100?”

The first target should **not** be a complete inference engine from scratch. Build on llama.cpp as the reference runtime and replace one bottleneck at a time.

### Hardware target

- GPU: AMD Radeon RX 7900 XT
- Architecture: RDNA3
- LLVM target: `gfx1100`
- VRAM: 20 GB
- Backend: ROCm / HIP
- Reference runtime: llama.cpp

AMD's current ROCm documentation lists the RX 7900 XT as RDNA3 / `gfx1100` with Runtime and HIP SDK support. llama.cpp's HIP build supports selecting a specific AMD GPU target and has an explicit `gfx1100` example for the RX 7900 XTX/XT/GRE family.

### Initial success criteria

Do not define success as “I wrote a kernel.”

Use measurable targets:

- No regression in model correctness beyond an agreed numerical tolerance.
- Establish a reproducible llama.cpp baseline.
- Achieve a measurable speedup in at least one important workload.
- Or reduce VRAM use enough to enable a larger context / better KV cache.
- Publish benchmark results against the baseline.
- Keep every optimization switchable so regressions can be bisected.

---

# Phase 1 — Environment and hardware baseline

## 1.1 Verify the exact GPU target

Record:

```bash
rocminfo | grep -i gfx | head
```

Expected target:

```text
gfx1100
```

Also record:

```bash
rocminfo
hipconfig --full
rocm-smi
```

Save the output under:

```text
benchmarks/environment/
```

## 1.2 Build a reference llama.cpp

Use the HIP backend and explicitly target `gfx1100`.

Example:

```bash
cmake -S . -B build \
  -G Ninja \
  -DGGML_HIP=ON \
  -DGPU_TARGETS=gfx1100 \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build
```

Do not optimize anything yet.

## 1.3 Establish a baseline model

Use at least:

```text
Qwen3.8-27B Q4_K_M
```

Keep the exact GGUF filename, source repo, commit/date, and quant metadata in:

```text
models/README.md
```

Also test:

```text
Q4_K_S
Q5_K_M
```

if VRAM permits.

## 1.4 Baseline metrics

Measure separately:

### Prompt processing

```text
prompt tokens / second
```

### Generation

```text
generated tokens / second
```

### Memory

```text
peak VRAM
model VRAM
KV cache VRAM
runtime/buffer VRAM
```

### Quality

Use a fixed evaluation set.

At minimum:

- perplexity on a fixed text set
- deterministic coding prompts
- reasoning prompts
- Turkish prompts if Turkish use is important

Store every run as CSV/JSON.

Suggested layout:

```text
benchmarks/
├── baseline/
│   ├── q4_k_s.json
│   ├── q4_k_m.json
│   └── q5_k_m.json
└── environment/
    ├── rocm.txt
    ├── rocminfo.txt
    └── rocm-smi.txt
```

---

# Phase 2 — Understand the existing GPU path

Before writing custom code, understand what llama.cpp is already doing.

Study:

```text
ggml/src/ggml-hip/
ggml/src/ggml-cuda/
ggml/src/ggml-common.h
ggml/src/ggml-quants.c
```

Pay special attention to:

- quantized matrix multiplication
- dequantization
- attention
- Flash Attention
- memory allocation
- tensor scheduling
- GPU/CPU synchronization
- graph execution

The current llama.cpp HIP backend reuses many CUDA kernel sources through HIP and links against HIP, hipBLAS and rocBLAS. That makes it useful as both the reference implementation and a place to test targeted changes.

---

# Phase 3 — Profiling before optimization

## 3.1 Find the real bottleneck

Do not assume attention is the bottleneck.

Profile real Qwen3.8-27B workloads.

Use:

- ROCm profiling tools
- rocprof / rocprofv2 where available
- Radeon GPU Profiler where practical
- CPU-side timers already exposed by llama.cpp
- VRAM monitoring via ROCm tooling

Create profiles for:

```text
short prompt + short generation
long prompt + short generation
short prompt + long generation
long context generation
```

## 3.2 Build a bottleneck table

Example:

| Kernel / operation | % runtime | Memory bound? | Compute bound? | Candidate? |
|---|---:|---|---|---|
| Q4 matmul | 38% | yes/no | yes/no | high |
| attention | 21% | yes | maybe | high |
| KV cache | 12% | yes | no | high |
| sampling | 1% | no | no | low |

Do not optimize low-impact operations.

---

# Phase 4 — HIP kernel playground

Create a standalone benchmark directory:

```text
kernels/
├── common/
├── dequant/
├── matmul/
├── attention/
└── benchmarks/
```

Each kernel should first have a CPU reference.

For example:

```text
CPU reference
     ↓
HIP implementation
     ↓
numerical comparison
     ↓
microbenchmark
     ↓
integration test
```

This makes debugging much easier than starting inside the full LLM runtime.

---

# Phase 5 — First custom kernel: Q4 dequantization

This should be the first serious optimization.

## Goal

Take Q4_K weights and produce the values needed by the matmul without unnecessary global-memory traffic.

Instead of:

```text
VRAM
  ↓
dequant to large FP16 buffer
  ↓
VRAM again
  ↓
matmul
```

aim toward:

```text
VRAM Q4
   ↓
fused dequant
   ↓
matmul
```

## Experiments

Test:

1. standalone dequant
2. dequant + matmul
3. different tile sizes
4. LDS/shared-memory staging
5. vectorized loads
6. register usage
7. wave-level reductions where useful

Benchmark against the existing llama.cpp kernel.

---

# Phase 6 — Quantized GEMM / matrix multiplication

This is likely the highest-value stage.

Implement a small matrix-multiply kernel specialized for:

```text
Q4_K × FP16/BF16
Q5_K × FP16/BF16
```

Do not immediately support every quant format.

Start with the exact format used by your target Qwen GGUF.

## Questions to answer experimentally

- Which tile shape works best on gfx1100?
- Is LDS helping?
- Are wave-level operations useful?
- Where does register pressure become harmful?
- What is the best occupancy point?
- Is rocBLAS better for this shape?
- When does a custom kernel beat rocBLAS?
- Does batch size change the winner?

Create a shape sweep.

Example:

```text
M × N × K

1 × hidden × hidden
8 × hidden × hidden
16 × hidden × hidden
32 × hidden × hidden
```

This matters because LLM decoding often operates at very small M while prompt processing has much larger M.

---

# Phase 7 — Decode-specific optimization

LLM inference has two very different modes.

## Prompt processing

High token count:

```text
M is relatively large
```

## Generation

Usually:

```text
M ≈ 1
```

A kernel optimized for prompt processing can be bad for single-token decode.

Therefore create two paths:

```text
prefill kernel
decode kernel
```

This is an important design goal.

For your 7900 XT, optimize decode separately rather than chasing one “universal” GEMM.

---

# Phase 8 — Attention kernel experiments

Once GEMM is under control, work on attention.

Start with correctness, then speed.

Implement experiments around:

```text
QKᵀ
softmax
PV
```

Eventually aim for a fused attention path that minimizes trips to global memory.

Measure:

- 2K context
- 8K
- 16K
- 32K
- higher if VRAM allows

Do not claim success from a tiny context benchmark.

---

# Phase 9 — KV cache optimization

This is especially important for a 20 GB GPU.

Track VRAM as:

```text
weights
+
KV cache
+
temporary buffers
+
runtime
```

## Experiments

Compare:

```text
FP16 KV
Q8 KV
Q6 KV
Q5 KV
Q4 KV
```

where supported.

Test different KV layouts and allocation strategies.

Measure:

```text
VRAM/token
tokens/sec
quality loss
maximum context
```

## Goal

Find a sweet spot such as:

```text
Q4 model
+
lower-precision KV cache
+
optimized attention
```

that lets the 20 GB card run larger contexts without a major quality or speed penalty.

---

# Phase 10 — Qwen-specific optimization

Only after generic kernels are working should you specialize for Qwen3.8-27B.

Map the model graph.

Record:

```text
embedding
attention
MLP
norm
output
```

For every major tensor:

```text
shape
dtype
quant format
frequency
bytes moved
kernel used
```

Build a report:

```text
tensor_name
shape
dtype
quant
runtime
VRAM
kernel
```

Then optimize the operations that dominate actual Qwen runtime.

---

# Phase 11 — Quantization and imatrix experiments

This is where your earlier quantization idea comes back.

Build:

```text
Q4_K_S
Q4_K_M
Q5_K_M
```

with and without an importance matrix.

llama.cpp's `llama-imatrix` supports creating an importance matrix from a calibration text set, and that matrix can be supplied to `llama-quantize` during quantization.

Example:

```bash
./llama-imatrix \
  -m model-f16.gguf \
  -f calibration-data.txt \
  -o imatrix.gguf \
  -ngl 99
```

Then:

```bash
./llama-quantize \
  --imatrix imatrix.gguf \
  model-f16.gguf \
  qwen-q4_k_m-imatrix.gguf \
  q4_k_m
```

## Build domain-specific calibration data

Create separate datasets:

```text
calibration/
├── general.txt
├── coding.txt
├── reasoning.txt
├── turkish.txt
└── mixed.txt
```

Then compare.

This lets you test whether a calibration set tuned to your actual workload improves the final quant.

---

# Phase 12 — Runtime integration

At this point you should have independent kernels.

Now integrate only the winners into llama.cpp.

Recommended order:

```text
baseline
  ↓
custom matmul
  ↓
custom fused dequant/matmul
  ↓
custom attention
  ↓
KV optimization
  ↓
runtime scheduling improvements
```

Every change should have:

```text
ENABLE_CUSTOM_KERNEL=ON/OFF
```

or an equivalent runtime switch.

Never destroy the baseline.

---

# Phase 13 — Autotuning

Once the kernels work, build a tiny autotuner.

Inputs:

```text
M
N
K
batch size
context length
quant type
```

Candidate parameters:

```text
tile M/N/K
workgroup size
LDS usage
vector width
pipeline stages
```

Run a short benchmark and select the fastest configuration.

Store results:

```text
autotune/
├── gfx1100.json
└── q4_k_m.json
```

Eventually the runtime can do:

```text
shape
 ↓
lookup tuned configuration
 ↓
launch optimized kernel
```

This is where the project starts looking like a real GPU-specific library rather than a collection of hand-tuned kernels.

---

# Phase 14 — Numerical validation

Every kernel must be tested against a reference.

For each operation compare:

```text
max absolute error
mean absolute error
relative error
cosine similarity
```

For full-model tests compare:

```text
same prompt
same seed
same sampling
```

and verify that differences are acceptable.

Do not optimize first and validate later.

---

# Phase 15 — End-to-end benchmark suite

Create fixed benchmark profiles.

## Profile A — chat

```text
4K context
512 generated tokens
```

## Profile B — coding

```text
16K context
1K generated tokens
```

## Profile C — long context

```text
32K context
1K generated tokens
```

## Profile D — long generation

```text
4K context
4K generated tokens
```

For every benchmark report:

```text
model
quant
ROCm version
driver
GPU
kernel version
context
prompt tokens
generated tokens
prompt tok/s
generation tok/s
peak VRAM
temperature
power
```

---

# Phase 16 — Compare against real baselines

At minimum compare:

```text
llama.cpp stock HIP
your optimized build
```

Also compare multiple quantizations:

```text
Q4_K_S
Q4_K_M
Q5_K_M
```

And multiple runtime configurations:

```text
stock KV
optimized KV
```

Do not publish only the best number.

Publish the complete matrix.

---

# Phase 17 — Package the result

Suggested repository:

```text
qwen3.8-27b-7900xt/
│
├── README.md
├── ROADMAP.md
├── LICENSE
│
├── kernels/
│   ├── dequant/
│   ├── matmul/
│   ├── attention/
│   └── kv/
│
├── runtime/
│
├── quant/
│   ├── calibration/
│   └── imatrix/
│
├── benchmarks/
│   ├── baseline/
│   ├── optimized/
│   └── plots/
│
├── scripts/
│
└── docs/
    ├── architecture.md
    ├── benchmark.md
    └── kernel-notes.md
```

Publish:

1. exact build commands
2. ROCm version
3. GPU target
4. model/quant files
5. benchmark methodology
6. raw benchmark data
7. kernel source
8. known limitations

---

# Phase 18 — Stretch goals

These should come only after the core project works.

## A. Custom sampler

Move sampling work to GPU and reduce CPU synchronization.

## B. Speculative decoding

Explore a smaller model as a draft model.

Example:

```text
small draft model
       ↓
Qwen3.8-27B verifier
```

Measure whether the additional complexity actually improves tokens/sec.

## C. Multi-GPU support

Possible later extension:

```text
7900 XT + second AMD GPU
```

Do not start here.

## D. Custom GGUF format / metadata

Only consider this if existing GGUF layouts block a useful optimization.

## E. Persistent kernels / advanced scheduling

Investigate only after ordinary kernel tuning is exhausted.

---

# Recommended execution order

Do **not** attempt all of this at once.

Use this sequence:

```text
1. ROCm setup
   ↓
2. llama.cpp gfx1100 baseline
   ↓
3. benchmark suite
   ↓
4. profiler
   ↓
5. identify top 1 bottleneck
   ↓
6. standalone HIP kernel
   ↓
7. numerical validation
   ↓
8. microbenchmark
   ↓
9. llama.cpp integration
   ↓
10. end-to-end benchmark
   ↓
11. repeat for next bottleneck
```

The first serious milestone should be:

> **Beat stock llama.cpp HIP on one Qwen3.8-27B workload with a custom gfx1100 kernel while preserving model output quality.**

Do not start by writing a new inference engine.

---

# Suggested first 7 milestones

## Milestone 1
Get stock llama.cpp running on `gfx1100`.

## Milestone 2
Create reproducible Q4_K_M/Q5_K_M benchmark scripts.

## Milestone 3
Profile Qwen3.8-27B and identify the top three kernels by wall time.

## Milestone 4
Implement one standalone Q4_K_M HIP kernel and verify numerical correctness.

## Milestone 5
Beat the relevant stock kernel in a microbenchmark.

## Milestone 6
Integrate it behind a runtime flag and run the full model.

## Milestone 7
Publish before/after numbers.

Only after Milestone 7 move to attention/KV/autotuning.

---

# Rules for the project

1. **Benchmark before optimizing.**
2. **One optimization at a time.**
3. **Keep a stock baseline forever.**
4. **Test prefill and decode separately.**
5. **Measure VRAM as carefully as throughput.**
6. **Do not assume CUDA optimization ideas map directly to RDNA3.**
7. **Prefer fused memory-efficient kernels when they actually win.**
8. **Keep correctness tests next to every kernel.**
9. **Record compiler/ROCm/driver versions.**
10. **Publish failed experiments too.**

---

# Final target

The ideal end state is not merely:

> “Qwen3.8-27B runs on my RX 7900 XT.”

It is:

> **“Qwen3.8-27B has an independently benchmarked gfx1100-tuned inference path, with custom HIP kernels, optimized KV-cache behavior, reproducible quantization, and published performance/VRAM results against stock llama.cpp.”**

That is the point where the project becomes a genuine AMD GPU optimization project rather than a custom quantization experiment.
