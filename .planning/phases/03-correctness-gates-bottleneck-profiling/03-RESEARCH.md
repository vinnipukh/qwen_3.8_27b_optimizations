# Phase 3: Correctness Gates & Bottleneck Profiling - Research

**Date:** 2026-08-24
**Scope:** QUAL-01, QUAL-02, PROF-01, PROF-02

---

## 1. Op-Level Correctness Gate (`test-backend-ops`)

### Capabilities & CLI Options
- Binary: `/root/llama.cpp/build-ci/bin/test-backend-ops` (archived at `baseline/binaries/v0.2.0-bb4caa75/test-backend-ops`).
- Target backend: `-b ROCm0` (selects RX 7900 XT gfx1100).
- Modes:
  - `test` (compares GPU math against 64-bit precision CPU reference).
  - `perf` (measures microsecond latency and memory bandwidth GB/s across tensor shapes).
  - `support` (probes op support matrix).
- Coverage: 128 mathematical operations available in GGML.
- Error formula: Normalized Mean Squared Error (NMSE) $\le 10^{-7}$ against CPU IEEE-754 FP64 reference.

### Key Op Checks for Qwen 3.8 Hybrid Architecture:
1. `GATED_DELTA_NET` (linear attention recurrent state scan)
2. `SOLVE_TRI` (triangular solver in DeltaNet)
3. `SSM_CONV` / `SSM_SCAN` (state-space conv/scan)
4. `FLASH_ATTN_EXT` (full self-attention for the 16 full-attention layers)
5. `MUL_MAT` (quantized matrix multiplication for IQ4_XS weights)
6. `RMS_NORM`, `ROPE`, `SWIGLU`, `SILU`

---

## 2. Model-Level Quality Gate (WikiText-2 + Golden Canary)

### Dataset Acquisition
- Canonical dataset: `https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip`
- Target file: `wiki.test.raw` (~250 KB uncompressed) placed in `benchmarks/data/wiki.test.raw`.
- SHA256 verification recorded at download.

### Perplexity Metric (`llama-perplexity`)
- Invocation: `llama-perplexity -m models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf -f benchmarks/data/wiki.test.raw -c 2048 -ngl 99 --no-mmap`
- Published baseline reference for IQ4_XS: `7.1583`
- Tolerance threshold: $\pm 1\%$ (acceptable interval: `7.087` – `7.230`).

### Golden Canary & Logits
- Built-in `llama-perplexity` features:
  - `--save-all-logits <file>`: dumps raw logits on prompt completion boundaries.
  - `--kl-divergence --kl-divergence-base <file>`: measures exact KL-divergence between baseline logits and candidate logits.
- Prompt Corpus: 6 deterministic prompt files in `benchmarks/prompts/` evaluated for:
  - Exact token match on first 32 tokens of structured prompts (JSON / Code).
  - Logit cosine similarity $> 0.999$ on completion boundaries.

---

## 3. WSL2 Profiling Ladder (PROF-01) & Upstream Practices

### Rung (a) Hardware Profiler Probe:
- Probed `rocprofv3 --hip-trace` under stock ROCm 7.2.1 / WSL2.
- Outcome: Aborts with `sysfs nodes path '/sys/class/kfd/kfd/topology/nodes' does not exist` (unsupported under `/dev/dxg` paravirtualization).
- Verdict: Expected; proceed to Rung (b).

### Rung (b) High-Precision Graph Evaluation & Op Timers:
- Upstream mechanism: `ggml_backend_sched_set_eval_callback` (demonstrated in `examples/eval-callback`).
- `llama-eval-callback` intercepts every node in the 64-layer compute graph on ROCm0 with microsecond accuracy.
- Zero virtualization friction, zero host/guest impedance mismatch.

### Dispatch Overhead & Graph Capture Audit:
- Upstream issue #20292 showed small DeltaNet dispatches can be 99% CPU-dispatch-bound without graph capture.
- We will test `GGML_HIP_GRAPHS=ON` vs `OFF` to isolate launch latency from kernel execution duration.

---

## 4. The 4 Canonical Workload Shapes (PROF-02)

| Shape ID | Name | Prompt ($p$) | Generation ($n$) | Primary Profile Dimension |
|---|---|---|---|---|
| **S1** | Short/Short | 128 | 128 | Interactive turn / small-matrix latency |
| **S2** | Short/Long | 128 | 1024 | Sustained decode & recurrent scan throughput |
| **S3** | Long/Short | 4096 | 128 | Heavy prefill GEMM & prompt ingestion |
| **S4** | Long/Long | 4096 | 1024 | Agentic multi-turn (prefill + decode) |

---

## 5. Output Deliverables & Schema

- `benchmarks/bin/run_op_gate.py` — Op-level gate CLI.
- `benchmarks/bin/run_model_gate.py` — Model quality gate CLI.
- `benchmarks/bin/profile_workload.py` — 4-shape profiler CLI.
- `benchmarks/golden/stock_baseline_golden.json` — Baseline logits & token outputs.
- `benchmarks/profiling/BOTTLENECK-TABLE.md` — Final published ranking naming Target #1.
