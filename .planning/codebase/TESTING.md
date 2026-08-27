<!-- refreshed: 2026-08-25 -->
# Testing & Verification Architecture

**Analysis Date:** 2026-08-25 (Updated Phase 6 / v1.0.0-gfx1100)

## 1. Seven-Level Testing Hierarchy

```
Level 0: Unit & Guard Regression Suite (55 tests in benchmarks/tests/)
Level 1: Platform & Device Pre-flight (rocminfo, hipconfig, versions.txt)
Level 2: Constraint & Integrity Verification (llabench argument construction, row signature checks)
Level 3: VRAM Overcommit & RSS Guard (preflight buffer estimation, 3-signal watchdog)
Level 4: Isolation Gate (scripts/check_no_ggml.sh — zero ggml/llama includes in kernels/)
Level 5: Standalone Kernel Oracles (ref_cpu.cpp vs impl*.hip — test_gemv_compare, test_gemm_compare)
Level 6: End-to-End Op Gate (QUAL-01: run_op_gate.py on test-backend-ops)
Level 7: Model-Level Quality Gate (QUAL-02: run_model_gate.py for WikiText-2 PPL and 6/6 canaries)
```

## 2. Test Suites & Commands

### A. Python Unit & Regression Suite (Level 0)
55 unit tests validate harness logic, guard thresholds, RunStore crash resilience, fingerprint generation, and metric calculations in pure CPU mode.

```bash
# Run complete unit test suite:
PYTHONPATH=. python3 -m pytest benchmarks/tests/ -q
```

### B. Header Isolation Gate (Level 4)
Enforces zero external llama.cpp / ggml headers in `kernels/`:

```bash
bash scripts/check_no_ggml.sh
```

### C. Kernel Playground Correctness Tests (Level 5)
Standalone binaries assert numerical correctness against CPU FP64 reference oracles:

```bash
export HSA_ENABLE_DXG_DETECTION=1

# Stock comparator test (16/16 PASS, cosine=1.0):
./kernels/build/matmul_iq4xs/matmul_test_baseline

# Custom GEMV decode test (10/10 PASS, cosine=1.0):
./kernels/build/matmul_iq4xs/test_gemv_compare

# Custom GEMM prefill test (11/11 PASS, cosine=1.0):
./kernels/build/matmul_iq4xs/test_gemm_compare
```

### D. Op-Level Correctness Gate (Level 6 / QUAL-01)
Runs `test-backend-ops` across 127 operation types and 21,093 tests, asserting 0 errors on ROCm:

```bash
python3 benchmarks/bin/run_op_gate.py --bin /root/llama.cpp/build-custom/bin/test-backend-ops
```

### E. Model-Level Quality Gate (Level 7 / QUAL-02)
Evaluates WikiText-2 perplexity within ±1.0% tolerance of golden baseline (6.4271) and asserts exact greedy token match on 6 canary prompts:

```bash
python3 benchmarks/bin/run_model_gate.py \
  --cli-bin /root/llama.cpp/build-custom/bin/llama-cli \
  --ppl-bin /root/llama.cpp/build-custom/bin/llama-perplexity
```

## 3. Quality Gate Thresholds & Criteria

| Gate | Target Metric | Required Threshold | Result |
|---|---|---|---|
| **Demo Dequant** | `max_abs`, `cosine` | `max_abs < 1e-5`, `cosine > 0.99999` | **PASS** (`max_abs=0`, `cosine=1.0`) |
| **Demo Red Test** | Broken mutant error | `>1000x` higher error (`max_abs > 1e3`) | **PASS** (exit code 1, discriminative) |
| **GEMV Decode** | `cosine`, `max_rel` | `cosine \ge 0.999`, `max_rel \le 1e-3` | **PASS** (`cosine=1.0`, `max_rel=0`) |
| **GEMM Prefill** | `cosine`, `max_rel` | `cosine \ge 0.999` vs FP64 oracle | **PASS** (`cosine=1.0` on all shapes) |
| **Op Gate (`QUAL-01`)** | Backend ops error count | **0 Errors** across all ops | **PASS** (4,243 supported, 0 errors) |
| **PPL Gate (`QUAL-02`)** | WikiText-2 PPL (145 chunks) | $6.4271 \pm 1.0\%$ $[6.3628, 6.4914]$ | **PASS** ($6.4271 \pm 0.00\%$) |
| **Canary Gate (`QUAL-02`)** | Greedy prompt hashes | 6 / 6 exact SHA256 match | **PASS** (6 / 6 match golden) |
