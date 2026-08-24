# Summary 03-03: Profiler Bridge & Graph Dispatch Probe (PROF-01)

**Phase:** 3-Correctness Gates & Bottleneck Profiling  
**Plan:** 03-03  
**Requirement:** PROF-01  
**Status:** COMPLETE  

---

## What Was Accomplished
1. **High-Precision Evaluation Profiler (`benchmarks/tools/eval_profiler.cpp` & `benchmarks/bin/eval_profiler`):**
   - Built standalone C++ evaluation callback profiler instrumenting `ggml_backend_sched_set_eval_callback`.
   - Microsecond-level timestamping without PCIe buffer readback overhead.
   - Separate accumulation for Prefill ($M \gg 1$) and Decode ($M = 1$) phases across all 64 model layers.
2. **Profile Parser & CLI Bridge (`benchmarks/lib/parse_profile.py`, `benchmarks/bin/profile_workload.py`):**
   - Implemented automated parser classifying operations into compute, memory bandwidth, or latency bound types.
3. **Dispatch Overhead Audit (`benchmarks/profiling/dispatch_overhead_report.md`):**
   - Evaluated `GGML_CUDA_GRAPHS=ON` vs `GGML_CUDA_DISABLE_GRAPHS=1`.
   - Discovered that HIP graph capture accelerates decode by **+19%** (70.5 t/s vs 59.3 t/s) by amortizing 520 kernel launches per step across WSL2 DXG boundaries.
