# Summary 03-04: 4-Shape Workload Profiling & Bottleneck Attribution Table (PROF-02)

**Phase:** 3-Correctness Gates & Bottleneck Profiling  
**Plan:** 03-04  
**Requirement:** PROF-02  
**Status:** COMPLETE  

---

## What Was Accomplished
1. **4-Shape Inference Profiling Sweep (`benchmarks/bin/profile_matrix.py`):**
   - Profiled canonical inference shapes S1 (128/128), S2 (128/256), S3 (4096/128), and S4 (4096/256) on RX 7900 XT.
   - Saved raw profiling logs to `benchmarks/profiling/raw/*.json`.
2. **Ranked Bottleneck Table (`benchmarks/profiling/BOTTLENECK-TABLE.md`):**
   - Published full attribution breakdown mapping top operations by % runtime, invocation count, latency, and bound type.
3. **Formal Designation of Target #1 (`benchmarks/profiling/bottleneck_summary.json`):**
   - Formally designated `MUL_MAT` (quantized IQ4_XS matrix-vector/matrix-matrix multiplication) as **Optimization Target #1**.
   - `MUL_MAT` represents **31.12%** of cumulative GPU wall time (50.89% of prefill, 30.04% of decode).
   - Followed by elementwise gate modulation `MUL` (13.59%), `RMS_NORM` (9.76%), `ADD` (8.45%), and SSM ops (`GATED_DELTA_NET`, `SSM_CONV`, `SOLVE_TRI`).
4. **Unit Test Suite (`benchmarks/tests/test_bottleneck_profiling.py`):**
   - Verified profile parser and existence/consistency of published bottleneck artifacts.
