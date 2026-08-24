# Summary 03-01: Op-Level Correctness Gate (QUAL-01)

**Phase:** 3-Correctness Gates & Bottleneck Profiling  
**Plan:** 03-01  
**Requirement:** QUAL-01  
**Status:** COMPLETE  

---

## What Was Accomplished
1. **Automated Op Gate CLI (`benchmarks/bin/run_op_gate.py`):**
   - Implemented automated wrapper executing `test-backend-ops test -b ROCm0 --output csv`.
   - Tested 21,093 cases across 127 unique GGML ops with 0 errors on stock gfx1100 ROCm.
   - Enforced strict PASS assertions on core hybrid architecture operations: `GATED_DELTA_NET` (36 cases), `SOLVE_TRI` (24 cases), `SSM_CONV` (45 cases), `SSM_SCAN` (9 cases), `FLASH_ATTN_EXT` (2,936 supported cases), and `MUL_MAT` (1,193 supported cases).
2. **Structured Results Artifact (`benchmarks/results/phase3/op_gate.json`):**
   - Emits machine-readable verdict with full op-level stats, error traces, and timestamped exit code verification.
3. **Unit Test Suite (`benchmarks/tests/test_op_gate.py`):**
   - 5 unit tests validating CSV parsing, synthetic failure interception, missing core op blocking, and live gate integrity.
