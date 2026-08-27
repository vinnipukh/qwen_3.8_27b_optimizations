<!-- refreshed: 2026-08-25 -->
# Known Concerns, Constraints & Risk Mitigations

**Analysis Date:** 2026-08-25 (Updated Phase 6 / v1.0.0-gfx1100)

## 1. Resolved Technical Concerns (Phase 6 Fixes)

| Issue | Root Cause | Resolution in Phase 6 | Status |
|---|---|---|---|
| **Barrier Divergence (M-1)** | Early `return` in GEMV/WMMA workgroups before `__syncthreads()` caused divergence on non-full blocks. | Replaced early exits with thread masking (`if (row < N)`) so all 256 threads participate in barriers uniformly. | **RESOLVED** |
| **Misaligned 16B Loads (M-2)** | `reinterpret_cast<const uint4*>` on `blk->qs + ib*16` violated 16-byte alignment (`alignof(uint4)==16`). | Replaced with 8-byte aligned `uint64_t[2]` pairs (guaranteed 8-byte aligned at offset `8 + ib*16`). | **RESOLVED** |
| **WMMA LDS Bounds (H-1)** | Lanes $\ge 16$ in WMMA workgroup indexed LDS rows $16..31$, which were uninitialized. | Added bounds clamping (`elem_mod / 16`) and full LDS zero-initialization; verified on `wmma_gate_pass_5120_1024_512` (`cosine=1.0`). | **RESOLVED** |
| **Unguarded `can_handle` (H-4)** | Provisional patch called `custom_*_can_handle` outside `#if` block, breaking `OFF` compilation. | Moved all dispatch and handle logic inside `#if defined(GGML_CUDA_ENABLE_CUSTOM_GFX1100)`. | **RESOLVED** |
| **Fixture Seed Non-Determinism** | `hash(name)` in `dump_matmul_fixtures.py` used salted Python hash. | Replaced with stable SHA256-derived integer seeds and NumPy `SeedSequence`. | **RESOLVED** |
| **CMake Target Redundancy (F6)** | `kernels/matmul_iq4xs/CMakeLists.txt` contained repeated include/link declarations. | Collapsed dependencies into `matmul_common_iface` INTERFACE target. | **RESOLVED** |

## 2. Documented Performance Trade-Offs (Rule #10)

- **Small Batch Prefill Penalty ($M=16$ on large $K=17408$):**
  - **Measurement:** `ffn_down` ($17408 \times 5120$) and `attn_q` ($5120 \times 5120$) measure **0.82×** (an 18% slowdown) at $M=16$.
  - **Reason:** For $M=16$, the overhead of double-precision accumulator initialization and threadblock setup exceeds the cache benefit over stock scalar L1 hits.
  - **Mitigation:** In real inference workloads, prefill batches operate at $M \ge 128$ (prompt lengths 512–4096 tokens), where custom kernels achieve **1.76× to 9.27× speedups**. Small $M$ adaptive tuning is documented for v2 autotuning.

## 3. Platform & Hardware Constraints

- **WSL2 Memory Allocation Deficit:**
  - `.wslconfig` requires `memory=28GB`. Guest allocations below 20 GB fail with DXG `ENOMEM` (`-12`).
  - Context tier `32768` exceeds the physical 20 GiB VRAM allocation envelope under WSL2 DXG. The preflight check (`preflight.py`) intercepts this tier to prevent host driver crashes.
- **PTY Hang on Headless CLI Runs:**
  - Pinned `llama-cli` v0.2.0 defaults to interactive chat. Headless benchmark invocations must supply `--single-turn --simple-io --load-mode none` to avoid dead PTY stalls in `n_tty_write`.
- **DrvFs File-Lock Incompatibility:**
  - Git locks and CMake temporary builds on `/mnt/e` can deadlock. The `llama.cpp` build tree must reside on guest ext4 filesystem (`/root/llama.cpp`).

## 4. Frozen Configuration Lock

To maintain reproducibility and prevent silent performance regressions:
- AMD Adrenalin Driver: `26.2.2` (`32.0.31041.1004`)
- ROCm: `7.2.1` (`librocdxg 1.2.2`)
- Upstream llama.cpp commit: `bb4caa75`
- Model Artifact: `JonathanColetti/Qwen3.8-27B-Uncensored-IQ4_XS.gguf` (SHA256: `53adc4bb...`)
- WSL2 Snapshot: `E:\wsl-snapshots\ubuntu-2404-rocm721-phase1.tar`
