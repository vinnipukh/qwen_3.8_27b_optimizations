# freetoken-rocm-probe

Bandwidth profiling harness implementing **step zero of the FreeToken method**
(arXiv 2608.16157): measure the two deployed-machine bandwidths that drive its
q\* bandwidth-adaptive decode policy, then project what edge MoE serving could
achieve on this exact box before porting anything.

Target machine: **Ryzen 7 5700X + RX 7900 XT (20 GB) + 32 GB DDR4-3200, Windows**

## Files

| file | purpose |
|---|---|
| `src/bench_bh.cpp` | B_H probe — AVX2 FMA streaming over 2 GiB (simulates CPU-side MoE expert kernel, memory-bound) |
| `src/bench_bp.cpp` | B_P probe — pinned-style H2D/D2H transfers over PCIe via OpenCL.dll (runtime-loaded, no SDK needed) |
| `src/qstar.mjs`   | q\* policy projector — splits misses into PCIe fills vs CPU execution, projects batch-1 decode tok/s |
| `bin/`            | compiled exes |
| `tools/`          | portable zig 0.16 (compiler only, nothing installed system-wide) |

## Rebuild / rerun

```bash
tools/zig-x86_64-windows-0.16.0/zig.exe c++ -O3 -mavx2 -mfma -o bin/bench_bh.exe src/bench_bh.cpp
tools/zig-x86_64-windows-0.16.0/zig.exe c++ -O3 -o bin/bench_bp.exe src/bench_bp.cpp

./bin/bench_bh.exe 2 12 8,16          # GiB working set, passes, thread counts
./bin/bench_bp.exe 16,64,256 12       # transfer sizes MiB, reps
node src/qstar.mjs --bp <BP> --bh <BH> --preset qwen30a3
```

## Measured results (2026-08-22)

### B_H — host DRAM streaming (bench_bh)

| mode | threads | median GB/s |
|---|---|---|
| read | 8 | **38.2** |
| read | 16 | 36.0 |
| fma (MoE-like) | 8 | **36.4** ← use this |
| fma | 16 | 36.4 |

Physical cores only wins; SMT adds nothing (memory-bound). ≈71% of DDR4-3200
theoretical — normal dual-channel Zen3.

### B_P — PCIe transfers (bench_bp, OpenCL/WDDM path)

| dir | 16 MiB | 64 MiB | 256 MiB |
|---|---|---|---|
| H2D med | 23.5 | 19.6 | **18.8** GB/s |
| D2H med | 7.5 | 14.2 | 19.0 GB/s |

Steady-state H2D ≈ **19–24 GB/s**. WDDM taxes small transfers; native
Linux/ROCm typically lands higher (Gen4 x16 theoretical ≈32 GB/s; expect
~25–28 on Linux). Conservative value used in projections: **22 GB/s**.

### q\* projection (Qwen3-30B-A3B-class @ MXFP4, pool 13.5 GB, S=2.1 MiB/expert)

```
q* = B_P/B_H = 0.604  ->  ~60% of misses become PCIe cache fills,
                          ~40% execute in place on CPU
hit%   tok/s (projected, batch 1)
 50%      58
 70%     102
 85%     137   <- paper measured 84% hit rate at 37% pool cached on RTX 5090
```

The 85% row exceeds reality by design: it omits attention/shared-expert/sampling
time. The useful reading is that **at high hit rates the memory system stops
being the bottleneck** — matching the paper's finding that engine overhead
dominates once locality is good.

## Fit check for this machine

- Expert pool must fit host RAM alongside OS: ≤ ~13–16 GB on 32 GB total.
  A 30B-class A3B MoE at MXFP4 fits; 284B-class does not.
- 20 GB VRAM comfortably holds non-expert weights + KV + ~40–60% of a
  13.5 GB expert pool → paper-level (84%) hit rates are plausible targets.

## Caveats

- WDDM numbers understate what Linux/ROCm will do; re-run `bench_bp` logic as
  HIP (`hipMemcpyAsync` on registered pinned memory) after moving to Linux.
- Projections are first-order (fixed hit rate across layers, no scheduling
  jitter, batch 1).
- The OpenCL path measures the same DMA window an engine would use, but a real
  FreeToken-style runtime overlaps fills with compute; exposed latency can be
  lower than raw transfer time suggests when pipelined.

## Next steps toward a real ROCm port

1. Linux install (dual-boot), ROCm + PyTorch ROCm wheels; verify gfx1100.
2. Port bench_bp to HIP to get true Linux B_P/B_H.
3. Stand up Triton fused-MoE decode (vLLM ROCm kernels are the reference),
   wrap with LRU expert cache + q\* scheduler outside any graph capture.
4. Add HIP Graph replay per layer once numerics are validated.
