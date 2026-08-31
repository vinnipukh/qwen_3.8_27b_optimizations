# Race Harness Note — Phase 7 High-Yield Variant Racing (07-04)

**Date:** 2026-08-30  
**Harness:** `benchmarks/results/phase7/race.py --repeats 10` interleaved A,B,A,B  
**Gate:** REQ-PERF-07 >=1.10x pp+tg at {512,1024,2048,4096,8192} (8192 conditional), REQ-STAT-07 N=10/N=15

## What it does
- **Interleaved A,B,A,B not AAAA BBBB** across `--repeats 10` to kill thermal bias (adelj88/rocm_wmma_gemm pattern). For each repeat r in 0..9, runs all 5 variants back-to-back per tier per split (pp and tg separately) inside ONE thermal window.
- **5 variants raced:** `64x32_P2+33` (baseline +33 padded, b128 float4), `64x32_P4_XOR` (quad-buffer + XOR preshuffle `x'=(y%(64/8))^x`), `64x64_P4_XOR` (64x reuse B-stationary weight in VGPR, T=64), `128x32` (8x2 warps for M=8192), `LUT_mu4` (16-entry half LUT `d*(ls-32)` baked via `tools/swizzle_iq4xs.py`). Optional W8A8 SmoothQuant alpha=0.5 deferred.
- **TIERS 512,1024,2048,4096,8192** with `-ngl 99 -b 2048 --single-turn --simple-io --load-mode none -r 10` per tier per build, stock OFF vs custom ON interleaved.
- **VRAM preflight >2GB for 8192:** `hipMemGetInfo` + `hipMalloc` probe before 8192; SKIPPED if <2GB free with FA+GQA rationale (15.3GB model + 128 KiB/tok KV -> 18.5GB on 20GB, 800 GiB lying + BSOD risk per microsoft/WSL#40732). No retry loops.
- **Thermal pairing:** `hwinfo_daemon` 1Hz (`Global\HWiNFO_SENS_SM2`, 1 Hz, clocks/power/temps per row, record-don't-control) + `thermal_watchdog` 90C (kill @ 90C per race spec, RUNBOOK kill @ 95C). WSL2 fallback: WinError5 HWiNFO access denied -> degraded polling but still logged; no hwmon in WSL2 still warns.
- **RunStore:** `benchmarks/results/phase7/rows.jsonl` append-only fsynced + `CHECKSUMS.sha256` + `manifest.json` fingerprint (commit, ROCm/driver, GGUF sha256). Single `wsl --export` snapshot discipline.
- **Winner pick:** median N=10 per tier per split (pp and tg separately) where both `median >=1.10x stock` AND `mean-1sigma >=1.10x`. Overall winner is variant winning most tiers with that gate.

## Honest result (2026-08-29, Windows host + WSL2 gfx1100 hardware)
- **Synthetic on Windows (no HIP):** `race.py --repeats 10` produced `rows.jsonl` 250 lines median **1.03-1.07x ALL FAIL <1.10x** (no PASS fabricated). Example: 512 pp ~1.08x FAIL, 1024 pp ~1.07x FAIL, 2048 pp ~1.08x FAIL, 4096 pp ~1.08x FAIL.
- **Hardware WSL2 gfx1100 (HSA_ENABLE_DXG_DETECTION=1, runs:10):** `bench_real_stock` 99.5us vs 543us 5.46x VERIFIED (runs:10), `bench_gemv_dp4a` avg **0.94x FAIL <1.2x** (peak 1.048 ffn_up, attn_q 0.965, mean-1sigma 0.40-0.61), `bench_gemm_wmma` 0.04x FAIL (M128 736us vs 17537us 24x slower, truncated 12288B needs `timeout 90` regen).
- **Paired llama-bench N=10:** stock 808.18±13.18 pp / 33.25±0.21 tg vs custom 849.75±34.60 pp / 34.79±0.44 tg at p4096 -> **1.051x pp / 1.046x tg FAIL <10%** (median and mean-1sigma <1.10x at {512,1024,2048,4096,8192}). Single-run banned; all numbers N>=10 median/mean/stddev/p95.
- **Verdict:** REQ-PERF-07 **FAIL** kept honest, REQ-STAT-07 harness-ready but hardware unverified (no 15x LLM QA yet), REQ-WIN-07 build_windows.bat not executed on this host. Do not fabricate 1.10x PASS.

## How to run (bare-metal WSL2 gfx1100)
```bash
python benchmarks/results/phase7/race.py --repeats 10 --tiers 512,1024,2048,4096,8192
# also: ./kernels/build/matmul_iq4xs/bench_gemv_dp4a --runs 10 --json
#       ./kernels/build/matmul_iq4xs/bench_gemm_wmma --runs 10 --json
#       ./build-custom/bin/llama-bench -m model.gguf -p 512..8192 -n 128 -ngl 99 -b 2048 -r 10 (interleaved A,B)
# LLM QA N=15: llama-cli fixed prompt temp=0 -n 128 repeated 15x -> avg tok/s + per-run 15-row table
```

## References
- `benchmarks/profiling/KERNEL-BENCH-DIFF.md §8` honest tables (synthetic vs hardware FAIL, 1.051x)
- `docs/PUBLICATION.md` §8 honest N=10 hardware vs synthetic, N=15 LLM QA harness
- `benchmarks/results/phase7/README.md` N=10/N=15 rigour, thermal pairing
- `docs/research/deep-research/high-yield/RDNA3-high-yield-keywords-synthesis.md` §A.3 race pattern
