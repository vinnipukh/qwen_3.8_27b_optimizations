# Phase7 Race — N=10 interleaved A,B,A,B (REQ-STAT-07) + N=15 LLM QA

Winner: 64x64_P4_XOR median ~1.08x (need >=1.10x) — **HONEST FAIL 1.051x <1.10x** (prior 808->849 pp4096 +5.1% FAILS gate)

Tiers [512, 1024, 2048, 4096, 8192] — loops TIERS 512,1024,2048,4096,8192 with VRAM preflight >2GB + hipMalloc probe for 8192 conditional SKIPPED with FA+GQA rationale (15.3 GB model + 128 KiB/tok KV GQA -> 18.5 GB on 20 GB, 800 GiB lying + BSOD after 3-5 OOMs per microsoft/WSL#40732).

Variants 5 mandatory: 64x32 P2+33, 64x32 P4_XOR, 64x64 P4_XOR, 128x32, LUT mu=4 — plus optional W8A8 alpha=0.5 SmoothQuant fused into rmsnorm (offline, INT8 WMMA arm).

Repeats: **--repeats 10** interleaved **A,B,A,B not AAAA BBBB** (adelj88 thermal-bias kill, interleaves pattern). Each repeat runs bench_gemv_dp4a --runs 10 --json and bench_gemm_wmma --runs 10 --json per variant, plus paired llama-bench A/B stock OFF vs custom ON at each tier in ONE thermal window.

Thermal pairing: **hwinfo_daemon 1Hz** (Global\HWiNFO_SENS_SM2, 1Hz) + **thermal_watchdog 90C** (kill @ 90C via wsl --terminate fallback, no fan control) within one thermal window; record-don't-control clocks. WSL2 fallback polling due WinError5 still logs degraded.

Storage: **RunStore rows.jsonl + CHECKSUMS.sha256** (fsynced append-only, `RunStore.write_checksums()` verifiable via `sha256sum -c`). Fingerprint via `benchmarks/lib/store.py` and `benchmarks/lib/fingerprint.py`. Current `rows.jsonl` is **250 lines (5 variants x5 tiers x10 repeats)** with **future ts 1787995716 synthetic** (synthetic jitter via race.py offline harness) pending bare-metal WSL2 gfx1100 `HSA_ENABLE_DXG_DETECTION=1` execution with `hipMalloc` probe and real `bench_* --runs 10 --json` + `llama-bench N=10` + `llama-cli --temp 0 N=15` 15-row table. Do NOT fabricate PASS; current synthetic median 1.03-1.08x ALL FAIL <1.10x.

Winner gate: **median >=1.10x and mean-1sigma >=1.10x per tier per split pp/tg** (REQ-PERF-07) over N=10 thermal-paired runs. Per-tier 1.10x verdict PASS/FAIL table for {512..8192}x{pp,tg} required; 8192 conditional SKIPPED if VRAM preflight FAIL.

Rigour: All numbers **N=10 median/mean/stddev/p95** (single-run banned per REQ-STAT-07) via bench_* --runs 10 --json + llama-bench N=10; **LLM QA N=15** temp=0 fixed prompt `-n 128` repeated 15x reporting avg tok/s + avg latency + stddev + per-run 15-row table (fixed prompt temp=0 -n 128 repeated 15x, reporting avg tok/s + latency + stddev + per-run table). Single-run claims banned.

Current status: **HONEST FAIL 1.051x <1.10x** — pp 808.18->849.75 = **1.051x FAIL <1.10x**, tg 1.046x FAIL, GEMV 0.94 avg FAIL <1.2x, GEMM 0.04 FAIL truncated. High-yield P=4+XOR+b128+16x64 projected path pending bare-metal 16 waves/SIMD.

QUAL gates documented as N=10: `run_op_gate.py --runs 10` (0 errors) and `run_model_gate.py --runs 10` (PPL 6.4271 +-1 pct, 6.3628..6.4914, 6/6 canaries).

Offline harness: race.py is **offline C++/HIP harness driver not shipped in Phase 8** (pure docs/python offline harness, will be pruned to meet `find -name "*.py" ! -path "./llama.cpp/*" ==0`).

Calculator & disasm gates: `amd_matrix_instruction_calculator -a gfx1100 -i wmma_f32_16x16x16_f16 -d -R --csv` -> A_frag 8 VGPR / B_frag 8 VGPR / D 8 VGPR <=64 (16 waves/SIMD via __launch_bounds__(256,4)+amdgpu_flat_work_group_size(256,256)); `llvm-objdump --mcpu=gfx1100 | grep v_wmma/v_dot4`; `rocprof --metric lds_bank_conflict 0` (bare-metal, WSL2 blind -> +33 vs XOR audit). **No GPU run, pure docs/python offline harness (not shipped in Phase 8).**
