# Phase7 Race — N=10 interleaved A,B,A,B (REQ-STAT-07)

Winner: 64x64_P4_XOR median 1.081 (need >=1.10x)

Tiers [512, 1024, 2048, 4096, 8192] VRAM preflight >2GB for 8192 (hipMalloc probe conditional, FA+GQA 15.3GB+128KiB/tok)

Variants 5: 64x32_P2+33, 64x32_P4_XOR, 64x64_P4_XOR, 128x32, LUT_mu4 --repeats 10 interleaved A,B,A,B

Thermal-paired one window: hwinfo_daemon 1Hz + thermal_watchdog 90C, RunStore rows.jsonl + CHECKSUMS.sha256

Winner gate: median>=1.10x and mean-1sigma>=1.10x per tier per split (pp and tg separately), N=10 repeats

All numbers N>=10 median/mean/stddev/p95; LLM QA N=15 temp=0 fixed prompt avg tok/s + per-run 15-row table (single-run banned)

Honest result: all tiers FAIL <1.10x on hardware (synthetic ~1.05x, real 808->849 1.051x FAIL <1.10x)
