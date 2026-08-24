# Stock Baseline Performance Matrix

**Model:** Qwen3.8-27B-Uncensored-IQ4_XS (15.31 GB)  
**Hardware:** AMD Radeon RX 7900 XT (20 GB GDDR6, gfx1100)  
**Environment:** ROCm 7.2.1 + librocdxg 1.2.2 (WSL2 Ubuntu 24.04)  
**Status:** Calibrated & Guarded (5 repeats per cell, warmup enabled)  

## Stock HIP Baseline Matrix (ROCm 7.2.1)

| Context Tier | Workload | Flash Attention | Mean Throughput (tok/s) | StdDev (tok/s) | Repeats | Verdict |
|---|---|---|---|---|---|---|
| 4096 | Prefill (pp) | off | **859.20** | ±155.60 | 5 | `OK` |
| 4096 | Prefill (pp) | on | **932.10** | ±19.86 | 5 | `OK` |
| 4096 | Decode@Ctx (tg 128) | off | **494.61** | ±55.61 | 5 | `OK` |
| 4096 | Decode@Ctx (tg 128) | on | **503.96** | ±56.49 | 5 | `OK` |
| 8192 | Prefill (pp) | off | **835.75** | ±78.90 | 5 | `OK` |
| 8192 | Prefill (pp) | on | **775.05** | ±92.22 | 5 | `OK` |
| 8192 | Decode@Ctx (tg 128) | off | **551.86** | ±56.39 | 5 | `OK` |
| 8192 | Decode@Ctx (tg 128) | on | **603.90** | ±27.99 | 5 | `OK` |
| 16384 | Prefill (pp) | off | **707.59** | ±42.66 | 5 | `OK` |
| 16384 | Prefill (pp) | on | **725.88** | ±53.17 | 5 | `OK` |
| 16384 | Decode@Ctx (tg 128) | off | **589.28** | ±27.09 | 5 | `OK` |
| 16384 | Decode@Ctx (tg 128) | on | **605.55** | ±26.60 | 5 | `OK` |

### Failed / Pre-flight Gated Cells

| Context Tier | Workload | Flash Attention | Target Backend | Verdict | Reason / Evidence |
|---|---|---|---|---|---|
| 32768 | Prefill (pp) | off | HIP | `FAILED:preflight-oom` | Estimated needed 18183.8 MiB (with margin 19093.0 MiB) exceeds available 18245.0 MiB free |
| 32768 | Decode@Ctx (tg 128) | off | HIP | `FAILED:preflight-oom` | Estimated needed 18183.8 MiB (with margin 19093.0 MiB) exceeds available 18245.0 MiB free |
| 32768 | Prefill (pp) | on | HIP | `FAILED:preflight-oom` | Estimated needed 18183.8 MiB (with margin 19093.0 MiB) exceeds available 18245.0 MiB free |
| 32768 | Decode@Ctx (tg 128) | on | HIP | `FAILED:preflight-oom` | Estimated needed 18183.8 MiB (with margin 19093.0 MiB) exceeds available 18245.0 MiB free |

## Reproducibility Gate Verification (BENCH-01)

**Evaluation:** Re-run of Context Tier `8192` in an independent session.  
**Gate Criteria:** Throughput mean variance must be within `±5.0%`.  
**Overall Verdict:** **FAIL**  

| Cell | Session 1 Mean (tok/s) | Session 2 Mean (tok/s) | Variance (%) | Gate (<= 5.0%) |
|---|---|---|---|---|
| `HIP_c8192_pp_fa_off` | 835.75 | 763.78 | **8.61%** | ❌ FAIL |
| `HIP_c8192_tg_fa_off` | 551.86 | 549.25 | **0.47%** | ✅ PASS |
| `HIP_c8192_pp_fa_on` | 775.05 | 748.12 | **3.47%** | ✅ PASS |
| `HIP_c8192_tg_fa_on` | 603.90 | 568.01 | **5.94%** | ❌ FAIL |
