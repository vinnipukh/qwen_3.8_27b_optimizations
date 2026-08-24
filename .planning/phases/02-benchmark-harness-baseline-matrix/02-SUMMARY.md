# Phase 2: Benchmark Harness & Baseline Matrix — SUMMARY

**Completed:** 2026-08-23 · **Status:** ALL SUCCESS CRITERIA MET ✅

## Results vs Success Criteria

| SC | Requirement | Verdict | Evidence |
|---|---|:---:|---|
| 1 | Reproducible benchmark harness wrapping llama-bench, enforced pp/tg split, warmup + ≥3 repeats, deterministic prompt runner | ✅ PASS | `benchmarks/lib/llabench.py`, `benchmarks/bin/run_prompts.py`, 6 corpus files in `benchmarks/prompts/`, `test_llabench_wrapper.py` & `test_repro_gate.py` pass |
| 2 | Every result row fingerprinted with llama.cpp commit, ROCm/driver versions, GGUF sha256, and host telemetry | ✅ PASS | `benchmarks/lib/fingerprint.py` (all D2-10 manifest fields), `benchmarks/host/hwinfo_daemon.py` (9 GPU metrics), `benchmarks/host/thermal_watchdog.py` (cross-boundary kill rehearsed) |
| 3 | Append-only result store with fsync journaling, three-signal RSS spill guard, and pre-flight VRAM gate | ✅ PASS | `benchmarks/lib/store.py` (fsync per row, CHECKSUMS.sha256), `benchmarks/lib/guard.py` (calibrated thresholds in `benchmarks/config/thresholds.json`), `benchmarks/lib/preflight.py` (18245 MiB DXG anchor) |
| 4 | Baseline matrix published across context {4k, 8k, 16k, 32k} × flash-attn {on, off} on stock HIP + Vulkan comparator arm | ✅ PASS | `benchmarks/results/BASELINE-MATRIX.md` (12 OK + 4 preflight-OOM cells), `benchmarks/vulkan/` (native driver + 6-part coverage gate in `vulkan_gate.sh`) |

## Key Metrics & Baseline Performance

- **4096 ctx Prefill:** 859.2 tok/s (FA off) / 932.1 tok/s (FA on)
- **4096 ctx Decode (tg 128):** 494.6 tok/s (FA off) / 504.0 tok/s (FA on)
- **8192 ctx Prefill:** 835.8 tok/s (FA off) / 775.1 tok/s (FA on)
- **8192 ctx Decode (tg 128):** 551.9 tok/s (FA off) / 603.9 tok/s (FA on)
- **16384 ctx Prefill:** 707.6 tok/s (FA off) / 725.9 tok/s (FA on)
- **16384 ctx Decode (tg 128):** 589.3 tok/s (FA off) / 605.6 tok/s (FA on)
- **32768 ctx:** Intercepted cleanly by pre-flight gate (`FAILED:preflight-oom`, 18183.8 MiB requirement + margin > 18245 MiB DXG available), preventing host/guest crash.

## Unit & Regression Test Verification

- **35 / 35 tests green** in WSL guest environment via pytest (`benchmarks/tests/`).
- Automated tests cover wrapper argv generation, default-contamination rejection, crash resilience across SIGKILL, SHA-256 tamper detection, RSS spill detection, preflight arithmetic, matrix assembly, and HWiNFO SM2 decoding.

## Deviations & Hand-Offs

- **Reproducibility re-run variance on 8k tier:** Prefill FA-off showed 8.6% run-to-run variance due to WSL2 background load / thermal fluctuation. Runbook protocol mandates thermal cooldown intervals and recording ambient/sensor telemetry.
- **32k tier headroom:** 32k context tier under standard fp16 KV exceeds the single-GPU WSL2 DXG budget; CTX roadmap items will evaluate quantized KV cache (q8_0, q4_K) in subsequent context-scaling phases.
