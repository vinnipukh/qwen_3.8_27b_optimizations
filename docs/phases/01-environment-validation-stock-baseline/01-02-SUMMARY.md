---
phase: 1
plan: 01-02
status: done
---

# Plan 01-02 Summary: Pinned llama.cpp Build & Baseline Archive

**Result:** ✅ ALL TASKS COMPLETE

## What was delivered
- llama.cpp pinned to **v0.2.0 @ bb4caa7540188872173c44d161602d9271386413** (lineage ≥ b8394 verified)
- Built for gfx1100: `-DGGML_HIP=ON -DGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_SERVER=OFF` (+ccache; deviations documented)
- 4 binaries archived: `baseline/binaries/v0.2.0-bb4caa75/{llama-cli, llama-bench, llama-perplexity, test-backend-ops}`
- **test-backend-ops PASSED on ROCm0 backend** (op-level gate green)
- Provenance in `benchmarks/environment/llamacpp-pin.txt`; repo payload well under 750 MB (binaries ~2.4 MB total, dynamic-linked)

## Incidents resolved
- DrvFs git-lock failure → source tree at guest ext4 `/root/llama.cpp`
- WebUI npm-via-Windows-interop build failure → server/UI disabled + HF dist fallback
- First op-test run saw CPU only → fixed by sourcing rocdxg env before run

## Verification
- HEAD sha matches pin exactly
- `test-backend-ops-phase1.txt`: "Backend 1/2: ROCm0 ... OK ... backends passed"
