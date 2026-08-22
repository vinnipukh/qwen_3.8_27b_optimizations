---
status: passed
phase: 1
verified_date: 2026-08-22
verifier: orchestrator-fallback (gsd-verifier wedged on provider; documented deviation)
---

# Phase 1 Verification Report

**Verified:** 2026-08-22 · **Method:** goal-backward artifact inspection
**Note (process):** spawned gsd-verifier wedged on provider stream failure (same as three planner children + researcher today); fallback to orchestrator-authored verification per documented deviation in STATE.md. All evidence below was re-checked against on-disk artifacts.

## Verdict: VERIFIED ✅

## SC1 — rocminfo gfx1100 + HIP device execution (ENV-01)
- `benchmarks/environment/rocminfo.txt`: contains `AMD Radeon RX 7900 XT`, `Name: gfx1100`, GPU agent ✅ (grep -c gfx1100 = 2)
- HIP smoke: source archived at `hipsmoke.cpp`; session log records `RESULT=1 ARCH=gfx1100 NAME=AMD Radeon RX 7900 XT`, exit 0 ✅
- Device-execution proof is REAL (kernel launched, result verified correct)

## SC2 — pinned build + tools run (ENV-02)
- Pin: tag v0.2.0, commit bb4caa7540188872173c44d161602d9271386413 recorded in `llamacpp-pin.txt` ✅
- 4 binaries present: `baseline/binaries/v0.2.0-bb4caa75/{llama-cli,llama-bench,llama-perplexity,test-backend-ops}` ✅
- `test-backend-ops-phase1.txt`: "Testing 2 devices / Backend 1/2: ROCm0 / OK / backends passed" ✅ — green on ROCm backend specifically

## SC3 — full-GPU residency, zero CPU fallback, both layer families (ENV-03)
- `startup-log.txt` (verbose capture): repeated `D load_tensors: layer N assigned to device ROCm0`; count **132 ROCm0 assignments, 0 CPU assignments** ✅
- Hybrid-arch coverage: Gated DeltaNet layers and gated-attention layers both among assigned set (64-layer qwen35 graph fully mapped) ✅
- Generation succeeded, clean single-turn exit 0; **pp 111.5 tok/s · tg 33.5 tok/s** recorded in `vram-probe.txt` with bandwidth sanity argument (~457 GB/s effective ⇒ VRAM-only) ✅

## SC4 — provenance completeness (ENV-04)
- `models/README.md`: repo id, file, size 15,309,039,008 B, sha256 53adc4bb…f5 (verified OK via sha256sum -c during execution), HF rev dee0a3164d9e11bbbebf5b63f52ba99443d14fc3, quantizer a94d563ed, imatrix, base model, arch facts, verified date ✅
- `versions.txt`: ROCm version, frozen driver pairing (32.0.31041.1004), D-04 amendment, librocdxg 1.2.2, smoke verdict, snapshot ledger entry (49,415,639,040 B) ✅
- Snapshot exists at E:\wsl-snapshots\ubuntu-2404-rocm721-phase1.tar ✅ (serial-last ordering respected)

## Accepted deviations (weighed, not failures)
| Deviation | Disposition |
|---|---|
| D-04 update-pause registry command pending owner elevation | Documented in versions.txt as PENDING-OWNER-ACTION; detection net (fingerprint gates) compensates |
| `--usecase=rocm --no-dkms` instead of `wsl` usecase | amdgpu-install 30.30.x lacks `wsl`; equivalent semantics, recorded |
| Source tree at guest ext4 `/root/llama.cpp`, not repo | DrvFs git-lock incompatibility; binaries archived in-repo; provenance records location |
| `-DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF` | Build-scope reduction beyond roadmap flag set; required for clean build (npm interop contamination); server out of scope |
| `.wslconfig memory=28GB` (D-02 escalation applied) | Evidence-driven per decision D-02; dmesg ENOMEM proof captured in session record |

## Requirement closure
ENV-01 ✅ · ENV-02 ✅ · ENV-03 ✅ · ENV-04 ✅ — no gaps found.
