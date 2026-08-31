# Phase 1: Environment Validation & Stock Baseline — SUMMARY

**Completed:** 2026-08-22 · **Status:** ALL SUCCESS CRITERIA MET ✅

## Results vs Success Criteria

| SC | Requirement | Verdict | Evidence |
|----|-------------|---------|----------|
| 1 | rocminfo enumerates gfx1100 + HIP smoke executes on device | ✅ PASS | `rocminfo.txt` (2× gfx1100); hipsmoke `RESULT=1 ARCH=gfx1100`, exit 0 |
| 2 | llama-cli/bench/perplexity/test-backend-ops build+run at pinned commit | ✅ PASS | v0.2.0 @ `bb4caa75`; 4 binaries in `baseline/binaries/v0.2.0-bb4caa75/`; test-backend-ops OK on ROCm0 backend (`test-backend-ops-phase1.txt`) |
| 3 | IQ4_XS fully on GPU, zero CPU fallback, both layer families | ✅ PASS | 132 tensor-layer assignments → ROCm0, **0 → CPU**; GDN + gated-attn both resident; pp 111.5 t/s / tg 33.5 t/s; exit 0 (`startup-log.txt`) |
| 4 | Full provenance + versions recorded | ✅ PASS | `models/README.md` (sha256 verified), `versions.txt`, `llamacpp-pin.txt`, `vram-probe.txt` |

## Key numbers
- Stock baseline throughput (2048 ctx, single turn): **111.5 tok/s prompt · 33.5 tok/s decode**
- Effective bandwidth implied by decode: ~457 GB/s → VRAM-only speed (sanity: DDR4 would yield ~3 tok/s)
- Environment frozen: `E:\wsl-snapshots\ubuntu-2404-rocm721-phase1.tar` (49.4 GB)

## Deviations & incidents (all documented, all resolved)
1. amdgpu-install 30.30.x has no `wsl` usecase → `--usecase=rocm --no-dkms`
2. DrvFs breaks git lock-files → source tree moved to guest ext4 `/root/llama.cpp`
3. WebUI subbuild invokes Windows npm via interop → `-DLLAMA_BUILD_SERVER=OFF`
4. **15 GB guest RAM caused DXG ENOMEM during VRAM allocation** (dmesg `-12`) → D-02 escalation applied: `.wslconfig memory=28GB` → 27 GB visible; resolved
5. llama-cli hung writing to dead PTY (`n_tty_write` wait) → headless runs require `setsid` + `--simple-io`
6. v0.2.0 defaults to interactive chat mode → use `--single-turn` for gates
7. Model loads from `/mnt/e` can stall under mmap → canonical model copy at guest `/root/models/`
8. D-04 amended (owner): no-silent-updates scope, detection via fingerprint gates; elevation pending owner

## Open items carried to Phase 2
- Driver-update pause registry command still needs one elevated shell (owner action; commands in `versions.txt`)
- `.wslconfig` change must be recorded as part of benchmark environment fingerprint going forward
- llama.cpp v0.2.0 logs lack classic "offloaded N/N" line — residency evidence = per-layer device assignment lines (verbose mode)
