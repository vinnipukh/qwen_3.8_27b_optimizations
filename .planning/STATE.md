---
gsd_state_version: '1.0'
status: executing
progress:
  total_phases: 6
  completed_phases: 0
  total_plans: 3
  completed_plans: 0
  percent: 5
---

# Project State — Phase 1 RESUME NOTES (2026-08-22, session interrupted by owner shutdown)

## Where we are in Phase 1 (plans at .planning/phases/01-environment-validation-stock-baseline/)

**DONE (committed):**
- Plan 01-01 COMPLETE: ROCm 7.2.1 + librocdxg v1.2.2 installed (root, Ubuntu-24.04). rocminfo sees RX 7900 XT gfx1100. HIP device smoke PASSED (`RESULT=1 ARCH=gfx1100`, source kept at benchmarks/environment/hipsmoke.cpp). Fingerprints archived. D-04 update-pause = PENDING OWNER (registry writes need elevation — exact commands in benchmarks/environment/versions.txt).
- Plan 01-02 COMPLETE: llama.cpp v0.2.0 @ bb4caa75 built for gfx1100 (source tree at guest /root/llama.cpp — DrvFs git-lock issue documented; configure flags in llamacpp-pin.txt incl. -DLLAMA_CURL=OFF -DLLAMA_BUILD_SERVER=OFF). 4 binaries archived baseline/binaries/v0.2.0-bb4caa75/. test-backend-ops PASS on ROCm0 backend.
- Plan 01-03 T1+T2 DONE: model downloaded to models/, sha256 VERIFIED OK vs locked digest; models/README.md provenance written.

**REMAINING (resume here):**
1. Plan 01-03 T3 — ENV-03 runtime gate: run archived binary with model fully on GPU:
   script template was C:\Users\arhan\AppData\Local\Temp\g1.sh (copy pattern: wsl cp to /root, chmod, run):
   `cd /root/llama.cpp/build-ci/bin && source /etc/profile.d/rocdxg.sh && export LD_LIBRARY_PATH=$B:$LD_LIBRARY_PATH && ./llama-cli -m /mnt/e/.../models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf -ngl 99 -c 2048 -p "Hello" -n 32 --temp 0 -e > startup-log.txt`
   Objective predicates per plan 01-03 T3: offloaded 64/64 line; CUDA0 buffer ≥ ~13.7 GiB total; ZERO fallback-suspect lines; exit 0 + ≥16 chars output. On OOM: D-02 .wslconfig escalation FIRST (memory=24GB + wsl --shutdown), then -ngl descent ladder.
   WARNING: loading from /mnt/e is slow (~9p); first attempt hit the owner's shutdown timer mid-load. Consider copying GGUF into guest ext4 (~/models) for the gate run, or just be patient.
2. Plan 01-03 T4 — serial-last: `wsl --export Ubuntu-24.04 E:\wsl-snapshots\ubuntu-2404-rocm721-phase1.tar` (+ append size to versions.txt).
3. Write phase SUMMARY.md, mark plans complete, update ROADMAP progress table.

## Gotchas learned this session (do not relearn)
- Git-bash on Windows strips ${VAR} from args passed through wsl.exe → ALWAYS write scripts to files (C:\Users\arhan\AppData\Local\Temp maps to /mnt/c/...), copy into guest, execute.
- printf with C-code containing %d gets eaten — use write-tool files not heredocs.
- amdgpu-install 30.30.x has no 'wsl' usecase → use --usecase=rocm --no-dkms.
- npm/webui subbuild invokes WINDOWS npm via interop → keep -DLLAMA_BUILD_SERVER=OFF.
- git clone onto /mnt/e fails (lock file) → clone/build in ext4, copy binaries out.
- test-backend-ops needs rocdxg env sourced or it silently tests CPU only.
- Subagent provider (ox-alpha-free) dropped streams repeatedly today → long children unreliable; direct orchestration used instead (documented deviation).

## Session Continuity
Last session: 2026-08-22 · Stopped at: runtime-gate launch aborted by owner shutdown
Next command after resume: `/gsd-execute-phase` continuation of plan 01-03 (steps above)
