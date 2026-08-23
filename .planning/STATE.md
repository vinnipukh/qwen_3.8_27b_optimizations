---
gsd_state_version: 1.0
current_phase: 2
current_phase_name: Benchmark Harness & Baseline Matrix
status: executing
stopped_at: Phase 2 context gathered
last_updated: "2026-08-23T14:13:21.176Z"
last_activity: 2026-08-23
last_activity_desc: Phase 1 complete, transitioned to Phase 2
state_head: 668a4771cd1d681dc1931a20427e41765b8f9cd4
progress:
  total_phases: 6
  completed_phases: 1
  total_plans: 8
  completed_plans: 3
  percent: 17
---

# Project State

## Project Reference

See: .planning/PROJECT.md

**Core value:** Beat stock llama.cpp HIP on at least one important Qwen3.8-27B workload on the RX 7900 XT with a custom gfx1100 kernel, within agreed numerical tolerance — measured, reproducible, bisectable.

**Current focus:** Phase 1 EXECUTED (all 3 plans done) — pending verifier report → phase-complete gate. Next up: Phase 2 Benchmark Harness & Baseline Matrix.

## Current Position

Phase: 2 (Benchmark Harness & Baseline Matrix) — READY TO EXECUTE
Status: Ready to execute
Last activity: 2026-08-23 — Phase 1 complete, transitioned to Phase 2

Progress: [██░░░░░░░] ~17% (1 of 6 phases)

## What Phase 1 proved

- Platform kill-gate CLEARED: ROCm 7.2.1 + librocdxg 1.2.2 under WSL2, gfx1100 enumerated, HIP kernels execute on device
- Stock baseline EXISTS: llama.cpp v0.2.0 @ bb4caa75, 4 binaries archived, op-tests green on ROCm0 backend
- Model runs FULLY on GPU: 132/132 tensor layers on ROCm0, zero CPU assignments
- **Stock numbers to beat: pp 111.5 tok/s · tg 33.5 tok/s** (2048 ctx, single turn)
- Environment frozen: E:\wsl-snapshots\ubuntu-2404-rocm721-phase1.tar (49 GB; slim re-export offered to owner)

## Key environment facts (carry forward)

- .wslconfig memory=28GB was REQUIRED (15 GB caused DXG ENOMEM during VRAM alloc)
- Canonical model copy for runs: guest /root/models/ (mmap over /mnt/e stalls)
- Headless llama-cli needs: setsid + --simple-io + --single-turn
- D-04 amended: no-silent-updates scope; one elevated registry command still pending owner

## Pending Todos

- Owner: elevated-shell registry command (see benchmarks/environment/versions.txt)
- Snapshot slimming decision (49 GB → optional ~15-25 GB)
- Verifier report → then `phase complete 1`

## Session Continuity

**Resume file:** .planning/phases/02-benchmark-harness-baseline-matrix/02-CONTEXT.md

Last session: 2026-08-23T10:36:14.989Z
Stopped at: Phase 2 context gathered
Next command after verifier lands: `phase complete 1` → `/gsd-plan-phase 2`
