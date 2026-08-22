---
gsd_state_version: 1.0
current_phase: 2
current_phase_name: Benchmark Harness & Baseline Matrix
status: planning
stopped_at: Phase 1 complete, ready to plan Phase 2
last_updated: "2026-08-22T21:09:12.801Z"
last_activity: 2026-08-23
last_activity_desc: Phase 1 complete, transitioned to Phase 2
state_head: 09cc7de2edb01bf338d2f5153d82653885af4ce9
progress:
  total_phases: 6
  completed_phases: 1
  total_plans: 3
  completed_plans: 3
  percent: 17
---

# Project State

## Project Reference

See: .planning/PROJECT.md

**Core value:** Beat stock llama.cpp HIP on at least one important Qwen3.8-27B workload on the RX 7900 XT with a custom gfx1100 kernel, within agreed numerical tolerance — measured, reproducible, bisectable.

**Current focus:** Phase 1 EXECUTED (all 3 plans done) — pending verifier report → phase-complete gate. Next up: Phase 2 Benchmark Harness & Baseline Matrix.

## Current Position

Phase: 2 of 6 (Benchmark Harness & Baseline Matrix)
Status: Ready to plan
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

Last session: 2026-08-22
Stopped at: Phase 1 complete, ready to plan Phase 2
Next command after verifier lands: `phase complete 1` → `/gsd-plan-phase 2`
