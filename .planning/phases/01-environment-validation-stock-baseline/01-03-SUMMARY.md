---
phase: 1
plan: 01-03
status: done
---

# Plan 01-03 Summary: Model Artifact, Runtime Gate, Provenance & Snapshot

**Result:** ✅ ALL TASKS COMPLETE

## What was delivered
- IQ4_XS artifact downloaded (owner delegated D-07 supersession) → `models/`, **sha256 VERIFIED OK** vs locked digest `53adc4bb…f5`
- Provenance: `models/README.md` (repo, rev dee0a316, size 15,309,039,008 B, quantizer a94d563ed, imatrix, base model, arch facts)
- **ENV-03 runtime gate PASSED**: 132 tensor-layer assignments → ROCm0 (gfx1100), **0 → CPU**; both Gated DeltaNet and gated-attention layer families resident; single-turn generation "Hello! How can I help you today?" exit 0; **pp 111.5 tok/s / tg 33.5 tok/s** stock baseline numbers recorded
- Empirical VRAM evidence in `vram-probe.txt`; verbose per-layer assignment log preserved in `startup-log.txt`
- Serial-last snapshot: `E:\wsl-snapshots\ubuntu-2404-rocm721-phase1.tar` (49,415,639,040 B), ledger entry appended to `versions.txt`

## Incidents resolved (important for Phase 2+)
1. DXG ENOMEM during VRAM allocation at 15 GB guest RAM (dmesg `dxgkio_create_allocation: -12`) → **D-02 escalation applied**: `.wslconfig memory=28GB` → 27 GB visible; resolved
2. llama-cli hang blocked in `n_tty_write` on dead PTY → headless runs need `setsid` + `--simple-io`
3. v0.2.0 interactive-mode default caused prompt-flood → use `--single-turn`
4. Model reads from `/mnt/e` stall under mmap → canonical copy at guest `/root/models/`

## Verification
- sha256sum -c → OK; startup-log.txt contains per-layer ROCm0 assignment lines; vram-probe.txt gate verdict PASS
