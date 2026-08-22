<!-- generated-by: gsd-doc-writer -->
# qwen_3.8_27b_optimizations

Custom HIP kernel development for Qwen3.8-27B IQ4_XS inference on an AMD RX 7900 XT
(gfx1100), running stock llama.cpp under WSL2 ROCm 7.2.1. Goal: at least one custom
gfx1100 kernel that beats the pinned stock build on a real workload, gated by
numerical-correctness checks — benchmark before optimize, one change at a time,
prefill and decode measured separately, failures published like wins.

## Current status

| Item | State |
|---|---|
| Phase | **1 of 6 complete** — Environment Validation & Stock Baseline (2026-08-22) |
| Stock baseline | pp **111.5 tok/s** · tg **33.5 tok/s** (2048 ctx, single turn, fully GPU-resident) |
| Baseline archive | `baseline/binaries/v0.2.0-bb4caa75/` (llama.cpp v0.2.0 @ `bb4caa75`) |
| Model | Locked: `JonathanColetti/Qwen3.8-27B-Uncensored-IQ4_XS.gguf`, 15.31 GB, sha256-verified (`models/README.md`) |
| Frozen env | `E:\wsl-snapshots\ubuntu-2404-rocm721-phase1.tar` (49.4 GB WSL snapshot) |

Remaining phases: benchmark harness → correctness gates & profiling → kernel
playground → first custom kernel → integration & publication. See `.planning/ROADMAP.md`.

## Quick links

| Path | Contents |
|---|---|
| `.planning/ROADMAP.md` | 6-phase plan, methodology rules, merge map |
| `.planning/REQUIREMENTS.md`, `.planning/PROJECT.md` | Scope, decisions, success criteria |
| `.planning/phases/01-*` | Phase 1 plans, summaries, verification evidence |
| `benchmarks/environment/` | rocminfo, hipconfig, versions, VRAM probe, backend-ops logs |
| `baseline/binaries/v0.2.0-bb4caa75/` | Archived stock binaries (gitignored artifacts) |
| `models/README.md` | Model provenance: HF revision, sha256, quantizer details |
| `docs/GETTING-STARTED.md` | Full setup from Windows to first generation |

## Hardware & software requirements

| Component | Requirement |
|---|---|
| OS | Windows 11 with WSL2, Ubuntu 24.04 guest (root-only) |
| GPU | AMD RX 7900 XT (gfx1100); driver 32.0.31041.1004 / Adrenalin 26.10.41 — frozen, no silent updates |
| ROCm | 7.2.1 in guest + librocdxg 1.2.2; `HSA_ENABLE_DXG_DETECTION=1` via `/etc/profile.d/rocdxg.sh` |
| RAM | `.wslconfig` `[wsl2] memory=28GB` is **required** — 15 GB guest RAM caused DXG ENOMEM during VRAM allocation |
| VRAM | 20 GB class card; model runs fully on GPU (`-ngl 99`), zero CPU fallback |
| Build | cmake + HIP toolchain, llama.cpp pinned v0.2.0 @ `bb4caa75`, built `-DGGML_HIP=ON -DGPU_TARGETS=gfx1100 -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF` |
| Source tree | Guest ext4 `/root/llama.cpp` (DrvFs breaks git lock-files — do not clone into `/mnt/*`) |

## One-command smoke test

From Windows, against the guest-side model copy and archived binary:

```bash
wsl -- setsid /root/llama.cpp/build/bin/llama-cli \
  -m /root/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf \
  -ngl 99 -c 2048 -p "Hello" -n 32 --temp 0 -e \
  --single-turn --simple-io --no-mmap
```

Headless runs require `setsid` + `--simple-io` (dead-PTY hang otherwise),
`--single-turn` (v0.2.0 defaults to interactive chat), and `--no-mmap`
(mmap stalls on `/mnt/*`). Pass = exit 0, full offload lines, no CPU buffer lines.

## License

No project license has been chosen yet. The base model `Qwen/Qwen3.8-27B` is
Apache-2.0; see `models/README.md` for artifact provenance.
