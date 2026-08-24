<!-- generated-by: gsd-doc-writer -->
# qwen_3.8_27b_optimizations

Custom HIP kernel development for Qwen3.8-27B IQ4_XS inference on an AMD Radeon RX 7900 XT
(gfx1100), running stock llama.cpp under WSL2 ROCm 7.2.1. Goal: at least one custom
gfx1100 kernel that beats the pinned stock build on a real workload, gated by
numerical-correctness checks — benchmark before optimize, one change at a time,
prefill and decode measured separately, failures published like wins.

## Current status

| Item | State |
|---|---|
| Phase | **3 of 6 complete (50%)** — Phase 4 (Kernel Playground Scaffold) ready for planning & implementation |
| Optimization Target #1 | Designated: **`MUL_MAT`** (quantized IQ4_XS GEMV/GEMM, 31.12% cumulative GPU time) |
| Quality Gates | Armed & Green: Op-gate (21,093 ops, 0 errors) + PPL 6.4271 ± 0.04103 + 6/6 golden canaries |
| Stock baseline matrix | Published in `benchmarks/results/BASELINE-MATRIX.md` (12 OK + 4 pre-flight intercepted cells) |
| Baseline archive | `baseline/binaries/v0.2.0-bb4caa75/` (llama.cpp v0.2.0 @ `bb4caa75`) |
| Model | Locked: `JonathanColetti/Qwen3.8-27B-Uncensored-IQ4_XS.gguf`, 15.31 GB, sha256-verified (`models/README.md`) |
| Frozen env | `E:\wsl-snapshots\ubuntu-2404-rocm721-phase1.tar` (49.4 GB WSL snapshot) |

Remaining phases: kernel playground scaffold → first custom kernel (`MUL_MAT` attack) → integration & publication. See `.planning/ROADMAP.md`.

## Stock Performance Matrix (HIP ROCm 7.2.1)

| Context Tier | Workload | Flash Attention | Mean tok/s | StdDev | Repeats | Verdict |
|---|---|---|---|---|---|---|
| **4096** | Prefill (`pp`) | off / on | 859.20 / **932.10** | ±19.86 | 5 | `OK` |
| **4096** | Decode (`tg 128`) | off / on | 494.61 / **503.96** | ±56.49 | 5 | `OK` |
| **8192** | Prefill (`pp`) | off / on | **835.75** / 775.05 | ±78.90 | 5 | `OK` |
| **8192** | Decode (`tg 128`) | off / on | 551.86 / **603.90** | ±27.99 | 5 | `OK` |
| **16384** | Prefill (`pp`) | off / on | 707.59 / **725.88** | ±53.17 | 5 | `OK` |
| **16384** | Decode (`tg 128`) | off / on | 589.28 / **605.55** | ±26.60 | 5 | `OK` |
| **32768** | All cells | off / on | N/A | N/A | 0 | `FAILED:preflight-oom` |

## Quick links

| Path | Contents |
|---|---|
| `.planning/ROADMAP.md` | 6-phase plan, methodology rules, merge map |
| `.planning/REQUIREMENTS.md`, `.planning/PROJECT.md` | Scope, decisions, success criteria |
| `benchmarks/RUNBOOK.md` | Binding session protocol, guard thresholds, thermal policy |
| `benchmarks/results/BASELINE-MATRIX.md` | Published stock baseline matrix + reproducibility verification |
| `benchmarks/profiling/BOTTLENECK-TABLE.md` | Phase 3 published 4-shape bottleneck attribution report |
| `benchmarks/golden/` | Golden perplexity baseline and prompt canary store |
| `.planning/phases/03-correctness-gates-bottleneck-profiling/` | Phase 3 context, discussion log, research, plans, and summaries |
| `.planning/reference/GPU-KERNEL-RESOURCES.md` | Master index of AMD RDNA3 ISA, HIP optimization, and kernel tuning docs |
| `benchmarks/lib/` | Harness modules (`llabench.py`, `fingerprint.py`, `guard.py`, `store.py`, `preflight.py`, `parse_profile.py`, `toast.py`) |
| `benchmarks/bin/` | Orchestration CLIs (`run_session.py`, `run_op_gate.py`, `run_model_gate.py`, `profile_workload.py`, `profile_matrix.py`) |
| `benchmarks/environment/` | rocminfo, hipconfig, versions, VRAM probe, backend-op support CSVs |
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

From Windows, against the guest-side model copy and pinned binary:

```bash
wsl -- setsid /root/llama.cpp/build-ci/bin/llama-cli \
  -m /root/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf \
  -ngl 99 -c 2048 -p "Hello" -n 32 --temp 0 -e \
  --single-turn --simple-io --load-mode none
```

Headless runs require `setsid` + `--simple-io` (dead-PTY hang otherwise),
`--single-turn` (v0.2.0 defaults to interactive chat), and `--load-mode none`
(avoids mmap stalls on `/mnt/*`). Pass = exit 0, full offload lines, no CPU buffer lines.

## Running the benchmark harness

To run a full guarded benchmark session from the repository root:

```bash
# In WSL2 guest:
python3 benchmarks/bin/run_session.py --tiers 4096 8192 16384 32768 --repeats 5 --delay 10
```

## License

No project license has been chosen yet. The base model `Qwen/Qwen3.8-27B` is
Apache-2.0; see `models/README.md` for artifact provenance.
