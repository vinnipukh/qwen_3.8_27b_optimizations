# Phase 1: Environment Validation & Stock Baseline - Research

**Researched:** 2026-08-22
**Domain:** WSL2 ROCm/HIP toolchain · llama.cpp gfx1100 build · model artifact logistics
**Confidence:** HIGH (all load-bearing facts verified live today)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- D-01: root-only Ubuntu 24.04 (no named user)
- D-02: `.wslconfig` tuning only when evidence demands (watch ROCm#6022-class RAM-clamp bug)
- D-03: librocdxg quickstart route, ROCm pinned EXACTLY 7.2.1
- D-04: Adrenalin auto-updates paused; frozen pairing recorded (26.10.41 / 32.0.31041.1004 — verified present on host today)
- D-05: pin newest llama.cpp release with ≥b8394 lineage (owner delegated choice)
- D-06: baseline binaries inside repo while total < 750 MB
- D-07: superseded by owner instruction 2026-08-22 — orchestrator performs the download during execution ("install the model or whatever you need in wsl"); storage/sha256 rules unchanged

### Claude's Discretion
- Exact release tag (within ≥b8394 lineage) → resolved below
- Build flags beyond `-DGGML_HIP=ON -DGPU_TARGETS=gfx1100`, Release, Ninja+ccache
- `.wslconfig` values if tuning becomes necessary

### Deferred Ideas
- none
</user_constraints>

## Q1 — llama.cpp pin (HIGH)

- **Newest release tag: `v0.2.0`** — published 2026-08-21T18:32Z, commit `bb4caa7540188872173c44d161602d9271386413` (GitHub API, checked 2026-08-22).
- Lineage check: b8394 = commit `3a5cb629` dated **2026-03-17T13:27Z** (verified via tags/ref API); PR #20518 merge confirmed ~2026-03-17 by researcher run before it failed. v0.2.0 is ~5 months downstream ⇒ constraint satisfied.
- **RECOMMENDED PIN: tag `v0.2.0`, commit `bb4caa75`.** Note: llama.cpp moved to semver release scheme; v0.2.0 is fresh (<24h old at research time) — if build breaks mysteriously, fallback pin = newest `b8xxx`-era tag ≥ b850 (record any deviation + rationale in environment fingerprint).

## Q2 — ROCm 7.2.1 exact install route (HIGH)

Verified against AMD docs scrape (06-amd-wsl-howto.md) + live repo/API checks:

1. Host driver already installed: Adrenalin 26.10.41 (`32.0.31041.1004`) — verified via WMI today. No host work needed.
2. Guest repo setup (Ubuntu 24.04 "noble", root):
   `https://repo.radeon.com/amdgpu-install/7.2.1/ubuntu/noble/amdgpu-install_7.2.1.70201-1_all.deb`
   → `apt install ./amdgpu-install_7.2.1.70201-1_all.deb` then `amdgpu-install --usecase=wsl,rocm --no-dkms` (wsl usecase = no kernel/dkms components).
   Newer 7.2.2–7.2.4 exist upstream — deliberately NOT used (D-03 freeze).
3. librocdxg: pre-built debs from **v1.2.2** release:
   - `https://github.com/ROCm/librocdxg/releases/download/v1.2.2/rocdxg-roct_1.2.2_amd64.deb`
   - `https://github.com/ROCm/librocdxg/releases/download/v1.2.2/rocdxg-amd-smi-lib_1.2.2_amd64.deb`
   (`dpkg -i`; Option B of quickstart — no source build, no Windows SDK needed.)
4. **ENV VAR REQUIRED for 7.2.x-era ROCk:** `export HSA_ENABLE_DXG_DETECTION=1` (requirement removed only in newer ROCk; set it system-wide e.g. `/etc/profile.d/rocdxg.sh`).
5. Verification: `rocminfo | grep -i gfx` must list Agent `gfx1100`. `/dev/dxg` already present (verified today).

## Q3 — llama.cpp build prerequisites (HIGH)

- apt: `build-essential cmake ninja-build ccache git libcurl4-openssl-dev pkg-config`
- ROCm runtime+devel via amdgpu-install usecase above (provides hipcc, hipBLASLt, rocBLAS)
- Configure exactly per roadmap: `-G Ninja -DGGML_HIP=ON -DGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release`; add `-DLLAMA_CURL=ON` if libcurl present (default ON at modern pins; harmless either way)
- Targets to produce: `llama-cli`, `llama-bench`, `llama-perplexity`, `test-backend-ops` (all built by default `cmake --build`)
- Known gotcha class (from scrapes 01/02): RDNA3 wave-size assumptions and fa-on/off behavior differences are Phase-3 concerns, not build blockers.

## Q4 — Artifact availability & identity (HIGH — exact match)

- Repo `JonathanColetti/Qwen3.8-27B-Uncensored-GGUF` is PUBLIC, ungated (401 earlier was wrong repo-id with `.gguf` suffix; correct id verified).
- File `Qwen3.8-27B-Uncensored-IQ4_XS.gguf`: size **15,309,039,008 B**, LFS sha256 **`53adc4bbed67044d662273356bbf3a50fdec667ac21bbf18d13e5815fbccc7f5`** — byte-exact match to MODEL-DECISION.md lock record.
- Download URL: `https://huggingface.co/JonathanColetti/Qwen3.8-27B-Uncensored-GGUF/resolve/main/Qwen3.8-27B-Uncensored-IQ4_XS.gguf`
- HF metadata corroborates lock record: arch `qwen35`, ctx 262144, MTP/imatrix variants present, repo rev `dee0a316`.
- Storage: `models/` in repo working tree (gitignored via `models/*.gguf`). Download on Windows side to `E:\...\models\` then verify sha256 in guest via `/mnt/e/...` — avoids 15 GB through the WSL disk image and sidesteps the 15 GB WSL RAM ceiling for hashing (host RAM is fine for streaming hash).
- Bonus artifacts available in same repo if ever needed: noMTP-IQ4_XS (15.08 GB), draft-Q8_0 (3.16 GB, for later spec-decode v2), imatrix.dat.

## Q5 — Empirical free-VRAM probe method (MEDIUM-HIGH)

DXG-reported free VRAM overstates usable by ~1.5–3 GB (scrape 05-dxg-vram-loss.md measured −2.9 GB on XTX sibling). Ground-truth sequence for ENV gates:
1. Guest: `rocminfo` (agent present) + llama.cpp startup log line reporting computed VRAM budget (`ggml_backend_cuda... buffer size` / `VRAM 0/x MiB` lines).
2. Escalating `-ngl` + fixed small `-c` runs watching for OOM vs full residency; record largest fully-resident config.
3. Cross-check Windows side via `nvidia-smi`-equivalent: `Get-Counter`/HWiNFO CSV export (per BENCH-02 telemetry rule; rocm-smi non-functional under ROCDXG).
4. Record measured numbers — never DXG-reported — into `benchmarks/environment/vram-probe.txt`.

## Sources

- GitHub API: releases/tags/commits for ggml-org/llama.cpp (2026-08-22)
- https://repo.radeon.com/amdgpu-install/7.2.1/ubuntu/noble/ (2026-08-22)
- https://api.github.com/repos/ROCm/librocdxg/releases (v1.2.2, 2026-08-22)
- https://huggingface.co/api/models/JonathanColetti/Qwen3.8-27B-Uncensored-GGUF?blobs=true (2026-08-22)
- Local scrapes 01–07 under .planning/research/deep-research/raw/scrapes/

## Research note (process)

Primary researcher subagent was killed twice by upstream provider 503s mid-run; questions were
completed directly by the orchestrator using the same sources. Planner/checker role separation
is unaffected (both remain independent spawned agents).
