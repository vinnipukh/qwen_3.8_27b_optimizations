<!-- generated-by: gsd-doc-writer -->

# Configuration Reference

Every knob that matters for reproducing this project's environment. All values below are verified
against the frozen Phase 1 environment.

## Windows side: `.wslconfig`

`C:\Users\<user>\.wslconfig`:

```ini
[wsl2]
memory=28GB
swap=16GB
```

**Required.** With the default ~15 GB guest RAM allocation, VRAM allocation fails with
DXG ENOMEM (`dmesg`: `dxgkio_create_allocation: -12`). At 28 GB the guest sees 27 GB and
full-model GPU residency succeeds (132/132 tensor layers offloaded). Apply, then run
`wsl --shutdown` before restarting the distro.

## Guest environment

| Setting | Location | Purpose |
|---|---|---|
| `HSA_ENABLE_DXG_DETECTION=1` | `/etc/profile.d/rocdxg.sh` | Required for ROCr to enumerate the GPU through the WSL2 DXG path |

The distro is root-only Ubuntu 24.04; everything runs as root in the guest.

## llama.cpp build flags

Pin: **v0.2.0 @ bb4caa7540188872173c44d161602d9271386413**. Source tree lives guest-side at
`/root/llama.cpp` (DrvFs has a git-lock incompatibility; do not build from `/mnt/e`).

```bash
cmake -B build -G Ninja \
  -DGGML_HIP=ON \
  -DGPU_TARGETS=gfx1100 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_CURL=OFF
```

| Flag | Why |
|---|---|
| `-DGGML_HIP=ON` | HIP backend (ROCm) instead of CUDA |
| `-DGPU_TARGETS=gfx1100` | RX 7900 XT codegen only |
| `-DLLAMA_BUILD_SERVER=OFF` | Server unused; smaller build surface |
| `-DLLAMA_CURL=OFF` | No network dependency in binaries |

Compiler: gcc 13.3.0 / hipcc 7.2.53211-e1a6bc5663. Note: `amdgpu-install` usecase `wsl` is
invalid in the 30.30.x build — use `--usecase=rocm --no-dkms`.

## Runtime flags

Headless interactive runs need process isolation flags; omitting them hangs waiting on TTY input:

```bash
setsid llama-cli -m /root/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf \
  -ngl 99 -c 2048 -p 'Hello' -n 32 --temp 0 \
  --no-mmap --simple-io --single-turn -t <threads>
```

| Flag | Purpose |
|---|---|
| `-ngl 99` | Offload all layers to GPU (target: fully resident) |
| `-c N` | Context size — **set explicitly, always**; OOM arrives at first long prompt, not load |
| `--no-mmap` | Avoid mmap stalls; canonical model copy is guest-side `/root/models/` (mmap over `/mnt/e` DrvFs stalls) |
| `--simple-io` | Required for headless/non-TTY execution |
| `--single-turn` | One prompt, one completion — no interactive loop |
| `-t <threads>` | CPU thread count (note: ROCr busy-spin under WSL2 consumes ~2 cores per GPU context; consider taskset isolation for decode-latency runs) |

## Frozen versions (Phase 1)

Do not upgrade any component silently — see D-04 below.

| Component | Version |
|---|---|
| ROCm (guest) | 7.2.1 |
| Windows driver | 32.0.31041.1004 (Adrenalin 26.10.41) |
| librocdxg | 1.2.2 |
| llama.cpp | v0.2.0 @ bb4caa7540188872173c44d161602d9271386413 |
| hipcc | 7.2.53211-e1a6bc5663 |
| OS | Ubuntu 24.04 (guest, root-only), WSL2 host |

Full fingerprint files: `benchmarks/environment/` (versions.txt, hipconfig.txt, rocminfo.txt).

## D-04 update policy

No silent driver updates. Scope (as amended): prevent *silent* updates so the ROCm/driver
pairing stays frozen; notification-only behavior is acceptable. Sanctioned mechanism (**PENDING — requires an elevated shell, owner action**; not yet applied):

```powershell
reg add HKLM\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate /v ExcludeWUDriversInQualityUpdate /t REG_DWORD /d 1 /f
```

Detection net if drift occurs anyway: every benchmark row carries a driver fingerprint, and the
environment version gates are re-run on any detected mismatch.

## Snapshot restore procedure

Frozen post-validation image: `E:\wsl-snapshots\ubuntu-2404-rocm721-phase1.tar`
(49 GB — includes ROCm 7.2.1, librocdxg 1.2.2, and the pinned llama.cpp build tree).

Restore into a fresh distro:

```powershell
wsl --import <DistroName> <InstallPath> E:\wsl-snapshots\ubuntu-2404-rocm721-phase1.tar
wsl -d <DistroName>
```

After restore, verify the pairing before trusting any run: re-run the environment checks
(`benchmarks/environment/hipsmoke.cpp`, `rocminfo.txt` comparison, `test-backend-ops`).
