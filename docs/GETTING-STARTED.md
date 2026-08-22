<!-- generated-by: gsd-doc-writer -->
# Getting Started

Setup path from a bare Windows machine to the first GPU-resident generation of
Qwen3.8-27B IQ4_XS on RX 7900 XT. Two routes exist: restore the frozen Phase 1
snapshot (fast, exact) or install manually (slow, documented in phase summaries).

## Prerequisites

| Component | Version / requirement |
|---|---|
| Windows | Windows 11 with WSL2 enabled |
| GPU | AMD RX 7900 XT (gfx1100) |
| Windows driver | Adrenalin 26.10.41 (32.0.31041.1004) — **frozen**; block silent updates per D-04 amendment (commands in `benchmarks/environment/versions.txt`) |
| Guest OS | Ubuntu 24.04, root account only |
| ROCm | 7.2.1 + librocdxg 1.2.2 |
| Host RAM | ≥ 32 GB physical (28 GB must be assignable to WSL2) |
| Disk | ~70 GB free (49.4 GB snapshot OR ROCm install + 15.31 GB model + build tree) |

**Required:** `%UserProfile%\.wslconfig` must contain:

```ini
[wsl2]
memory=28GB
```

With less guest RAM, VRAM allocation fails (`dxgkio_create_allocation: -12`
in dmesg). After editing: `wsl --shutdown`.

## Option A: Restore the frozen snapshot

Phase 1's validated environment (ROCm 7.2.1, librocdxg, pinned llama.cpp build
tree, `/etc/profile.d/rocdxg.sh`) was exported to
`E:\wsl-snapshots\ubuntu-2404-rocm721-phase1.tar`:

```bash
wsl --import qwen-opt E:\wsl\qwen-opt E:\wsl-snapshots\ubuntu-2404-rocm721-phase1.tar
wsl -d qwen-opt
```

Verify inside the guest: `rocminfo | grep gfx1100` lists the device.
Environment details: `.planning/phases/01-environment-validation-stock-baseline/01-01-SUMMARY.md`.

## Option B: Manual install

Follow the exact commands recorded in
`.planning/phases/01-environment-validation-stock-baseline/01-01-SUMMARY.md`.
Key steps:

1. Install pinned ROCm: `amdgpu-install_7.2.1.70201-1_all.deb` with
   `--usecase=rocm --no-dkms` (the `wsl` usecase does not exist in 30.30.x).
2. Install librocdxg v1.2.2 (roct + amd-smi-lib debs).
3. Create `/etc/profile.d/rocdxg.sh` setting `HSA_ENABLE_DXG_DETECTION=1`
   and the ROCm PATH; re-login or `source` it.

## Get the model

Download from Hugging Face, then verify the checksum against the locked value
in `models/README.md`:

```bash
curl -L -o models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf \
  https://huggingface.co/JonathanColetti/Qwen3.8-27B-Uncensored-GGUF/resolve/main/Qwen3.8-27B-Uncensored-IQ4_XS.gguf

echo "53adc4bbed67044d662273356bbf3a50fdec667ac21bbf18d13e5815fbccc7f5  models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf" | sha256sum -c -
```

Expected: `models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf: OK` (15,309,039,008 bytes).
Then copy it to the guest-local canonical location:

```bash
mkdir -p /root/models && cp /mnt/e/Projects/qwen_3.8_27b_optimizations/models/*.gguf /root/models/
```

Loading from `/mnt/*` can stall under mmap — always use the `/root/models/` copy.

## Build llama.cpp (pinned)

Clone into guest ext4, not `/mnt/*` (DrvFs breaks git lock-files):

```bash
cd /root/llama.cpp   # already present in the snapshot at commit bb4caa75
cmake -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1100 \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF
cmake --build build --config Release -j$(nproc)
```

Sanity check: `build/bin/test-backend-ops` passes on the ROCm backend
(reference log: `benchmarks/environment/test-backend-ops-phase1.txt`).

## First generation

```bash
setsid /root/llama.cpp/build/bin/llama-cli \
  -m /root/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf \
  -ngl 99 -c 2048 -p "Hello" -n 32 --temp 0 -e \
  --single-turn --simple-io --no-mmap
```

Pass criteria: exit code 0, all layers assigned to ROCm0, no CPU-buffer lines,
coherent output. Reference run: `benchmarks/environment/startup-log.txt`
(pp 111.5 / tg 33.5 tok/s). Flags explained: `setsid` + `--simple-io` avoid the
headless PTY hang; `--single-turn` because v0.2.0 defaults to interactive chat;
`--no-mmap` avoids mmap stalls.

## Next steps

- `docs/DEVELOPMENT.md` — build commands and conventions (when available)
- `.planning/ROADMAP.md` — phases 2–6: harness, gates/profiling, kernel work
