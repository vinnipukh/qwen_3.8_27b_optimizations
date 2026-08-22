<!-- generated-by: gsd-doc-writer -->

# Development

Development happens in the WSL2 guest (root-only Ubuntu 24.04) against the pinned
llama.cpp tree at `/root/llama.cpp`. Windows-side tooling drives the guest via `wsl.exe`.

## WSL2 gotchas (learned in Phase 1)

| Gotcha | Symptom | Fix |
|---|---|---|
| DrvFs git locks | `git` index-lock operations fail on `/mnt/e` | Keep source trees on guest ext4 (`/root/llama.cpp`) |
| PTY hang | `llama-cli` blocks in `n_tty_write` on dead PTY | Headless runs: `setsid --simple-io --single-turn --no-mmap` |
| mmap stall over `/mnt/e` | Model load from DrvFs stalls | Use canonical copy at `/root/models/` |
| Guest RAM too low | DXG ENOMEM at VRAM alloc (`dxgkio_create_allocation: -12`) | `.wslconfig` `memory=28GB` (REQUIRED; ~27 GB visible) |
| npm via Windows interop | WebUI build fails through interop | Server/UI disabled at build time; HF dist fallback |
| Env not sourced | HIP ops see CPU only | Source `/etc/profile.d/rocdxg.sh` (`HSA_ENABLE_DXG_DETECTION=1`) before runs |

Also: Git-bash mangles `$VARS` passed through `wsl.exe` args — write scripts to guest
files and execute them instead of inlining shell with variables.

## Rebuild llama.cpp from pin

Pin record: `benchmarks/environment/llamacpp-pin.txt` — tag `v0.2.0`, commit
`bb4caa7540188872173c44d161602d9271386413`. Never rebuild over the archived stock
binaries in `baseline/binaries/v0.2.0-bb4caa75/`; build to a separate directory.

```bash
# inside WSL2, as root
source /etc/profile.d/rocdxg.sh
cd /root/llama.cpp && git checkout bb4caa7540188872173c44d161602d9271386413
cmake -B build -G Ninja \
  -DGGML_HIP=ON -DGPU_TARGETS=gfx1100 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF
cmake --build build --target llama-cli llama-bench llama-perplexity test-backend-ops
```

Record any deviation from these flags plus compiler versions in the result notes.

## Re-run correctness gates

Three gates from Phase 1, re-run after every environment or kernel change:

1. **HIP smoke** — compile and run `benchmarks/environment/hipsmoke.cpp`; stdout must
   contain `gfx1100`:

   ```bash
   hipcc benchmarks/environment/hipsmoke.cpp -o /tmp/hipsmoke && /tmp/hipsmoke | grep gfx1100
   ```

2. **Backend op tests** — full numerical validation of the GGML backends:

   ```bash
   ./build/bin/test-backend-ops
   ```

   Reference output for the stock pin: `benchmarks/environment/test-backend-ops-phase1.txt`.

3. **Runtime gate script pattern** — headless generation must produce coherent tokens
   with all tensors on GPU (startup log shows zero CPU fallback):

   ```bash
   setsid /root/llama.cpp/build/bin/llama-cli \
     -m /root/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf \
     -ngl 99 --simple-io --single-turn --no-mmap \
     -p "prompt" -n 64 2>&1 | tee logs/run-$(date -u +%Y%m%dT%H%M%SZ).log
   ```

   Check the log: device assignment lines list `gfx1100`, no fallback warnings.

## Benchmarking discipline

- Prefill (M≫1) and decode (M≈1) are measured separately — blended tok/s is banned.
- One change per benchmark run; always compare against the archived stock baseline.
- Record ROCm/driver/toolchain versions with every result (`benchmarks/environment/versions.txt`
  is the template).

## Planning workflow

Work is planned and executed phase-by-phase via the GSD commands:
`/gsd-plan-phase N` produces the phase plan(s) under `.planning/phases/NN-*/`;
`/gsd-execute-phase N` executes them. Current state lives in `.planning/STATE.md`.
The roadmap and binding methodology rules are in `.planning/ROADMAP.md`.

## Commit discipline

- **Atomic commits** — one logical change (one kernel, one gate fix, one harness change)
  per commit.
- **Evidence-carrying messages** — include what changed, the benchmark numbers
  (prefill/tg split), gate results, and versions. Failed experiments get committed and
  documented too; publishing failures is a project rule.
- The model GGUF and stock binaries stay gitignored; provenance lives in
  `models/README.md`, fingerprints in `benchmarks/environment/`.
