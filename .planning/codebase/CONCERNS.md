# Codebase Concerns

**Analysis Date:** 2026-08-25

## Tech Debt

**Hardcoded absolute paths scattered across modules (no central config):**
- Issue: Guest-side binary/model paths and host user paths are duplicated as module-level constants in at least 8 files instead of one config module or env-var contract.
- Files: `benchmarks/lib/llabench.py` (`BIN_PATH = "/root/llama.cpp/build-ci/bin/llama-bench"`, `MODEL_PATH = "/root/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf"`), `benchmarks/lib/fingerprint.py:19-21` (including `DEFAULT_WSLCONFIG = "/mnt/c/Users/arhan/.wslconfig"`, which embeds a username), `benchmarks/bin/profile_workload.py:24,93`, `benchmarks/bin/run_model_gate.py:22-24`, `benchmarks/bin/run_op_gate.py:21`, `benchmarks/bin/run_prompts.py:21-22`, `tools/ask_model.py:9-14`.
- Impact: Any change of guest model path, binary location, or Windows username breaks multiple harness entry points silently or via confusing failures; blocks portability beyond this single WSL2 guest.
- Fix approach: Introduce a single `benchmarks/lib/config.py` (env-overridable: `LLAMA_BIN`, `MODEL_PATH`, etc.) and make all CLIs import from it; keep current values as defaults.

**Duplicated harness/session logic for the Vulkan backend:**
- Issue: `benchmarks/vulkan/run_session_vulkan.py` reimplements the tier/cell/guard session loop that `benchmarks/bin/run_session.py` owns, importing only the `benchmarks/lib/*` contracts. The two session loops can drift (guard evaluation, fail-row emission, PID-file handling).
- Files: `benchmarks/vulkan/run_session_vulkan.py` (244 lines) vs `benchmarks/bin/run_session.py` (342 lines).
- Impact: Bug fixes applied to the HIP session loop will not reach the Vulkan path (or vice versa); the Vulkan path also hardcodes `DEFAULT_VULKAN_BIN = r"E:\vulkan-arm\llama.cpp\build\bin\Release\llama-bench.exe"`, tying it to one machine.
- Fix approach: Extract the shared session skeleton into `benchmarks/lib/session.py` parameterized by backend (binary path, env, arm label), or explicitly mark the Vulkan driver as frozen/archived if it is no longer part of the plan.

**Stub function left in fixture tooling:**
- Issue: `dequant_row_cpu()` in `tools/dump_matmul_fixtures.py` (~line 62) has a docstring and a bare `pass` body — it silently returns `None`. Any caller would produce empty/None references.
- Files: `tools/dump_matmul_fixtures.py`
- Impact: Latent trap for future maintainers extending fixture generation; currently appears uncalled, but it sits directly next to reference-generation code where silent `None` is dangerous.
- Fix approach: Delete the stub or implement it against `gguf.quants.dequantize`; add a guard assertion so `y_ref` generation cannot emit `None`.

**Silent exception swallowing in environment fingerprinting and thresholds:**
- Issue: Five broad `except Exception:` blocks in `benchmarks/lib/fingerprint.py` (lines ~67, 105, 123, 136, 214) fall back to placeholder values (`"unknown"`, host time defaults) with no warning output. `guard.Thresholds.from_json` (`benchmarks/lib/guard.py:44-56`) returns `None` on *any* parse error, silently downgrading runs to observe-only mode.
- Files: `benchmarks/lib/fingerprint.py`, `benchmarks/lib/guard.py`
- Impact: A malformed `benchmarks/config/thresholds.json` produces unguarded benchmark sessions with no signal that protection was disabled; fingerprints can be incomplete without anyone noticing, weakening reproducibility records.
- Fix approach: Log a stderr warning on every fallback; distinguish "file absent" (legitimate observe-only) from "file corrupt" (should fail loudly or abort).

**Root-level compiler artifacts not covered by `.gitignore`:**
- Issue: HIP build outputs from WMMA experiments sit in the repo root and are not ignored: `test_wmma-hip-amdgcn-amd-amdhsa-gfx1100.{bc,hipi,o,s,out}`, `test_wmma-host-x86_64-unknown-linux-gnu.{bc,hipi,s}`, `test_wmma.hip-hip-amdgcn-amd-amdhsa.hipfb` (two ~2.4 MB `.hipi` blobs). Also stray directories `.rocprofv3/`, `scrape_out/`, and untracked result dirs under `benchmarks/results/kernels_*` accumulate.
- Files: repo root `test_wmma*`, `.rocprofv3/`, `.gitignore`
- Impact: Risk of accidentally committing multi-megabyte binary artifacts; clutters `git status` (currently 30+ dirty entries including untracked phase docs, `CONTRIBUTING.md`, `LICENSE`).
- Fix approach: Add `test_wmma*`, `.rocprofv3/`, `scrape_out/` patterns to `.gitignore`, move experiment artifacts out of the root, and commit the pending doc/test work.

## Known Bugs

**No process-level timeout inside the harness itself (relies on external watchdog):**
- Symptoms: `benchmarks/bin/run_session.py` uses `subprocess.Popen(...)` followed by an unconditional `proc.wait()` (~line 174-186) with no timeout parameter. A hung `llama-bench` (the known dead-PTY / DXG ENOMEM hang class documented in `AGENTS.md`) blocks the session indefinitely unless the separate `benchmarks/host/thermal_watchdog.py` daemon is running.
- Files: `benchmarks/bin/run_session.py`, `benchmarks/host/thermal_watchdog.py`
- Trigger: GPU/DXG pipeline hang during any tier invocation while the watchdog is not active.
- Workaround: Always launch `thermal_watchdog.py` alongside sessions (per `benchmarks/RUNBOOK.md`); AGENTS.md mandates timeouts for interactive commands but the harness code does not enforce one itself.

**WMMA kernel lane mapping is admitted demonstration-grade:**
- Symptoms: `gemm_iq4xs_wmma_kernel` in `kernels/matmul_iq4xs/impl_gemm_wmma.hip` contains self-documented approximations: lane→element mapping uses `elem % 256` wraparound ("we map 256 with duplication"), B-fragment LDS indexing carries confused inline comments ("actually warp_m offset already in block_m") plus a dead `(void)bv` load, and comments concede "This is technically not lane-replicated correctly for optimal throughput but functional."
- Files: `kernels/matmul_iq4xs/impl_gemm_wmma.hip` (WMMA section, ~lines 100-230)
- Trigger: Shapes meeting the launcher gate `M >= 512 && N >= 1024 && M%16==0 && N%16==0 && K%16==0` route to the WMMA path (`kernels/matmul_iq4xs/impl_gemm_wmma.hip`, `gemm_iq4xs_wmma_gpu`).
- Workaround: Host launcher falls back to the tiled kernel on launch/sync failure, and small/medium shapes stay on the well-verified tiled path; correctness gates (cosine ≥ 0.999) still apply downstream.

**Potential misaligned 128-bit load in GEMV/tiled kernels for odd super-block counts:**
- Symptoms: Both `impl_gemv_gfx1100.hip` and the tiled GEMM reinterpret `blk->qs + ib * 16` as `uint4` (128-bit load). With `#pragma pack(1)` the block struct is 136 bytes, so a row's base is aligned to 16 bytes only when `blocks_per_row * 136 ≡ 0 (mod 16)`, i.e., when `K/256` is even. All current fixtures use even counts (e.g., K=5120 → 20 blocks), but an odd `blocks_per_row` shape would produce a misaligned vector load (fault or wrong results depending on target).
- Files: `kernels/matmul_iq4xs/impl_gemv_gfx1100.hip`, `kernels/matmul_iq4xs/impl_gemm_wmma.hip`, `kernels/common/block_iq4_xs.h` (136-byte packed layout)
- Trigger: Running the kernels on a weight matrix whose K is an odd multiple of 256.
- Workaround: None in code; restrict shapes to even `K/256` (all canonical shapes in `tools/dump_matmul_fixtures.py` satisfy this) or add a runtime alignment assert in the host launchers.

**Canary extraction regex couples to model chat-template markers:**
- Symptoms: `extract_generation()` in `benchmarks/bin/run_model_gate.py` greps generations between `[Start thinking]` and `[ Prompt:`/`Exiting` markers specific to this model's output format; on mismatch it falls back to the whole stdout strip.
- Files: `benchmarks/bin/run_model_gate.py` (~line 50)
- Trigger: Swapping models or llama.cpp versions changing exit-log wording silently changes canary comparison input.
- Workaround: None; golden canaries (`benchmarks/golden/stock_baseline_golden.json`) were captured against this exact pairing.

## Security Considerations

**Cross-boundary process kill surface:**
- Risk: `benchmarks/host/thermal_watchdog.py` builds `wsl.exe`/`taskkill` kill commands targeting PIDs; a wrong PID could kill an unrelated process.
- Files: `benchmarks/host/thermal_watchdog.py` (`build_kill_command`)
- Current mitigation: PID is strictly integer-validated and must be positive; PID file written/read locally by the session loop (`benchmarks/bin/run_session.py` writes `<run_dir>/run/<pid>`).
- Recommendations: Record the killed process's argv at spawn time and verify identity before killing; log every kill action to the run store.

**Host interop via PowerShell subprocess calls:**
- Risk: `benchmarks/lib/fingerprint.py` and `benchmarks/lib/toast.py` shell out to `powershell.exe` for clock/driver/toast operations. Injection risk is low today (fixed argument vectors, no string interpolation of user data), but any future interpolation would create a command-injection path.
- Files: `benchmarks/lib/fingerprint.py`, `benchmarks/lib/toast.py`
- Current mitigation: Fixed argv lists with `capture_output=True` and 5 s timeouts.
- Recommendations: Keep argument vectors static; never format strings into PowerShell `-Command`.

**Secrets posture:**
- Risk: None observed — no `.env` files exist, no credentials in tracked sources, model artifact provenance is hash-recorded rather than credentialed.
- Files: `models/README.md` (sha256 record), `.gitignore`
- Current mitigation: Large binaries and toolchains gitignored (`freetoken-rocm-probe/tools/`, `baseline/binaries/`, `models/*.gguf`).
- Recommendations: Keep it that way; if HF tokens are ever added for downloads, route through env vars excluded from the run-store manifests (which currently snapshot full environment).

**License ambiguity resolved inconsistently:**
- Risk: `README.md` states "No project license has been chosen yet" while an untracked `LICENSE` file (12 KB, looks like GPL-style given llama.cpp lineage) now exists in the working tree.
- Files: `README.md`, `LICENSE`
- Current mitigation: Vendored `kernels/common/block_iq4_xs.h` carries Apache-2.0 attribution comment.
- Recommendations: Reconcile README and LICENSE; verify the vendored header's license compatibility with the chosen project license before publication (Phase 6).

## Performance Bottlenecks

**FP64 accumulation in custom kernels on a consumer RDNA3 card:**
- Problem: GEMV and tiled-GEMM kernels accumulate in `double` (`thread_sum`, `acc[16]`) for CPU-oracle parity.
- Files: `kernels/matmul_iq4xs/impl_gemv_gfx1100.hip`, `kernels/matmul_iq4xs/impl_gemm_wmma.hip` (tiled fallback)
- Cause: gfx1100 FP64 throughput is 1/16 of FP32; double accumulators also double VGPR pressure against the stated ≤96 VGPR budget.
- Improvement path: Match stock numerics with float accumulation + pairwise/Kahan splitting only if the cosine ≥ 0.999 / max_rel ≤ 1e-3 gate allows; measure before/after per project rules.

**Tiled GEMM fallback uses scalar global loads for activations:**
- Problem: The X-tile loads in the tiled path are scalar per-element reads relying on L1 ("For this fallback we keep simple and rely on L1 spatial reuse").
- Files: `kernels/matmul_iq4xs/impl_gemm_wmma.hip` (tiled kernel inner loop)
- Cause: No LDS staging of X tiles on the fallback path, despite the WMMA path demonstrating LDS buffering.
- Improvement path: Stage the X tile in LDS once per block (as the WMMA path sketches) to multiply weight-dequant reuse across rows.

**WMMA path reloads the B tile from global memory every K-step without real double buffering:**
- Problem: Comments claim LDS double-buffering, but the implementation performs a synchronous cooperative load followed by `__syncthreads()` twice per K-tile with `ping ^= 1` never overlapping anything.
- Files: `kernels/matmul_iq4xs/impl_gemm_wmma.hip` (WMMA main loop)
- Cause: Simplified demonstrator implementation.
- Improvement path: Implement genuine ping-pong prefetch (load K-tile t+1 during compute of t) before claiming double-buffer wins in publication.

## Fragile Areas

**Log-format parsing of the pinned llama.cpp binary:**
- Files: `benchmarks/lib/preflight.py` (`parse_free_mib`, `parse_buffer_lines` regexes), `benchmarks/lib/llabench.py` (JSONL output parsing, `MatrixContaminationError` matrix assertions), `benchmarks/bin/run_model_gate.py` (PPL stderr regex, generation extraction)
- Why fragile: Every parser is coupled to exact upstream v0.2.0 (`bb4caa75`) log/output wording; any re-pin or rebuild with different flags breaks them silently (regex non-match → zero values or fallbacks).
- Safe modification: Add golden-fixture unit tests per parser (some exist in `benchmarks/tests/test_preflight.py`, `test_llabench_wrapper.py` — extend them whenever touching parsing) and fail loudly on unparsed-but-present buffer lines.
- Test coverage: Partial — happy paths covered; unmatched-format fallbacks mostly untested.

**HWiNFO Shared Memory v2 binary decoding:**
- Files: `benchmarks/host/hwinfo_daemon.py` (hand-packed structs `HEADER_FMT`, `READING_FMT`, fixed sensor-label list in `benchmarks/config/hwinfo_sensor_labels.txt`)
- Why fragile: Depends on exact HWiNFO version's SM2 layout and sensor naming; a HWiNFO update shifts offsets or labels and telemetry degrades to missing mandatory fields.
- Safe modification: Validate signature/version fields before interpreting (partially done via `HWIS_SIG`/`DEAD_SIG`); keep the CSV fallback path exercised.
- Test coverage: `benchmarks/tests/` includes decoding tests for pure functions, but live-layout drift is only detectable at runtime.

**Guest-side state outside version control:**
- Files: `/root/llama.cpp` (pinned source + `build-ci` binaries), `/root/models/*.gguf` — both referenced by hardcoded paths throughout `benchmarks/`
- Why fragile: The entire benchmark baseline lives in WSL guest ext4, recoverable only via the 49.4 GB snapshot tar (`E:\wsl-snapshots\ubuntu-2404-rocm721-phase1.tar`). Losing the guest or snapshot invalidates every recorded baseline.
- Safe modification: Never rebuild the pinned binary without re-running the golden/model gates (`benchmarks/bin/run_model_gate.py`, `run_op_gate.py`); keep the snapshot archived.
- Test coverage: `benchmarks/lib/fingerprint.py` records commit hashes in manifests, which detects drift after the fact.

**Tensor-name candidate mapping in fixture extraction:**
- Files: `tools/dump_matmul_fixtures.py` (`TENSOR_CANDIDATES` — e.g., `attn_v` falls back through `"blk.0.ssm_out.weight"`)
- Why fragile: First-hit-wins heuristics over GGUF tensor names can silently bind a canonical shape name to a semantically different tensor when re-running against another model revision.
- Safe modification: Assert extracted tensor dtype/shape exactly matches the canonical (K, N) and IQ4_XS type before writing fixtures; keep `kernels/fixtures/manifest_matmul.json` checksums authoritative.

## Scaling Limits

**VRAM ceiling at 32768 context:**
- Current capacity: 20 GB card; weights ~14.6 GiB (`WEIGHTS_BYTES` in `benchmarks/lib/preflight.py`); 4096–16384 tiers pass with margin 1.05.
- Limit: All 32768-tier cells return `FAILED:preflight-oom` (published in `benchmarks/results/BASELINE-MATRIX.md`); KV cache at 64 KiB/token f16 plus compute buffers exceed free VRAM.
- Scaling path: Quantized KV cache, lower `-ngl`, or a larger card — all out of scope for the current frozen-environment methodology; treat 32768 as permanently out-of-matrix.

**Harness single-run serialization:**
- Current capacity: One guarded session at a time (`benchmarks/results/.session.lock`), RSS polling at 1 Hz per process.
- Limit: Parallel sessions would contend for VRAM and corrupt thermal/shared-memory guard signals.
- Scaling path: Not needed for the single-GPU project scope; do not remove the lock.

## Dependencies at Risk

**Pinned llama.cpp v0.2.0 @ `bb4caa75`:**
- Risk: Ancient pin; upstream log formats, bench flags (`--delay`, `-pg`, `-fa`), and quant kernels evolve. The provisional integration patch (`patches/phase5_mul_mat_custom.patch`, gated behind `GGML_CUDA_ENABLE_CUSTOM_GFX1100`) applies only to this commit.
- Impact: Any rebase invalidates parsers, the patch, and the published baseline matrix comparability.
- Migration plan: If re-pinning becomes necessary, re-run the full gate chain (op gate, PPL gate, golden canaries) and republish `BASELINE-MATRIX.md`; treat old results as superseded, not merged.

**ROCm 7.2.1 over WSL2 DXG stack:**
- Risk: Known fragility documented in `AGENTS.md`/`README.md` — DXG ENOMEM below 28 GB host RAM assignment, dead-PTY hangs, driver updates forbidden by policy.
- Impact: Environment-level hangs that no in-repo test can catch; the frozen WSL snapshot is the sole recovery artifact.
- Migration plan: Keep driver/Adrenalin versions locked; validate the snapshot restores correctly before Phase 6 publication.

**Optional `gguf-py` import:**
- Risk: `tools/dump_matmul_fixtures.py` degrades to `HAS_GGUF = False` with only a stderr warning; fixture regeneration silently skips real-tensor extraction.
- Files: `tools/dump_matmul_fixtures.py` (top-of-file try/import)
- Impact: Regenerated fixture manifests could end up synthetic-only without a hard failure.
- Migration plan: Make gguf-py absence a hard error when `--from-gguf` regeneration is requested.

## Missing Critical Features

**No CI pipeline:**
- Problem: All quality gates (55 pytest cases, op gate, model gate, `scripts/check_no_ggml.sh`) run manually on the single WSL guest; nothing enforces them on commit.
- Blocks: Multi-machine reproduction, contributor contributions (relevant now that `CONTRIBUTING.md` exists), drift detection between docs and code.
- Note: GPU-dependent tests cannot run in hosted CI cheaply, but the pure-Python suites (`benchmarks/tests/test_llabench_wrapper.py`, `test_manifest.py`, `test_matrix_assembly.py`, `test_preflight.py`, etc.) could.

**Kernel C++/HIP tests depend on out-of-band build:**
- Problem: `benchmarks/tests/test_demo_iq4xs_dequant.py` (and similar) assert prebuilt binaries exist (`assert os.path.exists(bin_path)... Run cmake build.`) — pytest gives an opaque failure when `kernels/build/` wasn't built first, and there is no CTest wiring invoked from Python.
- Blocks: One-command validation; new-contributor onboarding.
- Files: `benchmarks/tests/test_demo_iq4xs_dequant.py`, `kernels/CMakeLists.txt`

## Test Coverage Gaps

**WMMA kernel path (large-shape correctness):**
- What's not tested: The hardware-matrix-core path gated to `M >= 512 && N >= 1024` in `kernels/matmul_iq4xs/impl_gemm_wmma.hip` — comparisons focus on tiled/GEMV paths at small/medium M; WMMA-vs-stock at M=512+ shapes lacks a dedicated correctness test in `benchmarks/tests/`.
- Files: `kernels/matmul_iq4xs/impl_gemm_wmma.hip`, `kernels/matmul_iq4xs/test_gemm_compare.cpp`
- Risk: A numerically wrong WMMA fragment mapping (see Known Bugs) reaching e2e publication.
- Priority: High

**Vulkan session driver:**
- What's not tested: `benchmarks/vulkan/run_session_vulkan.py` has no dedicated tests; it imports the lib contracts but its own session loop is uncovered.
- Files: `benchmarks/vulkan/run_session_vulkan.py`
- Risk: Silent rot of an alternate-backend capability.
- Priority: Low (only if the Vulkan path stays in scope)

**Fixture dumpers:**
- What's not tested: `tools/dump_gguf_fixtures.py` and `tools/dump_matmul_fixtures.py` have no unit tests; the tensor-candidate mapping and slicing logic (including the dead `dequant_row_cpu` stub) are verified only by their outputs.
- Files: `tools/dump_matmul_fixtures.py`, `tools/dump_gguf_fixtures.py`
- Risk: Regenerating fixtures against a new model revision could silently bind wrong tensors (see Fragile Areas).
- Priority: Medium

**Thermal watchdog kill path:**
- What's not tested: Cross-boundary kill command construction is validated for PID format, but the actual `wsl.exe`/`taskkill` execution branch is untested (understandably destructive).
- Files: `benchmarks/host/thermal_watchdog.py`
- Risk: Watchdog failing exactly when a thermal abort is needed.
- Priority: Medium (add dry-run mode test)

---

*Concerns audit: 2026-08-25*
