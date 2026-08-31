# Phase 1: Environment Validation & Stock Baseline - Context

**Gathered:** 2026-08-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver a validated WSL2 ROCm/HIP toolchain that drives a pinned stock llama.cpp build (gfx1100 target) with the locked JonathanColetti IQ4_XS artifact fully resident on the RX 7900 XT — producing archived baseline binaries, environment fingerprint files, and a post-validation WSL snapshot. Zero optimization work; this phase is the platform kill-gate.

Requirements in scope: ENV-01, ENV-02, ENV-03, ENV-04.
</domain>

<decisions>
## Implementation Decisions

### WSL environment
- **D-01:** Stay root-only in the Ubuntu 24.04 distro — no named user account will be created. All project commands run as root. (Owner: "stay on root".)
- **D-02:** `.wslconfig` VRAM/RAM pool tuning happens as-needed during validation, driven by evidence — specifically watch the ROCm#6022-class bug where WSL RAM limits clamp the dGPU pool. Default is untouched until a measured problem appears. (Owner: "tune vram pool as needed".)

### ROCm install & freeze policy
- **D-03:** Install via the librocdxg quickstart path (AMD's blessed route). Pin **exactly ROCm 7.2.1** — no floating to newer 7.2.x without an explicit re-decision. (Owner: "lets freeze for now".) — **Reversibility: costly** — mid-project guest-stack upgrades force a full re-run of the environment version gate and invalidate prior fingerprints/benchmarks.
- **D-04:** Pause/hold Windows Adrenalin auto-updates for the project duration; record the current driver version (26.10.41 / 32.0.31041.1004) as the frozen pairing. If a driver update forces its way through, treat it as an environment-breaking event → re-run ENV gates before any benchmarking continues.

### llama.cpp pin & baseline archive
- **D-05:** Pin strategy delegated to Claude: pin the newest GitHub release tag at execution-start time, verifying its lineage includes the qwen35 correctness fix (≥ build b8394 / PR #20518 per deep-research). Record exact SHA + tag in `benchmarks/environment/`. (Owner: "you decide the one with the least issues".)
- **D-06:** Archived stock binaries live inside the repo under `baseline/` **while total repo size stays < 750 MB**; if that threshold is breached, move binaries outside git and gitignore the path. (Owner rule.)

### Model artifact logistics
- **D-07:** The 15.31 GB IQ4_XS download does NOT start now. Owner will trigger/download manually before phase execution begins. Stored at `models/` inside the repo, gitignored. sha256 (`53adc4bb…`) verification remains a phase deliverable regardless of who downloads. (Owner: "I can do it before phase execution".)

### Claude's Discretion
- Exact release-tag choice for the llama.cpp pin (within the ≥b8394-lineage constraint).
- Build flag details beyond the roadmap-fixed set (`-DGGML_HIP=ON -DGPU_TARGETS=gfx1100`, Release, Ninja + ccache).
- `.wslconfig` specific values if tuning becomes necessary (evidence-driven only).
- Timing/order of snapshot (`wsl --export`) relative to validation steps — must be serial-last per roadmap.

### Deferred Ideas
*(none surfaced during discussion)*

</decisions>

<canonical_refs>
## Canonical References

- `docs/reference/ROADMAP-original.md` — original methodology rules (binding, inherited via ROADMAP.md)
- `docs/research/MODEL-DECISION.md` — locked artifact identity, sha256, VRAM envelope incl. DXG deficit
- `docs/research/deep-research/REPORT.md` — validated platform assumptions; PROF-01 op-timer baseline decision; ≥b8394 lineage requirement
- `docs/research/deep-research/raw/scrapes/06-amd-wsl-howto.md` — AMD's official ROCm-on-WSL install procedure (primary reference for D-03)
- `docs/research/deep-research/raw/scrapes/07-librocdxg.md` — librocdxg quickstart + compatibility matrix
- https://github.com/ggml-org/llama.cpp/discussions/27047 — community Windows/HIP install guide (secondary reference)
</canonical_refs>

<specifics>
## Project-Specific Details

- Host pairing already verified: Adrenalin 26.10.41 (32.0.31041.1004), WSL 2.7.12, kernel 6.18.33, `/dev/dxg` present in guest, Ubuntu 24.04.4 LTS root-only.
- Expected `rocminfo` output: `gfx1100` agent (per librocdxg compatibility matrix).
- Startup-log fallback check must cover BOTH Gated DeltaNet and gated full-attention layer paths (ENV-03).
- Free-VRAM probe must be empirical (DXG-reported numbers overstate by ~1.5–3 GB).
</specifics>
