# Project notes — why this skill is installed (qwen_3.8_27b_optimizations)

Installed 2026-08-23 from https://github.com/amd/skills (`skills/magpie-kernel-evaluator`,
MIT). Upstream `SKILL.md`/`reference.md`/`examples.md` are verbatim; only this file is local.

## Why

`magpie analyze` / `magpie compare` for standalone HIP kernels is a near 1:1 external
analogue of our Phase 4 KERN-01 quartet (`ref_cpu → impl_gfx1100 → test_compare →
bench_sweep`) and Phase 5 head-to-head microbenchmarks. Primary value = **design prior
art** for harness conventions: explicit baseline, correctness-gate before ranking,
identical warmup/tolerances/iterations, full config provenance per row.

## Caveats — read before invoking

- The **Magpie tool itself is NOT installed** in this project. Treat the skill as a
  conventions/prior-art reference by default. Probing `magpie analyze --type hip` inside
  the Phase 4 playground is sanctioned upside, never a dependency.
- Magpie is validated on Instinct MI3xx; the skill's own preflight rule says treat
  unlisted hardware as **unverified until tested** — gfx1100 under WSL2/DXG is unlisted.
- Its profiling legs depend on exactly what our PROF-01 flags as the soft spot under DXG.
- Do not let its vLLM/SGLang benchmark workflows leak into our llama.cpp-based harness.

See `.planning/research/EXTERNAL-RESOURCES-ASSESSMENT.md` §1 for the full evaluation.
