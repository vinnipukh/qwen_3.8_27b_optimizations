# Project notes — why this skill is installed (qwen_3.8_27b_optimizations)

Installed 2026-08-23 from https://github.com/amd/skills (`staging/rocm-doctor`, MIT;
status: staging/planned upstream). Upstream files are verbatim; only this file is local.

## Why

Diagnoses ROCm/HIP/PyTorch/**llama.cpp** failures against a closed misconfiguration
catalog via the `rocm` CLI — the closest match to our stack if environment debugging
is ever needed on **native Linux**.

## Critical caveat — scope gate excludes WSL2

The skill's own Scope gate **declines WSL2 entirely, by design**. Our frozen dev env is
WSL2 (ROCm 7.2.1 + librocdxg), so day-to-day it will (correctly) refuse. Do not weaken
that gate locally — the refusal is protective.

**Sanctioned use:** PROF-01 escalation ladder step (c) — if Phase 3 profiling attribution
proves insufficient and the recorded decision moves work to a native-Linux session,
run rocm-doctor there first before touching drivers/ROCm versions.

See `.planning/research/EXTERNAL-RESOURCES-ASSESSMENT.md` §1 and `.planning/ROADMAP.md`
Phase 3 WSL2 risk notes.
