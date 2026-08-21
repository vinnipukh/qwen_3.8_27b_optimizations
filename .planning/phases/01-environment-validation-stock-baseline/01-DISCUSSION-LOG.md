# Phase 1 Discussion Log

**Date:** 2026-08-21 · **Mode:** default (text) · **Areas presented:** 4 · **All selected**

| Area | Question | Decision |
|------|----------|----------|
| WSL account & memory | root vs named user; .wslconfig tuning | Stay root; tune pool as-needed, evidence-driven |
| ROCm install & freeze | exact version pin; Adrenalin updates | Freeze 7.2.1 exactly; hold Adrenalin auto-updates |
| llama.cpp pin & layout | tag vs SHA; binary archive location | Claude decides (least-issues release ≥ b8394 lineage); binaries in-repo `baseline/` while repo < 750 MB |
| Model download | start now? storage? | Not now — owner downloads before execution; `models/` gitignored in-repo |
