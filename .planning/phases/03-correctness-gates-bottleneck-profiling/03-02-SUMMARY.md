# Summary 03-02: Model-Level Quality Gate & Golden Baseline (QUAL-02)

**Phase:** 3-Correctness Gates & Bottleneck Profiling  
**Plan:** 03-02  
**Requirement:** QUAL-02  
**Status:** COMPLETE  

---

## What Was Accomplished
1. **Dataset Ingestion (`benchmarks/data/wiki.test.raw`):**
   - Downloaded and verified standard WikiText-2 test dataset (1,290,590 bytes, sha256 `173c87a5...`).
2. **Quality Gate Runner (`benchmarks/bin/run_model_gate.py`):**
   - Perplexity evaluation executing `llama-perplexity -m ... -f benchmarks/data/wiki.test.raw -c 2048 -ngl 99 --load-mode none`.
   - Measured stock baseline PPL: **6.4271 +/- 0.04103** across 145 chunks.
   - Enforced $\pm 1.0\%$ acceptance band: $[6.3628, 6.4914]$.
   - Golden greedy decode canary runner across all 6 corpus prompt files (`benchmarks/prompts/`) capturing exact token sequences and sha256 checksums.
3. **Golden Baseline Store (`benchmarks/golden/stock_baseline_golden.json`):**
   - Recorded reference PPL, allowable bounds, and golden prompt outputs.
4. **Structured Results & Tests (`benchmarks/results/phase3/model_gate.json`, `benchmarks/tests/test_model_gate.py`):**
   - 4 unit tests verifying tolerance checking, canary mismatch fail-fast, and live gate compliance.
