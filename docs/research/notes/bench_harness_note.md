# Bench Harness N=10 Rigour — Verification Note (07-01)

**Scope:** Phase 7 Fix for Bench Harness N=10 rigour (07-01). Pure C++/HIP, no GPU execution required for static validation; hardware timing is quoted from `bench_real_stock.hardware.json` (WSL2 gfx1100, HSA_ENABLE_DXG_DETECTION=1). Timeout 90s per bench to avoid DXG deadlock.

## Artifacts Under Verification

| File | Role |
|------|------|
| `kernels/matmul_iq4xs/bench_real_stock.cpp` | `--runs <int>` default 10, `--json` flag, aggregates `BenchStats` median/mean/stddev/p95 per 8 canonical shapes, computes `speedup_vs_naive`, emits JSON `runs:10` + `note` |
| `kernels/common/bench.h` | `BenchStats` with `median_us`, `mean_us`, `stddev_us`/`stdev_us`, `p95_us`, `gb_s`; `bench_hip_event(warmup 50, iters 200)` |
| `kernels/matmul_iq4xs/bench_real_stock.hardware.json` | Valid JSON, 8 entries, each `runs:10` + `real_dp4a_median_us` + `real_dp4a_stddev_us` + `real_dp4a_p95_us` + `naive_median_us` + `speedup_vs_naive` |
| `kernels/matmul_iq4xs/baseline_dp4a.json` | Copy of `bench_real_stock.hardware.json` (verbatim, `baseline==hardware` python equality), same 8×N=10 schema |
| `kernels/matmul_iq4xs/real_stock_dp4a_comparator.hip` | `ggml_cuda_dp4a_real` via `__builtin_amdgcn_sudot4` + 6× `__builtin_amdgcn_perm`, `vec_dot_iq4_xs_q8_1_device`, `quantize_row_q8_1_standalone` with `half2 ds` (`pack_half2`/`low2float`) |
| `kernels/matmul_iq4xs/BASELINE_DP4A.md` | Title + Source + Device + Benchmark + table `Shape|K|N|naive median±stddev|real DP4A median±stddev|p95|GB/s|speedup` + `runs:10` + interpretation 84-105us vs 543us DP4A proof |
| `kernels/matmul_iq4xs/CMakeLists.txt` | `matmul_real_stock_hip OBJECT` correctly linked to `bench_real_stock` and `test_real_stock_compare` |
| `docs/research/notes/bench_harness_note.md` | This file |

## Verification Commands — Copy-Paste

### 1. CMake configure + build (WSL2 Ubuntu-24.04 ROCm 7.2.1 or Windows HIP SDK 6.4)

```bash
# WSL2 (ROCm at /opt/rocm)
HSA_ENABLE_DXG_DETECTION=1 cmake -S kernels -B kernels/build -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_BUILD_TYPE=Release
cmake --build kernels/build --parallel 4 --target bench_real_stock test_real_stock_compare
# Windows (HIP SDK at %HIP_PATH%, Ninja, no cl)
cmake -S kernels -B kernels/build -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_BUILD_TYPE=Release -G Ninja -DCMAKE_CXX_COMPILER="%HIP_PATH%/bin/clang++.exe" -DCMAKE_HIP_COMPILER="%HIP_PATH%/bin/clang++.exe"
cmake --build kernels/build --parallel 4 --target bench_real_stock test_real_stock_compare
```

`--parallel 4` avoids OOM on 32GB; timeout 90s per bench handles DXG jitter.

### 2. Static grep gates (no GPU)

```bash
grep -n "ggml_cuda_dp4a_real" kernels/matmul_iq4xs/real_stock_dp4a_comparator.hip
grep -c "__builtin_amdgcn_perm" kernels/matmul_iq4xs/real_stock_dp4a_comparator.hip   # expect 6
grep -c "__builtin_amdgcn_sudot4" kernels/matmul_iq4xs/real_stock_dp4a_comparator.hip # expect >=1
grep -n "vec_dot_iq4_xs_q8_1_device" kernels/matmul_iq4xs/real_stock_dp4a_comparator.hip
grep -n "quantize_row_q8_1_standalone" kernels/matmul_iq4xs/real_stock_dp4a_comparator.hip
grep -n "pack_half2\|low2float\|half2" kernels/matmul_iq4xs/real_stock_dp4a_comparator.hip
grep -n "stddev_us\|stdev_us" kernels/common/bench.h
grep -n "matmul_real_stock_hip" kernels/matmul_iq4xs/CMakeLists.txt
grep -n "Shape.*K.*N.*naive.*real DP4A.*p95.*GB/s.*speedup" kernels/matmul_iq4xs/BASELINE_DP4A.md
grep -n "84-105us vs 543us" kernels/matmul_iq4xs/BASELINE_DP4A.md
grep -n "runs.*10\|N=10\|--runs 10" kernels/matmul_iq4xs/BASELINE_DP4A.md
```

### 3. JSON schema validation (no GPU)

```bash
python -c "import json; d=json.load(open('kernels/matmul_iq4xs/bench_real_stock.hardware.json')); assert len(d)==8; assert all(x.get('runs')==10 for x in d); assert all('real_dp4a_median_us' in x and 'real_dp4a_stddev_us' in x and 'real_dp4a_p95_us' in x and 'naive_median_us' in x and 'speedup_vs_naive' in x and 'note' in x for x in d); print('bench_real_stock.hardware.json OK', [x['shape'] for x in d])"
python -c "import json; h=json.load(open('kernels/matmul_iq4xs/bench_real_stock.hardware.json')); b=json.load(open('kernels/matmul_iq4xs/baseline_dp4a.json')); assert h==b; print('baseline_dp4a.json == hardware JSON')"
python -c "import json; d=json.load(open('kernels/matmul_iq4xs/baseline_dp4a.json')); print({k: d[0][k] for k in ['real_dp4a_median_us','real_dp4a_stddev_us','real_dp4a_p95_us','naive_median_us','speedup_vs_naive','runs']})"
```

### 4. Bench execution N=10 (requires gfx1100 GPU, timeout 90)

```bash
timeout 90 bash -c 'HSA_ENABLE_DXG_DETECTION=1 ./kernels/build/matmul_iq4xs/bench_real_stock --runs 10 --json > /tmp/brs.json && python -c "import json; d=json.load(open(\"/tmp/brs.json\")); assert len(d)==8; assert all(x[\"runs\"]==10 for x in d); assert all(\"real_dp4a_stddev_us\" in x and \"real_dp4a_p95_us\" in x for x in d); print(\"bench_real_stock --runs 10 --json OK\", d[0][\"real_dp4a_median_us\"], \"vs naive\", d[0][\"naive_median_us\"])"'
timeout 90 bash -c 'HSA_ENABLE_DXG_DETECTION=1 ./kernels/build/matmul_iq4xs/test_real_stock_compare 2>&1 | tail -20'
```

### 5. Windows compile probe (no GPU, no cl)

```bash
# WSL2 syntax probe
hipcc --offload-arch=gfx1100 -c kernels/matmul_iq4xs/real_stock_dp4a_comparator.hip -o /tmp/rs.o && echo "hipcc gfx1100 OK"
# Windows native (PowerShell, HIP SDK 6.4, VS Build Tools, Ninja)
# $env:HIP_PATH/bin/clang++.exe --offload-arch=gfx1100 -I kernels/common -I kernels/matmul_iq4xs -c kernels/matmul_iq4xs/real_stock_dp4a_comparator.hip -o rs.o
# Expected: clean (warnings only), no cl.exe in PATH, --offload-arch=gfx1100 present
```

### 6. BASELINE_DP4A.md reproduce reference

```bash
cat kernels/matmul_iq4xs/BASELINE_DP4A.md | head -n 25
# Expect: Title, Source: real_stock_dp4a_comparator.hip exact quantize_row_q8_1 + vec_dot_iq4_xs_q8_1 via __builtin_amdgcn_sudot4 + 6x __builtin_amdgcn_perm,
# Device: RX 7900 XT gfx1100 ROCm 7.2.1, Benchmark: bench_real_stock --runs 10 --json (50 warmup/200 iters hipEvent),
# Table columns Shape|K|N|naive median ± stddev|real DP4A median ± stddev|p95|GB/s|speedup with median ± stddev and runs:10,
# Interpretation: 84-105us DP4A vs 543us naive proof (bare-metal 84us ±4us tight, WSL2 99.55 ±28.56 with DXG jitter, p95 231 vs 780).
```

## Expected Outputs (from captured hardware JSON, not re-measured)

- `bench_real_stock.hardware.json` 8 entries: `real_dp4a_median_us` 92-135 band (attn_q 99.55 ±28.56 p95 231.54, attn_k 105.64 ±43.82, etc.) vs `naive_median_us` 543 ±84 p95 780 → `speedup 5.46x` (ffn_down 16.06x). `runs:10` per object, `note` field present.
- `BASELINE_DP4A.md` table shows `median ± stddev` (not point) per shape, `p95` column, `GB/s` (130-421), `speedup` 5.14-16x, `runs:10` noted; interpretation states **84-105us vs 543us DP4A proof**.
- `real_stock_dp4a_comparator.hip` greps: `ggml_cuda_dp4a_real` ≥1, `__builtin_amdgcn_sudot4` ≥1, `__builtin_amdgcn_perm` ==6 (2 low even/odd + 2 high even/odd + 2 mask perm), `vec_dot_iq4_xs_q8_1_device` ≥1, `quantize_row_q8_1_standalone` ≥1 with `pack_half2`/`low2float` half2 ds.
- `bench.h` has `stddev_us` alias alongside `stdev_us` (both kept in sync).
- `CMakeLists.txt` has `add_library(matmul_real_stock_hip OBJECT real_stock_dp4a_comparator.hip)` and `target_link_libraries(bench_real_stock PRIVATE ... $<TARGET_OBJECTS:matmul_real_stock_hip>)`.

## Notes

- **No Python shipped for Windows gate:** The harness is pure C++/HIP; JSON validation uses offline python only (`python -c json.load`) and is not part of the build.
- **DXG deadlock:** `bench_gemm_wmma` previously hung 271s; all benches use `timeout 90` wrapper.
- **Single-run banned:** Per REQ-STAT-07, table must show `median ± stddev` + `p95` over N=10, not a single point; `test_real_stock_compare` cosine 0.999985 PASS 15/15 is the correctness proof for the DP4A decode `d*(ls-32)*kvalues_iq4nl`.
