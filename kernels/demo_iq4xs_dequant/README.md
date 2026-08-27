# Demo Op: IQ4_XS Dequantization (`dequant_iq4_xs`)

This op demonstrates the end-to-end kernel playground workflow:
1. **CPU Reference (`ref_cpu.cpp`):** Pure C++17 golden reference oracle.
2. **HIP Implementation (`impl.hip`):** WarpSize-templated (`__launch_bounds__(256, 4)`) HIP kernel for gfx1100.
3. **Broken HIP Implementation (`impl_broken.hip`):** Deliberate mutant with scale shift and nibble swap bugs.
4. **Numerical Comparison (`test_compare.cpp`):** Validates all fixtures against tight numerical bounds.
5. **Microbenchmark (`bench_sweep.cpp`):** Measures execution time (median/p95/min/max/stdev) across wave32 and wave64 configurations.

Zero `llama.cpp` or `ggml` headers included.

---

## IQ4_XS Layout & Dequantization Formula

Super-block size: 136 bytes = 256 quantized elements.
8 sub-blocks of 32 elements each.

- Scale computation for sub-block `s ∈ [0..7]`:
  $$\text{ls\_low} = (\text{scales\_l}[s / 2] \gg (4 \times (s \pmod 2))) \ \& \ 0\text{xF}$$
  $$\text{ls\_high} = (\text{scales\_h} \gg (2 \times s)) \ \& \ 0\text{x}3$$
  $$\text{ls} = \text{ls\_low} \mid (\text{ls\_high} \ll 4)$$
  $$\text{scale} = \text{fp16\_to\_fp32}(d) \times (\text{ls} - 32)$$

- Split-half weight unpacking for index `j ∈ [0..15]`:
  $$\text{weight}[s \times 32 + j] = \text{scale} \times \text{kvalues\_iq4nl}[\text{qs}[s \times 16 + j] \ \& \ 0\text{xF}]$$
  $$\text{weight}[s \times 32 + j + 16] = \text{scale} \times \text{kvalues\_iq4nl}[\text{qs}[s \times 16 + j] \gg 4]$$

---

## Numerical Acceptance Criteria

- **`max_abs`**: $< 1.0 \times 10^{-5}$
- **`mean_abs`**: $< 1.0 \times 10^{-6}$
- **`cosine`**: $\ge 0.99999$
- **`has_nan_or_inf`**: `false`

The deliberately broken implementation `demo_test_broken` triggers $> 10\times$ higher `max_abs` and fails the gate, proving the test harness discriminates and catches bugs.

---

## Building and Running

### Build
```bash
cmake -S kernels -B kernels/build -G Ninja -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_BUILD_TYPE=Release
cmake --build kernels/build
```

### Run Correct Implementation (GREEN)
```bash
export HSA_ENABLE_DXG_DETECTION=1
./kernels/build/demo_iq4xs_dequant/demo_test
# Exits 0 with PASS across all fixtures
```

### Run Broken Implementation (RED)
```bash
export HSA_ENABLE_DXG_DETECTION=1
./kernels/build/demo_iq4xs_dequant/demo_test_broken
# Exits 1 with FAIL
```

### Run Benchmark Sweep
```bash
export HSA_ENABLE_DXG_DETECTION=1
./kernels/build/demo_iq4xs_dequant/demo_bench
```
