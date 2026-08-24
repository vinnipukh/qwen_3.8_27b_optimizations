# Bottleneck Attribution & Optimization Target #1 Table (PROF-02)

**Date:** 2026-08-24 21:22:17 UTC  
**Target Hardware:** AMD Radeon RX 7900 XT (`gfx1100`, RDNA3, 20 GiB VRAM)  
**Model Artifact:** `/root/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf`  
**Host Stack:** Windows 11 Pro / WSL2 Ubuntu 24.04 (Adrenalin 26.2.2 + ROCm 7.2.1)  

---

## 1. Executive Designation of Optimization Target #1

🎯 **PRIMARY OPTIMIZATION TARGET #1:** `MUL_MAT`

> **Attribution Rationale:** Across all four canonical inference shapes (S1–S4), `MUL_MAT` accounts for **31.12%** of total cumulative GPU execution time (88033.03 ms). In the Decode phase (M=1), it constitutes **30.04%** of runtime, and in Prefill (M>>1) it constitutes **50.89%**. Custom gfx1100 kernel development in Phase 4 (scaffolding) and Phase 5 (kernel attack) will directly target this bottleneck.

---

## 2. Cumulative Op Bottleneck Ranking

| Rank | GGML Operation | % Total GPU Time | Cumulative Time (ms) | % Prefill Time | % Decode Time | Primary Bound Classification |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| 1 | `MUL_MAT` | **31.12%** | 88033.03 ms | 50.89% | 30.04% | Memory Bandwidth / Dequant Bound |
| 2 | `MUL` | **13.59%** | 38439.19 ms | 8.73% | 13.85% | Memory Bandwidth Bound |
| 3 | `RMS_NORM` | **9.76%** | 27619.77 ms | 6.01% | 9.97% | Memory Bandwidth Bound |
| 4 | `ADD` | **8.45%** | 23901.92 ms | 4.80% | 8.65% | Memory Bandwidth Bound |
| 5 | `SILU` | **4.96%** | 14028.50 ms | 2.73% | 5.08% | Compute / Elementwise Bound |
| 6 | `GET_ROWS` | **4.28%** | 12093.74 ms | 2.93% | 4.35% | Memory Bandwidth / Gathering Bound |
| 7 | `CPY` | **4.14%** | 11720.08 ms | 2.55% | 4.23% | Memory Bandwidth Bound |
| 8 | `L2_NORM` | **4.03%** | 11409.25 ms | 2.34% | 4.13% | Memory Bandwidth Bound |
| 9 | `SWIGLU` | **2.79%** | 7885.69 ms | 2.53% | 2.80% | Compute / Elementwise Bound |
| 10 | `SIGMOID` | **2.70%** | 7625.45 ms | 1.57% | 2.76% | Compute / Elementwise Bound |
| 11 | `GATED_DELTA_NET` | **2.25%** | 6376.70 ms | 4.55% | 2.13% | Compute / Register Bound |
| 12 | `CONCAT` | **2.07%** | 5860.47 ms | 1.71% | 2.09% | Memory / Compute Bound |

---

## 3. Breakdown by Inference Workload Shape

### Interactive (128 / 128)
- **Prefill Latency:** 521.10 ms (103 tokens)
- **Decode Latency:** 46545.20 ms (128 tokens)

#### Top Operations (Decode Phase M=1)
| Rank | GGML Operation | % Runtime | Total Time (ms) | Invocation Count | Avg Latency (μs) | Bound Classification |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| 1 | `MUL_MAT` | 27.05% | 12406.70 ms | 63616 | 195.0 μs | Memory Bandwidth / Dequant Bound |
| 2 | `RMS_NORM` | 14.67% | 6729.60 ms | 26752 | 251.6 μs | Memory Bandwidth Bound |
| 3 | `MUL` | 14.12% | 6473.67 ms | 41088 | 157.6 μs | Memory Bandwidth Bound |
| 4 | `ADD` | 7.73% | 3546.49 ms | 22528 | 157.4 μs | Memory Bandwidth Bound |
| 5 | `GET_ROWS` | 4.46% | 2044.95 ms | 24832 | 82.3 μs | Memory Bandwidth / Gathering Bound |
| 6 | `CPY` | 4.30% | 1971.87 ms | 24576 | 80.2 μs | Memory Bandwidth Bound |

#### Top Operations (Prefill Phase M>>1)
| Rank | GGML Operation | % Runtime | Total Time (ms) | Invocation Count | Avg Latency (μs) | Bound Classification |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| 1 | `MUL_MAT` | 41.27% | 211.66 ms | 497 | 425.9 μs | Memory Bandwidth / Dequant Bound |
| 2 | `MUL` | 10.50% | 53.85 ms | 321 | 167.8 μs | Memory Bandwidth Bound |
| 3 | `RMS_NORM` | 6.17% | 31.64 ms | 209 | 151.4 μs | Memory Bandwidth Bound |
| 4 | `ADD` | 5.30% | 27.20 ms | 176 | 154.6 μs | Memory Bandwidth Bound |
| 5 | `GET_ROWS` | 4.27% | 21.91 ms | 194 | 113.0 μs | Memory Bandwidth / Gathering Bound |
| 6 | `SILU` | 3.81% | 19.54 ms | 96 | 203.5 μs | Compute / Elementwise Bound |

### Decode Heavy (128 / 256)
- **Prefill Latency:** 459.65 ms (103 tokens)
- **Decode Latency:** 91189.10 ms (256 tokens)

#### Top Operations (Decode Phase M=1)
| Rank | GGML Operation | % Runtime | Total Time (ms) | Invocation Count | Avg Latency (μs) | Bound Classification |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| 1 | `MUL_MAT` | 30.78% | 27625.80 ms | 127232 | 217.1 μs | Memory Bandwidth / Dequant Bound |
| 2 | `MUL` | 14.46% | 12981.50 ms | 82176 | 158.0 μs | Memory Bandwidth Bound |
| 3 | `RMS_NORM` | 9.43% | 8461.18 ms | 53504 | 158.1 μs | Memory Bandwidth Bound |
| 4 | `ADD` | 7.94% | 7124.21 ms | 45056 | 158.1 μs | Memory Bandwidth Bound |
| 5 | `GET_ROWS` | 4.56% | 4090.02 ms | 49664 | 82.3 μs | Memory Bandwidth / Gathering Bound |
| 6 | `CPY` | 4.43% | 3975.38 ms | 49152 | 80.9 μs | Memory Bandwidth Bound |

#### Top Operations (Prefill Phase M>>1)
| Rank | GGML Operation | % Runtime | Total Time (ms) | Invocation Count | Avg Latency (μs) | Bound Classification |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| 1 | `MUL_MAT` | 46.69% | 211.54 ms | 497 | 425.6 μs | Memory Bandwidth / Dequant Bound |
| 2 | `MUL` | 9.40% | 42.57 ms | 321 | 132.6 μs | Memory Bandwidth Bound |
| 3 | `RMS_NORM` | 5.46% | 24.75 ms | 209 | 118.4 μs | Memory Bandwidth Bound |
| 4 | `ADD` | 4.29% | 19.44 ms | 176 | 110.4 μs | Memory Bandwidth Bound |
| 5 | `SILU` | 3.87% | 17.52 ms | 96 | 182.6 μs | Compute / Elementwise Bound |
| 6 | `GET_ROWS` | 3.78% | 17.11 ms | 194 | 88.2 μs | Memory Bandwidth / Gathering Bound |

### Prefill Heavy (4096 / 128)
- **Prefill Latency:** 7331.89 ms (3308 tokens)
- **Decode Latency:** 46815.70 ms (128 tokens)

#### Top Operations (Decode Phase M=1)
| Rank | GGML Operation | % Runtime | Total Time (ms) | Invocation Count | Avg Latency (μs) | Bound Classification |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| 1 | `MUL_MAT` | 26.24% | 12104.00 ms | 63616 | 190.3 μs | Memory Bandwidth / Dequant Bound |
| 2 | `MUL` | 13.42% | 6192.68 ms | 41088 | 150.7 μs | Memory Bandwidth Bound |
| 3 | `SILU` | 9.37% | 4321.21 ms | 12288 | 351.7 μs | Compute / Elementwise Bound |
| 4 | `RMS_NORM` | 8.75% | 4036.92 ms | 26752 | 150.9 μs | Memory Bandwidth Bound |
| 5 | `ADD` | 7.37% | 3399.42 ms | 22528 | 150.9 μs | Memory Bandwidth Bound |
| 6 | `FLASH_ATTN_EXT` | 5.07% | 2337.18 ms | 2048 | 1141.2 μs | Compute / Memory Bandwidth Bound |

#### Top Operations (Prefill Phase M>>1)
| Rank | GGML Operation | % Runtime | Total Time (ms) | Invocation Count | Avg Latency (μs) | Bound Classification |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| 1 | `MUL_MAT` | 50.95% | 3703.95 ms | 3479 | 1064.7 μs | Memory Bandwidth / Dequant Bound |
| 2 | `MUL` | 8.75% | 635.96 ms | 2247 | 283.0 μs | Memory Bandwidth Bound |
| 3 | `RMS_NORM` | 6.13% | 445.76 ms | 1463 | 304.7 μs | Memory Bandwidth Bound |
| 4 | `ADD` | 4.85% | 352.36 ms | 1232 | 286.0 μs | Memory Bandwidth Bound |
| 5 | `GATED_DELTA_NET` | 4.64% | 337.63 ms | 336 | 1004.9 μs | Compute / Register Bound |
| 6 | `GET_ROWS` | 2.89% | 210.32 ms | 1358 | 154.9 μs | Memory Bandwidth / Gathering Bound |

### Agentic Multi-Turn (4096 / 256)
- **Prefill Latency:** 6460.39 ms (3308 tokens)
- **Decode Latency:** 87745.40 ms (256 tokens)

#### Top Operations (Decode Phase M=1)
| Rank | GGML Operation | % Runtime | Total Time (ms) | Invocation Count | Avg Latency (μs) | Bound Classification |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| 1 | `MUL_MAT` | 32.89% | 28445.60 ms | 127232 | 223.6 μs | Memory Bandwidth / Dequant Bound |
| 2 | `MUL` | 13.31% | 11513.60 ms | 82176 | 140.1 μs | Memory Bandwidth Bound |
| 3 | `ADD` | 10.56% | 9129.19 ms | 45056 | 202.6 μs | Memory Bandwidth Bound |
| 4 | `RMS_NORM` | 8.69% | 7511.63 ms | 53504 | 140.4 μs | Memory Bandwidth Bound |
| 5 | `GET_ROWS` | 4.17% | 3605.41 ms | 49664 | 72.6 μs | Memory Bandwidth / Gathering Bound |
| 6 | `CPY` | 4.07% | 3518.26 ms | 49152 | 71.6 μs | Memory Bandwidth Bound |

#### Top Operations (Prefill Phase M>>1)
| Rank | GGML Operation | % Runtime | Total Time (ms) | Invocation Count | Avg Latency (μs) | Bound Classification |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| 1 | `MUL_MAT` | 51.90% | 3323.77 ms | 3479 | 955.4 μs | Memory Bandwidth / Dequant Bound |
| 2 | `MUL` | 8.52% | 545.35 ms | 2247 | 242.7 μs | Memory Bandwidth Bound |
| 3 | `RMS_NORM` | 5.91% | 378.30 ms | 1463 | 258.6 μs | Memory Bandwidth Bound |
| 4 | `ADD` | 4.74% | 303.61 ms | 1232 | 246.4 μs | Memory Bandwidth Bound |
| 5 | `GATED_DELTA_NET` | 4.68% | 299.76 ms | 336 | 892.1 μs | Compute / Register Bound |
| 6 | `GET_ROWS` | 2.80% | 179.54 ms | 1358 | 132.2 μs | Memory Bandwidth / Gathering Bound |
