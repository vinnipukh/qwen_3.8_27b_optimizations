<!-- generated-by: gsd-doc-writer -->

# Getting Started

Step-by-step setup to take a clean Windows 11 machine with an AMD Radeon RX 7900 XT
to running the full benchmark harness against Qwen3.8-27B.

## 1. Windows Host Configuration

1. **Configure `.wslconfig`:**
   Create or edit `C:\Users\<user>\.wslconfig`:
   ```ini
   [wsl2]
   memory=28GB
   swap=16GB
   ```
   Run `wsl --shutdown` in PowerShell.

2. **Driver Verification:**
   Ensure AMD Adrenalin driver `32.0.31041.1004` (26.10.41) or compatible WSL2 driver is installed.

3. **HWiNFO64 Telemetry (Optional but Recommended):**
   Start HWiNFO64, enable "Shared Memory Support" in Settings, and leave sensors running in the background.

## 2. WSL2 Guest Environment Setup

1. **Start Ubuntu 24.04:**
   ```bash
   wsl -d Ubuntu-24.04 -u root
   ```

2. **ROCm & librocdxg Installation:**
   Verify ROCm 7.2.1 and `librocdxg` 1.2.2 are installed. Ensure `/etc/profile.d/rocdxg.sh` contains:
   ```bash
   export HSA_ENABLE_DXG_DETECTION=1
   ```

3. **Verify GPU Enumeration:**
   ```bash
   rocminfo | grep gfx1100
   ```

## 3. Clone Repository & Setup Model

1. **Clone Repo:**
   Clone to Windows drive (e.g. `E:\Projects\qwen_3.8_27b_optimizations`), mounted at `/mnt/e/Projects/qwen_3.8_27b_optimizations` in WSL.

2. **Model Copy:**
   Ensure the locked model file exists on the fast guest ext4 filesystem:
   `/root/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf` (15.31 GB, SHA-256 `53adc4bb...`).

## 4. Install Test Dependencies & Run Suite

1. **Install pytest in WSL2:**
   ```bash
   apt-get update && apt-get install -y python3-pytest
   ```

2. **Run Pytest Suite:**
   ```bash
   cd /mnt/e/Projects/qwen_3.8_27b_optimizations
   python3 -m pytest benchmarks/tests/ -q
   ```
   All 35 tests should pass.

## 5. Execute Smoke Test & Baseline Session

1. **Run Smoke Matrix:**
   ```bash
   bash benchmarks/tests/smoke_matrix.sh
   ```

2. **Run Full Baseline Matrix:**
   ```bash
   python3 benchmarks/bin/run_session.py --tiers 4096 8192 16384 32768 --repeats 5 --delay 10
   ```

3. **Check Results:**
   Results land in `benchmarks/results/<timestamp>_baseline_hip/` with `manifest.json`, `rows.jsonl`, and `CHECKSUMS.sha256`. The published aggregate table is generated in `benchmarks/results/BASELINE-MATRIX.md`.
