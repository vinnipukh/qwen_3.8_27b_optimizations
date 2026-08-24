# Project Instructions & Agent Guidelines: Qwen3.8-27B on RX 7900 XT (gfx1100)

## Critical Operational Rules & Failure Preventions

### 1. Mandatory Bash Command Timeouts
- **YOU MUST PUT A STRICT TIMEOUT ON EVERY SINGLE BASH COMMAND YOU LAUNCH.**
- Every invocation of the `bash` tool must specify an explicit, bounded `timeout` parameter (e.g. `timeout: 60`, `timeout: 90`, max `timeout: 300` for batch sweeps).
- Never execute a GPU inference script or benchmark without a process-level timeout (e.g., `timeout 90s ...` or tool timeout).
- Unbounded / hanging bash commands are strictly prohibited.

### 2. Device & Pipeline Pre-Flight Checks
- Always validate that the ROCm / DXG pipeline is responding BEFORE running heavy model workloads:
  ```bash
  export HSA_ENABLE_DXG_DETECTION=1
  /opt/rocm/bin/rocminfo | grep -E "gfx1100|RX 7900"
  ```
- If GPU fails to respond or hangs, check `dmesg` and abort before launching full models.

### 3. Step-Up Verification Discipline (CPU → Partial GPU → Full GPU)
When testing new kernel builds, binary modifications, or untested execution paths:
1. **CPU Smoke Check:** Verify baseline CPU execution first to confirm binary, flags, and model graph integrity:
   ```bash
   llama-cli -m <model> -p "Hi" -n 2 -ngl 0 --threads 4 --single-turn --simple-io
   ```
2. **Partial GPU Offload:** Incrementally offload layers (e.g., `-ngl 10`, then `-ngl 32`) to verify memory mapping and kernel launch stability without risking TDR or DXG deadlocks.
3. **Full GPU Offload:** Only transition to `-ngl 99` after partial offload passes cleanly.

### 4. WSL2 Memory & DXG Allocation Safety
- Always specify `--single-turn --simple-io --load-mode none` when using `llama-cli` to prevent interactive stdin stalls and disk-swapping thrash during mmap.
- For context sizes $\ge 4096$, always set `-b 2048` or match `n_batch` to avoid internal buffer assertion panics.
- Maintain `.wslconfig` awareness: host memory and swap allocations must support 20B+ parameter model initialization without I/O thrashing.
