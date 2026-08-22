## Key insights

- AMD ROCm 7.2.4's new DXG connector (librocdxg) lets WSL2 Linux use an RX 7900 XTX for llama.cpp inference without dual-booting.
- The WSL2 path via librocdxg reports roughly 3GB less free VRAM than native Windows on the same RX 7900 XTX under ROCm 7.2.4.
- Developer Diablo-D3 published full build instructions including librocdxg compilation steps and the required HSA\_ENABLE\_DXG\_DETECTION=1 environment variable.

## Why this matters

AMD GPU users on Windows have had no documented WSL2 GPU compute path for llama.cpp, making this DXG connector the first community-verified mechanism to run AMD GPU inference without leaving Windows. The 3GB VRAM shortfall is a concrete constraint: an RX 7900 XTX reporting 24,136 MiB free on native Windows drops to 21,191 MiB under WSL2, directly shrinking the context window and limiting which models fit in memory. Community-documented setups like this one typically feed into official driver and tooling support, placing Diablo-D3's issue in position to become a reference point for both AMD ROCm and llama.cpp maintainers.


## Summary

AMD ROCm gained a DXG connector (librocdxg) letting WSL2 Linux reach Windows GPU hardware directly, enabling llama.cpp inference on AMD GPUs without dual-booting. Developer Diablo-D3 documented the full build process and a persistent VRAM reporting anomaly in llama.cpp GitHub issue #23999, opened June 1, 2026.

On an RX 7900 XTX with ROCm 7.2.4, native Windows reports 24,136 MiB of free VRAM; the WSL2 path via librocdxg shows only 21,191 MiB -- a nearly 3GB gap that directly reduces the context window llama.cpp can allocate.

Essentially: (AMD, llama.cpp community) now have a working WSL2 GPU path, but memory accounting is not yet accurate.

\- Build requires librocdxg compiled from source with Windows SDK headers and the HSA\_ENABLE\_DXG\_DETECTION=1 environment variable.
\- The issue carries a bug-unconfirmed label; Diablo-D3 notes it is "probably not a llama.cpp bug, but worth documenting."

The gap shows AMD's WSL2 path is functional enough to test and document, but not yet a transparent substitute for native Linux inference.


## Potential risks and opportunities

### Risks

- Windows users following Diablo-D3's build instructions may encounter reduced context allocation without understanding the VRAM underreporting cause, incorrectly attributing model loading failures to llama.cpp.
- The librocdxg build dependency on Windows SDK headers creates a fragile setup that could break with AMD Adrenaline driver updates beyond version 26.5.2.
- The bug-unconfirmed status means AMD and llama.cpp maintainers may deprioritize a fix, leaving Windows AMD users with a persistent 3GB context penalty relative to native Linux setups indefinitely.

### Opportunities

- llama.cpp maintainers could add WSL2 DXG-aware VRAM detection logic to correct the 21,191 MiB underreport and restore full context allocation for Windows AMD GPU users.
- AMD could officially document and support the librocdxg WSL2 path in ROCm 7.x release notes, converting a community workaround into a supported configuration and expanding their Windows inference user base.
- Windows-focused local inference tools could integrate the DXG connector path to unlock AMD GPU acceleration without requiring dual-boot, opening a user segment currently blocked by the absence of a native WSL2 AMD compute path.

## What we don't know yet

- Whether the gap between native Windows (24,136 MiB free) and WSL2 (21,191 MiB free) is a ROCm 7.2.4 bug, a Windows DXG memory reservation, or a librocdxg accounting error -- not resolved in the issue.
- Whether librocdxg will be distributed as a prebuilt package or will continue to require users to compile from source with Windows SDK headers.
- Inference throughput benchmarks comparing WSL2 ROCm 7.2.4 vs. native Windows ROCm on the same RX 7900 XTX hardware -- entirely absent from the issue report.

Originally reported by
**github.com**

[Read the original article →](https://github.com/ggml-org/llama.cpp/issues/23999)

Original headline:r/LocalLLaMA: AMD ROCm Now Runs Sanely Inside WSL2 for llama.cpp — Community Posts Working Build Instructions, Known Bugs Remain

Free AI alerts in your inbox
Breaking AI news 3x/week. 50,000+ subscribers.

EmailGet them

Leave this field blank

×

We use essential cookies to keep the site working (login, form security). With your permission, we also use analytics cookies to understand how you use the site.
[Privacy policy](https://aiweekly.co/privacy)

Accept AllEssential OnlyReject Non-Essential