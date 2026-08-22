[Skip to main content](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html#main-content)

Back to top

`Ctrl` + `K`

[Use ROCm on Radeon and Ryzen](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/index.html)

Search`Ctrl` + `K`

# WSL How to guide - Use ROCm on Radeon

## Contents

# WSL How to guide - Use ROCm on Radeon [\#](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html\#wsl-how-to-guide-use-rocm-on-radeon "Link to this heading")

## New in Adrenalin 26.2.2 + ROCm 7.2.1 [\#](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html\#new-in-adrenalin-26-2-2-rocm-7-2-1 "Link to this heading")

AMD is introducing production support for the open-source ROCDXG (librocdxg) WSL solution with Adrenalin 26.2.2 + ROCm 7.2.1. ROCDXG is a user-mode library that enables ROCm functionality inside WSL for GPU-accelerated workloads, including AI and HPC use cases. It leverages a ROCm runtime feature introduced in ROCm 7.1 and is designed to remain loosely coupled from both ROCm releases and Windows display drivers—allowing the solution to evolve independently.
This release also marks the first time AMD is supporting Ryzen Strix and Strix Halo SKUs versus the legacy WSL solution.

- [AMD Software: Adrenalin Edition™ 26.2.2 for WSL2](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-26-2-2.html)


## AMD ROCDXG library [\#](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html\#amd-rocdxg-library "Link to this heading")

Production support is now available for the open-source ROCDXG (librocdxg) solution on WSL—enabling GPU-accelerated Linux workloads under WSL with a decoupled design that can evolve independently.

- [ROCDXG (librocdxg) GitHub repo and compatibility matrix](https://github.com/ROCm/librocdxg/)


### What is ROCDXG? [\#](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html\#what-is-rocdxg "Link to this heading")

ROCDXG (librocdxg) is an open-source, user-mode library that enables ROCm GPU-compute functionality for Windows Subsystem for Linux (WSL). It serves as the translation layer between the Linux ROCm runtime and the Windows GPU driver stack, allowing developers to run GPU-accelerated workloads—including AI inference, machine learning training, and HPC applications—directly within WSL on a Windows host machine.

ROCDXG is designed as a modular, loosely coupled, community-friendly solution. It evolves independently from both AMD ROCm releases and Windows display driver versions, meaning updates to either do not break the WSL compute path. It is hosted publicly on the \[ROCm GitHub organization\] ( [ROCm/librocdxg](https://github.com/ROCm/librocdxg)) and welcomes community contributions.

This release also marks the first time AMD officially supports Ryzen Strix and Strix Halo SKUs on WSL, in addition to the existing discrete Radeon GPU lineup.

### Why ROCDXG? [\#](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html\#why-rocdxg "Link to this heading")

**Modernized WSL Integration Model**

ROCDXG communicates with the Windows GPU driver through Microsoft’s DXCore interface (/dev/dxg), which enables GPU acceleration inside WSL.
This architecture replaces the legacy roc4wsl packaging model and aligns ROCm on WSL with Microsoft’s supported GPU virtualization framework. Hardware support remains defined by the official ROCm compatibility matrix and validated platform SKUs.

**No Display Driver Installation Required**

ROCDXG functions as the user-mode bridge between the ROCm runtime and the Windows DXCore interface provided by the installed display driver.
Unlike the legacy roc4wsl approach, ROCDXG does not require installation of Radeon Software for Linux (RSL) packages to enable WSL compute. The Windows display driver remains the authoritative GPU driver, while ROCDXG enables the compute interface inside WSL.

**Loosely Coupled, Independent Release Lifecycle**

For users and enterprises, this translates directly into a more stable, predictable, and lower-maintenance WSL GPU compute environment. There are no version-pair constraints to track, no risk of a driver update silently breaking the compute path, and no dependency on AMD’s internal release schedule to receive improvements.

**Simplified Setup — No Version Specification Required**

ROCDXG does not require users to specify or match a ROCm driver version during integration. The library is designed for compatibility across multiple ROCm releases simultaneously, eliminating an entire category of setup complexity.

**Legacy vs. ROCDXG Comparison**

| Problem (Legacy roc4wsl) | Solution (ROCDXG) |
| --- | --- |
| Required full RSL package installation to enable WSL compute | Standalone user-mode library integrated with standard Windows Adrenalin driver |
| Required tight coupling to Linux driver packaging model | Uses DXCore interface for WSL GPU enablement (supported hardware defined by compatibility matrix) |
| Tightly coupled to RSL and Windows driver release cycles | Loosely coupled — evolves independently from ROCm and Windows driver versions |
| Closed, monolithic distribution | Open-source, community-driven, hosted on GitHub |
| Required matching specific RSL + ROCm version pairs | No manual driver version pairing required |
| No path for native Windows ROCm support | Architected as a downstream integration path toward native Windows ROCm |

### Key Features & Improvements [\#](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html\#key-features-improvements "Link to this heading")

**Open-Source & Community-Driven**

ROCDXG is fully open-source and hosted on ( [ROCm/librocdxg](https://github.com/ROCm/librocdxg)). This is a deliberate architectural decision that aligns with AMD’s commitment to open-source GPU compute. Developers can inspect, contribute to, and extend the library, fostering innovation across AI, HPC, and experimental computing scenarios.

**Loosely Coupled Architecture**

ROCDXG is intentionally designed to remain loosely coupled from both AMD ROCm releases and Windows display driver versions. This means:

> - ROCDXG can be updated independently without requiring a new ROCm release.
>
> - Windows driver updates do not force a corresponding ROCDXG update.
>
> - Users and enterprises can adopt new ROCDXG versions at their own pace.

**Simplified Integration**

Unlike the legacy roc4wsl approach, ROCDXG does not require specifying a ROCm driver version during integration. This significantly reduces complexity and maintenance overhead for both end users and downstream integrators.

Note

Non-WSL framework guides can be used to install the frameworks.

## Install WSL with ROCDXG [\#](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html\#install-wsl-with-rocdxg "Link to this heading")

1. Install Adrenalin driver. To install the compatible driver, refer to [AMD Software: Adrenalin Edition™ 26.2.2 for WSL2](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-26-2-2.html).

2. Install WSL. To install WSL, refer to [Windows Subsystem for Linux Documentation](https://learn.microsoft.com/en-us/windows/wsl/).


> Ubuntu 24.04 and 22.04 are the supported distros on WSL.

3. Open the [librocdxg github](https://github.com/ROCm/librocdxg/) link and follow the **Quickstart** installation.

4. Use Radeon framework how-tos to install on WSL.

To install frameworks on WSL, refer to the [Radeon How-To Guides](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/howto_native_linux.html).


Note

The ROCm-supported version of JAX is not currently enabled or validated under WSL. As a result, JAX workloads on WSL may fail to install, initialize, or execute correctly.

## Legacy known issues [\#](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html\#legacy-known-issues "Link to this heading")

As of ROCm 7.2.1 AMD is using the ROCDXG implementation. Refer here to view 7.2 known issues:

- [Legacy WSL known issues 7.2](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/limitations/limitationsrad.html#wsl)


## Legacy compatibility matrix [\#](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html\#legacy-compatibility-matrix "Link to this heading")

As of ROCm 7.2.1 AMD is using the ROCDXG implementation. Refer here to view 7.2 compatibility matrix:

- [Legacy WSL compatibility matrix 7.2](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html)


## Optional: Legacy WSL how-to guide [\#](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html\#optional-legacy-wsl-how-to-guide "Link to this heading")

As of ROCm 7.2.1 AMD is using the ROCDXG implementation. Users who wish to continue using the legacy implementation must use ROCm 7.2.

- [Legacy WSL how-to guide with ROCm 7.2](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installrad/wsl/howto_wsl.html)


Note

- MIGraphX and mGPU configuration are not currently supported by WSL


Contents


Versions**[latest](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html)**[docs-7.2.1](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/install/installrad/wsl/howto_wsl.html)[docs-7.2](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installrad/wsl/howto_wsl.html)[docs-7.1.1](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.1.1/docs/install/installrad/wsl/howto_wsl.html)[docs-7.1](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.1/docs/install/installrad/wsl/howto_wsl.html)[docs-7.0.2](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.0.2/docs/install/installrad/wsl/howto_wsl.html)[docs-6.4.4](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-6.4.4/docs/install/installrad/wsl/howto_wsl.html)[docs-6.4.2](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-6.4.2/docs/install/installrad/wsl/howto_wsl.html)[docs-6.3.4](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-6.3.4/docs/install/installrad/wsl/howto_wsl.html)[docs-6.3.2](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-6.3.2/docs/install/installrad/wsl/howto_wsl.html)[docs-6.2.3](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-6.2.3/docs/install/installrad/wsl/howto_wsl.html)[docs-6.1.3](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-6.1.3/docs/install/installrad/wsl/howto_wsl.html)[docs-6.0.2](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-6.0.2/docs/install/installrad/wsl/howto_wsl.html)[docs-5.7.0](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-5.7.0/docs/install/installrad/wsl/howto_wsl.html)Downloads[PDF](https://rocm.docs.amd.com/_/downloads/radeon-ryzen/en/latest/pdf/)[HTML](https://rocm.docs.amd.com/_/downloads/radeon-ryzen/en/latest/htmlzip/)[EPUB](https://rocm.docs.amd.com/_/downloads/radeon-ryzen/en/latest/epub/)On Read the Docs[Project Home](https://app.readthedocs.com/projects/advanced-micro-devices-radeon-ryzen/?utm_source=advanced-micro-devices-radeon-ryzen&utm_content=flyout)[Builds](https://app.readthedocs.com/projects/advanced-micro-devices-radeon-ryzen/builds/?utm_source=advanced-micro-devices-radeon-ryzen&utm_content=flyout)Search

* * *

[Addons documentation](https://docs.readthedocs.io/page/addons.html?utm_source=advanced-micro-devices-radeon-ryzen&utm_content=flyout) ― Hosted by
[Read the Docs](https://about.readthedocs.com/?utm_source=advanced-micro-devices-radeon-ryzen&utm_content=flyout)