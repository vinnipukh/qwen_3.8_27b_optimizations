[Skip to content](https://github.com/ROCm/librocdxg#start-of-content)

You signed in with another tab or window. [Reload](https://github.com/ROCm/librocdxg) to refresh your session.You signed out in another tab or window. [Reload](https://github.com/ROCm/librocdxg) to refresh your session.You switched accounts on another tab or window. [Reload](https://github.com/ROCm/librocdxg) to refresh your session.Dismiss alert

{{ message }}

### Uh oh!

There was an error while loading. [Please reload this page](https://github.com/ROCm/librocdxg).

[ROCm](https://github.com/ROCm)/ **[librocdxg](https://github.com/ROCm/librocdxg)** Public

- [Notifications](https://github.com/login?return_to=%2FROCm%2Flibrocdxg) You must be signed in to change notification settings
- [Fork\\
15](https://github.com/login?return_to=%2FROCm%2Flibrocdxg)
- [Star\\
137](https://github.com/login?return_to=%2FROCm%2Flibrocdxg)


develop

[**6** Branches](https://github.com/ROCm/librocdxg/branches) [**7** Tags](https://github.com/ROCm/librocdxg/tags)

[Go to Branches page](https://github.com/ROCm/librocdxg/branches)[Go to Tags page](https://github.com/ROCm/librocdxg/tags)

Go to file

Code

Open more actions menu

## Latest commit

[![fcui-amd](https://avatars.githubusercontent.com/u/229691314?v=4&size=40)](https://github.com/fcui-amd)[fcui-amd](https://github.com/ROCm/librocdxg/commits?author=fcui-amd)

[topology: set AQL emulation over PM4 capability](https://github.com/ROCm/librocdxg/commit/6612a86f5279f557d04bf6a50db616f675d4b170)

Open commit details

3 days agoAug 18, 2026

[6612a86](https://github.com/ROCm/librocdxg/commit/6612a86f5279f557d04bf6a50db616f675d4b170) · 3 days agoAug 18, 2026

## History

[368 Commits](https://github.com/ROCm/librocdxg/commits/develop/)

Open commit details

[View commit history for this file.](https://github.com/ROCm/librocdxg/commits/develop/) 368 Commits

## Folders and files

| Name | Name | Last commit message | Last commit date |
| --- | --- | --- | --- |
| [DEBIAN](https://github.com/ROCm/librocdxg/tree/develop/DEBIAN "DEBIAN") | [DEBIAN](https://github.com/ROCm/librocdxg/tree/develop/DEBIAN "DEBIAN") | [wsl/hsakmt: initial commit](https://github.com/ROCm/librocdxg/commit/aa5a08b1d749a2e118c95203d0885ff9fb4799ff "wsl/hsakmt: initial commit  Signed-off-by: lyndonli <Lyndon.Li@amd.com> Signed-off-by: Horatio Zhang <Hongkun.Zhang@amd.com> Signed-off-by: Shi.Leslie <Yuliang.Shi@amd.com> Signed-off-by: LonglongYao <Longlong.Yao@amd.com> Signed-off-by: tiancyin <tianci.yin@amd.com> Signed-off-by: Frank Min <Frank.Min@amd.com> Signed-off-by: Aaron Liu <aaron.liu@amd.com> Signed-off-by: Shane Xiao <shane.xiao@amd.com> Signed-off-by: Lang Yu <lang.yu@amd.com> Signed-off-by: Feifei Xu <Feifei.Xu@amd.com> Signed-off-by: Ruili Ji <ruiliji2@amd.com> Signed-off-by: Qiang Yu <qiang.yu@amd.com> Signed-off-by: Flora Cui <flora.cui@amd.com>") | 9 months agoNov 5, 2025 |
| [amdsmi](https://github.com/ROCm/librocdxg/tree/develop/amdsmi "amdsmi") | [amdsmi](https://github.com/ROCm/librocdxg/tree/develop/amdsmi "amdsmi") | [amdsmi: load WSL env for interactive bash shells](https://github.com/ROCm/librocdxg/commit/846768a851ea50ceffbb9e50516f5970755beb69 "amdsmi: load WSL env for interactive bash shells  Ensure rocdxg-amd-smi-lib environment setup applies to both login shells and docker exec interactive bash sessions, and make env exports idempotent to avoid duplicate path entries.  Signed-off-by: Flora Cui <flora.cui@amd.com>") | 2 months agoJul 1, 2026 |
| [cmake\_modules](https://github.com/ROCm/librocdxg/tree/develop/cmake_modules "cmake_modules") | [cmake\_modules](https://github.com/ROCm/librocdxg/tree/develop/cmake_modules "cmake_modules") | [cmake: share PACKAGE\_VERSION\_SUFFIX for deb package versions](https://github.com/ROCm/librocdxg/commit/f2f2beed175c10a8ed420a343d6bec78fbd816d6 "cmake: share PACKAGE_VERSION_SUFFIX for deb package versions  Replace ROCDXG_VERSION_PRERELEASE with a common PACKAGE_VERSION_SUFFIX and apply_package_version_suffix() helper used by rocdxg and amdsmi.  Signed-off-by: Flora Cui <flora.cui@amd.com>") | 3 months agoJun 2, 2026 |
| [data](https://github.com/ROCm/librocdxg/tree/develop/data "data") | [data](https://github.com/ROCm/librocdxg/tree/develop/data "data") | [packaging: install share/rocdxg/dids.conf as conffile](https://github.com/ROCm/librocdxg/commit/2428c61ddc192446e873b8e11f226b0750459458 "packaging: install share/rocdxg/dids.conf as conffile  Ship the default user DID template from data/dids.conf to share/rocdxg at install time and register it as a Debian conffile so local edits survive package upgrades.  Signed-off-by: Flora Cui <flora.cui@amd.com>") | 3 months agoMay 29, 2026 |
| [include](https://github.com/ROCm/librocdxg/tree/develop/include "include") | [include](https://github.com/ROCm/librocdxg/tree/develop/include "include") | [wsl/librocdxg: Add API hsaKmtGetCoreDeviceInfo](https://github.com/ROCm/librocdxg/commit/5229bcf5354becb8208bd617404e4b19e5044ea0 "wsl/librocdxg: Add API hsaKmtGetCoreDeviceInfo  Export the API for ROCr thunk loader; return NOT_SUPPORTED on the WSL DXG path.  Signed-off-by: Yang Su <Yang.Su2@amd.com>") | last monthJul 17, 2026 |
| [shared](https://github.com/ROCm/librocdxg/tree/develop/shared "shared") | [shared](https://github.com/ROCm/librocdxg/tree/develop/shared "shared") | [shared: add missing FARPROC typedef for Windows SDK >= 10.0.28000.0](https://github.com/ROCm/librocdxg/commit/fe3456032dc3e53dd701190f88b39c345edaeb79 "shared: add missing FARPROC typedef for Windows SDK >= 10.0.28000.0") | 2 months agoJun 8, 2026 |
| [src](https://github.com/ROCm/librocdxg/tree/develop/src "src") | [src](https://github.com/ROCm/librocdxg/tree/develop/src "src") | [topology: set AQL emulation over PM4 capability](https://github.com/ROCm/librocdxg/commit/6612a86f5279f557d04bf6a50db616f675d4b170 "topology: set AQL emulation over PM4 capability  Signed-off-by: Flora Cui <flora.cui@amd.com>") | 3 days agoAug 18, 2026 |
| [CMakeLists.txt](https://github.com/ROCm/librocdxg/blob/develop/CMakeLists.txt "CMakeLists.txt") | [CMakeLists.txt](https://github.com/ROCm/librocdxg/blob/develop/CMakeLists.txt "CMakeLists.txt") | [cmake: share PACKAGE\_VERSION\_SUFFIX for deb package versions](https://github.com/ROCm/librocdxg/commit/f2f2beed175c10a8ed420a343d6bec78fbd816d6 "cmake: share PACKAGE_VERSION_SUFFIX for deb package versions  Replace ROCDXG_VERSION_PRERELEASE with a common PACKAGE_VERSION_SUFFIX and apply_package_version_suffix() helper used by rocdxg and amdsmi.  Signed-off-by: Flora Cui <flora.cui@amd.com>") | 3 months agoJun 2, 2026 |
| [CONTRIBUTING.md](https://github.com/ROCm/librocdxg/blob/develop/CONTRIBUTING.md "CONTRIBUTING.md") | [CONTRIBUTING.md](https://github.com/ROCm/librocdxg/blob/develop/CONTRIBUTING.md "CONTRIBUTING.md") | [wsl/librocdxg: add CONTRIBUTING.md](https://github.com/ROCm/librocdxg/commit/f71feb692250a26f1e31f298c884dd0c0a897e44 "wsl/librocdxg: add CONTRIBUTING.md  Signed-off-by: Flora Cui <flora.cui@amd.com>") | 8 months agoDec 17, 2025 |
| [LICENSE.md](https://github.com/ROCm/librocdxg/blob/develop/LICENSE.md "LICENSE.md") | [LICENSE.md](https://github.com/ROCm/librocdxg/blob/develop/LICENSE.md "LICENSE.md") | [wsl/hsakmt: rename rocr\_proxy to thunk\_proxy](https://github.com/ROCm/librocdxg/commit/8e06c833827e027023874086550b36bb81438ab0 "wsl/hsakmt: rename rocr_proxy to thunk_proxy  Signed-off-by: Longlong Yao <Longlong.Yao@amd.com> Part-of: <http://10.67.69.192/wsl/libhsakmt/-/merge_requests/26>") | 9 months agoNov 5, 2025 |
| [README.md](https://github.com/ROCm/librocdxg/blob/develop/README.md "README.md") | [README.md](https://github.com/ROCm/librocdxg/blob/develop/README.md "README.md") | [docs(readme): update compatibility matrix](https://github.com/ROCm/librocdxg/commit/4c9e8bccc9954878864b02f739acb398999194a9 "docs(readme): update compatibility matrix  Signed-off-by: Flora Cui <flora.cui@amd.com>") | last monthJul 15, 2026 |
| [VERSION](https://github.com/ROCm/librocdxg/blob/develop/VERSION "VERSION") | [VERSION](https://github.com/ROCm/librocdxg/blob/develop/VERSION "VERSION") | [version: bump to 1.2.2](https://github.com/ROCm/librocdxg/commit/4955d12888a3ec57057f1cf8660c2485e415e74c "version: bump to 1.2.2  Signed-off-by: Flora Cui <flora.cui@amd.com>") | 2 weeks agoAug 5, 2026 |
| [librocdxg.pc.in](https://github.com/ROCm/librocdxg/blob/develop/librocdxg.pc.in "librocdxg.pc.in") | [librocdxg.pc.in](https://github.com/ROCm/librocdxg/blob/develop/librocdxg.pc.in "librocdxg.pc.in") | [librocdxg: rename hsakmt to rocdxg](https://github.com/ROCm/librocdxg/commit/f78387b62de3efb38d25aa021b9903d6320aa9fc "librocdxg: rename hsakmt to rocdxg  Signed-off-by: Flora Cui <flora.cui@amd.com>") | 9 months agoNov 5, 2025 |
| [rocdxg-config.cmake.in](https://github.com/ROCm/librocdxg/blob/develop/rocdxg-config.cmake.in "rocdxg-config.cmake.in") | [rocdxg-config.cmake.in](https://github.com/ROCm/librocdxg/blob/develop/rocdxg-config.cmake.in "rocdxg-config.cmake.in") | [librocdxg: rename hsakmt to rocdxg](https://github.com/ROCm/librocdxg/commit/f78387b62de3efb38d25aa021b9903d6320aa9fc "librocdxg: rename hsakmt to rocdxg  Signed-off-by: Flora Cui <flora.cui@amd.com>") | 9 months agoNov 5, 2025 |
| View all files |

## Repository files navigation

# AMD ROCDXG Libary

[Permalink: AMD ROCDXG Libary](https://github.com/ROCm/librocdxg#amd-rocdxg-libary)

A user-mode library that enables ROCm functionality on Windows Subsystem for Linux (WSL). This library allows users to run GPU-accelerated Linux workloads under WSL, supporting AI, HPC, and other experimental use cases.

## Prerequisites

[Permalink: Prerequisites](https://github.com/ROCm/librocdxg#prerequisites)

- Download the compatible Windows driver from [AMD Drivers](https://www.amd.com/en/support/download/drivers.html)
- Download and install the latest stable version of WSL2 [WSL Install](https://learn.microsoft.com/en-us/windows/wsl/install)
- The following tools are required to build librocdxg:
  - CMake >= 3.15
  - GCC >= 11.4

## Quickstart

[Permalink: Quickstart](https://github.com/ROCm/librocdxg#quickstart)

### 1\. Install AMD ROCm package

[Permalink: 1. Install AMD ROCm package](https://github.com/ROCm/librocdxg#1-install-amd-rocm-package)

Install the ROCm package by following the official ROCm Installation Quick Guide:

[ROCm Installation Quick Start](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html)

> _**Note**_
>
> - This step may take several minutes, depending on internet connection and system speed.
> - Follow the quick-start guide for package repository setup and ROCm package installation.
> - **Important**: Post-installation validation (Step 3) must only be performed after the successful completion of **Step 2**. Executing the validation prior to this will lead to failure.

### 2\. Install librocdxg

[Permalink: 2. Install librocdxg](https://github.com/ROCm/librocdxg#2-install-librocdxg)

> _**Note**_
>
> - For legacy ROCm releases, `HSA_ENABLE_DXG_DETECTION=1` MUST be set; this requirement is removed starting with the ROCk release 7.13. It applies to both installation options below.
>
>
>
>   ```
>   export HSA_ENABLE_DXG_DETECTION=1
>   ```

#### Option A — Build from source

[Permalink: Option A — Build from source](https://github.com/ROCm/librocdxg#option-a--build-from-source)

1. Install Windows SDK

Download and install the Windows SDK from [Windows SDK](https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/).

2. Clone the librocdxg repository to your local WSL.

```
git clone https://github.com/ROCm/librocdxg.git
cd librocdxg
```

3. Build and install librocdxg.

```
# Set the Windows SDK path (adjust version number if different)
export win_sdk='/mnt/c/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0'

# Build the library
mkdir -p build
cd build
cmake .. -DWIN_SDK="${win_sdk}/shared"
make
sudo make install
```

> _**Note**_
>
> - The Windows SDK path may vary depending on the version you installed. Common locations include:
>   - C:\\Program Files (x86)\\Windows Kits\\10\\Include\\10.0.26100.0\
> - Ensure you have the necessary permissions to access the Windows SDK directory from WSL

#### Option B — Install pre-built deb package

[Permalink: Option B — Install pre-built deb package](https://github.com/ROCm/librocdxg#option-b--install-pre-built-deb-package)

Download the `rocdxg-roct` runtime package from [GitHub Releases](https://github.com/ROCm/librocdxg/releases):

```
sudo dpkg -i rocdxg-roct_<version>_amd64.deb
```

### 3\. Post-install verification checks

[Permalink: 3. Post-install verification checks](https://github.com/ROCm/librocdxg#3-post-install-verification-checks)

Run these post-installation checks to verify that the installation is complete.

Check if the GPU is listed as an agent:

```
rocminfo
```

Expected result:

```
[...]
*******
Agent 2
*******
  Name:                    gfx1100
  Marketing Name:          Radeon RX 7900 XTX
  Vendor Name:             AMD
  [...]
[...]
```

### 4\. Container Launch – WSL-Specific Flags

[Permalink: 4. Container Launch – WSL-Specific Flags](https://github.com/ROCm/librocdxg#4-container-launch--wsl-specific-flags)

When you launch the container, add these WSL-specific arguments (they do not replace the native-Linux GPU flags):

| Flag | Purpose |
| --- | --- |
| `--device /dev/dxg` | Pass the `/dev/dxg` device node into the container so applications inside the container can access the GPU. |
| `-v /usr/lib/wsl/lib/libdxcore.so:/usr/lib/libdxcore.so`<br>`-v /opt/rocm/lib/librocdxg.so:/usr/lib/librocdxg.so`<br>`-v /opt/rocm/share/rocdxg/dids.conf:/usr/share/rocdxg/dids.conf` | Make the AMD ROCDXG and Microsoft DXCore libraries available inside the container so that ROCm/HIP applications can route their GPU compute calls through ROCDXG and DXCore to communicate with the GPU. |
| `-e HSA_ENABLE_DXG_DETECTION=1` | For legacy ROCm releases, HSA\_ENABLE\_DXG\_DETECTION=1 MUST be set; this requirement is removed starting with the ROCk release 7.13. |

Example docker run command:

```
docker run -it  \
    -v /usr/lib/wsl/lib/libdxcore.so:/usr/lib/libdxcore.so \
    -v /opt/rocm/lib/librocdxg.so:/usr/lib/librocdxg.so \
    -v /opt/rocm/share/rocdxg/dids.conf:/usr/share/rocdxg/dids.conf \
    --device=/dev/dxg \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --ipc=host \
    --shm-size 8G \
    rocm/pytorch:latest
```

> _**Note**_
>
> - For ROCm releases prior to 7.13, pass `-e HSA_ENABLE_DXG_DETECTION=1` to the `docker run` command:
>
>
>
>   ```
>   docker run -it  \
>       -v /usr/lib/wsl/lib/libdxcore.so:/usr/lib/libdxcore.so \
>       -v /opt/rocm/lib/librocdxg.so:/usr/lib/librocdxg.so \
>       -v /opt/rocm/share/rocdxg/dids.conf:/usr/share/rocdxg/dids.conf \
>       -e HSA_ENABLE_DXG_DETECTION=1 \
>       --device=/dev/dxg \
>       --cap-add=SYS_PTRACE \
>       --security-opt seccomp=unconfined \
>       --ipc=host \
>       --shm-size 8G \
>       rocm/pytorch:latest
>   ```

## Known Issues and Limitations

[Permalink: Known Issues and Limitations](https://github.com/ROCm/librocdxg#known-issues-and-limitations)

- JAX is supported from version 0.9.1 onwards.
- AMD-SMI currently provides a limited set of features on WSL2. The source code is available in the develop branch, and a formal release plan is under development.
- Debugging/Profiling: `ROCm-profiler`, `Debugger` are not supported.
- vLLM can fail during initialization on WSL2 when using the V2 model runner because pinned memory is disabled by default under WSL, while the V2 runner requires pinned memory/UVA support. This may result in errors such as `RuntimeError: UVA is not available`. The issue is addressed by the vLLM fix in [vLLM PR #41496](https://github.com/vllm-project/vllm/pull/41496).

## WSL Compatibility Matrix

[Permalink: WSL Compatibility Matrix](https://github.com/ROCm/librocdxg#wsl-compatibility-matrix)

- Windows 11
- Ubuntu 26.04 LTS / Ubuntu 24.04 LTS / Ubuntu 22.04 LTS
- The AMD ROCDXG library utilizes a ROCm runtime feature introduced in ROCm 7.1, which loads _**librocdxg**_ to enable ROCm functionality within the WSL environment. This design keeps the _**librocdxg**_ solution loosely coupled with both AMD ROCm release and Windows display driver. As a result, the AMD ROCDXG library can evolve independently, following its own development schedule without impacting the existing ROCm solution.

| AMD Rocdxg Lib Version | AMD ROCm Version | AMD Windows Driver Version | Supported AMD GPU Products |
| --- | --- | --- | --- |
| 1.2.1 | 7.14 | AMD Windows x86 drivers<br>can be directly downloaded<br>from [AMD Driver](https://www.amd.com/en/support/download/drivers.html) | Additional AMD GPU support includes all ASICs for previous versions, plus the following:<br>_**Ryzen**_<br>AMD Ryzen™ AI 5 PRO 435<br>AMD Ryzen™ AI 7 445<br>AMD Ryzen™ AI 5 435<br>AMD Ryzen™ AI 5 430<br>AMD Ryzen AI MAX+ PRO 495<br>AMD Ryzen AI MAX PRO 490<br>AMD Ryzen AI MAX PRO 485<br>AMD Ryzen AI Halo |
| 1.2.0 | 7.13 | AMD Windows x86 drivers<br>can be directly downloaded<br>from [AMD Driver](https://www.amd.com/en/support/download/drivers.html) | Additional AMD GPU support includes all ASICs for previous versions, plus the following:<br>_**Ryzen**_<br>AMD Ryzen AI 7 PRO 360<br>AMD Ryzen AI 7 PRO 350<br>AMD Ryzen AI 5 PRO 340<br>AMD Ryzen AI 7 350<br>AMD Ryzen AI 7 345<br>AMD Ryzen AI 5 340<br>AMD Ryzen AI 5 330 |
| 1.2.0 | 7.2.x | AMD Windows x86 drivers<br>can be directly downloaded<br>from [AMD Driver](https://www.amd.com/en/support/download/drivers.html) | _**Radeon**_<br>AMD Radeon RX 9070<br>AMD Radeon RX 9070 XT<br>AMD Radeon RX 9070 GRE<br>AMD Radeon AI PRO R9700<br>AMD Radeon RX 9060<br>AMD Radeon RX 9060 XT<br>AMD Radeon RX 7900 XTX<br>AMD Radeon RX 7900 XT<br>AMD Radeon RX 7900 GRE<br>AMD Radeon PRO W7900<br>AMD Radeon PRO W7900 Dual Slot<br>AMD Radeon PRO W7800<br>AMD Radeon PRO W7800 48GB<br>AMD Radeon RX 7800 XT<br>AMD Radeon PRO W7700<br>_**Ryzen**_<br>AMD Ryzen AI Max+ 395<br>AMD Ryzen AI Max 390<br>AMD Ryzen AI Max 385<br>AMD Ryzen AI 9 HX 375<br>AMD Ryzen AI 9 HX 370<br>AMD Ryzen AI 9 365 |

## Documentation

[Permalink: Documentation](https://github.com/ROCm/librocdxg#documentation)

For detailed documentation—including ROCm installation guides, configuration options, and metric descriptions—see " [Use ROCm on Radeon and Ryzen](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/index.html#)".

## Contributing

[Permalink: Contributing](https://github.com/ROCm/librocdxg#contributing)

See [CONTRIBUTING.md](https://github.com/ROCm/librocdxg/blob/develop/CONTRIBUTING.md) for guidelines on setting up your WSL environment, building, and submitting pull requests.

## About

repo for AMD ROCDXG project

### Resources

[Readme](https://github.com/ROCm/librocdxg#readme-ov-file)

[License](https://github.com/ROCm/librocdxg#License-1-ov-file)

### Contributing

[Contributing](https://github.com/ROCm/librocdxg#contributing-ov-file)

[Activity](https://github.com/ROCm/librocdxg/activity)

[Custom properties](https://github.com/ROCm/librocdxg/custom-properties)

### Stars

**137** stars

### Watchers

**1** watching

### Forks

[**15** forks](https://github.com/ROCm/librocdxg/forks)

[Report repository](https://github.com/contact/report-content?content_url=https%3A%2F%2Fgithub.com%2FROCm%2Flibrocdxg&report=ROCm+%28user%29)

## Releases

## Packages

## Contributors

## Languages

You can’t perform that action at this time.