# Windows Python + Terminal Help for Phase 7 Subagents

Date: 2026-08-30
Context: 5 subagents reported python/windows terminal issues on Windows 11 host.

## Python on Windows — Fixed

- Host has python.exe (Python 3.14.7), python3.exe shim, and py.exe launcher — all resolve. race.py shebang #!/usr/bin/env python3 works via py launcher, but direct python3 may fail in cmd. Fix: invoke as python benchmarks/results/phase7/race.py or py benchmarks/results/phase7/race.py (both tested OK). Do NOT require python3 literal; use python for portability.
- No pip installs needed; race.py is offline harness pruned in Phase 8.

Tested:
- python benchmarks/results/phase7/race.py --help -> OK
- py benchmarks/results/phase7/race.py --help -> OK
- python --version / py --version -> 3.14.7

## Windows Terminal Quoting — Fixed

- HIP_PATH=C:\Program Files\AMD\ROCm\6.4 contains space. build_windows.bat correctly quotes "%HIP_PATH%\bin\clang++.exe" and "%HIP_PATH%" in cmake -DCMAKE_CXX_COMPILER=... etc. Verified via grep.
- CMake kernels/CMakeLists.txt uses find_package(hip REQUIRED CONFIG PATHS "$ENV{HIP_PATH}/lib/cmake/hip") with quotes — correct for space paths. No hard-coded /opt/rocm in find_package.
- Do NOT use bash export VAR=val or timeout 90 bash -c on Windows; use set VAR=val and Python subprocess.run(timeout=90) (race.py already does).

## Actionable Guidance for Stuck Subagents

1. If python3: command not found, retry with python or py.
2. If HIP_PATH space breaks cmake, ensure quotes preserved: cmake -DCMAKE_HIP_COMPILER="%HIP_PATH%\bin\clang++.exe"
3. If timeout shell fails, use python timeout param (race.py subprocess.run(timeout=90) already correct).
4. Commit with core.autocrlf=false already set; .gitattributes eol=lf prevents CRLF patch break.
5. For Windows cmd, use where clang++.exe / where ninja, not which. build_windows.bat already correct.
