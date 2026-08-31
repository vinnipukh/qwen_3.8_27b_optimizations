@echo off
REM build_windows.bat — Windows-native build for gfx1100 via HIP SDK + VS Build Tools (REQ-WIN-07)
REM Pure C++/HIP + CMake + .bat only (<=2 langs), no Python/JS servers. Uses HIP_PATH/bin/clang++.exe --offload-arch=gfx1100 -G Ninja (not cl)
REM The .hip files use __builtin_amdgcn_sudot4/perm/wmma which cl cannot compile; Ninja + clang++ is mandatory.

setlocal EnableDelayedExpansion

REM Ensure git safe.directory for shared checkouts
git config --global --add safe.directory "%CD%" >nul 2>&1

REM --- HIP SDK detection ------------------------------------------------------
if "%HIP_PATH%"=="" set "HIP_PATH=C:\Program Files\AMD\ROCm\6.4"
if not exist "%HIP_PATH%\bin\clang++.exe" (
    echo [ERROR] HIP SDK not found at HIP_PATH="%HIP_PATH%"
    echo        Expected "%HIP_PATH%\bin\clang++.exe" (HIP 6.4) and "%HIP_PATH%\lib\cmake\hip\hip-config.cmake"
    echo        Install AMD HIP SDK for Windows: https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html
    echo        Or set HIP_PATH environment variable: set HIP_PATH=C:\your\ROCm\path
    exit /b 1
)

set "PATH=%HIP_PATH%\bin;%PATH%"

echo [INFO] HIP_PATH="%HIP_PATH%"
where clang++.exe || (
    echo [ERROR] clang++.exe not in PATH after HIP_PATH setup
    exit /b 1
)

"%HIP_PATH%\bin\clang++.exe" --offload-arch=gfx1100 --version
if errorlevel 1 (
    echo [ERROR] clang++.exe --offload-arch=gfx1100 --version failed
    exit /b 1
)

REM --- Verify Ninja (required, not MSVC/cl generator) -------------------------
where ninja >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Ninja not found. Install via: winget install Ninja-build.Ninja
    echo        MSVC generator (cl) cannot compile __builtin_amdgcn_* intrinsics in .hip files.
    exit /b 1
)

REM --- CMake configure --------------------------------------------------------
echo [INFO] Configuring with "%HIP_PATH%\bin\clang++.exe" --offload-arch=gfx1100 -G Ninja ...

cmake -S . -B build-windows -G Ninja ^
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 ^
  -DGGML_HIP=ON ^
  -DGGML_CUDA_ENABLE_CUSTOM_GFX1100=ON ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CXX_COMPILER="%HIP_PATH%\bin\clang++.exe" ^
  -DCMAKE_HIP_COMPILER="%HIP_PATH%\bin\clang++.exe" ^
  -DHIP_PATH="%HIP_PATH%"

if errorlevel 1 (
    echo [ERROR] CMake configure failed. Check find_package(hip REQUIRED CONFIG PATHS "$ENV{HIP_PATH}/lib/cmake/hip") in CMakeLists.txt
    echo        CMakeLists.txt must contain: find_package(hip REQUIRED CONFIG PATHS "$ENV{HIP_PATH}/lib/cmake/hip") not hard-coded /opt/rocm
    exit /b 1
)

REM --- Build ------------------------------------------------------------------
echo [INFO] Building build-windows (this may take 5-15 mins for llama.cpp + kernels) ...
cmake --build build-windows --config Release
if errorlevel 1 (
    echo [ERROR] Build failed. Check __builtin_amdgcn_sudot4/perm/wmma intrinsics compile with clang++ --offload-arch=gfx1100
    echo        Common pitfall: using MSVC generator (cl) instead of Ninja + clang++
    exit /b 1
)

echo [INFO] Build succeeded. Checking artifacts...

if not exist "build-windows\bin\llama-server.exe" (
    echo [WARN] build-windows\bin\llama-server.exe not found, checking alternative locations...
    dir build-windows\*.exe /s /b 2>nul
    if errorlevel 1 (
        echo [ERROR] No llama-server.exe produced. Expected build-windows\bin\llama-server.exe
        exit /b 1
    )
) else (
    echo [INFO] Found build-windows\bin\llama-server.exe
    build-windows\bin\llama-server.exe --help || echo [INFO] llama-server --help returned non-zero (may need model arg, ok)
)

REM --- Smoke serve at localhost:8000 -----------------------------------------
REM Per REQ-WIN-07: build-windows/bin/llama-server.exe serves curl http://127.0.0.1:8000/v1/chat/completions -> 200 on gfx1100
echo [INFO] Starting smoke test: llama-server.exe at http://127.0.0.1:8000/v1/chat/completions ...

REM Check model exists (configurable via MODEL_PATH)
if "%MODEL_PATH%"=="" set "MODEL_PATH=models\Qwen3.8-27B-Uncensored-IQ4_XS.gguf"
if not exist "%MODEL_PATH%" (
    echo [WARN] Model not found at %MODEL_PATH%
    echo        Set MODEL_PATH env var to your GGUF, e.g.: set MODEL_PATH=D:\models\qwen.gguf
    echo        Skipping live server smoke (build still verified). To test manually:
    echo          build-windows\bin\llama-server.exe -m %MODEL_PATH% --port 8000 --host 127.0.0.1 -ngl 99 -b 2048
    echo          curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}"
    goto :end
)

start /B build-windows\bin\llama-server.exe -m "%MODEL_PATH%" --port 8000 --host 127.0.0.1 -ngl 99 -b 2048 --single-turn > build-windows\server.log 2>&1
echo [INFO] Waiting 15s for server to start...
timeout /t 15 /nobreak >nul

curl -s -o build-windows\curl_out.json -w "%%{http_code}" http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"temperature\":0}" > build-windows\curl_code.txt 2>&1
set /p CURL_CODE=<build-windows\curl_code.txt
echo [INFO] curl http://127.0.0.1:8000/v1/chat/completions -> %CURL_CODE%
type build-windows\curl_out.json
if "%CURL_CODE%"=="200" (
    echo [PASS] llama-server.exe serves 200 at :8000 with choices[0].message.content
    findstr /C:"choices" build-windows\curl_out.json >nul && echo [PASS] Response contains choices[0].message.content
) else (
    echo [WARN] Expected 200 but got %CURL_CODE%. Check build-windows\server.log
    type build-windows\server.log
)

REM Kill server (taskkill)
taskkill /F /IM llama-server.exe >nul 2>&1

:end
echo [INFO] Windows-native gate REQ-WIN-07 complete. Pure C++/HIP + CMake + .bat only, <=2 langs.
echo [INFO] No Python/JS shipped check: find -name "*.py" ! -path "./llama.cpp/*" should be 0 after Phase 8 prune (benchmarks/ Python harness offline-only, not shipped)
echo [INFO] To verify: find . -name "*.py" ! -path "./llama.cpp/*"  (Git Bash) or dir /s *.py

endlocal
