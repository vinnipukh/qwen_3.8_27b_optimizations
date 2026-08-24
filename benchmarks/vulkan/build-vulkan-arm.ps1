# PowerShell script to clone and build native-Windows Vulkan comparator arm (D2-02, D2-03)
param(
    [string]$TargetDir = "E:\vulkan-arm\llama.cpp",
    [string]$Commit = "bb4caa7540188872173c44d161602d9271386413",
    [string]$LogFile = "benchmarks\vulkan\build-log.txt"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Get-Item .).FullName
$LogPath = Join-Path $RepoRoot $LogFile

New-Item -ItemType Directory -Force -Path (Split-Path $LogPath) | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $TargetDir) | Out-Null

Write-Host "=== Building Native Windows Vulkan Arm ==="
Write-Host "Target: $TargetDir @ commit: $Commit"
Write-Host "Logging to: $LogPath"

# Step 1: Check compiler and tools
$CMake = Get-Command cmake -ErrorAction SilentlyContinue
$GLSLC = Get-Command glslc -ErrorAction SilentlyContinue

if (-not $CMake) {
    Write-Host "Error: cmake not found in PATH. Please install VS2022 / CMake." -ForegroundColor Red
    "ERROR: cmake not found in PATH" | Out-File -FilePath $LogPath -Encoding utf8
    exit 1
}

# Step 2: Clone if needed
if (-not (Test-Path "$TargetDir\.git")) {
    Write-Host "Cloning llama.cpp to $TargetDir ..."
    git clone https://github.com/ggml-org/llama.cpp $TargetDir 2>&1 | Tee-Object -FilePath $LogPath -Append
}

Push-Location $TargetDir
try {
    Write-Host "Checking out commit $Commit ..."
    git fetch origin 2>&1 | Tee-Object -FilePath $LogPath -Append
    git checkout $Commit 2>&1 | Tee-Object -FilePath $LogPath -Append

    $CurRev = (git rev-parse HEAD).Trim()
    if ($CurRev -ne $Commit) {
        throw "Commit mismatch! Expected $Commit but got $CurRev"
    }

    $BuildDir = Join-Path $TargetDir "build"
    New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

    Write-Host "Configuring CMake with GGML_VULKAN=ON ..."
    cmake -B $BuildDir `
        -G "Visual Studio 17 2022" `
        -A x64 `
        -DGGML_VULKAN=ON `
        -DCMAKE_BUILD_TYPE=Release `
        -DLLAMA_CURL=OFF `
        -DLLAMA_BUILD_SERVER=OFF 2>&1 | Tee-Object -FilePath $LogPath -Append

    Write-Host "Building Release binaries (llama-cli, llama-bench, test-backend-ops) ..."
    cmake --build $BuildDir --config Release --target llama-cli llama-bench test-backend-ops 2>&1 | Tee-Object -FilePath $LogPath -Append

    Write-Host "Build complete! Release binaries located in: $BuildDir\bin\Release"
} finally {
    Pop-Location
}
