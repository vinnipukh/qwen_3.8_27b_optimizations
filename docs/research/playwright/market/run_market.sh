set -e
PWCLI="playwright-cli"
OUT="E:/Projects/qwen_3.8_27b_optimizations/docs/research/playwright/market"
mkdir -p "$OUT"

run_extract() {
  url="$1"
  outname="$2"
  echo "=== OPEN $url -> $outname ==="
  "$PWCLI" --session market open "$url" 2>&1 | tail -n 5
  sleep 4
  "$PWCLI" --session market eval "() => document.documentElement.innerText.slice(0,15000)" 2>&1 | head -c 16000 > "$OUT/$outname.txt" || true
  echo "--- saved $outname head ---"
  head -n 80 "$OUT/$outname.txt" || true
  echo "=== done $outname ==="
}

run_extract "https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7900xt.html" "amd-7900xt-spec"
run_extract "https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7900-xtx.html" "amd-7900xtx-spec"
run_extract "https://rocm.blogs.amd.com/artificial-intelligence/llm-inference-optimization/README.html" "rocm-llm-inference-opt"
run_extract "https://github.com/microsoft/WSL/issues/40732" "wsl-40732-bsod"
run_extract "https://github.com/microsoft/WSL/issues/40401" "wsl-40401-vram-lie"
run_extract "https://gpuopen.com/learn/wmma_on_rdna3/" "gpuopen-wmma-rdna3"
run_extract "https://rocm.docs.amd.com/projects/rocWMMA/en/latest/" "rocm-rocwmma-docs"
echo "ALL_DONE"
