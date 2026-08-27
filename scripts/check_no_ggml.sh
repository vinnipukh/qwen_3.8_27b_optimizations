#!/usr/bin/env bash
set -euo pipefail

# Isolation gate: Ensure kernels/ has ZERO ggml or llama headers / includes
# KERN-01 requirement: standalone HIP execution with zero llama.cpp headers

echo "Checking kernels/ directory for forbidden ggml/llama header includes..."

if grep -rnE --exclude-dir=build '#include\s*[<"](ggml|llama)' kernels/; then
    echo "ERROR: Found forbidden ggml/llama header include in kernels/!"
    exit 1
fi

# Check for #include "ggml" or "llama" specifically
if grep -rnE --exclude-dir=build '#include.*(ggml|llama)' kernels/; then
    echo "ERROR: Found forbidden ggml/llama include pattern in kernels/!"
    exit 1
fi

echo "PASS: Zero ggml/llama includes found in kernels/."
exit 0
