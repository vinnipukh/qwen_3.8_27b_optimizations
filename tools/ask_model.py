#!/usr/bin/env python3
import os
import subprocess
import sys

default_prompt = "<|im_start|>system\\nYou are an expert in computational neuroscience and neuromorphic computing. Provide a direct, detailed, rigorous explanation of integrated subcompartmentalized liquid neural networks.<|im_end|>\\n<|im_start|>user\\nExplain the concept, architecture, and theoretical mechanisms of integrated subcompartmentalized liquid neural networks in detail.<|im_end|>\\n<|im_start|>assistant\\n"
prompt = sys.argv[1] if len(sys.argv) > 1 else default_prompt

bin_path = "/root/llama.cpp/build-ci/bin/llama-cli"
model_path = "/root/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf"

env = dict(os.environ)
env["HSA_ENABLE_DXG_DETECTION"] = "1"
bin_dir = "/root/llama.cpp/build-ci/bin"
curr_ld = env.get("LD_LIBRARY_PATH", "")
env["LD_LIBRARY_PATH"] = f"{bin_dir}:{curr_ld}" if curr_ld else bin_dir

cmd = [
    bin_path,
    "-m", model_path,
    "-c", "8192",
    "-ngl", "99",
    "--temp", "0.3",
    "-n", "4500",
    "--single-turn",
    "--simple-io",
    "--no-display-prompt",
    "--load-mode", "none",
    "-p", prompt
]

print(f"Loading Qwen3.8-27B on RX 7900 XT (gfx1100) and running prompt...\n")
res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=240)

if res.returncode != 0:
    print(f"Error (code {res.returncode}):\n{res.stderr}")
    sys.exit(res.returncode)

print("=== Model Output ===")
print(res.stdout.strip())
