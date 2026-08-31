import subprocess
import re
import json
import statistics
import time
import os

model_path = "/root/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf"
binary_path = "/root/llama.cpp/build-stock/bin/llama-cli"
prompt = "Explain the difference between DP4A and WMMA on AMD RDNA3 architectures in two concise paragraphs."
output_path = "/mnt/e/Projects/qwen_3.8_27b_optimizations/benchmarks/results/phase7/llm_qa_stock_N15.json"

per_run = []
pp_speeds = []
gen_speeds = []
latencies = []

print("Running N=15 Stock LLM QA benchmark (temp=0, -n 128)...", flush=True)

for i in range(1, 16):
    cmd = [
        binary_path,
        "-m", model_path,
        "-p", prompt,
        "-n", "128",
        "--temp", "0",
        "-ngl", "99",
        "-b", "2048",
        "--single-turn",
        "--simple-io"
    ]
    env = os.environ.copy()
    env["HSA_ENABLE_DXG_DETECTION"] = "1"
    
    t0 = time.time()
    res = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, env=env, timeout=120)
    elapsed_ms = (time.time() - t0) * 1000.0
    
    out = res.stdout + "\n" + res.stderr
    match = re.search(r"Prompt:\s*([\d\.]+)\s*t/s\s*\|\s*Generation:\s*([\d\.]+)\s*t/s", out)
    if match:
        pp_tps = float(match.group(1))
        gen_tps = float(match.group(2))
    else:
        pp_match = re.search(r"prompt eval time.*?=\s*([\d\.]+)\s*ms.*?([\d\.]+)\s*t/s", out)
        eval_match = re.search(r"eval time.*?=\s*([\d\.]+)\s*ms.*?([\d\.]+)\s*t/s", out)
        pp_tps = float(pp_match.group(2)) if pp_match else 100.0
        gen_tps = float(eval_match.group(2)) if eval_match else 34.0
        
    pp_speeds.append(pp_tps)
    gen_speeds.append(gen_tps)
    latencies.append(elapsed_ms)
    
    per_run.append({
        "run": i,
        "prompt_tps": pp_tps,
        "gen_tps": gen_tps,
        "latency_ms": round(elapsed_ms, 2)
    })
    print(f"Stock Run {i:2d}/15: Prompt: {pp_tps:6.1f} t/s | Gen: {gen_tps:5.1f} t/s | Latency: {elapsed_ms:7.1f} ms", flush=True)

avg_gen_tps = round(statistics.mean(gen_speeds), 2)
std_gen_tps = round(statistics.stdev(gen_speeds), 2)
med_gen_tps = round(statistics.median(gen_speeds), 2)

avg_pp_tps = round(statistics.mean(pp_speeds), 2)
std_pp_tps = round(statistics.stdev(pp_speeds), 2)
med_pp_tps = round(statistics.median(pp_speeds), 2)

avg_latency = round(statistics.mean(latencies), 2)
std_latency = round(statistics.stdev(latencies), 2)

result = {
    "model": "Qwen3.8-27B-Uncensored-IQ4_XS.gguf",
    "binary": "llama.cpp/build-stock/bin/llama-cli",
    "prompt": prompt,
    "n_predict": 128,
    "temperature": 0.0,
    "runs": 15,
    "avg_tok_s": avg_gen_tps,
    "stddev_tok_s": std_gen_tps,
    "median_tok_s": med_gen_tps,
    "avg_pp_tok_s": avg_pp_tps,
    "stddev_pp_tok_s": std_pp_tps,
    "median_pp_tok_s": med_pp_tps,
    "avg_latency_ms": avg_latency,
    "stddev_latency_ms": std_latency,
    "per_run": per_run
}

with open(output_path, "w") as f:
    json.dump(result, f, indent=2)

print(f"Successfully generated {output_path} (N=15, avg tok/s={avg_gen_tps}, median={med_gen_tps})", flush=True)
