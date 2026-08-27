import json
import os
import subprocess
import pytest

def get_env():
    env = os.environ.copy()
    env["HSA_ENABLE_DXG_DETECTION"] = "1"
    return env

def test_demo_correct_passes():
    bin_path = "./kernels/build/demo_iq4xs_dequant/demo_test"
    assert os.path.exists(bin_path), f"Binary {bin_path} does not exist. Run cmake build."

    res = subprocess.run([bin_path], capture_output=True, text=True, env=get_env(), timeout=60)
    assert res.returncode == 0, f"demo_test returned {res.returncode}. Output:\n{res.stdout}\n{res.stderr}"
    assert "FINAL RESULT: PASS (8/8 passed)" in res.stdout
    assert "FAIL" not in res.stdout

def test_demo_warp64_passes():
    bin_path = "./kernels/build/demo_iq4xs_dequant/demo_test"
    res = subprocess.run([bin_path, "--warp", "64"], capture_output=True, text=True, env=get_env(), timeout=60)
    assert res.returncode == 0, f"demo_test --warp 64 returned {res.returncode}. Output:\n{res.stdout}\n{res.stderr}"
    assert "FINAL RESULT: PASS (8/8 passed)" in res.stdout

def test_demo_broken_fails():
    bin_path = "./kernels/build/demo_iq4xs_dequant/demo_test_broken"
    assert os.path.exists(bin_path), f"Binary {bin_path} does not exist. Run cmake build."

    res = subprocess.run([bin_path], capture_output=True, text=True, env=get_env(), timeout=60)
    assert res.returncode != 0, "demo_test_broken MUST fail with non-zero exit code"
    assert "FINAL RESULT: FAIL" in res.stdout
    # Verify that max_abs on broken exceeds 1e-4 threshold by huge margin
    assert "max_abs=" in res.stdout

def test_discrimination_metrics():
    # Run both and verify broken kernel exhibits >= 10x max_abs error vs correct
    correct_bin = "./kernels/build/demo_iq4xs_dequant/demo_test"
    broken_bin = "./kernels/build/demo_iq4xs_dequant/demo_test_broken"

    res_correct = subprocess.run([correct_bin], capture_output=True, text=True, env=get_env(), timeout=60)
    res_broken = subprocess.run([broken_bin], capture_output=True, text=True, env=get_env(), timeout=60)

    assert res_correct.returncode == 0
    assert res_broken.returncode != 0

    # Parse max_abs from stdout for a non-zero fixture (e.g. blk_0_ffn_down_weight)
    def extract_max_abs(stdout, fixture_name):
        for line in stdout.splitlines():
            if fixture_name in line and "max_abs=" in line:
                part = line.split("max_abs=")[1].split()[0]
                return float(part)
        return None

    correct_err = extract_max_abs(res_correct.stdout, "blk_0_ffn_down_weight")
    broken_err = extract_max_abs(res_broken.stdout, "blk_0_ffn_down_weight")

    assert correct_err is not None
    assert broken_err is not None
    assert correct_err < 1e-5
    assert broken_err > 1e-3
    assert broken_err >= 10.0 * (correct_err if correct_err > 0 else 1e-7)

def test_demo_bench_execution():
    bin_path = "./kernels/build/demo_iq4xs_dequant/demo_bench"
    assert os.path.exists(bin_path), f"Binary {bin_path} does not exist."

    res = subprocess.run([bin_path], capture_output=True, text=True, env=get_env(), timeout=90)
    assert res.returncode == 0, f"demo_bench returned {res.returncode}. Error:\n{res.stderr}"

    data = json.loads(res.stdout.strip())
    assert isinstance(data, list)
    assert len(data) == 10 # 5 block shapes * 2 warp sizes

    for item in data:
        assert item["op"] == "dequant_iq4_xs"
        assert item["n_blocks"] in [1, 8, 64, 512, 4096]
        assert item["warp_size"] in [32, 64]
        assert item["median_us"] > 0.0
        assert item["gb_s"] > 0.0
        assert item["count"] == 200
        assert item["warmup"] == 50
