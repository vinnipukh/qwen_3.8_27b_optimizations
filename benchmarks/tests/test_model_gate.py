import json
import pytest
from pathlib import Path
from benchmarks.bin import run_model_gate


MOCK_GOLDEN = {
    "recorded_at": "2026-08-24T00:00:00.000Z",
    "model_path": "/root/models/Qwen3.8-27B-Uncensored-IQ4_XS.gguf",
    "dataset_path": "benchmarks/data/wiki.test.raw",
    "dataset_sha256": "173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08",
    "perplexity": {
        "reference_ppl": 6.4271,
        "ppl_stddev": 0.04103,
        "chunks": 145,
        "allowed_tolerance_pct": 1.0,
        "min_allowed_ppl": 6.3628,
        "max_allowed_ppl": 6.4914,
    },
    "canaries": {
        "short_code_01.txt": {
            "prompt_sha256": "3b96a2c2b3482b7dd56ced1ae4b26f1707b7b26b4052e61e7efdfea07e85ddf8",
            "generated_text": "We need answer user. User provided code task: Implement fast vector dot product with AVX2 intrinsics.",
            "output_sha256": "0453d0ca2260f05ab7a830535ba5308b12dca6f5cbac7e90efb704d0b84e76ab",
            "returncode": 0,
        }
    },
}


def test_model_gate_valid_mock(tmp_path: Path):
    golden_path = tmp_path / "golden.json"
    out_json = tmp_path / "model_gate.json"
    with open(golden_path, "w") as f:
        json.dump(MOCK_GOLDEN, f)

    res = run_model_gate.evaluate_model_gate(
        golden_path=str(golden_path),
        out_json=str(out_json),
        mock_ppl=6.4200,
        mock_canaries={
            "short_code_01.txt": "We need answer user. User provided code task: Implement fast vector dot product with AVX2 intrinsics.",
        },
    )

    assert res["status"] == "PASS"
    assert res["perplexity"]["status"] == "PASS"
    assert out_json.exists()


def test_model_gate_ppl_out_of_bounds_fails(tmp_path: Path):
    golden_path = tmp_path / "golden.json"
    out_json = tmp_path / "model_gate_ppl_fail.json"
    with open(golden_path, "w") as f:
        json.dump(MOCK_GOLDEN, f)

    # 6.60 is > 6.4914 (+1% threshold)
    res = run_model_gate.evaluate_model_gate(
        golden_path=str(golden_path),
        out_json=str(out_json),
        mock_ppl=6.6000,
        mock_canaries={
            "short_code_01.txt": "We need answer user. User provided code task: Implement fast vector dot product with AVX2 intrinsics.",
        },
    )

    assert res["status"] == "FAIL"
    assert res["perplexity"]["status"] == "FAIL"


def test_model_gate_canary_mismatch_fails(tmp_path: Path):
    golden_path = tmp_path / "golden.json"
    out_json = tmp_path / "model_gate_canary_fail.json"
    with open(golden_path, "w") as f:
        json.dump(MOCK_GOLDEN, f)

    res = run_model_gate.evaluate_model_gate(
        golden_path=str(golden_path),
        out_json=str(out_json),
        mock_ppl=6.4271,
        mock_canaries={
            "short_code_01.txt": "CORRUPTED TOKEN OUTPUT MISMATCH",
        },
    )

    assert res["status"] == "FAIL"
    assert res["canaries"]["status"] == "FAIL"
    assert res["canaries"]["passed"] == 0


def test_saved_model_gate_and_golden_baseline():
    golden_file = Path("benchmarks/golden/stock_baseline_golden.json")
    gate_file = Path("benchmarks/results/phase3/model_gate.json")
    wiki_file = Path("benchmarks/data/wiki.test.raw")

    assert wiki_file.exists(), "wiki.test.raw dataset must exist"
    assert golden_file.exists(), "stock_baseline_golden.json must exist"
    assert gate_file.exists(), "model_gate.json must exist"

    with open(golden_file, "r") as f:
        gdata = json.load(f)
    assert gdata["perplexity"]["reference_ppl"] == 6.4271
    assert len(gdata["canaries"]) == 6

    with open(gate_file, "r") as f:
        mdata = json.load(f)
    assert mdata["status"] == "PASS"
    assert mdata["perplexity"]["status"] == "PASS"
    assert mdata["canaries"]["status"] == "PASS"
    assert mdata["canaries"]["passed"] == 6
