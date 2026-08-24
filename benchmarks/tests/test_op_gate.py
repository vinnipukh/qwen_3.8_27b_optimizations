import json
import pytest
from pathlib import Path
from benchmarks.bin import run_op_gate


MOCK_VALID_CSV = """ggml_cuda_init: found 1 ROCm devices
"backend_name","op_name","op_params","test_mode","supported","error_message","backend_reg_name"
"ROCm0","GATED_DELTA_NET","type=f32,head_count=32","test","1","",""
"ROCm0","SOLVE_TRI","type=f32,n=16","test","1","",""
"ROCm0","SSM_CONV","type=f32,dim=64","test","1","",""
"ROCm0","SSM_SCAN","type=f32,dim=64","test","1","",""
"ROCm0","FLASH_ATTN_EXT","type_K=f16","test","1","",""
"ROCm0","MUL_MAT","type_A=f16,type_B=f32","test","1","",""
"ROCm0","ADD","type=f16,ne=[1,1,8,1]","test","1","",""
"ROCm0","MUL","type=f16,ne=[1,1,8,1]","test","0","not supported",""
"""

MOCK_FAILING_OP_CSV = """ggml_cuda_init: found 1 ROCm devices
"backend_name","op_name","op_params","test_mode","supported","error_message","backend_reg_name"
"ROCm0","GATED_DELTA_NET","type=f32,head_count=32","test","1","",""
"ROCm0","SOLVE_TRI","type=f32,n=16","test","1","",""
"ROCm0","SSM_CONV","type=f32,dim=64","test","1","",""
"ROCm0","SSM_SCAN","type=f32,dim=64","test","1","",""
"ROCm0","FLASH_ATTN_EXT","type_K=f16","test","1","",""
"ROCm0","MUL_MAT","type_A=f16,type_B=f32","test","0","numerical mismatch max_abs=0.042",""
"ROCm0","ADD","type=f16,ne=[1,1,8,1]","test","1","",""
"""

MOCK_MISSING_CORE_OP_CSV = """ggml_cuda_init: found 1 ROCm devices
"backend_name","op_name","op_params","test_mode","supported","error_message","backend_reg_name"
"ROCm0","ADD","type=f16,ne=[1,1,8,1]","test","1","",""
"""


def test_op_gate_parsing_valid(tmp_path: Path):
    out_json = tmp_path / "op_gate.json"
    res = run_op_gate.run_op_gate(
        mock_csv=MOCK_VALID_CSV,
        mock_exit_code=0,
        out_json=out_json,
    )
    assert res["status"] == "PASS"
    assert res["total_cases"] == 8
    assert res["supported_cases"] == 7
    assert res["unsupported_cases"] == 1
    assert res["error_cases"] == 0
    assert out_json.exists()

    with open(out_json, "r") as f:
        saved = json.load(f)
    assert saved["status"] == "PASS"
    assert saved["core_ops_status"]["GATED_DELTA_NET"]["status"] == "PASS"


def test_op_gate_failing_op_triggers_fail(tmp_path: Path):
    out_json = tmp_path / "op_gate_fail.json"
    res = run_op_gate.run_op_gate(
        mock_csv=MOCK_FAILING_OP_CSV,
        mock_exit_code=0,
        out_json=out_json,
    )
    assert res["status"] == "FAIL"
    assert res["error_cases"] == 1
    assert len(res["errors"]) == 1
    assert res["errors"][0]["op"] == "MUL_MAT"
    assert "numerical mismatch" in res["errors"][0]["error"]
    assert res["core_ops_status"]["MUL_MAT"]["status"] == "FAIL"


def test_op_gate_missing_core_op_triggers_fail(tmp_path: Path):
    out_json = tmp_path / "op_gate_missing.json"
    res = run_op_gate.run_op_gate(
        mock_csv=MOCK_MISSING_CORE_OP_CSV,
        mock_exit_code=0,
        out_json=out_json,
    )
    assert res["status"] == "FAIL"
    assert res["core_ops_status"]["GATED_DELTA_NET"]["status"] == "FAIL"


def test_op_gate_nonzero_exit_code_triggers_fail(tmp_path: Path):
    out_json = tmp_path / "op_gate_err_exit.json"
    res = run_op_gate.run_op_gate(
        mock_csv=MOCK_VALID_CSV,
        mock_exit_code=1,
        out_json=out_json,
    )
    assert res["status"] == "FAIL"
    assert res["exit_code"] == 1


def test_saved_op_gate_result_exists_and_passes():
    gate_file = Path("benchmarks/results/phase3/op_gate.json")
    assert gate_file.exists(), "op_gate.json must exist from live run"
    with open(gate_file, "r") as f:
        data = json.load(f)
    assert data["status"] == "PASS"
    assert data["error_cases"] == 0
    for op in run_op_gate.CORE_HYBRID_OPS:
        assert op in data["core_ops_status"]
        assert data["core_ops_status"][op]["status"] == "PASS"
