"""Unit tests for matrix aggregation, assembly, and Markdown publishing."""
import json
import pytest
from benchmarks.bin import publish_matrix
from benchmarks.lib import guard, llabench
from benchmarks.lib.store import RunStore
from benchmarks.tests.fixtures.gen_llabench_jsonl import make_jsonl_row


def test_calculate_stats():
    # Identical samples
    mean, stddev = publish_matrix.calculate_stats([100.0, 100.0, 100.0])
    assert mean == 100.0
    assert stddev == 0.0

    # Known variance sample
    samples = [10.0, 12.0, 23.0, 23.0, 16.0, 23.0, 21.0, 16.0]
    mean, stddev = publish_matrix.calculate_stats(samples)
    assert pytest.approx(mean, 0.01) == 18.0
    assert pytest.approx(stddev, 0.01) == 5.24


def test_matrix_assembly_and_ordering(tmp_path):
    run_dir = tmp_path / "run_hip"
    store = RunStore(run_dir)

    # 4k pp and tg
    store.append_row({
        "backend_arm": "HIP",
        "n_prompt": 4096,
        "n_gen": 0,
        "flash_attn": 0,
        "avg_ts": 111.5,
        "samples_ts": [111.0, 112.0],
        "guard": {"verdict": "OK"},
    })
    store.append_row({
        "backend_arm": "HIP",
        "n_prompt": 4096,
        "n_gen": 128,
        "flash_attn": 0,
        "avg_ts": 33.5,
        "samples_ts": [33.0, 34.0],
        "guard": {"verdict": "OK"},
    })
    # 32k failed preflight
    store.append_row({
        "backend_arm": "HIP",
        "n_prompt": 32768,
        "n_gen": 0,
        "flash_attn": 0,
        "avg_ts": 0.0,
        "samples_ts": [],
        "guard": {"verdict": guard.VERDICT_PREFLIGHT, "evidence": "Allocation limit exceeded"},
    })
    # Review row
    store.append_row({
        "backend_arm": "HIP",
        "n_prompt": 8192,
        "n_gen": 0,
        "flash_attn": 0,
        "avg_ts": 90.0,
        "samples_ts": [50.0, 130.0],
        "guard": {"verdict": guard.VERDICT_REVIEW},
    })

    mat = publish_matrix.assemble_matrix([run_dir])

    # Check ok cells
    assert len(mat["ok_cells"]) == 2
    assert "HIP_c4096_pp_fa_off" in mat["ok_cells"]
    assert "HIP_c4096_tg_fa_off" in mat["ok_cells"]

    # Check failed cells
    assert len(mat["failed_cells"]) == 1
    assert mat["failed_cells"][0]["verdict"] == guard.VERDICT_PREFLIGHT

    # Check review cells
    assert len(mat["review_cells"]) == 1
    assert mat["review_cells"][0]["verdict"] == guard.VERDICT_REVIEW

    # Check generated markdown
    md = publish_matrix.generate_markdown(mat)
    assert "## Stock HIP Baseline Matrix (ROCm 7.2.1)" in md
    assert "### Failed / Pre-flight Gated Cells" in md
    assert "FAILED:preflight-oom" in md
    assert "### Cells Flagged for Review" in md


def test_matrix_assembly_contamination_rejection(tmp_path):
    run_dir = tmp_path / "run_contaminated"
    store = RunStore(run_dir)

    # Injected contaminated default 512 row
    store.append_row({
        "backend_arm": "HIP",
        "n_prompt": 512,
        "n_gen": 128,
        "flash_attn": 0,
        "avg_ts": 30.0,
        "samples_ts": [30.0],
        "guard": {"verdict": "OK"},
    })

    with pytest.raises(llabench.MatrixContaminationError, match="Cannot publish contaminated rows"):
        publish_matrix.assemble_matrix([run_dir])
