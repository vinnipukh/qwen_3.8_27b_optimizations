"""Tests for crash resilience and append-only result store verification."""
import multiprocessing
import os
import signal
import subprocess
import time
from pathlib import Path
import pytest
from benchmarks.lib.store import RunStore


def _child_append_and_die(run_dir_str: str, row_count: int, kill_after: int):
    store = RunStore(Path(run_dir_str))
    for i in range(row_count):
        store.append_row({"row_id": i, "data": f"content_{i}"})
        if i + 1 == kill_after:
            # Simulate sudden uncatchable kill (SIGKILL)
            os.kill(os.getpid(), signal.SIGKILL)


def test_journal_crash_resilience(tmp_path):
    run_dir = tmp_path / "crash_run"
    store = RunStore(run_dir)

    total_rows = 10
    kill_after = 5

    proc = multiprocessing.Process(
        target=_child_append_and_die,
        args=(str(run_dir), total_rows, kill_after),
    )
    proc.start()
    proc.join()

    # Process should have died by SIGKILL (exitcode == -signal.SIGKILL)
    assert proc.exitcode == -signal.SIGKILL or proc.exitcode != 0

    # Verify rows written up to kill_after are intact
    assert store.rows_file.exists()
    lines = store.rows_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == kill_after

    for idx, line in enumerate(lines):
        import json
        data = json.loads(line)
        assert data["row_id"] == idx
        assert data["data"] == f"content_{idx}"


def test_checksums_and_tamper_detection(tmp_path):
    run_dir = tmp_path / "checksum_run"
    store = RunStore(run_dir)

    store.append_row({"sample": 1, "score": 100.0})
    store.append_row({"sample": 2, "score": 105.0})
    (store.logs_dir / "sample.log").write_text("benchmark log content", encoding="utf-8")

    chk_file = store.write_checksums()
    assert chk_file.exists()

    # Run sha256sum -c in run_dir
    res = subprocess.run(
        ["sha256sum", "-c", "CHECKSUMS.sha256"],
        cwd=run_dir,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"sha256sum failed: {res.stderr}"

    # Tamper with rows.jsonl
    with open(store.rows_file, "a", encoding="utf-8") as f:
        f.write('{"tampered": true}\n')

    # Re-verify sha256sum -c must fail
    res_tamper = subprocess.run(
        ["sha256sum", "-c", "CHECKSUMS.sha256"],
        cwd=run_dir,
        capture_output=True,
        text=True,
    )
    assert res_tamper.returncode != 0


def test_supersede_metadata(tmp_path):
    run_dir = tmp_path / "run_v2"
    store = RunStore(run_dir)
    store.supersede("run_v1", "Hardware thermal re-calibration")

    assert store.meta_file.exists()
    import json
    meta = json.loads(store.meta_file.read_text(encoding="utf-8"))
    assert "supersedes" in meta
    assert meta["supersedes"]["run_id"] == "run_v1"
    assert meta["supersedes"]["reason"] == "Hardware thermal re-calibration"
