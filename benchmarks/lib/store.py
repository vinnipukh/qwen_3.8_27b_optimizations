"""Crash-resilient append-only result store (D2-09, D2-11, BENCH-03).

Provides atomic fsynced row journaling, verifiable CHECKSUMS.sha256 creation,
supersede metadata recording, and index tracking.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any


class RunStore:
    """Manages an append-only benchmark run directory."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.logs_dir = self.run_dir / "logs"
        self.telemetry_dir = self.run_dir / "telemetry"
        self.rows_file = self.run_dir / "rows.jsonl"
        self.ledger_file = self.run_dir / "vram_ledger.jsonl"
        self.meta_file = self.run_dir / "meta.json"
        self.checksums_file = self.run_dir / "CHECKSUMS.sha256"

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(
        cls,
        results_root: str | Path = "benchmarks/results",
        run_id: str | None = None,
        label: str = "run",
    ) -> RunStore:
        """Create a new run directory in results_root."""
        root = Path(results_root)
        if run_id is None:
            ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
            run_id = f"{ts}_{label}"
        run_p = root / run_id
        return cls(run_p)

    def append_row(self, row: dict[str, Any]) -> None:
        """Serialize compact JSON, append line to rows.jsonl, and fsync immediately."""
        line = json.dumps(row, separators=(",", ":")) + "\n"
        with open(self.rows_file, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    def append_ledger_row(self, row: dict[str, Any]) -> None:
        """Append VRAM ledger row with fsync."""
        line = json.dumps(row, separators=(",", ":")) + "\n"
        with open(self.ledger_file, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    def supersede(self, old_run_id: str, reason: str) -> None:
        """Record supersede relation in meta.json."""
        meta: dict[str, Any] = {}
        if self.meta_file.exists():
            try:
                meta = json.loads(self.meta_file.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        meta["supersedes"] = {
            "run_id": old_run_id,
            "reason": reason,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

    def write_checksums(self) -> Path:
        """Walk run directory and emit CHECKSUMS.sha256 in sha256sum -c compatible format."""
        entries: list[tuple[str, str]] = []

        for p in sorted(self.run_dir.rglob("*")):
            if p.is_file() and p.name != "CHECKSUMS.sha256":
                # Compute sha256
                h = hashlib.sha256()
                with open(p, "rb") as f:
                    while chunk := f.read(1024 * 1024):
                        h.update(chunk)
                digest = h.hexdigest()
                rel_path = p.relative_to(self.run_dir).as_posix()
                entries.append((digest, rel_path))

        with open(self.checksums_file, "w", encoding="utf-8") as f:
            for digest, rel_path in sorted(entries, key=lambda x: x[1]):
                f.write(f"{digest}  {rel_path}\n")
            f.flush()
            os.fsync(f.fileno())

        return self.checksums_file

    def index_entry(
        self,
        ok_count: int,
        failed_count: int,
        backend_arm: str = "HIP",
    ) -> dict[str, Any]:
        """Generate summary dictionary for results/index.jsonl."""
        return {
            "run_id": self.run_dir.name,
            "close_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "backend_arm": backend_arm,
            "ok_count": ok_count,
            "failed_count": failed_count,
        }
