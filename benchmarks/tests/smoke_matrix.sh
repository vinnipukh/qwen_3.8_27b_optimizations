#!/usr/bin/env bash
set -euo pipefail

echo "=== Running Smoke Matrix Test ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SMOKE_OUT_DIR="benchmarks/results/smoke_tracer_matrix"
rm -rf "${SMOKE_OUT_DIR}"

python3 benchmarks/bin/run_session.py --smoke --out-dir "${SMOKE_OUT_DIR}"

echo "Validating smoke matrix predicates..."

# Predicate 1: manifest.json exists with required fields
test -s "${SMOKE_OUT_DIR}/manifest.json"
python3 -c "
import json
m = json.load(open('${SMOKE_OUT_DIR}/manifest.json'))
assert m['backend_arm'] == 'HIP'
assert m['binary_sha256'] != ''
assert m['model_sha256'] != ''
print('Manifest validation: PASS')
"

# Predicate 2: rows.jsonl has expected row count (2 rows for smoke: pp and tg at tier 1024, fa off)
test -s "${SMOKE_OUT_DIR}/rows.jsonl"
python3 -c "
import json
from benchmarks.lib.llabench import scan_banned_signatures
rows = [json.loads(line) for line in open('${SMOKE_OUT_DIR}/rows.jsonl') if line.strip()]
assert len(rows) == 2, f'Expected 2 rows, got {len(rows)}'
violations = scan_banned_signatures(rows)
assert len(violations) == 0, f'Found violations: {violations}'
print('Rows validation: PASS')
"

# Predicate 3: Checksums verify
cd "${SMOKE_OUT_DIR}"
sha256sum -c CHECKSUMS.sha256
cd "${REPO_ROOT}"

echo "SMOKE_MATRIX_PASS"
