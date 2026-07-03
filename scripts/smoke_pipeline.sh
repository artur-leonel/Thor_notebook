#!/usr/bin/env bash
# Validate inputs + full THOR notebook pipeline (no Marimo server)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python3
fi

"$PYTHON" "${ROOT}/scripts/sync_mrv_thor_inputs.py"
"$PYTHON" "${ROOT}/scripts/audit_thor_pipeline.py"
"$PYTHON" "${ROOT}/scripts/run_thor_notebooks.py" --reset
"$PYTHON" "${ROOT}/scripts/validate_pipeline.py"
"$PYTHON" "${ROOT}/scripts/audit_thor_pipeline.py" --require-tables
"$PYTHON" "${ROOT}/scripts/smoke_check_mrv_oml.py"
"$PYTHON" "${ROOT}/scripts/audit_mrv_output_provenance.py"
