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
CAD_PYTHON="${ROOT}/.cad-venv/bin/python"
if [[ ! -x "$CAD_PYTHON" ]]; then
  CAD_PYTHON="$PYTHON"
fi
if PYTHONPATH="${ROOT}:${ROOT}/.cad-venv/lib/python3.10/site-packages" "$CAD_PYTHON" -c "import OCP" >/dev/null 2>&1; then
  PYTHONPATH="${ROOT}:${ROOT}/.cad-venv/lib/python3.10/site-packages" "$CAD_PYTHON" "${ROOT}/scripts/export_onshape_step.py" \
    --spec "${ROOT}/outputs/mrv3_notebook/onshape_brep_spec.json" \
    --output-dir "${ROOT}/outputs/mrv3_notebook/onshape_step"
else
  echo "OCP unavailable; skipping optional Onshape STEP/BREP solid export"
fi
"$PYTHON" "${ROOT}/scripts/audit_mrv_output_provenance.py"
