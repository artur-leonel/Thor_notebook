from pathlib import Path

from scripts.audit_thor_pipeline import audit_static
from thor.pipeline import THOR_NOTEBOOKS, expected_tables


def test_thor_manifest_lists_every_notebook_once():
    manifest_paths = [spec.path for spec in THOR_NOTEBOOKS]
    disk_paths = sorted(str(path) for path in Path("notebooks").glob("*.py"))

    assert len(manifest_paths) == len(set(manifest_paths))
    assert manifest_paths == disk_paths


def test_thor_manifest_expected_tables_are_unique():
    tables = expected_tables()
    assert len(tables) == len(set(tables))
    assert "mrv_oml" in tables
    assert "control_authority" in tables


def test_thor_pipeline_static_audit_passes():
    assert audit_static() == []
