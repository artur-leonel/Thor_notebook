#!/usr/bin/env python3
"""Audit that the THOR notebook chain is complete and input-driven."""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from thor.io.handoff import PARQUET_DIR, load_table
from thor.io.inputs import load_inputs
from thor.pipeline import THOR_NOTEBOOKS, expected_tables, manifest_by_path


FORBIDDEN_NOTEBOOK_PATTERNS = {
    r"\bor\s+3500\b": "Use estimated_dry_mass_kg()/estimated_wet_mass_kg() instead of a fixed mass fallback.",
    r"\bor\s+4000\b": "Use estimated_wet_mass_kg() instead of a fixed wet-mass fallback.",
    r"\bor\s+5000\b": "Use total_phase_energy_wh() instead of a fixed energy fallback.",
    r"\bor\s+600\b": "Use mass.propellant_kg from inputs instead of a fixed propellant fallback.",
    r"\bor\s+7800(?:\.0)?\b": "Use entry_velocity_m_s() instead of a fixed entry velocity fallback.",
}


def _saved_tables(text: str) -> set[str]:
    """Return literal table names passed to save_table(...) in a notebook."""
    tree = ast.parse(text)
    tables: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue

        func = node.func
        is_save_table = (
            isinstance(func, ast.Name) and func.id == "save_table"
        ) or (
            isinstance(func, ast.Attribute) and func.attr == "save_table"
        )
        if not is_save_table:
            continue

        first_arg = node.args[0]
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            tables.add(first_arg.value)
    return tables


def audit_static() -> list[str]:
    errors: list[str] = []
    manifest = manifest_by_path()
    declared_paths = set(manifest)
    actual_paths = {str(path) for path in sorted(Path("notebooks").glob("*.py"))}

    missing = sorted(declared_paths - actual_paths)
    extra = sorted(actual_paths - declared_paths)
    errors.extend(f"manifest notebook missing on disk: {path}" for path in missing)
    errors.extend(f"notebook not listed in THOR manifest: {path}" for path in extra)

    try:
        load_inputs()
    except Exception as exc:  # noqa: BLE001 - report as audit error
        errors.append(f"could not load thor_inputs.csv: {exc}")

    for spec in THOR_NOTEBOOKS:
        path = Path(spec.path)
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        try:
            saved_tables = _saved_tables(text)
        except SyntaxError as exc:
            errors.append(f"{path}: syntax error: {exc}")
            saved_tables = set()

        for pattern, message in FORBIDDEN_NOTEBOOK_PATTERNS.items():
            if re.search(pattern, text):
                errors.append(f"{path}: {message}")

        for table in spec.tables:
            if table not in saved_tables:
                errors.append(f"{path}: expected save_table({table!r}) from manifest")

    return errors


def audit_tables() -> list[str]:
    errors: list[str] = []
    for table in expected_tables():
        path = PARQUET_DIR / f"{table}.parquet"
        if not path.exists():
            errors.append(f"missing notebook output table: {path}")
            continue
        try:
            df = load_table(table)
        except Exception as exc:  # noqa: BLE001 - report as audit error
            errors.append(f"could not read table {table}: {exc}")
            continue
        if df is None:
            errors.append(f"missing notebook output table: {path}")
        elif df.height == 0:
            errors.append(f"empty notebook output table: {path}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-tables", action="store_true", help="also require every manifest table to exist and be nonempty")
    args = parser.parse_args()

    errors = audit_static()
    if args.require_tables:
        errors.extend(audit_tables())

    if errors:
        print("THOR pipeline audit FAILED")
        for error in errors:
            print(f" - {error}")
        return 1

    print("THOR pipeline audit OK")
    print(f"notebooks: {len(THOR_NOTEBOOKS)}")
    print(f"expected tables: {len(expected_tables())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
