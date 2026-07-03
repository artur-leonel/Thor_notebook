"""Shared schema helpers for THOR→MRV handoff validation and reporting."""

from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import yaml


REQUIRED_MRV_OML_FIELDS = [
    "out_dir",
    "scad_path",
    "obj_path",
    "stl_path",
    "geometry_json_path",
    "metrics_json_path",
    "three_view_path",
    "shaded_render_path",
    "engineering_views_path",
]

PATH_FIELDS = REQUIRED_MRV_OML_FIELDS

REQUIRED_METRIC_FIELDS = [
    "provenance",
    "geometry",
    "heating",
    "constraints",
    "assumptions",
    "openscad",
    "previews",
]


@dataclass(frozen=True)
class MassSanityConfig:
    enabled: bool = True
    warn_bulk_density_kg_m3: float = 1200.0
    fail_bulk_density_kg_m3: float = 3000.0
    warn_payload_fraction_below: float = 0.10
    fail_payload_fraction_below: float = 0.03
    fail_on_mass_sanity: bool = False


DEFAULT_MASS_SANITY_CONFIG = MassSanityConfig()
DEFAULT_MASS_SANITY_WARNING_MESSAGE = (
    "Dry mass appears high relative to vehicle volume and payload target. "
    "Treat current mass model as placeholder until mass breakdown is reviewed."
)


MASS_SANITY_LEVEL_OK = "ok"
MASS_SANITY_LEVEL_WARNING = "warning"
MASS_SANITY_LEVEL_FAIL = "fail"


def load_mass_sanity_config(config_path: str | Path) -> MassSanityConfig:
    """Load optional mass sanity thresholds from a MRV YAML config."""
    path = Path(config_path)
    if not path.exists():
        return DEFAULT_MASS_SANITY_CONFIG

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = data.get("mass_sanity")
    if not isinstance(raw, dict):
        return DEFAULT_MASS_SANITY_CONFIG

    def _as_bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(default if value is None else value)

    def _as_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    return MassSanityConfig(
        enabled=_as_bool(raw.get("enabled"), DEFAULT_MASS_SANITY_CONFIG.enabled),
        warn_bulk_density_kg_m3=_as_float(
            raw.get("warn_bulk_density_kg_m3"), DEFAULT_MASS_SANITY_CONFIG.warn_bulk_density_kg_m3
        ),
        fail_bulk_density_kg_m3=_as_float(
            raw.get("fail_bulk_density_kg_m3"), DEFAULT_MASS_SANITY_CONFIG.fail_bulk_density_kg_m3
        ),
        warn_payload_fraction_below=_as_float(
            raw.get("warn_payload_fraction_below"),
            DEFAULT_MASS_SANITY_CONFIG.warn_payload_fraction_below,
        ),
        fail_payload_fraction_below=_as_float(
            raw.get("fail_payload_fraction_below"),
            DEFAULT_MASS_SANITY_CONFIG.fail_payload_fraction_below,
        ),
        fail_on_mass_sanity=_as_bool(
            raw.get("fail_on_mass_sanity"), DEFAULT_MASS_SANITY_CONFIG.fail_on_mass_sanity
        ),
    )


def as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def as_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _row_field(row: Mapping[str, Any] | None, key: str) -> str:
    if not row:
        return ""
    value = row.get(key, "")
    return as_string(value).strip()


def _row_field_is_str(row: Mapping[str, Any] | None, key: str) -> bool:
    if not row:
        return False
    value = row.get(key)
    return isinstance(value, str) and bool(as_text(value))


def _artifact_exists(row: Mapping[str, Any] | None, field: str, value: str) -> bool:
    if not value:
        return False
    path = Path(value)
    if field == "out_dir":
        return path.exists() and path.is_dir()
    return path.exists()


def validate_mrv_oml_row(
    row: Mapping[str, Any] | None,
    require_files: bool = True,
) -> tuple[bool, list[str], list[str]]:
    missing_fields: list[str] = []
    missing_files: list[str] = []

    if row is None:
        return False, REQUIRED_MRV_OML_FIELDS.copy(), []

    for field in REQUIRED_MRV_OML_FIELDS:
        if not _row_field_is_str(row, field):
            missing_fields.append(field)
            continue
        value = _row_field(row, field)
        if require_files and not _artifact_exists(row, field, value):
            missing_files.append(value)

    return len(missing_fields) == 0 and len(missing_files) == 0, missing_fields, missing_files


def _markdown_link(path: str) -> str:
    path = as_string(path)
    if not path:
        return "<n/a>"
    return f"[{path}]({path})"


def artifact_markdown_table(row: Mapping[str, Any] | None, include_optional: bool = True) -> str:
    """Create a markdown handoff table suitable for notebook rendering."""
    required_rows = [
        ("Output directory", "out_dir"),
        ("OpenSCAD model", "scad_path"),
        ("OBJ mesh", "obj_path"),
        ("STL mesh", "stl_path"),
        ("Geometry JSON", "geometry_json_path"),
        ("Metrics JSON", "metrics_json_path"),
        ("Three-view preview", "three_view_path"),
        ("Shaded render", "shaded_render_path"),
        ("Engineering views", "engineering_views_path"),
    ]

    rows = [["Artifact", "Exists", "Path"]]
    for artifact, field in required_rows:
        value = _row_field(row, field)
        exists = _artifact_exists(row, field, value)
        rows.append([artifact, "✅" if exists else "❌", _markdown_link(value)])

    if include_optional:
        for field, value in (row or {}).items():
            if field in REQUIRED_MRV_OML_FIELDS or not field.endswith("_path"):
                continue
            exists = Path(as_string(value)).exists() if as_string(value) else False
            rows.append([field, "✅" if exists else "❌", _markdown_link(value)])

    lines = ["## MRV OML Artifact Handoff", "", "| Artifact | Exists | Path |", "| --- | --- | --- |"]
    lines += [f"| {r[0]} | {r[1]} | {r[2]} |" for r in rows[1:]]
    return "\n".join(lines)


def load_metrics(metrics_json_path: str | Path) -> dict[str, Any]:
    path = Path(metrics_json_path)
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_json_path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Metrics payload is not a mapping: {metrics_json_path}")
    return data


def evaluate_mass_sanity(
    dry_mass_kg: float,
    payload_mass_kg: float,
    volume_m3: float,
    config: MassSanityConfig = DEFAULT_MASS_SANITY_CONFIG,
) -> dict[str, Any]:
    if not config.enabled:
        return {
            "status": MASS_SANITY_LEVEL_OK,
            "enabled": False,
            "bulk_density_kg_m3": 0.0,
            "payload_fraction": 0.0,
            "warnings": [],
            "failures": [],
            "message": "",
            "thresholds": vars(config),
        }

    bulk_density = dry_mass_kg / max(volume_m3, 1e-9)
    payload_fraction = payload_mass_kg / max(dry_mass_kg + payload_mass_kg, 1e-9)
    warnings: list[str] = []
    failures: list[str] = []

    if bulk_density >= config.fail_bulk_density_kg_m3:
        failures.append(
            f"dry mass bulk density {bulk_density:.1f} kg/m^3 exceeds fail threshold {config.fail_bulk_density_kg_m3:.0f}"
        )
    elif bulk_density >= config.warn_bulk_density_kg_m3:
        warnings.append(
            f"dry mass bulk density {bulk_density:.1f} kg/m^3 exceeds warning threshold {config.warn_bulk_density_kg_m3:.0f}"
        )

    if payload_fraction <= config.fail_payload_fraction_below:
        failures.append(
            f"payload fraction {payload_fraction:.3f} is below fail threshold {config.fail_payload_fraction_below:.2f}"
        )
    elif payload_fraction <= config.warn_payload_fraction_below:
        warnings.append(
            f"payload fraction {payload_fraction:.3f} is below warning threshold {config.warn_payload_fraction_below:.2f}"
        )

    status = MASS_SANITY_LEVEL_OK
    if failures:
        status = MASS_SANITY_LEVEL_FAIL
    elif warnings:
        status = MASS_SANITY_LEVEL_WARNING

    summary_message = DEFAULT_MASS_SANITY_WARNING_MESSAGE if (warnings or failures) else ""

    return {
        "config": asdict(config),
        "status": status,
        "enabled": True,
        "bulk_density_kg_m3": float(bulk_density),
        "payload_fraction": float(payload_fraction),
        "warnings": warnings,
        "failures": failures,
        "message": summary_message,
        "summary_message": summary_message if status != MASS_SANITY_LEVEL_OK else "",
        "fail_on_mass_sanity": bool(config.fail_on_mass_sanity),
        "thresholds": vars(config),
    }
