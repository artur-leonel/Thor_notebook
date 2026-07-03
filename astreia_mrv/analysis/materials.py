from __future__ import annotations

import yaml


def load_materials(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def recommended_zone_material(zone: str, materials: dict) -> str:
    zones = materials.get("zones", {})
    return str(zones.get(zone, {}).get("preferred", "unassigned"))
