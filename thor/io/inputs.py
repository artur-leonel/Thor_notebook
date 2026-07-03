"""Central input spreadsheet — single source of truth for THOR sizing."""

from __future__ import annotations

from pathlib import Path

import polars as pl

INPUTS_DIR = Path(__file__).resolve().parents[2] / "data" / "inputs"
INPUTS_CSV = INPUTS_DIR / "thor_inputs.csv"


def load_inputs() -> pl.DataFrame:
    """Load `data/inputs/thor_inputs.csv`."""
    if not INPUTS_CSV.exists():
        raise FileNotFoundError(f"Input spreadsheet not found: {INPUTS_CSV}")
    df = pl.read_csv(INPUTS_CSV, infer_schema_length=None)
    return df.with_columns(pl.col("item").fill_null("").cast(pl.Utf8))


def _filter(section: str, item: str | None = None, parameter: str | None = None) -> pl.DataFrame:
    df = load_inputs().filter(pl.col("section") == section)
    if item is not None:
        df = df.filter(pl.col("item") == item)
    if parameter is not None:
        df = df.filter(pl.col("parameter") == parameter)
    return df


def _parse_value(raw: str) -> float | str:
    s = str(raw).strip()
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return float(int(s))
    except ValueError:
        return s


def get(section: str, parameter: str, item: str = "", default: float | str | None = None) -> float | str:
    rows = _filter(section, item, parameter)
    if rows.is_empty():
        if default is not None:
            return default
        raise KeyError(f"Input not found: section={section!r} item={item!r} parameter={parameter!r}")
    return _parse_value(rows["value"][0])


def num(section: str, parameter: str, item: str = "", default: float | None = None) -> float:
    val = get(section, parameter, item, default)
    return float(val)


def txt(section: str, parameter: str, item: str = "", default: str = "") -> str:
    val = get(section, parameter, item, default)
    return str(val)


def section_table(section: str) -> pl.DataFrame:
    """All rows for a section (spreadsheet view)."""
    return load_inputs().filter(pl.col("section") == section).select(
        "item", "parameter", "value", "unit", "notes"
    )


def items_with_params(section: str, *parameters: str) -> list[dict[str, float | str]]:
    """Group rows by item; each dict has item keys + requested parameters."""
    df = _filter(section)
    if parameters:
        df = df.filter(pl.col("parameter").is_in(list(parameters)))
    out: list[dict[str, float | str]] = []
    for item in df["item"].unique().to_list():
        if item in ("", "_config"):
            continue
        row: dict[str, float | str] = {"item": item}
        sub = df.filter(pl.col("item") == item)
        for r in sub.iter_rows(named=True):
            row[r["parameter"]] = _parse_value(r["value"])
        out.append(row)
    return out


def phase_rows() -> list[dict[str, float | str]]:
    """Mission phases from section `phase`."""
    return items_with_params("phase", "duration_s", "delta_v_m_s", "power_wh")


def txt_list(section: str, parameter: str, item: str = "", sep: str = ";") -> list[str]:
    raw = txt(section, parameter, item)
    return [x.strip() for x in raw.split(sep) if x.strip()]


def float_list(section: str, parameter: str, item: str = "", sep: str = ";") -> list[float]:
    return [float(x) for x in txt_list(section, parameter, item, sep)]


def item_table(section: str, name_col: str = "item") -> pl.DataFrame:
    """Pivot item/parameter rows into one row per item (mixed types)."""
    rows = []
    df = _filter(section)
    for item in df["item"].unique().to_list():
        if not item or item == "_config":
            continue
        row: dict = {name_col: item}
        sub = df.filter(pl.col("item") == item)
        for r in sub.iter_rows(named=True):
            row[r["parameter"]] = _parse_value(r["value"])
        rows.append(row)
    return pl.DataFrame(rows)


def estimated_dry_mass_kg() -> float:
    """Dry mass estimate from the mass rows in `thor_inputs.csv`.

    This mirrors notebook 01 when downstream notebooks are run on their own
    before `vehicle_state.json` has been populated.
    """
    growth = num("mass", "growth_allowance", "_config")
    return sum(float(row["dry_kg"]) * (1.0 + growth) for row in items_with_params("mass", "dry_kg"))


def estimated_wet_mass_kg() -> float:
    return estimated_dry_mass_kg() + num("mass", "propellant_kg", "_config")


def total_phase_energy_wh() -> float:
    return sum(float(row["power_wh"]) for row in phase_rows())


def entry_velocity_m_s() -> float:
    """Configured entry-interface velocity, deriving the auto case from orbit inputs."""
    override = num("entry", "velocity_m_s")
    if override > 0.0:
        return override

    from thor.physics.astro import circular_velocity

    v_circ = circular_velocity(num("orbit", "altitude_km") * 1000.0)
    dv_deorbit = num("entry", "deorbit_dv_m_s")
    return max(v_circ - dv_deorbit, 0.0)
