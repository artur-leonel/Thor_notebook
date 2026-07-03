from __future__ import annotations

from pathlib import Path

import polars as pl

INPUTS_DIR = Path(__file__).resolve().parents[2] / "data" / "inputs"
INPUTS_CSV = INPUTS_DIR / "astreia_inputs.csv"


def load_inputs() -> pl.DataFrame:
    if not INPUTS_CSV.exists():
        raise FileNotFoundError(f"Input spreadsheet not found: {INPUTS_CSV}")
    return pl.read_csv(INPUTS_CSV, infer_schema_length=None).with_columns(
        pl.col("item").fill_null("").cast(pl.Utf8)
    )


def _filter(section: str, item: str | None = None, parameter: str | None = None) -> pl.DataFrame:
    df = load_inputs().filter(pl.col("section") == section)
    if item is not None:
        df = df.filter(pl.col("item") == item)
    if parameter is not None:
        df = df.filter(pl.col("parameter") == parameter)
    return df


def _parse_value(raw: str) -> float | str:
    text = str(raw).strip()
    try:
        if "." in text or "e" in text.lower():
            return float(text)
        return float(int(text))
    except ValueError:
        return text


def get(section: str, parameter: str, item: str = "", default: float | str | None = None) -> float | str:
    rows = _filter(section, item, parameter)
    if rows.is_empty():
        if default is not None:
            return default
        raise KeyError(f"Input not found: section={section!r} item={item!r} parameter={parameter!r}")
    return _parse_value(rows["value"][0])


def num(section: str, parameter: str, item: str = "", default: float | None = None) -> float:
    return float(get(section, parameter, item, default))


def txt(section: str, parameter: str, item: str = "", default: str = "") -> str:
    return str(get(section, parameter, item, default))


def section_table(section: str) -> pl.DataFrame:
    return load_inputs().filter(pl.col("section") == section).select("item", "parameter", "value", "unit", "notes")


def item_table(section: str, name_col: str = "item") -> pl.DataFrame:
    rows = []
    df = _filter(section)
    for item in df["item"].unique().to_list():
        if not item or item == "_config":
            continue
        row: dict[str, float | str] = {name_col: item}
        sub = df.filter(pl.col("item") == item)
        for r in sub.iter_rows(named=True):
            row[r["parameter"]] = _parse_value(r["value"])
        rows.append(row)
    return pl.DataFrame(rows)


def txt_list(section: str, parameter: str, item: str = "", sep: str = ";") -> list[str]:
    return [x.strip() for x in txt(section, parameter, item).split(sep) if x.strip()]


def float_list(section: str, parameter: str, item: str = "", sep: str = ";") -> list[float]:
    return [float(x) for x in txt_list(section, parameter, item, sep)]
