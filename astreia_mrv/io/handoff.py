from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl

from astreia_mrv.models.notebook_state import AstreiaNotebookState

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
STATE_FILE = DATA_DIR / "vehicle_state.json"
PARQUET_DIR = DATA_DIR / "parquet"
DUCKDB_PATH = DATA_DIR / "astreia.duckdb"


def state_path() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return STATE_FILE


def save_state(state: AstreiaNotebookState) -> Path:
    path = state_path()
    path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_state() -> AstreiaNotebookState:
    path = state_path()
    if not path.exists():
        return AstreiaNotebookState()
    return AstreiaNotebookState.model_validate_json(path.read_text(encoding="utf-8"))


def save_table(name: str, df: pl.DataFrame) -> Path:
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    out = PARQUET_DIR / f"{name}.parquet"
    df.write_parquet(out)
    register_duckdb(name, df)
    return out


def load_table(name: str) -> pl.DataFrame | None:
    path = PARQUET_DIR / f"{name}.parquet"
    if not path.exists():
        return None
    return pl.read_parquet(path)


def register_duckdb(name: str, df: pl.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(DUCKDB_PATH)) as con:
        con.register("_tmp", df.to_arrow())
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _tmp")
