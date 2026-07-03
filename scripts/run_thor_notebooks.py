#!/usr/bin/env python3
"""Execute all THOR notebooks in manifest order via Marimo session export."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from thor.io.handoff import PARQUET_DIR, STATE_FILE
from thor.pipeline import THOR_NOTEBOOKS


def reset_handoff_state() -> None:
    if STATE_FILE.exists():
        STATE_FILE.unlink()
    if PARQUET_DIR.exists():
        shutil.rmtree(PARQUET_DIR)


def run_notebook(path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "marimo",
            "export",
            "session",
            "--force-overwrite",
            "--no-continue-on-error",
            str(path),
        ],
        cwd=ROOT,
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true", help="remove generated vehicle_state/parquet tables before running")
    args = parser.parse_args()

    if args.reset:
        reset_handoff_state()

    for index, spec in enumerate(THOR_NOTEBOOKS, start=1):
        print(f"[{index:02d}/{len(THOR_NOTEBOOKS)}] {spec.path}")
        run_notebook(ROOT / spec.path)

    print(f"executed {len(THOR_NOTEBOOKS)} THOR notebooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
