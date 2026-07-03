from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save_sweep_scatter(rows: list[dict[str, float]], path: str | Path) -> None:
    if not rows:
        return
    heat = [r["peak_heat_flux_w_m2"] for r in rows]
    vol = [r["payload_volume_efficiency"] for r in rows]
    feasible = [r["feasible"] for r in rows]
    colors = ["tab:green" if f else "tab:red" for f in feasible]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(heat, vol, c=colors, s=18, alpha=0.75)
    ax.set_xlabel("Peak heat flux W/m^2")
    ax.set_ylabel("Payload volume efficiency")
    ax.grid(True, linewidth=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
