from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

from astreia_mrv.analysis.aerodynamics import alpha_sweep, write_polar_csv
from astreia_mrv.analysis.constraints import evaluate_constraints
from astreia_mrv.analysis.heating import heating_summary, write_heating_csv
from astreia_mrv.analysis.mass import conceptual_mass_breakdown
from astreia_mrv.analysis.objectives import objective_summary
from astreia_mrv.config import VehicleDesign, load_design
from astreia_mrv.geometry.cad_export import export_geometry
from astreia_mrv.geometry.capsule import generate_capsule
from astreia_mrv.geometry.lifting_body import generate_lifting_body
from astreia_mrv.optimization.sampling import sample_lifting_body_rx
from astreia_mrv.viz.plot_geometry import save_three_view
from astreia_mrv.viz.plot_performance import save_sweep_scatter


def generate_geometry(design: VehicleDesign):
    if design.geometry_family == "capsule":
        return generate_capsule(design)
    return generate_lifting_body(design)


def _write_metrics(design: VehicleDesign, geometry, out: Path) -> dict:
    heat = heating_summary(design, geometry)
    constraints = evaluate_constraints(design, geometry)
    polar = alpha_sweep(geometry, mach=20.0)
    metrics = {
        "geometry": {
            "reference_area_m2": geometry.reference_area_m2,
            "reference_length_m": geometry.reference_length_m,
            "volume_m3": geometry.volume_m3,
            "wetted_area_m2": geometry.wetted_area_m2,
            "payload_box": geometry.payload_box,
        },
        "mass": conceptual_mass_breakdown(design, geometry),
        "heating": heat,
        "constraints": [c.__dict__ for c in constraints],
        "objectives": objective_summary(design, geometry),
        "aero_alpha_sweep": polar,
        "assumptions": [
            "Conceptual modified-Newtonian aerodynamics on sampled mesh panels.",
            "Conceptual cold-wall heating; no material response or ablation regression.",
            "Terminal landing accuracy is represented by authorized recovery-zone recovery only.",
            "TODO: CFD/FEA/6DOF integration.",
        ],
    }
    with (out / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    write_polar_csv(polar, out / "aero_polar.csv")
    write_heating_csv(heat, out / "heating_summary.csv")
    return metrics


def cmd_generate(args: argparse.Namespace) -> None:
    design = load_design(args.config)
    geometry = generate_geometry(design)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    export_geometry(geometry, out)
    save_three_view(geometry, out / "three_view.png")
    metrics = _write_metrics(design, geometry, out)
    print(json.dumps({"out": str(out), "volume_m3": geometry.volume_m3, "constraints_passed": all(c["passed"] for c in metrics["constraints"])}, indent=2))


def cmd_analyze(args: argparse.Namespace) -> None:
    design = load_design(args.config)
    geometry = generate_geometry(design)
    out = Path(args.geometry).parent if args.geometry else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics = _write_metrics(design, geometry, out)
    print(json.dumps({"out": str(out), "peak_heat_flux_w_m2": metrics["objectives"]["minimize_peak_heat_flux_w_m2"]}, indent=2))


def cmd_sweep(args: argparse.Namespace) -> None:
    base = load_design(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, rx in enumerate(sample_lifting_body_rx(args.n, seed=args.seed)):
        design = replace(base, rx=rx, geometry_family="lifting_body")
        geometry = generate_lifting_body(design)
        constraints = evaluate_constraints(design, geometry)
        objectives = objective_summary(design, geometry)
        rows.append(
            {
                "i": i,
                "feasible": float(all(c.passed for c in constraints)),
                "peak_heat_flux_w_m2": objectives["minimize_peak_heat_flux_w_m2"],
                "payload_volume_efficiency": objectives["maximize_payload_volume_efficiency"],
                "crossrange_proxy": objectives["maximize_crossrange_proxy"],
                "volume_m3": geometry.volume_m3,
            }
        )
    with (out / "sweep.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    save_sweep_scatter(rows, out / "pareto_style_scatter.png")
    print(json.dumps({"out": str(out), "n": args.n, "feasible": int(sum(r["feasible"] for r in rows))}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="astreia-mrv")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate")
    gen.add_argument("--config", default="configs/mrv3.yaml")
    gen.add_argument("--out", default="outputs/mrv3_baseline")
    gen.set_defaults(func=cmd_generate)

    analyze = sub.add_parser("analyze")
    analyze.add_argument("--config", default="configs/mrv3.yaml")
    analyze.add_argument("--geometry", default=None)
    analyze.add_argument("--out", default="outputs/analysis")
    analyze.set_defaults(func=cmd_analyze)

    sweep = sub.add_parser("sweep")
    sweep.add_argument("--config", default="configs/mrv3.yaml")
    sweep.add_argument("--n", type=int, default=100)
    sweep.add_argument("--seed", type=int, default=7)
    sweep.add_argument("--out", default="outputs/sweep_mrv3")
    sweep.set_defaults(func=cmd_sweep)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
