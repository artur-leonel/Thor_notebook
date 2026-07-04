from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from astreia_mrv.config import VehicleDesign


def interp_from_rx(rx_value: float, low: float, high: float) -> float:
    """Book-style normalized mapping x = x_min + rx * (x_max - x_min)."""
    if not 0.0 <= rx_value <= 1.0:
        raise ValueError(f"rx value {rx_value} is outside [0, 1]")
    return low + rx_value * (high - low)


def lifting_body_bounds(body_length_m: float) -> dict[str, tuple[float, float]]:
    l = body_length_m
    return {
        "R_N": (0.05 * l, 0.09 * l),
        "theta_N_deg": (25.0, 45.0),
        "dx1": (0.26 * l, 0.42 * l),
        "dx2": (0.25 * l, 0.40 * l),
        "body_half_width": (0.110 * l, 0.155 * l),
        "body_top_height": (0.085 * l, 0.130 * l),
        "body_belly_depth": (0.080 * l, 0.130 * l),
        "R_LE": (max(0.025, 0.012 * l), max(0.12, 0.04 * l)),
        "theta_LE_deg": (5.0, 30.0),
        "L_w": (0.15 * l, 0.32 * l),
        "L_mid_frac": (0.25, 0.6875),
        "t_mid": (max(0.03, 0.012 * l), max(0.18, 0.08 * l)),
        "semi_span": (0.012 * l, 0.035 * l),
        "winglet_bent_span_frac": (0.20, 0.50),
        "theta_f_deg": (35.0, 85.0),
        "f_el": (0.50, 0.95),
        "L_bf": (0.05 * l, 0.25 * l),
        "sweep_deg": (55.0, 75.0),
        "x_cg_frac": (0.46, 0.58),
    }


def capsule_bounds(body_length_m: float) -> dict[str, tuple[float, float]]:
    l = body_length_m
    return {
        "R_N": (0.35 * l, 0.95 * l),
        "R_S": (0.02 * l, 0.12 * l),
        "theta_c_deg": (10.0, 35.0),
        "R_m": (0.25 * l, 0.48 * l),
        "L_c": (0.10 * l, 0.42 * l),
        "x_cg_frac": (0.42, 0.58),
    }


LIFTING_DEFAULT_RX = {
    "R_N": 0.80,
    "theta_N": 0.50,
    "dx1": 0.45,
    "dx2": 0.55,
    "z_2_1": 0.55,
    "y_2_2": 0.65,
    "z_2_2": 0.40,
    "y_2_3": 0.55,
    "z_2_3": 0.35,
    "z_2_4": 0.15,
    "z_3_1": 0.55,
    "y_3_2": 0.75,
    "z_3_2": 0.45,
    "y_3_3": 0.60,
    "z_3_3": 0.25,
    "R_LE": 0.50,
    "theta_LE": 0.30,
    "t_mid": 0.60,
    "L_mid": 0.35,
    "L_w": 0.45,
    "xw2_xw1": 0.45,
    "xw3_xw1": 0.35,
    "xw4_xw1": 0.30,
    "yw1": 0.45,
    "yw3": 0.55,
    "dx_f_xw4": 0.60,
    "theta_f": 0.35,
    "f_el": 0.75,
    "L_bf": 0.35,
    "x_cg": 0.50,
}

CAPSULE_DEFAULT_RX = {
    "R_N": 0.55,
    "R_S": 0.35,
    "theta_c": 0.45,
    "R_m": 0.60,
    "L_c": 0.50,
    "x_cg": 0.50,
}


@dataclass(frozen=True)
class PhysicalParameters:
    family: str
    values: dict[str, float | str]
    normalized: dict[str, float]
    bounds: dict[str, tuple[float, float]]


def _rx(design: VehicleDesign, key: str, default: dict[str, float]) -> float:
    return float(design.rx.get(key, default[key]))


def map_lifting_body_parameters(design: VehicleDesign) -> PhysicalParameters:
    bounds = lifting_body_bounds(design.scale.body_length_m)
    rx = {**LIFTING_DEFAULT_RX, **design.rx}
    body_top_height = 0.90 * interp_from_rx(rx["z_3_1"], *bounds["body_top_height"])
    raw_body_belly_depth = interp_from_rx(rx["z_2_4"], *bounds["body_belly_depth"])
    payload_clearance = 0.01
    min_internal_height = design.payload.height_m + 2.0 * payload_clearance
    body_belly_depth = max(raw_body_belly_depth, 0.82 * body_top_height, min_internal_height - body_top_height)
    raw_body_half_width = interp_from_rx(rx["y_3_2"], *bounds["body_half_width"])
    min_payload_half_width = (design.payload.width_m + 2.0 * payload_clearance) / 1.35
    target_cylindrical_half_width = 1.05 * (body_top_height + body_belly_depth)
    values: dict[str, float | str] = {
        "L_body": design.scale.body_length_m,
        "R_N": interp_from_rx(_rx(design, "R_N", LIFTING_DEFAULT_RX), *bounds["R_N"]),
        "theta_N_deg": interp_from_rx(_rx(design, "theta_N", LIFTING_DEFAULT_RX), *bounds["theta_N_deg"]),
        "dx1": interp_from_rx(_rx(design, "dx1", LIFTING_DEFAULT_RX), *bounds["dx1"]),
        "dx2": interp_from_rx(_rx(design, "dx2", LIFTING_DEFAULT_RX), *bounds["dx2"]),
        "body_half_width": max(min_payload_half_width, min(raw_body_half_width, target_cylindrical_half_width)),
        "body_top_height": body_top_height,
        "body_belly_depth": body_belly_depth,
        "R_LE": interp_from_rx(rx["R_LE"], *bounds["R_LE"]),
        "theta_LE_deg": interp_from_rx(rx["theta_LE"], *bounds["theta_LE_deg"]),
        "L_w": interp_from_rx(rx["L_w"], *bounds["L_w"]),
        "L_mid_frac": interp_from_rx(rx["L_mid"], *bounds["L_mid_frac"]),
        "t_mid": interp_from_rx(rx["t_mid"], *bounds["t_mid"]),
        "semi_span": interp_from_rx(rx["yw1"], *bounds["semi_span"]),
        "winglet_bent_span_frac": interp_from_rx(rx["yw3"], *bounds["winglet_bent_span_frac"]),
        "theta_f_deg": interp_from_rx(rx["theta_f"], *bounds["theta_f_deg"]),
        "f_el": interp_from_rx(rx["f_el"], *bounds["f_el"]),
        "L_bf": interp_from_rx(rx["L_bf"], *bounds["L_bf"]),
        "sweep_deg": interp_from_rx(rx["xw2_xw1"], *bounds["sweep_deg"]),
        "x_cg_frac": interp_from_rx(rx["x_cg"], *bounds["x_cg_frac"]),
    }
    nose_bluntness = 0.5 * (rx["R_N"] + rx["theta_N"])
    nose_profile_codes = {"rounded": 0.0, "conic": 1.0, "triangular": 2.0}
    values["nose_profile"] = design.nose_profile
    values["nose_profile_code"] = nose_profile_codes[design.nose_profile]
    values["nose_station_frac"] = {
        "rounded": 0.0,
        "conic": 0.135,
        "triangular": 0.120,
    }[design.nose_profile]
    values["nose_ogive_exponent"] = interp_from_rx(1.0 - nose_bluntness, 1.0, 1.55)
    values["nose_top_scale"] = interp_from_rx(nose_bluntness, 0.62, 0.78)
    values["nose_shoulder_scale"] = interp_from_rx(nose_bluntness, 0.36, 0.52)
    values["nose_chine_scale"] = interp_from_rx(nose_bluntness, 0.62, 0.86)
    values["nose_belly_scale"] = interp_from_rx(nose_bluntness, 0.56, 0.70)
    values["forebody_width_scale"] = interp_from_rx(nose_bluntness, 0.74, 0.88)
    values["station2_shoulder_y_scale"] = interp_from_rx(rx["y_2_2"], 0.48, 0.68)
    values["station2_shoulder_z_scale"] = interp_from_rx(rx["z_2_2"], 0.42, 0.62)
    values["station2_chine_y_scale"] = interp_from_rx(rx["y_2_3"], 0.86, 1.06)
    values["station2_chine_z_scale"] = interp_from_rx(rx["z_2_3"], -0.16, -0.04)
    values["station2_belly_scale"] = interp_from_rx(rx["z_2_4"], 0.82, 1.02)
    values["station3_width_scale"] = interp_from_rx(rx["y_3_2"], 0.88, 1.06)
    values["station3_shoulder_y_scale"] = interp_from_rx(rx["y_3_3"], 0.48, 0.68)
    values["station3_shoulder_z_scale"] = interp_from_rx(rx["z_3_3"], 0.34, 0.54)
    values["fin_span_scale"] = interp_from_rx(rx["yw1"], 0.62, 1.02)
    values["dx_f_xw4"] = 0.05 * values["L_w"] + rx["dx_f_xw4"] * 0.20 * values["L_w"]
    values["payload_bay_length"] = design.payload.length_m
    values["payload_bay_width"] = design.payload.width_m
    values["payload_bay_height"] = design.payload.height_m
    values["payload_clearance_m"] = payload_clearance
    return PhysicalParameters("lifting_body", values, rx, bounds)


def map_capsule_parameters(design: VehicleDesign) -> PhysicalParameters:
    bounds = capsule_bounds(design.scale.body_length_m)
    rx = {**CAPSULE_DEFAULT_RX, **design.rx}
    values: dict[str, float | str] = {
        "L_body": design.scale.body_length_m,
        "R_N": interp_from_rx(rx["R_N"], *bounds["R_N"]),
        "R_S": interp_from_rx(rx["R_S"], *bounds["R_S"]),
        "theta_c_deg": interp_from_rx(rx["theta_c"], *bounds["theta_c_deg"]),
        "R_m": interp_from_rx(rx["R_m"], *bounds["R_m"]),
        "L_c": interp_from_rx(rx["L_c"], *bounds["L_c"]),
        "x_cg_frac": interp_from_rx(rx["x_cg"], *bounds["x_cg_frac"]),
    }
    values["payload_bay_length"] = design.payload.length_m
    values["payload_bay_width"] = design.payload.width_m
    values["payload_bay_height"] = design.payload.height_m
    values["payload_clearance_m"] = 0.05
    return PhysicalParameters("capsule", values, rx, bounds)


def map_design_parameters(design: VehicleDesign) -> PhysicalParameters:
    if design.geometry_family == "capsule":
        return map_capsule_parameters(design)
    return map_lifting_body_parameters(design)


def latin_hypercube(n: int, keys: list[str], seed: int | None = None) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    samples = np.zeros((n, len(keys)))
    for j in range(len(keys)):
        samples[:, j] = (np.arange(n) + rng.random(n)) / n
        rng.shuffle(samples[:, j])
    return [{key: float(samples[i, j]) for j, key in enumerate(keys)} for i in range(n)]
