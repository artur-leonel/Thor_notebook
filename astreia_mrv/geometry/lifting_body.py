from __future__ import annotations

import math

import numpy as np

from astreia_mrv.config import VehicleDesign
from astreia_mrv.geometry.mesh import (
    SurfaceMesh,
    VehicleGeometry,
    drop_tiny_face_islands,
    make_trimesh,
    orient_faces_away_from_centroid,
    validate_mesh,
    weld_vertices,
)
from astreia_mrv.math.hermite import sample_closed_hermite, sample_open_hermite
from astreia_mrv.parameters import PhysicalParameters, map_lifting_body_parameters


def _nose_station_x(params: dict[str, float | str]) -> float:
    l = float(params["L_body"])
    rn = float(params["R_N"])
    theta_n = math.radians(float(params["theta_N_deg"]))
    spherical_match = rn * (1.0 - math.cos(theta_n))
    station_frac = float(params.get("nose_station_frac", 0.0))
    return max(spherical_match, station_frac * l)


def _contour_points(params: dict[str, float | str], station: int, x: float, r1: float | None = None) -> np.ndarray:
    if station == 1:
        r = float(r1)
        nose_profile_code = int(float(params.get("nose_profile_code", 0.0)))
        if nose_profile_code == 1:
            yz = np.array(
                [
                    [0.0, 0.98 * r],
                    [0.90 * r, 0.52 * r],
                    [0.86 * r, -0.34 * r],
                    [0.0, -0.92 * r],
                    [-0.86 * r, -0.34 * r],
                    [-0.90 * r, 0.52 * r],
                ]
            )
        elif nose_profile_code == 2:
            yz = np.array(
                [
                    [0.0, 1.02 * r],
                    [1.04 * r, 0.02 * r],
                    [0.56 * r, -0.54 * r],
                    [0.0, -0.74 * r],
                    [-0.56 * r, -0.54 * r],
                    [-1.04 * r, 0.02 * r],
                ]
            )
        else:
            # Blunt hypersonic nose section: rounded enough for heating, but
            # with early shoulder/chine cues so it does not read as a plain tube.
            yz = np.array(
                [
                    [0.0, float(params["nose_top_scale"]) * r],
                    [float(params["nose_shoulder_scale"]) * r, 0.46 * r],
                    [float(params["nose_chine_scale"]) * r, -0.05 * r],
                    [0.0, -float(params["nose_belly_scale"]) * r],
                    [-float(params["nose_chine_scale"]) * r, -0.05 * r],
                    [-float(params["nose_shoulder_scale"]) * r, 0.46 * r],
                ]
            )
    else:
        if station == 2:
            hw_scale = float(params["forebody_width_scale"])
            top_scale = 0.96
            belly_scale = float(params["station2_belly_scale"])
            shoulder_y_scale = float(params["station2_shoulder_y_scale"])
            shoulder_z_scale = float(params["station2_shoulder_z_scale"])
            chine_y_scale = float(params["station2_chine_y_scale"])
            chine_z_scale = float(params["station2_chine_z_scale"])
        else:
            hw_scale = float(params["station3_width_scale"])
            top_scale = 0.64
            belly_scale = 0.70
            shoulder_y_scale = float(params["station3_shoulder_y_scale"])
            shoulder_z_scale = float(params["station3_shoulder_z_scale"])
            chine_y_scale = 1.0
            chine_z_scale = -0.08
        hw = float(params["body_half_width"]) * hw_scale
        top = float(params["body_top_height"]) * top_scale
        belly = float(params["body_belly_depth"]) * belly_scale
        yz = np.array(
            [
                [0.0, top],
                [shoulder_y_scale * hw, shoulder_z_scale * top],
                [chine_y_scale * hw, chine_z_scale * belly],
                [0.0, -0.92 * belly],
                [-chine_y_scale * hw, chine_z_scale * belly],
                [-shoulder_y_scale * hw, shoulder_z_scale * top],
            ]
        )
    return np.column_stack([np.full(6, x), yz[:, 0], yz[:, 1]])


def _sample_closed_polyline(points: np.ndarray, samples_per_segment: int = 8) -> np.ndarray:
    """Sample a closed polygon without Hermite rounding for the faceted variant."""
    points = np.asarray(points, dtype=float)
    samples = []
    for i, p0 in enumerate(points):
        p1 = points[(i + 1) % len(points)]
        for k in range(samples_per_segment):
            u = k / samples_per_segment
            samples.append((1.0 - u) * p0 + u * p1)
    return np.asarray(samples, dtype=float)


def _scale_z_signed(ring: np.ndarray, positive_scale: float, negative_scale: float) -> np.ndarray:
    scaled = ring.copy()
    positive = scaled[:, 2] >= 0.0
    scaled[positive, 2] *= positive_scale
    scaled[~positive, 2] *= negative_scale
    return scaled


def _conic_forebody_rings(
    ring_reference: np.ndarray,
    ring_midbody: np.ndarray,
    x_midbody: float,
    ring_count: int,
    size_exponent: float = 0.82,
    blend_start: float = 0.18,
    blend_end: float = 0.86,
) -> list[np.ndarray]:
    """Generate a single smooth forebody up to the first body station.

    Earlier versions ended the nose at an intermediate shoulder and then
    restarted the body expansion, which made a visible kink. This treats the
    shoulder as a shape reference only; station size evolves monotonically from
    the tip to the mid-body maximum.
    """
    ref = ring_reference.copy()
    mid = ring_midbody.copy()
    ref[:, 0] = 0.0
    mid[:, 0] = 0.0

    ref_y = max(float(np.max(np.abs(ref[:, 1]))), 1e-9)
    ref_top = max(float(np.max(ref[:, 2])), 1e-9)
    ref_belly = max(float(np.max(-ref[:, 2])), 1e-9)
    mid_y = max(float(np.max(np.abs(mid[:, 1]))), 1e-9)
    mid_top = max(float(np.max(mid[:, 2])), 1e-9)
    mid_belly = max(float(np.max(-mid[:, 2])), 1e-9)

    oval = ref.copy()
    oval[:, 1] *= mid_y / ref_y
    oval = _scale_z_signed(oval, mid_top / ref_top, mid_belly / ref_belly)

    rings: list[np.ndarray] = []
    for k in range(1, ring_count + 1):
        frac = k / ring_count
        x = x_midbody * frac
        size = math.sin(0.5 * math.pi * frac) ** size_exponent
        shape_blend = float(_smoothstep(blend_start, blend_end, frac))
        ring = (1.0 - shape_blend) * oval + shape_blend * mid
        ring[:, 0] = x
        ring[:, 1:] *= size
        rings.append(ring)
    return rings


def _connect_rings(ring_count: int, ring_size: int, start_index: int = 0, zone: str = "fuselage") -> tuple[list[list[int]], list[str]]:
    faces: list[list[int]] = []
    zones: list[str] = []
    for i in range(ring_count - 1):
        for j in range(ring_size):
            a = start_index + i * ring_size + j
            b = start_index + i * ring_size + (j + 1) % ring_size
            c = start_index + (i + 1) * ring_size + (j + 1) % ring_size
            d = start_index + (i + 1) * ring_size + j
            faces.append([a, d, c])
            faces.append([a, c, b])
            zones.extend([zone, zone])
    return faces, zones


def _smoothstep(edge0: float, edge1: float, value: float | np.ndarray) -> float | np.ndarray:
    t = np.clip((value - edge0) / np.maximum(np.asarray(edge1) - np.asarray(edge0), 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _smooth_window(value: float | np.ndarray, start: float, full_start: float, full_end: float, end: float) -> float | np.ndarray:
    return _smoothstep(start, full_start, value) * (1.0 - _smoothstep(full_end, end, value))


def _apply_integrated_chine_fins(surface: SurfaceMesh, params: dict[str, float]) -> SurfaceMesh:
    """Blend low-aspect swept control fins into the existing side/lower chine OML.

    This deliberately deforms the continuous fuselage vertex field instead of
    appending separate plates. The result stays one closed mesh, but the aft
    control zones have a real outboard planform silhouette instead of appearing
    only as denser tagged body faces.
    """
    vertices = surface.vertices.copy()
    l = params["L_body"]
    half_width = params["body_half_width"]
    top = params["body_top_height"]
    belly = params["body_belly_depth"]

    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    ay = np.abs(y)
    side_sign = np.sign(y)

    local_half_width = np.full_like(ay, half_width)
    rounded_x = np.round(x, 9)
    for x_value in np.unique(rounded_x):
        station = rounded_x == x_value
        station_half_width = float(np.max(ay[station]))
        if station_half_width > 1e-9:
            local_half_width[station] = station_half_width

    side_weight = np.clip((ay - 0.72 * local_half_width) / np.maximum(0.28 * local_half_width, 1e-9), 0.0, 1.0)
    side_weight = side_weight * side_weight * (3.0 - 2.0 * side_weight)
    lower_chine_weight = np.clip((0.34 * top - z) / max(0.34 * top + 0.72 * belly, 1e-9), 0.0, 1.0)
    lower_chine_weight = lower_chine_weight * lower_chine_weight * (3.0 - 2.0 * lower_chine_weight)

    mid_strake = _smooth_window(x, 0.52 * l, 0.68 * l, 0.82 * l, 0.91 * l)
    aft_fin = _smooth_window(x, 0.74 * l, 0.83 * l, 0.94 * l, 0.992 * l)
    planform = np.maximum(0.40 * mid_strake, 0.82 * aft_fin)

    max_outboard_extension = min(0.014 * l, 0.25 * half_width)
    outboard = max_outboard_extension * planform * side_weight * lower_chine_weight
    outboard[side_sign == 0.0] = 0.0

    vertices[:, 1] += side_sign * outboard
    vertices[:, 2] -= 0.18 * outboard * lower_chine_weight

    return orient_faces_away_from_centroid(SurfaceMesh(vertices, surface.faces.copy(), list(surface.face_zone)))


def _apply_futuristic_body_facets(surface: SurfaceMesh, params: dict[str, float]) -> SurfaceMesh:
    """Add subtle dart-like crown and chine faceting without changing topology."""
    vertices = surface.vertices.copy()
    l = params["L_body"]
    half_width = params["body_half_width"]
    top = params["body_top_height"]
    belly = params["body_belly_depth"]

    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    ay = np.abs(y)

    local_half_width = np.full_like(ay, half_width)
    rounded_x = np.round(x, 9)
    for x_value in np.unique(rounded_x):
        station = rounded_x == x_value
        station_half_width = float(np.max(ay[station]))
        if station_half_width > 1e-9:
            local_half_width[station] = station_half_width

    body_window = _smooth_window(x, 0.08 * l, 0.22 * l, 0.78 * l, 0.94 * l)
    crown_weight = np.exp(-((ay / np.maximum(0.30 * local_half_width, 1e-9)) ** 2))
    top_weight = _smoothstep(0.10 * top, 0.72 * top, z)
    spine = 0.010 * l * body_window * crown_weight * top_weight

    shoulder_weight = np.clip((ay - 0.45 * local_half_width) / np.maximum(0.45 * local_half_width, 1e-9), 0.0, 1.0)
    shoulder_weight = shoulder_weight * shoulder_weight * (3.0 - 2.0 * shoulder_weight)
    chine_window = _smooth_window(x, 0.18 * l, 0.36 * l, 0.84 * l, 0.98 * l)
    lower_weight = np.clip((0.24 * top - z) / max(0.24 * top + 0.82 * belly, 1e-9), 0.0, 1.0)
    chine_drop = 0.010 * l * chine_window * shoulder_weight * lower_weight

    vertices[:, 2] += spine
    vertices[:, 2] -= chine_drop
    return orient_faces_away_from_centroid(SurfaceMesh(vertices, surface.faces.copy(), list(surface.face_zone)))


def _local_station_extents(
    vertices: np.ndarray,
    default_half_width: float,
    default_top: float,
    default_belly: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local_half_width = np.full(len(vertices), default_half_width, dtype=float)
    local_top = np.full(len(vertices), default_top, dtype=float)
    local_belly = np.full(len(vertices), default_belly, dtype=float)
    rounded_x = np.round(vertices[:, 0], 9)
    for x_value in np.unique(rounded_x):
        station = rounded_x == x_value
        y = vertices[station, 1]
        z = vertices[station, 2]
        station_half_width = float(np.max(np.abs(y)))
        station_top = float(np.max(z))
        station_belly = float(np.max(-z))
        if station_half_width > 1e-9:
            local_half_width[station] = station_half_width
        if station_top > 1e-9:
            local_top[station] = station_top
        if station_belly > 1e-9:
            local_belly[station] = station_belly
    return local_half_width, local_top, local_belly


def _apply_tail_keel_stabilizers(surface: SurfaceMesh, params: dict[str, float]) -> SurfaceMesh:
    """Raise aft dorsal/ventral stabilizing keels from the same fuselage mesh.

    Appending a fin to a closed fuselage edge creates non-manifold four-face
    root edges. This deformation keeps the topology manifold while still
    giving the aft body a visible missile-like vertical stabilizer cue.
    """
    vertices = surface.vertices.copy()
    l = params["L_body"]
    half_width = params["body_half_width"]
    top = params["body_top_height"]
    belly = params["body_belly_depth"]

    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    ay = np.abs(y)
    local_half_width, local_top, local_belly = _local_station_extents(vertices, half_width, top, belly)

    tail_window = _smooth_window(x, 0.72 * l, 0.82 * l, 0.955 * l, 0.998 * l)
    center_weight = np.exp(-((ay / np.maximum(0.18 * local_half_width, 1e-9)) ** 4))
    dorsal_skin = _smoothstep(0.30 * local_top, 0.82 * local_top, z) * center_weight
    ventral_skin = _smoothstep(0.30 * local_belly, 0.82 * local_belly, -z) * center_weight

    dorsal_height = min(0.046 * l, 0.92 * half_width)
    ventral_height = min(0.038 * l, 0.82 * half_width)
    vertices[:, 2] += dorsal_height * tail_window * dorsal_skin
    vertices[:, 2] -= ventral_height * tail_window * ventral_skin

    keel_strength = tail_window * np.maximum(dorsal_skin, ventral_skin)
    vertices[:, 1] *= 1.0 - 0.12 * keel_strength
    return orient_faces_away_from_centroid(SurfaceMesh(vertices, surface.faces.copy(), list(surface.face_zone)))


def _root_band_local_indices(fuselage: SurfaceMesh, x0: float, x1: float, sign: float) -> tuple[list[np.ndarray], int, int]:
    candidate_groups = [
        group
        for group in _ring_vertex_groups(fuselage)
        if x0 <= float(fuselage.vertices[group[0], 0]) <= x1
    ]
    max_group_size = max((len(group) for group in candidate_groups), default=0)
    groups = [group for group in candidate_groups if len(group) >= max(12, int(0.75 * max_group_size))]
    if len(groups) < 3:
        raise ValueError("Not enough fuselage rings to root integrated fin panels")

    vertices = fuselage.vertices
    base = groups[len(groups) // 2]
    pts = vertices[base]
    side = sign * pts[:, 1] > 0.0
    lower_side = side & (pts[:, 2] <= 0.05)

    upper_score = sign * pts[:, 1] - 0.18 * np.abs(pts[:, 2] - 0.002)
    upper_score[~lower_side] = -np.inf
    j_upper = int(np.argmax(upper_score))

    lower_candidates = lower_side & (pts[:, 2] <= pts[j_upper, 2] + 0.006)
    lower_score = sign * pts[:, 1] - 0.10 * np.abs(pts[:, 2] + 0.055)
    lower_score[~lower_candidates] = -np.inf
    j_lower = int(np.argmax(lower_score))

    if j_lower == j_upper:
        candidates = np.flatnonzero(lower_side)
        lower_z = pts[candidates, 2]
        j_lower = int(candidates[np.argmin(lower_z)])
    if j_lower == j_upper:
        j_lower = (j_upper + (1 if sign > 0 else -1)) % len(base)
    return groups, j_upper, j_lower


def _add_integrated_fin_panels(fuselage: SurfaceMesh, params: dict[str, float]) -> SurfaceMesh:
    """Add explicit short aft canted fins rooted to fuselage ring vertices.

    The fins are meshed as thick, swept strips whose root edge reuses existing
    body vertices. They are intentionally fin-like rather than wing-like:
    shorter in y-span, with the tip canted upward/outboard to support
    conceptual hypersonic trim and low-speed roll/yaw allocation.
    """
    l = params["L_body"]
    half_width = params["body_half_width"]
    x0 = 0.70 * l
    x1 = 0.992 * l
    elevon_break = 0.865 * l
    span_max = min(0.090 * l, 1.45 * half_width) * params["fin_span_scale"]
    thickness = max(0.032, 0.30 * span_max)
    span_fracs = (0.45, 0.78, 1.0)

    vertices = fuselage.vertices.tolist()
    original_vertices = fuselage.vertices

    root_data = []
    for sign in (1.0, -1.0):
        groups, j_upper, j_lower = _root_band_local_indices(fuselage, x0, x1, sign)
        root_data.append((sign, groups, j_upper, j_lower))

    remove_vertices: set[int] = set()
    for _, groups, j_upper, j_lower in root_data:
        j0 = min(j_upper, j_lower)
        j1 = max(j_upper, j_lower)
        for group in groups:
            remove_vertices.update(int(group[j]) for j in range(j0, j1 + 1))

    kept_faces: list[list[int]] = []
    kept_zones: list[str] = []
    for face, zone in zip(fuselage.faces.tolist(), fuselage.face_zone):
        if all(int(vertex) in remove_vertices for vertex in face):
            continue
        kept_faces.append(face)
        kept_zones.append(zone)

    faces = kept_faces
    zones = kept_zones

    for sign, groups, j_upper, j_lower in root_data:
        root_upper = [int(group[j_upper]) for group in groups]
        root_lower = [int(group[j_lower]) for group in groups]

        upper_rows: list[list[int]] = [root_upper]
        lower_rows: list[list[int]] = [root_lower]
        for _ in span_fracs:
            upper_rows.append([])
            lower_rows.append([])

        for upper_idx, lower_idx in zip(root_upper, root_lower):
            upper = original_vertices[upper_idx]
            lower = original_vertices[lower_idx]
            root = 0.5 * (upper + lower)
            t = np.clip((root[0] - x0) / max(x1 - x0, 1e-9), 0.0, 1.0)
            leading = float(_smoothstep(0.0, 0.34, t))
            trailing = 1.0 - 0.38 * float(_smoothstep(0.74, 1.0, t))
            profile = leading * trailing
            span = span_max * profile
            root_side = max(abs(upper[1]), abs(lower[1]))
            for row_index, span_frac in enumerate(span_fracs, start=1):
                x_panel = root[0] + span_frac * span_max * 0.26 * (1.0 - t)
                y_panel = sign * (root_side + span * span_frac)
                z_center = root[2] + span * (0.18 + 0.92 * span_frac)
                local_thickness = thickness * (0.10 + 0.90 * profile) * (1.0 - 0.22 * span_frac)
                upper_rows[row_index].append(len(vertices))
                vertices.append([float(x_panel), float(y_panel), float(z_center + 0.5 * local_thickness)])
                lower_rows[row_index].append(len(vertices))
                vertices.append([float(x_panel), float(y_panel), float(z_center - 0.5 * local_thickness)])

        for i in range(len(groups) - 1):
            x_mid = 0.5 * (original_vertices[root_upper[i], 0] + original_vertices[root_upper[i + 1], 0])
            zone = "strake" if x_mid < elevon_break else "elevon"

            segment_faces = []
            for row in range(len(upper_rows) - 1):
                u00, u01 = upper_rows[row][i], upper_rows[row][i + 1]
                u10, u11 = upper_rows[row + 1][i], upper_rows[row + 1][i + 1]
                l00, l01 = lower_rows[row][i], lower_rows[row][i + 1]
                l10, l11 = lower_rows[row + 1][i], lower_rows[row + 1][i + 1]
                segment_faces.extend(
                    [
                        [u00, u01, u11],
                        [u00, u11, u10],
                        [l00, l10, l11],
                        [l00, l11, l01],
                    ]
                )

            tip_row = len(upper_rows) - 1
            tu0, tu1 = upper_rows[tip_row][i], upper_rows[tip_row][i + 1]
            tl0, tl1 = lower_rows[tip_row][i], lower_rows[tip_row][i + 1]
            segment_faces.extend([[tu0, tu1, tl1], [tu0, tl1, tl0]])
            faces.extend(segment_faces)
            zones.extend([zone] * len(segment_faces))

        for station_index, zone in ((0, "strake"), (-1, "elevon")):
            if j_upper <= j_lower:
                band = [int(groups[station_index][j]) for j in range(j_upper, j_lower + 1)]
            else:
                band = [int(groups[station_index][j]) for j in range(j_upper, j_lower - 1, -1)]
            cap = (
                [upper_rows[0][station_index]]
                + [row[station_index] for row in upper_rows[1:]]
                + [row[station_index] for row in reversed(lower_rows)]
                + list(reversed(band[1:-1]))
            )
            for i in range(1, len(cap) - 1):
                faces.append([cap[0], cap[i], cap[i + 1]])
                zones.append(zone)

    return orient_faces_away_from_centroid(
        SurfaceMesh(np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64), zones)
    )


def _build_fuselage(params: PhysicalParameters, contour_samples: int = 8, longitudinal_samples: int = 14) -> SurfaceMesh:
    p = params.values
    l = float(p["L_body"])
    rn = float(p["R_N"])
    theta_n = math.radians(float(p["theta_N_deg"]))
    spherical_r1 = rn * math.sin(theta_n)
    r1 = float(p.get("nose_station_radius", spherical_r1))
    spherical_x1 = rn * (1.0 - math.cos(theta_n))
    x1 = _nose_station_x(p)
    x2 = min(0.70 * l, spherical_x1 + float(p["dx1"]))
    x3 = min(0.94 * l, x2 + float(p["dx2"]))

    c1 = _contour_points(p, 1, x1, r1)
    c2 = _contour_points(p, 2, x2)
    c3 = _contour_points(p, 3, x3)
    nose_profile_code = int(float(p.get("nose_profile_code", 0.0)))
    ring1 = _sample_closed_polyline(c1, contour_samples) if nose_profile_code == 2 else sample_closed_hermite(c1, contour_samples)
    ring2 = sample_closed_hermite(c2, contour_samples)
    ring3 = sample_closed_hermite(c3, contour_samples)
    ring_size = len(ring1)

    if nose_profile_code == 1:
        nose_rings = _conic_forebody_rings(ring1, ring2, x2, max(16, longitudinal_samples + 4))
        path_rings = [ring2, ring3]
    elif nose_profile_code == 2:
        nose_rings = _conic_forebody_rings(
            ring1,
            ring2,
            x2,
            max(14, longitudinal_samples + 2),
            size_exponent=1.02,
            blend_start=0.30,
            blend_end=0.96,
        )
        path_rings = [ring2, ring3]
    else:
        nose_rings = []
        for k in range(1, max(4, longitudinal_samples // 2) + 1):
            frac = k / max(4, longitudinal_samples // 2)
            theta = frac * theta_n
            x = rn * (1.0 - math.cos(theta))
            scale = (rn * math.sin(theta)) / max(r1, 1e-9)
            scale = scale ** float(p["nose_ogive_exponent"])
            ring = ring1.copy()
            ring[:, 0] = x
            ring[:, 1:] *= scale
            nose_rings.append(ring)
        path_rings = [ring1, ring2, ring3]

    paths = []
    for j in range(ring_size):
        pts = np.vstack([ring[j] for ring in path_rings])
        paths.append(sample_open_hermite(pts, longitudinal_samples))
    fuselage_rings = [np.asarray([paths[j][i] for j in range(ring_size)]) for i in range(1, len(paths[0]))]

    taper_rings = []
    tail_start_frac = x3 / l
    tail_ring_fracs = (0.70, 0.76, 0.84, 0.91, 0.965, 0.99, 1.000)
    for x_frac in tail_ring_fracs:
        if x_frac <= tail_start_frac + 1e-6:
            continue
        taper = float(_smoothstep(tail_start_frac, 1.0, x_frac))
        y_scale = 1.0 - 0.60 * taper
        z_scale = 1.0 - 0.58 * taper
        ring = ring3.copy()
        ring[:, 0] = l * x_frac
        ring[:, 1] *= y_scale
        ring[:, 2] *= z_scale
        taper_rings.append(ring)

    all_rings = nose_rings + fuselage_rings + taper_rings
    vertices = [np.array([[0.0, 0.0, 0.0]])] + all_rings + [np.array([[l, 0.0, -0.006 * l]])]
    vertices_arr = np.vstack(vertices)

    faces: list[list[int]] = []
    zones: list[str] = []
    first_ring_start = 1
    for j in range(ring_size):
        a = first_ring_start + j
        b = first_ring_start + (j + 1) % ring_size
        faces.append([0, b, a])
        zones.append("nose_cap")

    ring_faces, ring_zones = _connect_rings(len(all_rings), ring_size, first_ring_start, "fuselage")
    faces.extend(ring_faces)
    zones.extend(ring_zones)

    tail_index = len(vertices_arr) - 1
    last_ring_start = first_ring_start + (len(all_rings) - 1) * ring_size
    for j in range(ring_size):
        a = last_ring_start + j
        b = last_ring_start + (j + 1) % ring_size
        faces.append([a, b, tail_index])
        zones.append("aft_closeout")

    surface = SurfaceMesh(vertices_arr, np.asarray(faces, dtype=np.int64), zones)
    return orient_faces_away_from_centroid(surface)


def _body_side_anchor(fuselage: SurfaceMesh, x_target: float, sign: float) -> np.ndarray:
    """Pick an existing side-body vertex near x_target for root-continuous wings."""
    vertices = fuselage.vertices
    for half_width in (0.08, 0.16, 0.32, 0.64):
        mask = (np.abs(vertices[:, 0] - x_target) <= half_width) & (sign * vertices[:, 1] > 0.0)
        candidates = vertices[mask]
        if len(candidates):
            side_index = int(np.argmax(sign * candidates[:, 1]))
            return candidates[side_index].copy()
    mask = sign * vertices[:, 1] > 0.0
    candidates = vertices[mask]
    side_index = int(np.argmax(sign * candidates[:, 1]))
    return candidates[side_index].copy()


def _ring_vertex_groups(fuselage: SurfaceMesh) -> list[np.ndarray]:
    vertices = fuselage.vertices
    rounded_x = np.round(vertices[:, 0], 9)
    groups = []
    for x_value in np.unique(rounded_x):
        indices = np.flatnonzero(rounded_x == x_value)
        if len(indices) > 3:
            groups.append(indices)
    return groups


def _side_root_path(fuselage: SurfaceMesh, x0: float, x1: float, sign: float) -> np.ndarray:
    path = []
    vertices = fuselage.vertices
    for group in _ring_vertex_groups(fuselage):
        x = float(vertices[group[0], 0])
        if x0 <= x <= x1:
            side = group[sign * vertices[group, 1] > 0.0]
            if len(side):
                path.append(vertices[side[np.argmax(sign * vertices[side, 1])]].copy())

    if len(path) < 3:
        path = [_body_side_anchor(fuselage, x0, sign), _body_side_anchor(fuselage, 0.5 * (x0 + x1), sign), _body_side_anchor(fuselage, x1, sign)]
    return np.asarray(path)


def _side_lower_root_path(fuselage: SurfaceMesh, x0: float, x1: float, sign: float) -> np.ndarray:
    path = []
    vertices = fuselage.vertices
    for group in _ring_vertex_groups(fuselage):
        x = float(vertices[group[0], 0])
        if not x0 <= x <= x1:
            continue
        side = group[(sign * vertices[group, 1] > 0.0) & (vertices[group, 2] <= 0.03)]
        if len(side):
            score = sign * vertices[side, 1] - 0.35 * np.abs(vertices[side, 2] + 0.025)
            path.append(vertices[side[int(np.argmax(score))]].copy())

    if len(path) < 3:
        return _side_root_path(fuselage, x0, x1, sign)
    return np.asarray(path)


def _nearest_ring(fuselage: SurfaceMesh, x_target: float) -> np.ndarray:
    groups = _ring_vertex_groups(fuselage)
    vertices = fuselage.vertices
    if not groups:
        return np.arange(len(vertices), dtype=np.int64)
    return min(groups, key=lambda group: abs(float(vertices[group[0], 0]) - x_target))


def _section_vertex(
    fuselage: SurfaceMesh,
    x_target: float,
    y_target: float,
    z_target: float,
    y_weight: float = 1.0,
    z_weight: float = 1.0,
) -> np.ndarray:
    vertices = fuselage.vertices
    ring = _nearest_ring(fuselage, x_target)
    pts = vertices[ring]
    score = y_weight * (pts[:, 1] - y_target) ** 2 + z_weight * (pts[:, 2] - z_target) ** 2
    return pts[int(np.argmin(score))].copy()


def _section_side_vertex(fuselage: SurfaceMesh, x_target: float, sign: float, z_target: float = -0.025) -> np.ndarray:
    vertices = fuselage.vertices
    ring = _nearest_ring(fuselage, x_target)
    pts = vertices[ring]
    side = pts[sign * pts[:, 1] > 0.0]
    if len(side) == 0:
        return _body_side_anchor(fuselage, x_target, sign)
    score = sign * side[:, 1] - 0.45 * np.abs(side[:, 2] - z_target)
    return side[int(np.argmax(score))].copy()


def _face_subset(mesh: SurfaceMesh, mask: np.ndarray, zone_name: str) -> SurfaceMesh:
    faces = mesh.faces[mask]
    if len(faces) == 0:
        return SurfaceMesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), [])
    used = np.unique(faces)
    remap = {old: new for new, old in enumerate(used)}
    remapped_faces = np.vectorize(remap.__getitem__)(faces)
    return SurfaceMesh(mesh.vertices[used], remapped_faces, [zone_name] * len(faces))


def _tag_integrated_control_surfaces(
    fuselage: SurfaceMesh, params: dict[str, float]
) -> tuple[SurfaceMesh, dict[str, SurfaceMesh]]:
    """Tag control regions on the continuous finned OML instead of adding separate islands."""
    l = params["L_body"]
    half_width = params["body_half_width"]
    outer_half_width = float(np.max(np.abs(fuselage.vertices[:, 1])))
    belly = params["body_belly_depth"]
    top = params["body_top_height"]
    tri = fuselage.vertices[fuselage.faces]
    centroids = tri.mean(axis=1)
    x = centroids[:, 0]
    y = centroids[:, 1]
    z = centroids[:, 2]
    ay = np.abs(y)
    existing_zones = np.asarray(fuselage.face_zone, dtype=object)

    dorsal_fin_mask = (
        (x >= 0.74 * l)
        & (x <= 0.990 * l)
        & (ay <= 0.34 * half_width)
        & (z >= 0.54 * top)
    ) | (existing_zones == "dorsal_fin")
    ventral_fin_mask = (
        (x >= 0.74 * l)
        & (x <= 0.990 * l)
        & (ay <= 0.34 * half_width)
        & (z <= -0.30 * belly)
    ) | (existing_zones == "ventral_fin")
    body_flap_mask = (
        (x >= 0.66 * l)
        & (x <= 0.985 * l)
        & (ay <= 0.82 * half_width)
        & (z <= -0.12 * belly)
        & ~ventral_fin_mask
    )
    right_elevon_mask = (
        (x >= 0.66 * l)
        & (x <= 0.975 * l)
        & (y >= 0.50 * half_width)
        & (y <= 1.01 * outer_half_width)
        & (z <= 0.06 * top)
        & ~body_flap_mask
    )
    left_elevon_mask = (
        (x >= 0.66 * l)
        & (x <= 0.975 * l)
        & (y <= -0.50 * half_width)
        & (y >= -1.01 * outer_half_width)
        & (z <= 0.06 * top)
        & ~body_flap_mask
    )
    right_strake_mask = (
        (x >= 0.52 * l)
        & (x < 0.72 * l)
        & (y >= 0.50 * half_width)
        & (y <= 1.01 * outer_half_width)
        & (z >= -0.62 * belly)
        & (z <= 0.28 * top)
    )
    left_strake_mask = (
        (x >= 0.52 * l)
        & (x < 0.72 * l)
        & (y <= -0.50 * half_width)
        & (y >= -1.01 * outer_half_width)
        & (z >= -0.62 * belly)
        & (z <= 0.28 * top)
    )

    right_elevon_mask = right_elevon_mask | ((existing_zones == "elevon") & (y > 0.0))
    left_elevon_mask = left_elevon_mask | ((existing_zones == "elevon") & (y < 0.0))
    right_strake_mask = right_strake_mask | ((existing_zones == "strake") & (y > 0.0))
    left_strake_mask = left_strake_mask | ((existing_zones == "strake") & (y < 0.0))

    zones = list(fuselage.face_zone)
    for idx in np.flatnonzero(right_strake_mask | left_strake_mask):
        zones[int(idx)] = "strake"
    for idx in np.flatnonzero(right_elevon_mask | left_elevon_mask):
        zones[int(idx)] = "elevon"
    for idx in np.flatnonzero(body_flap_mask):
        zones[int(idx)] = "body_flap"
    for idx in np.flatnonzero(dorsal_fin_mask):
        zones[int(idx)] = "dorsal_fin"
    for idx in np.flatnonzero(ventral_fin_mask):
        zones[int(idx)] = "ventral_fin"

    tagged = SurfaceMesh(fuselage.vertices.copy(), fuselage.faces.copy(), zones)
    controls = {
        "body_flap": _face_subset(tagged, body_flap_mask, "body_flap"),
        "right_elevon": _face_subset(tagged, right_elevon_mask, "elevon"),
        "left_elevon": _face_subset(tagged, left_elevon_mask, "elevon"),
        "right_strake": _face_subset(tagged, right_strake_mask, "strake"),
        "left_strake": _face_subset(tagged, left_strake_mask, "strake"),
        "dorsal_fin": _face_subset(tagged, dorsal_fin_mask, "dorsal_fin"),
        "ventral_fin": _face_subset(tagged, ventral_fin_mask, "ventral_fin"),
    }
    return tagged, controls


def generate_lifting_body(design: VehicleDesign) -> VehicleGeometry:
    params = map_lifting_body_parameters(design)
    fuselage = _build_fuselage(params)
    p = params.values
    fuselage = _apply_integrated_chine_fins(fuselage, p)
    fuselage = _apply_futuristic_body_facets(fuselage, p)
    fuselage = _apply_tail_keel_stabilizers(fuselage, p)
    fuselage = _add_integrated_fin_panels(fuselage, p)
    mesh, control_surfaces = _tag_integrated_control_surfaces(fuselage, p)
    mesh = weld_vertices(mesh)
    mesh = drop_tiny_face_islands(mesh, max_fraction=0.001)
    mesh = orient_faces_away_from_centroid(mesh)
    tri = make_trimesh(mesh)
    bounds = np.asarray(tri.bounds)
    width = bounds[1, 1] - bounds[0, 1]
    core_width = 2.0 * p["body_half_width"]
    fin_extension = max(0.0, 0.5 * width - p["body_half_width"])
    reference_area = p["L_body"] * width
    payload_box = (design.payload.length_m, design.payload.width_m, design.payload.height_m)
    metadata = {
        "family": "lifting_body",
        "nose_profile": design.nose_profile,
        "parameters": p,
        "normalized_rx": params.normalized,
        "stations": {
            "x_nose_match_m": _nose_station_x(p),
            "x_middle_m": min(
                0.70 * float(p["L_body"]),
                float(p["R_N"]) * (1.0 - math.cos(math.radians(float(p["theta_N_deg"])))) + float(p["dx1"]),
            ),
            "x_rear_m": min(
                0.94 * float(p["L_body"]),
                float(p["R_N"]) * (1.0 - math.cos(math.radians(float(p["theta_N_deg"]))))
                + float(p["dx1"])
                + float(p["dx2"]),
            ),
        },
        "control_surfaces": {
            "body_flap": {
                "deflection_limits_deg": [-25.0, 20.0],
                "role": "integrated aft windward pitch-trim face zone",
            },
            "elevons": {
                "deflection_limits_deg": [-20.0, 20.0],
                "role": "integrated aft lower-chine fin/elevon face zones on the continuous OML",
            },
            "strakes": {
                "role": "continuous low-aspect swept chine fins blended into the body mesh",
            },
            "dorsal_ventral_tail_fins": {
                "role": "Barracuda-inspired continuous aft keel stabilizers for conceptual yaw damping/allocation",
            },
            "geometry": {
                "core_body_width_m": float(core_width),
                "total_span_m": float(width),
                "per_side_fin_extension_m": float(fin_extension),
                "topology": "single continuous deformed fuselage mesh; no separate glued plate components",
            },
        },
        "recovery": {
            "landing_accuracy_mode": "terminal recovery into authorized recovery zone",
            "mode": design.recovery_mode,
        },
        "mesh_quality": validate_mesh(mesh),
    }
    return VehicleGeometry(
        mesh=mesh,
        reference_area_m2=float(reference_area),
        reference_length_m=p["L_body"],
        volume_m3=float(abs(tri.volume)),
        wetted_area_m2=float(tri.area),
        payload_box=payload_box,
        control_surfaces=control_surfaces,
        metadata=metadata,
    )
