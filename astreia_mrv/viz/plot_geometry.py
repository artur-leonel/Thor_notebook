from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.collections import LineCollection, PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from astreia_mrv.geometry.mesh import VehicleGeometry


BODY_COLOR = "#56616c"
BODY_EDGE_COLOR = "#303a43"
CONTROL_EDGE_COLOR = "#0f1720"
CONTROL_COLORS = {
    "strake": BODY_COLOR,
    "elevon": BODY_COLOR,
    "body_flap": BODY_COLOR,
    "dorsal_fin": BODY_COLOR,
    "ventral_fin": BODY_COLOR,
}

ZONE_COLORS = {
    "nose_cap": BODY_COLOR,
    "fuselage": BODY_COLOR,
    "aft_closeout": BODY_COLOR,
    "strake": CONTROL_COLORS["strake"],
    "winglet": BODY_COLOR,
    "body_flap": CONTROL_COLORS["body_flap"],
    "elevon": CONTROL_COLORS["elevon"],
    "dorsal_fin": CONTROL_COLORS["dorsal_fin"],
    "ventral_fin": CONTROL_COLORS["ventral_fin"],
    "capsule_aeroshell": BODY_COLOR,
    "default": BODY_COLOR,
}


def _mesh_tris_and_colors(geometry: VehicleGeometry):
    verts = geometry.mesh.vertices
    faces = geometry.mesh.faces
    tri = verts[faces]
    areas = geometry.mesh.face_areas
    keep = areas > 1e-10
    tri = tri[keep]
    zones = [z for z, k in zip(geometry.mesh.face_zone, keep) if k]
    colors = _face_colors(zones, len(tri))
    return tri, colors, zones


def _axis_limits(
    ax,
    coords: np.ndarray,
    pad_frac: float = 0.08,
    half_span: float | None = None,
    equal_scale: bool = False,
) -> None:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center = 0.5 * (mins + maxs)
    if equal_scale and half_span is None:
        span = max(float(np.max(maxs - mins)), 1e-6)
        half = 0.5 * span * (1.0 + pad_frac)
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
    elif half_span is None:
        spans = np.maximum(maxs - mins, 1e-6)
        halfs = 0.5 * spans * (1.0 + pad_frac)
        ax.set_xlim(center[0] - halfs[0], center[0] + halfs[0])
        ax.set_ylim(center[1] - halfs[1], center[1] + halfs[1])
    else:
        half = half_span * (1.0 + pad_frac)
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)


def _projection_bounds(verts: np.ndarray, axes: tuple[int, int], pad_frac: float = 0.18) -> tuple[float, float, float, float]:
    coords = verts[:, axes]
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    pad = np.maximum(pad_frac * spans, 0.018)
    return (
        float(mins[0] - pad[0]),
        float(maxs[0] + pad[0]),
        float(mins[1] - pad[1]),
        float(maxs[1] + pad[1]),
    )


def _set_projected_limits(
    ax,
    verts: np.ndarray,
    axes: tuple[int, int],
    equal_scale: bool = True,
    pad_frac: float = 0.18,
) -> None:
    xmin, xmax, ymin, ymax = _projection_bounds(verts, axes, pad_frac=pad_frac)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal" if equal_scale else "auto", adjustable="box")


def _projected_edge_segments(tri: np.ndarray, axes: tuple[int, int]) -> list[np.ndarray]:
    segments: list[np.ndarray] = []
    for face in tri:
        projected = face[:, axes]
        segments.extend([projected[[0, 1]], projected[[1, 2]], projected[[2, 0]]])
    return segments


def _face_colors(face_zone: list[str], count: int) -> list[str]:
    return [ZONE_COLORS.get(zone, ZONE_COLORS["default"]) for zone in face_zone[:count]]


def _lit_face_colors(tri: np.ndarray, base_colors: list[str] | str) -> np.ndarray:
    """Compute simple directional lighting from triangle normals."""
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    good = lengths > 1e-12
    normals[good] /= lengths[good, None]

    key = np.array([-0.20, -0.55, 0.80])
    fill = np.array([0.72, 0.12, 0.28])
    rim = np.array([0.15, 0.90, 0.35])
    key /= np.linalg.norm(key)
    fill /= np.linalg.norm(fill)
    rim /= np.linalg.norm(rim)

    intensity = (
        0.22
        + 0.82 * np.maximum(0.0, normals @ key)
        + 0.20 * np.maximum(0.0, normals @ fill)
        + 0.24 * np.maximum(0.0, normals @ rim)
    )
    intensity = np.clip(intensity, 0.20, 1.22)

    if isinstance(base_colors, str):
        rgb = np.tile(np.asarray(to_rgb(base_colors)), (len(tri), 1))
    else:
        rgb = np.asarray([to_rgb(color) for color in base_colors], dtype=float)
    lit = np.clip(rgb * intensity[:, None], 0.0, 1.0)
    alpha = np.full((len(tri), 1), 1.0)
    return np.hstack([lit, alpha])


def _set_render_axes(ax, verts: np.ndarray, aspect: tuple[float, float, float] | None = None) -> None:
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    pad = np.maximum(0.18 * spans, 0.030)
    ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
    ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
    ax.set_zlim(mins[2] - pad[2], maxs[2] + pad[2])
    ax.set_box_aspect(tuple(spans + 2.0 * pad if aspect is None else aspect))
    if hasattr(ax, "set_proj_type"):
        try:
            ax.set_proj_type("persp", focal_length=0.85)
        except TypeError:
            ax.set_proj_type("persp")


def _control_edge_colors(zones: list[str]) -> list[str]:
    return [CONTROL_EDGE_COLOR if zone in CONTROL_COLORS else BODY_EDGE_COLOR for zone in zones]


def _mesh_linewidths(zones: list[str], body_width: float = 0.035, control_width: float = 0.20) -> list[float]:
    return [control_width if zone in CONTROL_COLORS else body_width for zone in zones]


def _face_normals(tri: np.ndarray) -> np.ndarray:
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    good = lengths > 1e-12
    normals[good] /= lengths[good, None]
    return normals


def _visible_face_mask(tri: np.ndarray, view_axis: int, sign: float, threshold: float = -0.035) -> np.ndarray:
    """Cull back-facing triangles for clean orthographic drawing views."""
    normals = _face_normals(tri)
    return normals[:, view_axis] * sign >= threshold


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    pts = np.unique(np.round(points, 10), axis=0)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    if len(pts) <= 2:
        return pts

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for point in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)

    upper: list[np.ndarray] = []
    for point in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)

    return np.asarray(lower[:-1] + upper[:-1])


def _draw_projection_mesh(
    ax,
    tri: np.ndarray,
    zones: list[str],
    axes: tuple[int, int],
    view_axis: int,
    sign: float,
    title: str,
) -> np.ndarray:
    projected_vertices = tri.reshape(-1, 3)[:, axes]
    visible = _visible_face_mask(tri, view_axis, sign, threshold=-0.18)
    if not np.any(visible):
        visible = np.ones(len(tri), dtype=bool)

    visible_indices = np.flatnonzero(visible)
    # Draw far triangles first, near triangles last. This keeps every
    # orthographic view tied to the same mesh instead of to a projected hull.
    depth = sign * tri[visible_indices, :, view_axis].mean(axis=1)
    ordered_indices = visible_indices[np.argsort(depth)]
    ordered_tri = tri[ordered_indices]
    ordered_zones = [zones[int(idx)] for idx in ordered_indices]
    ordered_polys = [face[:, axes] for face in ordered_tri]

    footprint_polys = [face[:, axes] for face in tri]
    if footprint_polys:
        ax.add_collection(
            PolyCollection(
                footprint_polys,
                facecolors=[to_rgb(BODY_COLOR) + (0.26,)] * len(footprint_polys),
                edgecolors="none",
                linewidths=0.0,
                alpha=1.0,
                zorder=1,
            )
        )

    normals = _face_normals(ordered_tri)
    shade = np.clip(0.48 + 0.52 * np.abs(normals[:, view_axis]), 0.42, 1.0)
    base = np.asarray(to_rgb(BODY_COLOR))
    facecolors = [tuple(np.clip(base * value, 0.0, 1.0)) + (0.88,) for value in shade]
    edgecolors = _control_edge_colors(ordered_zones)
    linewidths = _mesh_linewidths(ordered_zones, body_width=0.060, control_width=0.22)

    if ordered_polys:
        ax.add_collection(
            PolyCollection(
                ordered_polys,
                facecolors=facecolors,
                edgecolors=edgecolors,
                linewidths=linewidths,
                alpha=1.0,
                zorder=3,
            )
        )

    visible_segments = _projected_edge_segments(ordered_tri, axes)
    if visible_segments:
        ax.add_collection(
            LineCollection(
                visible_segments,
                colors=BODY_EDGE_COLOR,
                linewidths=0.045,
                alpha=0.52,
                zorder=5,
                clip_on=False,
            )
        )

    ax.set_title(title)
    ax.grid(True, linewidth=0.25, alpha=0.28)
    return projected_vertices


def _finish_projection_axis(
    ax,
    coords: np.ndarray,
    labels: tuple[str, str],
    equal_scale: bool = False,
    pad_frac: float = 0.10,
) -> None:
    _axis_limits(ax, coords, pad_frac=pad_frac, equal_scale=False)
    ax.set_aspect("equal" if equal_scale else "auto", adjustable="box")
    ax.set_xlabel(f"{labels[0]} [m]")
    ax.set_ylabel(f"{labels[1]} [m]")


def _set_manual_limits(ax, xmin: float, xmax: float, ymin: float, ymax: float, equal_scale: bool = False) -> None:
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal" if equal_scale else "auto", adjustable="box")


def _dimension_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    text: str,
    text_offset: tuple[float, float] = (0.0, 0.0),
    rotation: float = 0.0,
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="<->", lw=0.85, color="#111827"),
        annotation_clip=False,
        zorder=8,
    )
    mid = (0.5 * (start[0] + end[0]) + text_offset[0], 0.5 * (start[1] + end[1]) + text_offset[1])
    ax.text(
        mid[0],
        mid[1],
        text,
        ha="center",
        va="center",
        rotation=rotation,
        fontsize=8,
        fontweight="bold",
        color="#111827",
        zorder=9,
    )


def _projected_polys(
    tri: np.ndarray,
    zones: list[str],
    i: int,
    j: int,
    hidden: int,
) -> tuple[list[np.ndarray], list[int]]:
    order = np.argsort(tri[:, :, hidden].mean(axis=1))
    return [tri[idx][:, [i, j]] for idx in order], order.tolist()


def _add_lit_surface(
    ax,
    tri: np.ndarray,
    colors: list[str],
    zones: list[str],
    verts: np.ndarray,
    control_edges: bool = True,
    control_fill: bool = False,
) -> None:
    edgecolors = _control_edge_colors(zones) if control_edges else "none"
    linewidths = _mesh_linewidths(zones, body_width=0.035, control_width=0.16) if control_edges else 0.0
    face_colors = colors if control_fill else [BODY_COLOR] * len(zones)
    ax.add_collection3d(
        Poly3DCollection(
            tri,
            facecolors=_lit_face_colors(tri, face_colors),
            edgecolors=edgecolors,
            linewidths=linewidths,
            alpha=1.0,
        )
    )


def _filter_triangles_by_x(
    tri: np.ndarray,
    colors: list[str],
    zones: list[str],
    x_min: float,
) -> tuple[np.ndarray, list[str], list[str]]:
    centroids = tri.mean(axis=1)
    keep = centroids[:, 0] >= x_min
    return tri[keep], [color for color, k in zip(colors, keep) if k], [zone for zone, k in zip(zones, keep) if k]


def _style_render_axis(ax, verts: np.ndarray, title: str, elev: float, azim: float) -> None:
    _set_render_axes(ax, verts)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, color="#111827", pad=8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
    ax.grid(visible=False)


def save_three_view(geometry: VehicleGeometry, path: str | Path) -> None:
    """Save clean orthographic projections with hidden faces culled."""
    verts = geometry.mesh.vertices
    tri, colors, zones = _mesh_tris_and_colors(geometry)
    fig = plt.figure(figsize=(14, 8.0))
    axes = [
        fig.add_axes((0.06, 0.62, 0.40, 0.26)),
        fig.add_axes((0.54, 0.62, 0.40, 0.26)),
        fig.add_axes((0.06, 0.16, 0.54, 0.26)),
        fig.add_axes((0.68, 0.10, 0.25, 0.40)),
    ]
    views = [
        ((0, 1), 2, 1.0, "Top x-y (+z skin)", ("x", "y")),
        ((0, 1), 2, -1.0, "Bottom x-y (-z skin)", ("x", "y")),
        ((0, 2), 1, 1.0, "Side x-z", ("x", "z")),
        ((1, 2), 0, 1.0, "Aft y-z", ("y", "z")),
    ]
    for ax, (projection_axes, view_axis, sign, title, labels) in zip(axes, views):
        _draw_projection_mesh(ax, tri, zones, projection_axes, view_axis, sign, title)
        _set_projected_limits(ax, verts, projection_axes, equal_scale=True, pad_frac=0.20)
        ax.set_xlabel(f"{labels[0]} [m]")
        ax.set_ylabel(f"{labels[1]} [m]")
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def save_engineering_sheet(geometry: VehicleGeometry, path: str | Path) -> None:
    """Create a drawing-like multi-view sheet with clean silhouettes and visible surface mesh."""
    verts = geometry.mesh.vertices
    tri, colors, zones = _mesh_tris_and_colors(geometry)

    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    x0, x1 = mins[0], maxs[0]
    y0, y1 = mins[1], maxs[1]
    z0, z1 = mins[2], maxs[2]
    l = x1 - x0
    b = y1 - y0
    h = z1 - z0

    fig = plt.figure(figsize=(14.0, 10.4))

    ax_top = fig.add_axes((0.07, 0.75, 0.40, 0.16))
    _draw_projection_mesh(ax_top, tri, zones, (0, 1), 2, 1.0, "Top planform (+z)")
    _set_projected_limits(ax_top, verts, (0, 1), equal_scale=True, pad_frac=0.24)
    ax_top.set_xlabel("x [m]")
    ax_top.set_ylabel("y [m]")
    _dimension_arrow(ax_top, (x0, y0 - 0.16 * b), (x1, y0 - 0.16 * b), f"L = {l:.3f} m", (0.0, -0.055 * b))
    _dimension_arrow(
        ax_top,
        (x1 + 0.05 * l, y0),
        (x1 + 0.05 * l, y1),
        f"B = {b:.3f} m",
        (0.045 * l, 0.0),
        rotation=90,
    )

    ax_bottom = fig.add_axes((0.53, 0.75, 0.40, 0.16))
    _draw_projection_mesh(ax_bottom, tri, zones, (0, 1), 2, -1.0, "Bottom planform (-z fins/flap)")
    _set_projected_limits(ax_bottom, verts, (0, 1), equal_scale=True, pad_frac=0.24)
    ax_bottom.set_xlabel("x [m]")
    ax_bottom.set_ylabel("y [m]")

    ax_side = fig.add_axes((0.08, 0.50, 0.84, 0.15))
    _draw_projection_mesh(ax_side, tri, zones, (0, 2), 1, 1.0, "Side profile")
    _set_projected_limits(ax_side, verts, (0, 2), equal_scale=True, pad_frac=0.24)
    ax_side.set_xlabel("x [m]")
    ax_side.set_ylabel("z [m]")
    _dimension_arrow(ax_side, (x0, z0 - 0.16 * h), (x1, z0 - 0.16 * h), f"L = {l:.3f} m", (0.0, -0.055 * h))
    _dimension_arrow(
        ax_side,
        (x1 + 0.05 * l, z0),
        (x1 + 0.05 * l, z1),
        f"H = {h:.3f} m",
        (0.045 * l, 0.0),
        rotation=90,
    )

    ax_aft = fig.add_axes((0.08, 0.09, 0.38, 0.30))
    _draw_projection_mesh(ax_aft, tri, zones, (1, 2), 0, 1.0, "Aft view")
    _set_projected_limits(ax_aft, verts, (1, 2), equal_scale=True, pad_frac=0.24)
    ax_aft.set_xlabel("y [m]")
    ax_aft.set_ylabel("z [m]")
    _dimension_arrow(ax_aft, (y0, z0 - 0.16 * h), (y1, z0 - 0.16 * h), f"B = {b:.3f} m", (0.0, -0.055 * h))
    _dimension_arrow(
        ax_aft,
        (y1 + 0.10 * b, z0),
        (y1 + 0.10 * b, z1),
        f"H = {h:.3f} m",
        (0.10 * b, 0.0),
        rotation=90,
    )

    ax_iso = fig.add_axes((0.54, 0.08, 0.38, 0.33), projection="3d")
    _add_lit_surface(ax_iso, tri, colors, zones, verts)
    _style_render_axis(ax_iso, verts, "Oblique lower-aft solid view", elev=-14, azim=-52)

    fig.suptitle("Astreia-MRV Conceptual Engineering Sheet", fontsize=14, y=0.96)
    fig.text(
        0.04,
        0.02,
        "Dimensions: L length, B span/width, H height\nUnits: m",
        fontsize=8,
        color="#111827",
    )
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.20)
    plt.close(fig)


def save_shaded_render(geometry: VehicleGeometry, path: str | Path) -> None:
    verts = geometry.mesh.vertices
    tri, colors, zones = _mesh_tris_and_colors(geometry)

    fig = plt.figure(figsize=(13, 6.8))
    ax_upper = fig.add_subplot(1, 2, 1, projection="3d")
    _add_lit_surface(ax_upper, tri, colors, zones, verts)
    _style_render_axis(ax_upper, verts, "Upper/aft control view", elev=25, azim=-48)

    ax_lower = fig.add_subplot(1, 2, 2, projection="3d")
    _add_lit_surface(ax_lower, tri, colors, zones, verts)
    _style_render_axis(ax_lower, verts, "Underside body-flap view", elev=-18, azim=-55)

    fig.suptitle("Astreia-MRV lit solid inspection render", fontsize=15, y=0.96)
    fig.text(0.50, 0.04, "Continuous gray OML; darkened mesh edges mark integrated control-zone faces", ha="center", color="#111827")
    fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.10, wspace=0.08)
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.20)
    plt.close(fig)


def save_control_surface_inspection(geometry: VehicleGeometry, path: str | Path) -> None:
    """Save an aft-focused inspection sheet for the integrated fins/control zones."""
    verts = geometry.mesh.vertices
    tri, colors, zones = _mesh_tris_and_colors(geometry)

    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    length = maxs[0] - mins[0]
    width = maxs[1] - mins[1]
    height = maxs[2] - mins[2]
    x_cut = mins[0] + 0.42 * length
    tri_aft, colors_aft, zones_aft = _filter_triangles_by_x(tri, colors, zones, x_cut)
    verts_aft = tri_aft.reshape(-1, 3)

    fig = plt.figure(figsize=(13.5, 8.8))

    ax_plan = fig.add_axes((0.07, 0.66, 0.40, 0.23))
    _draw_projection_mesh(ax_plan, tri_aft, zones_aft, (0, 1), 2, 1.0, "Aft upper planform")
    _set_projected_limits(ax_plan, verts_aft, (0, 1), equal_scale=True, pad_frac=0.26)
    ax_plan.set_xlabel("x [m]")
    ax_plan.set_ylabel("y [m]")

    ax_bottom = fig.add_axes((0.53, 0.66, 0.40, 0.23))
    _draw_projection_mesh(ax_bottom, tri_aft, zones_aft, (0, 1), 2, -1.0, "Aft underside planform")
    _set_projected_limits(ax_bottom, verts_aft, (0, 1), equal_scale=True, pad_frac=0.26)
    ax_bottom.set_xlabel("x [m]")
    ax_bottom.set_ylabel("y [m]")

    ax_aft = fig.add_axes((0.07, 0.14, 0.36, 0.35))
    _draw_projection_mesh(ax_aft, tri_aft, zones_aft, (1, 2), 0, 1.0, "Aft fin/control section")
    _set_projected_limits(ax_aft, verts_aft, (1, 2), equal_scale=True, pad_frac=0.26)
    ax_aft.set_xlabel("y [m]")
    ax_aft.set_ylabel("z [m]")

    ax_iso = fig.add_axes((0.52, 0.12, 0.40, 0.40), projection="3d")
    _add_lit_surface(ax_iso, tri_aft, colors_aft, zones_aft, verts_aft)
    _style_render_axis(ax_iso, verts_aft, "Lower-aft oblique close-up", elev=-20, azim=-55)

    fin_meta = geometry.metadata.get("control_surfaces", {}).get("geometry", {})
    summary = (
        f"core body width = {float(fin_meta.get('core_body_width_m', width)):.3f} m; "
        f"total span = {float(fin_meta.get('total_span_m', width)):.3f} m; "
        f"per-side fin extension = {float(fin_meta.get('per_side_fin_extension_m', 0.0)):.3f} m"
    )
    fig.suptitle("Astreia-MRV Aft Control-Surface Inspection", fontsize=14, y=0.96)
    fig.text(0.50, 0.04, summary, ha="center", color="#111827", fontsize=9)
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.20)
    plt.close(fig)
