import matplotlib.pyplot as plt
import numpy as np

from astreia_mrv.config import load_design
from astreia_mrv.geometry.lifting_body import generate_lifting_body
from astreia_mrv.viz.plot_geometry import (
    _projection_bounds,
    save_control_surface_inspection,
    save_engineering_sheet,
    save_shaded_render,
    save_three_view,
)


def test_projection_bounds_include_full_mesh_with_padding():
    geom = generate_lifting_body(load_design("configs/mrv3.yaml"))
    verts = geom.mesh.vertices

    for axes in ((0, 1), (0, 2), (1, 2)):
        xmin, xmax, ymin, ymax = _projection_bounds(verts, axes, pad_frac=0.20)
        projected = verts[:, axes]
        assert float(projected[:, 0].min()) > xmin
        assert float(projected[:, 0].max()) < xmax
        assert float(projected[:, 1].min()) > ymin
        assert float(projected[:, 1].max()) < ymax


def test_geometry_plot_exports_are_nonblank(tmp_path):
    geom = generate_lifting_body(load_design("configs/mrv3.yaml"))
    plotters = [
        save_three_view,
        save_engineering_sheet,
        save_shaded_render,
        save_control_surface_inspection,
    ]

    for plotter in plotters:
        path = tmp_path / f"{plotter.__name__}.png"
        plotter(geom, path)
        image = plt.imread(path)
        assert image.shape[0] > 100
        assert image.shape[1] > 100
        assert np.nanstd(image[..., :3]) > 0.01
