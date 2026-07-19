import matplotlib.pyplot as plt
import numpy as np

from dmri.utils.viz import (
    plot_stereographic_contour,
    plot_stereographic_scatter,
)


def test_stereographic_plots_accept_multiple_sample_sets():
    samples = [
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]),
    ]

    scatter_figure, scatter_axis = plot_stereographic_scatter(
        samples, point_color=["red", "blue"], draw_guides=False
    )
    contour_figure, contour_axis = plot_stereographic_contour(
        samples, bins=8, draw_guides=False
    )

    assert len(scatter_axis.collections) == 2
    assert len(contour_axis.collections) > 0
    plt.close(scatter_figure)
    plt.close(contour_figure)
