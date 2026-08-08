import re

import matplotlib.pyplot as plt
import numpy as np
import pytest

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


def test_orthoview_renders_a_scalar_volume():
    """The scalar map viewer: (x, y, z, channels)."""
    from dmri.utils.viz import orthoview

    volume = np.random.default_rng(0).random((6, 7, 5, 2)).astype(np.float32)
    fig = orthoview(volume)

    assert fig.data, "no traces produced"
    assert fig.to_json(), "figure does not serialise"


def test_orthoview_quiver_renders_directions_over_a_volume():
    """The map + fibre-direction viewer: (x, y, z, channels, 3) plus fractions."""
    from dmri.utils.viz import orthoview_quiver

    rng = np.random.default_rng(0)
    vectors = rng.random((6, 7, 5, 2, 3)).astype(np.float32)
    fractions = rng.random((6, 7, 5, 2)).astype(np.float32)
    fig = orthoview_quiver(vectors, fractions)

    assert fig.data
    assert fig.to_json()


def test_orthoview_quiver_rejects_the_wrong_shape():
    """A (x, y, z, channels) volume is not a vector field."""
    from dmri.utils.viz import orthoview_quiver

    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="shape"):
        orthoview_quiver(rng.random((6, 7, 5, 2)), rng.random((6, 7, 5, 2)))


def test_compact_options_shrink_the_output():
    """The downsampling knobs exist to keep notebook HTML small."""
    from dmri.utils.viz import orthoview

    volume = np.random.default_rng(0).random((16, 16, 8, 1)).astype(np.float32)
    full = len(orthoview(volume).to_json())
    compact = len(orthoview(volume, downsample_factor=2, heatmap_quality=0.5).to_json())
    assert compact < full, f"compact output ({compact}b) is not smaller than {full}b"


def _blob(shape=(12, 14, 8)):
    x, y, z = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
    centre = [s / 2 for s in shape]
    return np.exp(
        -(
            (x - centre[0]) ** 2 / 20
            + (y - centre[1]) ** 2 / 20
            + (z - centre[2]) ** 2 / 10
        )
    ).astype(np.float32)


def test_slice_viewer_has_one_frame_per_slice():
    from dmri.utils.viz import slice_viewer

    volume = _blob()
    fig = slice_viewer(volume)

    assert len(fig.frames) == volume.shape[2]
    assert len(fig.layout.sliders[0].steps) == volume.shape[2]
    assert fig.layout.sliders[0].active == volume.shape[2] // 2


def test_slice_viewer_embeds_slices_as_compressed_images():
    """The point of this viewer: images, not JSON float arrays."""
    from dmri.utils.viz import slice_viewer

    fig = slice_viewer(_blob())
    source = fig.frames[0].data[0].source
    assert source.startswith("data:image/png;base64,")
    # No raw numeric grid anywhere in the payload.
    assert '"z":' not in fig.to_json()


def test_slice_viewer_is_much_smaller_than_orthoview():
    from dmri.utils.viz import orthoview, slice_viewer

    volume = _blob((32, 32, 16))
    compact = len(slice_viewer(volume).to_json())
    full = len(orthoview(volume[..., None]).to_json())
    assert compact * 4 < full, f"expected a large saving, got {full} -> {compact}"


def test_slice_viewer_scales_each_volume_independently():
    """A map with a tiny value range must still use the full display range."""
    import base64
    import io

    from PIL import Image

    from dmri.utils.viz import slice_viewer

    blob = _blob()
    fig = slice_viewer({"unit": blob, "tiny": blob * 1e-4})
    assert len(fig.data) == 2
    assert [t.visible for t in fig.data] == [True, False]

    for index in (0, 1):
        payload = fig.frames[4].data[index].source.split(",", 1)[1]
        image = np.array(Image.open(io.BytesIO(base64.b64decode(payload))))
        assert image.max() > 200, "a small-valued map lost its display range"


def test_slice_viewer_rejects_mismatched_volumes():
    from dmri.utils.viz import slice_viewer

    with pytest.raises(ValueError, match="disagree"):
        slice_viewer({"a": _blob((8, 8, 4)), "b": _blob((8, 8, 6))})

    with pytest.raises(ValueError, match="3-D"):
        slice_viewer(np.zeros((4, 4)))


def test_slice_viewer_squeezes_a_trailing_singleton_axis():
    from dmri.utils.viz import slice_viewer

    volume = _blob()
    assert len(slice_viewer(volume[..., None]).frames) == volume.shape[2]


def test_slice_viewer_is_visually_minimal():
    """Chromeless by design: the image is the content, nothing competes with it."""
    from dmri.utils.viz import slice_viewer

    fig = slice_viewer(_blob(), title="demo")
    layout = fig.layout

    assert layout.paper_bgcolor == layout.plot_bgcolor, "two-tone background"
    assert layout.showlegend is False
    assert layout.xaxis.visible is False and layout.yaxis.visible is False
    assert not layout.xaxis.showgrid and not layout.yaxis.showgrid
    assert max(layout.margin.l, layout.margin.r, layout.margin.b) <= 10


def test_slice_viewer_keeps_voxels_square():
    """Without a scale anchor a non-cubic volume renders stretched."""
    from dmri.utils.viz import slice_viewer

    layout = slice_viewer(_blob((10, 40, 6))).layout
    assert layout.yaxis.scaleanchor == "x"
    assert layout.yaxis.scaleratio == 1


def test_slider_does_not_label_every_slice():
    """One tick label per slice turns the track into a smear on a real volume."""
    from dmri.utils.viz import slice_viewer

    volume = _blob((8, 8, 60))
    slider = slice_viewer(volume).layout.sliders[0]

    assert len(slider.steps) == 60
    assert slider.font.size <= 2, "per-tick labels are rendered at readable size"
    assert slider.ticklen == 0
    # The position is still reported, just once.
    assert slider.currentvalue.font.size >= 10
    assert slider.steps[slider.active].label == "31/60"


def test_dropdown_shows_bare_map_names():
    from dmri.utils.viz import slice_viewer

    blob = _blob()
    menu = slice_viewer({"f_sum": blob, "f0": blob}).layout.updatemenus[0]
    assert [b.label for b in menu.buttons] == ["f_sum", "f0"]


def test_direction_colour_follows_the_rgb_convention():
    """x/y/z -> red/green/blue, and an axis is not an arrow."""
    from dmri.utils.viz import direction_colour

    directions = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    np.testing.assert_allclose(direction_colour(directions), np.eye(3))

    # v and -v are the same fibre, so they must get the same colour.
    np.testing.assert_allclose(
        direction_colour(np.array([[-1.0, 0, 0]])),
        direction_colour(np.array([[1.0, 0, 0]])),
    )
    # Unnormalised input is normalised, not clipped.
    np.testing.assert_allclose(direction_colour(np.array([[0, 0, 4.0]])), [[0, 0, 1.0]])


def test_direction_colour_survives_zero_and_nan_vectors():
    """Outside the brain mask dyads are zero; that must not divide by zero."""
    from dmri.utils.viz import direction_colour

    colour = direction_colour(np.array([[0.0, 0, 0], [np.nan, 0, 0]]))
    assert np.all(np.isfinite(colour))
    np.testing.assert_allclose(colour[0], [0, 0, 0])


def test_direction_colour_dims_by_weight():
    from dmri.utils.viz import direction_colour

    directions = np.tile([1.0, 0, 0], (200, 1))
    weight = np.linspace(0.0, 1.0, 200)
    colour = direction_colour(directions, weight)[:, 0]

    # The top of the range is fully bright, and below that brightness is
    # proportional to weight -- check the ratio, so the test does not restate
    # the implementation's choice of reference percentile.
    assert colour[-1] == pytest.approx(1.0)
    assert colour[50] / colour[100] == pytest.approx(weight[50] / weight[100], rel=1e-5)
    assert colour[0] == pytest.approx(0.0)


def test_direction_colour_brightness_survives_an_outlier_voxel():
    """Scaling by the maximum lets one bad voxel black out the volume.

    A single spuriously large fraction is common at the mask edge; against the
    maximum every real voxel would be divided by it and vanish.
    """
    from dmri.utils.viz import direction_colour

    directions = np.tile([1.0, 0, 0], (200, 1))
    weight = np.full(200, 0.5)
    weight[0] = 1000.0

    colour = direction_colour(directions, weight)
    assert colour[1:, 0].min() == pytest.approx(1.0), "an outlier dimmed the volume"


def test_slice_viewer_accepts_colour_volumes():
    from dmri.utils.viz import direction_colour, slice_viewer

    rng = np.random.default_rng(0)
    scalar = rng.random((6, 7, 5))
    colour = direction_colour(rng.normal(size=(6, 7, 5, 3)))
    figure = slice_viewer({"f1": scalar, "dyads1": colour})

    assert len(figure.data) == 2
    assert all(t.source.startswith("data:image") for t in figure.data)
    assert len(figure.frames) == 5


def test_colour_volumes_are_not_rescaled_per_slice():
    """A direction map carries meaning in the hue, so a dim slice must stay dim.

    Scalar volumes are min/max scaled; doing that to colour would repaint every
    slice and make orientations incomparable across the volume.
    """
    from dmri.utils.viz import slice_viewer

    colour = np.zeros((4, 4, 2, 3), dtype=np.float32)
    colour[..., 0, 0] = 1.0  # slice 0: saturated red
    colour[..., 1, 0] = 0.2  # slice 1: dim red
    figure = slice_viewer({"dyads1": colour})

    first, second = (f.data[0].source for f in figure.frames)
    assert first != second, "the two slices were rescaled onto the same values"


def test_slice_viewer_rejects_a_non_colour_trailing_axis():
    from dmri.utils.viz import slice_viewer

    with pytest.raises(ValueError, match="colour volume"):
        slice_viewer({"bad": np.zeros((4, 4, 3, 7))})


def test_saved_page_is_dark_all_the_way_out(tmp_path):
    """The figure was already dark; the page around it was still white."""
    from dmri.utils.viz import _VIEWER_BACKGROUND, save_viewer, slice_viewer

    figure = slice_viewer(np.random.default_rng(0).random((4, 4, 3)))
    destination = save_viewer(figure, tmp_path / "v.html", title="t")
    html = destination.read_text()

    assert html.startswith("<!DOCTYPE html>")
    body = re.search(r"html, body \{(.*?)\}", html, re.S).group(1)
    assert _VIEWER_BACKGROUND in body
    assert "margin: 0" in body
    assert '<meta name="color-scheme" content="dark">' in html
    assert "<title>t</title>" in html


def test_saved_page_fills_the_window_without_resizing_the_caller(tmp_path):
    from dmri.utils.viz import save_viewer, slice_viewer

    figure = slice_viewer(np.random.default_rng(0).random((4, 4, 3)), height=520)
    html = save_viewer(figure, tmp_path / "v.html").read_text()

    assert '"autosize": true' in html or '"autosize":true' in html
    assert '"responsive": true' in html or '"responsive":true' in html
    assert figure.layout.height == 520, "save_viewer mutated the caller's figure"
