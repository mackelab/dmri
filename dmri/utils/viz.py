import base64
import io
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from dipy.data import get_sphere
from matplotlib.patches import Circle
from plotly.subplots import make_subplots

from dmri.utils.dmriutils import cart2sph, sph2cart

sphere_default = get_sphere(name="symmetric724")


def _ensure_axis(
    ax=None,
    *,
    projection: str | None = None,
    figsize: tuple[float, float] | None = None,
    subplot_kw: dict[str, Any] | None = None,
):
    """Return a Matplotlib axis, creating one if needed."""
    if ax is not None:
        return ax.figure, ax

    subplot_kw = dict(subplot_kw or {})
    if projection is not None:
        subplot_kw["projection"] = projection

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, **subplot_kw)
    return fig, ax


def orthoview_quiver(
    data,
    fractions,
    colors=None,
    step=1,
    downsample_factor=1,
    slider_step=10,
    heatmap_quality=1.0,
    simplified_ui=False,
    precision=np.float32,
    colorscale="gray",
    arrow_scale=1.0,
    arrow_width=1,
    height=None,
    width=None,
    show_all_channels=True,
):
    """Orthogonal slice views of a volume with fibre-direction arrows overlaid.

    Produces compact HTML: the downsampling, quality and precision options
    below trade rendering fidelity for file size, which matters in notebooks.

    Args:
        data: 5D numpy array (x, y, z, channels, 3) of vector data
        fractions: Fraction weights for each channel (x, y, z, channels)
        colors: List of colors for vector arrows
        step: Step size for vector arrow sampling
        downsample_factor: Factor to downsample data by
        slider_step: Step size for slice sliders
        heatmap_quality: Factor to reduce heatmap resolution (0.5 = 50% of original)
        simplified_ui: Whether to use simplified UI with fewer controls
        precision: Data precision for internal calculations ('float32', 'float16', or 'int8')
        colorscale: Colorscale for the background heatmap
        arrow_scale: Scale factor for vector arrows
        arrow_width: Width of the vector arrows
        height: Height of the figure in pixels
        width: Width of the figure in pixels
        show_all_channels: Whether to show all channels simultaneously (True) or use dropdown selector (False)

    Returns:
        plotly figure object
    """
    if data.ndim != 5 or data.shape[-1] != 3:
        raise ValueError(
            "Data must have shape (x, y, z, channels, 3) where last dim is vector (u, v, w)"
        )

    x_max, y_max, z_max, channels, _ = data.shape
    fractions = fractions.reshape(x_max, y_max, z_max, channels)

    # Convert data precision if requested
    if precision == np.float16 or precision == np.float32 or precision == np.float64:
        data = data.astype(precision)
        fractions = fractions.astype(precision)
    elif precision == np.int8:
        # For int8, scale vectors to unit length and use int8 for direction only
        # Note: For vectors, int8 is not ideal but can be used for direction
        vectors_norm = np.sqrt(np.sum(data**2, axis=-1, keepdims=True))
        vectors_norm = np.maximum(vectors_norm, 1e-10)  # Avoid division by zero
        data = ((data / vectors_norm) * 127).astype(np.int8)

        # Scale fractions to 0-255 range for int8
        fractions_min = fractions.min()
        fractions_max = fractions.max() + 1e-10
        fractions = (
            (fractions - fractions_min) / (fractions_max - fractions_min) * 255
        ).astype(np.int8)

    # Apply downsampling if requested
    if downsample_factor > 1:
        # Use simple striding for downsampling
        data = data[::downsample_factor, ::downsample_factor, ::downsample_factor, :, :]
        fractions = fractions[
            ::downsample_factor, ::downsample_factor, ::downsample_factor, :
        ]
        x_max, y_max, z_max, channels, _ = data.shape

    # Pad to cube
    max_dim = max(x_max, y_max, z_max)
    x_pad = (max_dim - x_max) // 2
    y_pad = (max_dim - y_max) // 2
    z_pad = (max_dim - z_max) // 2

    data = np.pad(
        data,
        ((x_pad, x_pad), (y_pad, y_pad), (z_pad, z_pad), (0, 0), (0, 0)),
        mode="constant",
    )
    fractions = np.pad(
        fractions,
        ((x_pad, x_pad), (y_pad, y_pad), (z_pad, z_pad), (0, 0)),
        mode="constant",
    )
    x_max, y_max, z_max, _, _ = data.shape

    # Weight vectors by fractions
    data = fractions[..., None] * data
    fsum = np.sum(fractions, axis=-1)

    # Initial slices (middle)
    x_slice = x_max // 2
    y_slice = y_max // 2
    z_slice = z_max // 2

    # Apply heatmap quality reduction if requested
    if heatmap_quality < 1.0:
        # Create downsampled views for the heatmaps
        def downsample_slice(slice_data, quality):
            # Skip if already small enough
            if min(slice_data.shape) < 10:
                return slice_data

            # Calculate new dimensions
            target_height = max(int(slice_data.shape[0] * quality), 10)
            target_width = max(int(slice_data.shape[1] * quality), 10)

            # Use simple striding for downsampling
            stride_y = max(1, slice_data.shape[0] // target_height)
            stride_x = max(1, slice_data.shape[1] // target_width)

            return slice_data[::stride_y, ::stride_x]

    else:
        # No downsampling function
        def downsample_slice(slice_data, quality):
            return slice_data

    # Optimized function to create quiver plots with fewer points
    def create_line_quiver_2d_optimized(X, Y, U, V, color="white", density_factor=1.0):
        # Adjust density based on the quality factor
        if density_factor < 1.0:
            skip = max(1, int(1.0 / density_factor))
            X = X[::skip, ::skip]
            Y = Y[::skip, ::skip]
            U = U[::skip, ::skip]
            V = V[::skip, ::skip]

        # Apply scaling
        U = U * arrow_scale
        V = V * arrow_scale

        # Flatten all arrays
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        U_flat = U.flatten()
        V_flat = V.flatten()

        # Calculate line endpoints
        x0 = X_flat - U_flat
        x1 = X_flat + U_flat
        y0 = Y_flat - V_flat
        y1 = Y_flat + V_flat

        # Create lines with NaN separators
        x_lines = np.vstack([x0, x1, np.full(x0.shape, np.nan)]).T.flatten()
        y_lines = np.vstack([y0, y1, np.full(y0.shape, np.nan)]).T.flatten()

        return go.Scatter(
            x=x_lines,
            y=y_lines,
            mode="lines",
            line=dict(color=color, width=arrow_width),
            showlegend=False,
        )

    if colors is None:
        colors = ["red", "blue", "green"]

    # Create figure
    fig = make_subplots(
        rows=2,
        cols=2,
        column_widths=[0.35, 0.65],
        specs=[[{}, {"rowspan": 2}], [{"colspan": 1}, None]],
        vertical_spacing=0.0,
        horizontal_spacing=0.0,
    )

    # Create reduced list of slider indices with the given step size
    # For simplified UI, use even larger step sizes
    effective_step = slider_step * 2 if simplified_ui else slider_step

    x_indices = list(range(0, x_max, effective_step))
    y_indices = list(range(0, y_max, effective_step))
    z_indices = list(range(0, z_max, effective_step))

    # Ensure the middle slice is included
    if x_slice not in x_indices:
        x_indices = sorted(x_indices + [x_slice])
    if y_slice not in y_indices:
        y_indices = sorted(y_indices + [y_slice])
    if z_slice not in z_indices:
        z_indices = sorted(z_indices + [z_slice])

    # For simplified UI, limit to a max of 5 slices
    if simplified_ui:

        def limit_indices(indices, middle):
            if len(indices) <= 5:
                return indices

            # Keep middle and select 4 more points evenly distributed
            middle_idx = indices.index(middle)
            result = [indices[0]]  # First

            if middle_idx > 0:
                result.append(indices[middle_idx // 2])  # Quarter

            result.append(middle)  # Middle

            if middle_idx < len(indices) - 1:
                result.append(
                    indices[(middle_idx + len(indices)) // 2]
                )  # Three-quarter

            result.append(indices[-1])  # Last

            # Deduplicate and sort
            return sorted(set(result))

        x_indices = limit_indices(x_indices, x_slice)
        y_indices = limit_indices(y_indices, y_slice)
        z_indices = limit_indices(z_indices, z_slice)

    # Keep track of trace indices for slider updates
    xy_plane_traces = {"heatmap": None, "quivers": []}
    xz_plane_traces = {"heatmap": None, "quivers": []}
    yz_plane_traces = {"heatmap": None, "quivers": []}

    # XY plane (axial) - row 1, col 1
    xy_heatmap = go.Heatmap(
        z=downsample_slice(fsum[x_slice, :, :].T, heatmap_quality),
        colorscale=colorscale,
        zmin=0,
        zmax=np.max(fsum) if np.max(fsum) > 0 else 1,
        showscale=False,
    )
    fig.add_trace(xy_heatmap, row=1, col=1)
    xy_plane_traces["heatmap"] = 0  # First trace added

    # XZ plane (coronal) - row 2, col 1
    xz_heatmap = go.Heatmap(
        z=downsample_slice(fsum[:, y_slice, :].T, heatmap_quality),
        colorscale=colorscale,
        zmin=0,
        zmax=np.max(fsum) if np.max(fsum) > 0 else 1,
        showscale=False,
    )
    fig.add_trace(xz_heatmap, row=2, col=1)
    xz_plane_traces["heatmap"] = 1  # Second trace added

    # YZ plane (sagittal) - row 1, col 2
    yz_heatmap = go.Heatmap(
        z=downsample_slice(fsum[:, :, z_slice], heatmap_quality),
        colorscale=colorscale,
        zmin=0,
        zmax=np.max(fsum) if np.max(fsum) > 0 else 1,
        showscale=False,
    )
    fig.add_trace(yz_heatmap, row=1, col=2)
    yz_plane_traces["heatmap"] = 2  # Third trace added

    # Current trace index counter
    trace_idx = 3

    # Add quiver traces for each channel
    for c in range(channels):
        # XY plane quivers
        U = data[x_slice, :, :, c, 0]
        V = data[x_slice, :, :, c, 1]
        Y, Z = np.mgrid[0:y_max:step, 0:z_max:step]
        quiver_trace = create_line_quiver_2d_optimized(
            Y,
            Z,
            U[::step, ::step],
            V[::step, ::step],
            colors[c % len(colors)],
            heatmap_quality,
        )
        fig.add_trace(quiver_trace, row=1, col=1)
        xy_plane_traces["quivers"].append(trace_idx)
        trace_idx += 1

        # XZ plane quivers
        U = data[:, y_slice, :, c, 0]
        V = data[:, y_slice, :, c, 2]
        X, Z = np.mgrid[0:x_max:step, 0:z_max:step]
        quiver_trace = create_line_quiver_2d_optimized(
            X,
            Z,
            U[::step, ::step],
            V[::step, ::step],
            colors[c % len(colors)],
            heatmap_quality,
        )
        fig.add_trace(quiver_trace, row=2, col=1)
        xz_plane_traces["quivers"].append(trace_idx)
        trace_idx += 1

        # YZ plane quivers
        U = data[:, :, z_slice, c, 0]
        V = data[:, :, z_slice, c, 1]
        X, Y_ = np.mgrid[0:x_max:step, 0:y_max:step]
        quiver_trace = create_line_quiver_2d_optimized(
            X,
            Y_,
            U[::step, ::step].T,
            V[::step, ::step].T,
            colors[c % len(colors)],
            heatmap_quality,
        )
        fig.add_trace(quiver_trace, row=1, col=2)
        yz_plane_traces["quivers"].append(trace_idx)
        trace_idx += 1

    # Initialize default visibility state
    for i in range(trace_idx):
        # Heatmaps are always visible
        if i in [
            xy_plane_traces["heatmap"],
            xz_plane_traces["heatmap"],
            yz_plane_traces["heatmap"],
        ]:
            fig.data[i].visible = True
        # Show all quivers or just first channel depending on settings
        elif show_all_channels:
            fig.data[i].visible = True
        # Otherwise, only show first channel's quivers
        elif i in [
            xy_plane_traces["quivers"][0],
            xz_plane_traces["quivers"][0],
            yz_plane_traces["quivers"][0],
        ]:
            fig.data[i].visible = True
        else:
            fig.data[i].visible = False

    # Create sliders with a completely different approach
    sliders = []

    # X slider
    x_steps = []
    for x_idx in x_indices:
        # Create a single update that includes all traces
        step_data = {}
        step_indices = []

        # Add heatmap update
        step_indices.append(xy_plane_traces["heatmap"])
        step_data["z"] = [downsample_slice(fsum[x_idx, :, :].T, heatmap_quality)]

        # Add quiver updates (separately for each channel)
        for c in range(channels):
            if show_all_channels or c == 0:
                U = data[x_idx, :, :, c, 0]
                V = data[x_idx, :, :, c, 1]
                Y, Z = np.mgrid[0:y_max:step, 0:z_max:step]
                quiver = create_line_quiver_2d_optimized(
                    Y,
                    Z,
                    U[::step, ::step],
                    V[::step, ::step],
                    colors[c % len(colors)],
                    heatmap_quality,
                )

                # Use index-based keys for multiple trace updates
                quiver_idx = xy_plane_traces["quivers"][c]
                step_indices.append(quiver_idx)

                # Each trace property needs a corresponding entry in the data array
                if "x" not in step_data:
                    step_data["x"] = []
                if "y" not in step_data:
                    step_data["y"] = []

                # Add None for heatmap (doesn't need x/y update)
                if len(step_data["x"]) < len(step_indices) - 1:
                    step_data["x"].append(None)
                    step_data["y"].append(None)

                # Add quiver data
                step_data["x"].append(quiver.x)
                step_data["y"].append(quiver.y)

        # Create a single step for this slice
        x_steps.append({
            "method": "restyle",
            "label": str(x_idx),
            "args": [step_data, step_indices],
        })

    sliders.append({
        "active": x_indices.index(x_slice) if x_slice in x_indices else 0,
        "currentvalue": {"prefix": "X: " if simplified_ui else "X Slice: "},
        "steps": x_steps,
        "x": 0.05,
        "y": 0.0,
        "len": 0.25,
        "pad": {"t": 50},
    })

    # Y slider
    y_steps = []
    for y_idx in y_indices:
        # Create a single update that includes all traces
        step_data = {}
        step_indices = []

        # Add heatmap update
        step_indices.append(xz_plane_traces["heatmap"])
        step_data["z"] = [downsample_slice(fsum[:, y_idx, :].T, heatmap_quality)]

        # Add quiver updates (separately for each channel)
        for c in range(channels):
            if show_all_channels or c == 0:
                U = data[:, y_idx, :, c, 0]
                V = data[:, y_idx, :, c, 2]
                X, Z = np.mgrid[0:x_max:step, 0:z_max:step]
                quiver = create_line_quiver_2d_optimized(
                    X,
                    Z,
                    U[::step, ::step],
                    V[::step, ::step],
                    colors[c % len(colors)],
                    heatmap_quality,
                )

                # Use index-based keys for multiple trace updates
                quiver_idx = xz_plane_traces["quivers"][c]
                step_indices.append(quiver_idx)

                # Each trace property needs a corresponding entry in the data array
                if "x" not in step_data:
                    step_data["x"] = []
                if "y" not in step_data:
                    step_data["y"] = []

                # Add None for heatmap (doesn't need x/y update)
                if len(step_data["x"]) < len(step_indices) - 1:
                    step_data["x"].append(None)
                    step_data["y"].append(None)

                # Add quiver data
                step_data["x"].append(quiver.x)
                step_data["y"].append(quiver.y)

        # Create a single step for this slice
        y_steps.append({
            "method": "restyle",
            "label": str(y_idx),
            "args": [step_data, step_indices],
        })

    sliders.append({
        "active": y_indices.index(y_slice) if y_slice in y_indices else 0,
        "currentvalue": {"prefix": "Y: " if simplified_ui else "Y Slice: "},
        "steps": y_steps,
        "x": 0.35,
        "y": 0.0,
        "len": 0.25,
        "pad": {"t": 50},
    })

    # Z slider
    z_steps = []
    for z_idx in z_indices:
        # Create a single update that includes all traces
        step_data = {}
        step_indices = []

        # Add heatmap update
        step_indices.append(yz_plane_traces["heatmap"])
        step_data["z"] = [downsample_slice(fsum[:, :, z_idx], heatmap_quality)]

        # Add quiver updates (separately for each channel)
        for c in range(channels):
            if show_all_channels or c == 0:
                U = data[:, :, z_idx, c, 0]
                V = data[:, :, z_idx, c, 1]
                X, Y_ = np.mgrid[0:x_max:step, 0:y_max:step]
                quiver = create_line_quiver_2d_optimized(
                    X,
                    Y_,
                    U[::step, ::step].T,
                    V[::step, ::step].T,
                    colors[c % len(colors)],
                    heatmap_quality,
                )

                # Use index-based keys for multiple trace updates
                quiver_idx = yz_plane_traces["quivers"][c]
                step_indices.append(quiver_idx)

                # Each trace property needs a corresponding entry in the data array
                if "x" not in step_data:
                    step_data["x"] = []
                if "y" not in step_data:
                    step_data["y"] = []

                # Add None for heatmap (doesn't need x/y update)
                if len(step_data["x"]) < len(step_indices) - 1:
                    step_data["x"].append(None)
                    step_data["y"].append(None)

                # Add quiver data
                step_data["x"].append(quiver.x)
                step_data["y"].append(quiver.y)

        # Create a single step for this slice
        z_steps.append({
            "method": "restyle",
            "label": str(z_idx),
            "args": [step_data, step_indices],
        })

    sliders.append({
        "active": z_indices.index(z_slice) if z_slice in z_indices else 0,
        "currentvalue": {"prefix": "Z: " if simplified_ui else "Z Slice: "},
        "steps": z_steps,
        "x": 0.65,
        "y": 0.0,
        "len": 0.25,
        "pad": {"t": 50},
    })

    # Channel dropdown - simplified if requested
    updatemenus = []
    if channels > 1 and not show_all_channels:
        buttons = []
        for c in range(channels):
            visibility = []

            # Set visibility for all traces
            for i in range(trace_idx):
                # Heatmaps are always visible
                if i in [
                    xy_plane_traces["heatmap"],
                    xz_plane_traces["heatmap"],
                    yz_plane_traces["heatmap"],
                ]:
                    visibility.append(True)
                # Only show current channel's quivers
                elif i in [
                    xy_plane_traces["quivers"][c],
                    xz_plane_traces["quivers"][c],
                    yz_plane_traces["quivers"][c],
                ]:
                    visibility.append(True)
                # Hide other channel's quivers
                else:
                    visibility.append(False)

            buttons.append({
                "method": "update",
                "label": str(c),
                "args": [{"visible": visibility}],
            })

        updatemenus.append({
            "buttons": buttons,
            "direction": "right" if simplified_ui else "up",
            "showactive": True,
            "active": 0,
            "x": 0.85,
            "y": 0.05,
            "xanchor": "left",
            "yanchor": "bottom",
        })

    # Update layout with all sliders and dropdown
    fig.update_layout(
        sliders=sliders,
        updatemenus=updatemenus,
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=10, b=50),
        height=height,
        width=width,
    )

    # Turn off axes
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    return fig


def plot_spherical_distribution_polar(
    distribution,
    n_samples: int = 1000,
    ax=None,
    color=None,
    levels: int = 3,
    figsize: tuple[float, float] | None = (6, 4),
    cmap: str = "viridis",
    filled: bool = False,
    contour_kwargs: dict[str, Any] | None = None,
    show_axis_labels: bool = True,
    title: str | None = "Spherical Distribution PDF",
):
    """Plot spherical distribution in polar coordinates.

    Args:
        distribution: Spherical distribution object with pdf and sample methods
        n_samples: Number of samples to generate
        ax: Matplotlib axis to plot on
        color: Backwards compatible alias for cmap
        levels: Number of contour levels
        figsize: Figure size when creating a new axis
        cmap: Matplotlib colormap name
        filled: Use filled contours instead of lines
        contour_kwargs: Additional kwargs forwarded to `tricontour`/`tricontourf`
        show_axis_labels: Whether to draw axis labels and ticks
        title: Title text (set to None to skip)

    Returns:
        Tuple of (Figure, Axes)
    """
    samples = distribution.sample(jax.random.key(0), (n_samples,))
    grid_y = np.linspace(0, np.pi, 100)
    grid_x = np.linspace(-np.pi, np.pi, 200)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    samples_rand = np.stack([grid_y.flatten(), grid_x.flatten()], axis=-1)
    samples_cart = jax.vmap(sph2cart)(samples_rand)
    samples = jnp.concatenate([samples, samples_cart], axis=0)
    pdf = distribution.pdf(samples)
    mu = jax.vmap(cart2sph)(samples)
    pdf = pdf / jnp.max(pdf)

    existing_ax = ax
    fig, ax = _ensure_axis(ax, figsize=figsize)
    contour_kwargs = dict(contour_kwargs or {})

    cmap_name = color or cmap
    contour_fn = ax.tricontourf if filled else ax.tricontour
    contour_fn(
        mu[:, 1],
        mu[:, 0],
        pdf,
        cmap=cmap_name,
        levels=levels,
        **contour_kwargs,
    )

    ax.set_ylim(0, np.pi)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_aspect("equal")

    if title is not None:
        ax.set_title(title)

    if show_axis_labels:
        ax.set_xlabel("Azimuthal Angle (phi)")
        ax.set_ylabel("Polar Angle (theta)")
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    if existing_ax is None:
        fig.tight_layout()

    return fig, ax


def plot_spherical_distribution_cartesian(
    distribution,
    n_samples: int = 1000,
    sphere=None,
    ax=None,
    figsize: tuple[float, float] | None = (6, 6),
    cmap: str = "viridis",
    vertex_kwargs: dict[str, Any] | None = None,
    show_samples: bool = True,
    sample_kwargs: dict[str, Any] | None = None,
    add_colorbar: bool = True,
    colorbar_kwargs: dict[str, Any] | None = None,
    title: str | None = "Spherical Distribution PDF",
):
    """Plot spherical distribution in Cartesian coordinates.

    Args:
        distribution: Spherical distribution object with pdf and sample methods
        n_samples: Number of samples to generate
        sphere: Sphere object for vertices
        ax: Matplotlib axis to plot on
        figsize: Figure size for new axes
        cmap: Colormap for pdf-colored vertices
        vertex_kwargs: Extra kwargs forwarded to the pdf scatter
        show_samples: Whether to overlay Monte Carlo samples
        sample_kwargs: Extra kwargs forwarded to the samples scatter
        add_colorbar: Whether to draw a colorbar for the pdf values
        colorbar_kwargs: Extra kwargs forwarded to `fig.colorbar`
        title: Title text (None to skip)

    Returns:
        Tuple of (Figure, Axes)
    """
    sphere = sphere_default if sphere is None else sphere
    existing_ax = ax
    fig, ax = _ensure_axis(ax, projection="3d", figsize=figsize)

    pdfs = distribution.pdf(sphere.vertices)
    samples = None
    if show_samples:
        samples = distribution.sample(jax.random.key(0), (n_samples,))

    vertex_kwargs = dict(vertex_kwargs or {})
    vertex_kwargs.setdefault("cmap", cmap)
    vertex_kwargs.setdefault("s", 10)
    vertex_kwargs.setdefault("alpha", 0.9)
    vertex_kwargs.setdefault("c", pdfs)
    sc = ax.scatter(
        sphere.vertices[:, 0],
        sphere.vertices[:, 1],
        sphere.vertices[:, 2],
        **vertex_kwargs,
    )

    if show_samples and samples is not None:
        sample_kwargs = dict(sample_kwargs or {})
        sample_kwargs.setdefault("color", "red")
        sample_kwargs.setdefault("s", 10)
        sample_kwargs.setdefault("alpha", 0.1)
        sample_kwargs.setdefault("label", "Samples")
        ax.scatter(
            samples[:, 0],
            samples[:, 1],
            samples[:, 2],
            **sample_kwargs,
        )

    if add_colorbar:
        colorbar_kwargs = dict(colorbar_kwargs or {})
        colorbar_kwargs.setdefault("label", "PDF value")
        colorbar_kwargs.setdefault("shrink", 0.6)
        fig.colorbar(sc, ax=ax, **colorbar_kwargs)

    if title is not None:
        ax.set_title(title)

    if existing_ax is None:
        fig.tight_layout()

    return fig, ax


def plot_spherical_distribution_fod(
    distribution,
    ax=None,
    alpha: float = 0.8,
    figsize: tuple[float, float] | None = (5, 5),
    cmap: str | None = "viridis",
    surface_kwargs: dict[str, Any] | None = None,
    hide_axes: bool = True,
):
    """Plot spherical distribution as a fiber orientation distribution (FOD).

    Args:
        distribution: Spherical distribution object with pdf and sample methods
        ax: Matplotlib axis to plot on
        alpha: Transparency of the surface
        figsize: Figure size when creating a new axis
        cmap: Colormap applied to the pdf evaluated on the sphere (set to None for default Matplotlib coloring)
        surface_kwargs: Additional kwargs forwarded to `plot_surface`
        hide_axes: Remove axis spines/ticks when True

    Returns:
        Tuple of (Figure, Axes)
    """
    # Create a grid of points on a sphere
    samples = distribution.sample(jax.random.key(0), (10,))
    u_samples = jax.vmap(cart2sph)(samples)

    u = np.linspace(0, 2 * np.pi, 200)
    u = np.concatenate([u, u_samples[:, 1]])
    v = np.linspace(0, np.pi, 200)
    v = np.concatenate([v, u_samples[:, 0]])
    u = np.sort(u)
    v = np.sort(v)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    # Reshape for evaluation
    points = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T

    # Evaluate PDF at sphere points
    pdf_values = distribution.pdf(points)
    radius = pdf_values.reshape(x.shape)

    # Scale the sphere by the pdf values
    x_surf = x * radius
    y_surf = y * radius
    z_surf = z * radius

    existing_ax = ax
    fig, ax = _ensure_axis(ax, projection="3d", figsize=figsize)
    surface_kwargs = dict(surface_kwargs or {})

    facecolors = None
    if cmap is not None:
        norm = (pdf_values - pdf_values.min()) / (np.ptp(pdf_values) + 1e-12)
        cmap_obj = plt.get_cmap(cmap)
        facecolors = cmap_obj(norm.reshape(x.shape))
        surface_kwargs.setdefault("facecolors", facecolors)

    ax.plot_surface(
        x_surf,
        y_surf,
        z_surf,
        alpha=alpha,
        **surface_kwargs,
    )

    # Plot the maxima of the PDF as stick
    # dir_max = points[np.argmax(pdf_values)]

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    if hide_axes:
        ax.axis("off")

    if existing_ax is None:
        fig.tight_layout()

    return fig, ax


def orthoview(
    data,
    vmin=None,
    vmax=None,
    channel_names=None,
    color_map="gray",
    downsample_factor=1,
    slider_step=10,
    heatmap_quality=1.0,
    simplified_ui=False,
    precision=np.float32,
    height=None,
    width=None,
):
    """Orthogonal slice views of a scalar volume.

    Produces compact HTML: the downsampling, quality and precision options
    below trade rendering fidelity for file size, which matters in notebooks.

    Args:
        data: 3D or 4D numpy array (x, y, z, [channels])
        vmin: Minimum value for colorscale
        vmax: Maximum value for colorscale
        channel_names: Names for channels in dropdown
        downsample_factor: Factor to downsample data by
        slider_step: Step size for slice sliders
        heatmap_quality: Factor to reduce heatmap resolution (0.5 = 50% of original)
        simplified_ui: Whether to use simplified UI with fewer controls
        precision: Data precision ('float32', 'float16', or 'int8')

    Returns:
        plotly figure object
    """
    # Check for 4D data (x, y, z, channels)
    if data.ndim == 3:
        data = data[..., np.newaxis]  # Add channel dimension if missing

    # Convert data precision if requested
    if precision == np.float16 or precision == np.float32 or precision == np.float64:
        # Compute quantiles before converting to avoid float16 overflow
        if vmin is None:
            vmin = np.quantile(data, 0.01)
            vmin = vmin.astype(precision)
        if vmax is None:
            vmax = np.quantile(data, 0.99)
            vmax = vmax.astype(precision)
        data = data.astype(precision)
    elif precision == np.int8:
        # Scale to 0-255 range for int8
        data_min = data.min()
        data_max = data.max() + 1e-2
        data = ((data - data_min) / (data_max - data_min) * 255).astype(np.int8)
        vmin = np.quantile(data, 0.01)
        vmax = np.quantile(data, 0.99)
    # Apply downsampling if requested
    if downsample_factor > 1:
        # Use simple striding for downsampling
        data = data[::downsample_factor, ::downsample_factor, ::downsample_factor, :]
        if vmin is None:
            vmin = np.quantile(data, 0.01)
        if vmax is None:
            vmax = np.quantile(data, 0.99)

    x_max, y_max, z_max, channels = data.shape

    # Pad to cube
    max_dim = max(x_max, y_max, z_max)
    x_pad = (max_dim - x_max) // 2
    y_pad = (max_dim - y_max) // 2
    z_pad = (max_dim - z_max) // 2

    data = np.pad(
        data, ((x_pad, x_pad), (y_pad, y_pad), (z_pad, z_pad), (0, 0)), mode="constant"
    )
    x_max, y_max, z_max, _ = data.shape

    # Initial slices (middle)
    x_slice = x_max // 2
    y_slice = y_max // 2
    z_slice = z_max // 2

    # Apply heatmap quality reduction if requested
    if heatmap_quality < 1.0:
        # Create downsampled views for the heatmaps
        def downsample_slice(slice_data, quality):
            # Skip if already small enough
            if min(slice_data.shape) < 10:
                return slice_data

            # Calculate new dimensions
            target_height = max(int(slice_data.shape[0] * quality), 10)
            target_width = max(int(slice_data.shape[1] * quality), 10)

            # Use simple striding for downsampling
            stride_y = max(1, slice_data.shape[0] // target_height)
            stride_x = max(1, slice_data.shape[1] // target_width)

            return slice_data[::stride_y, ::stride_x]

    else:
        # No downsampling function
        def downsample_slice(slice_data, quality):
            return slice_data

    # Create figure with existing layout
    fig = make_subplots(
        rows=2,
        cols=2,
        column_widths=[0.35, 0.65],
        specs=[[{}, {"rowspan": 2}], [{"colspan": 1}, None]],
        vertical_spacing=0.0,
        horizontal_spacing=0.0,
    )

    # Create traces for all channels (always 3 per channel)
    traces = []
    for c in range(channels):
        # Apply quality reduction to heatmaps
        xy_data = downsample_slice(data[x_slice, :, :, c].T, heatmap_quality)
        xz_data = downsample_slice(data[:, y_slice, :, c].T, heatmap_quality)
        yz_data = downsample_slice(data[:, :, z_slice, c], heatmap_quality)

        traces.append(
            go.Heatmap(
                z=xy_data, colorscale=color_map, zmin=vmin, zmax=vmax, showscale=False
            )
        )
        traces.append(
            go.Heatmap(
                z=xz_data, colorscale=color_map, zmin=vmin, zmax=vmax, showscale=False
            )
        )
        traces.append(
            go.Heatmap(
                z=yz_data, colorscale=color_map, zmin=vmin, zmax=vmax, showscale=False
            )
        )

    # Add all traces to figure
    for i, trace in enumerate(traces):
        if i % 3 == 0:
            fig.add_trace(trace, row=1, col=1)
        elif i % 3 == 1:
            fig.add_trace(trace, row=2, col=1)
        else:
            fig.add_trace(trace, row=1, col=2)

    # Initialize visibility (only show channel 0)
    initial_visibility = []
    for c in range(channels):
        initial_visibility.extend([c == 0] * 3)

    fig.update_traces(visible=False)
    for i, v in enumerate(initial_visibility):
        fig.data[i].visible = v

    # Create reduced list of slider indices with the given step size
    # For simplified UI, use even larger step sizes
    effective_step = slider_step * 2 if simplified_ui else slider_step

    x_indices = list(range(0, x_max, effective_step))
    y_indices = list(range(0, y_max, effective_step))
    z_indices = list(range(0, z_max, effective_step))

    # Ensure the middle slice is included
    if x_slice not in x_indices:
        x_indices = sorted(x_indices + [x_slice])
    if y_slice not in y_indices:
        y_indices = sorted(y_indices + [y_slice])
    if z_slice not in z_indices:
        z_indices = sorted(z_indices + [z_slice])

    # For simplified UI, limit to a max of 5 slices
    if simplified_ui:

        def limit_indices(indices, middle):
            if len(indices) <= 5:
                return indices

            # Keep middle and select 4 more points evenly distributed
            middle_idx = indices.index(middle)
            result = [indices[0]]  # First

            if middle_idx > 0:
                result.append(indices[middle_idx // 2])  # Quarter

            result.append(middle)  # Middle

            if middle_idx < len(indices) - 1:
                result.append(
                    indices[(middle_idx + len(indices)) // 2]
                )  # Three-quarter

            result.append(indices[-1])  # Last

            # Deduplicate and sort
            return sorted(set(result))

        x_indices = limit_indices(x_indices, x_slice)
        y_indices = limit_indices(y_indices, y_slice)
        z_indices = limit_indices(z_indices, z_slice)

    # Sliders with reduced number of steps
    sliders = [
        dict(
            active=x_indices.index(x_slice) if x_slice in x_indices else 0,
            currentvalue={
                "prefix": "X: " if simplified_ui else "X Slice: ",
                "suffix": "",
            },
            steps=[
                dict(
                    method="restyle",
                    label=str(i),
                    args=[
                        {
                            "z": [
                                downsample_slice(data[i, :, :, c].T, heatmap_quality)
                                for c in range(channels)
                            ]
                        },
                        list(range(0, 3 * channels, 3)),
                    ],
                )
                for i in x_indices
            ],
            x=0.0,
            y=0.13,
            len=0.25,
        ),
        dict(
            active=y_indices.index(y_slice) if y_slice in y_indices else 0,
            currentvalue={
                "prefix": "Y: " if simplified_ui else "Y Slice: ",
                "suffix": "",
            },
            steps=[
                dict(
                    method="restyle",
                    label=str(i),
                    args=[
                        {
                            "z": [
                                downsample_slice(data[:, i, :, c].T, heatmap_quality)
                                for c in range(channels)
                            ]
                        },
                        list(range(1, 3 * channels, 3)),
                    ],
                )
                for i in y_indices
            ],
            x=0.3,
            y=0.13,
            len=0.25,
        ),
        dict(
            active=z_indices.index(z_slice) if z_slice in z_indices else 0,
            currentvalue={
                "prefix": "Z: " if simplified_ui else "Z Slice: ",
                "suffix": "",
            },
            steps=[
                dict(
                    method="restyle",
                    label=str(i),
                    args=[
                        {
                            "z": [
                                downsample_slice(data[:, :, i, c], heatmap_quality)
                                for c in range(channels)
                            ]
                        },
                        list(range(2, 3 * channels, 3)),
                    ],
                )
                for i in z_indices
            ],
            x=0.6,
            y=0.13,
            len=0.25,
        ),
    ]

    # Channel dropdown - simplified if requested
    updatemenus = []
    if channels > 1:
        # For simplified UI, use simpler button layout
        channel_dropdown = dict(
            buttons=[
                dict(
                    label=str(c) if channel_names is None else channel_names[c],
                    method="update",
                    args=[{"visible": [(i // 3) == c for i in range(3 * channels)]}],
                )
                for c in range(channels)
            ],
            direction="up" if not simplified_ui else "right",
            showactive=True,
            active=0,
            x=0.875,
            y=0.05,
            xanchor="left",
            yanchor="bottom",
            font=dict(color="white"),
        )
        updatemenus.append(channel_dropdown)

    # Update layout with all sliders and dropdown
    fig.update_layout(
        sliders=sliders,
        updatemenus=updatemenus,
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=10, b=50),
        height=height,
        width=width,
    )

    # Turn off axes
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    return fig


# ---------------------- helpers ----------------------


# --------------------- plotting ----------------------


def _norm_rows(X, eps=1e-12):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)


def _stereo_project(X):
    x, y, z = X.T
    d = np.clip(1 + z, 1e-9, None)
    return np.c_[x / d, y / d]


def _as_sample_sets(samples):
    if isinstance(samples, (list, tuple)):
        return [np.asarray(sample_set, dtype=float) for sample_set in samples]
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 3 and samples.shape[-1] == 3:
        return list(samples)
    return [samples]


def _broadcast_plot_param(value, count, name):
    if isinstance(value, (list, tuple)) and not isinstance(value, str):
        if len(value) == count:
            return list(value)
        if len(value) == 1:
            return [value[0]] * count
        raise ValueError(f"{name} must have length {count} or be a scalar.")
    return [value] * count


def _draw_stereographic_guides(ax, ring_radii, angle_step, frame_color, draw_guides):
    ax.add_artist(Circle((0, 0), 1.0, fill=False, lw=1.8, zorder=2, color=frame_color))
    if not draw_guides:
        return
    for radius in ring_radii:
        ax.add_artist(
            Circle(
                (0, 0),
                radius,
                fill=False,
                lw=0.8,
                ls="--",
                alpha=0.6,
                zorder=1,
                color=frame_color,
            )
        )
    for degrees in range(0, 360, angle_step):
        angle = np.deg2rad(degrees)
        ax.plot(
            [0, np.cos(angle)],
            [0, np.sin(angle)],
            lw=0.6,
            ls="--",
            alpha=0.6,
            zorder=1,
            color=frame_color,
        )


def _prepare_stereographic_samples(samples, axial):
    samples = _norm_rows(np.asarray(samples, dtype=float))
    if axial:
        samples = np.where(samples[:, 2:3] < 0, -samples, samples)
    projected = _stereo_project(samples)
    inside = np.sum(projected**2, axis=1) <= 1.0 + 1e-9
    return samples[inside], projected[inside], inside


def _finish_stereographic_axis(fig, ax, created_axis):
    clip_circle = Circle((0, 0), 1.0, transform=ax.transData)
    for collection in ax.collections:
        collection.set_clip_path(clip_circle)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_frame_on(False)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if created_axis:
        fig.tight_layout()


def plot_stereographic_scatter(
    V,
    axial=True,
    angle_step=30,
    ring_radii=(0.25, 0.5, 0.75, 1.0),
    s=14,
    point_alpha=0.9,
    point_color="orientation",
    edge=True,
    ax=None,
    figsize: tuple[float, float] | None = (5, 5),
    facecolor: str | None = "black",
    frame_color: str = "white",
    draw_guides: bool = True,
):
    """
    Scatter-only stereographic plot with better visibility.
    Args:
        point_color: "orientation" maps |x|,|y|,|z| to RGB, otherwise any Matplotlib color
        facecolor: Background color for the stereographic disk (None to keep default)
        frame_color: Color used for boundary, guides and annotations

    Returns:
        Tuple of (Figure, Axes)
    """
    sample_sets = _as_sample_sets(V)
    point_colors = _broadcast_plot_param(point_color, len(sample_sets), "point_color")
    sizes = _broadcast_plot_param(s, len(sample_sets), "s")
    alphas = _broadcast_plot_param(point_alpha, len(sample_sets), "point_alpha")
    edges = _broadcast_plot_param(edge, len(sample_sets), "edge")

    created_axis = ax is None
    fig, ax = _ensure_axis(ax, figsize=figsize)
    if facecolor is not None:
        ax.set_facecolor(facecolor)
    _draw_stereographic_guides(ax, ring_radii, angle_step, frame_color, draw_guides)

    for index, sample_set in enumerate(sample_sets):
        if sample_set.size == 0:
            continue
        samples, projected, _ = _prepare_stereographic_samples(sample_set, axial)
        if projected.size == 0:
            continue
        color = point_colors[index]
        if isinstance(color, str) and color == "orientation":
            color = np.abs(samples)
            color /= np.linalg.norm(color, axis=1, keepdims=True) + 1e-12
        ax.scatter(
            projected[:, 0],
            projected[:, 1],
            s=sizes[index],
            c=color,
            alpha=alphas[index],
            linewidths=0.3 if edges[index] else 0,
            edgecolors=frame_color if edges[index] else None,
            zorder=5,
        )
    _finish_stereographic_axis(fig, ax, created_axis)
    return fig, ax


def plot_stereographic_contour(
    V,
    axial=True,
    angle_step=30,
    ring_radii=(0.25, 0.5, 0.75, 1.0),
    bins=128,
    levels=6,
    filled=True,
    cmap="viridis",
    weights=None,
    alpha=0.85,
    ax=None,
    figsize: tuple[float, float] | None = (5, 5),
    facecolor: str | None = "black",
    frame_color: str = "white",
    draw_guides: bool = True,
    normalize: bool = True,
):
    """Plot stereographic density contours for one or more orientation samples."""
    sample_sets = _as_sample_sets(V)
    bins_per_set = _broadcast_plot_param(bins, len(sample_sets), "bins")
    cmaps = _broadcast_plot_param(cmap, len(sample_sets), "cmap")
    alphas = _broadcast_plot_param(alpha, len(sample_sets), "alpha")
    weight_sets = _broadcast_plot_param(weights, len(sample_sets), "weights")

    created_axis = ax is None
    fig, ax = _ensure_axis(ax, figsize=figsize)
    if facecolor is not None:
        ax.set_facecolor(facecolor)
    _draw_stereographic_guides(ax, ring_radii, angle_step, frame_color, draw_guides)

    for index, sample_set in enumerate(sample_sets):
        if sample_set.size == 0:
            continue
        _, projected, inside = _prepare_stereographic_samples(sample_set, axial)
        if projected.size == 0:
            continue
        sample_weights = weight_sets[index]
        if sample_weights is not None:
            sample_weights = np.asarray(sample_weights).reshape(-1)
            if sample_weights.shape[0] != sample_set.shape[0]:
                raise ValueError("weights must match the number of samples.")
            sample_weights = sample_weights[inside]
        histogram, x_edges, y_edges = np.histogram2d(
            projected[:, 0],
            projected[:, 1],
            bins=bins_per_set[index],
            range=[[-1, 1], [-1, 1]],
            weights=sample_weights,
        )
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        X, Y = np.meshgrid(x_centers, y_centers, indexing="xy")
        density = histogram.T
        if normalize and np.max(density) > 0:
            density = density / np.max(density)
        density = np.where(X**2 + Y**2 <= 1.0 + 1e-9, density, np.nan)
        contour = ax.contourf if filled else ax.contour
        contour(
            X,
            Y,
            density,
            levels=levels,
            cmap=cmaps[index],
            alpha=alphas[index],
        )

    _finish_stereographic_axis(fig, ax, created_axis)
    return fig, ax


#: A restrained, chromeless palette: the image is the content, everything else
#: should recede.
_VIEWER_BACKGROUND = "#111111"
_VIEWER_FOREGROUND = "#d0d0d0"
_VIEWER_TRACK = "#2a2a2a"
_VIEWER_FONT = "ui-sans-serif, -apple-system, Segoe UI, Helvetica, Arial, sans-serif"


def _to_png_uri(plane: np.ndarray, vmin: float, vmax: float) -> str:
    """One 2-D slice as a base64 PNG data URI.

    Quantising to uint8 and letting PNG compress it is what makes this viewer
    small: plotly serialises numeric arrays as JSON floats at ~12 bytes per
    voxel, against well under one byte here.
    """
    from PIL import Image

    span = (vmax - vmin) or 1.0
    scaled = (np.nan_to_num(plane, nan=vmin) - vmin) / span
    quantised = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(np.flipud(quantised.T)).save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def slice_viewer(
    volumes,
    axis: int = 2,
    title: str | None = None,
    height: int = 520,
    width: int | None = None,
):
    """Browse volumes slice by slice, as a small self-contained figure.

    An alternative to :func:`orthoview` for looking at output maps. Each slice
    is embedded once as a compressed image rather than as a JSON float array,
    which is roughly 16x smaller for a typical volume; a slider steps through
    slices and, given several volumes, a dropdown switches between them.

    The trade-off is quantisation: slices are mapped to 256 grey levels for
    display, so this is a viewer, not a way to read exact voxel values. Each
    volume is scaled by its own min/max, reported in the dropdown label.

    Args:
        volumes: a 3-D array, or a mapping of name -> 3-D array. Arrays with a
            trailing singleton axis are accepted and squeezed.
        axis: axis to slice along (0=sagittal, 1=coronal, 2=axial).
        title: figure title.
        height: figure height in pixels.
        width: figure width in pixels, or None to let plotly decide.

    Returns:
        A plotly figure.
    """
    if not isinstance(volumes, Mapping):
        volumes = {"volume": volumes}
    if not volumes:
        raise ValueError("No volumes to display")

    prepared = {}
    for name, volume in volumes.items():
        array = np.asarray(volume)
        if array.ndim == 4 and array.shape[-1] == 1:
            array = array[..., 0]
        if array.ndim != 3:
            raise ValueError(f"{name!r} must be a 3-D volume, got shape {array.shape}")
        prepared[name] = np.moveaxis(array, axis, -1)

    depths = {name: array.shape[-1] for name, array in prepared.items()}
    if len(set(depths.values())) != 1:
        raise ValueError(f"Volumes disagree on the sliced axis: {depths}")
    num_slices = next(iter(depths.values()))

    ranges = {
        name: (float(np.nanmin(array)), float(np.nanmax(array)))
        for name, array in prepared.items()
    }
    names = list(prepared)

    def images_for(index):
        return [
            go.Image(source=_to_png_uri(prepared[name][..., index], *ranges[name]))
            for name in names
        ]

    initial = num_slices // 2
    figure = go.Figure(
        data=images_for(initial),
        frames=[go.Frame(name=str(i), data=images_for(i)) for i in range(num_slices)],
    )
    for position, trace in enumerate(figure.data):
        trace.visible = position == 0

    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor=_VIEWER_BACKGROUND,
        plot_bgcolor=_VIEWER_BACKGROUND,
        font=dict(family=_VIEWER_FONT, size=12, color=_VIEWER_FOREGROUND),
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=13))
        if title
        else None,
        height=height,
        width=width,
        margin=dict(l=8, r=8, t=34 if title else 8, b=8),
        showlegend=False,
        sliders=[
            {
                "active": initial,
                "currentvalue": {
                    "prefix": "",
                    "font": {"size": 12, "color": _VIEWER_FOREGROUND},
                    "offset": 6,
                },
                "len": 0.96,
                "x": 0.02,
                "pad": {"t": 6, "b": 6},
                "bgcolor": _VIEWER_TRACK,
                "activebgcolor": _VIEWER_FOREGROUND,
                "bordercolor": _VIEWER_BACKGROUND,
                "borderwidth": 0,
                "tickcolor": _VIEWER_BACKGROUND,
                "ticklen": 0,
                "minorticklen": 0,
                "font": {"size": 1, "color": _VIEWER_BACKGROUND},
                "steps": [
                    {
                        "args": [
                            [str(i)],
                            {
                                "mode": "immediate",
                                "frame": {"redraw": True, "duration": 0},
                            },
                        ],
                        # Labelling every slice turns the track into a smear of
                        # numbers; the current value is shown above it instead.
                        "label": f"{i + 1}/{num_slices}",
                        "method": "animate",
                    }
                    for i in range(num_slices)
                ],
            }
        ],
    )
    if len(names) > 1:
        figure.update_layout(
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [{"visible": [n == name for n in names]}],
                            "label": name,
                            "method": "restyle",
                        }
                        for name in names
                    ],
                    "direction": "down",
                    "showactive": True,
                    "x": 0.0,
                    "xanchor": "left",
                    "y": 1.0,
                    "yanchor": "bottom",
                    "bgcolor": _VIEWER_TRACK,
                    "bordercolor": _VIEWER_TRACK,
                    "font": {"size": 12, "color": _VIEWER_FOREGROUND},
                    "pad": {"l": 2, "r": 2, "t": 2, "b": 2},
                }
            ]
        )
    figure.update_xaxes(visible=False, showgrid=False, zeroline=False)
    figure.update_yaxes(
        visible=False, showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1
    )
    return figure
