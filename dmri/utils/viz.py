import os
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from dipy.data import get_sphere
from jax.typing import ArrayLike
from plotly.subplots import make_subplots

from dmri.utils.dmriutils import cart2sph, sph2cart

sphere_default = get_sphere(name="symmetric724")



def set_style(style="dark"):
    # Directory where this file lives
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if style == "white":
        style_path = os.path.join(base_dir, "pyloric.mplstyle")
    elif style == "dark":
        style_path = os.path.join(base_dir, "pyloric_black.mplstyle")
    else:
        raise ValueError("Style must be 'white' or 'dark'")

    plt.style.use(style_path)


set_style()


def _ensure_axis(
    ax=None,
    *,
    projection: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    subplot_kw: Optional[Dict[str, Any]] = None,
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


def orthoview_quiver_plotly(
    data,
    fractions,
    step=1,
    colors=None,
    xy_slice=None,
    xz_slice=None,
    yz_slice=None,
    height=None,
    width=None,
):
    if data.ndim != 5 or data.shape[-1] != 3:
        raise ValueError(
            "Data must have shape (x, y, z, channels, 3) where last dim is vector (u, v, w)"
        )

    x_max, y_max, z_max, channels, _ = data.shape
    fractions = fractions.reshape(x_max, y_max, z_max, channels)

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

    data = fractions[..., None] * data
    fsum = np.sum(fractions, axis=-1)

    def create_line_quiver_2d_vectorized(X, Y, U, V, color="white", scale=1.0):
        x0 = X - U * scale
        x1 = X + U * scale
        y0 = Y - V * scale
        y1 = Y + V * scale

        x_lines = np.vstack([
            x0.flatten(),
            x1.flatten(),
            np.full(X.size, np.nan),
        ]).T.flatten()
        y_lines = np.vstack([
            y0.flatten(),
            y1.flatten(),
            np.full(X.size, np.nan),
        ]).T.flatten()

        return [
            go.Scatter(
                x=x_lines,
                y=y_lines,
                mode="lines",
                line=dict(color=color, width=1),
                showlegend=False,
            )
        ]

    fig = make_subplots(
        rows=2,
        cols=2,
        column_widths=[0.35, 0.65],
        specs=[[{}, {"rowspan": 2}], [{"colspan": 1}, None]],
        vertical_spacing=0.0,
        horizontal_spacing=0.0,
    )

    if colors is None:
        colors = ["red", "blue", "green"]

    # --- Generalized slice plotting with per-axis control ---
    all_trace_groups = {"XY": [], "XZ": [], "YZ": []}

    def handle_axis(axis_name, axis_size, slice_selector, plot_func, row, col):
        if slice_selector is None:
            return

        if isinstance(slice_selector, int):
            slice_list = [slice_selector]
        elif isinstance(slice_selector, (tuple, list)) and len(slice_selector) == 2:
            slice_list = list(range(slice_selector[0], slice_selector[1] + 1))
        else:
            raise ValueError(f"{axis_name}_slice must be int or (start, end) tuple")

        trace_groups = []
        for idx in slice_list:
            traces = plot_func(idx)
            for trace in traces:
                fig.add_trace(trace, row=row, col=col)
            trace_groups.append([
                len(fig.data) - len(traces) + i for i in range(len(traces))
            ])
        all_trace_groups[axis_name] = trace_groups

        if len(slice_list) > 1:
            steps = []
            for i, group in enumerate(trace_groups):
                vis = [trace.visible for trace in fig.data]
                # Turn off only this axis' traces
                for g in trace_groups:
                    for idx_ in g:
                        vis[idx_] = False
                # Turn on selected group
                for idx_ in group:
                    vis[idx_] = True
                steps.append({
                    "method": "update",
                    "label": f"{slice_list[i]}",
                    "args": [{"visible": vis}],
                })
            fig.update_layout(
                sliders=list(fig.layout.sliders)
                + [
                    {
                        "active": 0,
                        "currentvalue": {"prefix": f"{axis_name} Slice: "},
                        "steps": steps,
                        "x": 0.05
                        if axis_name == "XY"
                        else 0.35
                        if axis_name == "XZ"
                        else 0.65,
                        "y": 0.0,
                        "len": 0.25,
                    }
                ]
            )
            # Set all but first slice of this axis invisible
            for group in trace_groups[1:]:
                for idx_ in group:
                    fig.data[idx_].visible = False

    # Define plotting functions per axis
    def plot_xy(x_idx):
        traces = [
            go.Heatmap(
                z=fsum[x_idx, :, :].T,
                colorscale="gray",
                zmin=0,
                zmax=1,
                showscale=False,
            )
        ]
        for c in range(channels):
            U = data[x_idx, :, :, c, 0]
            V = data[x_idx, :, :, c, 1]
            Y, Z = np.mgrid[0:y_max:step, 0:z_max:step]
            traces += create_line_quiver_2d_vectorized(
                Y, Z, U[::step, ::step], V[::step, ::step], colors[c % len(colors)]
            )
        return traces

    def plot_xz(y_idx):
        traces = [
            go.Heatmap(
                z=fsum[:, y_idx, :].T,
                colorscale="gray",
                zmin=0,
                zmax=1,
                showscale=False,
            )
        ]
        for c in range(channels):
            U = data[:, y_idx, :, c, 0]
            V = data[:, y_idx, :, c, 2]
            X, Z = np.mgrid[0:x_max:step, 0:z_max:step]
            traces += create_line_quiver_2d_vectorized(
                X, Z, U[::step, ::step], V[::step, ::step], colors[c % len(colors)]
            )
        return traces

    def plot_yz(z_idx):
        traces = [
            go.Heatmap(
                z=fsum[:, :, z_idx], colorscale="gray", zmin=0, zmax=1, showscale=False
            )
        ]
        for c in range(channels):
            U = data[:, :, z_idx, c, 0]
            V = data[:, :, z_idx, c, 1]
            X, Y_ = np.mgrid[0:x_max:step, 0:y_max:step]
            traces += create_line_quiver_2d_vectorized(
                X, Y_, U[::step, ::step].T, V[::step, ::step].T, colors[c % len(colors)]
            )
        return traces

    # Process axes
    handle_axis("XY", x_max, xy_slice, plot_xy, row=1, col=1)
    handle_axis("XZ", y_max, xz_slice, plot_xz, row=2, col=1)
    handle_axis("YZ", z_max, yz_slice, plot_yz, row=1, col=2)

    fig.update_layout(
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=10, b=20),
        height=height,
        width=width,
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    return fig


def orthoview_quiver_ultracompact(
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
    """Ultra-optimized version of orthoview_quiver_plotly that produces minimal HTML files.

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


def plot_volume(data, vmin=None, vmax=None, color_map="gray"):
    """Plot a 3D volume visualization of the data.

    Args:
        data: 3D numpy array to visualize
        vmin: Minimum value for colorscale
        vmax: Maximum value for colorscale
        color_map: Colormap to use for visualization
    """
    # Create coordinate meshgrid
    x = np.arange(data.shape[0])
    y = np.arange(data.shape[1])
    z = np.arange(data.shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Create a 3D volume plot
    fig = go.Figure(
        data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=data.flatten(),
            isomin=vmin
            if vmin is not None
            else 0.4,  # Increase minimum threshold to filter out noise
            isomax=vmax if vmax is not None else np.max(data),
            opacity=1.0,  # Increase opacity
            opacityscale=[[0.4, 0.1], [0.5, 0.3], [0.8, 1.0]],
            surface_count=10,  # Increase surface resolution
            colorscale=color_map,
            caps=dict(x_show=False, y_show=False, z_show=False),
            showscale=False,  # Hide colorbar
        )
    )

    # Update the layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                showline=False,
                showspikes=False,
                showbackground=False,
                showaxeslabels=False,
                title="",
            ),
            yaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                showline=False,
                showspikes=False,
                showbackground=False,
                showaxeslabels=False,
                title="",
            ),
            zaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                showline=False,
                showspikes=False,
                showbackground=False,
                showaxeslabels=False,
                title="",
            ),
            camera=dict(
                eye=dict(x=2, y=2, z=2)  # Move camera further out
            ),
            aspectmode="data",  # Preserve data aspect ratio
            bgcolor="black",  # Set background color to black
        ),
        paper_bgcolor="black",  # Set paper background to black
        plot_bgcolor="black",  # Set plot background to black
    )

    fig.update_traces(hoverinfo="skip", hovertemplate=None)
    return fig


def plot_3d_quiver(
    fractions,
    directions,
    step=1,
    length_threshold=0.6,
    fraction_threshold=0.15,
    colors=None,
    opacity=0.5,
    line_width=1,
    height=None,
    width=None,
):
    """Plot a 3D quiver visualization of vector field data.

    Args:
        fractions: 4D numpy array (x, y, z, channels) of fraction weights
        directions: 5D numpy array (x, y, z, channels, 3) of direction vectors
        step: Step size for subsampling points
        length_threshold: Minimum sum of fractions to show vector
        fraction_threshold: Minimum fraction value to show vector
        colors: List of colors for each channel
        opacity: Opacity of the vectors
        line_width: Width of the vector lines
        height: Height of the figure in pixels
        width: Width of the figure in pixels
    """
    if fractions.shape[:-1] != directions.shape[:-2]:
        raise ValueError("Spatial dimensions of fractions and directions must match")
    if fractions.shape[-1] != directions.shape[-2]:
        raise ValueError("Number of channels in fractions and directions must match")
    if directions.shape[-1] != 3:
        raise ValueError("Directions must have 3 components (x,y,z)")

    # Create coordinate meshgrid for vector field
    x = np.arange(fractions.shape[0])
    y = np.arange(fractions.shape[1])
    z = np.arange(fractions.shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Create a 3D vector field plot
    fig = go.Figure()

    # Define colors for each channel
    if colors is None:
        colors = ["red", "green", "blue", "yellow", "cyan", "magenta"]
    colors = colors[: fractions.shape[-1]]  # Limit colors to number of channels

    # Add vectors for each channel
    for channel in range(fractions.shape[-1]):
        # Scale vectors by their fractions and subsample to reduce data
        channel_fractions = fractions[::step, ::step, ::step, channel : channel + 1][
            ..., 0
        ]
        u = channel_fractions * directions[::step, ::step, ::step, channel, 0]
        v = channel_fractions * directions[::step, ::step, ::step, channel, 1]
        w = channel_fractions * directions[::step, ::step, ::step, channel, 2]

        # Create mask for vectors above threshold length
        fsum = fractions[::step, ::step, ::step].sum(axis=-1)
        mask = (fsum > length_threshold) & (channel_fractions > fraction_threshold)

        # Apply mask to coordinates and vectors
        x_coords = X[::step, ::step, ::step][mask]
        y_coords = Y[::step, ::step, ::step][mask]
        z_coords = Z[::step, ::step, ::step][mask]
        u = u[mask]
        v = v[mask]
        w = w[mask]

        # Create start and end points for lines
        x_start = x_coords - u * 2
        y_start = y_coords - v * 2
        z_start = z_coords - w * 2

        x_end = x_start + u * 2
        y_end = y_start + v * 2
        z_end = z_start + w * 2

        # Add lines connecting start and end points in a vectorized way
        x_lines = np.vstack([x_start, x_end, np.full_like(x_start, np.nan)]).T.flatten()
        y_lines = np.vstack([y_start, y_end, np.full_like(y_start, np.nan)]).T.flatten()
        z_lines = np.vstack([z_start, z_end, np.full_like(z_start, np.nan)]).T.flatten()

        fig.add_trace(
            go.Scatter3d(
                x=x_lines,
                y=y_lines,
                z=z_lines,
                mode="lines",
                line=dict(color=colors[channel], width=line_width),
                opacity=opacity,
                showlegend=False,
            )
        )

    # Update the layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                showline=False,
                showspikes=False,
                showbackground=False,
                showaxeslabels=False,
                title="",
            ),
            yaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                showline=False,
                showspikes=False,
                showbackground=False,
                showaxeslabels=False,
                title="",
            ),
            zaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                showline=False,
                showspikes=False,
                showbackground=False,
                showaxeslabels=False,
                title="",
            ),
            camera=dict(eye=dict(x=2, y=2, z=2)),
            aspectmode="data",
            bgcolor="black",
        ),
        paper_bgcolor="black",
        plot_bgcolor="black",
        height=height,
        width=width,
    )

    fig.update_traces(hoverinfo="skip", hovertemplate=None)
    return fig


def plot_spherical_function(
    theta: ArrayLike,
    phi: ArrayLike,
    func_values: ArrayLike,
    elev: float = 30,
    azim: float = 30,
    ax=None,
    cmap: str = "viridis",
    alpha: float = 0.7,
    figsize: Optional[Tuple[float, float]] = (6, 6),
    hide_axes: bool = True,
    add_colorbar: bool = False,
    surface_kwargs: Optional[Dict[str, Any]] = None,
):
    """Plot a scalar function that is defined on the sphere.

    Returns:
        Tuple[matplotlib.figure.Figure, matplotlib.axes._subplots.Axes3DSubplot]
    """
    surface_kwargs = dict(surface_kwargs or {})
    fig, ax = _ensure_axis(ax, projection="3d", figsize=figsize)

    ax.view_init(elev=elev, azim=azim)

    theta = np.asarray(theta)
    phi = np.asarray(phi)
    func_values = np.asarray(func_values)

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    norm_vals = (func_values - func_values.min()) / (np.ptp(func_values) + 1e-15)
    cmap_obj = plt.get_cmap(cmap)
    facecolors = cmap_obj(norm_vals)

    ax.plot_surface(
        x,
        y,
        z,
        facecolors=facecolors,
        rstride=1,
        cstride=1,
        alpha=alpha,
        **surface_kwargs,
    )

    if hide_axes:
        ax.set_axis_off()

    if add_colorbar:
        mappable = plt.cm.ScalarMappable(cmap=cmap_obj)
        mappable.set_array(func_values)
        fig.colorbar(mappable, ax=ax, shrink=0.6)

    return fig, ax


def plot_spherical_distribution_polar(
    distribution,
    n_samples: int = 1000,
    ax=None,
    color=None,
    levels: int = 3,
    figsize: Optional[Tuple[float, float]] = (6, 4),
    cmap: str = "viridis",
    filled: bool = False,
    contour_kwargs: Optional[Dict[str, Any]] = None,
    show_axis_labels: bool = True,
    title: Optional[str] = "Spherical Distribution PDF",
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
    figsize: Optional[Tuple[float, float]] = (6, 6),
    cmap: str = "viridis",
    vertex_kwargs: Optional[Dict[str, Any]] = None,
    show_samples: bool = True,
    sample_kwargs: Optional[Dict[str, Any]] = None,
    add_colorbar: bool = True,
    colorbar_kwargs: Optional[Dict[str, Any]] = None,
    title: Optional[str] = "Spherical Distribution PDF",
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
    figsize: Optional[Tuple[float, float]] = (7, 7),
    cmap: Optional[str] = "viridis",
    surface_kwargs: Optional[Dict[str, Any]] = None,
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


def save_orthoview_html(fig, filepath, include_plotlyjs="cdn"):
    """Save orthoview figure to HTML file with optimized settings.

    Args:
        fig: plotly figure object
        filepath: path to save HTML file
        include_plotlyjs: plotly.js inclusion method
            'cdn': use CDN (smallest file size)
            'directory': save plotly.js in a directory
            True: include plotly.js in the HTML file (largest file size)
    """
    fig.write_html(
        filepath,
        include_plotlyjs=include_plotlyjs,
        full_html=True,
        auto_play=False,
        include_mathjax=False,
    )


def orthoview_ultracompact(
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
    """Ultra-optimized version of orthoview that produces minimal HTML files.

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


from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ---------------------- helpers ----------------------
def normalize_rows(X: np.ndarray, eps: float = 1e-12):
    n = np.linalg.norm(X, axis=1, keepdims=True) + eps
    return X / n


def fibonacci_sphere(n: int) -> np.ndarray:
    # quasi-uniform points on S^2
    i = np.arange(n)
    phi = (1 + 5**0.5) / 2
    z = 1 - 2 * (i + 0.5) / n
    r = np.sqrt(1 - z * z)
    theta = 2 * np.pi * i / phi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.vstack([x, y, z]).T


def spherical_kde(
    points: np.ndarray, grid: np.ndarray, kappa: float, axial: bool = True
) -> np.ndarray:
    # vMF KDE: sum_i exp(kappa * (μ_i · x)); for axial, also add exp(kappa * (-μ_i · x))
    # All inputs assumed unit vectors.
    MU = points  # (N,3)
    X = grid  # (M,3)
    dots = X @ MU.T  # (M,N)
    s = np.exp(kappa * dots)
    if axial:
        s = s + np.exp(-kappa * dots)
    f = s.sum(axis=1)
    # normalize to [0,1] for display
    f = (f - f.min()) / (f.max() - f.min() + 1e-12)
    return f


def orientation_rgb(dirs: np.ndarray) -> np.ndarray:
    c = np.abs(dirs)
    c = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-12)
    return c


def nonmax_suppression_on_sphere(
    values: np.ndarray, dirs: np.ndarray, k_neighbors: int = 12
):
    # crude NMS: keep points whose value is greater than their k nearest angular neighbors
    # Use dot similarity to approximate neighbor search
    D = dirs @ dirs.T  # cosine similarity
    np.fill_diagonal(D, -np.inf)
    idx = np.argpartition(-D, kth=k_neighbors, axis=1)[
        :, :k_neighbors
    ]  # neighbors with highest cosine
    keep = np.ones(len(values), dtype=bool)
    for i in range(len(values)):
        if not np.all(values[i] >= values[idx[i]]):
            keep[i] = False
    return keep


def _set_view(ax, view):
    # Accept 'xy','xz','yz' or 3-vector
    if isinstance(view, str):
        view = view.lower()
        if view == "xy":  # look along +z
            elev, azim = 90, -90  # top-down
        elif view == "xz":  # look along +y
            elev, azim = 0, -90
        elif view == "yz":  # look along +x
            elev, azim = 0, 180
        else:
            # default nice isometric
            elev, azim = 20, -60
    else:
        # view is a direction vector -> compute spherical angles
        v = np.asarray(view, float)
        v = v / (np.linalg.norm(v) + 1e-12)
        # Matplotlib's view is defined by elev (degrees from xy) and azim (degrees CCW from x)
        elev = np.degrees(np.arcsin(v[2]))  # z component -> elevation
        azim = np.degrees(np.arctan2(v[1], v[0]))  # y,x -> azimuth
    ax.view_init(elev=elev, azim=azim)


# --------------------- plotting ----------------------
def plot_glyph_from_sticks(
    V: np.ndarray,
    view: str | np.ndarray = "xy",
    axial: bool = False,
    kappa: float = 20.0,
    grid_points: int = 4000,
    r_scale: float = 1.0,
    peak_nms_neighbors: int = 16,
    show_peaks: bool = False,
    ax=None,
    figsize: Optional[Tuple[float, float]] = (7, 7),
    surface_kwargs: Optional[Dict[str, Any]] = None,
    peaks_kwargs: Optional[Dict[str, Any]] = None,
    hide_axes: bool = True,
):
    """Render a spherical glyph built from discrete stick directions.

    Returns:
        Tuple of (Figure, Axes)
    """
    V = normalize_rows(np.asarray(V, float))
    # grid on sphere
    G = fibonacci_sphere(grid_points)
    # KDE
    f = spherical_kde(V, G, kappa=kappa, axial=axial)
    # radius field
    R = 0.2 + r_scale * f  # small base radius + scaled KDE
    verts = R[:, None] * G
    # orientation color
    colors = orientation_rgb(G)

    # crude triangulation for sphere: use matplotlib trisurf via spherical parameterization indices
    # We'll parametrize with lon/lat sorted mapping to a Delaunay in 2D for nicer surface.
    # Convert to spherical coords for triangulation
    x, y, z = G.T
    lon = np.arctan2(y, x)
    lat = np.arcsin(z)
    # stack as 2D points
    P2 = np.vstack([lon, lat]).T

    # Use matplotlib.tri for triangulation
    import matplotlib.tri as mtri

    tri = mtri.Triangulation(P2[:, 0], P2[:, 1])

    fig, ax = _ensure_axis(ax, projection="3d", figsize=figsize)
    surface_kwargs = dict(surface_kwargs or {})

    base_surface_kwargs = dict(
        triangles=tri.triangles,
        linewidth=0.1,
        antialiased=True,
        shade=True,
        alpha=1.0,
        edgecolor="none",
    )
    base_surface_kwargs.update(surface_kwargs)

    surf = ax.plot_trisurf(
        verts[:, 0],
        verts[:, 1],
        verts[:, 2],
        **base_surface_kwargs,
    )
    # set vertex colors via face colors approximation
    # Map per-vertex RGB to per-triangle by averaging
    face_rgb = colors[tri.triangles].mean(axis=1)
    surf.set_facecolors(face_rgb)

    # optional: show peak sticks
    if show_peaks:
        keep = nonmax_suppression_on_sphere(f, G, k_neighbors=peak_nms_neighbors)
        peaks = G[keep]
        # keep only top K peaks for cleanliness
        K = min(6, len(peaks))
        top_idx = np.argsort(f[keep])[-K:]
        peaks = peaks[top_idx]
        peaks_kwargs = dict(peaks_kwargs or {})
        peaks_kwargs.setdefault("linewidth", 2)
        peaks_kwargs.setdefault("color", "white")
        for p in peaks:
            ax.plot(
                [-p[0], p[0]],
                [-p[1], p[1]],
                [-p[2], p[2]],
                **peaks_kwargs,
            )

    # cosmetics
    lim = 1.25 * (0.2 + r_scale)
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    ax.set_box_aspect([1, 1, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    _set_view(ax, view)
    if hide_axes:
        ax.set_axis_off()

    return fig, ax


from matplotlib.patches import Circle


def _norm_rows(X, eps=1e-12):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)


def _stereo_project(X):
    x, y, z = X.T
    d = np.clip(1 + z, 1e-9, None)
    return np.c_[x / d, y / d]


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
    figsize: Optional[Tuple[float, float]] = (5, 5),
    facecolor: Optional[str] = "black",
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
    V = _norm_rows(np.asarray(V, float))
    if axial:
        V = np.where(V[:, 2:3] < 0, -V, V)

    UV = _stereo_project(V)
    inside = np.sum(UV**2, axis=1) <= 1.0 + 1e-9
    UV = UV[inside]
    C = None
    if point_color == "orientation":
        C = np.abs(V[inside])
        C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    else:
        C = point_color

    existing_ax = ax
    fig, ax = _ensure_axis(ax, figsize=figsize)
    if facecolor is not None:
        ax.set_facecolor(facecolor)

    # circular frame
    boundary = Circle((0, 0), 1.0, fill=False, lw=1.8, zorder=2, color=frame_color)
    ax.add_artist(boundary)

    # concentric rings
    if draw_guides:
        for r in ring_radii:
            ax.add_artist(
                Circle(
                    (0, 0),
                    r,
                    fill=False,
                    lw=0.8,
                    ls="--",
                    alpha=0.6,
                    zorder=1,
                    color=frame_color,
                )
            )
            ax.text(
                r / np.sqrt(2),
                r / np.sqrt(2),
                f"{r:.2f}",
                ha="left",
                va="bottom",
                fontsize=9,
                alpha=0.7,
                zorder=3,
                color=frame_color,
            )

    # spokes
    if draw_guides:
        for deg in range(0, 360, angle_step):
            th = np.deg2rad(deg)
            ax.plot(
                [0, np.cos(th)],
                [0, np.sin(th)],
                lw=0.6,
                ls="--",
                alpha=0.6,
                zorder=1,
                color=frame_color,
            )
            rlab = 1.1
            ax.text(
                rlab * np.cos(th),
                rlab * np.sin(th),
                f"{deg}°",
                ha="center",
                va="center",
                fontsize=9,
                zorder=3,
                color=frame_color,
            )

    # scatter on top
    zord = 5
    if edge:
        ax.scatter(
            UV[:, 0],
            UV[:, 1],
            s=s,
            c=C,
            alpha=point_alpha,
            linewidths=0.3,
            edgecolors=frame_color,
            zorder=zord,
        )
    else:
        ax.scatter(
            UV[:, 0], UV[:, 1], s=s, c=C, alpha=point_alpha, linewidths=0, zorder=zord
        )

    # clip to circle
    clip_circle = Circle((0, 0), 1.0, transform=ax.transData)
    for col in ax.collections:
        col.set_clip_path(clip_circle)

    # clean look
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_frame_on(False)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    if existing_ax is None:
        fig.tight_layout()

    return fig, ax
