import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from dipy.data import get_sphere
from jax.typing import ArrayLike

from dmri.utils.dmriutils import cartesian_to_unitsphere, unitsphere_to_cartesian

sphere_default = get_sphere(name="symmetric724")

import plotly.graph_objects as go
from plotly.subplots import make_subplots


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

        x_lines = np.vstack(
            [x0.flatten(), x1.flatten(), np.full(X.size, np.nan)]
        ).T.flatten()
        y_lines = np.vstack(
            [y0.flatten(), y1.flatten(), np.full(X.size, np.nan)]
        ).T.flatten()

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
            trace_groups.append(
                [len(fig.data) - len(traces) + i for i in range(len(traces))]
            )
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
                steps.append(
                    {
                        "method": "update",
                        "label": f"{slice_list[i]}",
                        "args": [{"visible": vis}],
                    }
                )
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
            return sorted(list(set(result)))

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
        x_steps.append(
            {
                "method": "restyle",
                "label": str(x_idx),
                "args": [step_data, step_indices],
            }
        )

    sliders.append(
        {
            "active": x_indices.index(x_slice) if x_slice in x_indices else 0,
            "currentvalue": {"prefix": "X: " if simplified_ui else "X Slice: "},
            "steps": x_steps,
            "x": 0.05,
            "y": 0.0,
            "len": 0.25,
            "pad": {"t": 50},
        }
    )

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
        y_steps.append(
            {
                "method": "restyle",
                "label": str(y_idx),
                "args": [step_data, step_indices],
            }
        )

    sliders.append(
        {
            "active": y_indices.index(y_slice) if y_slice in y_indices else 0,
            "currentvalue": {"prefix": "Y: " if simplified_ui else "Y Slice: "},
            "steps": y_steps,
            "x": 0.35,
            "y": 0.0,
            "len": 0.25,
            "pad": {"t": 50},
        }
    )

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
        z_steps.append(
            {
                "method": "restyle",
                "label": str(z_idx),
                "args": [step_data, step_indices],
            }
        )

    sliders.append(
        {
            "active": z_indices.index(z_slice) if z_slice in z_indices else 0,
            "currentvalue": {"prefix": "Z: " if simplified_ui else "Z Slice: "},
            "steps": z_steps,
            "x": 0.65,
            "y": 0.0,
            "len": 0.25,
            "pad": {"t": 50},
        }
    )

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

            buttons.append(
                {"method": "update", "label": str(c), "args": [{"visible": visibility}]}
            )

        updatemenus.append(
            {
                "buttons": buttons,
                "direction": "right" if simplified_ui else "up",
                "showactive": True,
                "active": 0,
                "x": 0.85,
                "y": 0.05,
                "xanchor": "left",
                "yanchor": "bottom",
            }
        )

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
):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Set camera angle
    ax.view_init(elev=elev, azim=azim)

    # Convert spherical to Cartesian for plotting
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    # Plot the surface (color it by the function value)
    # Ensure func_values is normalized for color mapping
    norm_vals = (func_values - func_values.min()) / (np.ptp(func_values) + 1e-15)

    ax.plot_surface(
        x, y, z, facecolors=plt.cm.viridis(norm_vals), rstride=1, cstride=1, alpha=0.7
    )
    plt.axis("off")
    plt.show()


def plot_spherical_distribution_polar(
    distribution,
    n_samples: int = 1000,
    ax=None,
    color=None,
    levels: int = 3,
):
    """Plot spherical distribution in polar coordinates.

    Args:
        distribution: Spherical distribution object with pdf and sample methods
        n_samples: Number of samples to generate
        ax: Matplotlib axis to plot on
        color: Colormap to use
        levels: Number of contour levels
    """
    samples = distribution.sample(jax.random.key(0), (n_samples,))
    grid_y = np.linspace(0, np.pi, 100)
    grid_x = np.linspace(-np.pi, np.pi, 200)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    samples_rand = np.stack([grid_y.flatten(), grid_x.flatten()], axis=-1)
    samples_cart = jax.vmap(unitsphere_to_cartesian)(samples_rand)
    samples = jnp.concatenate([samples, samples_cart], axis=0)
    pdf = distribution.pdf(samples)
    mu = jax.vmap(cartesian_to_unitsphere)(samples)
    pdf = pdf / jnp.max(pdf)

    if ax is None:
        fig = plt.figure()
        ax = plt.gca()

    cmap = "viridis" if color is None else color
    ax.tricontour(
        mu[:, 1],
        mu[:, 0],
        pdf,
        cmap=cmap,
        levels=levels,
    )
    ax.set_ylim(0, np.pi)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_aspect("equal")
    ax.set_title("Spherical Distribution PDF")
    ax.set_xlabel("Azimuthal Angle (phi)")
    ax.set_ylabel("Polar Angle (theta)")


def plot_spherical_distribution_cartesian(
    distribution,
    n_samples: int = 1000,
    sphere=None,
    ax=None,
):
    """Plot spherical distribution in Cartesian coordinates.

    Args:
        distribution: Spherical distribution object with pdf and sample methods
        n_samples: Number of samples to generate
        sphere: Sphere object for vertices
        ax: Matplotlib axis to plot on
    """
    sphere = sphere_default if sphere is None else sphere
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

    pdfs = distribution.pdf(sphere.vertices)
    samples = distribution.sample(jax.random.key(0), (n_samples,))

    # Plot sphere vertices colored by their pdf
    sc = ax.scatter(
        sphere.vertices[:, 0],
        sphere.vertices[:, 1],
        sphere.vertices[:, 2],
        c=pdfs,
        cmap="viridis",
    )

    # Overlay sample points in red
    ax.scatter(
        samples[:, 0],
        samples[:, 1],
        samples[:, 2],
        color="red",
        s=10,
        alpha=0.1,
        label="Samples",
    )

    plt.colorbar(sc, label="PDF value")
    ax.set_title("Spherical Distribution PDF")


def plot_spherical_distribution_fod(
    distribution,
    ax=None,
    alpha=None,
):
    """Plot spherical distribution as a fiber orientation distribution (FOD).

    Args:
        distribution: Spherical distribution object with pdf and sample methods
        ax: Matplotlib axis to plot on
        alpha: Transparency of the surface
    """
    # Create a grid of points on a sphere
    samples = distribution.sample(jax.random.key(0), (10,))
    u_samples = jax.vmap(cartesian_to_unitsphere)(samples)

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

    # Plot the surface
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(x_surf, y_surf, z_surf, alpha=alpha)

    # Plot the maxima of the PDF as stick
    dir_max = points[np.argmax(pdf_values)]

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    ax.axis("off")
    return ax


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
            return sorted(list(set(result)))

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
