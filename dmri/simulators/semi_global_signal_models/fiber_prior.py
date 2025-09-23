from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .curves3d import (
    Curve3D,
    VoxelGrid,
    VoxelizedCurve,
    curve_points,
    sample_splines,
)


class FiberPrior:
    r"""
    A class for generating fiber prior distributions in 3D space.
    """

    def __init__(
        self,
        n_voxels: int,
        max_degree: int,
        diameter: float | None = None,
        degree_probs: Sequence[float] | None = None,
    ) -> None:
        self.max_degree = int(max_degree)
        if self.max_degree <= 0:
            raise ValueError("max_degree must be positive")
        self.grid = VoxelGrid(n_voxels, n_voxels, n_voxels)
        self.degree_values = jnp.arange(1, self.max_degree + 1)
        self.degree_probs = self._normalize_degree_probs(degree_probs)
        self.diameter = float(diameter) if diameter is not None else 1.0 / n_voxels

    def _normalize_degree_probs(
        self, degree_probs: Sequence[float] | None
    ) -> jnp.ndarray:
        if degree_probs is None:
            probs = jnp.ones(self.max_degree, dtype=jnp.float32)
        else:
            probs = jnp.asarray(degree_probs, dtype=jnp.float32)
            if probs.shape != (self.max_degree,):
                raise ValueError(
                    "degree_probs must have length equal to max_degree"
                )
        probs = jnp.clip(probs, 1e-8, None)
        return probs / jnp.sum(probs)

    def sample(
        self, rng: Any,
    ) -> VoxelizedCurve | tuple[VoxelizedCurve, Any]:
        """Sample from the fiber prior distribution and voxelize the result."""
        rng1, rng2 = jax.random.split(rng)
        degree = jax.random.choice(rng1, self.degree_values, p=self.degree_probs)
        spline = sample_splines(rng2, degree=degree, max_degree=self.max_degree)
        voxelized = VoxelizedCurve(
            spline,
            diameter=self.diameter,
            grid=self.grid,
        )
        return voxelized


def plot_voxelized_fiber_field(
    volume_fractions: Any,
    grid: VoxelGrid,
    tangents: Any | None = None,
    *,
    threshold: float = 1e-4,
    ax=None,
    cmap: str = "viridis",
    size_scale: float = 600.0,
    show_tangents_only: bool = False,
    curves: Curve3D | VoxelizedCurve | Sequence[Curve3D | VoxelizedCurve] | None = None,
    curve_num_points: int = 200,
    volume_opacity: float | None = None,
):
    """Visualize discretized fiber volume fractions and optional tangents.

    Args:
        volume_fractions: 3-D voxel grid of fiber volume occupancy.
        grid: The grid instance used to discretize the spline.
        tangents: Optional (Nx, Ny, Nz, 3) array of average tangents per voxel.
        threshold: Minimum volume fraction to display.
        ax: Optional Plotly figure to append traces to.
        cmap: Colormap / colorscale name for the voxel coloring.
        size_scale: Controls voxel opacity (legacy name retained for API stability).
        show_tangents_only: Skip volume fractions and only draw tangent arrows.
        curves: Optional analytic curves (or voxelized curves) to overlay.
        curve_num_points: Number of samples per curve when rendering lines.
        volume_opacity: Override for voxel mesh opacity (0–1). Defaults to
            ``size_scale`` behaviour when ``None``.

    Returns:
        A tuple ``(fig, fig)`` with the Plotly figure (duplicated for backward
        compatibility with the previous ``(fig, ax)`` signature).
    """
    try:
        import plotly.graph_objects as go
        from plotly import colors as plotly_colors
    except ImportError as exc:  # pragma: no cover - plotting helper
        raise ImportError(
            "plot_voxelized_fiber_field requires plotly to be installed"
        ) from exc

    def resolve_colorscale(name: str) -> str:
        try:
            available = {n.lower(): n for n in plotly_colors.named_colorscales()}
            return available.get(name.lower(), "Viridis")
        except Exception:  # pragma: no cover - defensive fallback
            return "Viridis"

    vf = np.asarray(volume_fractions)
    if vf.ndim != 3:
        raise ValueError("volume_fractions must be a 3-D array")

    indices = np.argwhere(vf > threshold)
    fig = ax if isinstance(ax, go.Figure) else go.Figure()

    def add_grid_wireframe(target_fig: go.Figure, *, line_color: str = "rgba(160,160,160,0.6)", line_width: float = 1.0) -> None:
        x_edges = np.linspace(grid.x_min, grid.x_max, grid.n_voxels_x + 1)
        y_edges = np.linspace(grid.y_min, grid.y_max, grid.n_voxels_y + 1)
        z_edges = np.linspace(grid.z_min, grid.z_max, grid.n_voxels_z + 1)

        xs: list[float | None] = []
        ys: list[float | None] = []
        zs: list[float | None] = []

        for x in x_edges:
            for y in y_edges:
                xs.extend([x, x, None])
                ys.extend([y, y, None])
                zs.extend([z_edges[0], z_edges[-1], None])
        for x in x_edges:
            for z in z_edges:
                xs.extend([x, x, None])
                ys.extend([y_edges[0], y_edges[-1], None])
                zs.extend([z, z, None])
        for y in y_edges:
            for z in z_edges:
                xs.extend([x_edges[0], x_edges[-1], None])
                ys.extend([y, y, None])
                zs.extend([z, z, None])

        target_fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=dict(color=line_color, width=line_width),
                hoverinfo="skip",
                name="voxel grid",
                showlegend=False,
            )
        )

    add_grid_wireframe(fig)

    if indices.size == 0:
        fig.update_layout(
            title="No voxels above threshold",
            scene=dict(
                xaxis_title="x",
                yaxis_title="y",
                zaxis_title="z",
            ),
        )
        return fig, fig

    centers = np.array(
        [grid.get_voxel_center(int(ix), int(iy), int(iz)) for ix, iy, iz in indices]
    )
    strengths = vf[indices[:, 0], indices[:, 1], indices[:, 2]]

    if not show_tangents_only:
        if volume_opacity is None:
            voxel_opacity = float(np.clip(size_scale / 600.0, 0.1, 0.9))
        else:
            voxel_opacity = float(np.clip(volume_opacity, 0.0, 1.0))

        vertex_x: list[float] = []
        vertex_y: list[float] = []
        vertex_z: list[float] = []
        intensities: list[float] = []
        triangulation_i: list[int] = []
        triangulation_j: list[int] = []
        triangulation_k: list[int] = []

        cube_tris = (
            (0, 1, 2),
            (0, 2, 3),
            (4, 5, 6),
            (4, 6, 7),
            (0, 1, 5),
            (0, 5, 4),
            (1, 2, 6),
            (1, 6, 5),
            (2, 3, 7),
            (2, 7, 6),
            (3, 0, 4),
            (3, 4, 7),
        )

        for idx, (ix, iy, iz) in enumerate(indices):
            lower, upper = grid.get_voxel_boundaries(int(ix), int(iy), int(iz))
            lower = np.asarray(lower)
            upper = np.asarray(upper)
            x0, y0, z0 = lower
            x1, y1, z1 = upper
            voxel_vertices = [
                (x0, y0, z0),
                (x1, y0, z0),
                (x1, y1, z0),
                (x0, y1, z0),
                (x0, y0, z1),
                (x1, y0, z1),
                (x1, y1, z1),
                (x0, y1, z1),
            ]
            base_index = len(vertex_x)
            for vx, vy, vz in voxel_vertices:
                vertex_x.append(float(vx))
                vertex_y.append(float(vy))
                vertex_z.append(float(vz))
                intensities.append(float(strengths[idx]))
            for tri in cube_tris:
                triangulation_i.append(base_index + tri[0])
                triangulation_j.append(base_index + tri[1])
                triangulation_k.append(base_index + tri[2])

        if vertex_x:
            fig.add_trace(
                go.Mesh3d(
                    x=vertex_x,
                    y=vertex_y,
                    z=vertex_z,
                    i=triangulation_i,
                    j=triangulation_j,
                    k=triangulation_k,
                    intensity=intensities,
                    colorscale=resolve_colorscale(cmap),
                    showscale=True,
                    colorbar=dict(title="volume fraction"),
                    opacity=voxel_opacity,
                    flatshading=True,
                    name="volume fractions",
                    hoverinfo="skip",
                )
            )

    if tangents is not None:
        tangents_arr = np.asarray(tangents)
        if tangents_arr.shape[:3] != vf.shape:
            raise ValueError(
                "tangents must have the same spatial shape as volume_fractions"
            )
        tangent_vectors = tangents_arr[
            indices[:, 0], indices[:, 1], indices[:, 2], :
        ]
        norms = np.linalg.norm(tangent_vectors, axis=1, keepdims=True)
        safe_norms = np.where(norms > 0, norms, 1.0)
        tangent_vectors = tangent_vectors / safe_norms
        vx, vy, vz = grid.get_voxel_size()
        length = float(min(vx, vy, vz))
        ends = centers + tangent_vectors * length
        xs = []
        ys = []
        zs = []
        for start, end in zip(centers, ends):
            xs.extend([start[0], end[0], None])
            ys.extend([start[1], end[1], None])
            zs.extend([start[2], end[2], None])
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=dict(color="black", width=3),
                name="tangents",
                hoverinfo="skip",
            )
        )

    if curves is not None:
        if not isinstance(curves, Sequence) or isinstance(curves, (Curve3D, VoxelizedCurve)):
            curve_list = [curves]
        else:
            curve_list = list(curves)

        try:
            from plotly import colors as plotly_colors

            base_colors = list(plotly_colors.qualitative.Plotly)
        except Exception:
            base_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

        for idx, curve_obj in enumerate(curve_list):
            if isinstance(curve_obj, VoxelizedCurve):
                curve = curve_obj.curve
            else:
                curve = curve_obj
            if curve is None:
                continue
            points = curve_points(curve, num_points=curve_num_points)
            if points.size == 0:
                continue
            color = base_colors[idx % len(base_colors)]
            fig.add_trace(
                go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode="lines",
                    line=dict(color=color, width=4),
                    name=f"curve {idx + 1}",
                    hoverinfo="skip",
                )
            )

    title = "Fiber tangents" if show_tangents_only else "Voxelized fiber volume fractions"
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="cube",
        ),
    )
    return fig, fig
