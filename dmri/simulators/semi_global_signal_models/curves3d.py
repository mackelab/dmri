from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKey
from jax.typing import ArrayLike


class Curve3D(ABC):
    """Abstract interface for parametric 3D curves."""

    @abstractmethod
    def __call__(self, t: ArrayLike) -> ArrayLike:
        """Evaluate curve coordinates at parameter ``t``."""

    def tangent(self, t: ArrayLike, *, epsilon: float = 1e-3) -> ArrayLike:
        """Finite-difference tangent for curves without an analytic derivative."""
        t_arr = jnp.asarray(t)
        eps = float(epsilon)
        forward = self(t_arr + eps)
        backward = self(t_arr - eps)
        return (forward - backward) / (2.0 * eps)

    def sample_points(
        self,
        *,
        num_points: int = 200,
        t_min: float = 0.0,
        t_max: float = 1.0,
    ) -> np.ndarray:
        ts = jnp.linspace(t_min, t_max, num_points)
        return np.asarray(self(ts))


@jax.tree_util.register_pytree_node_class
class Spline3D(Curve3D):
    """A 3D spline representation using separate x, y, and z polynomial coefficients."""

    def __init__(
        self,
        spline_coeffs_x: ArrayLike,
        spline_coeffs_y: ArrayLike,
        spline_coeffs_z: ArrayLike,
    ) -> None:
        """
        Parameters:
            spline_coeffs_x (ArrayLike): Polynomial coefficients for x dimension.
            spline_coeffs_y (ArrayLike): Polynomial coefficients for y dimension.
            spline_coeffs_z (ArrayLike): Polynomial coefficients for z dimension.
        """
        self.spline_coeffs_x = spline_coeffs_x
        self.spline_coeffs_y = spline_coeffs_y
        self.spline_coeffs_z = spline_coeffs_z

    def tree_flatten(self):
        children = (self.spline_coeffs_x, self.spline_coeffs_y, self.spline_coeffs_z)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        spline_coeffs_x, spline_coeffs_y, spline_coeffs_z = children
        return cls(spline_coeffs_x, spline_coeffs_y, spline_coeffs_z)

    def __call__(self, t: ArrayLike) -> ArrayLike:
        """
        Evaluate the spline at parameter t.

        Parameters:
            t (ArrayLike): The parameter(s) at which to evaluate.

        Returns:
            ArrayLike: An array of shape (..., 3) containing x, y, and z.
        """
        x = jnp.polyval(self.spline_coeffs_x, t)
        y = jnp.polyval(self.spline_coeffs_y, t)
        z = jnp.polyval(self.spline_coeffs_z, t)
        return jnp.stack([x, y, z], axis=-1)

    def tangent(self, t: ArrayLike, *, epsilon: float = 1e-3) -> ArrayLike:
        ts = jnp.asarray(t)
        dx_dt = jnp.polyval(jnp.polyder(self.spline_coeffs_x), ts)
        dy_dt = jnp.polyval(jnp.polyder(self.spline_coeffs_y), ts)
        dz_dt = jnp.polyval(jnp.polyder(self.spline_coeffs_z), ts)
        return jnp.stack([dx_dt, dy_dt, dz_dt], axis=-1)


class VoxelGrid:
    def __init__(
        self,
        n_voxels_x: int,
        n_voxels_y: int,
        n_voxels_z: int,
        x_min: float = 0.0,
        x_max: float = 1.0,
        y_min: float = 0.0,
        y_max: float = 1.0,
        z_min: float = 0.0,
        z_max: float = 1.0,
    ):
        self.n_voxels_x = n_voxels_x
        self.n_voxels_y = n_voxels_y
        self.n_voxels_z = n_voxels_z
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max

    def get_voxel_indices(self):
        return jnp.array(
            [
                (ix, iy, iz)
                for ix in range(self.n_voxels_x)
                for iy in range(self.n_voxels_y)
                for iz in range(self.n_voxels_z)
            ]
        )

    def get_voxel_size(self):
        return (
            (self.x_max - self.x_min) / self.n_voxels_x,
            (self.y_max - self.y_min) / self.n_voxels_y,
            (self.z_max - self.z_min) / self.n_voxels_z,
        )

    def get_voxel_center(self, ix: int, iy: int, iz: int):
        voxel_size_x, voxel_size_y, voxel_size_z = self.get_voxel_size()
        x = self.x_min + (ix + 0.5) * voxel_size_x
        y = self.y_min + (iy + 0.5) * voxel_size_y
        z = self.z_min + (iz + 0.5) * voxel_size_z
        return x, y, z

    def get_voxel_index_for_point(
        self,
        point: ArrayLike,
    ) -> jnp.ndarray:
        """
        Return the voxel index for a given 3D point.

        Parameters:
            point (ArrayLike): The 3D point (x, y, z).

        Returns:
            jnp.ndarray: A 1D array (3,) containing the voxel indices (ix, iy, iz).
        """
        voxel_size_x, voxel_size_y, voxel_size_z = self.get_voxel_size()
        shifted_point = point - jnp.array([self.x_min, self.y_min, self.z_min])
        voxel_coords = shifted_point / jnp.array(
            [voxel_size_x, voxel_size_y, voxel_size_z]
        )
        voxel_indices = jnp.floor(voxel_coords).astype(int)
        return voxel_indices

    def get_grid_lines(self):
        """
        Return 3D meshgrid (xx, yy, zz) of voxel boundaries for plotting in matplotlib.
        """
        x_edges = jnp.linspace(self.x_min, self.x_max, self.n_voxels_x + 1)
        y_edges = jnp.linspace(self.y_min, self.y_max, self.n_voxels_y + 1)
        z_edges = jnp.linspace(self.z_min, self.z_max, self.n_voxels_z + 1)
        xx, yy, zz = jnp.meshgrid(x_edges, y_edges, z_edges, indexing="ij")
        return xx, yy, zz

    def get_voxel_boundaries(self, ix: int, iy: int, iz: int):
        """
        Return the (x0, x1, y0, y1, z0, z1) boundaries of the voxel at (ix, iy, iz).
        Useful for filling the voxel interior with a color.
        """
        vx, vy, vz = self.get_voxel_size()
        x0 = self.x_min + ix * vx
        x1 = x0 + vx
        y0 = self.y_min + iy * vy
        y1 = y0 + vy
        z0 = self.z_min + iz * vz
        z1 = z0 + vz
        return jnp.array([x0, y0, z0]), jnp.array([x1, y1, z1])


@jax.tree_util.register_pytree_node_class
@dataclass
class VoxelizedCurve:
    """Container for voxelized representations of a parametric curve."""

    curve: Curve3D
    diameter: ArrayLike
    grid: VoxelGrid
    metadata: dict[str, Any] | None = None

    def tree_flatten(self):
        curve_flat, curve_tree = jax.tree_util.tree_flatten(self.curve)
        children =   (curve_flat, self.diameter,)
        aux_data = (curve_tree, self.metadata, self.grid)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        curve_flat, diameter = children
        curve_tree, metadata, grid = aux_data
        curve = jax.tree_util.tree_unflatten(curve_tree, curve_flat)
        return cls(curve=curve, diameter=diameter, grid=grid, metadata=metadata)

    @property
    def voxel_shape(self) -> tuple[int, int, int]:
        """Return the voxel grid shape as (Nx, Ny, Nz)."""
        return (
            self.grid.n_voxels_x,
            self.grid.n_voxels_y,
            self.grid.n_voxels_z,
        )

    @property
    def volume_fraction(self) -> jnp.ndarray:
        """Lazily compute and return the volume fraction in each voxel."""
        volume_fraction = get_curve_volume_fraction_in_voxels(
                self.curve,
                self.grid,
                self.diameter,
            )
        return volume_fraction

    @property
    def tangents(self) -> jnp.ndarray:
        """Lazily compute and return the average tangent in each voxel."""
        tangents = get_average_tangent_in_voxels(self.curve, self.grid)
        return tangents

    def nonzero_indices(self, threshold: float = 0.0) -> np.ndarray:
        """Return voxel indices with volume fraction above ``threshold``."""
        vf_np = np.asarray(self.volume_fraction)
        return np.argwhere(vf_np > threshold)

    def as_tuple(self) -> tuple[jnp.ndarray, jnp.ndarray | None]:
        """Legacy helper returning ``(volume_fraction, tangents)``."""
        return self.volume_fraction, self.tangents


def random_surface_point(
    rng_face: PRNGKey,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> ArrayLike:
    """
    Return a random coordinate on the surface of the cube defined by x_min/x_max,
    y_min/y_max, and z_min/z_max.

    Parameters:
        rng_face (PRNGKeyArray): Random key for sampling the face and coordinates.
        x_min, x_max, y_min, y_max, z_min, z_max (float): The cube boundaries.

    Returns:
        ArrayLike: A 3D coordinate (x, y, z) lying on the cube's surface.
    """
    face = jax.random.randint(rng_face, shape=(), minval=0, maxval=6)
    rand_vals = jax.random.uniform(rng_face, shape=(2,))

    # Determine which coordinate is fixed to min/max, others span full range
    def pick_vals(f):
        return jax.lax.switch(
            f,
            [
                lambda: (
                    x_min,
                    y_min + rand_vals[0] * (y_max - y_min),
                    z_min + rand_vals[1] * (z_max - z_min),
                ),
                lambda: (
                    x_max,
                    y_min + rand_vals[0] * (y_max - y_min),
                    z_min + rand_vals[1] * (z_max - z_min),
                ),
                lambda: (
                    x_min + rand_vals[0] * (x_max - x_min),
                    y_min,
                    z_min + rand_vals[1] * (z_max - z_min),
                ),
                lambda: (
                    x_min + rand_vals[0] * (x_max - x_min),
                    y_max,
                    z_min + rand_vals[1] * (z_max - z_min),
                ),
                lambda: (
                    x_min + rand_vals[0] * (x_max - x_min),
                    y_min + rand_vals[1] * (y_max - y_min),
                    z_min,
                ),
                lambda: (
                    x_min + rand_vals[0] * (x_max - x_min),
                    y_min + rand_vals[1] * (y_max - y_min),
                    z_max,
                ),
            ],
        )

    return pick_vals(face)


def random_surface_point_for_face(
    rng_val: PRNGKey,
    face: ArrayLike,
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
    z_min: float = 0.0,
    z_max: float = 1.0,
) -> ArrayLike:
    """
    Return a random point on the specified face of the cube.

    Parameters:
        rng_val (PRNGKeyArray): Random key for sampling.
        face (jnp.int32): Cube face index [0..5].
        x_min, x_max, y_min, y_max, z_min, z_max (float): The cube boundaries.

    Returns:
        ArrayLike: A 3D coordinate (x, y, z) on the cube's specified face.
    """
    rand_vals = jax.random.uniform(rng_val, shape=(2,))

    def pick_vals(f):
        return jax.lax.switch(
            f,
            [
                lambda: (
                    x_min,
                    y_min + rand_vals[0] * (y_max - y_min),
                    z_min + rand_vals[1] * (z_max - z_min),
                ),
                lambda: (
                    x_max,
                    y_min + rand_vals[0] * (y_max - y_min),
                    z_min + rand_vals[1] * (z_max - z_min),
                ),
                lambda: (
                    x_min + rand_vals[0] * (x_max - x_min),
                    y_min,
                    z_min + rand_vals[1] * (z_max - z_min),
                ),
                lambda: (
                    x_min + rand_vals[0] * (x_max - x_min),
                    y_max,
                    z_min + rand_vals[1] * (z_max - z_min),
                ),
                lambda: (
                    x_min + rand_vals[0] * (x_max - x_min),
                    y_min + rand_vals[1] * (y_max - y_min),
                    z_min,
                ),
                lambda: (
                    x_min + rand_vals[0] * (x_max - x_min),
                    y_min + rand_vals[1] * (y_max - y_min),
                    z_max,
                ),
            ],
        )

    return pick_vals(face)


def _fit_polynomial_coeffs(ts: ArrayLike, values: ArrayLike, degree: int) -> jnp.ndarray:
    """Solve for polynomial coefficients interpolating ``values`` at ``ts``."""
    ts_arr = jnp.asarray(ts)
    values_arr = jnp.asarray(values, dtype=ts_arr.dtype)
    # Build a square Vandermonde system and solve ``V @ coeffs = values``.
    exponents = jnp.arange(degree, -1, -1)
    vander = jnp.power(ts_arr[..., None], exponents)
    if values_arr.ndim == 1:
        return jnp.linalg.solve(vander, values_arr)

    flat_rhs = values_arr.reshape((-1, values_arr.shape[-1]))
    solve_fn = lambda rhs: jnp.linalg.solve(vander, rhs)
    coeffs_flat = jax.vmap(solve_fn)(flat_rhs)
    return coeffs_flat.reshape(values_arr.shape[:-1] + (degree + 1,))


def sample_splines(
    rng: PRNGKey,
    degree: int = 3,
    max_degree: int | None = None,
    t_min: float = 0.0,
    t_max: float = 1.0,
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
    z_min: float = 0.0,
    z_max: float = 1.0,
    p_opposite_face: float = 0.85,
) -> Spline3D:
    """
    Sample a 3D spline that starts and ends on random faces (biased to end on the opposite face).

    Parameters:
        rng (PRNGKeyArray): Random key for all sampling.
        t_min, t_max (float): Domain of the spline parameter.
        x_min, x_max, y_min, y_max, z_min, z_max (float): Boundaries for random points.
        degree (int): Polynomial degree to use for fitting each dimension.
        max_degree (int | None): Static upper bound for ``degree`` when JIT compiling.
            If ``None`` the value defaults to ``degree``. Providing a static
            ``max_degree`` keeps sampling shapes constant, enabling ``jax.jit``
            compilation while choosing ``degree`` dynamically.

    Returns:
        Spline3D: A spline instance that can be evaluated at any parameter t.
    """
    if max_degree is None:
        try:
            max_degree = int(degree)
        except TypeError as exc:
            raise TypeError(
                "max_degree must be provided as a static integer when degree is a tracer"
            ) from exc
    max_degree = int(max_degree)
    if max_degree <= 0:
        raise ValueError("max_degree must be positive")

    if isinstance(degree, (int, np.integer)) and not (1 <= int(degree) <= max_degree):
        raise ValueError("degree must be between 1 and max_degree (inclusive)")

    rng_x, rng_y, rng_z, rng_surface_start, rng_surface_end = jax.random.split(rng, 5)

    # Opposite faces array: index = face, value = opposite face
    opposite_arr = jnp.array([1, 0, 3, 2, 5, 4])

    start_face = jax.random.randint(rng_surface_start, shape=(), minval=0, maxval=6)
    rand_prob = jax.random.uniform(rng_surface_end, shape=())

    def pick_opposite():
        return opposite_arr[start_face]

    def pick_random():
        return jax.random.randint(rng_surface_end, shape=(), minval=0, maxval=6)

    end_face = jax.lax.cond(rand_prob < p_opposite_face, pick_opposite, pick_random)

    num_internal_ctrl = max_degree - 1
    x_samples = jax.nn.sigmoid(
        jax.random.normal(rng_x, shape=(num_internal_ctrl,)) * 0.5
    )
    y_samples = jax.nn.sigmoid(
        jax.random.normal(rng_y, shape=(num_internal_ctrl,)) * 0.5
    )
    z_samples = jax.nn.sigmoid(
        jax.random.normal(rng_z, shape=(num_internal_ctrl,)) * 0.5
    )

    x_start, y_start, z_start = random_surface_point_for_face(
        rng_surface_start, start_face, x_min, x_max, y_min, y_max, z_min, z_max
    )
    x_end, y_end, z_end = random_surface_point_for_face(
        rng_surface_end, end_face, x_min, x_max, y_min, y_max, z_min, z_max
    )

    start_xyz = jnp.stack((x_start, y_start, z_start), axis=-1)
    end_xyz = jnp.stack((x_end, y_end, z_end), axis=-1)
    t_bounds = jnp.array([t_min, t_max])

    def _make_branch(deg: int):
        pad_width = max_degree - deg

        def _branch(data):
            x_ctrl, y_ctrl, z_ctrl, s_xyz, e_xyz, bounds = data
            t0, t1 = bounds
            deg_internal = deg - 1
            x_mid = x_ctrl[..., :deg_internal]
            y_mid = y_ctrl[..., :deg_internal]
            z_mid = z_ctrl[..., :deg_internal]

            x_points = jnp.concatenate((s_xyz[..., 0:1], x_mid, e_xyz[..., 0:1]), axis=-1)
            y_points = jnp.concatenate((s_xyz[..., 1:2], y_mid, e_xyz[..., 1:2]), axis=-1)
            z_points = jnp.concatenate((s_xyz[..., 2:3], z_mid, e_xyz[..., 2:3]), axis=-1)

            ts = jnp.linspace(t0, t1, deg + 1)
            coeffs_x = _fit_polynomial_coeffs(ts, x_points, deg)
            coeffs_y = _fit_polynomial_coeffs(ts, y_points, deg)
            coeffs_z = _fit_polynomial_coeffs(ts, z_points, deg)
            if pad_width:
                zeros_x = jnp.zeros((pad_width,), dtype=coeffs_x.dtype)
                zeros_y = jnp.zeros((pad_width,), dtype=coeffs_y.dtype)
                zeros_z = jnp.zeros((pad_width,), dtype=coeffs_z.dtype)
                coeffs_x = jnp.concatenate((zeros_x, coeffs_x))
                coeffs_y = jnp.concatenate((zeros_y, coeffs_y))
                coeffs_z = jnp.concatenate((zeros_z, coeffs_z))
            return coeffs_x, coeffs_y, coeffs_z

        return _branch

    branches = tuple(_make_branch(int(d)) for d in range(1, max_degree + 1))

    degree_index = jnp.asarray(degree, dtype=jnp.int32) - 1
    branch_operand = (
        x_samples,
        y_samples,
        z_samples,
        start_xyz,
        end_xyz,
        t_bounds,
    )

    coeffs_x, coeffs_y, coeffs_z = jax.lax.switch(degree_index, branches, branch_operand)
    return Spline3D(coeffs_x, coeffs_y, coeffs_z)


def get_voxels_along_curve(
    curve: Curve3D,
    voxel_grid: VoxelGrid,
    t_min: float = 0.0,
    t_max: float = 1.0,
    num_samples: int = 100,
) -> ArrayLike:
    """Determine which voxels a curve passes through using JAX operations."""
    ts = jnp.linspace(t_min, t_max, num_samples)
    points = curve(ts)
    return jax.vmap(voxel_grid.get_voxel_index_for_point)(points)


def get_average_tangent_in_voxels(
    curve: Curve3D,
    voxel_grid: VoxelGrid,
    t_min: float = 0.0,
    t_max: float = 1.0,
    num_samples: int = 1000,
) -> jnp.ndarray:
    """
    Compute an average 3D tangent vector in each voxel, using VoxelGrid utilities.

    Returns:
        A 4D array of shape (Nx, Ny, Nz, 3), where Nx, Ny, Nz come from the voxel grid.
    """
    Nx, Ny, Nz = (
        voxel_grid.n_voxels_x,
        voxel_grid.n_voxels_y,
        voxel_grid.n_voxels_z,
    )
    vx, vy, vz = voxel_grid.get_voxel_size()

    # Generate sample parameters
    ts = jnp.linspace(t_min, t_max, num_samples)

    # Derivatives for tangents
    tangents = curve.tangent(ts)

    # Map each spline point to voxel indices
    points = curve(ts)
    shifted = points - jnp.array([voxel_grid.x_min, voxel_grid.y_min, voxel_grid.z_min])
    voxel_coords = shifted / jnp.array([vx, vy, vz])
    voxel_indices = jnp.floor(voxel_coords).astype(jnp.int32)

    # Scatter-add tangents and counts
    sums_init = jnp.zeros((Nx, Ny, Nz, 3), dtype=jnp.float32)
    counts_init = jnp.zeros((Nx, Ny, Nz), dtype=jnp.int32)
    sums = sums_init.at[
        voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]
    ].add(tangents)
    counts = counts_init.at[
        voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]
    ].add(1)

    # Compute averages
    counts_expanded = counts[..., None]
    avg_tangents = jnp.where(counts_expanded > 0, sums / counts_expanded, 0.0)
    return avg_tangents


def get_curve_volume_fraction_in_voxels(
    curve: Curve3D,
    grid: VoxelGrid,
    diameter: float,
    t_min: float = 0.0,
    t_max: float = 1.0,
    num_samples: int = 100,
    sub_segments: int = 5,
    cross_section_step: float | None = None,
) -> jnp.ndarray:
    """Approximate curve volume occupancy with voxel-aware cross-section sampling.

    Large diameters are handled by tiling the cross-sectional disk and distributing
    sub-segment volumes across all voxels that intersect that disk.
    """
    if diameter <= 0.0:
        shape = (
            grid.n_voxels_x,
            grid.n_voxels_y,
            grid.n_voxels_z,
        )
        return jnp.zeros(shape, dtype=jnp.float32)

    radius = diameter / 2.0
    voxel_size = np.asarray(grid.get_voxel_size(), dtype=float)
    min_step = float(np.min(voxel_size))
    step = cross_section_step or (min_step / 2.0)
    step = max(step, min_step / 4.0)

    coords = np.arange(-radius, radius + step * 0.5, step)
    if coords.size == 0:
        offsets_arr = np.zeros((0, 2), dtype=float)
    else:
        ox_grid, oy_grid = np.meshgrid(coords, coords, indexing="xy")
        mask = ox_grid * ox_grid + oy_grid * oy_grid <= radius * radius + (step * step * 0.25)
        offsets_arr = np.stack([ox_grid[mask], oy_grid[mask]], axis=-1)

    if offsets_arr.size == 0:
        offsets = [(0.0, 0.0)]
        weights = [np.pi * radius * radius]
    else:
        offsets = offsets_arr
        weights = np.full(offsets_arr.shape[0], step * step, dtype=float)
    offsets_arr = np.asarray(offsets, dtype=float)
    weights_arr = np.asarray(weights, dtype=float)
    area_total = float(weights_arr.sum())

    cross_section = np.pi * (radius**2)
    if area_total <= 0.0:
        area_total = 1.0

    finer_samples = num_samples * sub_segments
    ts = jnp.linspace(t_min, t_max, finer_samples + 1)
    points = np.asarray(curve(ts))

    diffs = points[1:] - points[:-1]
    lengths = np.linalg.norm(diffs, axis=1)
    midpoints = (points[1:] + points[:-1]) / 2.0

    grid_min = np.asarray([grid.x_min, grid.y_min, grid.z_min], dtype=float)
    Nx, Ny, Nz = grid.n_voxels_x, grid.n_voxels_y, grid.n_voxels_z
    voxel_volume = (
        (grid.x_max - grid.x_min)
        * (grid.y_max - grid.y_min)
        * (grid.z_max - grid.z_min)
        / (Nx * Ny * Nz)
    )

    volume_accum = np.zeros((Nx, Ny, Nz), dtype=np.float64)

    def orthonormal_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        aux = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(direction[0]) > 0.9:
            aux = np.array([0.0, 1.0, 0.0], dtype=float)
        u = np.cross(direction, aux)
        norm_u = np.linalg.norm(u)
        if norm_u < 1e-8:
            aux = np.array([0.0, 0.0, 1.0], dtype=float)
            u = np.cross(direction, aux)
            norm_u = np.linalg.norm(u)
        u = u / (norm_u if norm_u > 0 else 1.0)
        v = np.cross(direction, u)
        v_norm = np.linalg.norm(v)
        v = v / (v_norm if v_norm > 0 else 1.0)
        return u, v

    for seg_idx, length in enumerate(lengths):
        if length <= 0.0:
            continue
        direction = diffs[seg_idx] / length
        u_vec, v_vec = orthonormal_basis(direction)
        midpoint = midpoints[seg_idx]
        segment_volume = cross_section * length
        weight_scale = segment_volume / area_total

        for (ox, oy), weight in zip(offsets_arr, weights_arr):
            offset_vec = u_vec * ox + v_vec * oy
            sample_point = midpoint + offset_vec
            rel = (sample_point - grid_min) / voxel_size
            ix, iy, iz = np.floor(rel).astype(int)
            if 0 <= ix < Nx and 0 <= iy < Ny and 0 <= iz < Nz:
                volume_accum[ix, iy, iz] += weight * weight_scale

    fraction = volume_accum / voxel_volume
    fraction = np.clip(fraction, 0.0, 1.0)
    return jnp.asarray(fraction, dtype=jnp.float32)

def curve_points(
    curve: Curve3D,
    *,
    num_points: int = 200,
    t_min: float = 0.0,
    t_max: float = 1.0,
) -> np.ndarray:
    """Return discretized points along a curve for visualization."""
    ts = jnp.linspace(t_min, t_max, num_points)
    return np.asarray(curve(ts))


def plot_curves(
    curves: Curve3D | Sequence[Curve3D],
    *,
    num_points: int = 200,
    t_min: float = 0.0,
    t_max: float = 1.0,
    labels: Sequence[str] | None = None,
    grid: VoxelGrid | None = None,
    ax=None,
    show: bool = False,
    **plot_kwargs,
):
    """Plot one or more curves in 3D using Plotly."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - plotting helper
        raise ImportError("plot_curves requires plotly to be installed") from exc

    if isinstance(curves, Curve3D):
        curves = [curves]
    else:
        curves = list(curves)
    if not curves:
        raise ValueError("curves must contain at least one curve")
    if labels is not None and len(labels) != len(curves):
        raise ValueError("labels must match number of curves")

    fig = ax if isinstance(ax, go.Figure) else go.Figure()

    # Extract commonly used matplotlib-style arguments for compatibility.
    opacity = plot_kwargs.pop("alpha", None)
    color_override = (
        plot_kwargs.pop("color", None)
        or plot_kwargs.pop("line_color", None)
        or None
    )
    width_override = (
        plot_kwargs.pop("linewidth", None)
        or plot_kwargs.pop("line_width", None)
        or None
    )
    line_kwargs = plot_kwargs.pop("line", {})

    for idx, curve in enumerate(curves):
        pts = curve_points(
            curve, num_points=num_points, t_min=t_min, t_max=t_max
        )
        label = labels[idx] if labels is not None else None
        line_dict = {k: v for k, v in line_kwargs.items()}
        if color_override is not None:
            line_dict.setdefault("color", color_override)
        if width_override is not None:
            line_dict.setdefault("width", width_override)
        trace_kwargs = {
            "x": pts[:, 0],
            "y": pts[:, 1],
            "z": pts[:, 2],
            "mode": "lines",
            "name": label,
        }
        if line_dict:
            trace_kwargs["line"] = line_dict
        if opacity is not None:
            trace_kwargs["opacity"] = opacity
        trace_kwargs.update(plot_kwargs)
        fig.add_trace(go.Scatter3d(**trace_kwargs))

    if grid is not None:
        _add_grid_box(fig, grid)

    fig.update_layout(
        title="Curve trajectories",
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="cube",
        ),
        legend=dict(itemsizing="constant"),
    )
    if labels is None or not any(label is not None for label in labels):
        fig.update_layout(showlegend=False)
    if show:
        fig.show()
    return fig, fig


def plot_splines(*args, **kwargs):  # pragma: no cover - shim
    """Backward-compatible wrapper around :func:`plot_curves`."""
    return plot_curves(*args, **kwargs)


def voxelize_curve(
    curve: Curve3D,
    grid: VoxelGrid,
    diameter: float,
    *,
    t_min: float = 0.0,
    t_max: float = 1.0,
    volume_num_samples: int = 100,
    volume_sub_segments: int = 5,
    cross_section_step: float | None = None,
    compute_tangents: bool = True,
    tangent_num_samples: int = 100,
    metadata: dict[str, Any] | None = None,
) -> VoxelizedCurve:
    """Generate a voxelized representation for a curve."""

    volume_fraction = get_curve_volume_fraction_in_voxels(
        curve,
        grid,
        diameter,
        t_min=t_min,
        t_max=t_max,
        num_samples=volume_num_samples,
        sub_segments=volume_sub_segments,
        cross_section_step=cross_section_step,
    )
    tangents = (
        get_average_tangent_in_voxels(
            curve,
            grid,
            t_min=t_min,
            t_max=t_max,
            num_samples=tangent_num_samples,
        )
        if compute_tangents
        else None
    )
    return VoxelizedCurve(
        curve=curve,
        grid=grid,
        diameter=diameter,
        volume_fraction=volume_fraction,
        tangents=tangents,
        metadata=metadata,
    )


def _add_grid_box(
    fig,
    grid: VoxelGrid,
    *,
    color: str = "lightgray",
    alpha: float = 0.4,
    linewidth: float = 0.8,
) -> None:
    """Overlay the voxel grid bounding box on a Plotly figure."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - plotting helper
        raise ImportError("plot_curves requires plotly to be installed") from exc

    x0, x1 = grid.x_min, grid.x_max
    y0, y1 = grid.y_min, grid.y_max
    z0, z1 = grid.z_min, grid.z_max
    corners = np.array(
        [
            [x0, y0, z0],
            [x0, y0, z1],
            [x0, y1, z0],
            [x0, y1, z1],
            [x1, y0, z0],
            [x1, y0, z1],
            [x1, y1, z0],
            [x1, y1, z1],
        ]
    )
    edges = [
        (0, 1),
        (0, 2),
        (0, 4),
        (3, 1),
        (3, 2),
        (3, 7),
        (5, 1),
        (5, 4),
        (5, 7),
        (6, 2),
        (6, 4),
        (6, 7),
    ]

    xs = []
    ys = []
    zs = []
    for start_idx, end_idx in edges:
        start = corners[start_idx]
        end = corners[end_idx]
        xs.extend([start[0], end[0], None])
        ys.extend([start[1], end[1], None])
        zs.extend([start[2], end[2], None])

    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color=color, width=linewidth),
            opacity=alpha,
            name="voxel grid",
            hoverinfo="skip",
            showlegend=False,
        )
    )
