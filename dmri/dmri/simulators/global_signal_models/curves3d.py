import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax.random import PRNGKey


class Spline3D:
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


def sample_splines(
    rng: PRNGKey,
    t_min: float = 0.0,
    t_max: float = 1.0,
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
    z_min: float = 0.0,
    z_max: float = 1.0,
    degree: int = 3,
    p_opposite_face: float = 0.85,
) -> Spline3D:
    """
    Sample a 3D spline that starts and ends on random faces (biased to end on the opposite face).

    Parameters:
        rng (PRNGKeyArray): Random key for all sampling.
        t_min, t_max (float): Domain of the spline parameter.
        x_min, x_max, y_min, y_max, z_min, z_max (float): Boundaries for random points.
        degree (int): Polynomial degree to use for fitting each dimension.

    Returns:
        Spline3D: A spline instance that can be evaluated at any parameter t.
    """
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

    x_points = jax.random.logistic(
        rng_x, shape=(degree - 1,)
    ) * (x_max - x_min) / 6 + (x_max + x_min) / 2
    y_points = jax.random.logistic(
        rng_y, shape=(degree - 1,)
    ) * (y_max - y_min) / 6 + (y_max + y_min) / 2
    z_points = jax.random.logistic(
        rng_z, shape=(degree - 1,)
    ) * (z_max - z_min) / 6 + (z_max + z_min) / 2

    x_start, y_start, z_start = random_surface_point_for_face(
        rng_surface_start, start_face, x_min, x_max, y_min, y_max, z_min, z_max
    )
    x_end, y_end, z_end = random_surface_point_for_face(
        rng_surface_end, end_face, x_min, x_max, y_min, y_max, z_min, z_max
    )

    x_points = jnp.concatenate((jnp.array([x_start]), x_points, jnp.array([x_end])))
    y_points = jnp.concatenate((jnp.array([y_start]), y_points, jnp.array([y_end])))
    z_points = jnp.concatenate((jnp.array([z_start]), z_points, jnp.array([z_end])))

    spline_coeffs_x = jnp.polyfit(
        jnp.linspace(t_min, t_max, degree + 1), x_points, degree
    )
    spline_coeffs_y = jnp.polyfit(
        jnp.linspace(t_min, t_max, degree + 1), y_points, degree
    )
    spline_coeffs_z = jnp.polyfit(
        jnp.linspace(t_min, t_max, degree + 1), z_points, degree
    )

    return Spline3D(spline_coeffs_x, spline_coeffs_y, spline_coeffs_z)


def get_voxels_along_spline(
    spline: Spline3D,
    voxel_grid: VoxelGrid,
    t_min: float = 0.0,
    t_max: float = 1.0,
    num_samples: int = 100,
) -> ArrayLike:
    """
    Determine which voxels the spline passes through using only JAX operations.

    Returns:
        ArrayLike: A 2D array of shape (N, 3) containing the voxel indices (ix, iy, iz).
    """
    # Generate sample parameters
    ts = jnp.linspace(t_min, t_max, num_samples)

    # Compute spline points
    points = spline(ts)

    # Determine voxel indices for each point
    voxel_indices = jax.vmap(voxel_grid.get_voxel_index_for_point)(points)

    return voxel_indices


def get_average_tangent_in_voxels(
    spline: Spline3D,
    voxel_grid: VoxelGrid,
    t_min: float = 0.0,
    t_max: float = 1.0,
    num_samples: int = 100,
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
    dx_dt = jnp.polyval(jnp.polyder(spline.spline_coeffs_x), ts)
    dy_dt = jnp.polyval(jnp.polyder(spline.spline_coeffs_y), ts)
    dz_dt = jnp.polyval(jnp.polyder(spline.spline_coeffs_z), ts)
    tangents = jnp.stack([dx_dt, dy_dt, dz_dt], axis=-1)

    # Map each spline point to voxel indices
    points = spline(ts)
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


def get_spline_volume_fraction_in_voxels(
    spline: Spline3D,
    grid: VoxelGrid,
    diameter: float,
    t_min: float = 0.0,
    t_max: float = 1.0,
    num_samples: int = 100,
    sub_segments: int = 5,
) -> jnp.ndarray:
    """
    Approximate the fraction of volume occupied by a cylindrical spline (of given diameter)
    in each voxel by subdividing each segment further. Returns an array of shape (Nx, Ny, Nz).
    """
    # Cross-sectional area
    radius = diameter / 2.0
    cross_section = jnp.pi * (radius**2)

    # Increase sampling
    finer_samples = num_samples * sub_segments
    ts = jnp.linspace(t_min, t_max, finer_samples + 1)
    points = spline(ts)

    # Compute sub-segment lengths, midpoints
    diffs = points[1:] - points[:-1]
    seg_lengths = jnp.sqrt(jnp.sum(diffs**2, axis=-1))
    midpoints = (points[1:] + points[:-1]) / 2.0

    # Map each midpoint to a voxel, scatter-add
    voxel_indices = jax.vmap(grid.get_voxel_index_for_point)(midpoints)
    Nx, Ny, Nz = grid.n_voxels_x, grid.n_voxels_y, grid.n_voxels_z
    voxel_vol = (
        (grid.x_max - grid.x_min)
        * (grid.y_max - grid.y_min)
        * (grid.z_max - grid.z_min)
        / (Nx * Ny * Nz)
    )

    volume_init = jnp.zeros((Nx, Ny, Nz), dtype=jnp.float32)
    segment_volumes = cross_section * seg_lengths
    volume_sums = volume_init.at[
        voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]
    ].add(segment_volumes.astype(jnp.float32))

    fraction = volume_sums / voxel_vol
    fraction = jnp.clip(fraction, 0.0, 1.0)
    return fraction
