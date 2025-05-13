import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from dipy.data import get_sphere
from jax.typing import ArrayLike

from dmri.utils.dmriutils import cartesian_to_unitsphere, unitsphere_to_cartesian

sphere_default = get_sphere(name="symmetric724")


def orthoview(data, x_slice=None, y_slice=None, z_slice=None, vmin=None, vmax=None):

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    if x_slice is None:
        x_slice = data.shape[0] // 2
    if y_slice is None:
        y_slice = data.shape[1] // 2
    if z_slice is None:
        z_slice = data.shape[2] // 2

    if vmin is None:
        vmin = np.min(data)
    if vmax is None:
        vmax = np.max(data)



    axs[0].imshow(data[x_slice, :, :].T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axs[1].imshow(data[:, y_slice, :].T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axs[2].imshow(data[:, :, z_slice].T, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)

    # Disable axis for all subplots
    for ax in axs:
        ax.axis('off')

    # Background color is black
    fig.patch.set_facecolor('black')

    return fig, axs





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
