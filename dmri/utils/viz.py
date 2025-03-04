import matplotlib.pyplot as plt
import numpy as np

from jax.typing import ArrayLike


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
