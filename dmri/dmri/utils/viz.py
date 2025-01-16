import matplotlib.pyplot as plt
import numpy as np


def plot_odf(odf_callable, sphere, norm=False, colormap="viridis"):
    """
    Plot an ODF on the sphere in 3D.

    Parameters
    ----------
    odf_callable : callable
        A function that returns the ODF value for a given direction.
    sphere : Sphere
        The sphere.
    scale : float, optional
        The scale of the ODF. Default: 1.0.
    norm : bool, optional
        If True, the ODF will be normalized. Default: False.
    colormap : str, optional
        The colormap. Default: 'plasma'.
    """
    # Compute ODF values using the callable for each direction
    # Get the angles of the vertices
    phi = sphere.phi
    theta = sphere.theta
    odf_values = odf_callable(phi, theta)
    if norm:
        odf_values = odf_values / odf_values.max()
    odf_values = odf_values

    face_odf_values = np.mean(odf_values[sphere.faces], axis=1)

    # Plotting the ODF on the sphere
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(
        sphere.vertices[:, 0],
        sphere.vertices[:, 1],
        sphere.vertices[:, 2],
        triangles=sphere.faces,
        cmap=colormap,
        antialiased=False,
        linewidth=0.2,
        facecolors=plt.cm.get_cmap(colormap)(face_odf_values),
    )
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    plt.axis("off")
    plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.scatter(
    #     sphere.vertices[:, 0],
    #     sphere.vertices[:, 1],
    #     sphere.vertices[:, 2],
    #     c=odf_values,
    #     cmap=colormap,
    #     s=50,
    # )
    # ax.set_xlim([-1, 1])
    # ax.set_ylim([-1, 1])
    # ax.set_zlim([-1, 1])
    # ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    # plt.axis("off")

    return fig, ax
