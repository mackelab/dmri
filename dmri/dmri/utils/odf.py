# NOTE: We can directly derive our fODE approximation from posterior samples
# NOTE: We might can benchmark against deconvolution based methods ?

# NOTE: This requires to define P(n|\theta) i.e. the likelihood of a orientation
# for a given model.

import numpy as np


class ODF:
    pass


def general_odf(dirs, evals, evecs):
    """
    Compute the Orientation Distribution Function (ODF) for a general three‐dimensional
    Gaussian diffusion process modeled by a single diffusion tensor.

    The diffusion propagator is given by

    .. math::

        p(\mathbf{r}) = \frac{1}{\sqrt{(4\pi\tau)^3\,\det(D)}}
        \exp\!\left[-\frac{1}{4\tau}\,\mathbf{r}^T D^{-1}\mathbf{r}\right],

    where the diffusion tensor is constructed from its eigenvalues and eigenvectors:

    .. math::

        D = R\,\mathrm{diag}(\lambda_1,\lambda_2,\lambda_3)\,R^T.

    The Orientation Distribution Function (ODF) is defined as the radial integral of the
    propagator:

    .. math::

        \text{ODF}(\mathbf{u}) = \int_0^\infty p(r\,\mathbf{u})\,r^2\,dr,

    where :math:`\mathbf{u}` is a unit direction vector. One can show that, after performing
    the radial integration (and assuming that the diffusion time :math:`\tau` has been absorbed
    into the eigenvalues), the ODF can be written as

    .. math::

        \text{ODF}(\mathbf{u}) = \frac{1}{4\pi\,\sqrt{\lambda_1\lambda_2\lambda_3}\,
        \left(\mathbf{u}^T D^{-1}\mathbf{u}\right)^{3/2}}.

    Parameters
    ----------
    dirs : ndarray, shape (N, 3)
        Array of unit vectors (directions) at which to evaluate the ODF.
    evals : array-like, shape (3,)
        The eigenvalues (typically all positive) of the diffusion tensor.
    evecs : array-like, shape (3, 3)
        The eigenvectors of the diffusion tensor (each column is an eigenvector).

    Returns
    -------
    odf : ndarray, shape (N,)
        The ODF evaluated at each direction in `dirs`.
    """
    # Build the diffusion tensor D from the eigenvalues and eigenvectors.
    R = np.asarray(evecs)
    D = R @ np.diag(evals) @ R.T

    # Invert the tensor.
    D_inv = np.linalg.inv(D)

    # The normalization factor involves the square root of the product of the eigenvalues,
    # which is equivalent to sqrt(det(D)) when D is diagonal.
    det_factor = np.sqrt(np.prod(evals))

    # For each measurement direction u (a row in 'dirs'), compute the quadratic form:
    #   Q(u) = u^T D^{-1} u
    quad = np.sum(dirs @ D_inv * dirs, axis=1)

    # The ODF is then given by:
    #   ODF(u) = 1 / (4*pi * sqrt(det(D)) * Q(u)^(3/2))
    odf = 1.0 / (4 * np.pi * det_factor * (quad ** (1.5)))
    return odf


def one_dimensional_odf(dirs, lambda_val, v, tau=1.0, tol=1e-6):
    """
    Compute the Orientation Distribution Function (ODF) for a one‐dimensional (stick model)
    diffusion process.

    In the stick model the diffusion tensor is degenerate and takes the form

    .. math::

        D = \lambda\,\mathbf{v}\mathbf{v}^T,

    with eigenvalues :math:`\lambda, 0, 0`, meaning that diffusion occurs only along the
    unit vector :math:`\mathbf{v}`.

    The diffusion propagator in this case is given by

    .. math::

        p(\mathbf{r}) = \frac{1}{\sqrt{4\pi\lambda\tau}}
        \exp\!\left[-\frac{r_\parallel^2}{4\lambda\tau}\right]
        \delta(\mathbf{r}_\perp),

    where :math:`r_\parallel = \mathbf{v}^T\mathbf{r}` is the displacement along the fiber
    and :math:`\mathbf{r}_\perp` represents the perpendicular components. Because of the
    Dirac delta function :math:`\delta(\mathbf{r}_\perp)`, only displacements along :math:`\mathbf{v}`
    contribute.

    The ODF is defined as

    .. math::

        \text{ODF}(\mathbf{u}) = \int_0^\infty p(r\,\mathbf{u})\,r^2\,dr.

    Since the propagator is nonzero only when :math:`\mathbf{u}` is collinear with :math:`\mathbf{v}`,
    one finds that

    .. math::

        \text{ODF}(\pm\mathbf{v}) = \lambda \tau,

    and for any other direction the ODF is zero.

    Parameters
    ----------
    dirs : ndarray, shape (N, 3)
        Array of unit vectors (directions) at which to evaluate the ODF.
    lambda_val : float
        Diffusivity along the fiber (stick) direction.
    v : array-like, shape (3,)
        Unit vector representing the principal (fiber) diffusion direction.
    tau : float, optional
        Diffusion time (default is 1.0).
    tol : float, optional
        Tolerance for determining collinearity (default is 1e-6).

    Returns
    -------
    odf : ndarray, shape (N,)
        The ODF evaluated at each direction in `dirs`. It will have the value
        :math:`\lambda\tau` when :math:`\mathbf{u}` is (approximately) collinear with `v`
        (or its opposite), and 0 for all other directions.
    """
    dirs = np.asarray(dirs)
    v = np.asarray(v)
    # Check for collinearity by computing the absolute dot product.
    # A dot product of 1 (within tolerance) means the vectors are collinear.
    dot_prod = np.abs(np.dot(dirs, v))
    odf = np.zeros(len(dirs))
    aligned = np.abs(dot_prod - 1) < tol
    odf[aligned] = lambda_val * tau
    return odf


def two_dimensional_odf(dirs, evals2d, evecs, tau=1.0, tol=1e-6):
    """
    Compute the Orientation Distribution Function (ODF) for a two‐dimensional diffusion process.

    In a 2D diffusion process, diffusion is confined to a plane. This can be modeled by a
    degenerate diffusion tensor with two nonzero eigenvalues and one zero eigenvalue. The tensor
    can be written as:

    .. math::

        D = R\,\mathrm{diag}(\lambda_1,\lambda_2, 0)\,R^T,

    where :math:`\lambda_1` and :math:`\lambda_2` are the in‐plane diffusivities, and `R` is a
    3×3 rotation matrix. Its first two columns span the diffusion plane and its third column is the
    normal vector :math:`\mathbf{n}` to that plane.

    The 2D diffusion propagator (restricted to the plane) is given by

    .. math::

        p(\mathbf{r}) = \frac{1}{(4\pi\tau)\sqrt{\lambda_1\lambda_2}}
        \exp\!\left[-\frac{1}{4\tau}\,\mathbf{r}_{||}^T D_{2d}^{-1}\mathbf{r}_{||}\right]\,
        \delta(r_\perp),

    where :math:`\mathbf{r}_{||}` is the displacement in the plane and
    :math:`r_\perp` is the perpendicular component.

    The ODF is defined as the radial integral in the plane:

    .. math::

        \text{ODF}(\mathbf{u}) = \int_0^\infty p(r\,\mathbf{u})\,r\,dr,

    where the 2D measure is :math:`r\,dr` (instead of :math:`r^2dr` in 3D). For directions
    :math:`\mathbf{u}` that lie in the plane, if we define

    .. math::

        Q(\mathbf{u}) = \mathbf{u}_{2d}^T\,\mathrm{diag}(1/\lambda_1, 1/\lambda_2)\,\mathbf{u}_{2d},

    with :math:`\mathbf{u}_{2d}` the representation of :math:`\mathbf{u}` in the plane basis,
    one obtains the integral

    .. math::

        \int_0^\infty \exp\!\left[-\frac{r^2}{4\tau}Q(\mathbf{u})\right]\,r\,dr
        = \frac{2\tau}{Q(\mathbf{u})}.

    Therefore, the ODF for directions in the plane is

    .. math::

        \text{ODF}(\mathbf{u}) = \frac{1}{2\pi\sqrt{\lambda_1\lambda_2}}\,
        \frac{1}{Q(\mathbf{u})}.

    For any direction :math:`\mathbf{u}` that is not in the diffusion plane (i.e. not orthogonal to
    the normal :math:`\mathbf{n}`), the delta function forces the propagator (and thus the ODF) to be zero.

    Parameters
    ----------
    dirs : ndarray, shape (N, 3)
        Array of unit vectors (directions) at which to evaluate the ODF.
    evals2d : array-like, shape (2,)
        The two nonzero eigenvalues (in‐plane diffusivities) of the diffusion tensor.
    evecs : ndarray, shape (3, 3)
        A rotation matrix whose first two columns span the diffusion plane and whose third column
        is the normal vector to the plane.
    tau : float, optional
        Diffusion time (default is 1.0).
    tol : float, optional
        Tolerance for determining whether a direction lies in the diffusion plane (default is 1e-6).

    Returns
    -------
    odf : ndarray, shape (N,)
        The ODF evaluated at each direction in `dirs`. For directions that lie in the diffusion plane,
        the ODF is given by the derived formula. For directions outside the plane, the ODF is zero.
    """
    # Ensure inputs are arrays.
    dirs = np.asarray(dirs)
    evals2d = np.asarray(evals2d)
    evecs = np.asarray(evecs)

    # The plane basis: the first two columns of evecs span the plane.
    B = evecs[:, :2]  # shape (3,2)
    # The plane normal: the third column of evecs.
    n = evecs[:, 2]

    # For each direction, check if it lies in the plane (i.e., its dot product with n is near zero)
    dot_with_normal = np.abs(np.dot(dirs, n))
    odf = np.zeros(len(dirs))

    # Indices of directions that are in the plane (within tolerance)
    in_plane = dot_with_normal < tol

    if np.any(in_plane):
        # For directions in the plane, project them into the 2D basis.
        dirs_in_plane = dirs[in_plane]  # shape (M, 3)
        # Compute 2D coordinates in the plane basis:
        coords = dirs_in_plane @ B  # shape (M, 2)
        # The quadratic form is:
        #   Q(u) = u_1^2/lambda1 + u_2^2/lambda2.
        Q = (coords[:, 0] ** 2) / evals2d[0] + (coords[:, 1] ** 2) / evals2d[1]
        # Then the ODF is given by:
        odf[in_plane] = 1.0 / (2 * np.pi * np.sqrt(evals2d[0] * evals2d[1]) * Q)

    # Directions not in the plane remain zero.
    return odf

