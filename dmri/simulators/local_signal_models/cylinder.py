import math

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from dmri.simulators.base import SignalCompartment
from dmri.utils.dmriutils import unitsphere_to_cartesian

# Constants
gamma = 2.6752218744 * 1e8  # [sec]^-1 * [T]^-1
gamma_ms = gamma * 1e-3  # [ms]^-1 *[T]^-1
MINIMUM_RADIUS = 1e-10  # to avoid numerical issues

# First 60 roots of j'1(am*x)=0 from Camino source
am = np.array(
    [
        1.84118307861360,
        5.33144196877749,
        8.53631578218074,
        11.7060038949077,
        14.8635881488839,
        18.0155278304879,
        21.1643671187891,
        24.3113254834588,
        27.4570501848623,
        30.6019229722078,
        33.7461812269726,
        36.8899866873805,
        40.0334439409610,
        43.1766274212415,
        46.3195966792621,
        49.4623908440429,
        52.6050411092602,
        55.7475709551533,
        58.8900018651876,
        62.0323477967829,
        65.1746202084584,
        68.3168306640438,
        71.4589869258787,
        74.6010956133729,
        77.7431620631416,
        80.8851921057280,
        84.0271895462953,
        87.1691575709855,
        90.3110993488875,
        93.4530179063458,
        96.5949155953313,
        99.7367932203820,
        102.878653768715,
        106.020498619541,
        109.162329055405,
        112.304145672561,
        115.445950418834,
        118.587744574512,
        121.729527118091,
        124.871300497614,
        128.013065217171,
        131.154821965250,
        134.296570328107,
        137.438311926144,
        140.580047659913,
        143.721775748727,
        146.863498476739,
        150.005215971725,
        153.146928691331,
        156.288635801966,
        159.430338769213,
        162.572038308643,
        165.713732347338,
        168.855423073845,
        171.997111729391,
        175.138794734935,
        178.280475036977,
        181.422152668422,
        184.563828222242,
        187.705499575101,
    ]
)


def compute_GPDsum(am_r, pulse_duration, diffusion_time, diffusivity, radius):
    """Compute the GPD sum for perpendicular diffusion."""
    dam = diffusivity * am_r * am_r
    e11 = -dam * pulse_duration
    e2 = -dam * diffusion_time
    e3 = e2 - e11
    e4 = e2 + e11

    nom = (
        -2 * e11
        - 2
        + (2 * jnp.exp(e11))
        + (2 * jnp.exp(e2))
        - jnp.exp(e3)
        - jnp.exp(e4)
    )
    denom = dam**2 * am_r**2 * (radius**2 * am_r**2 - 1)
    return jnp.sum(nom / denom, axis=0)


class Cylinder(SignalCompartment):
    r"""
    The Stejskal-Tanner approximation of the cylinder model with finite
    radius. Assumes finite pulse duration and diffusion time.
    """

    theta_dim = 4
    lam_par_max: float = 0.1
    radius_mean = math.log(0.01)
    radius_scale = 0.5

    def __init__(self, mu: ArrayLike, lam_par: float, radius: float):
        self.mu = mu
        self.lam_par = lam_par
        self.radius = radius

    @classmethod
    def log_signal_fn(
        cls,
        acq,
        mu: ArrayLike,
        lam_par: float,
        radius: float,
        rng=None,
    ):
        """Compute the log signal attenuation."""
        q = acq.qvals
        bvecs = acq.bvecs
        pulse_duration = acq.pulse_duration
        diffusion_time = acq.diffusion_time
        G = acq.G

        # Convert units
        G_T_per_micron = G * 1e-3 * 1e-6  # [T] * [um]^-1
        radius_clip = jnp.clip(radius, MINIMUM_RADIUS, None)

        # Compute perpendicular diffusion
        am_r = am[:, jnp.newaxis] / radius_clip
        GPDsum = compute_GPDsum(
            am_r, pulse_duration, diffusion_time, lam_par, radius_clip
        )

        # Get orientation components
        mu_cart = unitsphere_to_cartesian(mu)
        c2 = jnp.sum(bvecs * mu_cart, axis=1) ** 2  # cosine^2
        s2 = 1 - c2  # sine^2

        # Compute signal components
        log_att_perp = -2.0 * gamma_ms**2 * G_T_per_micron**2 * GPDsum * s2
        log_att_para = -(q**2) * lam_par * c2

        return log_att_perp + log_att_para

    @classmethod
    def to_theta(cls, mu: ArrayLike, lam_par: float, radius: float) -> ArrayLike:
        """Convert parameters to the parameter space theta."""
        lambda_par_theta = jax.scipy.stats.norm.ppf(lam_par / cls.lam_par_max)
        radius_theta = (jnp.log(radius) - cls.radius_mean) / cls.radius_scale
        theta_mu0 = jax.scipy.stats.norm.ppf(mu[0] / jnp.pi)
        theta_mu1 = jax.scipy.stats.norm.ppf((mu[1] + jnp.pi) / (2 * jnp.pi))
        return jnp.array([lambda_par_theta, radius_theta, theta_mu0, theta_mu1])

    @classmethod
    def to_params(cls, theta: ArrayLike) -> tuple:
        """Convert the parameter space theta to the model parameters."""
        lambda_par = jax.scipy.stats.norm.cdf(theta[0]) * cls.lam_par_max
        radius = jnp.exp(theta[1] * cls.radius_scale + cls.radius_mean)
        theta_mu = jax.scipy.stats.norm.cdf(theta[2:])
        mu0 = theta_mu[0] * jnp.pi
        mu1 = theta_mu[1] * 2 * jnp.pi - jnp.pi
        mu = jnp.array([mu0, mu1])
        return mu, lambda_par, radius
