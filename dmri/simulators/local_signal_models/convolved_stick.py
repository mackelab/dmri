from dmri.simulators.convolved_models import ConvolvedSignalCompartment
from dmri.simulators.local_signal_models.signal_response_kernels import StickKernel
from dmri.simulators.sphereical_distributions import Bingham, Watson


class WatsonStick(ConvolvedSignalCompartment):
    """Watson distribution with stick kernel."""

    fod_type = Watson
    signal_kernel_type = StickKernel


class BinghamStick(ConvolvedSignalCompartment):
    """Bingham distribution with stick kernel."""

    fod_type = Bingham
    signal_kernel_type = StickKernel
