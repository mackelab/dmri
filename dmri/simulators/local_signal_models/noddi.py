from dmri.simulators.convolved_models import ConvolvedSignalCompartment
from dmri.simulators.local_signal_models.signal_response_kernels import NODDIKernel
from dmri.simulators.sphereical_distributions import Bingham, Watson


class NoddiW(ConvolvedSignalCompartment):
    """Watson distribution with NODDI kernel."""

    fod_type = Watson
    signal_kernel_type = NODDIKernel


class NoddiB(ConvolvedSignalCompartment):
    """Bingham distribution with NODDI kernel."""

    fod_type = Bingham
    signal_kernel_type = NODDIKernel
