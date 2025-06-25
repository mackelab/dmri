from dmri.simulators.convolved_models import ConvolvedSignalCompartment
from dmri.simulators.local_signal_models.signal_response_kernels import (
    SimpleSANDIKernel,
)
from dmri.simulators.sphereical_distributions import Watson, Bingham


class SandiW(ConvolvedSignalCompartment):
    """Watson distribution with SANDI kernel."""

    fod_type = Watson
    signal_kernel_type = SimpleSANDIKernel


class SandiB(ConvolvedSignalCompartment):
    """Bingham distribution with SANDI kernel."""

    fod_type = Bingham
    signal_kernel_type = SimpleSANDIKernel
