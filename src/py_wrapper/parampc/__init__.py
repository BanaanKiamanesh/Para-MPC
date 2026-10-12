"""ParaMPC: parallel-in-horizon model predictive control.

An interior-point solver for MPC whose Newton steps are associative scans over the horizon: Theta(log N) depth
instead of the Riccati recursion's Theta(N). It runs in JAX (CPU or GPU) or as a single CUDA kernel.

    backends        available_backends

Subpackages: parampc.core (the method), parampc.cuda (the CUDA kernel's wrapper), parampc.testing (dense references
for checking solutions).
"""
import jax

jax.config.update('jax_enable_x64', True)   # the cost-to-go recursion needs float64

from . import core, testing  # noqa: E402
from .backends import available_backends  # noqa: E402

__version__ = '0.1.0'
__all__ = ['available_backends', 'core', 'testing']
