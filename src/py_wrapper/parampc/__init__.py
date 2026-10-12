"""ParaMPC: parallel-in-horizon model predictive control.

An interior-point solver for MPC whose Newton steps are associative scans over the horizon: Theta(log N) depth
instead of the Riccati recursion's Theta(N). It runs in JAX (CPU or GPU) or as a single CUDA kernel, and solves
linear MPC (QPs) and nonlinear MPC (SQP or the real-time iteration).

    linear MPC      OCPQP, make_qp, add_bounds, soften, solve_qp, Solution, kkt_score, STATUS
    nonlinear MPC   OCP, NMPC, State, euler, midpoint, rk4
    backends        available_backends

Subpackages: parampc.core (the method), parampc.cuda (the CUDA kernel's wrapper), parampc.models (example
problems), parampc.testing (dense references for checking solutions).
"""
import jax

jax.config.update('jax_enable_x64', True)   # the cost-to-go recursion needs float64

from . import core, models, testing  # noqa: E402
from .backends import available_backends  # noqa: E402
from .integrators import euler, midpoint, rk4  # noqa: E402
from .nmpc import NMPC, OCP, State  # noqa: E402
from .qp import OCPQP, STATUS, Solution, add_bounds, kkt_score, make_qp, soften  # noqa: E402
from .solver import solve_qp  # noqa: E402

__version__ = '0.1.0'
__all__ = ['NMPC', 'OCP', 'OCPQP', 'STATUS', 'Solution', 'State', 'add_bounds', 'available_backends', 'core',
           'euler', 'kkt_score', 'make_qp', 'midpoint', 'models', 'rk4', 'soften', 'solve_qp', 'testing']
