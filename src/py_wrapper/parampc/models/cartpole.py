"""Cart-pole swing-up: the pole starts hanging and is swung up to upright and balanced, with |force| <= 15 N and the
cart on a 3 m track. The cost 1 - cos(theta) is non-convex (concave around the hanging position), so the NMPC
convexifies its Hessian (NMPC(hessian='convexify'), the default).

State  x = (cart position, pole angle from upright, their rates)    nx = 4
Input  u = (force,)                                                 nu = 1
"""
import jax.numpy as jnp
import numpy as np

from ..integrators import rk4
from ..nmpc import OCP

M_CART, M_POLE, LENGTH, GRAV = 1.0, 0.1, 0.5, 9.81
DT = 0.05
U_MAX, X_MAX = 15.0, 1.5
NMPC_OPTIONS = dict(sqp_iters=3, lm=0.1)


def dynamics(x, u, p=None):
    """Continuous-time cart-pole dynamics."""
    pos, th, v, w = x
    s, c = jnp.sin(th), jnp.cos(th)
    den = M_CART + M_POLE * s ** 2
    acc = (u[0] - M_POLE * s * (LENGTH * w ** 2 - GRAV * c)) / den
    alpha = (u[0] * c - M_POLE * LENGTH * w ** 2 * c * s + (M_CART + M_POLE) * GRAV * s) / (LENGTH * den)
    return jnp.array([v, w, acc, alpha])


def state_cost(x):
    """2 pos^2 + 10 (1 - cos theta) + 0.1 (v^2 + w^2): zero only at the upright equilibrium."""
    return 2.0 * x[0] ** 2 + 10.0 * (1 - jnp.cos(x[1])) + 0.1 * (x[2] ** 2 + x[3] ** 2)


def ocp(N=60, dt=DT):
    """The swing-up problem: RK4 prediction model, terminal cost 10 x the state cost, force and track limits."""
    return OCP(nx=4, nu=1, N=N, dynamics=rk4(dynamics, dt),
               stage_cost=lambda x, u, p: state_cost(x) + 0.01 * u[0] ** 2,
               terminal_cost=lambda x, p: 10.0 * state_cost(x),
               u_min=-U_MAX, u_max=U_MAX, x_min=[-X_MAX, -np.inf, -np.inf, -np.inf],
               x_max=[X_MAX, np.inf, np.inf, np.inf])


def plant(dt=DT, substeps=4):
    """The closed loop's plant(x, u, k): RK4 with dt / substeps steps."""
    step = rk4(dynamics, dt, substeps=substeps)
    return lambda x, u, k: step(x, u, None)


def hanging(offset=0.05):
    """The hanging initial state, the pole `offset` rad from straight down."""
    return jnp.array([0.0, np.pi - offset, 0.0, 0.0])


def wrap(theta):
    """Angles wrapped to [-pi, pi): the distance from upright."""
    return (np.asarray(theta) + np.pi) % (2 * np.pi) - np.pi
