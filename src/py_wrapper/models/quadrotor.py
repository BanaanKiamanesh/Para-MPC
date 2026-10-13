"""A planar quadrotor carrying a cable-suspended load: follow a sinusoidal path with the load swing below 10 degrees
and each rotor's thrust in [0, 9] N.

State  x = (px, pz, theta, phi, px', pz', theta', phi')   position, pitch, cable angle, and their rates (nx = 8)
Input  u = (T1, T2)                                       rotor thrusts (nu = 2)

    ocp = quadrotor.ocp(N=30)                              # the NMPC problem (an OCP)
    ctrl = parampc.NMPC(ocp, **quadrotor.NMPC_OPTIONS)
    xs, us, infos = ctrl.closed_loop(quadrotor.plant(K), quadrotor.initial_state(), K, params_fn=...)

The prediction model is one explicit-midpoint step per sample; the plant is RK4 with 5 substeps, a lateral wind
gust and measurement noise, so the controller never sees its own model.
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from ..integrators import midpoint
from ..nmpc import OCP

# physics
M_QUAD, M_LOAD, L_CABLE, I_YY, ARM, GRAV = 1.00, 0.25, 0.60, 0.012, 0.16, 9.81
M_TOT = M_QUAD + M_LOAD
DT = 0.05
NX, NU = 8, 2

# limits
T_MAX = 9.0                   # hard: 0 <= T_i <= T_MAX
PHI_MAX = np.deg2rad(10.0)    # soft: |phi|  <= PHI_MAX
PHID_MAX = 1.20               # soft: |phi'| <= PHID_MAX
T_HOVER = M_TOT * GRAV / 2.0
U_HOVER = jnp.array([T_HOVER, T_HOVER])

# task and weights
AMP, PERIOD, Z_REF = 1.5, 6.0, 1.50
Q_DIAG = jnp.array([60., 60., 1., 5., 2., 2., .5, .5])
Q_TERM = 5.0 * Q_DIAG
R_DIAG = jnp.array([0.02, 0.02])
SOFT_L1 = 5000.0              # l1 penalty of the swing slacks (an exact penalty)
SOFT_L2 = 1e-2                # small quadratic penalty that keeps the QP's input Hessian positive definite

# the cost is a convex quadratic, so convexifying its Hessian would be a no-op (bit for bit): skip the check
NMPC_OPTIONS = dict(sqp_iters=3, lm=0.5, merit_weight=500.0, qp_tol=1e-6, hessian='exact')
WIND = dict(start=8.0, end=12.0, force=1.8)       # the closed loop's lateral gust [s, s, N]
NOISE = 1e-3                                      # measurement noise (standard deviation)


def reference(t):
    """px_ref = 1.5 sin(2 pi t / 6) at the constant height 1.5 m."""
    w = 2.0 * jnp.pi / PERIOD
    return jnp.array([AMP * jnp.sin(w * t), Z_REF, 0., 0., AMP * w * jnp.cos(w * t), 0., 0., 0.])


def horizon_reference(tk, N):
    """The reference over the horizon from time tk: (N+1, nx), the OCP's params."""
    return jax.vmap(reference)(tk + jnp.arange(N + 1) * DT)


def initial_state():
    return jnp.array([0., Z_REF, 0., 0., 0., 0., 0., 0.])


def dynamics(x, u, wind=0.0):
    """Continuous-time Euler-Lagrange dynamics x' = f(x, u), u = (T1, T2), with an optional lateral force."""
    th, ph, phid = x[2], x[3], x[7]
    f = u[0] + u[1]
    tau = (u[0] - u[1]) * ARM
    c, sn = jnp.cos(ph), jnp.sin(ph)
    mass = jnp.array([[M_TOT, 0., 0., M_LOAD * L_CABLE * c],
                      [0., M_TOT, 0., M_LOAD * L_CABLE * sn],
                      [0., 0., I_YY, 0.],
                      [c, sn, 0., L_CABLE]])
    force = jnp.array([-f * jnp.sin(th) + M_LOAD * L_CABLE * sn * phid**2 + wind,
                       f * jnp.cos(th) - M_TOT * GRAV - M_LOAD * L_CABLE * c * phid**2,
                       tau,
                       -GRAV * sn])
    qdd = jnp.linalg.solve(mass, force)
    return jnp.array([x[4], x[5], x[6], x[7], qdd[0], qdd[1], qdd[2], qdd[3]])


def ocp(N=30):
    """The tracking NMPC problem: params p_t = horizon_reference(...)[t]; hard thrust bounds; |phi| and |phi'| soft,
    one slack per pair of rows; hover thrust as the initial guess (zero thrust would free-fall)."""
    return OCP(
        nx=NX, nu=NU, N=N,
        dynamics=midpoint(lambda x, u, p: dynamics(x, u), DT),
        stage_cost=lambda x, u, p: 0.5 * jnp.sum(Q_DIAG * (x - p) ** 2) + 0.5 * jnp.sum(R_DIAG * (u - U_HOVER) ** 2),
        terminal_cost=lambda x, p: 0.5 * jnp.sum(Q_TERM * (x - p) ** 2),
        u_min=0.0, u_max=T_MAX,
        constraints=lambda x, u, p: jnp.array([x[3] - PHI_MAX, -x[3] - PHI_MAX, x[7] - PHID_MAX, -x[7] - PHID_MAX]),
        soft=[[0, 1], [2, 3]], soft_l1=SOFT_L1, soft_l2=SOFT_L2, u_guess=U_HOVER)


def step_plant(x, u, wind=0.0, substeps=5):
    """The 'true' plant over one sample: RK4 with DT / substeps steps and an optional lateral wind force."""
    h = DT / substeps

    def rk4(x, _):
        k1 = dynamics(x, u, wind)
        k2 = dynamics(x + 0.5 * h * k1, u, wind)
        k3 = dynamics(x + 0.5 * h * k2, u, wind)
        k4 = dynamics(x + h * k3, u, wind)
        return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), None

    return lax.scan(rk4, x, None, length=substeps)[0]


def plant(K, key=None, noise=NOISE, wind=WIND):
    """The closed loop's plant(x, u, k) for K steps: step_plant with the wind gust over [wind['start'], wind['end'])
    and Gaussian measurement noise (one key per step, from `key`, default PRNGKey(0))."""
    keys = jax.random.split(jax.random.PRNGKey(0) if key is None else key, K)

    def f(x, u, k):
        tk = k * DT
        force = jnp.where((tk >= wind['start']) & (tk < wind['end']), wind['force'], 0.0)
        return step_plant(x, u, force) + noise * jax.random.normal(keys[k], (NX,))
    return f
