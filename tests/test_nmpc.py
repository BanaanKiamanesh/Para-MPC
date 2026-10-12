"""Nonlinear MPC (parampc.NMPC): the quadrotor's open-loop optimum and closed loop, convexification, the cart-pole
swing-up, the real-time iteration, and batches."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import parampc as pm
from parampc.models import cartpole
from parampc.models import quadrotor as quad

N = 30


@pytest.fixture(scope='module')
def quad_ctrl():
    return pm.NMPC(quad.ocp(N), backend='sweep', **quad.NMPC_OPTIONS)


def test_quadrotor_open_loop_optimum(quad_ctrl):
    """39 SQP iterations from the hover guess reach the optimum (277.4224, reproduced independently by acados)."""
    x0, SR = quad.initial_state(), quad.horizon_reference(0.0, N)
    state = quad_ctrl.init(x0, SR)
    for _ in range(13):
        state, info = quad_ctrl.solve(state, x0, SR)
    assert abs(float(info['cost']) - 277.4224) < 1e-2
    assert abs(float(info['merit']) - float(info['cost'])) < 1e-3      # dynamics defect and violation ~ 0


def test_quadrotor_closed_loop(quad_ctrl):
    """20 s with a wind gust and sensor noise: tracking RMS 0.2141 m, the hard thrust limits hold exactly, the soft
    swing limit within 5%."""
    x0 = quad.initial_state()
    params_fn = lambda k: quad.horizon_reference(k * quad.DT, N)
    xs, us, _ = quad_ctrl.closed_loop(quad.plant(400), x0, 400, params_fn=params_fn)
    refs = jax.vmap(lambda k: quad.reference(k * quad.DT)[:2])(jnp.arange(400))
    rms = float(jnp.sqrt(jnp.mean(jnp.sum((xs[:, :2] - refs) ** 2, axis=1))))
    assert abs(rms - 0.21412) < 1e-4, rms
    assert float(us.min()) >= -1e-9 and float(us.max()) <= quad.T_MAX + 1e-9
    assert float(jnp.rad2deg(jnp.max(jnp.abs(xs[:, 3])))) < 10.5


def test_convexify_is_exact_noop_on_convex_costs():
    """The quadrotor's cost is convex: hessian='convexify' leaves its QPs unchanged, bit for bit (which is why its
    NMPC_OPTIONS skip the check with hessian='exact')."""
    convex = pm.NMPC(quad.ocp(N), lm=0.5, backend='sweep', hessian='convexify')
    exact = pm.NMPC(quad.ocp(N), lm=0.5, backend='sweep', hessian='exact')
    x0, SR = quad.initial_state(), quad.horizon_reference(0.0, N)
    state = convex.init(x0, SR)
    a, b = convex.qp(state.X, state.U, SR, x0), exact.qp(state.X, state.U, SR, x0)
    for f in ('Q', 'S', 'R', 'q', 'r', 'QN', 'qN'):
        np.testing.assert_array_equal(getattr(a, f), getattr(b, f))


def test_convexify_lifts_negative_curvature():
    """The cart-pole's cost 1 - cos(theta) is concave near the hanging position: the QP's Hessians become
    Hc + V max(-lambda, 0) V' + lm I (minimum eigenvalue lm), the identity where Hc is already convex."""
    ocp = cartpole.ocp(N=10)
    ctrl = pm.NMPC(ocp, lm=0.1, backend='sweep')
    X = jnp.zeros((11, 4)).at[:, 1].set(jnp.linspace(np.pi, 0.0, 11))          # from hanging to upright
    sub = ctrl.qp(X, jnp.zeros((10, 1)), None, X[0])
    Z = jnp.concatenate([X[:-1], jnp.zeros((10, 1))], 1)
    Hc = jax.vmap(jax.hessian(lambda z: ocp.stage_cost(z[:4], z[4:], None)))(Z)
    H = jnp.concatenate([jnp.concatenate([sub.Q, jnp.swapaxes(sub.S, 1, 2)], 2), jnp.concatenate([sub.S, sub.R], 2)], 1)
    lam_c, V = np.linalg.eigh(np.asarray(Hc))
    expect = np.einsum('tij,tj,tkj->tik', V, np.maximum(lam_c, 0.0), V) + 0.1 * np.eye(5)
    np.testing.assert_allclose(H, expect, atol=1e-10)
    assert np.min(lam_c) < -1.0                                                  # some stages were non-convex


def test_cartpole_swingup():
    ctrl = pm.NMPC(cartpole.ocp(), backend='sweep', **cartpole.NMPC_OPTIONS)
    x0 = cartpole.hanging()
    state = ctrl.init(x0)
    for _ in range(10):
        state, _ = ctrl.solve(state, x0)
    xs, us, infos = ctrl.closed_loop(cartpole.plant(), x0, 160, state=state)
    th = cartpole.wrap(xs[:, 1])
    assert np.abs(th[-20:]).max() < 0.05                                         # upright over the last second
    assert float(jnp.max(jnp.abs(us))) <= cartpole.U_MAX + 1e-9
    assert float(jnp.max(jnp.abs(xs[:, 0]))) <= cartpole.X_MAX


def test_rti_tracks_like_sqp():
    """sqp_iters = 1 (the real-time iteration) tracks the reference in closed loop about as well as 3 SQP
    iterations per step."""
    x0 = quad.initial_state()
    params_fn = lambda k: quad.horizon_reference(k * quad.DT, N)
    plant = lambda x, u, k: quad.step_plant(x, u)
    refs = jax.vmap(lambda k: quad.reference(k * quad.DT)[:2])(jnp.arange(120))
    rms = {}
    for it in (1, 3):
        ctrl = pm.NMPC(quad.ocp(N), backend='sweep', **dict(quad.NMPC_OPTIONS, sqp_iters=it))
        xs, us, _ = ctrl.closed_loop(plant, x0, 120, params_fn=params_fn)
        rms[it] = float(jnp.sqrt(jnp.mean(jnp.sum((xs[:, :2] - refs) ** 2, 1))))
        assert float(us.min()) >= -1e-9 and float(us.max()) <= quad.T_MAX + 1e-9
    assert rms[1] < 1.1 * rms[3], rms


@pytest.mark.parametrize('bk', ['sweep', 'scan'])
def test_vmap_step(bk):
    """jax.vmap over initial states batches the SQP subproblems; sweep: per-QP iterations (equal to the single
    steps), scan: one long horizon with shared iterations (equal up to the QP tolerance)."""
    x0, SR = quad.initial_state(), quad.horizon_reference(0.0, N)
    ctrl = pm.NMPC(quad.ocp(N), backend=bk, **dict(quad.NMPC_OPTIONS, sqp_iters=2))
    x0s = x0 + 0.1 * jnp.asarray(np.random.default_rng(0).normal(size=(4, 8)))
    states = jax.vmap(lambda x: ctrl.init(x, SR))(x0s)
    u0s, _, _ = jax.vmap(lambda st, x: ctrl.step(st, x, SR))(states, x0s)
    for i in range(4):
        u0, _, _ = ctrl.step(jax.tree.map(lambda a: a[i], states), x0s[i], SR)
        np.testing.assert_allclose(u0s[i], u0, atol=1e-9 if bk == 'sweep' else 1e-3)
