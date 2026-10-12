"""Linear MPC (parampc.solve_qp) on every backend: optimality certified on the dense QP, agreement between backends,
batches, warm starts, soft constraints, statuses."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import parampc as pm
from parampc.testing import dense_qp, kkt_certificate, objective, solve_equality_qp
from problems import CASES, case, lti


@pytest.mark.parametrize('name', CASES)
def test_certificate(backend, name):
    """Status 0, and the dense KKT conditions hold with the returned multipliers."""
    qp = case(name)
    sol = pm.solve_qp(qp, backend=backend, tol=1e-9)
    cert = kkt_certificate(qp, sol)
    assert int(sol.status) == 0, (int(sol.status), float(sol.kkt))
    for key in ('stationarity', 'equality', 'inequality', 'dual_sign', 'complementarity'):
        assert cert[key] < 1e-7, (key, cert[key])
    assert float(pm.kkt_score(qp, sol)) < 1e-8


@pytest.mark.parametrize('name', CASES)
def test_backends_agree(name):
    from conftest import BACKENDS
    qp = case(name)
    objs = [objective(qp, pm.solve_qp(qp, backend=b, tol=1e-10)) for b in BACKENDS]
    assert np.ptp(objs) <= 1e-7 * (1 + abs(objs[0])), objs


def test_terminal_box_is_active():
    qp = case('terminal')
    sol = pm.solve_qp(qp, backend='sweep', tol=1e-10)
    assert float(jnp.max(sol.lamN)) > 1e-2                        # the case tests what it claims to
    assert float(jnp.max(qp.CN @ sol.x[-1] - qp.dN)) < 1e-9


def test_lqr_without_rows(backend):
    """No inequality rows: the equality-constrained QP, solved directly."""
    A, B, x_ref, u_ref, x0 = lti(N=15)
    qp = pm.make_qp(A, B, Q=np.eye(6), R=0.1 * np.eye(3), N=15, x0=x0, x_ref=x_ref)
    sol = pm.solve_qp(qp, backend=backend)
    w, _ = solve_equality_qp(dense_qp(qp))
    np.testing.assert_allclose(np.concatenate([np.ravel(sol.x), np.ravel(sol.u)]), w, atol=1e-8)
    assert int(sol.status) == 0


def test_batch_equals_individual(backend):
    """jax.vmap over initial states (a long horizon on scan and cuda, a plain vmap on sweep) gives the individual
    optima; warm starts can be batched too."""
    qp = case('boxes', N=16)
    rng = np.random.default_rng(5)
    x0s = jnp.asarray(qp.x0 + rng.normal(size=(6, qp.nx)))
    solve = jax.jit(jax.vmap(lambda x0: pm.solve_qp(qp.replace(x0=x0), backend=backend, tol=1e-10)))
    out = solve(x0s)
    assert np.all(np.asarray(out.status) == 0)
    for i in range(len(x0s)):
        q = qp.replace(x0=x0s[i])
        one = pm.solve_qp(q, backend=backend, tol=1e-10)
        sol_i = jax.tree.map(lambda a: a[i], out)
        assert abs(objective(q, sol_i) - objective(q, one)) < 1e-8 * (1 + abs(objective(q, one)))
        np.testing.assert_allclose(sol_i.u, one.u, atol=1e-4)
        assert kkt_certificate(q, sol_i)['stationarity'] < 1e-7
    warm = jax.jit(jax.vmap(lambda x0, w: pm.solve_qp(qp.replace(x0=x0), w.shift(), backend=backend)))(x0s, out)
    assert np.all(np.asarray(warm.status) == 0)


def test_warm_start_saves_iterations(backend):
    """The next MPC step from the shifted solution takes fewer IPM iterations than a cold start."""
    A, B, x_ref, u_ref, x0 = lti(N=30)
    qp = pm.make_qp(A, B, Q=np.eye(6), R=0.1 * np.eye(3), N=30, x0=x0, QN=10 * np.eye(6), x_ref=x_ref, u_ref=u_ref)
    qp = pm.add_bounds(qp, u_min=-0.5, u_max=0.5)
    sol = pm.solve_qp(qp, backend=backend)
    x1 = A @ np.asarray(qp.x0) + B @ np.asarray(sol.u[0])
    cold = pm.solve_qp(qp.replace(x0=x1), backend=backend)
    warm = pm.solve_qp(qp.replace(x0=x1), sol.shift(), backend=backend)
    assert int(warm.status) == 0 and int(cold.status) == 0
    assert int(warm.iters) < int(cold.iters)
    assert abs(objective(qp.replace(x0=x1), warm) - objective(qp.replace(x0=x1), cold)) < 1e-6


def test_mehrotra_start(backend):
    qp = case('boxes')
    a = pm.solve_qp(qp, backend=backend, tol=1e-10)
    b = pm.solve_qp(qp, backend=backend, tol=1e-10, start='mehrotra')
    assert int(b.status) == 0
    assert abs(objective(qp, a) - objective(qp, b)) < 1e-8 * (1 + abs(objective(qp, a)))


def test_soften(backend):
    """A state bound that cannot hold: infeasible hard (status != 0), solved soft, slack = the violation."""
    A, B, x_ref, u_ref, x0 = lti(N=20, seed=7)
    qp = pm.make_qp(A, B, Q=np.eye(6), R=0.1 * np.eye(3), N=20, x0=x0, x_ref=x_ref, u_ref=u_ref)
    bound = 0.5 * float(np.abs(A @ x0).max())                     # x_1 cannot satisfy it with |u| <= 0.1
    hard = pm.add_bounds(qp, u_min=-0.1, u_max=0.1, x_min=-bound, x_max=bound)
    assert int(pm.solve_qp(hard, backend=backend).status) != 0
    rows = list(range(2 * 6))                                     # the state rows come first in add_bounds
    soft = pm.soften(hard, rows, l1=1e3, l2=1e-2)
    sol = pm.solve_qp(soft, backend=backend, tol=1e-6)            # soft QPs are ill-conditioned: on scan and cuda
    assert int(sol.status) == 0                                   # the attainable KKT score is ~1e-8, not 1e-10
    sig = np.asarray(sol.u[:, 3:])
    viol = np.asarray(jnp.einsum('tij,tj->ti', hard.C, sol.x[:-1]) + jnp.einsum('tij,tj->ti', hard.D, sol.u[:, :3])
                      - hard.d)[:, rows]
    assert sig.min() > -1e-9 and sig.max() > 1e-2
    np.testing.assert_allclose(sig, np.maximum(viol, 0.0), atol=1e-4)


def test_infeasible_is_not_solved():
    A, B, x_ref, u_ref, x0 = lti(N=10, seed=8)
    qp = pm.make_qp(A, B, Q=np.eye(6), R=np.eye(3), N=10, x0=x0)
    qp = pm.add_bounds(qp, u_min=-0.01, u_max=0.01, xN_min=x_ref + 10.0, xN_max=x_ref + 10.1)
    assert int(pm.solve_qp(qp, backend='sweep').status) != 0


def test_ipm_options_pass_through():
    qp = case('boxes')
    a = pm.solve_qp(qp, backend='scan', tol=1e-10)
    b = pm.solve_qp(qp, backend='scan', tol=1e-10, ipm_options=dict(n_corr=0, refine_pred=1))
    assert int(b.status) == 0
    assert abs(objective(qp, a) - objective(qp, b)) < 1e-8 * (1 + abs(objective(qp, a)))


def test_add_bounds_rows():
    A, B = np.eye(2), np.ones((2, 1))
    qp = pm.make_qp(A, B, Q=np.eye(2), R=np.eye(1), N=4, x0=np.zeros(2))
    qp = pm.add_bounds(qp, x_min=[-1.0, -np.inf], x_max=[2.0, np.inf], u_max=3.0, xN_max=[5.0, 6.0])
    assert qp.M == 3 and qp.MN == 2                               # x_0 <= 2, -x_0 <= 1, u <= 3; terminal x <= (5, 6)
    np.testing.assert_array_equal(qp.C[0], 0.0)                   # no state rows at t = 0 (x_0 is fixed) ...
    np.testing.assert_array_equal(qp.d[0, :2], 1.0)               # ... they are 0 <= 1 there
    np.testing.assert_array_equal(qp.d[1], [2.0, 1.0, 3.0])
    np.testing.assert_array_equal(qp.dN, [5.0, 6.0])


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError):
        pm.solve_qp(case('boxes'), backend='gpu')
