"""The solver core (parampc.core): the scan's algebra, the Newton step against dense linear algebra, and the IPM
against an independent dense KKT certificate."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from parampc import core, testing

newton = jax.jit(core.newton_step, static_argnames=('method',))
iteration = jax.jit(core.ipm_iteration)


def test_combine_is_associative():
    rng = np.random.default_rng(0)

    def element(k=5):
        L1, L2 = rng.normal(size=(2, k, k))
        return rng.normal(size=(k, k)), L1 @ L1.T / k, L2 @ L2.T / k + 0.1 * np.eye(k)     # (A, C >= 0, J > 0)
    for _ in range(16):
        a, b, c = element(), element(), element()
        left = core.vf_combine(core.vf_combine(a, b), c)
        right = core.vf_combine(a, core.vf_combine(b, c))
        assert max(float(jnp.max(jnp.abs(x - y))) for x, y in zip(left, right)) < 1e-12


def test_scan_equals_sweep_and_lu_combine():
    data, z0 = testing.random_problem(40, 5, 2, 6, seed=1)
    lq = core.barrier_hessians(data, z0)
    P_scan = core.factor_lq(lq, data, 'scan')['P']
    np.testing.assert_allclose(P_scan, core.riccati_sweep(lq, data), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(core.factor_lq(lq, data, 'scan', combine='lu')['P'], P_scan, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize('method', ['scan', 'sweep'])
def test_newton_step_solves_the_newton_system(method):
    """Relative residual |J dz + F| / |F| of the step on the full Newton matrix, at the start and six iterations
    in (barrier weights over many decades): round-off level, like dense LU."""
    data, z = testing.random_problem(32, 4, 2, 8, seed=32, u_max=0.05, x_max=0.5)
    for k in range(7):
        if k in (0, 6):
            tau = 0.1 * float(jnp.mean(z['lam'] * z['s']))
            J, F, _ = testing.newton_matrix(data, z, tau)
            dz = np.asarray(ravel_pytree(newton(data, z, tau, method=method))[0])
            assert np.max(np.abs(J @ dz + F)) / np.max(np.abs(F)) < 1e-10
        z = iteration(data, z)


def test_one_step_without_inequalities():
    """Without inequality rows the KKT system is affine: one Newton step from any point is the optimum."""
    data, z0 = testing.random_problem(40, 4, 2, 0, seed=2)
    z0 = dict(z0, x=z0['x'] + 1.0, u=z0['u'] - 0.5)
    z1 = jax.tree.map(jnp.add, z0, newton(data, z0, 0.0, method='scan'))
    assert max(float(jnp.max(jnp.abs(v))) for v in core.residuals(data, z1, 0.0).values() if v.size) < 1e-12
    w, _ = testing.solve_equality_qp(testing.dense_qp(data))
    assert np.max(np.abs(testing.pack(z1)[0] - w)) < 1e-12


@pytest.mark.parametrize('N,n,m,M,u_max,x_max', [(64, 4, 2, 8, 0.05, 0.5), (128, 6, 3, 12, 0.05, 0.3)])
def test_ipm_reaches_certified_optimum(N, n, m, M, u_max, x_max):
    """Tight bounds, many active rows: the scan and the sweep reach the same optimum, certified on the dense QP."""
    data, z0 = testing.random_problem(N, n, m, M, seed=3, u_max=u_max, x_max=x_max)
    dense = testing.dense_qp(data)
    objectives = []
    for method in ('scan', 'sweep'):
        z, k, score = core.ipm(data, z0, tol=1e-10, method=method)
        cert = testing.kkt_certificate(dense, z)
        assert float(score) <= 1e-10 and abs(float(score) - float(core.kkt_score(data, z))) < 1e-15
        for key in ('stationarity', 'equality', 'inequality', 'dual_sign', 'complementarity'):
            assert cert[key] < 1e-8, (method, key, cert[key])
        objectives.append(cert['objective'])
    assert abs(objectives[0] - objectives[1]) < 1e-10 * (1 + abs(objectives[0]))


@pytest.mark.parametrize('terminal_only', [False, True])
def test_semidefinite_cost_falls_back_to_lu(terminal_only):
    """Output-only (Q = QN = c c') or terminal-only (Q = 0) cost: the Cholesky combine breaks down, the scan is
    redone with the LU combine, and the IPM still reaches the certified optimum."""
    data, z0 = testing.random_problem(32, 4, 2, 4, seed=5)
    if terminal_only:
        data = dict(data, Q=0 * data['Q'], QN=jnp.eye(4))
    else:
        cc = jnp.outer(jnp.eye(4)[0], jnp.eye(4)[0])
        data = dict(data, Q=jnp.broadcast_to(cc, data['Q'].shape), QN=cc)
    lq = core.barrier_hessians(data, z0)
    assert not np.all(np.isfinite(np.asarray(core.factor_lq(lq, data, 'scan', combine='chol')['P'])))
    assert not bool(core.cholesky_ok(data))
    objectives = []
    for method in ('scan', 'sweep'):
        z, k, score = core.ipm(data, z0, tol=1e-10, method=method)
        cert = testing.kkt_certificate(data, z)
        assert max(cert[key] for key in ('stationarity', 'equality', 'inequality', 'complementarity')) < 1e-8
        objectives.append(cert['objective'])
    assert abs(objectives[0] - objectives[1]) < 1e-9 * (1 + abs(objectives[0]))


@pytest.mark.parametrize('options', [dict(refine_pred=1, n_corr=0), dict(n_corr=2), dict(start='mehrotra'),
                                     dict(combine='chol')])
def test_ipm_options_reach_the_same_optimum(options):
    data, z0 = testing.random_problem(64, 4, 2, 8, seed=3, u_max=0.05, x_max=0.5)
    z_ref, _, _ = core.ipm(data, z0, tol=1e-10)
    z, k, score = core.ipm(data, z0, tol=1e-10, **options)
    assert float(score) <= 1e-10
    assert abs(testing.objective(data, z) - testing.objective(data, z_ref)) < 1e-9


def test_cholesky_ok():
    data, _ = testing.random_problem(16, 4, 2, 4, seed=0)
    assert bool(core.cholesky_ok(data))
    assert not bool(core.cholesky_ok(dict(data, QN=jnp.zeros((4, 4)))))


@pytest.mark.parametrize('start', ['given', 'mehrotra'])
def test_float32_phase_reaches_the_certified_optimum(start):
    """precision='float32': a single-precision factorization while mu > mu_switch, double precision after. The
    iterate and the solves stay double, so the solution is certified like a float64 one, in at most one extra
    iteration."""
    data, z0 = testing.random_problem(64, 4, 2, 8, seed=3, u_max=0.05, x_max=0.5)
    z64, k64, _ = core.ipm(data, z0, tol=1e-10, start=start)
    z32, k32, score = core.ipm(data, z0, tol=1e-10, start=start, precision='float32')
    assert float(score) <= 1e-10 and int(k32) <= int(k64) + 1, (int(k32), int(k64))
    cert = testing.kkt_certificate(testing.dense_qp(data), z32)
    assert max(cert[key] for key in ('stationarity', 'equality', 'inequality', 'complementarity')) < 1e-8
    obj = testing.objective(data, z64)
    assert abs(testing.objective(data, z32) - obj) < 1e-9 * (1 + abs(obj))


def test_float32_phase_with_lu_fallback():
    """A semidefinite cost in the float32 phase: the single-precision Cholesky combine fails and the scan is redone
    with LU combines in single precision; the IPM still reaches the certified optimum."""
    data, z0 = testing.random_problem(32, 4, 2, 4, seed=5)
    cc = jnp.outer(jnp.eye(4)[0], jnp.eye(4)[0])
    data = dict(data, Q=jnp.broadcast_to(cc, data['Q'].shape), QN=cc)
    z, k, score = core.ipm(data, z0, tol=1e-10, precision='float32')
    cert = testing.kkt_certificate(data, z)
    assert max(cert[key] for key in ('stationarity', 'equality', 'inequality', 'complementarity')) < 1e-8


def test_unknown_precision_is_rejected():
    data, z0 = testing.random_problem(8, 2, 1, 2)
    with pytest.raises(ValueError):
        core.ipm(data, z0, precision='float16')


def test_breakdown_message_is_opt_in(capsys):
    """The message after a non-finite step is a host callback after every solve: off unless report_breakdowns()."""
    data, z0 = testing.random_problem(8, 2, 1, 2, seed=4)
    bad = dict(z0, lam=z0['lam'].at[0, 0].set(jnp.nan))
    try:
        for on in (False, True):
            core.report_breakdowns(on)
            jax.clear_caches()                        # the setting is read when the solve is traced
            jax.block_until_ready(core.ipm(data, bad))
            jax.effects_barrier()
            assert ('non-finite Newton step' in capsys.readouterr().out) == on
    finally:
        core.report_breakdowns(False)
        jax.clear_caches()
