"""The CUDA kernel (src/cuda) against the JAX core. Needs a GPU: run with JAX_PLATFORMS=cuda,cpu.

The factorization is checked directly (P_t read from the kernel's workspace): iterative refinement makes the final
solutions accurate even with a wrong factorization, which once hid a scratch-buffer bug for m > n.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from parampc import core, cuda, testing

pytestmark = pytest.mark.gpu


def p_offset(N, M, n, m):
    """Offset (in doubles) of the cost-to-go matrices P_t in the kernel's workspace (carve() in src/cuda/src/ipm.cu)."""
    E, N2, ks_alloc = N + 1, n * n, 2048
    sizes = [E * n, N * m, E * n, N * M, N * M, N * M, N * N2, N * m * n, N * m * m, E * 3 * N2,
             min(E, ks_alloc) * 3 * N2]
    return sum((s + 31) // 32 * 32 for s in sizes)


def kernel_P(data, z0):
    """P_t of the kernel's factorization at z0 (one IPM iteration, so its only factorization is at z0)."""
    N, n, m = data['B'].shape
    M = data['C'].shape[1]
    out = jax.jit(lambda d, z: cuda.call(d, z, 1e-8, 1e-12, 1))(data, z0)
    ws = np.asarray(out[-1])
    off = p_offset(N, M, n, m)
    return ws[off:off + (N + 1) * n * n].reshape(N + 1, n, n)


@pytest.mark.parametrize('n,m', [(4, 2), (3, 7)])                  # (3, 7): more inputs than states
@pytest.mark.parametrize('N', [1, 2, 5, 33, 200, 1000])             # N + 1 > 120: the hybrid Blelloch/Kogge-Stone scan
def test_factorization_matches_jax(n, m, N):
    data, z0 = testing.random_problem(N, n, m, 2 * m + 2, seed=N)
    cpu = jax.devices('cpu')[0]
    with jax.default_device(cpu):
        P = np.asarray(core.factorize(jax.device_put(data, cpu), jax.device_put(z0, cpu), method='sweep')['P'])
    Pk = kernel_P(data, z0)
    assert np.all(np.isfinite(Pk))
    assert np.max(np.abs(Pk - P)) <= 1e-10 * np.max(np.abs(P)), np.max(np.abs(Pk - P)) / np.max(np.abs(P))


@pytest.mark.parametrize('N', [1, 3, 64, 600, 3000])
def test_solve_matches_jax(N):
    data, z0 = testing.random_problem(N, 4, 2, 6, seed=N + 1)
    zc, kc, sc = core.ipm(data, z0, tol=1e-8, method='cuda')
    zs, ks, ss = core.ipm(data, z0, tol=1e-8, method='scan')
    assert float(core.kkt_score(data, zc)) <= 1e-8 and float(ss) <= 1e-8, (float(sc), float(ss))
    assert abs(int(kc) - int(ks)) <= 1, (int(kc), int(ks))
    for key in ('x', 'u'):
        np.testing.assert_allclose(zc[key], zs[key], atol=1e-6)


def test_deterministic():
    """Fixed reduction order: bit-identical results run to run (also on the LU fallback of a semidefinite cost)."""
    data, z0 = testing.random_problem(300, 4, 2, 6, seed=3)
    c = np.ones((1, 4)) / 2.0
    semi = dict(data, Q=jnp.broadcast_to(jnp.asarray(c.T @ c), data['Q'].shape), QN=jnp.zeros((4, 4)))
    run = jax.jit(lambda d, z: core.ipm(d, z, tol=1e-10, method='cuda'))
    for d in (data, semi):
        outs = [jax.tree.map(np.asarray, run(d, z0)) for _ in range(4)]
        for o in outs[1:]:
            for key in ('x', 'u', 'nu', 'lam', 's'):
                np.testing.assert_array_equal(o[0][key], outs[0][0][key])
            assert o[1] == outs[0][1]


def test_mehrotra_start_matches_jax():
    data, z0 = testing.random_problem(64, 4, 2, 6, seed=9)
    zc, kc, _ = core.ipm(data, z0, tol=1e-8, method='cuda', start='mehrotra')
    zs, ks, _ = core.ipm(data, z0, tol=1e-8, method='scan', start='mehrotra')
    assert float(core.kkt_score(data, zc)) <= 1e-8
    assert abs(int(kc) - int(ks)) <= 1, (int(kc), int(ks))
    np.testing.assert_allclose(zc['u'], zs['u'], atol=1e-6)


def test_nondefault_options_are_rejected():
    data, z0 = testing.random_problem(8, 2, 1, 2)
    with pytest.raises(ValueError):
        core.ipm(data, z0, method='cuda', n_corr=0)
