"""The Newton step on the perturbed KKT system: condense to the LQ problem, solve it by the scans, recover the
multiplier and slack steps, and refine iteratively with the same factorization."""
import jax
import jax.numpy as jnp

from .kkt import barrier_hessians, lq_rhs, mv, residuals
from .lq import factor_lq, solve_lq


def factorize(data, z, method='scan', combine='auto', fac_dtype=None):
    """Matrix part of the Newton step at z: the barrier Hessians and `factor_lq`. fac_dtype='float32' runs
    `factor_lq` in single precision; the barrier Hessians and every solve stay in the iterate's precision, and
    iterative refinement corrects the single-precision operator. Throughout a solve it cannot reach tight
    tolerances on ill-conditioned problems; `ipm` with precision='float32' uses it while the duality measure is
    large."""
    lq = barrier_hessians(data, z)
    if fac_dtype is None:
        return dict(lq, **factor_lq(lq, data, method, combine))
    low = lambda t: {k: v.astype(fac_dtype) for k, v in t.items()}
    with jax.default_matmul_precision('float32'):       # not TF32, the GPU's default for float32 products
        fac = factor_lq(low(lq), low({'A': data['A'], 'B': data['B']}), method, combine)
    return dict(lq, **{k: v.astype(data['A'].dtype) for k, v in fac.items()})


def newton_direction(data, z, fac, res, method='scan'):
    """Solve grad F(z) dz = -res with the factorization `fac` of z: right-hand side, vector part of the LQ solve,
    recovery of (dlam, ds). Linear in res."""
    rhs = lq_rhs(data, z, res)
    dx, du, dnu = solve_lq(rhs, data, fac, method)
    v = mv(data['C'], dx[:-1]) + mv(data['D'], du)
    return dict(x=dx, u=du, nu=dnu, lam=fac['W'] * v + rhs['w'], s=-res['rc'] - v)


def refine_step(data, z, fac, dz, tau, corr=0.0, method='scan', refine=1):
    """`refine` steps of iterative refinement of a computed step dz for the Newton system on F_tau: the linear
    residual F + grad F dz (a stage-local map) is corrected with the same factorization."""
    F = lambda zz: residuals(data, zz, tau, corr)
    for _ in range(refine):
        r, Jdz = jax.jvp(F, (z,), (dz,))
        dz = jax.tree.map(jnp.add, dz, newton_direction(data, z, fac, jax.tree.map(jnp.add, r, Jdz), method))
    return dz


def solve_newton(data, z, fac, tau, corr=0.0, method='scan', refine=1):
    """The Newton step on F_tau with `refine` steps of iterative refinement. One step recovers the accuracy that
    large barrier weights cost the condensed solve."""
    dz = newton_direction(data, z, fac, residuals(data, z, tau, corr), method)
    return refine_step(data, z, fac, dz, tau, corr, method, refine)


def newton_step(data, z, tau, corr=0.0, method='scan', refine=1):
    """One Newton step on F_tau: factorize at z, then solve with refinement."""
    return solve_newton(data, z, factorize(data, z, method), tau, corr, method, refine)
