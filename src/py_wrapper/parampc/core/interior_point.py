"""Mehrotra's predictor-corrector interior-point method with Gondzio's centrality correctors.

Each iteration factorizes once and solves several times with that factorization: an unrefined affine-scaling
predictor (it only sets the centering parameter and the second-order term), the corrector, and the centrality
correctors, whose sum is refined once against the system it solves. The loop runs on the device
(`lax.while_loop`) and returns the most accurate iterate it has seen: near the limit of floating-point accuracy
the barrier weights lam / s ~ 1 / mu become so large that a Newton step can make the iterate worse.
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from .kkt import kkt_score, mv
from .newton import factorize, newton_direction, refine_step, solve_newton

METHODS = ('scan', 'sweep', 'cuda')


def initial_iterate(data, x, u, lam=1.0, floor=1.0):
    """A starting point: the given (x, u), nu = 0, the multipliers lam and the slacks s = d - C x - D u, both
    floored at `floor`. The defaults (lam = 1, floor = 1) are a cold start; a warm start passes the previous
    solution's multipliers and a small floor, which keeps more of it."""
    g = mv(data['C'], x[:-1]) + mv(data['D'], u)
    return dict(x=x, u=u, nu=jnp.zeros_like(x),
                lam=jnp.maximum(lam * jnp.ones_like(data['d']), floor), s=jnp.maximum(data['d'] - g, floor))


def mehrotra_start(data, z, method='scan', combine='auto'):
    """Mehrotra's starting-point heuristic: one affine-scaling Newton step from z, then s and lam shifted so that
    both are positive and their products balanced. Costs one factorization and one solve."""
    zt = jax.tree.map(jnp.add, z, solve_newton(data, z, factorize(data, z, method, combine), 0.0, 0.0, method, 1))
    s = zt['s'] + jnp.maximum(-1.5 * jnp.min(zt['s']), 0.0)
    lam = zt['lam'] + jnp.maximum(-1.5 * jnp.min(zt['lam']), 0.0)
    ls = jnp.sum(lam * s)
    return dict(zt, s=s + 0.5 * ls / jnp.sum(lam), lam=lam + 0.5 * ls / jnp.sum(s))


def step_to_boundary(v, dv, frac=0.995):
    """The largest step in (0, 1] that keeps v + alpha dv > 0, shortened by `frac`."""
    ratio = jnp.where(dv < 0, -v / dv, jnp.inf)
    return jnp.minimum(frac * jnp.min(ratio), 1.0)


def centrality_correctors(data, z, fac, step, ap, ad, target, method, n_corr, d_alpha=0.1, beta=(0.1, 10.0),
                          gamma=0.1):
    """Gondzio's multiple centrality correctors, with the iteration's factorization. Each aims the complementarity
    products of a longer trial step (alpha + d_alpha) at [beta_min, beta_max] * target and is kept only if it
    lengthens the step by gamma * d_alpha. Each costs one solve; the count is fixed (no data-dependent control
    flow). Returns the step, its step lengths, and the accepted change dv of the complementarity right-hand side
    (the step solves the Newton system with r_t lowered by dv, which the caller's refinement uses)."""
    x, u = z['x'], z['u']
    zero = dict(r0=jnp.zeros_like(x[0]), rx=jnp.zeros_like(x[:-1]), rxN=jnp.zeros_like(x[0]),
                ru=jnp.zeros_like(u), rd=jnp.zeros_like(x[1:]), rc=jnp.zeros_like(z['s']))
    dv_total = jnp.zeros_like(z['s'])
    for _ in range(n_corr):
        tp, td = jnp.minimum(ap + d_alpha, 1.0), jnp.minimum(ad + d_alpha, 1.0)
        v = (z['lam'] + td * step['lam']) * (z['s'] + tp * step['s'])
        dv = jnp.maximum(jnp.clip(v, beta[0] * target, beta[1] * target) - v, -beta[1] * target)
        new = jax.tree.map(jnp.add, step, newton_direction(data, z, fac, dict(zero, rt=-dv), method))
        ap_n, ad_n = step_to_boundary(z['s'], new['s']), step_to_boundary(z['lam'], new['lam'])
        keep = jnp.minimum(ap_n, ad_n) >= jnp.minimum(ap, ad) + gamma * d_alpha
        step = jax.tree.map(lambda a, b: jnp.where(keep, a, b), new, step)
        ap, ad = jnp.where(keep, ap_n, ap), jnp.where(keep, ad_n, ad)
        dv_total = dv_total + jnp.where(keep, dv, 0.0)
    return step, ap, ad, dv_total


def ipm_iteration(data, z, method='scan', refine=1, refine_pred=0, n_corr=1, combine='auto', fac_dtype=None):
    """One predictor-corrector iteration. The predictor gets `refine_pred` refinement steps (default none); with
    n_corr = 0 the corrector is refined directly, otherwise the corrector and the n_corr centrality correctors are
    computed unrefined and their sum is refined against the system it solves, so the final step has the accuracy
    of a refined solve."""
    mu = jnp.mean(z['lam'] * z['s'])
    fac = factorize(data, z, method, combine, fac_dtype)            # one factorization, several solves

    aff = solve_newton(data, z, fac, 0.0, 0.0, method, refine_pred)                     # predictor
    ap = step_to_boundary(z['s'], aff['s'], 1.0)
    ad = step_to_boundary(z['lam'], aff['lam'], 1.0)
    mu_aff = jnp.mean((z['lam'] + ad * aff['lam']) * (z['s'] + ap * aff['s']))
    sigma = jnp.clip((mu_aff / mu) ** 3, 0.0, 1.0)

    corr = aff['lam'] * aff['s']
    if n_corr == 0:
        step = solve_newton(data, z, fac, sigma * mu, corr, method, refine)               # corrector
    else:
        step = solve_newton(data, z, fac, sigma * mu, corr, method, 0)
        ap = step_to_boundary(z['s'], step['s'])
        ad = step_to_boundary(z['lam'], step['lam'])
        step, ap, ad, dv = centrality_correctors(data, z, fac, step, ap, ad, sigma * mu, method, n_corr)
        step = refine_step(data, z, fac, step, sigma * mu, corr - dv, method, refine)
    ap = step_to_boundary(z['s'], step['s'])
    ad = step_to_boundary(z['lam'], step['lam'])
    return dict(x=z['x'] + ap * step['x'], u=z['u'] + ap * step['u'], s=z['s'] + ap * step['s'],
                nu=z['nu'] + ad * step['nu'], lam=z['lam'] + ad * step['lam'])


@partial(jax.jit, static_argnames=('method', 'refine', 'refine_pred', 'n_corr', 'start', 'combine', 'fac_dtype'))
def ipm(data, z0, tol=1e-8, mu_min=1e-12, max_iter=30, method='scan', refine=1, refine_pred=0, n_corr=1,
        start='given', combine='auto', fac_dtype=None):
    """Iterate until the KKT score is <= tol, the duality measure is <= mu_min, or max_iter iterations.

    Returns (z, iterations, score): the most accurate iterate seen and its KKT score. A non-finite Newton step (a
    breakdown) also ends the loop, and a message says so.

    method: 'scan' (associative scans), 'sweep' (Riccati recursion) or 'cuda' (the whole solve as one CUDA kernel,
    with the default options). start='mehrotra' replaces z0 by Mehrotra's starting point (counted as one
    iteration). The other options are those of `ipm_iteration` and `factorize`."""
    if method == 'cuda':
        if (refine, refine_pred, n_corr, combine, fac_dtype) != (1, 0, 1, 'auto', None):
            raise ValueError("method='cuda' implements the default options (and either start) only")
        from .. import cuda
        z, k, score, mu = cuda.ipm(data, z0, tol, mu_min, max_iter, start=start)
        _report_breakdown(jnp.isfinite(mu), k, score)
        return z, k, score
    if method not in METHODS:
        raise ValueError(f'method must be one of {METHODS}, not {method!r}')
    k0 = 0
    if start == 'mehrotra':
        z0, k0 = mehrotra_start(data, z0, method, combine), 1
    elif start != 'given':
        raise ValueError(f"start must be 'given' or 'mehrotra', not {start!r}")

    def not_done(carry):
        z, best, best_score, k = carry
        return (k < max_iter) & (best_score > tol) & (jnp.mean(z['lam'] * z['s']) > mu_min)

    def iterate(carry):
        z, best, best_score, k = carry
        z = ipm_iteration(data, z, method, refine, refine_pred, n_corr, combine, fac_dtype)
        score = kkt_score(data, z)
        better = score < best_score
        best = jax.tree.map(lambda new, old: jnp.where(better, new, old), z, best)
        return z, best, jnp.where(better, score, best_score), k + 1

    z, best, best_score, k = lax.while_loop(not_done, iterate, (z0, z0, kkt_score(data, z0), k0))
    _report_breakdown(jnp.isfinite(jnp.mean(z['lam'] * z['s'])), k, best_score)
    return best, k, best_score


def _report_breakdown(finite, k, score):
    """Print a message after a non-finite Newton step. A host callback, not a lax.cond around jax.debug.print:
    under jax.vmap a cond runs both branches, which would print the message for every QP of a batch."""
    def warn(finite, k, score):
        if not finite:
            print(f'parampc: non-finite Newton step at iteration {int(k)}; returning the best iterate '
                  f'(KKT score {float(score):.1e})')
    jax.debug.callback(warn, finite, k, score)
