"""Solve an OCPQP (or a batch of them) with the interior-point core on a backend (parampc.backends).

Three exact reductions bring every OCPQP to the core's form:
  - terminal rows: a copy stage x_{N+1} = x_N carries CN x_N <= dN, and the terminal cost is split between the two;
  - stages without inequality rows: one always-inactive row 0 <= 1;
  - a batch of B QPs (jax.vmap) on 'scan' or 'cuda': one QP with horizon B (N+1) - 1, in which each QP's terminal
    stage becomes a stage with dynamics A = 0, B = 0, c = x0 of the next QP. The problem is separable, so its
    solution is the B individual solutions, and a long horizon is what the parallel method is good at. The batch
    shares the IPM's step lengths and iterations, so it stops when the hardest QP is solved. (On 'sweep' a batch is
    an ordinary vmap with per-QP iterations: the sequential recursion gains nothing from a long horizon.)
"""
import dataclasses
import functools

import jax
import jax.numpy as jnp

from . import core
from .backends import resolve_backend
from .qp import OCPQP, Solution, kkt_score

STAGE_FIELDS = ('A', 'B', 'c', 'Q', 'S', 'R', 'q', 'r', 'C', 'D', 'd')


# --------------------------------------------------------- reductions to the core ---
def _pad_last(a, M, v):
    """Pad the last axis of a to length M with the value v."""
    k = M - a.shape[-1]
    return a if k == 0 else jnp.concatenate([a, jnp.full(a.shape[:-1] + (k,), v, a.dtype)], axis=-1)


def _pad_rows(C, D, d, M):
    """Pad the stage rows to M with inactive rows 0 <= 1."""
    k = M - C.shape[-2]
    if k == 0:
        return C, D, d
    pad = lambda a, v: jnp.concatenate([a, jnp.full(a.shape[:-2] + (k,) + a.shape[-1:], v)], axis=-2)
    return pad(C, 0.0), pad(D, 0.0), jnp.concatenate([d, jnp.ones(d.shape[:-1] + (k,))], axis=-1)


def to_core(qp):
    """The core's data dict for qp: terminal rows by a copy stage, at least one row per stage."""
    n, m = qp.nx, qp.nu
    M = max(qp.M, qp.MN, 1)
    C, D, d = _pad_rows(qp.C, qp.D, qp.d, M)
    data = dict(A=qp.A, B=qp.B, c=qp.c, Q=qp.Q, S=qp.S, R=qp.R, q=qp.q, r=qp.r, C=C, D=D, d=d,
                QN=qp.QN, qN=qp.qN, x0=qp.x0)
    if qp.MN:
        CN, DN, dN = _pad_rows(qp.CN[None], jnp.zeros((1, qp.MN, m)), qp.dN[None], M)
        last = dict(A=jnp.eye(n), B=jnp.zeros((n, m)), c=jnp.zeros(n), Q=0.5 * qp.QN, S=jnp.zeros((m, n)),
                    R=jnp.eye(m), q=0.5 * qp.qN, r=jnp.zeros(m), C=CN[0], D=DN[0], d=dN[0])
        data = {k: jnp.concatenate([data[k], last[k][None]]) if k in last else data[k] for k in data}
        data['QN'], data['qN'] = 0.5 * qp.QN, 0.5 * qp.qN
    return data


def starting_point(qp, data, warm, floor):
    """The core's starting point: the warm start's (x, u, lam), or a cold start x = (x0, 0, ..., 0), u = 0, lam = 1
    (an open-loop rollout is a worse start for unstable systems). lam and s are floored at `floor`."""
    m, M = qp.nu, data['C'].shape[-2]
    if warm is None:
        x = jnp.zeros((qp.N + 1, qp.nx)).at[0].set(qp.x0)
        u, lam, lamN = jnp.zeros((qp.N, m)), jnp.ones((qp.N, M)), jnp.ones(M)
    else:
        x, u, lam, lamN = warm.x, warm.u, _pad_last(warm.lam, M, 1.0), _pad_last(warm.lamN, M, 1.0)
    if qp.MN:
        x, u = jnp.concatenate([x, x[-1:]]), jnp.concatenate([u, jnp.zeros((1, m))])
        lam = jnp.concatenate([lam, lamN[None]])
    return core.initial_iterate(data, x, u, lam, floor)


def _status(score, k, tol, max_iter):
    return jnp.where(score <= tol, 0, jnp.where((k >= max_iter) & jnp.isfinite(score), 1, 2)).astype(jnp.int32)


def from_core(qp, z, k, score, tol, max_iter):
    """Map the core's iterate back to qp's variables, with the statistics (status from the core's own score)."""
    N, M, MN = qp.N, qp.M, qp.MN
    lamN, sN = (z['lam'][N, :MN], z['s'][N, :MN]) if MN else (jnp.zeros(0), jnp.zeros(0))
    sol = Solution(x=z['x'][:N + 1], u=z['u'][:N], nu=z['nu'][:N + 1], lam=z['lam'][:N, :M], s=z['s'][:N, :M],
                   lamN=lamN, sN=sN, iters=jnp.asarray(k, jnp.int32), kkt=jnp.zeros(()),
                   status=jnp.zeros((), jnp.int32))
    return dataclasses.replace(sol, kkt=kkt_score(qp, sol), status=_status(score, k, tol, max_iter))


# --------------------------------------------------------------------- batches ---
def batch_qp(qps):
    """B QPs (fields with a leading axis B) as one QP with horizon B (N+1) - 1: QP b's terminal stage becomes a stage
    with its terminal cost and terminal rows, dynamics A = 0, B = 0, c = x0 of QP b+1, and R = I (its input has no
    effect). Separable, so its solution is the B solutions (`split_solution`)."""
    Bn, N, n, m = qps.A.shape[0], qps.N, qps.nx, qps.nu
    M = max(qps.M, qps.MN)
    C, D, d = _pad_rows(qps.C, qps.D, qps.d, M) if M else (qps.C, qps.D, qps.d)
    if qps.MN:
        CN, DN, dN = _pad_rows(qps.CN, jnp.zeros((Bn, qps.MN, m)), qps.dN, M)
    else:
        CN, DN, dN = jnp.zeros((Bn, M, n)), jnp.zeros((Bn, M, m)), jnp.ones((Bn, M))
    x0_next = jnp.concatenate([qps.x0[1:], jnp.zeros((1, n))])
    reset = dict(A=jnp.zeros((Bn, n, n)), B=jnp.zeros((Bn, n, m)), c=x0_next, Q=qps.QN, S=jnp.zeros((Bn, m, n)),
                 R=jnp.broadcast_to(jnp.eye(m), (Bn, m, m)), q=qps.qN, r=jnp.zeros((Bn, m)), C=CN, D=DN, d=dN)
    stage = dict(A=qps.A, B=qps.B, c=qps.c, Q=qps.Q, S=qps.S, R=qps.R, q=qps.q, r=qps.r, C=C, D=D, d=d)
    flat = {k: jnp.concatenate([stage[k], reset[k][:, None]], axis=1).reshape((Bn * (N + 1),) + stage[k].shape[2:])
            [:Bn * (N + 1) - 1] for k in STAGE_FIELDS}
    return OCPQP(**flat, QN=qps.QN[-1], qN=qps.qN[-1], x0=qps.x0[0], CN=qps.CN[-1], dN=qps.dN[-1])


def batch_warm(qps, warm):
    """A batch of warm starts (a Solution with a leading axis) for batch_qp(qps)."""
    Bn, N, m = qps.A.shape[0], qps.N, qps.nu
    M = max(qps.M, qps.MN)
    lam, lamN = _pad_last(warm.lam, M, 1.0), _pad_last(warm.lamN, M, 1.0)
    u = jnp.concatenate([warm.u, jnp.zeros((Bn, 1, m))], axis=1).reshape(Bn * (N + 1), m)[:-1]
    lam = jnp.concatenate([lam, lamN[:, None]], axis=1).reshape(Bn * (N + 1), M)[:-1]
    z = lambda a: jnp.zeros_like(a)
    return Solution(x=warm.x.reshape(Bn * (N + 1), -1), u=u, nu=z(warm.x.reshape(Bn * (N + 1), -1)), lam=lam,
                    s=z(lam), lamN=warm.lamN[-1], sN=warm.sN[-1], iters=warm.iters, kkt=warm.kkt, status=warm.status)


def split_solution(qps, sol):
    """The B solutions inside the solution of batch_qp(qps)."""
    Bn, N, M, MN = qps.A.shape[0], qps.N, qps.M, qps.MN
    Mb = max(M, MN)
    pad = lambda a: jnp.concatenate([a, jnp.zeros((1,) + a.shape[1:])])
    u = pad(sol.u).reshape(Bn, N + 1, -1)[:, :N]
    lam_all = jnp.concatenate([sol.lam, jnp.zeros((1, Mb))]).reshape(Bn, N + 1, Mb)
    s_all = jnp.concatenate([sol.s, jnp.zeros((1, Mb))]).reshape(Bn, N + 1, Mb)
    lamN = jnp.concatenate([lam_all[:-1, N, :MN], sol.lamN[None]]) if MN else jnp.zeros((Bn, 0))
    sN = jnp.concatenate([s_all[:-1, N, :MN], sol.sN[None]]) if MN else jnp.zeros((Bn, 0))
    out = Solution(x=sol.x.reshape(Bn, N + 1, -1), u=u, nu=sol.nu.reshape(Bn, N + 1, -1), lam=lam_all[:, :N, :M],
                   s=s_all[:, :N, :M], lamN=lamN, sN=sN, iters=jnp.broadcast_to(sol.iters, (Bn,)),
                   kkt=jnp.zeros(Bn), status=jnp.zeros(Bn, jnp.int32))
    return dataclasses.replace(out, kkt=jax.vmap(kkt_score)(qps, out), status=jnp.broadcast_to(sol.status, (Bn,)))


# ----------------------------------------------------------------------- solve ---
@functools.lru_cache(maxsize=None)
def _solver(tol, mu_min, max_iter, method, start, floor, precision, ipm_options):
    opts = dict(ipm_options)

    def solve_one(qp, warm):
        data = to_core(qp)
        z0 = starting_point(qp, data, warm, floor)
        z, k, score = core.ipm(data, z0, tol=tol, mu_min=mu_min, max_iter=max_iter, method=method, start=start,
                               precision=precision, **opts)
        return from_core(qp, z, k, score, tol, max_iter)

    @jax.custom_batching.custom_vmap
    def solve(qp, warm):
        return solve_one(qp, warm)

    @solve.def_vmap
    def solve_batch(axis_size, in_batched, qp, warm):
        bq, bw = in_batched
        if method == 'sweep':                                   # sequential: a plain vmap is faster
            axes = lambda tree: jax.tree.map(lambda b: 0 if b else None, tree)
            out = jax.vmap(solve_one, in_axes=(axes(bq), axes(bw)))(qp, warm)
            return out, jax.tree.map(lambda _: True, out)
        qp = jax.tree.map(lambda a, b: a if b else jnp.broadcast_to(a, (axis_size,) + a.shape), qp, bq)
        if warm is not None:
            warm = jax.tree.map(lambda a, b: a if b else jnp.broadcast_to(a, (axis_size,) + a.shape), warm, bw)
            warm = batch_warm(qp, warm)
        out = split_solution(qp, solve(batch_qp(qp), warm))     # nested vmaps batch the batch
        return out, jax.tree.map(lambda _: True, out)

    return jax.jit(solve)


def solve_qp(qp, warm=None, *, tol=1e-8, mu_min=1e-14, max_iter=50, backend='auto', start='given', floor=None,
             precision='float64', ipm_options=None):
    """Solve an OCPQP. Returns a Solution.

    warm: a Solution of a similar QP (e.g. the previous MPC step's, `sol.shift()`): its x, u and multipliers start
        the IPM, the multipliers and slacks floored at `floor` (default 1e-3 with a warm start, 1 without).
    start: 'given' or 'mehrotra' (Mehrotra's starting-point heuristic: fewer iterations on cold starts of hard QPs).
    precision: 'float64', or 'float32': the factorization in single precision while the duality measure is above
        ipm_options['mu_switch'] (default 1e-5), then in double precision; the iterate, the residuals and the solves
        stay double, so the solution has the float64 accuracy. Faster on the 'cuda' backend (a consumer GPU's
        float32 arithmetic has a tenth of float64's latency); on 'scan' and 'sweep' it is the reference
        implementation.
    The IPM stops at KKT score <= tol, at duality measure <= mu_min (past the attainable accuracy the iterates
    degrade; the best one is returned), at max_iter, or on a non-finite step. Infeasibility is not detected: an
    infeasible QP ends with status 2 or 1.
    ipm_options: advanced settings of the IPM (parampc.core.ipm: n_corr, refine, refine_pred, combine, mu_switch);
        the 'cuda' backend supports only the defaults and mu_switch.
    Under jax.vmap a batch is solved as one long horizon on 'scan' and 'cuda' (module docstring)."""
    if floor is None:
        floor = 1.0 if warm is None else 1e-3
    opts = tuple(sorted((ipm_options or {}).items()))
    return _solver(float(tol), float(mu_min), int(max_iter), resolve_backend(backend), start, float(floor),
                   precision, opts)(qp, warm)
