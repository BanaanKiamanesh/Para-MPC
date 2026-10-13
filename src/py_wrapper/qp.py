"""Linear MPC: the MPC-structured QP, its solution, and helpers to build one.

    qp = make_qp(A, B, Q, R, N=50, x0=x0, QN=QN)          # stage-invariant data is broadcast to N stages
    qp = add_bounds(qp, u_min=-1.0, u_max=1.0, x_min=..., x_max=..., xN_min=..., xN_max=...)
    qp = soften(qp, rows=[...])                            # make some rows soft (slack inputs, exact penalty)
    sol = parampc.solve_qp(qp)                             # sol.x, sol.u, sol.status, sol.kkt, sol.iters

The problem (time-varying; N stages, n states, m inputs, M inequality rows per stage, MN terminal rows):

    min  sum_t 1/2 x_t'Q_t x_t + u_t'S_t x_t + 1/2 u_t'R_t u_t + q_t'x_t + r_t'u_t  +  1/2 x_N'QN x_N + qN'x_N
    s.t. x_0 = x0,   x_{t+1} = A_t x_t + B_t u_t + c_t,   C_t x_t + D_t u_t <= d_t,   CN x_N <= dN

It needs R_t > 0 (or R_t + D_t'W D_t > 0 for every positive diagonal W); Q_t and QN may be semidefinite.
"""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from .core import mtv, mv

STATUS = {0: 'solved', 1: 'iteration limit', 2: 'stopped (mu_min) or numerical breakdown'}


@dataclasses.dataclass(frozen=True)
class OCPQP:
    """An MPC-structured QP (see the module docstring). A pytree of arrays, so it can be traced, vmapped and
    passed through jax.jit; build one with `make_qp`."""
    A: jax.Array        # (N, n, n)
    B: jax.Array        # (N, n, m)
    c: jax.Array        # (N, n)
    Q: jax.Array        # (N, n, n)
    S: jax.Array        # (N, m, n)
    R: jax.Array        # (N, m, m)
    q: jax.Array        # (N, n)
    r: jax.Array        # (N, m)
    QN: jax.Array       # (n, n)
    qN: jax.Array       # (n,)
    C: jax.Array        # (N, M, n)
    D: jax.Array        # (N, M, m)
    d: jax.Array        # (N, M)
    x0: jax.Array       # (n,)
    CN: jax.Array       # (MN, n)
    dN: jax.Array       # (MN,)

    @property
    def N(self):
        return self.A.shape[-3]

    @property
    def nx(self):
        return self.A.shape[-1]

    @property
    def nu(self):
        return self.B.shape[-1]

    @property
    def M(self):
        return self.C.shape[-2]

    @property
    def MN(self):
        return self.CN.shape[-2]

    def replace(self, **fields):
        return dataclasses.replace(self, **fields)


@dataclasses.dataclass(frozen=True)
class Solution:
    """The solution of an OCPQP: primal (x, u), multipliers (nu of the dynamics and the initial condition, lam of
    the stage rows, lamN of the terminal rows), slacks (s, sN) and the solve's statistics. A pytree, so it can
    warm-start the next solve: solve_qp(qp, warm=sol.shift()).

    status 0 means the solver's own test passed: the KKT score of the problem it solved internally is <= tol.
    `kkt` measures this QP itself, so for a solved QP it can sit a small factor above tol (a terminal row's
    residual is split in two by the internal reformulation, and a batch averages the duality measure over its
    QPs)."""
    x: jax.Array        # (N+1, n)
    u: jax.Array        # (N, m)
    nu: jax.Array       # (N+1, n)
    lam: jax.Array      # (N, M)
    s: jax.Array        # (N, M)
    lamN: jax.Array     # (MN,)
    sN: jax.Array       # (MN,)
    iters: jax.Array    # IPM iterations (shared by a batch on the 'scan' and 'cuda' backends)
    kkt: jax.Array      # max(duality measure, largest KKT residual) of this point on the QP's own rows
    status: jax.Array   # 0 solved, 1 iteration limit, 2 stopped at mu_min or numerical breakdown (see STATUS)

    def shift(self):
        """The receding-horizon warm start for the next MPC step: drop the first stage, repeat the last."""
        sh = lambda Y: jnp.concatenate([Y[1:], Y[-1:]])
        return dataclasses.replace(self, x=sh(self.x), u=sh(self.u), nu=sh(self.nu), lam=sh(self.lam), s=sh(self.s))


for _cls in (OCPQP, Solution):
    jax.tree_util.register_dataclass(_cls, data_fields=[f.name for f in dataclasses.fields(_cls)], meta_fields=[])


# ---------------------------------------------------------------- construction ---
def make_qp(A, B, Q, R, N, x0, QN=None, S=None, q=None, r=None, qN=None, c=None, C=None, D=None, d=None,
            CN=None, dN=None, x_ref=None, u_ref=None):
    """An OCPQP from stage-invariant or stage-varying data: arrays without the leading N axis are broadcast to the
    N stages. Defaults: QN = Q (of the last stage), S = 0, c = 0, no inequality rows. x_ref ((N+1, n) or (n,)) and
    u_ref ((N, m) or (m,)) add the tracking terms q = -Q x_ref, r = -R u_ref, qN = -QN x_ref[N] to whatever q, r, qN
    are given."""
    A, B = jnp.asarray(A, jnp.float64), jnp.asarray(B, jnp.float64)
    n, m = A.shape[-1], B.shape[-1]
    st = lambda v, shape: jnp.broadcast_to(jnp.asarray(v, jnp.float64), (N,) + shape)
    Q, R = st(Q, (n, n)), st(R, (m, m))
    QN = Q[-1] if QN is None else jnp.asarray(QN, jnp.float64)
    q = st(0.0 if q is None else q, (n,))
    r = st(0.0 if r is None else r, (m,))
    qN = jnp.zeros(n) if qN is None else jnp.asarray(qN, jnp.float64)
    if x_ref is not None:
        xr = jnp.broadcast_to(jnp.asarray(x_ref, jnp.float64), (N + 1, n))
        q, qN = q - mv(Q, xr[:-1]), qN - QN @ xr[-1]
    if u_ref is not None:
        r = r - mv(R, jnp.broadcast_to(jnp.asarray(u_ref, jnp.float64), (N, m)))
    if C is None and D is None:
        C, D, d = jnp.zeros((N, 0, n)), jnp.zeros((N, 0, m)), jnp.zeros((N, 0))
    else:
        M = (jnp.asarray(C) if C is not None else jnp.asarray(D)).shape[-2]
        C = st(jnp.zeros((M, n)) if C is None else C, (M, n))
        D = st(jnp.zeros((M, m)) if D is None else D, (M, m))
        d = st(d, (M,))
    CN = jnp.zeros((0, n)) if CN is None else jnp.asarray(CN, jnp.float64)
    dN = jnp.zeros((0,)) if dN is None else jnp.asarray(dN, jnp.float64)
    return OCPQP(A=st(A, (n, n)), B=st(B, (n, m)), c=st(0.0 if c is None else c, (n,)), Q=Q,
                 S=st(0.0 if S is None else S, (m, n)), R=R, q=q, r=r, QN=QN, qN=qN, C=C, D=D, d=d,
                 x0=jnp.asarray(x0, jnp.float64), CN=CN, dN=dN)


def box_rows(lo, hi, size):
    """Rows e_j'v <= hi_j and -e_j'v <= -lo_j for the finite entries of the bounds lo, hi (scalars, (size,) or
    stage-varying (..., size) arrays; decided from concrete values). Returns (rows, bounds): lists of unit
    vectors and of the matching bound arrays."""
    rows, bounds = [], []
    for bound, sign in ((hi, 1.0), (lo, -1.0)):
        if bound is None:
            continue
        b = np.broadcast_to(np.asarray(bound, np.float64), np.broadcast_shapes(np.shape(bound), (size,)))
        flat = b.reshape(-1, size)
        for j in range(size):
            if np.all(np.isfinite(flat[:, j])):
                e = np.zeros(size)
                e[j] = sign
                rows.append(e)
                bounds.append(sign * b[..., j])
    return rows, bounds


def add_bounds(qp, x_min=None, x_max=None, u_min=None, u_max=None, xN_min=None, xN_max=None, include_x0=False):
    """Append box constraints as inequality rows. Bounds are scalars, (n,)/(m,) arrays or stage-varying
    (N, n)/(N, m) arrays; infinite entries are skipped (the bounds must be concrete, not traced). x bounds apply to
    x_1..x_{N-1} (and to x_0 if include_x0: x_0 is fixed, so a bound on it is either void or infeasible);
    xN_min/xN_max to x_N (terminal rows)."""
    N, n, m = qp.N, qp.nx, qp.nu
    xr, xb = box_rows(x_min, x_max, n)
    ur, ub = box_rows(u_min, u_max, m)
    C_new = [np.concatenate([e, np.zeros(m)]) for e in xr] + [np.concatenate([np.zeros(n), e]) for e in ur]
    d_new = [np.array(np.broadcast_to(b, (N,))) for b in xb + ub]
    if C_new:
        rows = np.broadcast_to(np.array(C_new), (N, len(C_new), n + m)).copy()
        d_rows = np.stack(d_new, axis=1)
        if not include_x0 and xr:                                  # stage 0: the state rows become 0 <= 1
            rows[0, :len(xr)] = 0.0
            d_rows[0, :len(xr)] = 1.0
        qp = qp.replace(C=jnp.concatenate([qp.C, jnp.asarray(rows[:, :, :n])], axis=1),
                        D=jnp.concatenate([qp.D, jnp.asarray(rows[:, :, n:])], axis=1),
                        d=jnp.concatenate([qp.d, jnp.asarray(d_rows)], axis=1))
    nr, nb = box_rows(xN_min, xN_max, n)
    if nr:
        qp = qp.replace(CN=jnp.concatenate([qp.CN, jnp.asarray(np.array(nr))]),
                        dN=jnp.concatenate([qp.dN, jnp.asarray(np.array(nb, dtype=np.float64))]))
    return qp


def soften(qp, rows, l1=1e3, l2=1e-2):
    """Make the stage rows `rows` (indices) soft: row i becomes C x + D u - sigma_i <= d with a slack input
    sigma_i >= 0 penalized by l1 sigma_i + l2/2 sigma_i^2 (an exact penalty when l1 exceeds the row's multiplier).
    The slacks are appended to the inputs: solution.u[:, m:] are the slacks."""
    rows = list(rows)
    N, n, m, M, k = qp.N, qp.nx, qp.nu, qp.M, len(rows)
    E = np.zeros((M, k))
    E[rows, range(k)] = -1.0
    st = lambda a: jnp.broadcast_to(jnp.asarray(a), (N,) + np.shape(a))
    D = jnp.concatenate([qp.D, st(E)], axis=2)
    D = jnp.concatenate([D, st(np.concatenate([np.zeros((k, m)), -np.eye(k)], axis=1))], axis=1)   # sigma >= 0
    C = jnp.concatenate([qp.C, jnp.zeros((N, k, n))], axis=1)
    d = jnp.concatenate([qp.d, jnp.zeros((N, k))], axis=1)
    R = jnp.zeros((N, m + k, m + k)).at[:, :m, :m].set(qp.R).at[:, m:, m:].set(l2 * jnp.eye(k))
    return qp.replace(B=jnp.concatenate([qp.B, jnp.zeros((N, n, k))], axis=2),
                      S=jnp.concatenate([qp.S, jnp.zeros((N, k, n))], axis=1), R=R,
                      r=jnp.concatenate([qp.r, jnp.broadcast_to(jnp.asarray(l1, jnp.float64), (N, k))], axis=1),
                      C=C, D=D, d=d)


# --------------------------------------------------------------------- checks ---
def kkt_score(qp, sol):
    """max(duality measure, largest KKT residual) of a Solution, on the QP's own variables and rows."""
    x, u, nu, lam, s = sol.x, sol.u, sol.nu, sol.lam, sol.s
    xt = x[:-1]
    res = [x[0] - qp.x0,
           nu[:-1] - mv(qp.Q, xt) - mtv(qp.S, u) - qp.q - mtv(qp.A, nu[1:]) - mtv(qp.C, lam),
           nu[-1] - qp.QN @ x[-1] - qp.qN - qp.CN.T @ sol.lamN,
           mv(qp.R, u) + mv(qp.S, xt) + qp.r + mtv(qp.B, nu[1:]) + mtv(qp.D, lam),
           x[1:] - mv(qp.A, xt) - mv(qp.B, u) - qp.c,
           mv(qp.C, xt) + mv(qp.D, u) + s - qp.d,
           qp.CN @ x[-1] + sol.sN - qp.dN]
    feas = jnp.max(jnp.stack([jnp.max(jnp.abs(r), initial=0.0) for r in res]))
    rows = lam.size + sol.lamN.size
    mu = (jnp.sum(lam * s) + jnp.sum(sol.lamN * sol.sN)) / max(rows, 1)
    return jnp.maximum(mu, feas)
