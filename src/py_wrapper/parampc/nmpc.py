"""Nonlinear MPC: sequential quadratic programming (or the real-time iteration) on top of `solve_qp`.

    ocp = OCP(nx=4, nu=1, N=50,
              dynamics=rk4(f, dt=0.05),                       # discrete-time x_{t+1} = F(x, u, p)
              stage_cost=lambda x, u, p: ..., terminal_cost=lambda x, p: ...,
              constraints=lambda x, u, p: g,                  # g(x, u, p) <= 0, optional
              u_min=..., u_max=..., x_min=..., x_max=...,     # bounds (exact, linear)
              soft=[[0, 1]], soft_l1=1e3, soft_l2=1e-2)       # groups of constraints sharing one slack
    ctrl = NMPC(ocp, sqp_iters=3)
    state = ctrl.init(x0, params)                             # initial guess: u_guess rolled out
    u0, state, info = ctrl.step(state, x0, params)            # one MPC step: SQP from the warm start, then shift
    xs, us, infos = ctrl.closed_loop(plant, x0, K=400, params_fn=params_fn)    # a closed loop as one lax.scan

params: a pytree whose leaves have a leading axis N+1 (stage t gets p_t, the terminal cost p_N), or None.

Each SQP iteration linearizes the dynamics and constraints at the current point (X, U) and expands the costs to
second order: their Hessians plus a Levenberg-Marquardt term lm I, the proximal term lm/2 |z - Z|^2
(hessian='convexify' first lifts negative curvature to zero, so non-convex costs give convex QPs). It solves the
QP with `solve_qp` (multiple shooting: the QP's states are kept, never re-rolled; the IPM starts from (X, U) with
the previous multipliers floored at warm_floor), projects the inputs onto their bounds, and takes the best step
of a backtracking ladder on the l1 merit function cost + merit_weight (sum |x_{t+1} - F(x_t, u_t)| +
sum max(g, 0)); a step that does not lower it is rejected. Soft constraints get slack inputs (one per group,
penalized by soft_l1 s + soft_l2/2 s^2, an exact penalty when soft_l1 exceeds the multipliers); they are part of
the state's U (U[:, nu:]).

Everything is jit-compatible, and jax.vmap over states and initial states batches the QPs into one long horizon.
"""
import dataclasses
from typing import Any, Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from .qp import OCPQP, Solution, box_rows
from .solver import solve_qp


@dataclasses.dataclass(frozen=True, eq=False)
class OCP:
    """A nonlinear optimal control problem over N stages (see the module docstring). Bounds are scalars or arrays
    (infinite entries are ignored); x bounds apply to x_1..x_{N-1}, xN bounds to x_N."""
    nx: int
    nu: int
    N: int
    dynamics: Callable                                  # (x, u, p) -> x_next
    stage_cost: Callable                                # (x, u, p) -> scalar
    terminal_cost: Optional[Callable] = None            # (x, p) -> scalar
    constraints: Optional[Callable] = None              # (x, u, p) -> g (ng,), g <= 0
    terminal_constraints: Optional[Callable] = None     # (x, p) -> gN (ngN,), gN <= 0 (hard)
    u_min: Any = None
    u_max: Any = None
    x_min: Any = None
    x_max: Any = None
    xN_min: Any = None
    xN_max: Any = None
    soft: Sequence[Sequence[int]] = ()                  # groups of constraint indices sharing a slack
    soft_l1: float = 1e3
    soft_l2: float = 1e-2
    u_guess: Any = None                                 # initial input guess, (nu,) or (N, nu)

    @property
    def ns(self):
        """Number of slack inputs (one per soft group)."""
        return len(self.soft)


def _bound_rows(lo, hi, size):
    """Static rows e_j'v <= hi_j and -e_j'v <= -lo_j for the finite entries of constant bounds: (E (k, size),
    b (k,))."""
    rows, b = box_rows(lo, hi, size)
    if not rows:
        return np.zeros((0, size)), np.zeros(0)
    return np.array(rows), np.array([float(np.asarray(v).reshape(-1)[0]) for v in b])


@dataclasses.dataclass(frozen=True)
class State:
    """The SQP iterate: states X (N+1, nx), inputs including slacks U (N, nu + ns), and the last QP's multipliers
    (the warm start of the next QP). A pytree."""
    X: jax.Array
    U: jax.Array
    lam: jax.Array
    lamN: jax.Array


jax.tree_util.register_dataclass(State, data_fields=['X', 'U', 'lam', 'lamN'], meta_fields=[])


class NMPC:
    """SQP-based nonlinear MPC for an OCP (module docstring). sqp_iters = 1 is the real-time iteration (RTI).

    lm: Levenberg-Marquardt weight; merit_weight and ladder: the line search; hessian: 'convexify' or 'exact';
    backend, qp_tol, qp_max_iter, warm_floor and ipm_options: the QP solves (see `solve_qp`)."""

    def __init__(self, ocp, sqp_iters=3, lm=0.5, merit_weight=500.0, ladder=(1.0, 0.5, 0.25, 0.125), qp_tol=1e-6,
                 qp_max_iter=30, warm_floor=1e-3, backend='auto', hessian='convexify', ipm_options=None):
        if hessian not in ('convexify', 'exact'):
            raise ValueError("hessian must be 'convexify' or 'exact'")
        self.ocp, self.sqp_iters, self.lm, self.merit_weight = ocp, sqp_iters, lm, merit_weight
        self.convexify = hessian == 'convexify'
        self.ladder = jnp.asarray(ladder, jnp.float64)
        self.qp_opts = dict(tol=qp_tol, max_iter=qp_max_iter, backend=backend, ipm_options=ipm_options)
        self.warm_floor = warm_floor
        nx, nu, ns = ocp.nx, ocp.nu, ocp.ns
        self.nv = nu + ns
        self.Ex, self.bx = _bound_rows(ocp.x_min, ocp.x_max, nx)
        self.Eu, self.bu = _bound_rows(ocp.u_min, ocp.u_max, nu)
        self.ExN, self.bxN = _bound_rows(ocp.xN_min, ocp.xN_max, nx)
        soft_idx = [i for grp in ocp.soft for i in grp]
        self.soft_idx = np.array(soft_idx, dtype=int)
        S = np.zeros((len(soft_idx), ns))
        for k, grp in enumerate(ocp.soft):
            for i in grp:
                S[soft_idx.index(i), k] = 1.0
        self.S = S                                           # soft row -> its slack
        self.u_lo = jnp.asarray(np.concatenate([np.broadcast_to(-np.inf if ocp.u_min is None else ocp.u_min, (nu,)),
                                                np.zeros(ns)]), jnp.float64)
        self.u_hi = jnp.asarray(np.concatenate([np.broadcast_to(np.inf if ocp.u_max is None else ocp.u_max, (nu,)),
                                                np.full(ns, np.inf)]), jnp.float64)
        self._jit_solve = jax.jit(self._solve)
        self._jit_step = jax.jit(self._step)
        self._loops = {}

    # ------------------------------------------------------------- the problem ---
    def _p(self, params, t):
        return None if params is None else jax.tree.map(lambda a: a[t], params)

    def _cost(self, x, v, p):
        """Stage cost including the slack penalty."""
        o, ns = self.ocp, self.ocp.ns
        J = o.stage_cost(x, v[:o.nu], p)
        if ns:
            sig = v[o.nu:]
            J = J + o.soft_l1 * jnp.sum(sig) + 0.5 * o.soft_l2 * jnp.sum(sig * sig)
        return J

    def _term(self, x, p):
        return 0.0 if self.ocp.terminal_cost is None else self.ocp.terminal_cost(x, p)

    def _g(self, x, v, p):
        """All inequality rows as h(x, v) <= 0: the hard constraints, and the soft ones minus their slacks."""
        o = self.ocp
        if o.constraints is None:
            return jnp.zeros(0)
        g = o.constraints(x, v[:o.nu], p)
        hard = np.array([i for i in range(g.shape[0]) if i not in set(self.soft_idx.tolist())], dtype=int)
        parts = [g[hard]] if len(hard) else []
        if len(self.soft_idx):
            parts.append(g[self.soft_idx] - jnp.asarray(self.S) @ v[o.nu:])
        return jnp.concatenate(parts) if parts else jnp.zeros(0)

    def cost(self, X, U, params):
        """The objective at (X, U), slack penalty included."""
        N = self.ocp.N
        stage = jax.vmap(lambda t, x, v: self._cost(x, v, self._p(params, t)))(jnp.arange(N), X[:-1], U)
        return jnp.sum(stage) + self._term(X[-1], self._p(params, N))

    def merit(self, X, U, params):
        """The l1 merit function: cost + merit_weight (dynamics defects + constraint violations)."""
        o = self.ocp
        P = jnp.arange(o.N)
        F = jax.vmap(lambda t, x, v: o.dynamics(x, v[:o.nu], self._p(params, t)))(P, X[:-1], U)
        viol = jnp.sum(jnp.abs(X[1:] - F))
        if o.constraints is not None:
            viol += jnp.sum(jnp.maximum(jax.vmap(lambda t, x, v: self._g(x, v, self._p(params, t)))(P, X[:-1], U), 0.0))
        if o.terminal_constraints is not None:
            viol += jnp.sum(jnp.maximum(o.terminal_constraints(X[-1], self._p(params, o.N)), 0.0))
        return self.cost(X, U, params) + self.merit_weight * viol

    def _psd(self, H):
        """hessian='convexify': lift the negative eigenvalues of the cost Hessians H (..., k, k) to 0
        (H + V max(-lambda, 0) V'; the Levenberg-Marquardt term then makes them positive definite). The
        eigendecompositions run only if a Cholesky test finds an indefinite H (H + 1e-12 (1 + max|H|) I not positive
        definite): a convex cost keeps its Hessians, and its iterates, bit for bit, at the price of one batched
        Cholesky."""
        if not self.convexify:
            return H
        H = 0.5 * (H + jnp.swapaxes(H, -1, -2))
        eye = jnp.eye(H.shape[-1])

        def lift(H):
            lam, V = jnp.linalg.eigh(H)
            return H + (V * jnp.maximum(-lam, 0.0)[..., None, :]) @ jnp.swapaxes(V, -1, -2)

        shift = 1e-12 * (1.0 + jnp.max(jnp.abs(H)))
        convex = jnp.all(jnp.isfinite(jnp.linalg.cholesky(H + shift * eye)))
        return lax.cond(convex, lambda H: H, lift, H)

    # ---------------------------------------------------------- the subproblem ---
    def qp(self, X, U, params, x0):
        """The SQP subproblem at (X, U) as an OCPQP (in absolute variables, the proximal term included)."""
        o, lm, N, nx, nv = self.ocp, self.lm, self.ocp.N, self.ocp.nx, self.nv

        def stage(t, x, v):
            p = self._p(params, t)
            F = lambda x, v: o.dynamics(x, v[:o.nu], p)
            A, Bv = jax.jacfwd(F, 0)(x, v), jax.jacfwd(F, 1)(x, v)
            c = F(x, v) - A @ x - Bv @ v
            z = jnp.concatenate([x, v])
            ell = lambda z: self._cost(z[:nx], z[nx:], p)
            h = self._g(x, v, p)
            Gx, Gv = jax.jacfwd(lambda x: self._g(x, v, p))(x), jax.jacfwd(lambda v: self._g(x, v, p))(v)
            d = -h + Gx @ x + Gv @ v
            return A, Bv, c, z, jax.grad(ell)(z), jax.hessian(ell)(z), Gx, Gv, d

        A, B, c, Z, G, Hc, Gx, Gv, dg = jax.vmap(stage)(jnp.arange(N), X[:-1], U)
        Hc = self._psd(Hc)
        H = Hc + lm * jnp.eye(nx + nv)                          # cost Hessian + Levenberg-Marquardt
        lin = G - jnp.einsum('tij,tj->ti', Hc, Z) - lm * Z      # second-order expansion at Z, proximal term
        Q, S, R, q, r = H[:, :nx, :nx], H[:, nx:, :nx], H[:, nx:, nx:], lin[:, :nx], lin[:, nx:]
        # inequality rows per stage: state bounds (not at t = 0), input bounds, constraints, slacks >= 0
        rows_C, rows_D, rows_d = [], [], []
        if len(self.bx):
            Ex = jnp.broadcast_to(jnp.asarray(self.Ex), (N,) + self.Ex.shape)
            bx = jnp.broadcast_to(jnp.asarray(self.bx), (N, len(self.bx)))
            Ex, bx = Ex.at[0].set(0.0), bx.at[0].set(1.0)
            rows_C.append(Ex), rows_D.append(jnp.zeros((N, len(self.bx), nv))), rows_d.append(bx)
        if len(self.bu):
            Eu = np.concatenate([self.Eu, np.zeros((len(self.bu), o.ns))], axis=1)
            rows_C.append(jnp.zeros((N, len(self.bu), nx)))
            rows_D.append(jnp.broadcast_to(jnp.asarray(Eu), (N,) + Eu.shape))
            rows_d.append(jnp.broadcast_to(jnp.asarray(self.bu), (N, len(self.bu))))
        if Gx.shape[1]:
            rows_C.append(Gx), rows_D.append(Gv), rows_d.append(dg)
        if o.ns:
            Es = np.concatenate([np.zeros((o.ns, o.nu)), -np.eye(o.ns)], axis=1)
            rows_C.append(jnp.zeros((N, o.ns, nx)))
            rows_D.append(jnp.broadcast_to(jnp.asarray(Es), (N,) + Es.shape))
            rows_d.append(jnp.zeros((N, o.ns)))
        cat = lambda rows, shape: jnp.concatenate(rows, axis=1) if rows else jnp.zeros(shape)
        C, D, d = cat(rows_C, (N, 0, nx)), cat(rows_D, (N, 0, nv)), cat(rows_d, (N, 0))
        # terminal stage: cost expansion, bounds, constraints
        pN, xN = self._p(params, N), X[-1]
        gN, HN = jax.grad(lambda x: self._term(x, pN))(xN), self._psd(jax.hessian(lambda x: self._term(x, pN))(xN))
        QN = HN + lm * jnp.eye(nx)
        qN = gN - HN @ xN - lm * xN
        CN, dN = [], []
        if len(self.bxN):
            CN.append(jnp.asarray(self.ExN)), dN.append(jnp.asarray(self.bxN))
        if o.terminal_constraints is not None:
            h = o.terminal_constraints(xN, pN)
            G = jax.jacfwd(lambda x: o.terminal_constraints(x, pN))(xN)
            CN.append(G), dN.append(-h + G @ xN)
        CN = jnp.concatenate(CN) if CN else jnp.zeros((0, nx))
        dN = jnp.concatenate(dN) if dN else jnp.zeros(0)
        return OCPQP(A=A, B=B, c=c, Q=Q, S=S, R=R, q=q, r=r, QN=QN, qN=qN, C=C, D=D, d=d, x0=x0, CN=CN, dN=dN)

    # --------------------------------------------------------------------- SQP ---
    def sqp_iteration(self, state, x0, params):
        """One SQP iteration from `state` (module docstring). Returns (state, QP solution)."""
        X, U = state.X.at[0].set(x0), state.U
        sub = self.qp(X, U, params, x0)
        warm = Solution(x=X, u=U, nu=jnp.zeros_like(X), lam=state.lam, s=jnp.zeros_like(state.lam),
                        lamN=state.lamN, sN=jnp.zeros_like(state.lamN), iters=jnp.zeros((), jnp.int32),
                        kkt=jnp.zeros(()), status=jnp.zeros((), jnp.int32))
        sol = solve_qp(sub, warm, floor=self.warm_floor, **self.qp_opts)
        Xn, Un = sol.x, jnp.clip(sol.u, self.u_lo, self.u_hi)
        phis = jax.vmap(lambda a: self.merit(X + a * (Xn - X), U + a * (Un - U), params))(self.ladder)
        j = jnp.argmin(phis)
        a = jnp.where(phis[j] < self.merit(X, U, params), self.ladder[j], 0.0)
        return State(X=X + a * (Xn - X), U=U + a * (Un - U), lam=sol.lam, lamN=sol.lamN), sol

    def _solve(self, state, x0, params):
        def body(i, carry):
            st, _, iters = carry
            st, sol = self.sqp_iteration(st, x0, params)
            return st, sol.kkt, iters + sol.iters
        st, kkt, iters = lax.fori_loop(0, self.sqp_iters, body, (state, jnp.zeros(()), jnp.zeros((), jnp.int32)))
        info = dict(qp_iters=iters, qp_kkt=kkt, cost=self.cost(st.X, st.U, params),
                    merit=self.merit(st.X, st.U, params))
        return st, info

    def solve(self, state, x0, params=None):
        """sqp_iters SQP iterations from `state` with x_0 = x0. Returns (state, info)."""
        return self._jit_solve(state, x0, params)

    @staticmethod
    def shift(state):
        """Receding-horizon warm start: drop the first stage, repeat the last (states, inputs, multipliers)."""
        sh = lambda Y: jnp.concatenate([Y[1:], Y[-1:]])
        return State(X=sh(state.X), U=sh(state.U), lam=sh(state.lam), lamN=state.lamN)

    def _step(self, state, x0, params):
        st, info = self._solve(state, x0, params)
        return st.U[0, :self.ocp.nu], self.shift(st), info

    def step(self, state, x0, params=None):
        """One MPC step: solve from the warm start; returns (u0, the shifted state for the next step, info)."""
        return self._jit_step(state, x0, params)

    def init(self, x0, params=None, X=None, U=None):
        """An initial state: U = u_guess (or 0) with zero slacks, X its rollout from x0 (or the given X, U)."""
        o = self.ocp
        if U is None:
            ug = jnp.zeros(o.nu) if o.u_guess is None else jnp.asarray(o.u_guess, jnp.float64)
            U = jnp.concatenate([jnp.broadcast_to(ug, (o.N, o.nu)), jnp.zeros((o.N, o.ns))], axis=1)
        if X is None:
            def roll(x, tv):
                t, v = tv
                xn = o.dynamics(x, v[:o.nu], self._p(params, t))
                return xn, xn
            _, xs = lax.scan(roll, jnp.asarray(x0, jnp.float64), (jnp.arange(o.N), U))
            X = jnp.concatenate([jnp.asarray(x0, jnp.float64)[None], xs])
        p0, pN = self._p(params, 0), self._p(params, o.N)
        size = lambda f: jax.eval_shape(f).shape[0]
        ng = 0 if o.constraints is None else size(lambda: o.constraints(X[0], U[0, :o.nu], p0))
        ngN = 0 if o.terminal_constraints is None else size(lambda: o.terminal_constraints(X[-1], pN))
        M, MN = len(self.bx) + len(self.bu) + ng + o.ns, len(self.bxN) + ngN
        return State(X=X, U=U, lam=jnp.ones((o.N, M)), lamN=jnp.ones(MN))

    def closed_loop(self, plant, x0, K, params_fn=None, state=None):
        """K steps of receding-horizon control as one lax.scan. plant(x, u, k) -> next state; params_fn(k) -> the
        horizon's params at step k (None: no params). Returns (xs (K, nx), us (K, nu), infos). The compiled loop is
        cached per (plant, params_fn, K) object identity: pass the same function objects to reuse it (a new lambda
        per call recompiles every time)."""
        x0 = jnp.asarray(x0, jnp.float64)
        pf = (lambda k: None) if params_fn is None else params_fn
        state = self.init(x0, pf(0)) if state is None else state
        key = (plant, params_fn, K)
        if key not in self._loops:
            def run(state, x0):
                def body(carry, k):
                    x, st = carry
                    u, st, info = self._step(st, x, pf(k))
                    return (plant(x, u, k), st), (x, u, info)
                return lax.scan(body, (x0, state), jnp.arange(K))[1]
            self._loops[key] = jax.jit(run)
        return self._loops[key](state, x0)
