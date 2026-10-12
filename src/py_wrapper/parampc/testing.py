"""Independent checks of solver output: dense reference solutions, KKT certificates, random test problems.

The dense references are built in NumPy from the problem data and share no code with the solver.

    dense_qp           the MPC QP as one dense QP
    kkt_certificate    optimality residuals of a primal-dual point on the dense QP
    solve_equality_qp  direct KKT solve of a dense QP without inequality rows
    newton_matrix      the full Newton matrix of the core's KKT system (autodiff of its residual)
    random_problem     a random time-varying problem in the core's data layout, and a starting point
"""
import jax
import numpy as np
from jax.flatten_util import ravel_pytree

FIELDS = ('A', 'B', 'c', 'Q', 'S', 'R', 'q', 'r', 'QN', 'qN', 'C', 'D', 'd', 'x0')


def _data(problem):
    """The problem's fields as NumPy arrays, from an OCPQP or a core data dict (no terminal rows)."""
    get = problem.get if isinstance(problem, dict) else lambda k, default=None: getattr(problem, k, default)
    d = {k: np.asarray(get(k), np.float64) for k in FIELDS}
    n = d['A'].shape[-1]
    d['CN'] = np.asarray(get('CN', np.zeros((0, n))), np.float64)
    d['dN'] = np.asarray(get('dN', np.zeros(0)), np.float64)
    return d


def dense_qp(problem):
    """min 1/2 w'Hw + g'w  s.t.  Aeq w = beq,  G w <= h,  w = (x_0..x_N, u_0..u_{N-1}), for an OCPQP or a core data
    dict. The equality rows are (x_0 = x0, x_1 - A_0 x_0 - B_0 u_0 = c_0, ...), so their multipliers are
    nu = (nu_0, ..., nu_N) in the solver's sign convention: stationarity reads H w + g - Aeq'nu + G'lam = 0. The rows
    of G are the stage rows, stage by stage, then the terminal rows."""
    d = _data(problem)
    N, n, m = d['B'].shape
    M, MN = d['C'].shape[1], d['CN'].shape[0]
    nw = (N + 1) * n + N * m
    ix = lambda t: slice(t * n, (t + 1) * n)
    iu = lambda t: slice((N + 1) * n + t * m, (N + 1) * n + (t + 1) * m)
    H, g = np.zeros((nw, nw)), np.zeros(nw)
    Aeq, beq = np.zeros(((N + 1) * n, nw)), np.zeros((N + 1) * n)
    G, h = np.zeros((N * M + MN, nw)), np.zeros(N * M + MN)
    Aeq[ix(0), ix(0)], beq[ix(0)] = np.eye(n), d['x0']
    for t in range(N):
        H[ix(t), ix(t)], H[iu(t), iu(t)] = d['Q'][t], d['R'][t]
        H[iu(t), ix(t)], H[ix(t), iu(t)] = d['S'][t], d['S'][t].T
        g[ix(t)], g[iu(t)] = d['q'][t], d['r'][t]
        Aeq[ix(t + 1), ix(t + 1)] = np.eye(n)
        Aeq[ix(t + 1), ix(t)], Aeq[ix(t + 1), iu(t)], beq[ix(t + 1)] = -d['A'][t], -d['B'][t], d['c'][t]
        rows = slice(t * M, (t + 1) * M)
        G[rows, ix(t)], G[rows, iu(t)], h[rows] = d['C'][t], d['D'][t], d['d'][t]
    H[ix(N), ix(N)], g[ix(N)] = d['QN'], d['qN']
    G[N * M:, ix(N)], h[N * M:] = d['CN'], d['dN']
    return dict(H=H, g=g, Aeq=Aeq, beq=beq, G=G, h=h)


def pack(point):
    """(w, nu, lam) of a Solution or a core iterate dict, in the ordering of dense_qp."""
    get = point.get if isinstance(point, dict) else lambda k, default=None: getattr(point, k, default)
    a = lambda v: np.asarray(v, np.float64).ravel()
    lamN = get('lamN', np.zeros(0))
    return np.concatenate([a(get('x')), a(get('u'))]), a(get('nu')), np.concatenate([a(get('lam')), a(lamN)])


def _dense(problem):
    return problem if isinstance(problem, dict) and 'H' in problem else dense_qp(problem)


def kkt_certificate(problem, point):
    """Optimality residuals of a primal-dual point (a Solution or a core iterate dict) on the dense QP of `problem`
    (an OCPQP, a core data dict, or dense_qp's output): for a convex QP, stationarity, equality, inequality,
    dual_sign and complementarity all near zero certify optimality. Also returns the objective."""
    D = _dense(problem)
    w, nu, lam = pack(point)
    slack = D['h'] - D['G'] @ w
    return dict(stationarity=np.max(np.abs(D['H'] @ w + D['g'] - D['Aeq'].T @ nu + D['G'].T @ lam)),
                equality=np.max(np.abs(D['Aeq'] @ w - D['beq'])),
                inequality=max(0.0, -np.min(slack, initial=0.0)),
                dual_sign=max(0.0, -np.min(lam, initial=0.0)),
                complementarity=np.max(np.abs(lam * slack), initial=0.0),
                objective=0.5 * w @ D['H'] @ w + D['g'] @ w)


def objective(problem, point):
    """The QP's objective at a point."""
    D = _dense(problem)
    w = pack(point)[0]
    return 0.5 * w @ D['H'] @ w + D['g'] @ w


def solve_equality_qp(D):
    """Direct solve of [[H, -Aeq'], [Aeq, 0]] (w, nu) = (-g, beq) for a dense QP (from dense_qp) without
    inequality rows. Returns (w, nu)."""
    H, Aeq = D['H'], D['Aeq']
    nw, ne = H.shape[0], Aeq.shape[0]
    K = np.block([[H, -Aeq.T], [Aeq, np.zeros((ne, ne))]])
    sol = np.linalg.solve(K, np.concatenate([-D['g'], D['beq']]))
    return sol[:nw], sol[nw:]


def newton_matrix(data, z, tau):
    """(J, F, unravel): the Newton system J dz = -F of the core's perturbed KKT system at z, as dense arrays."""
    from .core import residuals
    flat, unravel = ravel_pytree(z)
    F = lambda v: ravel_pytree(residuals(data, unravel(v), tau))[0]
    return np.asarray(jax.jacfwd(F)(flat)), np.asarray(F(flat)), unravel


def random_problem(N, n, m, M, seed=0, u_max=0.8, x_max=50.0):
    """A random stable LTV problem in the core's data layout, with input boxes (the first 2m rows, |u_j| <= u_max)
    and state boxes (|x_j| <= x_max). Returns (data, z0), z0 a cold starting point around the open-loop (u = 0)
    rollout."""
    import jax.numpy as jnp

    from .core import initial_iterate
    rng = np.random.default_rng(seed)
    A0 = rng.normal(size=(n, n))
    A0 = 0.9 * A0 / np.max(np.abs(np.linalg.eigvals(A0)))
    B0 = rng.normal(size=(n, m))
    A = A0 + 0.02 * rng.normal(size=(N, n, n))
    B = B0 + 0.02 * rng.normal(size=(N, n, m))
    c = 0.02 * rng.normal(size=(N, n))
    Lq = rng.normal(size=(N, n, n))
    Q = Lq @ np.swapaxes(Lq, -1, -2) / n + 0.5 * np.eye(n)
    Lr = rng.normal(size=(N, m, m))
    R = Lr @ np.swapaxes(Lr, -1, -2) / m + 0.5 * np.eye(m)
    q = 0.1 * rng.normal(size=(N, n))
    r = 0.1 * rng.normal(size=(N, m))
    C, D, d = np.zeros((N, M, n)), np.zeros((N, M, m)), np.zeros((N, M))
    for i in range(M):
        sign = 1.0 if i % 2 == 0 else -1.0
        if i < 2 * m:
            D[:, i, i // 2], d[:, i] = sign, u_max
        else:
            C[:, i, (i - 2 * m) % n], d[:, i] = sign, x_max
    x0 = 0.3 * rng.normal(size=n)
    data = {k: jnp.asarray(v) for k, v in dict(
        A=A, B=B, c=c, Q=Q, S=np.zeros((N, m, n)), R=R, q=q, r=r, QN=2.0 * Q[0], qN=q[0],
        C=C, D=D, d=d, x0=x0).items()}
    x = [x0]
    for t in range(N):
        x.append(A[t] @ x[t] + c[t])
    return data, initial_iterate(data, jnp.asarray(np.array(x)), jnp.zeros((N, m)))
