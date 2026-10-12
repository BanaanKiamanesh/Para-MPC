"""The LQ solve inside every Newton step, by associative scans over the horizon (Theta(log N) depth).

Matrix part (`factor_lq`): the cost-to-go matrices P_t. Each stage t defines a value-function element
a_t = (Abar_t, M_t, Qbar_t) with

    Abar_t = A_t - B_t Rt_t^-1 St_t,   M_t = B_t Rt_t^-1 B_t',   Qbar_t = Qt_t - St_t' Rt_t^-1 St_t,

and the terminal element is a_N = (0, 0, QN). Elements compose associatively (`vf_combine`, a1 earlier in time):

    (A1, C1, J1) (x) (A2, C2, J2) = (A2 (I + C1 J2)^-1 A1,
                                     A2 (I + C1 J2)^-1 C1 A2' + C2,
                                     A1' (I + J2 C1)^-1 J2 A1 + J1),

and P_t is the J part of the suffix product a_t (x) ... (x) a_N. One associative scan gives all of them. The
sequential alternative (`riccati_sweep`) is the classical Riccati recursion, Theta(N) depth.

Vector part (`solve_lq`): with the gains K_t and the closed loop A_t - B_t K_t, the costate offsets and the states
follow two affine recursions x_{t+1} = F_t x_t + f_t, each an associative scan over affine maps.
"""
import jax
import jax.numpy as jnp
from jax import lax

from .kkt import mtv, mv, sym


def tri_solve(L, B, lower=True):
    """Solve L X = B for a (batched) triangular L."""
    return jax.scipy.linalg.solve_triangular(L, B, lower=lower)


def spd_solve(L, B):
    """Solve (L L') X = B for the Cholesky factor L of a symmetric positive definite matrix."""
    return tri_solve(L.mT, tri_solve(L, B), lower=False)


# ----------------------------------------------------------------- matrix part ---
def stage_elements(lq, data):
    """The matrix parts (Abar_t, M_t, Qbar_t) of the stage elements, through the Cholesky factor Lr of Rt_t:
    M = U'U and Qbar = Qt - V'V are Gram forms (V = Lr^-1 St, U = Lr^-1 B')."""
    A, B = data['A'], data['B']
    Lr = jnp.linalg.cholesky(lq['Rt'])
    V, U = tri_solve(Lr, lq['St']), tri_solve(Lr, B.mT)
    return A - B @ tri_solve(Lr.mT, V, lower=False), sym(U.mT @ U), sym(lq['Qt'] - V.mT @ V)


def vf_combine(a1, a2):
    """Compose two value-function elements (A, C, J); a1 is earlier in time.

    Cholesky form, which keeps it accurate when J2 is huge (large barrier weights). With J2 = L L' and
    I + L' C1 L = R R' (>= I):
        (I + J2 C1)^-1 J2 = W'W,  W = R^-1 L'     (a Gram form: no cancellation)
        (I + C1 J2)^-1 = L^-T (R R')^-1 L'
    It needs J2 > 0, i.e. Qbar_t > 0 and QN > 0. For a semidefinite cost it returns NaN, and `factor_lq` falls
    back to `vf_combine_lu`."""
    A1, C1, J1 = a1
    A2, C2, J2 = a2
    n = A1.shape[-1]
    I = jnp.broadcast_to(jnp.eye(n, dtype=A1.dtype), A1.shape)
    L = jnp.linalg.cholesky(J2)
    Lt = L.mT
    R = jnp.linalg.cholesky(sym(I + Lt @ C1 @ L))
    W = tri_solve(R, Lt)
    Y = tri_solve(R, Lt @ jnp.concatenate([C1, A1], axis=-1))
    XR = tri_solve(Lt, tri_solve(R.mT, Y, lower=False), lower=False)       # (I + C1 J2)^-1 [C1, A1]
    H, XA1 = XR[..., :n], XR[..., n:]
    return A2 @ XA1, sym(A2 @ H @ A2.mT + C2), sym(A1.mT @ (W.mT @ W) @ A1 + J1)


def vf_combine_lu(a1, a2):
    """The same composition with LU solves. It needs no definiteness (I + C1 J2 is invertible whenever C1, J2 >= 0),
    so it handles a semidefinite cost such as output-only tracking; it is less accurate than `vf_combine` at huge
    barrier weights."""
    A1, C1, J1 = a1
    A2, C2, J2 = a2
    n = A1.shape[-1]
    I = jnp.broadcast_to(jnp.eye(n, dtype=A1.dtype), A1.shape)
    XR = jnp.linalg.solve(I + C1 @ J2, jnp.concatenate([C1, A1], axis=-1))
    G = jnp.linalg.solve(I + J2 @ C1, J2)                                   # (I + J2 C1)^-1 J2
    H, XA1 = XR[..., :n], XR[..., n:]
    return A2 @ XA1, sym(A2 @ H @ A2.mT + C2), sym(A1.mT @ G @ A1 + J1)


def _flip(tree):
    return jax.tree.map(lambda e: jnp.flip(e, 0), tree)


def riccati_scan(lq, data, combine=vf_combine):
    """P_0..P_N as the suffix products a_t (x) ... (x) a_N: one associative scan, Theta(log N) depth."""
    n, dt = data['A'].shape[-1], data['A'].dtype
    zero = jnp.zeros((1, n, n), dt)
    elems = [jnp.concatenate([e, last]) for e, last in
             zip(stage_elements(lq, data), (zero, zero, lq['QtN'][None]))]       # a_N = (0, 0, QtN)
    # suffix products: reverse time and swap the operator's arguments
    return _flip(lax.associative_scan(lambda a1, a2: combine(a2, a1), _flip(tuple(elems))))[2]


def riccati_sweep(lq, data):
    """P_0..P_N by the classical Riccati recursion, Theta(N) depth."""
    def backward(P, stage):
        A, B, Qt, St, Rt = stage
        Sig = Rt + B.T @ P @ B
        K = jnp.linalg.solve(Sig, B.T @ P @ A + St)
        P = sym(Qt + A.T @ P @ A - K.T @ Sig @ K)
        return P, P
    stages = (data['A'], data['B'], lq['Qt'], lq['St'], lq['Rt'])
    _, P = lax.scan(backward, lq['QtN'], stages, reverse=True)
    return jnp.concatenate([P, lq['QtN'][None]])


def factor_lq(lq, data, method='scan', combine='auto'):
    """Matrix part of the LQ solve: the cost-to-go P_t, the Cholesky factor L_t of Sigma_t = Rt_t + B_t'P_{t+1}B_t,
    the feedback gains K_t and the closed loop A_t - B_t K_t. None of it depends on the right-hand side, so the
    IPM's corrector steps reuse it.

    method: 'scan' (associative scan) or 'sweep' (Riccati recursion).
    combine (scan only): 'auto' redoes the scan with the LU combine if the Cholesky scan breaks down (semidefinite
    cost); 'chol' skips that check (valid when `cholesky_ok`); 'lu' uses the LU combine."""
    A, B = data['A'], data['B']
    if method == 'scan':
        if combine == 'lu':
            P = riccati_scan(lq, data, vf_combine_lu)
        else:
            P = riccati_scan(lq, data)
            if combine == 'auto':
                P = lax.cond(jnp.all(jnp.isfinite(P)), lambda: P, lambda: riccati_scan(lq, data, vf_combine_lu))
    elif method == 'sweep':
        P = riccati_sweep(lq, data)
    else:
        raise ValueError(f"method must be 'scan' or 'sweep', not {method!r}")
    BtP = B.mT @ P[1:]
    L = jnp.linalg.cholesky(lq['Rt'] + BtP @ B)
    K = spd_solve(L, BtP @ A + lq['St'])
    return dict(P=P, L=L, K=K, Acl=A - B @ K)


def cholesky_ok(data):
    """True if the Cholesky combine is valid at every IPM iterate in exact arithmetic: Qbar_t = Q_t - S_t'R_t^-1 S_t
    > 0 and QN > 0. The barrier adds [C D]'W[C D] >= 0 to the stage Hessian and the Schur complement is monotone, so
    Qbar_t can only grow. `combine='chol'` then drops the per-iteration check, which however also catches rare
    numerical breakdowns at extreme barrier weights; the default stays 'auto'."""
    Lr = jnp.linalg.cholesky(data['R'])
    V = tri_solve(Lr, data['S'])
    ok = jnp.all(jnp.isfinite(jnp.linalg.cholesky(sym(data['Q'] - V.mT @ V))))
    return ok & jnp.all(jnp.isfinite(jnp.linalg.cholesky(data['QN']))) & jnp.all(jnp.isfinite(Lr))


# ----------------------------------------------------------------- vector part ---
def affine_combine(a1, a2):
    """Compose the affine maps x -> Phi x + phi; a1 is applied first."""
    P1, p1 = a1
    P2, p2 = a2
    return P2 @ P1, (P2 @ p1[..., None])[..., 0] + p2


def affine_recursion(F, f, x0, method='scan'):
    """x_0..x_N of x_{t+1} = F_t x_t + f_t: an associative scan over the affine maps ('scan', Theta(log N) depth)
    or a sequential loop ('sweep')."""
    if method == 'scan':
        Phi, phi = lax.associative_scan(affine_combine, (F, f))
        xs = jnp.einsum('tij,j->ti', Phi, x0) + phi
    else:
        _, xs = lax.scan(lambda x, Ff: ((Ff[0] @ x + Ff[1]),) * 2, x0, (F, f))
    return jnp.concatenate([x0[None], xs])


def solve_lq(lq, data, fac, method='scan'):
    """Vector part of the LQ solve, given `factor_lq`. The costate offsets obey
    p_t = Acl_t'(p_{t+1} + P_{t+1} chat_t) + rhox_t - K_t' rhou_t backward from p_N, the states
    x_{t+1} = Acl_t x_t + chat_t - B_t k_t forward from dx0: two affine recursions through the closed loop.
    Returns (dx, du, dnu)."""
    B, P, K, Acl = data['B'], fac['P'], fac['K'], fac['Acl']
    g = mtv(Acl, mv(P[1:], lq['chat'])) + lq['rhox'] - mtv(K, lq['rhou'])
    p = _flip(affine_recursion(_flip(Acl.mT), _flip(g), lq['rhoxN'], method))
    k = spd_solve(fac['L'], (lq['rhou'] + mtv(B, mv(P[1:], lq['chat']) + p[1:]))[..., None])[..., 0]
    dx = affine_recursion(Acl, lq['chat'] - mv(B, k), lq['dx0'], method)
    du = -mv(K, dx[:-1]) - k
    return dx, du, mv(P, dx) + p
