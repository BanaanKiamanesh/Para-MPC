"""The perturbed KKT system of the MPC QP, its condensation to an LQ problem, and the KKT score.

The QP (stage t = 0..N-1, terminal stage N):

    min  sum_t 1/2 x_t'Q_t x_t + u_t'S_t x_t + 1/2 u_t'R_t u_t + q_t'x_t + r_t'u_t  +  1/2 x_N'QN x_N + qN'x_N
    s.t. x_0 = x0,   x_{t+1} = A_t x_t + B_t u_t + c_t,   C_t x_t + D_t u_t + s_t = d_t,   s_t >= 0

with multipliers nu_t of the dynamics (nu_0 of the initial condition) and lam_t >= 0 of the inequality rows.
The primal-dual interior-point method applies Newton's method to the KKT conditions with the complementarity
products lam * s perturbed to tau.
"""
import jax.numpy as jnp


def mv(M, v):
    """Batched M_t v_t."""
    return jnp.einsum('tij,tj->ti', M, v)


def mtv(M, v):
    """Batched M_t' v_t."""
    return jnp.einsum('tji,tj->ti', M, v)


def sym(M):
    """Symmetric part (M + M') / 2."""
    return 0.5 * (M + M.mT)


def residuals(data, z, tau, corr=0.0):
    """The perturbed KKT residual F_tau(z) (a dict of stage-wise blocks). `corr` is Mehrotra's second-order
    correction of the complementarity rows."""
    A, B, C, D = data['A'], data['B'], data['C'], data['D']
    x, u, nu, lam, s = z['x'], z['u'], z['nu'], z['lam'], z['s']
    xt = x[:-1]
    return dict(
        r0=x[0] - data['x0'],
        rx=nu[:-1] - mv(data['Q'], xt) - mtv(data['S'], u) - data['q'] - mtv(A, nu[1:]) - mtv(C, lam),
        rxN=nu[-1] - data['QN'] @ x[-1] - data['qN'],
        ru=mv(data['R'], u) + mv(data['S'], xt) + data['r'] + mtv(B, nu[1:]) + mtv(D, lam),
        rd=x[1:] - mv(A, xt) - mv(B, u) - data['c'],
        rc=mv(C, xt) + mv(D, u) + s - data['d'],
        rt=lam * s - tau + corr)


def kkt_score(data, z):
    """max(duality measure mean(lam * s), largest KKT residual at tau = 0)."""
    R = residuals(data, z, 0.0)
    feas = jnp.max(jnp.stack([jnp.max(jnp.abs(R[k])) for k in ('r0', 'rx', 'rxN', 'ru', 'rd', 'rc')]))
    return jnp.maximum(jnp.mean(z['lam'] * z['s']), feas)


def barrier_hessians(data, z):
    """The iterate-dependent matrices of the condensed LQ problem: the stage Hessians augmented by the barrier
    weights W = lam / s of the inequality rows. They depend on z only, not on the right-hand side."""
    C, D = data['C'], data['D']
    W = z['lam'] / z['s']
    WC, WD = W[..., None] * C, W[..., None] * D
    return dict(W=W, Qt=data['Q'] + C.mT @ WC, St=data['S'] + D.mT @ WC, Rt=data['R'] + D.mT @ WD, QtN=data['QN'])


def lq_rhs(data, z, res):
    """The right-hand side of the condensed LQ problem for the residual `res` (dlam and ds eliminated stage by
    stage). `w` is kept to recover dlam."""
    w = (z['lam'] * res['rc'] - res['rt']) / z['s']
    return dict(w=w, rhox=-res['rx'] + mtv(data['C'], w), rhou=res['ru'] + mtv(data['D'], w),
                chat=-res['rd'], dx0=-res['r0'], rhoxN=-res['rxN'])
