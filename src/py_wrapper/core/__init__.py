"""The solver core: a primal-dual interior-point method for the MPC QP whose Newton steps are associative scans
over the horizon. `parampc.solve_qp` reduces every supported problem to this form; use the core directly only to
study the method itself.

    kkt             the perturbed KKT system, its condensation to an LQ problem, the KKT score
    lq              the LQ solve: value-function scan (matrix part) and affine scans (vector part)
    newton          the Newton step with iterative refinement
    interior_point  Mehrotra's predictor-corrector method with Gondzio's centrality correctors

Problem data (a dict of arrays; N stages, n states, m inputs, M >= 1 inequality rows per stage):

    A (N,n,n)  B (N,n,m)  c (N,n)                         x_{t+1} = A_t x_t + B_t u_t + c_t
    Q (N,n,n)  S (N,m,n)  R (N,m,m)  q (N,n)  r (N,m)       stage cost
    QN (n,n)   qN (n,)                                    terminal cost
    C (N,M,n)  D (N,M,m)  d (N,M)                         C_t x_t + D_t u_t <= d_t
    x0 (n,)                                               initial state

Iterate (a dict): x (N+1,n), u (N,m), nu (N+1,n), lam (N,M), s (N,M).
"""
from .interior_point import centrality_correctors, initial_iterate, ipm, ipm_iteration, mehrotra_start, step_to_boundary
from .kkt import barrier_hessians, kkt_score, lq_rhs, mtv, mv, residuals, sym
from .lq import (
    affine_combine,
    affine_recursion,
    cholesky_ok,
    factor_lq,
    riccati_scan,
    riccati_sweep,
    solve_lq,
    stage_elements,
    vf_combine,
    vf_combine_lu,
)
from .newton import factorize, newton_direction, newton_step, refine_step, solve_newton

__all__ = ['affine_combine', 'affine_recursion', 'barrier_hessians', 'centrality_correctors', 'cholesky_ok',
           'factor_lq', 'factorize', 'initial_iterate', 'ipm', 'ipm_iteration', 'kkt_score', 'lq_rhs',
           'mehrotra_start', 'mtv', 'mv', 'newton_direction', 'newton_step', 'refine_step', 'residuals',
           'riccati_scan', 'riccati_sweep', 'solve_lq', 'solve_newton', 'stage_elements', 'step_to_boundary', 'sym',
           'vf_combine', 'vf_combine_lu']
