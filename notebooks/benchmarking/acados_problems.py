"""The two benchmark problems written for acados, so that both solvers solve the same problems.

    linear_solver     the linear MPC QP (LTI dynamics, box constraints, constant reference): one HPIPM solve
    quadrotor_solver  the slung-load quadrotor NMPC of parampc.models.quadrotor: the same midpoint model, cost and
                      constraints, the soft swing limits as acados slacks, the same Levenberg-Marquardt term

acados needs its libraries on the loader path (LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib). The generated C code goes
to acados_build/ next to this file.
"""
import os

import casadi as ca
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

from parampc.models import quadrotor as quad

BUILD = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'acados_build')


def _solver(ocp, name):
    ocp.code_export_directory = os.path.join(BUILD, name)
    return AcadosOcpSolver(ocp, json_file=os.path.join(BUILD, f'{name}.json'), verbose=False)


def _qp_options(ocp, N, tol):
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = float(N)                     # unused: discrete dynamics
    ocp.solver_options.cost_scaling = np.ones(N + 1)     # stage costs exactly as written
    ocp.solver_options.integrator_type = 'DISCRETE'
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.qp_solver_iter_max = 100
    ocp.solver_options.qp_tol = tol


def linear_solver(A, B, Q, R, QN, u_max, x_max, x_ref, N, tol=1e-8):
    """min sum |x - x_ref|_Q^2 + |u|_R^2 + |x_N - x_ref|_QN^2 (halved) s.t. x+ = Ax + Bu, |u| <= u_max on every
    stage, |x| <= x_max on x_1..x_{N-1}. SQP_RTI on a QP is exactly one QP solve."""
    n, m = B.shape
    model = AcadosModel()
    model.name = f'linear_{N}'
    model.x, model.u = ca.SX.sym('x', n), ca.SX.sym('u', m)
    model.disc_dyn_expr = ca.DM(A) @ model.x + ca.DM(B) @ model.u

    ocp = AcadosOcp()
    ocp.model = model
    ocp.cost.cost_type = ocp.cost.cost_type_e = 'LINEAR_LS'
    ocp.cost.W, ocp.cost.W_e = np.block([[Q, np.zeros((n, m))], [np.zeros((m, n)), R]]), QN
    ocp.cost.Vx = np.vstack([np.eye(n), np.zeros((m, n))])
    ocp.cost.Vu = np.vstack([np.zeros((n, m)), np.eye(m)])
    ocp.cost.Vx_e = np.eye(n)
    ocp.cost.yref, ocp.cost.yref_e = np.concatenate([x_ref, np.zeros(m)]), x_ref
    ocp.constraints.x0 = np.zeros(n)
    ocp.constraints.idxbu, ocp.constraints.lbu, ocp.constraints.ubu = (np.arange(m), -u_max * np.ones(m),
                                                                       u_max * np.ones(m))
    ocp.constraints.idxbx, ocp.constraints.lbx, ocp.constraints.ubx = (np.arange(n), -x_max * np.ones(n),
                                                                       x_max * np.ones(n))
    _qp_options(ocp, N, tol)
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    return _solver(ocp, model.name)


def quadrotor_model():
    """parampc.models.quadrotor.dynamics in CasADi, discretized with the same explicit-midpoint step."""
    s, u = ca.SX.sym('s', quad.NX), ca.SX.sym('u', quad.NU)

    def f(s, u):
        th, ph, phid = s[2], s[3], s[7]
        thrust, torque = u[0] + u[1], (u[0] - u[1]) * quad.ARM
        c, sn = ca.cos(ph), ca.sin(ph)
        mL = quad.M_LOAD * quad.L_CABLE
        mass = ca.vertcat(ca.horzcat(quad.M_TOT, 0, 0, mL * c),
                          ca.horzcat(0, quad.M_TOT, 0, mL * sn),
                          ca.horzcat(0, 0, quad.I_YY, 0),
                          ca.horzcat(c, sn, 0, quad.L_CABLE))
        force = ca.vertcat(-thrust * ca.sin(th) + mL * sn * phid**2,
                           thrust * ca.cos(th) - quad.M_TOT * quad.GRAV - mL * c * phid**2,
                           torque,
                           -quad.GRAV * sn)
        return ca.vertcat(s[4:8], ca.solve(mass, force))

    model = AcadosModel()
    model.x, model.u = s, u
    model.disc_dyn_expr = s + quad.DT * f(s + 0.5 * quad.DT * f(s, u), u)
    return model


def quadrotor_solver(N, rti=True, lm=0.5, tol=1e-6, max_iter=100):
    """The quadrotor NMPC. Thrusts hard-bounded; |phi| and |phi'| soft, with the same l1 + l2 slack penalty; the
    Gauss-Newton Hessian gets the same Levenberg-Marquardt term lm I, so each SQP iteration solves the same QP as
    parampc.NMPC. rti=True: one SQP iteration per call (SQP_RTI); rti=False: SQP with merit backtracking, run to
    convergence."""
    nx, nu = quad.NX, quad.NU
    model = quadrotor_model()
    model.name = f'quad_{"rti" if rti else "sqp"}_{N}'

    ocp = AcadosOcp()
    ocp.model = model
    ocp.cost.cost_type = ocp.cost.cost_type_e = 'LINEAR_LS'
    ocp.cost.W = np.diag(np.concatenate([np.asarray(quad.Q_DIAG), np.asarray(quad.R_DIAG)]))
    ocp.cost.W_e = np.diag(np.asarray(quad.Q_TERM))
    ocp.cost.Vx = np.vstack([np.eye(nx), np.zeros((nu, nx))])
    ocp.cost.Vu = np.vstack([np.zeros((nx, nu)), np.eye(nu)])
    ocp.cost.Vx_e = np.eye(nx)
    ocp.cost.yref, ocp.cost.yref_e = np.zeros(nx + nu), np.zeros(nx)   # set per stage later

    ocp.constraints.x0 = np.asarray(quad.initial_state())
    ocp.constraints.idxbu, ocp.constraints.lbu, ocp.constraints.ubu = (np.arange(nu), np.zeros(nu),
                                                                       quad.T_MAX * np.ones(nu))
    ocp.constraints.idxbx = np.array([3, 7])                                   # phi, phi'
    ocp.constraints.ubx = np.array([quad.PHI_MAX, quad.PHID_MAX])
    ocp.constraints.lbx = -ocp.constraints.ubx
    ocp.constraints.idxsbx = np.array([0, 1])                                  # both soft
    rho, eta = quad.SOFT_L1 * np.ones(2), quad.SOFT_L2 * np.ones(2)
    ocp.cost.zl, ocp.cost.zu, ocp.cost.Zl, ocp.cost.Zu = rho, rho, eta, eta

    _qp_options(ocp, N, tol)
    ocp.solver_options.nlp_solver_type = 'SQP_RTI' if rti else 'SQP'
    ocp.solver_options.levenberg_marquardt = lm
    if not rti:
        ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
    ocp.solver_options.nlp_solver_max_iter = max_iter
    ocp.solver_options.tol = tol
    return _solver(ocp, model.name)


def set_quadrotor_problem(solver, N, SR, X, U):
    """Load the reference SR (N+1, nx) and the initial guess (X, U) into the solver."""
    y_u = np.asarray(quad.U_HOVER)
    for t in range(N):
        solver.set(t, 'yref', np.concatenate([SR[t], y_u]))
        solver.set(t, 'x', X[t])
        solver.set(t, 'u', U[t, :quad.NU])
    solver.set(N, 'yref', SR[N])
    solver.set(N, 'x', X[N])
