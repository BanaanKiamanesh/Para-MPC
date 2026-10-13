"""One benchmark measurement in a fresh process, so that its peak memory is its own and the compilation of one
configuration cannot slow another:

    python worker.py <linear|quadrotor> <acados|cuda|scan|sweep> <N> <repeats>

The device comes from JAX_PLATFORMS. It prints one JSON line:

    time_ms     median time of one solve: wall clock for parampc (compiled call, block_until_ready), acados' own
                timer time_tot for acados; both after one second of untimed solves (clocks at their steady state)
    setup_s     one-off cost: JIT compilation (parampc) or code generation + C compilation (acados), including the
                first solve
    iters       IPM iterations (parampc) or QP iterations (acados)
    kkt         KKT score of the returned QP solution (parampc)
    objective   cost of the returned solution, to check that both solvers agree
    mem_mb      peak resident memory of the process minus the memory before the problem was built
    xla_mb      memory of the compiled XLA program (parampc): arguments, outputs and temporaries

Problems:
    linear      one linear MPC QP: an open-loop unstable LTI system (n = 8, m = 4), input and state boxes, a constant
                nonzero reference (with x_ref = 0 the optimal states decay below 1e-308 over long horizons, and
                subnormal arithmetic slows C solvers down by an order of magnitude)
    quadrotor   one SQP iteration of the slung-load quadrotor NMPC from the hover initial guess (cold multipliers)
"""
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import parampc as pm
from parampc.models import linear
from parampc.models import quadrotor as quad

WARMUP_S = 1.0


def peak_rss_mb():
    """Peak resident set size of this process (Linux: VmHWM), in MB."""
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmHWM'):
                return int(line.split()[1]) / 1024


def current_rss_mb():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS'):
                return int(line.split()[1]) / 1024


def warm_up(fn, seconds=WARMUP_S):
    """Run fn() for `seconds` before timing, so that CPU and GPU clocks are at their steady state, as in a running
    controller (a laptop GPU needs about half a second of load to reach its boost clock)."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        fn()


def median_time(fn, repeats):
    """Median wall-clock time of fn() in ms, after the warm-up."""
    warm_up(fn)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(times))


def median_acados_time(solver, solve, repeats):
    """Median of acados' internal solve time (time_tot) in ms, after the warm-up."""
    warm_up(solve)
    times = []
    for _ in range(repeats):
        solve()
        times.append(solver.get_stats('time_tot'))
    return 1e3 * float(np.median(times))


def xla_memory_mb(compiled):
    """Arguments + outputs + temporaries of a compiled XLA program, in MB."""
    mem = compiled.memory_analysis()
    return (mem.argument_size_in_bytes + mem.output_size_in_bytes + mem.temp_size_in_bytes) / 2**20


# ------------------------------------------------------------------------------------------------ linear ---
def lti_system(n=8, m=4, seed=0):
    """The random open-loop unstable LTI system (spectral radius 1.05), its weights and bounds, an initial state at
    which the input bounds are active, and the reference."""
    rng = np.random.default_rng(seed)
    A, B = linear.random_lti(n, m, rho=1.05, rng=rng)
    return dict(A=A, B=B, Q=np.eye(n), R=0.1 * np.eye(m), QN=10.0 * np.eye(n), u_max=1.0, x_max=5.0,
                x0=3.0 * rng.normal(size=n) / np.sqrt(n), x_ref=0.5 * rng.normal(size=n) / np.sqrt(n))


def linear_qp(sys_, N):
    qp = pm.make_qp(sys_['A'], sys_['B'], sys_['Q'], sys_['R'], N=N, x0=sys_['x0'], QN=sys_['QN'], x_ref=sys_['x_ref'])
    return pm.add_bounds(qp, u_min=-sys_['u_max'], u_max=sys_['u_max'], x_min=-sys_['x_max'], x_max=sys_['x_max'])


def linear_objective(sys_, x, u):
    Q, R, QN = sys_['Q'], sys_['R'], sys_['QN']
    e = x - sys_['x_ref']
    return float(0.5 * np.einsum('ti,ij,tj->', e[:-1], Q, e[:-1]) + 0.5 * np.einsum('ti,ij,tj->', u, R, u)
                 + 0.5 * e[-1] @ QN @ e[-1])


def run_parampc_linear(N, backend, repeats, tol=1e-8):
    sys_ = lti_system()
    qp = linear_qp(sys_, N)
    t0 = time.perf_counter()
    compiled = jax.jit(lambda qp: pm.solve_qp(qp, tol=tol, backend=backend)).lower(qp).compile()
    sol = jax.block_until_ready(compiled(qp))
    setup = time.perf_counter() - t0
    t = median_time(lambda: jax.block_until_ready(compiled(qp)), repeats)
    return dict(time_ms=t, setup_s=setup, iters=int(sol.iters), kkt=float(sol.kkt), status=int(sol.status),
                xla_mb=xla_memory_mb(compiled), objective=linear_objective(sys_, np.asarray(sol.x), np.asarray(sol.u)))


def run_acados_linear(N, repeats, tol=1e-8):
    from acados_problems import linear_solver
    sys_ = lti_system()
    t0 = time.perf_counter()
    solver = linear_solver(sys_['A'], sys_['B'], sys_['Q'], sys_['R'], sys_['QN'], sys_['u_max'], sys_['x_max'],
                           sys_['x_ref'], N, tol=tol)

    def solve():
        solver.reset()
        solver.solve_for_x0(sys_['x0'], fail_on_nonzero_status=False)

    solve()
    setup = time.perf_counter() - t0
    t = median_acados_time(solver, solve, repeats)
    x = np.array([solver.get(i, 'x') for i in range(N + 1)])
    u = np.array([solver.get(i, 'u') for i in range(N)])
    return dict(time_ms=t, setup_s=setup, iters=int(solver.get_stats('qp_iter').sum()),
                status=int(solver.get_status()), objective=linear_objective(sys_, x, u))


# --------------------------------------------------------------------------------------------- quadrotor ---
def quadrotor_controller(N, backend='auto'):
    """The NMPC, cold-started: its QPs start from multipliers 1, floored at 1."""
    return pm.NMPC(quad.ocp(N), backend=backend, warm_floor=1.0, **quad.NMPC_OPTIONS)


def run_parampc_quadrotor(N, backend, repeats):
    """One SQP iteration from the hover initial guess."""
    ctrl = quadrotor_controller(N, backend)
    x0, SR = quad.initial_state(), quad.horizon_reference(0.0, N)
    state = ctrl.init(x0, SR)
    t0 = time.perf_counter()
    compiled = jax.jit(lambda st: ctrl.sqp_iteration(st, x0, SR)).lower(state).compile()
    new, sol = jax.block_until_ready(compiled(state))
    setup = time.perf_counter() - t0
    t = median_time(lambda: jax.block_until_ready(compiled(state)), repeats)
    return dict(time_ms=t, setup_s=setup, iters=int(sol.iters), kkt=float(sol.kkt), status=int(sol.status),
                xla_mb=xla_memory_mb(compiled), objective=float(ctrl.cost(new.X, new.U, SR)))


def run_acados_quadrotor(N, repeats):
    """One SQP_RTI iteration from the same hover initial guess."""
    from acados_problems import quadrotor_solver, set_quadrotor_problem
    ctrl = quadrotor_controller(N)
    x0, SR = quad.initial_state(), quad.horizon_reference(0.0, N)
    guess = ctrl.init(x0, SR)
    X0, U0, SRn, x0n = map(np.asarray, (guess.X, guess.U, SR, x0))
    t0 = time.perf_counter()
    solver = quadrotor_solver(N, rti=True)

    def solve():
        solver.reset()
        set_quadrotor_problem(solver, N, SRn, X0, U0)
        solver.solve_for_x0(x0n, fail_on_nonzero_status=False)

    solve()
    setup = time.perf_counter() - t0
    t = median_acados_time(solver, solve, repeats)
    X = np.array([solver.get(i, 'x') for i in range(N + 1)])
    U = np.array([solver.get(i, 'u') for i in range(N)])
    U = np.concatenate([U, np.zeros((N, quad.ocp(N).ns))], axis=1)          # the cost of the unslacked point
    return dict(time_ms=t, setup_s=setup, iters=int(solver.get_stats('qp_iter').sum()),
                status=int(solver.get_status()), objective=float(ctrl.cost(jnp.asarray(X), jnp.asarray(U), SR)))


RUNNERS = {('linear', 'acados'): lambda N, r: run_acados_linear(N, r),
           ('quadrotor', 'acados'): lambda N, r: run_acados_quadrotor(N, r)}
for _backend in pm.backends.BACKENDS:
    RUNNERS[('linear', _backend)] = lambda N, r, b=_backend: run_parampc_linear(N, b, r)
    RUNNERS[('quadrotor', _backend)] = lambda N, r, b=_backend: run_parampc_quadrotor(N, b, r)

if __name__ == '__main__':
    problem, solver, N, repeats = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    if solver == 'acados':
        import acados_problems  # noqa: F401  (import before the memory baseline)
    base = current_rss_mb()
    out = RUNNERS[(problem, solver)](N, repeats)
    out.update(problem=problem, solver=solver, N=N, backend=jax.default_backend(), mem_mb=peak_rss_mb() - base)
    print(json.dumps(out))
