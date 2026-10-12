"""Linear MPC: an open-loop unstable LTI system tracks an equilibrium under input and state boxes and a terminal
box; a closed loop with warm starts; a batch of initial states solved in one call.

    python examples/linear_mpc.py [--backend auto|cuda|scan|sweep] [--N 64]
"""
import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

import parampc as pm
from parampc.models import linear

ap = argparse.ArgumentParser()
ap.add_argument('--backend', default='auto')
ap.add_argument('--N', type=int, default=64)
args = ap.parse_args()

rng = np.random.default_rng(0)
n, m, N = 8, 4, args.N
A, B = linear.random_lti(n, m, rho=1.05, rng=rng)                 # open-loop unstable
u_ref = 0.3 * rng.normal(size=m)
x_ref = np.linalg.solve(np.eye(n) - A, B @ u_ref)                # an equilibrium: x_ref = A x_ref + B u_ref
x0 = x_ref + 6.0 * rng.normal(size=n) / np.sqrt(n)

qp = pm.make_qp(A, B, Q=np.eye(n), R=0.1 * np.eye(m), N=N, x0=x0, QN=10 * np.eye(n), x_ref=x_ref, u_ref=u_ref)
qp = pm.add_bounds(qp, u_min=-1.0, u_max=1.0, x_min=-5.0, x_max=5.0, xN_min=x_ref - 0.05, xN_max=x_ref + 0.05)

solve = jax.jit(lambda qp, warm: pm.solve_qp(qp, warm, backend=args.backend))
print(f'backends here: {pm.available_backends()} (using {args.backend})')
sol = solve(qp, None)
print(f'one QP (N = {N}): {pm.STATUS[int(sol.status)]}, {int(sol.iters)} IPM iterations, KKT {float(sol.kkt):.1e}, '
      f'|x_N - x_ref| = {float(jnp.max(jnp.abs(sol.x[-1] - x_ref))):.3f} (box 0.05), '
      f'max |u| = {float(jnp.max(jnp.abs(sol.u))):.3f}')

# closed loop: apply u_0, simulate, warm-start the next QP from the shifted solution
x, iters = jnp.asarray(x0), []
for k in range(40):
    sol = solve(qp.replace(x0=x), sol.shift() if k else None)
    x = A @ x + B @ sol.u[0]
    iters.append(int(sol.iters))
print(f'closed loop, 40 steps: |x - x_ref| = {float(jnp.max(jnp.abs(x - x_ref))):.1e} at the end; '
      f'{iters[0]} IPM iterations cold, {np.mean(iters[1:]):.1f} per warm-started QP')


def timed(f, *a, reps=20):
    """Mean wall time of f(*a) after half a second of untimed calls (a laptop GPU's clock rises under load)."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 0.5:
        jax.block_until_ready(f(*a))
    t0 = time.perf_counter()
    for _ in range(reps):
        out = jax.block_until_ready(f(*a))
    return out, (time.perf_counter() - t0) / reps


# a batch: 64 initial states in one call (one long horizon inside, on scan and cuda)
x0s = jnp.asarray(x_ref + 6.0 * rng.normal(size=(64, n)) / np.sqrt(n))
batch = jax.jit(jax.vmap(lambda x0: pm.solve_qp(qp.replace(x0=x0), backend=args.backend)))
out, t = timed(batch, x0s)
one, t1 = timed(solve, qp.replace(x0=x0s[0]), None)
print(f'batch of 64 QPs: {1e3 * t:.1f} ms ({1e3 * t / 64:.2f} ms per QP; {1e3 * t1:.2f} ms for one QP alone), '
      f'{int(jnp.sum(out.status == 0))}/64 solved, {int(out.iters.max())} IPM iterations, '
      f'max deviation from the single solve {float(jnp.max(jnp.abs(out.u[0] - one.u))):.1e}')
