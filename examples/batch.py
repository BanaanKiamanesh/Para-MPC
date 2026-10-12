"""Batches: jax.vmap over QPs or over whole NMPC closed loops. On 'cuda' and 'scan' a batch of B QPs is solved as one
QP with horizon B (N+1) - 1, which is what the parallel-in-horizon method is good at; the batch shares the IPM's
iterations.

  1. QP throughput: a linear MPC QP from B random initial states, time per QP against B.
  2. Monte Carlo: 32 cart-pole swing-ups from different initial angles as one vmapped closed loop (each MPC step
     solves the 32 SQP subproblems together).

    python examples/batch.py [--backend auto|cuda|scan|sweep]
"""
import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

import parampc as pm
from parampc.models import cartpole, linear

ap = argparse.ArgumentParser()
ap.add_argument('--backend', default='auto')
args = ap.parse_args()
print(f'backend {args.backend} (available: {pm.available_backends()})')


def timed(f, *a, reps=10):
    """Mean wall time after half a second of untimed calls (a laptop GPU's clock rises under load)."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 0.5:
        jax.block_until_ready(f(*a))
    t0 = time.perf_counter()
    for _ in range(reps):
        out = jax.block_until_ready(f(*a))
    return out, (time.perf_counter() - t0) / reps


# 1. QP throughput -------------------------------------------------------------------------------------------------
rng = np.random.default_rng(0)
n, m, N = 8, 4, 32
A, B = linear.random_lti(n, m, rho=1.05, rng=rng)
qp = pm.make_qp(A, B, Q=np.eye(n), R=0.1 * np.eye(m), N=N, x0=np.zeros(n), QN=10 * np.eye(n))
qp = pm.add_bounds(qp, u_min=-1.0, u_max=1.0, x_min=-5.0, x_max=5.0)
x0s = jnp.asarray(2.0 * rng.normal(size=(256, n)) / np.sqrt(n))

one = jax.jit(lambda x0: pm.solve_qp(qp.replace(x0=x0), backend=args.backend))
many = jax.jit(jax.vmap(lambda x0: pm.solve_qp(qp.replace(x0=x0), backend=args.backend)))
_, t1 = timed(one, x0s[0])
print(f'linear MPC QP (n = {n}, m = {m}, N = {N}): one QP {1e3 * t1:.2f} ms')
for Bn in (8, 32, 128, 256):
    out, t = timed(many, x0s[:Bn])
    print(f'  batch of {Bn:3d}: {1e3 * t:7.2f} ms, {1e3 * t / Bn:.3f} ms per QP ({t1 * Bn / t:.1f}x the throughput of '
          f'one-by-one), {int(jnp.sum(out.status == 0))}/{Bn} solved, {int(out.iters.max())} IPM iterations')

# 2. Monte Carlo over NMPC closed loops ----------------------------------------------------------------------------
ctrl = pm.NMPC(cartpole.ocp(N=60), backend=args.backend, **cartpole.NMPC_OPTIONS)
plant, K = cartpole.plant(), 160


def swing_up(x0):
    state = ctrl.init(x0)
    state = jax.lax.fori_loop(0, 10, lambda i, st: ctrl.solve(st, x0)[0], state)        # the first plan
    return ctrl.closed_loop(plant, x0, K, state=state)


angles = jnp.pi + jnp.linspace(-0.6, 0.6, 32)                  # around hanging, both sides
x0s = jnp.stack([jnp.zeros(32), angles, jnp.zeros(32), jnp.zeros(32)], axis=1)
(xs, us, infos), t = timed(jax.jit(jax.vmap(swing_up)), x0s, reps=2)
th = cartpole.wrap(xs[:, :, 1])
up = np.abs(th[:, -20:]).max(axis=1) < 0.05                     # upright over the last second
print(f'cart-pole Monte Carlo: 32 closed loops of {K} steps in {t:.2f} s ({1e3 * t / K:.2f} ms per step for all 32); '
      f'{int(up.sum())}/32 balanced at the end, max |force| {float(jnp.max(jnp.abs(us))):.2f} N, '
      f'max |cart| {float(jnp.max(jnp.abs(xs[:, :, 0]))):.3f} m')
