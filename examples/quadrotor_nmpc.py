"""Nonlinear MPC: a planar quadrotor carrying a slung load follows a sinusoidal path. The reference enters as
per-stage params, the thrusts are hard-bounded, the load swing is soft-limited (one slack per pair of rows). The
open loop converges to the optimum; the 20 s closed loop has a wind gust and sensor noise.

    python examples/quadrotor_nmpc.py [--backend auto|cuda|scan|sweep] [--N 30]
"""
import argparse
import time

import jax
import jax.numpy as jnp

import parampc as pm
from parampc.models import quadrotor as quad

ap = argparse.ArgumentParser()
ap.add_argument('--backend', default='auto')
ap.add_argument('--N', type=int, default=30)
args = ap.parse_args()
N, K = args.N, 400

ctrl = pm.NMPC(quad.ocp(N), backend=args.backend, **quad.NMPC_OPTIONS)
params_fn = lambda k: quad.horizon_reference(k * quad.DT, N)      # the reference over the horizon at step k

# open loop: SQP convergence at t = 0
x0 = quad.initial_state()
state = ctrl.init(x0, params_fn(0))
for i in range(1, 14):                                             # 39 SQP iterations, 3 per solve
    state, info = ctrl.solve(state, x0, params_fn(0))
    if i in (1, 4, 13):
        print(f'open loop, {3 * i:2d} SQP iterations: cost {float(info["cost"]):.4f} '
              f'(optimum 277.4224), merit {float(info["merit"]):.4f}')

# closed loop: 20 s, lateral wind gust from 8 to 12 s, measurement noise
plant = quad.plant(K)
run = lambda: jax.block_until_ready(ctrl.closed_loop(plant, x0, K, params_fn=params_fn))
xs, us, infos = run()                                              # compiles
t0 = time.perf_counter()
xs, us, infos = run()
t = (time.perf_counter() - t0) / K
refs = jax.vmap(lambda k: quad.reference(k * quad.DT)[:2])(jnp.arange(K))
rms = float(jnp.sqrt(jnp.mean(jnp.sum((xs[:, :2] - refs) ** 2, axis=1))))
print(f'closed loop, {K} steps: tracking RMS {rms:.4f} m, '
      f'max |phi| {float(jnp.rad2deg(jnp.max(jnp.abs(xs[:, 3])))):.2f} deg (soft limit 10), '
      f'thrust in [{float(us.min()):.3f}, {float(us.max()):.3f}] N (hard limits 0, 9); '
      f'{1e3 * t:.2f} ms per MPC step, {float(jnp.mean(infos["qp_iters"])):.1f} IPM iterations per step')
