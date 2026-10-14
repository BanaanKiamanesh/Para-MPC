"""Nonlinear MPC: a planar quadrotor carrying a slung load follows a sinusoidal path. The reference enters as
per-stage params, the thrusts are hard-bounded, the load swing is soft-limited (one slack per pair of rows). The
open loop converges to the optimum; the 20 s closed loop has a wind gust and sensor noise.

    python examples/quadrotor_nmpc.py [--backend auto|cuda|scan|sweep] [--precision float64|float32] [--N 30]
                                      [--sqp-iters 3]

The last part runs the closed loop as a real-time controller would: one ctrl.step per sample, called from Python
with the new measurement, and reports the latency of each step. --sqp-iters 1 is the real-time iteration (RTI).
The open loop always takes 3 SQP iterations per solve.
"""
import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

import parampc as pm
from parampc.models import quadrotor as quad

ap = argparse.ArgumentParser()
ap.add_argument('--backend', default='auto')
ap.add_argument('--precision', default='float64', choices=('float64', 'float32'))
ap.add_argument('--N', type=int, default=30)
ap.add_argument('--sqp-iters', type=int, default=quad.NMPC_OPTIONS['sqp_iters'])
args = ap.parse_args()
N, K = args.N, 400

opts = dict(quad.NMPC_OPTIONS, backend=args.backend, precision=args.precision)
ctrl = pm.NMPC(quad.ocp(N), **opts)                                # the open loop: 3 SQP iterations per solve
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
ctrl = pm.NMPC(quad.ocp(N), **dict(opts, sqp_iters=args.sqp_iters))
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

# real time: the same closed loop, one compiled step per sample called from Python (the measurement in, u0 out;
# the reference over the horizon is built inside it), the plant simulated outside the timing; the latency of a step
# is what a sampling period has to cover
step = jax.jit(lambda state, x, k: ctrl.step(state, x, params_fn(k)))
plant_step = jax.jit(plant)


def realtime():
    x, state, lat, xs_rt = x0, ctrl.init(x0, params_fn(0)), [], []
    for k in range(K):
        t0 = time.perf_counter()
        u, state, _ = step(state, x, k)
        u.block_until_ready()
        lat.append(1e3 * (time.perf_counter() - t0))
        xs_rt.append(x)
        x = plant_step(x, u, k)
    return np.array(lat), jnp.stack(xs_rt)


realtime()                                                         # compiles
lat, xs_rt = realtime()
rms_rt = float(jnp.sqrt(jnp.mean(jnp.sum((xs_rt[:, :2] - refs) ** 2, axis=1))))
print(f'real time, {K} steps: latency per step median {np.median(lat):.2f} ms, 95th percentile '
      f'{np.percentile(lat, 95):.2f} ms, max {lat.max():.2f} ms; tracking RMS {rms_rt:.4f} m')
