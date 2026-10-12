"""Nonlinear MPC: cart-pole swing-up. The pole starts hanging and is swung up to upright and balanced, with
|force| <= 15 N and the cart on a 3 m track. The cost 1 - cos(theta) is non-convex; hessian='convexify' (the
default) keeps every SQP subproblem convex.

    python examples/cartpole_swingup.py [--backend auto|cuda|scan|sweep] [--plot]
"""
import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

import parampc as pm
from parampc.models import cartpole

ap = argparse.ArgumentParser()
ap.add_argument('--backend', default='auto')
ap.add_argument('--plot', action='store_true')
args = ap.parse_args()

ctrl = pm.NMPC(cartpole.ocp(N=60), backend=args.backend, **cartpole.NMPC_OPTIONS)
x0 = cartpole.hanging()
state = ctrl.init(x0)                        # u = 0 rolled out: the pole stays down
for _ in range(10):                          # 30 SQP iterations for the first plan
    state, info = ctrl.solve(state, x0)
print(f'first plan: cost {float(info["cost"]):.3f}, merit {float(info["merit"]):.3f}, '
      f'last QP KKT {float(info["qp_kkt"]):.1e}')

K, plant = 160, cartpole.plant()
xs, us, infos = jax.block_until_ready(ctrl.closed_loop(plant, x0, K, state=state))       # compiles
t0 = time.perf_counter()
xs, us, infos = jax.block_until_ready(ctrl.closed_loop(plant, x0, K, state=state))
t = (time.perf_counter() - t0) / K
th = cartpole.wrap(xs[:, 1])
up = K - np.argmax(np.abs(th[::-1]) >= 0.1)                  # first step after which |theta| stays below 0.1
print(f'closed loop, {K} steps of {cartpole.DT} s: |theta| < 0.1 rad from t = {up * cartpole.DT:.2f} s on, final '
      f'|theta| = {abs(th[-1]):.1e} rad, max |cart| = {float(jnp.max(jnp.abs(xs[:, 0]))):.3f} m (track 1.5), '
      f'max |force| = {float(jnp.max(jnp.abs(us))):.2f} N (limit 15)')
print(f'{1e3 * t:.2f} ms per MPC step ({ctrl.sqp_iters} SQP iterations, '
      f'{float(jnp.mean(infos["qp_iters"])):.1f} IPM iterations per step on average)')

if args.plot:
    import matplotlib.pyplot as plt
    ts = cartpole.DT * np.arange(K)
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(7, 6))
    ax[0].plot(ts, xs[:, 0]), ax[0].set_ylabel('cart [m]')
    ax[1].plot(ts, th), ax[1].set_ylabel('angle from upright [rad]')
    ax[2].step(ts, us[:, 0], where='post'), ax[2].set_ylabel('force [N]'), ax[2].set_xlabel('t [s]')
    plt.tight_layout()
    plt.show()
