# ParaMPC

Parallel-in-horizon model predictive control. ParaMPC solves MPC problems with a primal-dual interior-point method
whose Newton steps are **associative scans over the horizon**: each step has $\Theta(\log N)$ depth instead of the
$\Theta(N)$ of a Riccati recursion. It runs in JAX (CPU or GPU) or as a **single CUDA kernel** per solve, and
handles linear MPC (QPs) and nonlinear MPC (SQP and the real-time iteration).

```python
import parampc as pm

qp = pm.make_qp(A, B, Q, R, N=100, x0=x0)                    # linear MPC
qp = pm.add_bounds(qp, u_min=-1.0, u_max=1.0)
sol = pm.solve_qp(qp)                                       # sol.x, sol.u, sol.status

ctrl = pm.NMPC(pm.OCP(nx=4, nu=1, N=60, dynamics=..., stage_cost=...))      # nonlinear MPC
u0, state, info = ctrl.step(ctrl.init(x0), x0)
```

- [Installation](#installation)
- [Linear MPC](#linear-mpc)
- [Nonlinear MPC](#nonlinear-mpc)
- [Backends](#backends)
- [How it works](#how-it-works)
- [Performance](#performance)
- [Repository layout](#repository-layout)
- [Tests](#tests)
- [Limitations](#limitations)

## Installation

ParaMPC needs Linux, Python ≥ 3.11 and JAX ≥ 0.11 (float64 is switched on at import).

```bash
pip install -e .                 # CPU: the 'scan' and 'sweep' backends
```

```bash
pip install -e ".[cuda]"         # + JAX's CUDA build and nvcc: the 'cuda' backend
```

The CUDA kernel is compiled with the pip-installed `nvcc` the first time each problem size $(n, m)$ is solved
(a minute or so), and cached in `~/.cache/parampc` (or `$PARAMPC_CACHE`). On a GPU, run JAX with
`JAX_PLATFORMS=cuda,cpu`: the solver's diagnostic messages need a CPU device.

To reproduce the full environment (notebooks, tests, the acados benchmark):

```bash
conda env create -f environment.yml
```

## Linear MPC

`OCPQP` is the general time-varying MPC QP over $N$ stages, with $n$ states, $m$ inputs, $M$ inequality rows per
stage and $M_N$ terminal rows:

$$
\begin{aligned}
\min_{x,u}\ & \sum_{t=0}^{N-1} \tfrac12 x_t^\top Q_t x_t + u_t^\top S_t x_t + \tfrac12 u_t^\top R_t u_t + q_t^\top x_t + r_t^\top u_t
\ +\ \tfrac12 x_N^\top Q_N x_N + q_N^\top x_N \\
\text{s.t.}\ & x_0 = \bar x, \qquad x_{t+1} = A_t x_t + B_t u_t + c_t, \qquad C_t x_t + D_t u_t \le d_t, \qquad C_N x_N \le d_N .
\end{aligned}
$$

$R_t$ must be positive definite; $Q_t$ and $Q_N$ may be semidefinite (output tracking, terminal-only costs).

### Building a QP

```python
import numpy as np
import parampc as pm

qp = pm.make_qp(A, B, Q=np.eye(n), R=0.1 * np.eye(m), N=50, x0=x0,
                QN=10 * np.eye(n), x_ref=x_ref, u_ref=u_ref)        # tracking terms q, r, qN from the references
qp = pm.add_bounds(qp, u_min=-1.0, u_max=1.0, x_min=-5.0, x_max=5.0,
                   xN_min=x_ref - 0.05, xN_max=x_ref + 0.05)        # a terminal box
qp = pm.soften(qp, rows=[0, 1], l1=1e3, l2=1e-2)                    # rows 0, 1 soft: slack inputs, exact l1 penalty
```

- `make_qp(A, B, Q, R, N, x0, QN=None, S=None, q=None, r=None, qN=None, c=None, C=None, D=None, d=None, CN=None,
  dN=None, x_ref=None, u_ref=None)`: data without the leading stage axis is broadcast to all $N$ stages; give
  `(N, ...)` arrays for time-varying data. Defaults: $Q_N = Q$, $S = 0$, $c = 0$, no inequality rows.
- `add_bounds(qp, x_min, x_max, u_min, u_max, xN_min, xN_max)`: box constraints as rows; scalars, `(n,)`/`(m,)` or
  stage-varying `(N, n)`/`(N, m)` arrays; infinite entries are skipped. State bounds apply to $x_1..x_{N-1}$
  ($x_0$ is fixed), `xN_*` adds terminal rows.
- `soften(qp, rows, l1, l2)`: row $i$ becomes $C x + D u - \sigma_i \le d$ with a slack input $\sigma_i \ge 0$
  penalized by $l_1\sigma_i + \tfrac{l_2}{2}\sigma_i^2$; the slacks are appended to the inputs (`sol.u[:, m:]`).
- An `OCPQP` is a frozen dataclass and a JAX pytree: `qp.replace(x0=x)` gives the next step's QP, and QPs can be
  traced, jitted and vmapped. Properties: `N`, `nx`, `nu`, `M`, `MN`.

### Solving

```python
sol = pm.solve_qp(qp)                                         # backend='auto'
sol = pm.solve_qp(qp.replace(x0=x1), sol.shift())             # the next MPC step, warm-started
step = jax.jit(lambda qp, warm: pm.solve_qp(qp, warm))        # jit the whole solve
sols = jax.vmap(lambda x0: pm.solve_qp(qp.replace(x0=x0)))(x0s)   # a batch of QPs in one call
```

`solve_qp(qp, warm=None, *, tol=1e-8, mu_min=1e-14, max_iter=50, backend='auto', start='given', floor=None,
ipm_options=None)` returns a `Solution`:

| field | shape | meaning |
|---|---|---|
| `x`, `u` | `(N+1, n)`, `(N, m)` | the optimal states and inputs |
| `nu` | `(N+1, n)` | multipliers of the dynamics ($\nu_0$: of the initial condition) |
| `lam`, `s` | `(N, M)` | multipliers and slacks of the stage rows |
| `lamN`, `sN` | `(MN,)` | multipliers and slacks of the terminal rows |
| `iters` | `()` | interior-point iterations |
| `kkt` | `()` | max(duality measure, largest KKT residual) of the returned point on this QP |
| `status` | `()` | 0 solved, 1 iteration limit, 2 stopped at `mu_min` or numerical breakdown (`pm.STATUS`) |

- **Warm starts.** `warm` is a `Solution` of a similar QP; `sol.shift()` drops the first stage and repeats the last,
  the receding-horizon warm start. Its multipliers and slacks are floored at `floor` (default `1e-3`; `1` for a cold
  start). In a closed loop this saves a third to a half of the iterations.
- **Stopping.** The IPM stops at KKT score ≤ `tol`, at duality measure ≤ `mu_min`, at `max_iter`, or on a non-finite
  step, and returns the most accurate iterate it has seen (past the attainable accuracy the iterates degrade).
- **Options.** `start='mehrotra'` uses Mehrotra's starting-point heuristic (fewer iterations on cold starts of hard
  QPs). `ipm_options` reaches the core's settings (`n_corr` Gondzio correctors, `refine` refinement steps,
  `refine_pred`, `combine`); the `'cuda'` backend implements the defaults only.
- **Checking a solution.** `pm.kkt_score(qp, sol)` measures any point on the QP, and
  `pm.testing.kkt_certificate(qp, sol)` evaluates the KKT conditions on an independent dense build of the QP.

### Batches

Under `jax.vmap`, the `'scan'` and `'cuda'` backends solve $B$ QPs as **one QP with horizon $B(N+1)-1$**: each QP's
terminal stage becomes a reset stage leading into the next QP's initial state. The problem stays separable, so the
solution is the $B$ individual solutions, and a long horizon is what the parallel method is good at. The batch
shares the IPM's iterations and its stopping test (whose duality measure is averaged over the QPs), so it stops when
the hardest QP is solved. On `'sweep'` a batch is an ordinary vmap with per-QP iterations.

See [`notebooks/01_linear_mpc.ipynb`](notebooks/01_linear_mpc.ipynb) and [`examples/linear_mpc.py`](examples/linear_mpc.py).

## Nonlinear MPC

An `OCP` describes the problem; `NMPC` solves it by sequential quadratic programming.

```python
import jax.numpy as jnp
import parampc as pm

def f(x, u, p):                                   # continuous-time dynamics (or write F(x, u, p) directly)
    ...

ocp = pm.OCP(
    nx=4, nu=1, N=60,
    dynamics=pm.rk4(f, dt=0.05),                  # discrete-time x_{t+1} = F(x, u, p): pm.euler, pm.midpoint, pm.rk4
    stage_cost=lambda x, u, p: jnp.sum((x - p) ** 2) + 0.01 * u[0] ** 2,
    terminal_cost=lambda x, p: 10 * jnp.sum((x - p) ** 2),
    constraints=lambda x, u, p: jnp.array([x[1] - 0.5, -x[1] - 0.5]),     # g(x, u, p) <= 0
    u_min=-15.0, u_max=15.0, x_min=[-1.5, -jnp.inf, -jnp.inf, -jnp.inf], x_max=[1.5, jnp.inf, jnp.inf, jnp.inf],
    soft=[[0, 1]], soft_l1=1e3, soft_l2=1e-2)     # constraints 0 and 1 share one slack

ctrl = pm.NMPC(ocp, sqp_iters=3)                  # sqp_iters=1: the real-time iteration
state = ctrl.init(x0, params)                     # the initial guess: u_guess rolled out
u0, state, info = ctrl.step(state, x, params)     # one MPC step: SQP from the warm start, then shift
xs, us, infos = ctrl.closed_loop(plant, x0, K=400, params_fn=params_fn)    # a closed loop as one lax.scan
```

**The problem.** `OCP(nx, nu, N, dynamics, stage_cost, terminal_cost=None, constraints=None,
terminal_constraints=None, u_min, u_max, x_min, x_max, xN_min, xN_max, soft=(), soft_l1=1e3, soft_l2=1e-2,
u_guess=None)`. Functions take the stage parameters `p`: a pytree whose leaves have a leading axis $N+1$ (stage $t$
gets $p_t$, the terminal cost $p_N$), for example a reference trajectory; pass `None` when there are none. Bounds
are linear rows and stay exact; `constraints` are linearized; `soft` lists groups of constraint indices that share a
slack (penalized by `soft_l1 s + soft_l2/2 s^2`, an exact penalty when `soft_l1` exceeds the multipliers; the slacks
are `state.U[:, nu:]`). `terminal_constraints` are hard.

**The controller.** `NMPC(ocp, sqp_iters=3, lm=0.5, merit_weight=500.0, ladder=(1, 0.5, 0.25, 0.125),
qp_tol=1e-6, qp_max_iter=30, warm_floor=1e-3, backend='auto', hessian='convexify', ipm_options=None)`. Each SQP
iteration:

1. linearizes the dynamics and constraints at the current point (JAX derivatives) and expands the costs to second
   order, plus a Levenberg–Marquardt term `lm` (the proximal term $\tfrac{lm}{2}\|z - z_k\|^2$);
2. with `hessian='convexify'` (the default) lifts negative curvature of the cost Hessians to zero, so non-convex
   costs give convex QPs; a convex cost keeps its Hessians bit for bit (`hessian='exact'` skips the check);
3. solves the QP with `solve_qp`, warm-started from the current point and the previous multipliers (multiple
   shooting: the QP's states are kept, never re-rolled);
4. projects the inputs onto their bounds and takes the best step of the backtracking `ladder` on the $\ell_1$ merit
   function cost + `merit_weight` × (dynamics defects + constraint violations); a step that does not lower it is
   rejected.

Methods: `init(x0, params, X=None, U=None)`, `solve(state, x0, params)` (`sqp_iters` iterations, returns
`(state, info)` with `qp_iters`, `qp_kkt`, `cost`, `merit`), `step(state, x0, params)`, `shift(state)`,
`closed_loop(plant, x0, K, params_fn=None, state=None)` (cached per `plant`/`params_fn` object: pass the same
functions to reuse the compiled loop), and `qp`, `cost`, `merit` for inspection. Everything is jit-compatible, and
`jax.vmap` over states batches the SQP subproblems into one long horizon.

`parampc.models` defines the problems used throughout the repository: `quadrotor` (a planar quadrotor carrying a
slung load: hard thrust limits, soft swing limits, a time-varying reference) and `cartpole` (swing-up with the
non-convex cost $1-\cos\theta$). See [`notebooks/02_nonlinear_mpc.ipynb`](notebooks/02_nonlinear_mpc.ipynb),
[`examples/quadrotor_nmpc.py`](examples/quadrotor_nmpc.py) and [`examples/cartpole_swingup.py`](examples/cartpole_swingup.py).

## Backends

| `backend` | what runs | use it for |
|---|---|---|
| `'cuda'` | the whole interior-point solve as one persistent CUDA kernel (fp64) | a GPU: long horizons and batches |
| `'scan'` | the associative scans, compiled by XLA | a GPU without nvcc; runs on the CPU too |
| `'sweep'` | the sequential Riccati recursion inside the same IPM | the CPU, short horizons |
| `'auto'` | `'cuda'` on a GPU with nvcc, else `'sweep'` on a CPU, else `'scan'` | the default |

All backends run the same algorithm and agree to the solver tolerance; `pm.available_backends()` lists what can run
on the machine. The CUDA kernel handles $n, m \le 32$.

## How it works

The interior-point method applies Newton's method to the KKT conditions with the complementarity products
perturbed. Eliminating the multiplier and slack steps stage by stage turns every Newton system into an LQ problem
whose stage Hessians carry the barrier weights $W = \lambda / s$.

**The matrix part** is the cost-to-go $P_t$. Each stage defines a value-function element
$a_t = (\bar A_t, M_t, \bar Q_t)$ with $\bar A = A - B\tilde R^{-1}\tilde S$, $M = B\tilde R^{-1}B^\top$,
$\bar Q = \tilde Q - \tilde S^\top\tilde R^{-1}\tilde S$, and elements compose associatively:

$$
(A_1, C_1, J_1) \otimes (A_2, C_2, J_2) = \big(A_2 (I + C_1 J_2)^{-1} A_1,\ \ A_2 (I + C_1 J_2)^{-1} C_1 A_2^\top + C_2,\ \ A_1^\top (I + J_2 C_1)^{-1} J_2 A_1 + J_1\big).
$$

$P_t$ is the third component of the suffix product $a_t \otimes \cdots \otimes a_N$, so one associative scan gives
all of them in $\Theta(\log N)$ depth. The combine is evaluated in a Cholesky form (a Gram form for
$(I + J_2C_1)^{-1}J_2$) that stays accurate under barrier weights up to $10^{17}$; a semidefinite cost falls back to
an LU form.

**The vector part** is two affine recursions through the closed loop $A_t - B_tK_t$ (the costates backward, the
states forward), each an associative scan over affine maps.

**The iteration** is Mehrotra's predictor–corrector with one Gondzio centrality corrector: one factorization and
four solves per iteration, the corrector steps summed and refined once against the system they solve. The loop runs
on the device and returns its most accurate iterate.

**The CUDA kernel** (`src/cuda`) runs the whole solve in one cooperative launch, with grid-wide barriers between
phases: the factorization as a hybrid Blelloch / Kogge–Stone scan with a Riccati-form leaf level, the solves as
chunked recursions with a scan over the chunk boundaries, one warp per combine, fp64 throughout.

[`notebooks/03_verification.ipynb`](notebooks/03_verification.ipynb) checks the method numerically: the scan's
algebra, Newton steps against dense LU, certified optima, the numerical-stability claims, and the CUDA kernel
against the JAX implementation.

## Performance

Measured with [`notebooks/benchmarking/acados_benchmark.ipynb`](notebooks/benchmarking/acados_benchmark.ipynb)
against acados v0.6 (HPIPM, single-threaded C), on one laptop (Intel i7-11370H, NVIDIA RTX 3070 Laptop GPU),
performance power mode, no other GPU job. Each measurement solves one problem in a fresh process, after one second of
untimed solves; times are medians. Both solvers solve the same problems and return the same objective (to the solver
tolerances) at every horizon.

- **Linear MPC**: one QP, an open-loop unstable LTI system ($n=8$, $m=4$) with input and state boxes.
- **Quadrotor NMPC**: one SQP iteration of the slung-load quadrotor from the hover guess (the same SQP iterate in
  both solvers).

| $N$ | linear: acados | linear: `cuda` | linear: `scan` (GPU) | quadrotor: acados | quadrotor: `cuda` | quadrotor: `scan` (GPU) |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.16 ms | 2.05 ms (0.08×) | 10.4 ms (0.02×) | 0.35 ms | 4.59 ms (0.08×) | 25.2 ms (0.01×) |
| 64 | 0.65 ms | 2.42 ms (0.27×) | 11.8 ms (0.06×) | 1.73 ms | 5.88 ms (0.29×) | 32.9 ms (0.05×) |
| 256 | 2.81 ms | 3.20 ms (0.88×) | 16.3 ms (0.17×) | 7.58 ms | 8.57 ms (0.88×) | 45.2 ms (0.17×) |
| 1024 | 17.5 ms | 5.61 ms (**3.1×**) | 24.6 ms (0.71×) | 35.1 ms | 14.4 ms (**2.4×**) | 73.2 ms (0.48×) |
| 4096 | 93.5 ms | 16.2 ms (**5.8×**) | 52.8 ms (**1.8×**) | 203 ms | 40.8 ms (**5.0×**) | 150 ms (**1.4×**) |

In parentheses: acados' time divided by ParaMPC's (above 1: ParaMPC is faster).

- **GPU, `cuda`**: faster than acados from $N \approx 300$ (296 linear, 307 quadrotor, interpolated), 5.8× / 5.0×
  faster at $N = 4096$. From $N = 16$ to $4096$ its time grows 8–9×, acados' about 580×.
- **GPU, `scan`** (XLA): faster than acados from $N \approx 1600$ (linear) and $2700$ (quadrotor). Its floor at
  short horizons is kernel launches, which the single CUDA kernel removes.
- **Short horizons**: acados is about 13× faster at $N = 16$; the kernel's floor is the latency of double-precision
  arithmetic on a consumer GPU (about 0.25 ms per interior-point iteration).
- **CPU**: acados is 4.8–15× faster than the `scan` backend and 3.6–10× faster than `sweep` at every horizon: without
  parallel hardware the scan's extra work does not pay.
- **Batches** (`examples/batch.py`): on `cuda`, 256 linear MPC QPs ($N = 32$) solved together take about 0.15 ms per
  QP, 22× the throughput of one-at-a-time solves (3.3 ms each).

Caveats: acados runs with its default settings on one core (partial condensing tuned to the horizon could make it
faster); one consumer GPU with weak double precision (a data-centre GPU is untested); timings move by up to about 15%
between sessions, so the crossovers are approximate; batches are not compared with several acados instances in
parallel.

## Repository layout

```
src/
  py_wrapper/parampc/        the Python package
    core/                    the method in JAX: KKT system, scans (lq.py), Newton step, interior-point loop
    cuda.py                  the CUDA kernel's wrapper: build with nvcc, cache, XLA FFI registration
    qp.py, solver.py         linear MPC: OCPQP and its helpers; solve_qp, the reductions, batches
    nmpc.py, integrators.py  nonlinear MPC: OCP, NMPC
    backends.py              backend selection
    models/                  example problems: quadrotor, cart-pole, linear plants
    testing.py               dense references and KKT certificates
  cuda/
    include/                 the kernel's headers: configuration, context, and the declarations of each part
    src/                     their implementations: warp linear algebra, residual, factorization, solves, the
                             kernel (kernel.cu) and its XLA FFI handler (ffi.cu); ipm.cu, the file nvcc
                             compiles, includes them all as one translation unit
tests/                       pytest suite
notebooks/                   01 linear MPC, 02 nonlinear MPC, 03 verification
  benchmarking/              the benchmark against acados (notebook, worker, acados problem definitions)
examples/                    runnable scripts (each takes --backend)
```

## Tests

```bash
python -m pytest tests                              # CPU: the CUDA tests are skipped
```

```bash
JAX_PLATFORMS=cuda,cpu python -m pytest tests       # GPU: everything
```

The suite certifies every backend's answer on an independent dense KKT system (box, terminal, time-varying
coupled-row, semidefinite and $m > n$ problems), checks batches against individual solves, the quadrotor's optimum
and closed loop, convexification, and the CUDA kernel (its factorization read directly from its workspace, its
solutions against the JAX core, determinism).

## Limitations

- **Infeasibility** is not detected: an infeasible QP ends with status 2 or 1.
- **Soft constraints** make QPs ill-conditioned (slack weight `l2` ≈ 1e-2): on `'scan'` and `'cuda'` the attainable
  KKT score is then about 1e-8, not 1e-10; use `tol` ≥ 1e-7 (the NMPC uses 1e-6).
- **Batches** on `'scan'` and `'cuda'` share iterations: one hard QP slows the whole batch, and each QP's own KKT
  score can sit above `tol` (up to about the batch size times).
- **Sizes**: $n, m \le 32$ on the CUDA kernel; the JAX backends have no limit.
- **Guarantees**: no convergence guarantee for the SQP and no recursive-feasibility or closed-loop stability
  guarantee for the provided models (they have no terminal ingredients).

## License

MIT, see [LICENSE](LICENSE). Author: Banaan Kiamanesh.
