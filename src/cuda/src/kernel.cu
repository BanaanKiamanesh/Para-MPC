// The whole interior-point solve of parampc.core.ipm on the GPU, as one persistent CUDA kernel.
//
// Same algorithm and defaults as parampc.core.ipm: one factorization per iteration, an unrefined
// predictor, an unrefined corrector, one centrality corrector, the sum refined once; step to the boundary
// with fraction 0.995; stop at tol, mu_min or max_iter and return the best iterate. Everything runs inside
// a single cooperative launch, with grid-wide barriers between phases (a block barrier when the grid is one
// block), so a whole QP costs one kernel launch.
//
//   factorization   P_t as suffix products of the value-function elements, by a scan in three parts
//                   (`factor`): Blelloch up-sweep levels of general combines (Cholesky form) until at
//                   most 3 x SMs partial products remain, a Kogge-Stone scan over those (Riccati-type combines,
//                   where the partner reaches the end of the horizon), and the matching down-sweep
//                   levels; short horizons are one Kogge-Stone scan, ceil(log2(N+1)) levels. The leaf level
//                   combines stage elements, whose M = B Rt^-1 B' has rank m, in Riccati form (one m x m
//                   Cholesky instead of two n x n ones). If P is not finite (a semidefinite cost), the scan is
//                   redone with the LU combine, as parampc.core.factor_lq does.
//   solves          the two affine recursions of each solve by chunks: local recursions
//                   inside chunks of length L ~ sqrt(N), a Hillis-Steele scan over the chunk boundaries in one
//                   block, then a direct fix-up. The matrices of the boundary scan and the fix-up depend on the
//                   iterate only and are computed once per iteration (the matrix part), so the four solves of an
//                   iteration do matrix-vector work only.
//   linear algebra  one warp per combine, its lanes over matrix entries; one warp per stage, or SPW stages
//                   per warp (GP lanes each) when there are more stages than warps; matrices in shared
//                   memory; float64 throughout; sizes NX, NU fixed at compile time (-DPM_NX, -DPM_NU).
//
// The host side (launch configuration, workspace, the XLA FFI handler) is in ffi.cu.
#include "kernel.cuh"

#include "factor.cuh"
#include "grid.cuh"
#include "residual.cuh"
#include "solve.cuh"
#include "warp_linalg.cuh"

// -DPM_PROFILE: thread 0 adds the cycles since its last mark to slot i (phases end with a barrier)
#ifdef PM_PROFILE
#define MARK(i)                                                                \
  do {                                                                         \
    if (gw == 0 && lane == 0) {                                                \
      const long long t1 = clock64();                                          \
      c.prof[i] += (double)(t1 - t0);                                          \
      t0 = t1;                                                                 \
    }                                                                          \
  } while (0)
#else
#define MARK(i) do { } while (0)
#endif

// The compiler keeps each task's lane-dependent index arithmetic in registers for the whole kernel, so the
// kernel sits near the 255-register limit. Two things keep it there without spills: the launch bound states
// one block per multiprocessor (else ptxas may aim at 128 registers and spill kilobytes), and the phase lambdas
// are always inlined (else `c` is copied to local memory and every pointer load goes through it).
__global__ void __launch_bounds__(PM_THREADS, 1) ipm_kernel(Ctx c) {
  cg::grid_group grid = cg::this_grid();
  extern __shared__ double smem[];
  const int lane = threadIdx.x & 31, wib = threadIdx.x >> 5, wpb = blockDim.x >> 5;
  // warp rank, block-minor: consecutive tasks go to different multiprocessors (each has only two fp64 units)
  const int gw = wib * gridDim.x + blockIdx.x, TW = gridDim.x * wpb;
  double* ws = smem + (size_t)wib * c.WS;
  const double tol = c.prm[0], mu_min = c.prm[1];
  const int max_iter = (int)c.prm[2];
  const double NM = (double)c.N * c.M;
#ifdef PM_PROFILE
  long long t0 = clock64();
  if (gw == 0) for (int i = lane; i < 64; i += 32) c.prof[i] = 0.0;
#endif
  // per-stage phases: one stage per warp, or SPW stages per warp (GP lanes each) when there are more stages
  // than warps; stages(last, f) calls f(t, lanes, shared memory) for t = 0..last
  const bool pack = c.N + 1 > TW;
  const int gi = lane / GP;
  const Lanes<GP> lg{lane % GP, GP == 32 ? FULL : (FULL >> (32 - GP)) << (gi * GP)};
  double* wsg = ws + (size_t)gi * c.WSG;
  auto stages = [&](int last, auto f) __attribute__((always_inline)) {
    if (pack) {
      for (int t0 = gw * SPW; t0 <= last; t0 += TW * SPW)
        if (t0 + gi <= last) f(t0 + gi, lg, wsg);
    } else {
      for (int t = gw; t <= last; t += TW) f(t, Lanes<32>{lane, FULL}, ws);
    }
  };
  // element-wise loops over whole arrays: all lanes of all warps
  const size_t gt = (size_t)gw * 32 + lane, GT = (size_t)TW * 32;
#define EACH(i, n) for (size_t i = gt; i < (size_t)(n); i += GT)

  auto residual_phase = [&](double& score, double& mu) __attribute__((always_inline)) {
    double pmax = 0.0, psum = 0.0;
    stages(c.N, [&](int t, auto ln, double*) __attribute__((always_inline)) { task_residual(c, t, ln, pmax, psum); });
    put(c, gw, lane, 0, pmax, MAX);
    put(c, gw, lane, 1, psum, SUM);
    bar(grid);
    MARK(0);
    const double feas = get(c, TW, lane, 0, MAX);
    mu = get(c, TW, lane, 1, SUM) / NM;
    score = nmax(mu, feas);
  };

  auto solve = [&](const SolveArgs& a, double frac, double& ap, double& ad) __attribute__((always_inline)) {
    stages(c.N, [&](int t, auto ln, double* w) __attribute__((always_inline)) { task_s1(c, t, ln, w, a); });
    bar(grid);
    MARK(6);
    for (int ch = gw; ch < c.P; ch += TW) task_s2(c, ch, lane, ws);
    bar(grid);
    MARK(7);
    if (blockIdx.x == 0) task_s3(c, smem);
    bar(grid);
    MARK(8);
    stages(c.N - 1, [&](int t, auto ln, double* w) __attribute__((always_inline)) { task_s4a(c, t, ln, w); });
    bar(grid);
    MARK(9);
    for (int ch = gw; ch < c.P; ch += TW) task_s4b(c, ch, lane, ws);
    bar(grid);
    MARK(3);
    if (blockIdx.x == 0) task_s5(c, smem);
    bar(grid);
    MARK(10);
    double rs = INFINITY, rl = INFINITY;
    stages(c.N, [&](int t, auto ln, double* w) __attribute__((always_inline)) { task_s6(c, t, ln, w, a, rs, rl); });
    put(c, gw, lane, 0, rs, MIN);
    put(c, gw, lane, 1, rl, MIN);
    bar(grid);
    MARK(11);
    ap = nmin(frac * get(c, TW, lane, 0, MIN), 1.0);
    ad = nmin(frac * get(c, TW, lane, 1, MIN), 1.0);
  };

  auto build_phase = [&]() __attribute__((always_inline)) {
    stages(c.N, [&](int t, auto ln, double* w) __attribute__((always_inline)) { task_build(c, t, ln, w); });
    bar(grid);
  };

  double best_score, mu;
  residual_phase(best_score, mu);
  const bool mehrotra = c.prm[3] != 0.0;
  int k = 0;
  while ((mehrotra && k == 0) || (k < max_iter && best_score > tol && mu > mu_min)) {
    // With start = Mehrotra, the first iteration computes Mehrotra's starting point instead (counted as one
    // iteration, as core.mehrotra_start): the refined affine-scaling step at tau = 0, taken in full, then s and
    // lam shifted positive and balanced.
    const bool st = mehrotra && k == 0;
    if (st) EACH(i, c.N * c.M) c.corr[i] = 0.0;                // no second-order term (read after a barrier)
    // ---- factorization; a second pass with LU combines if P is not finite (a semidefinite cost). One loop, so
    // that the factorization is inlined once (two copies made the kernel slower everywhere)
    // The chunk transitions are computed in the same phase as the check of P (one barrier fewer); after the rare
    // LU pass they are computed again.
    for (int lu = 0;; lu = 1) {
      build_phase();
      MARK(1);
      double bad = 0.0;
      factor(c, grid, gw, TW, lane, ws, lu, bad);
      MARK(2);
      for (int task = gw; task < 2 * c.P; task += TW) task_chunk(c, task, lane);
      put(c, gw, lane, 2, bad, MAX);
      bar(grid);
      MARK(5);
      if (lu || get(c, TW, lane, 2, MAX) == 0.0) break;
    }
    for (int l = 1; (1 << l) < c.P; ++l) {                     // scan matrices over the chunk boundaries
      bnd_level(c, l, gw, TW, lane);
      bar(grid);
    }
    MARK(4);
    // ---- the solves, through one call site (each call site inlines the whole solve): predictor, centering;
    // corrector, centrality corrector, refinement of their sum. The starting point uses the corrector at
    // tau = 0 and its refinement.
    double ap = 0.0, ad = 0.0, tau = 0.0, tp = 0.0, td = 0.0;
    bool keep = false;
    for (int kind = st ? CORR : PRED; kind <= REFN; ++kind) {
      if (st && kind == GOND) continue;
      double a1, a2;
      solve(SolveArgs{kind, tau, tp, td, keep}, kind == PRED ? 1.0 : 0.995, a1, a2);
      if (kind == PRED) {                                      // centering: sigma = (mu_aff / mu)^3
        double psum = 0.0;
        EACH(i, c.N * c.M) psum += (c.lam[i] + a2 * c.Dlam[i]) * (c.s[i] + a1 * c.Ds[i]);
        put(c, gw, lane, 3, psum, SUM);
        bar(grid);
        MARK(12);
        const double ratio = (get(c, TW, lane, 3, SUM) / NM) / mu;
        tau = clip01(ratio * (ratio * ratio)) * mu;
      } else if (kind == CORR) {
        ap = a1; ad = a2;
        tp = nmin(ap + 0.1, 1.0); td = nmin(ad + 0.1, 1.0);
      } else if (kind == GOND) {
        keep = nmin(a1, a2) >= nmin(ap, ad) + 0.1 * 0.1;
      } else {
        ap = a1; ad = a2;
      }
    }
    // ---- update, score, best iterate
    if (st) {
      EACH(i, c.E * NX) { c.x[i] = c.x[i] + c.Sx[i]; c.nu[i] = c.nu[i] + c.Snu[i]; }
      EACH(i, c.N * NU) c.u[i] = c.u[i] + c.Su[i];
      double smin = INFINITY, lmin = INFINITY;
      EACH(i, c.N * c.M) {
        const double sv = c.s[i] + c.Ss[i], lv = c.lam[i] + c.Slam[i];
        c.s[i] = sv; c.lam[i] = lv;
        smin = nmin(smin, sv); lmin = nmin(lmin, lv);
      }
      put(c, gw, lane, 4, smin, MIN);
      put(c, gw, lane, 5, lmin, MIN);
      bar(grid);
      const double ds = nmax(-1.5 * get(c, TW, lane, 4, MIN), 0.0), dl = nmax(-1.5 * get(c, TW, lane, 5, MIN), 0.0);
      double ls = 0.0, lsum = 0.0, ssum = 0.0;
      EACH(i, c.N * c.M) {
        const double sv = c.s[i] + ds, lv = c.lam[i] + dl;
        ls += lv * sv; lsum += lv; ssum += sv;
      }
      put(c, gw, lane, 6, ls, SUM);
      put(c, gw, lane, 7, lsum, SUM);
      put(c, gw, lane, 0, ssum, SUM);
      bar(grid);
      const double lsv = get(c, TW, lane, 6, SUM), es = 0.5 * lsv / get(c, TW, lane, 7, SUM);
      const double el = 0.5 * lsv / get(c, TW, lane, 0, SUM);
      EACH(i, c.N * c.M) { c.s[i] = (c.s[i] + ds) + es; c.lam[i] = (c.lam[i] + dl) + el; }
    } else {
      EACH(i, c.E * NX) { c.x[i] = c.x[i] + ap * c.Sx[i]; c.nu[i] = c.nu[i] + ad * c.Snu[i]; }
      EACH(i, c.N * NU) c.u[i] = c.u[i] + ap * c.Su[i];
      EACH(i, c.N * c.M) { c.s[i] = c.s[i] + ap * c.Ss[i]; c.lam[i] = c.lam[i] + ad * c.Slam[i]; }
    }
    bar(grid);
    MARK(13);
    double score;
    residual_phase(score, mu);
    if (st || score < best_score) {
      best_score = score;
      EACH(i, c.E * NX) { c.bx[i] = c.x[i]; c.bnu[i] = c.nu[i]; }
      EACH(i, c.N * NU) c.bu[i] = c.u[i];
      EACH(i, c.N * c.M) { c.bs[i] = c.s[i]; c.blam[i] = c.lam[i]; }
    }
    ++k;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *c.kout = k;
    c.info[0] = best_score;
    c.info[1] = mu;
  }
}
#undef EACH
