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
// The XLA FFI handler `ParampcIpm` is called from Python by parampc/cuda.py (jax.ffi + ctypes).
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "xla/ffi/api/ffi.h"

#include "parampc/config.cuh"
#include "parampc/context.cuh"
#include "parampc/warp_linalg.cuh"
#include "parampc/grid.cuh"
#include "parampc/residual.cuh"
#include "parampc/factor.cuh"
#include "parampc/solve.cuh"

namespace ffi = xla::ffi;

// ------------------------------------------------------------- the kernel ---
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

// ----------------------------------------------------------- host side ---
namespace {

size_t carve(Ctx& c, double* base, int N, int M, int P, int maxwarps) {
  const size_t E = N + 1;
  size_t off = 0;
  auto take = [&](double*& ptr, size_t n) { ptr = base ? base + off : nullptr; off += (n + 31) / 32 * 32; };
  take(c.x, E * NX); take(c.u, (size_t)N * NU); take(c.nu, E * NX); take(c.lam, (size_t)N * M); take(c.s, (size_t)N * M);
  take(c.W, (size_t)N * M); take(c.Qt, (size_t)N * N2); take(c.St, (size_t)N * NU * NX); take(c.Rt, (size_t)N * NU * NU);
  take(c.T, E * 3 * N2); take(c.T2, (E < KS_ALLOC ? E : KS_ALLOC) * 3 * N2); take(c.Pm, E * N2); take(c.Ls, (size_t)N * NU * NU); take(c.Ldi, (size_t)N * NU); take(c.K, (size_t)N * NU * NX);
  take(c.Acl, (size_t)N * N2); take(c.Fp, (size_t)N * N2); take(c.Ps, (size_t)N * N2);
  int HP = 0;
  while ((1 << HP) < P) ++HP;
  const size_t nb = (size_t)(HP > 1 ? HP - 1 : 0) * P * N2;
  take(c.Tf, nb); take(c.Tb, nb);
  take(c.r0, NX); take(c.rx, (size_t)N * NX); take(c.rxN, NX); take(c.ru, (size_t)N * NU); take(c.rd, (size_t)N * NX);
  take(c.rc, (size_t)N * M);
  take(c.w, (size_t)N * M); take(c.rcs, (size_t)N * M); take(c.rhou, (size_t)N * NU); take(c.chat, (size_t)N * NX); take(c.pch, (size_t)N * NX); take(c.fv, (size_t)N * NX);
  take(c.g, (size_t)N * NX); take(c.ql, (size_t)N * NX); take(c.p, E * NX); take(c.kv, (size_t)N * NU);
  take(c.yl, E * NX); take(c.yend, (size_t)P * NX); take(c.dxb, E * NX); take(c.rhoxN, NX); take(c.dx0, NX);
  take(c.corr, (size_t)N * M); take(c.dv, (size_t)N * M);
  take(c.Sx, E * NX); take(c.Su, (size_t)N * NU); take(c.Snu, E * NX); take(c.Slam, (size_t)N * M); take(c.Ss, (size_t)N * M);
  take(c.Dx, E * NX); take(c.Du, (size_t)N * NU); take(c.Dnu, E * NX); take(c.Dlam, (size_t)N * M); take(c.Ds, (size_t)N * M);
  take(c.red, (size_t)maxwarps * RK);
  take(c.prof, 64);
  return off * sizeof(double);
}

struct Launch {
  int wpb = 0, ws = 0, wsg = 0, sms = 0, per_sm = 0;
  size_t smem = 0;
};

// smem (doubles) per stage task and per warp: a stage task needs its matrices and vectors of length M; a warp
// holds SPW of them, a combine (17 n^2 + 2n: G, Y, temporaries, Cholesky workspace) or a leaf
// (8 n^2 + 5 nm + 2 m^2 + m)
void smem_sizes(int M, int& wsg, int& ws) {
  const int build = N2 + 5 * NU * NX + 2 * NU * NU + NU + M;
  const int solve = 8 * NX + 4 * NU + 3 * M;
  wsg = ((build > solve ? build : solve) + 7) / 8 * 8;
  ws = 17 * N2 + 2 * NX;
  const int leaf = 8 * N2 + 5 * NX * NU + 2 * NU * NU + NU;
  if (leaf > ws) ws = leaf;
  if (SPW * wsg > ws) ws = SPW * wsg;
  ws = (ws + 7) / 8 * 8;
}

ffi::Error configure(Launch& lc, int M) {
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return ffi::Error::Internal("cudaGetDevice failed");
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return ffi::Error::Internal("cudaGetDeviceProperties failed");
  if (!prop.cooperativeLaunch) return ffi::Error::Internal("the GPU does not support cooperative launches");
  smem_sizes(M, lc.wsg, lc.ws);
  const size_t per_warp = (size_t)lc.ws * sizeof(double);
  const size_t cap = prop.sharedMemPerBlockOptin;
  lc.wpb = PM_THREADS / 32;
  while (lc.wpb > 1 && per_warp * lc.wpb > cap) lc.wpb /= 2;
  if (per_warp * lc.wpb > cap) return ffi::Error::InvalidArgument("state dimension too large for shared memory");
  lc.smem = per_warp * lc.wpb;
  if (cudaFuncSetAttribute(ipm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)lc.smem) != cudaSuccess)
    return ffi::Error::Internal("cudaFuncSetAttribute failed");
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&lc.per_sm, ipm_kernel, lc.wpb * 32, lc.smem) != cudaSuccess ||
      lc.per_sm < 1)
    return ffi::Error::Internal("the kernel does not fit on a multiprocessor");
  lc.sms = prop.multiProcessorCount;
  return ffi::Error::Success();
}

}  // namespace

static ffi::Error IpmImpl(cudaStream_t stream,
                          ffi::Buffer<ffi::F64> A, ffi::Buffer<ffi::F64> B, ffi::Buffer<ffi::F64> c_,
                          ffi::Buffer<ffi::F64> Q, ffi::Buffer<ffi::F64> S, ffi::Buffer<ffi::F64> R,
                          ffi::Buffer<ffi::F64> q, ffi::Buffer<ffi::F64> r, ffi::Buffer<ffi::F64> QN,
                          ffi::Buffer<ffi::F64> qN, ffi::Buffer<ffi::F64> C, ffi::Buffer<ffi::F64> D,
                          ffi::Buffer<ffi::F64> d, ffi::Buffer<ffi::F64> xbar, ffi::Buffer<ffi::F64> x0,
                          ffi::Buffer<ffi::F64> u0, ffi::Buffer<ffi::F64> nu0, ffi::Buffer<ffi::F64> lam0,
                          ffi::Buffer<ffi::F64> s0, ffi::Buffer<ffi::F64> prm, ffi::ResultBuffer<ffi::F64> x,
                          ffi::ResultBuffer<ffi::F64> u, ffi::ResultBuffer<ffi::F64> nu,
                          ffi::ResultBuffer<ffi::F64> lam, ffi::ResultBuffer<ffi::F64> s,
                          ffi::ResultBuffer<ffi::S32> kout, ffi::ResultBuffer<ffi::F64> info,
                          ffi::ResultBuffer<ffi::F64> work, int64_t blocks) {
  auto Ad = A.dimensions(), Bd = B.dimensions(), Cd = C.dimensions();
  if (Ad.size() != 3 || Ad[1] != NX || Ad[2] != NX || Bd.size() != 3 || Bd[2] != NU || Cd.size() != 3)
    return ffi::Error::InvalidArgument("parampc_ipm: data shapes do not match the compiled sizes (NX, NU)");
  const int N = (int)Ad[0], M = (int)Cd[1];
  if (N < 1 || M < 1) return ffi::Error::InvalidArgument("parampc_ipm: needs N >= 1 and at least one inequality row");
  if (prm.element_count() < 4) return ffi::Error::InvalidArgument("parampc_ipm: prm = (tol, mu_min, max_iter, start)");

  static Launch lc;
  static int lc_M = -1;
  if (lc_M != M) {
    ffi::Error err = configure(lc, M);
    if (err.failure()) return err;
    lc_M = M;
  }
  int L = (int)ceil(sqrt((double)N));
  if (L < 1) L = 1;
  while (2 * ((N + L - 1) / L) * NX > lc.wpb * lc.ws) ++L;   // the boundary pass holds 2 P NX doubles in a block
  const int P = (N + L - 1) / L;
  int H = 0;
  while ((1 << H) < N + 1) ++H;
  const int gmax = lc.sms * lc.per_sm;
  int G = (int)blocks;
  if (G <= 0) G = N + 1;                           // auto: as many blocks as elements, up to what fits
  if (G > gmax) G = gmax;
  if (G < 1) G = 1;

  Ctx c{};
  c.N = N; c.M = M; c.E = N + 1; c.L = L; c.P = P; c.H = H; c.WS = lc.ws; c.WSG = lc.wsg;
  int kmax = PM_KS_PER_SM * lc.sms;
  if (kmax > KS_ALLOC) kmax = KS_ALLOC;
  c.ksr = 0;
  while (((N + 1 + (1 << c.ksr) - 1) >> c.ksr) > kmax) ++c.ksr;
  c.A = A.typed_data(); c.B = B.typed_data(); c.c = c_.typed_data(); c.Q = Q.typed_data(); c.S = S.typed_data();
  c.R = R.typed_data(); c.q = q.typed_data(); c.r = r.typed_data(); c.QN = QN.typed_data(); c.qN = qN.typed_data();
  c.C = C.typed_data(); c.D = D.typed_data(); c.d = d.typed_data(); c.xbar = xbar.typed_data(); c.prm = prm.typed_data();
  c.bx = x->typed_data(); c.bu = u->typed_data(); c.bnu = nu->typed_data(); c.blam = lam->typed_data();
  c.bs = s->typed_data(); c.kout = kout->typed_data(); c.info = info->typed_data();

  if (G * lc.wpb > MAXWARPS) G = MAXWARPS / lc.wpb;
  if (work->size_bytes() < carve(c, nullptr, N, M, P, MAXWARPS))
    return ffi::Error::InvalidArgument("parampc_ipm: workspace too small (use parampc_ipm_workspace)");
  carve(c, work->typed_data(), N, M, P, MAXWARPS);

  // current iterate and best iterate both start at z0
  const size_t E = N + 1;
  struct { double* dst; double* dst2; const double* src; size_t n; } copies[] = {
      {c.x, c.bx, x0.typed_data(), E * NX}, {c.u, c.bu, u0.typed_data(), (size_t)N * NU},
      {c.nu, c.bnu, nu0.typed_data(), E * NX}, {c.lam, c.blam, lam0.typed_data(), (size_t)N * M},
      {c.s, c.bs, s0.typed_data(), (size_t)N * M}};
  for (auto& cp : copies) {
    cudaMemcpyAsync(cp.dst, cp.src, cp.n * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(cp.dst2, cp.src, cp.n * sizeof(double), cudaMemcpyDeviceToDevice, stream);
  }
  void* args[] = {&c};
  cudaError_t st = cudaLaunchCooperativeKernel((void*)ipm_kernel, dim3(G), dim3(lc.wpb * 32), args, lc.smem, stream);
  if (st != cudaSuccess) return ffi::Error::Internal(std::string("parampc_ipm launch: ") + cudaGetErrorString(st));
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(ParampcIpm, IpmImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::F64>>()   // A
                                  .Arg<ffi::Buffer<ffi::F64>>()   // B
                                  .Arg<ffi::Buffer<ffi::F64>>()   // c
                                  .Arg<ffi::Buffer<ffi::F64>>()   // Q
                                  .Arg<ffi::Buffer<ffi::F64>>()   // S
                                  .Arg<ffi::Buffer<ffi::F64>>()   // R
                                  .Arg<ffi::Buffer<ffi::F64>>()   // q
                                  .Arg<ffi::Buffer<ffi::F64>>()   // r
                                  .Arg<ffi::Buffer<ffi::F64>>()   // QN
                                  .Arg<ffi::Buffer<ffi::F64>>()   // qN
                                  .Arg<ffi::Buffer<ffi::F64>>()   // C
                                  .Arg<ffi::Buffer<ffi::F64>>()   // D
                                  .Arg<ffi::Buffer<ffi::F64>>()   // d
                                  .Arg<ffi::Buffer<ffi::F64>>()   // xbar
                                  .Arg<ffi::Buffer<ffi::F64>>()   // x0
                                  .Arg<ffi::Buffer<ffi::F64>>()   // u0
                                  .Arg<ffi::Buffer<ffi::F64>>()   // nu0
                                  .Arg<ffi::Buffer<ffi::F64>>()   // lam0
                                  .Arg<ffi::Buffer<ffi::F64>>()   // s0
                                  .Arg<ffi::Buffer<ffi::F64>>()   // (tol, mu_min, max_iter, start: 0 given, 1 Mehrotra)
                                  .Ret<ffi::Buffer<ffi::F64>>()   // x
                                  .Ret<ffi::Buffer<ffi::F64>>()   // u
                                  .Ret<ffi::Buffer<ffi::F64>>()   // nu
                                  .Ret<ffi::Buffer<ffi::F64>>()   // lam
                                  .Ret<ffi::Buffer<ffi::F64>>()   // s
                                  .Ret<ffi::Buffer<ffi::S32>>()   // iterations
                                  .Ret<ffi::Buffer<ffi::F64>>()   // (best KKT score, final mu)
                                  .Ret<ffi::Buffer<ffi::F64>>()   // workspace
                                  .Attr<int64_t>("blocks"));

// workspace size in doubles, for the caller to allocate (the last output of the FFI call)
extern "C" int64_t parampc_ipm_workspace(int64_t N, int64_t M) {
  Ctx c{};
  int L = (int)ceil(sqrt((double)N));
  if (L < 1) L = 1;
  return (int64_t)(carve(c, nullptr, (int)N, (int)M, (int)((N + L - 1) / L), MAXWARPS) / sizeof(double));
}
