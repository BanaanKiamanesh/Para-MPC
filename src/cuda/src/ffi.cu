// Host side of the kernel: workspace layout, launch configuration, and the XLA FFI handler `ParampcIpm` that
// parampc/cuda.py registers with jax.ffi (loaded with ctypes), plus the workspace size for the caller.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <string>

#include "xla/ffi/api/ffi.h"

#include "config.cuh"
#include "context.cuh"
#include "kernel.cuh"

namespace ffi = xla::ffi;

namespace {

size_t carve(Ctx& c, double* base, int N, int M, int P, int maxwarps) {
  const size_t E = N + 1;
  size_t off = 0;
  auto take = [&](double*& ptr, size_t n) { ptr = base ? base + off : nullptr; off += (n + 31) / 32 * 32; };
  take(c.x, E * NX); take(c.u, (size_t)N * NU); take(c.nu, E * NX); take(c.lam, (size_t)N * M);
  take(c.s, (size_t)N * M);
  take(c.W, (size_t)N * M); take(c.Qt, (size_t)N * N2); take(c.St, (size_t)N * NU * NX);
  take(c.Rt, (size_t)N * NU * NU);
  take(c.T, E * 3 * N2); take(c.T2, (E < KS_ALLOC ? E : KS_ALLOC) * 3 * N2); take(c.Pm, E * N2);
  take(c.Ls, (size_t)N * NU * NU); take(c.Ldi, (size_t)N * NU); take(c.K, (size_t)N * NU * NX);
  take(c.Acl, (size_t)N * N2); take(c.Fp, (size_t)N * N2); take(c.Ps, (size_t)N * N2);
  int HP = 0;
  while ((1 << HP) < P) ++HP;
  const size_t nb = (size_t)(HP > 1 ? HP - 1 : 0) * P * N2;
  take(c.Tf, nb); take(c.Tb, nb);
  take(c.r0, NX); take(c.rx, (size_t)N * NX); take(c.rxN, NX); take(c.ru, (size_t)N * NU); take(c.rd, (size_t)N * NX);
  take(c.rc, (size_t)N * M);
  take(c.w, (size_t)N * M); take(c.rcs, (size_t)N * M); take(c.rhou, (size_t)N * NU); take(c.chat, (size_t)N * NX);
  take(c.pch, (size_t)N * NX); take(c.fv, (size_t)N * NX);
  take(c.g, (size_t)N * NX); take(c.ql, (size_t)N * NX); take(c.p, E * NX); take(c.kv, (size_t)N * NU);
  take(c.yl, E * NX); take(c.yend, (size_t)P * NX); take(c.dxb, E * NX); take(c.rhoxN, NX); take(c.dx0, NX);
  take(c.corr, (size_t)N * M); take(c.dv, (size_t)N * M);
  take(c.Sx, E * NX); take(c.Su, (size_t)N * NU); take(c.Snu, E * NX); take(c.Slam, (size_t)N * M);
  take(c.Ss, (size_t)N * M);
  take(c.Dx, E * NX); take(c.Du, (size_t)N * NU); take(c.Dnu, E * NX); take(c.Dlam, (size_t)N * M);
  take(c.Ds, (size_t)N * M);
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
  c.C = C.typed_data(); c.D = D.typed_data(); c.d = d.typed_data(); c.xbar = xbar.typed_data();
  c.prm = prm.typed_data();
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
