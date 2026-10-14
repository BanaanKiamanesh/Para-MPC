// The kernel context: problem data, iterates and the workspace arrays, all in global memory.
#pragma once

#include "config.cuh"

struct Ctx {
  int N, M, E, L, P, H, WS, WSG;            // horizon, rows, N+1, chunk length / count, tree height, smem/warp, /group
  int ksr;                                  // Blelloch levels below the Kogge-Stone part of the scan
  const double *A, *B, *c, *Q, *S, *R, *q, *r, *QN, *qN, *C, *D, *d, *xbar, *prm;   // data (xbar: initial state)
  double *x, *u, *nu, *lam, *s;             // current iterate
  double *bx, *bu, *bnu, *blam, *bs;        // best iterate (the outputs)
  double* info;
  int* kout;
  double *W, *Qt, *St, *Rt, *T, *T2, *Pm, *Ls, *Ldi, *K, *Acl, *Fp, *Ps;   // factorization
  double *Tf, *Tb;                          // scan matrices over the chunk boundaries (forward, backward)
  double *r0, *rx, *rxN, *ru, *rd, *rc;                          // residuals at the current iterate
  double *w, *rcs, *rhou, *chat, *pch, *g, *ql, *p, *kv, *fv, *yl, *yend, *dxb, *rhoxN, *dx0, *corr, *dv;
  double *Sx, *Su, *Snu, *Slam, *Ss, *Dx, *Du, *Dnu, *Dlam, *Ds;  // step and direction
  double* red;
  double* prof;                             // per-phase cycle counts (-DPM_PROFILE)
};

// An array of the value-function scan (Qt, St, Rt, T, T2) in the scalar type F of the factorization: double, or
// float in the float32 phase (the workspace is carved in doubles, so a float array uses the first half of its
// slot). What the solves read (W, Pm, Ls, Ldi, K, Acl, Fp, Ps, Tf, Tb) is double in both.
template <typename F>
__device__ __forceinline__ F* fa(double* p) {
  return reinterpret_cast<F*>(p);
}
