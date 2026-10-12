// One solve with the factorization: right-hand side, the two affine recursions by chunks with a scan over
// the chunk boundaries, and the recovery of the step.
#pragma once

#include "parampc/context.cuh"
#include "parampc/warp_linalg.cuh"

// ------------------------------------------------------------- one solve ---
struct SolveArgs {
  int kind;
  double tau, tp, td;   // tau = sigma mu; tp, td: trial step lengths of the centrality corrector
  bool keep;            // refinement: whether the centrality correction was kept
};

// the step being refined: STEP, plus DIR if the centrality correction was kept
__device__ __forceinline__ double stp(const double* Sv, const double* Dv, size_t i, bool keep) {
  return keep ? Sv[i] + Dv[i] : Sv[i];
}

// S1: right-hand side of the solve (newton_direction's lq_rhs and the g of solve_lq)
template <int G>
__device__ __forceinline__ void task_s1(const Ctx& c, int t, Lanes<G> ln, double* ws, const SolveArgs& a) {
  const int lane = ln.sl;                                            // the lane within the group
  const int M = c.M;
  const bool ref = a.kind == REFN, zero = a.kind == GOND;
  if (t == c.N) {
    for (int i = lane; i < NX; i += G) {
      double rxN = 0.0, r0 = 0.0;
      if (!zero) {
        rxN = c.rxN[i];
        r0 = c.r0[i];
        if (ref) {
          double qx = 0.0;
          for (int j = 0; j < NX; ++j) qx += c.QN[i * NX + j] * stp(c.Sx, c.Dx, (size_t)c.N * NX + j, a.keep);
          rxN = rxN + (stp(c.Snu, c.Dnu, (size_t)c.N * NX + i, a.keep) - qx);
          r0 = r0 + stp(c.Sx, c.Dx, i, a.keep);
        }
      }
      c.rhoxN[i] = -rxN;
      c.dx0[i] = -r0;
    }
    __syncwarp(ln.gm);
    return;
  }
  const double *A = c.A + (size_t)t * N2, *B = c.B + (size_t)t * NX * NU, *Q = c.Q + (size_t)t * N2;
  const double *S = c.S + (size_t)t * NU * NX, *R = c.R + (size_t)t * NU * NU;
  const double *C = c.C + (size_t)t * M * NX, *D = c.D + (size_t)t * M * NU;
  const double *lt = c.lam + (size_t)t * M, *st = c.s + (size_t)t * M;
  double *sx0 = ws, *sx1 = sx0 + NX, *su = sx1 + NX, *sn0 = su + NU, *sn1 = sn0 + NX, *wv = sn1 + NX;
  double *rxv = wv + M, *ruv = rxv + NX, *chv = ruv + NU, *pc = chv + NX, *sl = pc + NX, *ss = sl + M;
  if (ref) {
    for (int i = lane; i < NX; i += G) {
      sx0[i] = stp(c.Sx, c.Dx, (size_t)t * NX + i, a.keep);
      sx1[i] = stp(c.Sx, c.Dx, (size_t)(t + 1) * NX + i, a.keep);
      sn0[i] = stp(c.Snu, c.Dnu, (size_t)t * NX + i, a.keep);
      sn1[i] = stp(c.Snu, c.Dnu, (size_t)(t + 1) * NX + i, a.keep);
    }
    for (int i = lane; i < NU; i += G) su[i] = stp(c.Su, c.Du, (size_t)t * NU + i, a.keep);
    for (int k = lane; k < M; k += G) {
      sl[k] = stp(c.Slam, c.Dlam, (size_t)t * M + k, a.keep);
      ss[k] = stp(c.Ss, c.Ds, (size_t)t * M + k, a.keep);
    }
    __syncwarp(ln.gm);
  }
  // complementarity rows: rc, rt, then w = (lam rc - rt) / s
  for (int k = lane; k < M; k += G) {
    const size_t i = (size_t)t * M + k;
    double rc = 0.0, rt;
    if (a.kind == GOND) {
      const double v = (lt[k] + a.td * c.Slam[i]) * (st[k] + a.tp * c.Ss[i]);
      const double lo = 0.1 * a.tau, hi = 10.0 * a.tau;
      const double dv = nmax(nmin(nmax(v, lo), hi) - v, -hi);
      c.dv[i] = dv;
      rt = -dv;
    } else {
      rc = c.rc[i];
      const double ls = lt[k] * st[k];
      if (a.kind == PRED) rt = ls - 0.0 + 0.0;
      else if (a.kind == CORR) rt = (ls - a.tau) + c.corr[i];
      else {
        rt = (ls - a.tau) + (c.corr[i] - (a.keep ? c.dv[i] : 0.0));
        double cx = 0.0, du = 0.0;
        for (int j = 0; j < NX; ++j) cx += C[k * NX + j] * sx0[j];
        for (int j = 0; j < NU; ++j) du += D[k * NU + j] * su[j];
        rc = rc + (cx + du + ss[k]);
        rt = rt + (sl[k] * st[k] + lt[k] * ss[k]);
      }
    }
    c.rcs[i] = rc;
    const double wk = (lt[k] * rc - rt) / st[k];
    wv[k] = wk;
    c.w[i] = wk;
  }
  // linear rows rx, ru, rd (plus J * step for the refinement)
  for (int i = lane; i < NX; i += G) {
    double rx = 0.0, rd = 0.0;
    if (!zero) {
      rx = c.rx[(size_t)t * NX + i];
      rd = c.rd[(size_t)t * NX + i];
      if (ref) {
        double qx = 0.0, su_ = 0.0, an = 0.0, cl = 0.0, ax = 0.0, bu = 0.0;
        for (int j = 0; j < NX; ++j) { qx += Q[i * NX + j] * sx0[j]; an += A[j * NX + i] * sn1[j]; ax += A[i * NX + j] * sx0[j]; }
        for (int j = 0; j < NU; ++j) { su_ += S[j * NX + i] * su[j]; bu += B[i * NU + j] * su[j]; }
        for (int k = 0; k < M; ++k) cl += C[k * NX + i] * sl[k];
        rx = rx + (sn0[i] - qx - su_ - an - cl);
        rd = rd + (sx1[i] - ax - bu);
      }
    }
    rxv[i] = rx;
    chv[i] = -rd;
  }
  for (int i = lane; i < NU; i += G) {
    double ru = 0.0;
    if (!zero) {
      ru = c.ru[(size_t)t * NU + i];
      if (ref) {
        double r_ = 0.0, sx = 0.0, bn = 0.0, dl = 0.0;
        for (int j = 0; j < NU; ++j) r_ += R[i * NU + j] * su[j];
        for (int j = 0; j < NX; ++j) { sx += S[i * NX + j] * sx0[j]; bn += B[j * NU + i] * sn1[j]; }
        for (int k = 0; k < M; ++k) dl += D[k * NU + i] * sl[k];
        ru = ru + (r_ + sx + bn + dl);
      }
    }
    ruv[i] = ru;
  }
  __syncwarp(ln.gm);
  // rhox = -rx + C'w (kept in rxv), rhou = ru + D'w, chat = -rd; P_{t+1} chat
  const double* P1 = c.Pm + (size_t)(t + 1) * N2;
  for (int i = lane; i < NX; i += G) {
    double cw = 0.0, pch = 0.0;
    for (int k = 0; k < M; ++k) cw += C[k * NX + i] * wv[k];
    for (int j = 0; j < NX; ++j) pch += P1[i * NX + j] * chv[j];
    pc[i] = pch;
    c.pch[(size_t)t * NX + i] = pch;
    c.chat[(size_t)t * NX + i] = chv[i];
    rxv[i] = -rxv[i] + cw;
  }
  for (int i = lane; i < NU; i += G) {
    double dw = 0.0;
    for (int k = 0; k < M; ++k) dw += D[k * NU + i] * wv[k];
    const double v = ruv[i] + dw;
    c.rhou[(size_t)t * NU + i] = v;
    ruv[i] = v;
  }
  __syncwarp(ln.gm);
  // g = Acl'(P_{t+1} chat) + rhox - K' rhou
  const double *Ac = c.Acl + (size_t)t * N2, *Kt = c.K + (size_t)t * NU * NX;
  for (int i = lane; i < NX; i += G) {
    double ap = 0.0, kr = 0.0;
    for (int j = 0; j < NX; ++j) ap += Ac[j * NX + i] * pc[j];
    for (int j = 0; j < NU; ++j) kr += Kt[j * NX + i] * ruv[j];
    c.g[(size_t)t * NX + i] = ap + rxv[i] - kr;
  }
  __syncwarp(ln.gm);
}

// lane i < NX: sum_j a[j] x[j] with two accumulators (shorter dependency chain)
__device__ __forceinline__ double dot2(const double* a, const double* x) {
  double s0 = 0.0, s1 = 0.0;
#pragma unroll
  for (int j = 0; j + 1 < NX; j += 2) { s0 += a[j] * x[j]; s1 += a[j + 1] * x[j + 1]; }
  if (NX % 2) s0 += a[NX - 1] * x[NX - 1];
  return s0 + s1;
}

// A sequential affine recursion y_{k+1} = M_k y_k + v_k, k = 0..len-1, on one warp: lane i < NX computes
// row i. fetch(k, i, row, v) loads row i of M_k and v_k[i]; emit(k, i, y) stores y_{k+1}[i]. The inputs of
// the next PF steps are loaded while the current step computes (PF = 2 or 3 was slower: register pressure).
constexpr int PF = 1;
template <class Fetch, class Emit>
__device__ void chain(int len, int lane, double* ws, const double* y0, Fetch fetch, Emit emit) {
  const int i = lane < NX ? lane : 0;
  double *ya = ws, *yb = ws + NX;
  double row[PF][NX], vv[PF];
  for (int k = lane; k < NX; k += 32) ya[k] = y0 ? y0[k] : 0.0;
#pragma unroll
  for (int b = 0; b < PF; ++b) if (b < len) fetch(b, i, row[b], vv[b]);
  __syncwarp();
  for (int k0 = 0; k0 < len; k0 += PF) {
#pragma unroll
    for (int b = 0; b < PF; ++b) {                   // unrolled ring: register indices stay compile-time
      const int k = k0 + b;
      if (k < len) {
        const double v = dot2(row[b], ya) + vv[b];
        if (k + PF < len) fetch(k + PF, i, row[b], vv[b]);
        if (lane < NX) { yb[lane] = v; emit(k, lane, v); }
      }
      __syncwarp();
      double* tmp = ya; ya = yb; yb = tmp;
      if (k >= len) { tmp = ya; ya = yb; yb = tmp; }  // undo the swap of a step that did not happen
    }
  }
}

// S2: local backward recursion q_t = Acl_t' q_{t+1} + g_t in a chunk, from q = 0 at its end
__device__ void task_s2(const Ctx& c, int ch, int lane, double* ws) {
  const int c0 = ch * c.L, c1 = min(c.N, c0 + c.L);
  chain(c1 - c0, lane, ws, nullptr,
        [&](int k, int i, double* row, double& v) {
          const int t = c1 - 1 - k;
#pragma unroll
          for (int j = 0; j < NX; ++j) row[j] = c.Acl[(size_t)t * N2 + j * NX + i];
          v = c.g[(size_t)t * NX + i];
        },
        [&](int k, int i, double y) { c.ql[(size_t)(c1 - 1 - k) * NX + i] = y; });
}

// S3 and S5: the passes over the chunk boundaries. Both are affine recursions z_{j+1} = A_j z_j + b_j,
// j = 0..P-1: S3 backward, z_0 = p_N, A_j = Ps of chunk P-1-j, b_j = q at its start, z_{j+1} = p at its start;
// S5 forward, z_0 = dx_0, A_j = Fp of chunk j, b_j = its yend, z_{j+1} = dx at its end. They run as a
// Hillis-Steele scan: at level l = 0..HP-1 (HP = ceil(log2 P)), b_j += Ahat^(l)_j b_{j-2^l} for j >= 2^l, with
// Ahat^(0) = A and Ahat^(l)_j = Ahat^(l-1)_j Ahat^(l-1)_{j-2^(l-1)}. The Ahat do not depend on the right-hand
// side, so they are computed once per iteration (bnd_level, over the whole grid), and each solve's pass is HP
// levels of matrix-vector products in one block (block barriers), instead of P sequential steps on one warp.

// Ahat^(l)_j, row-major; dir 0: forward (S5), 1: backward (S3)
__device__ __forceinline__ const double* bnd_A(const Ctx& c, int dir, int l, int j) {
  if (l == 0)
    return dir ? c.Ps + (size_t)(c.P - 1 - j) * c.L * N2 : c.Fp + (size_t)(min(c.N, (j + 1) * c.L) - 1) * N2;
  return (dir ? c.Tb : c.Tf) + ((size_t)(l - 1) * c.P + j) * N2;
}

// level l >= 1 of the matrix part, both directions: Ahat^(l)_j for j = 2^l..P-1
__device__ __forceinline__ void bnd_level(const Ctx& c, int l, int gw, int TW, int lane) {
  const int s = 1 << (l - 1), n = c.P - (1 << l);
  for (int task = gw; task < 2 * n; task += TW) {
    const int dir = task & 1, j = (1 << l) + (task >> 1);
    mm<NX, NX, NX, false, false>((dir ? c.Tb : c.Tf) + ((size_t)(l - 1) * c.P + j) * N2, NX, bnd_A(c, dir, l - 1, j),
                                 NX, bnd_A(c, dir, l - 1, j - s), NX, lane);
  }
}

// one pass, by the whole block (GP lanes per boundary); buf: 2 P NX doubles of shared memory.
// bv(j, i) gives b_j[i]; emit(j, i, v) stores z_{j+1}[i] = v.
template <class Bv, class Emit>
__device__ __forceinline__ void bnd_pass(const Ctx& c, int dir, const double* z0, double* buf, Bv bv, Emit emit) {
  const int ng = blockDim.x / GP, gi = threadIdx.x / GP, i = threadIdx.x % GP;
  double *src = buf, *dst = buf + (size_t)c.P * NX;
  for (int j = gi; j < c.P; j += ng)
    if (i < NX) {
      double v = bv(j, i);
      if (j == 0) v += dot2(bnd_A(c, dir, 0, 0) + i * NX, z0);              // the initial value enters b_0
      src[j * NX + i] = v;
    }
  __syncthreads();
  for (int l = 0, s = 1; s < c.P; ++l, s <<= 1) {
    for (int j = gi; j < c.P; j += ng)
      if (i < NX) {
        double v = src[j * NX + i];
        if (j >= s) v += dot2(bnd_A(c, dir, l, j) + i * NX, src + (j - s) * NX);
        dst[j * NX + i] = v;
      }
    __syncthreads();
    double* tmp = src; src = dst; dst = tmp;
  }
  for (int j = gi; j < c.P; j += ng)
    if (i < NX) emit(j, i, src[j * NX + i]);
}

// S3: costates at the chunk starts, p_c0 = Ps_c0 p_c1 + q_c0, from p_N = rhoxN
__device__ __forceinline__ void task_s3(const Ctx& c, double* buf) {
  if (threadIdx.x < NX) c.p[(size_t)c.N * NX + threadIdx.x] = c.rhoxN[threadIdx.x];
  bnd_pass(c, 1, c.rhoxN, buf,
           [&](int j, int i) __attribute__((always_inline)) { return c.ql[(size_t)(c.P - 1 - j) * c.L * NX + i]; },
           [&](int j, int i, double v) __attribute__((always_inline)) { c.p[(size_t)(c.P - 1 - j) * c.L * NX + i] = v; });
}

// S4a (per stage, parallel): the costate p_t inside its chunk, p_t = Ps_t p_c1 + q_t, then the gain
// k_t = Sigma_t^-1 (rhou_t + B_t'(P_{t+1} chat_t + p_{t+1})) and f_t = chat_t - B_t k_t
template <int G>
__device__ __forceinline__ void task_s4a(const Ctx& c, int t, Lanes<G> ln, double* ws) {
  const int lane = ln.sl;                                            // the lane within the group
  const int c0 = (t / c.L) * c.L, c1 = min(c.N, c0 + c.L);
  double *pe = ws, *tmp = pe + NX, *bt = tmp + NX, *di = bt + NU;
  for (int i = lane; i < NX; i += G) pe[i] = c.p[(size_t)c1 * NX + i];
  for (int i = lane; i < NU; i += G) di[i] = c.Ldi[(size_t)t * NU + i];
  __syncwarp(ln.gm);
  for (int i = lane; i < NX; i += G) {
    double p1 = pe[i];                                                 // p_{t+1}
    if (t + 1 < c1) {
      const double* Pc = c.Ps + (size_t)(t + 1) * N2;
      double v = 0.0;
      for (int j = 0; j < NX; ++j) v += Pc[i * NX + j] * pe[j];
      p1 = v + c.ql[(size_t)(t + 1) * NX + i];
    }
    if (t > c0) {                                                      // p_t (chunk starts come from S3)
      const double* Pc = c.Ps + (size_t)t * N2;
      double v = 0.0;
      for (int j = 0; j < NX; ++j) v += Pc[i * NX + j] * pe[j];
      c.p[(size_t)t * NX + i] = v + c.ql[(size_t)t * NX + i];
    }
    tmp[i] = c.pch[(size_t)t * NX + i] + p1;                          // P_{t+1} chat + p_{t+1}
  }
  __syncwarp(ln.gm);
  const double *B = c.B + (size_t)t * NX * NU, *Lg = c.Ls + (size_t)t * NU * NU;
  for (int i = lane; i < NU; i += G) {
    double v = 0.0;
    for (int j = 0; j < NX; ++j) v += B[j * NU + i] * tmp[j];
    bt[i] = c.rhou[(size_t)t * NU + i] + v;
  }
  __syncwarp(ln.gm);
  trsm_l<NU, G>(bt, 1, Lg, di, bt, 1, 1, lane, ln.gm);
  trsm_lt<NU, G>(bt, 1, Lg, di, bt, 1, 1, lane, ln.gm);                          // k = Sigma^-1 (...)
  for (int i = lane; i < NU; i += G) c.kv[(size_t)t * NU + i] = bt[i];
  for (int i = lane; i < NX; i += G) {
    double bk = 0.0;
    for (int j = 0; j < NU; ++j) bk += B[i * NU + j] * bt[j];
    c.fv[(size_t)t * NX + i] = c.chat[(size_t)t * NX + i] - bk;
  }
  __syncwarp(ln.gm);
}

// S4b: local forward recursion y_{t+1} = Acl_t y_t + f_t in a chunk, from y = 0 at its start
__device__ void task_s4b(const Ctx& c, int ch, int lane, double* ws) {
  const int c0 = ch * c.L, c1 = min(c.N, c0 + c.L);
  chain(c1 - c0, lane, ws, nullptr,
        [&](int k, int i, double* row, double& v) {
          const int t = c0 + k;
#pragma unroll
          for (int j = 0; j < NX; ++j) row[j] = c.Acl[(size_t)t * N2 + i * NX + j];
          v = c.fv[(size_t)t * NX + i];
        },
        [&](int k, int i, double y) {
          const int t = c0 + k;
          if (t + 1 < c1) c.yl[(size_t)(t + 1) * NX + i] = y; else c.yend[(size_t)ch * NX + i] = y;
        });
}

// S5: states at the chunk ends, x_c1 = Fp_{c1-1} x_c0 + yend, from x_0 = dx0
__device__ __forceinline__ void task_s5(const Ctx& c, double* buf) {
  if (threadIdx.x < NX) c.dxb[threadIdx.x] = c.dx0[threadIdx.x];
  bnd_pass(c, 0, c.dx0, buf, [&](int j, int i) __attribute__((always_inline)) { return c.yend[(size_t)j * NX + i]; },
           [&](int j, int i, double v) __attribute__((always_inline)) {
             c.dxb[(size_t)min(c.N, (j + 1) * c.L) * NX + i] = v;
           });
}

// S6: dx_t, du_t, dnu_t, dlam_t, ds_t; writes the result of this solve and the step-length partials
template <int G>
__device__ __forceinline__ void task_s6(const Ctx& c, int t, Lanes<G> ln, double* ws, const SolveArgs& a, double& rs,
                                        double& rl) {
  const int lane = ln.sl;                                            // the lane within the group
  const int M = c.M;
  double *dxv = ws, *duv = dxv + NX;
  // output: PRED and GOND into DIR, CORR and REFN into STEP
  const bool toS = a.kind == CORR || a.kind == REFN;
  double *ox = toS ? c.Sx : c.Dx, *ou = toS ? c.Su : c.Du, *on = toS ? c.Snu : c.Dnu;
  double *ol = toS ? c.Slam : c.Dlam, *os = toS ? c.Ss : c.Ds;
  if (t == c.N) {
    const double* PN = c.Pm + (size_t)c.N * N2;
    for (int i = lane; i < NX; i += G) dxv[i] = c.dxb[(size_t)c.N * NX + i];
    __syncwarp(ln.gm);
    for (int i = lane; i < NX; i += G) {
      double v = 0.0;
      for (int j = 0; j < NX; ++j) v += PN[i * NX + j] * dxv[j];
      double dn = v + c.p[(size_t)c.N * NX + i], dx = dxv[i];
      const size_t ix = (size_t)c.N * NX + i;
      if (a.kind == REFN) { dx = stp(c.Sx, c.Dx, ix, a.keep) + dx; dn = stp(c.Snu, c.Dnu, ix, a.keep) + dn; }
      ox[ix] = dx;
      on[ix] = dn;
    }
    __syncwarp(ln.gm);
    return;
  }
  const int ch = t / c.L, c0 = ch * c.L;
  if (t == c0) {
    for (int i = lane; i < NX; i += G) dxv[i] = c.dxb[(size_t)c0 * NX + i];
  } else {
    const double* F = c.Fp + (size_t)(t - 1) * N2;
    for (int i = lane; i < NX; i += G) {
      double v = 0.0;
      for (int j = 0; j < NX; ++j) v += F[i * NX + j] * c.dxb[(size_t)c0 * NX + j];
      dxv[i] = v + c.yl[(size_t)t * NX + i];
    }
  }
  __syncwarp(ln.gm);
  const double *Kt = c.K + (size_t)t * NU * NX, *Pt = c.Pm + (size_t)t * N2;
  for (int i = lane; i < NU; i += G) {
    double v = 0.0;
    for (int j = 0; j < NX; ++j) v += Kt[i * NX + j] * dxv[j];
    duv[i] = -v - c.kv[(size_t)t * NU + i];
  }
  __syncwarp(ln.gm);
  for (int i = lane; i < NX; i += G) {
    double v = 0.0;
    for (int j = 0; j < NX; ++j) v += Pt[i * NX + j] * dxv[j];
    double dn = v + c.p[(size_t)t * NX + i], dx = dxv[i];
    const size_t ix = (size_t)t * NX + i;
    if (a.kind == REFN) { dx = stp(c.Sx, c.Dx, ix, a.keep) + dx; dn = stp(c.Snu, c.Dnu, ix, a.keep) + dn; }
    if (a.kind != PRED) { ox[ix] = dx; on[ix] = dn; }
  }
  for (int i = lane; i < NU; i += G) {
    const size_t iu = (size_t)t * NU + i;
    double du = duv[i];
    if (a.kind == REFN) du = stp(c.Su, c.Du, iu, a.keep) + du;
    if (a.kind != PRED) ou[iu] = du;
  }
  const double *C = c.C + (size_t)t * M * NX, *D = c.D + (size_t)t * M * NU;
  const double *lt = c.lam + (size_t)t * M, *st = c.s + (size_t)t * M;
  for (int k = lane; k < M; k += G) {
    const size_t i = (size_t)t * M + k;
    double cx = 0.0, dd = 0.0;
    for (int j = 0; j < NX; ++j) cx += C[k * NX + j] * dxv[j];
    for (int j = 0; j < NU; ++j) dd += D[k * NU + j] * duv[j];
    const double v = cx + dd;
    double dl = c.W[i] * v + c.w[i], ds = -c.rcs[i] - v;
    if (a.kind == PRED) c.corr[i] = dl * ds;
    if (a.kind == REFN) { dl = stp(c.Slam, c.Dlam, i, a.keep) + dl; ds = stp(c.Ss, c.Ds, i, a.keep) + ds; }
    ol[i] = dl;
    os[i] = ds;
    double tl = dl, ts = ds;                                           // the step whose lengths are needed
    if (a.kind == GOND) { tl = c.Slam[i] + dl; ts = c.Ss[i] + ds; }
    if (ts < 0.0) rs = nmin(rs, -st[k] / ts);
    if (tl < 0.0) rl = nmin(rl, -lt[k] / tl);
  }
  __syncwarp(ln.gm);
}
