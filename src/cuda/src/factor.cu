// The matrix part of the Newton step (factor.cuh).
#include "factor.cuh"
#include "grid.cuh"
#include "warp_linalg.cuh"

template <int G>
__device__ __forceinline__ void task_build(const Ctx& c, int t, Lanes<G> ln, double* ws) {
  const int lane = ln.sl;                                            // the lane within the group
  double* T = c.T + (size_t)t * 3 * N2;
  if (t == c.N) {                                                   // terminal element (0, 0, QN)
    for (int e = lane; e < N2; e += G) { T[e] = 0.0; T[N2 + e] = 0.0; T[2 * N2 + e] = c.QN[e]; }
    __syncwarp(ln.gm);
    return;
  }
  const int M = c.M;
  const double *A = c.A + (size_t)t * N2, *B = c.B + (size_t)t * NX * NU;
  const double *C = c.C + (size_t)t * M * NX, *D = c.D + (size_t)t * M * NU;
  double *Qt = ws, *St = Qt + N2, *Rt = St + NU * NX, *Lr = Rt + NU * NU, *V = Lr + NU * NU, *U = V + NU * NX;
  double *X = U + NU * NX, *Bt = X + NU * NX, *di = Bt + NU * NX, *Wv = di + NU;
  for (int k = lane; k < M; k += G) {
    const double wk = c.lam[(size_t)t * M + k] / c.s[(size_t)t * M + k];
    Wv[k] = wk;
    c.W[(size_t)t * M + k] = wk;
  }
  for (int e = lane; e < NU * NX; e += G) Bt[e] = B[(e % NX) * NU + e / NX];
  __syncwarp(ln.gm);
  for (int e = lane; e < N2; e += G) {
    const int a = e / NX, b = e % NX;
    double acc = 0.0;
    for (int k = 0; k < M; ++k) acc += C[k * NX + a] * (Wv[k] * C[k * NX + b]);
    const double v = c.Q[(size_t)t * N2 + e] + acc;
    Qt[e] = v;
    c.Qt[(size_t)t * N2 + e] = v;
  }
  for (int e = lane; e < NU * NX; e += G) {
    const int a = e / NX, b = e % NX;
    double acc = 0.0;
    for (int k = 0; k < M; ++k) acc += D[k * NU + a] * (Wv[k] * C[k * NX + b]);
    const double v = c.S[(size_t)t * NU * NX + e] + acc;
    St[e] = v;
    c.St[(size_t)t * NU * NX + e] = v;
  }
  for (int e = lane; e < NU * NU; e += G) {
    const int a = e / NU, b = e % NU;
    double acc = 0.0;
    for (int k = 0; k < M; ++k) acc += D[k * NU + a] * (Wv[k] * D[k * NU + b]);
    const double v = c.R[(size_t)t * NU * NU + e] + acc;
    Rt[e] = v;
    c.Rt[(size_t)t * NU * NU + e] = v;
  }
  __syncwarp(ln.gm);
  chol<NU, G>(Lr, di, Rt, NU, Rt, lane, ln.gm);                              // in place: Rt is scratch here
  trsm_l<NU, G>(V, NX, Lr, di, St, NX, NX, lane, ln.gm);                      // V = Lr^-1 St
  trsm_l<NU, G>(U, NX, Lr, di, Bt, NX, NX, lane, ln.gm);                      // U = Lr^-1 B'
  trsm_lt<NU, G>(X, NX, Lr, di, V, NX, NX, lane, ln.gm);                      // Lr^-T V
  for (int e = lane; e < N2; e += G) {
    const int i = e / NX, j = e % NX;
    double bx = 0.0, uu = 0.0, vv = 0.0;
    for (int k = 0; k < NU; ++k) {
      bx += B[i * NU + k] * X[k * NX + j];
      uu += U[k * NX + i] * U[k * NX + j];
      vv += V[k * NX + i] * V[k * NX + j];
    }
    T[e] = A[e] - bx;                                               // Abar
    T[N2 + e] = uu;                                                 // M = U'U
    T[2 * N2 + e] = 0.5 * ((Qt[e] - vv) + (Qt[j * NX + i] - vv));   // Qbar = sym(Qt - V'V)
  }
  __syncwarp(ln.gm);
}

// G = (I + J2 C1)^-1 J2: Cholesky form W'W or LU; for the general combine also
// Y = (I + C1 J2)^-1 [C1, A1] (n x 2n). Operands in shared memory.
static __device__ void combine_core(const double* A1, const double* C1, const double* J2, double* G, double* Y,
                                    double* ws, bool lu, int lane) {
  double *Lm = ws, *Rm = Lm + N2, *Wm = Rm + N2, *T1 = Wm + N2, *T2 = T1 + N2, *dl = T2 + N2, *dr = dl + NX;
  if (!lu) {
    chol<NX>(Lm, dl, J2, NX, T2, lane);
    mm<NX, NX, NX, true, false>(T1, NX, Lm, NX, C1, NX, lane);                // L' C1
    mm<NX, NX, NX, false, false>(T2, NX, T1, NX, Lm, NX, lane);               // L' C1 L
    for (int e = lane; e < N2; e += 32) {
      const int i = e / NX, j = e % NX;
      const double id = i == j ? 1.0 : 0.0;
      T1[e] = 0.5 * ((id + T2[e]) + (id + T2[j * NX + i]));                   // sym(I + L' C1 L)
    }
    __syncwarp();
    chol<NX>(Rm, dr, T1, NX, T1, lane);
    for (int e = lane; e < N2; e += 32) Wm[e] = Lm[(e % NX) * NX + e / NX];   // L'
    __syncwarp();
    trsm_l<NX>(Wm, NX, Rm, dr, Wm, NX, NX, lane);                             // W = R^-1 L'
    mm<NX, NX, NX, true, false>(G, NX, Wm, NX, Wm, NX, lane);                 // G = W'W
    if (Y) {
      for (int e = lane; e < 2 * N2; e += 32) {
        const int i = e / (2 * NX), j = e % (2 * NX);
        const double* Bm = j < NX ? C1 : A1;
        const int jj = j < NX ? j : j - NX;
        double acc = 0.0;
        for (int k = 0; k < NX; ++k) acc += Wm[i * NX + k] * Bm[k * NX + jj];
        Y[e] = acc;                                                           // R^-1 L' [C1, A1] = W [C1, A1]
      }
      __syncwarp();
      trsm_lt<NX>(Y, 2 * NX, Rm, dr, Y, 2 * NX, 2 * NX, lane);
      trsm_lt<NX>(Y, 2 * NX, Lm, dl, Y, 2 * NX, 2 * NX, lane);                // L^-T R^-T R^-1 L' [C1, A1]
    }
  } else {
    // G = solve(I + J2 C1, J2)
    mm<NX, NX, NX, false, false>(T1, NX, J2, NX, C1, NX, lane);
    for (int e = lane; e < N2; e += 32) { T1[e] += (e / NX == e % NX) ? 1.0 : 0.0; G[e] = J2[e]; }
    __syncwarp();
    lu_solve(T1, G, NX, NX, lane);
    if (Y) {
      mm<NX, NX, NX, false, false>(T1, NX, C1, NX, J2, NX, lane);
      for (int e = lane; e < N2; e += 32) T1[e] += (e / NX == e % NX) ? 1.0 : 0.0;
      for (int e = lane; e < 2 * N2; e += 32) {
        const int i = e / (2 * NX), j = e % (2 * NX);
        Y[e] = j < NX ? C1[i * NX + j] : A1[i * NX + j - NX];
      }
      __syncwarp();
      lu_solve(T1, Y, 2 * NX, 2 * NX, lane);
    }
  }
}

// up-sweep node: out = a1 (x) a2 (the value-function combine); a1 earlier in time. The operands are staged in
// shared memory; out may be a1's slot.
static __device__ void task_combine(const double* a1, const double* a2, double* out, int lane, double* ws,
                                    bool lu) {
  double *A1 = ws, *C1 = A1 + N2, *J1 = C1 + N2, *A2 = J1 + N2, *C2 = A2 + N2, *J2 = C2 + N2;
  double *G = J2 + N2, *Y = G + N2, *T1 = Y + 2 * N2, *T2 = T1 + N2, *A12 = T2 + N2, *wk = A12 + N2;
  for (int e = lane; e < 3 * N2; e += 32) { A1[e] = a1[e]; A2[e] = a2[e]; }   // one coalesced load
  __syncwarp();
  combine_core(A1, C1, J2, G, Y, wk, lu, lane);
  mm<NX, NX, NX, false, false>(T1, NX, A2, NX, Y, 2 * NX, lane);              // A2 H
  mm<NX, NX, NX, false, true>(T2, NX, T1, NX, A2, NX, lane);                  // A2 H A2'
  mm<NX, NX, NX, false, false>(A12, NX, A2, NX, Y + NX, 2 * NX, lane);        // A12 = A2 (I + C1 J2)^-1 A1
  double *C12 = wk, *J12 = wk + N2;                                           // combine_core's workspace is free
  for (int e = lane; e < N2; e += 32) {
    const int i = e / NX, j = e % NX;
    C12[e] = 0.5 * ((T2[e] + C2[e]) + (T2[j * NX + i] + C2[j * NX + i]));
  }
  mm<NX, NX, NX, true, false>(T1, NX, A1, NX, G, NX, lane);                   // A1' G
  mm<NX, NX, NX, false, false>(T2, NX, T1, NX, A1, NX, lane);                 // A1' G A1
  for (int e = lane; e < N2; e += 32) {
    const int i = e / NX, j = e % NX;
    J12[e] = 0.5 * ((T2[e] + J1[e]) + (T2[j * NX + i] + J1[j * NX + i]));
  }
  __syncwarp();
  for (int e = lane; e < N2; e += 32) { out[e] = A12[e]; out[N2 + e] = C12[e]; out[2 * N2 + e] = J12[e]; }
  __syncwarp();
}

// down-sweep node: P_j = J of a1 (x) (0, 0, P_{j+s}), the Riccati map in information form
static __device__ void task_riccati(const double* a1, const double* J2, double* Pout, int lane, double* ws, bool lu,
                                    double& bad) {
  double *A1 = ws, *C1 = A1 + N2, *J1 = C1 + N2, *J2s = J1 + N2;
  double *G = J2s + N2, *T1 = G + N2, *T2 = T1 + N2, *wk = T2 + N2;
  for (int e = lane; e < 3 * N2; e += 32) A1[e] = a1[e];
  for (int e = lane; e < N2; e += 32) J2s[e] = J2[e];
  __syncwarp();
  combine_core(A1, C1, J2s, G, nullptr, wk, lu, lane);
  mm<NX, NX, NX, true, false>(T1, NX, A1, NX, G, NX, lane);
  mm<NX, NX, NX, false, false>(T2, NX, T1, NX, A1, NX, lane);
  for (int e = lane; e < N2; e += 32) {
    const int i = e / NX, j = e % NX;
    const double v = 0.5 * ((T2[e] + J1[e]) + (T2[j * NX + i] + J1[j * NX + i]));
    Pout[e] = v;
    if (!isfinite(v)) bad = 1.0;
  }
  __syncwarp();
}

// The classical Riccati step at stage t from P = P_{t+1} (shared memory):
//   Sigma = Rt + B'PB = Lg Lg',  Y = Lg^-1 (B'PA + St),  K = Lg^-T Y,  Acl = A - BK,
// and, if Pn, Pn = sym(Qt + A'PA - Y'Y) = R(P). The leaves of the tree are stage elements, whose M = B Rt^-1 B'
// has rank m, and their combines reduce to this form (Woodbury): one m x m Cholesky instead
// of two n x n ones, with the arithmetic of one step of the sweep. A, B: stage data staged in shared memory;
// wk: NU NX + NU^2 + 2 N2 scratch.
static __device__ __forceinline__ void riccati_step(const Ctx& c, int t, const double* P, double* Pn, double* A,
                                                    double* B, double* Lg, double* di, double* Y, double* K,
                                                    double* Acl, double* wk, int lane) {
  const double *Ag = c.A + (size_t)t * N2, *Bg = c.B + (size_t)t * NX * NU;
  for (int e = lane; e < N2; e += 32) A[e] = Ag[e];
  for (int e = lane; e < NX * NU; e += 32) B[e] = Bg[e];
  __syncwarp();
  double *BtP = wk, *Sg = BtP + NU * NX, *T1 = Sg + NU * NU, *T2 = T1 + N2;
  mm<NU, NX, NX, true, false>(BtP, NX, B, NU, P, NX, lane);                                         // B'P
  mm<NU, NX, NU, false, false>(Sg, NU, BtP, NX, B, NU, lane, c.Rt + (size_t)t * NU * NU, NU, 1.0); // Sigma
  chol<NU>(Lg, di, Sg, NU, Sg, lane);
  mm<NU, NX, NX, false, false>(Y, NX, BtP, NX, A, NX, lane, c.St + (size_t)t * NU * NX, NX, 1.0);  // B'PA + St
  trsm_l<NU>(Y, NX, Lg, di, Y, NX, NX, lane);
  trsm_lt<NU>(K, NX, Lg, di, Y, NX, NX, lane);
  mm<NX, NU, NX, false, false>(Acl, NX, B, NU, K, NX, lane, A, NX, -1.0);
  if (Pn) {
    mm<NX, NX, NX, false, false>(T1, NX, P, NX, A, NX, lane);                                         // PA
    mm<NX, NX, NX, true, false>(T2, NX, A, NX, T1, NX, lane);                                          // A'PA
    const double* Qt = c.Qt + (size_t)t * N2;
    for (int e = lane; e < N2; e += 32) {
      const int i = e / NX, j = e % NX;
      double yy = 0.0;
      for (int k = 0; k < NU; ++k) yy += Y[k * NX + i] * Y[k * NX + j];
      T1[e] = (Qt[e] + T2[e]) - yy;
    }
    __syncwarp();
    for (int e = lane; e < N2; e += 32) Pn[e] = 0.5 * (T1[e] + T1[(e % NX) * NX + e / NX]);
    __syncwarp();
  }
}

static __device__ __forceinline__ void store_gains(const Ctx& c, int t, const double* Lg, const double* di,
                                                   const double* K, const double* Acl, int lane) {
  for (int e = lane; e < NU * NU; e += 32) c.Ls[(size_t)t * NU * NU + e] = Lg[e];
  for (int i = lane; i < NU; i += 32) c.Ldi[(size_t)t * NU + i] = di[i];
  for (int e = lane; e < NU * NX; e += 32) c.K[(size_t)t * NU * NX + e] = K[e];
  for (int e = lane; e < N2; e += 32) c.Acl[(size_t)t * N2 + e] = Acl[e];
}

// up-sweep leaf: T_i = (stage i) (x) T_{i+1}. With P = J2 and the step's K, Lg (the combine by Woodbury):
//   A12 = A2 (A - BK),   C12 = A2 B Sigma^-1 B'A2' + C2 = Z'Z + C2 (Z = Lg^-1 B'A2'),   J12 = R(J2)
static __device__ __forceinline__ void task_leaf_up(const Ctx& c, int i, int lane, double* ws,
                                                    double* out) {
  double *A = ws, *B = A + N2, *Lg = B + NX * NU, *di = Lg + NU * NU, *Y = di + NU, *K = Y + NU * NX;
  double *Acl = K + NU * NX, *A2 = Acl + N2, *C2 = A2 + N2, *P = C2 + N2, *Jn = P + N2, *Z = Jn + N2, *wk = Z + NU * NX;
  const double* a2 = c.T + (size_t)(i + 1) * 3 * N2;
  for (int e = lane; e < 3 * N2; e += 32) A2[e] = a2[e];                       // A2, C2, J2 (= P)
  riccati_step(c, i, P, Jn, A, B, Lg, di, Y, K, Acl, wk, lane);                // its first barrier covers the load
  mm<NX, NX, NX, false, false>(out, NX, A2, NX, Acl, NX, lane);                // A12
  mm<NU, NX, NX, true, true>(Z, NX, B, NU, A2, NX, lane);                      // B'A2'
  trsm_l<NU>(Z, NX, Lg, di, Z, NX, NX, lane);
  for (int e = lane; e < N2; e += 32) {
    const int r = e / NX, q = e % NX;
    double zz = 0.0;
    for (int k = 0; k < NU; ++k) zz += Z[k * NX + r] * Z[k * NX + q];
    out[N2 + e] = zz + C2[e];                                                  // C12 (symmetric bit for bit)
    out[2 * N2 + e] = Jn[e];                                                   // J12
  }
  __syncwarp();
}

// down-sweep leaf: P_j = R(P_{j+1}) at stage j (P_N = QN), then the gains of stages j and j - 1, which need
// P_{j+1} and P_j: every gain is computed here, so the factorization needs no separate gains phase
static __device__ __forceinline__ void task_leaf_down(const Ctx& c, int j, int lane, double* ws,
                                                      double& bad) {
  double *A = ws, *B = A + N2, *Lg = B + NX * NU, *di = Lg + NU * NU, *Y = di + NU, *K = Y + NU * NX;
  double *Acl = K + NU * NX, *P = Acl + N2, *Pj = P + N2, *wk = Pj + N2;
  if (j < c.N) {
    for (int e = lane; e < N2; e += 32) P[e] = c.Pm[(size_t)(j + 1) * N2 + e];
    riccati_step(c, j, P, Pj, A, B, Lg, di, Y, K, Acl, wk, lane);
    store_gains(c, j, Lg, di, K, Acl, lane);
  } else {
    for (int e = lane; e < N2; e += 32) Pj[e] = c.T[(size_t)j * 3 * N2 + 2 * N2 + e];
  }
  for (int e = lane; e < N2; e += 32) {
    const double v = Pj[e];
    c.Pm[(size_t)j * N2 + e] = v;
    if (!isfinite(v)) bad = 1.0;
  }
  riccati_step(c, j - 1, Pj, nullptr, A, B, Lg, di, Y, K, Acl, wk, lane);
  store_gains(c, j - 1, Lg, di, K, Acl, lane);
  __syncwarp();
}

// gains of stage t from P_{t+1}
static __device__ __forceinline__ void task_gains(const Ctx& c, int t, int lane, double* ws) {
  double *A = ws, *B = A + N2, *Lg = B + NX * NU, *di = Lg + NU * NU, *Y = di + NU, *K = Y + NU * NX;
  double *Acl = K + NU * NX, *P = Acl + N2, *wk = P + N2;
  for (int e = lane; e < N2; e += 32) P[e] = c.Pm[(size_t)(t + 1) * N2 + e];
  riccati_step(c, t, P, nullptr, A, B, Lg, di, Y, K, Acl, wk, lane);
  store_gains(c, t, Lg, di, K, Acl, lane);
  __syncwarp();
}

// P_t = J of the element in `src`, with the finiteness check
static __device__ __forceinline__ void copy_J(const double* src, double* Pout, int lane, double& bad) {
  for (int e = lane; e < N2; e += 32) {
    const double v = src[2 * N2 + e];
    Pout[e] = v;
    if (!isfinite(v)) bad = 1.0;
  }
  __syncwarp();
}

// The factorization scan, as a schedule of levels with one barrier each. Every task type has one call site, so
// it is inlined once (a second copy of the combines changed the register allocation and slowed the whole
// kernel by 2%). Up-sweep, Kogge-Stone, down-sweep:
//   - r = c.ksr up-sweep levels l = 0..r-1 (Blelloch): task k combines elements i = 2^(l+1) k and i + 2^l into i,
//     the leaf level in Riccati form. Then T_{t 2^r} holds the product over [t 2^r, (t+1) 2^r): E' = ceil(E/2^r)
//     super elements, the last of which contains the terminal element and so covers the rest of the horizon.
//   - Kogge-Stone over the super elements, floor(log2(E'-1)) + 1 levels: at level l every unfinished element t
//     combines with t + 2^l. When that partner already covers the rest of the horizon, its (A, C) is zero, the
//     combine is the Riccati-type one, and it finishes t (P_{t 2^r}). Its elements alternate between T (at
//     stride 2^r, entries the down-sweep does not read) and T2. With r = 0, level 0 combines stage t with
//     stage t+1 in Riccati form, and a last level computes the gains.
//   - r down-sweep levels l = r-1..0: P_j = J of T_j (x) (0, 0, P_{j+2^l}) for j = 2^l (2k + 1); the leaf level
//     also gives the gains.
// The host picks the smallest r with E' <= PM_KS_PER_SM x multiprocessors: Kogge-Stone does E' log E' combines
// instead of 2 E', and it pays while a level's combines fit on the GPU without sharing its FP64 units much (on
// the RTX 3070 Laptop, 40 SMs, pure Kogge-Stone took 17-19% fewer cycles per IPM iteration for N <= 63, 8% at
// N = 191, none at 255; PM_KS_PER_SM = 2 or 3 was best for the hybrid, 5 1-3% slower, 8 6-15% slower). Long
// horizons keep their lowest levels, where the work is, in the Blelloch scan.
enum TaskKind { LEAF_UP, COMBINE, RICCATI, LEAF_DOWN, COPY, GAINS };
__device__ __forceinline__ void factor(const Ctx& c, cg::grid_group& grid, int gw, int TW, int lane, double* ws,
                                       bool lu, double& bad) {
  const int E = c.E, r = c.ksr, Ns = ((E + (1 << r) - 1) >> r) - 1;   // super elements 0..Ns
  int KL = 1;                                                  // Kogge-Stone levels 0..KL-1
  while ((1 << KL) <= Ns) ++KL;
  const int nlev = r + KL + (r ? r : 1);
  for (int lev = 0; lev < nlev; ++lev) {
#ifdef PM_PROFILE
    const long long tl = clock64();
#endif
    const int phase = lev < r ? 0 : (lev < r + KL ? 1 : 2);    // up-sweep, Kogge-Stone, down-sweep / gains
    const int l = phase == 0 ? lev : (phase == 1 ? lev - r : r + KL + r - 1 - lev), s = 1 << l;
    int ntask;
    if (phase == 1) {
      ntask = Ns - s + 1;
      if (l == 0 && gw == 0)                                   // the last super element is finished
        copy_J(c.T + ((size_t)Ns << r) * 3 * N2, c.Pm + ((size_t)Ns << r) * N2, lane, bad);
    } else if (phase == 2 && r == 0) {
      ntask = c.N;                                             // gains
    } else {
      ntask = (E - s + 2 * s - 1) / (2 * s);
    }
    double *ka = c.T, *kb = c.T2;                              // Kogge-Stone buffers: level l reads ka if l is even
    const size_t sa = (size_t)3 * N2 << r, sb = (size_t)3 * N2;
    for (int k = gw; k < ntask; k += TW) {
      int kind, a = 0;                                         // a: the task's element (stage index)
      const double *x1 = nullptr, *x2 = nullptr;
      double* out = nullptr;
      if (phase == 0) {
        a = 2 * s * k;
        if (l == 0) { kind = LEAF_UP; out = c.T + (size_t)a * 3 * N2; }   // stage a is not read again
        else {
          kind = COMBINE; x1 = c.T + (size_t)a * 3 * N2; x2 = c.T + (size_t)(a + s) * 3 * N2;
          out = c.T + (size_t)a * 3 * N2;
        }
      } else if (phase == 1) {
        const int t = k;
        const double* src = (l & 1) ? kb : ka;
        double* dst = (l & 1) ? ka : kb;
        const size_t ss = (l & 1) ? sb : sa, sd = (l & 1) ? sa : sb;
        a = t << r;
        if (l == 0 && r == 0) { kind = LEAF_UP; out = kb + (size_t)t * sb; }
        else if (t + 2 * s > Ns) {                             // partner finished: (0, 0, P)
          kind = RICCATI; x1 = src + t * ss; x2 = c.Pm + ((size_t)(t + s) << r) * N2; out = c.Pm + (size_t)a * N2;
        } else {
          kind = COMBINE; x1 = src + t * ss; x2 = src + (t + s) * ss; out = dst + t * sd;
        }
      } else if (r == 0) {
        a = k; kind = GAINS;
      } else {
        a = s + 2 * s * k;
        if (l == 0) { kind = LEAF_DOWN; }
        else if (a + s < E) {
          kind = RICCATI; x1 = c.T + (size_t)a * 3 * N2; x2 = c.Pm + (size_t)(a + s) * N2; out = c.Pm + (size_t)a * N2;
        } else {
          kind = COPY; x1 = c.T + (size_t)a * 3 * N2; out = c.Pm + (size_t)a * N2;
        }
      }
      switch (kind) {
        case LEAF_UP:
          task_leaf_up(c, a, lane, ws, out);
          if (phase == 1 && a == c.N - 1) copy_J(out, c.Pm + (size_t)a * N2, lane, bad);   // met the terminal
          break;
        case COMBINE: task_combine(x1, x2, out, lane, ws, lu); break;
        case RICCATI: task_riccati(x1, x2, out, lane, ws, lu, bad); break;
        case LEAF_DOWN: task_leaf_down(c, a, lane, ws, bad); break;
        case COPY: copy_J(x1, out, lane, bad); break;
        default: task_gains(c, a, lane, ws); break;
      }
    }
    bar(grid);
#ifdef PM_PROFILE
    if (gw == 0 && lane == 0 && lev < 32) c.prof[32 + lev] += (double)(clock64() - tl);
#endif
  }
}

__device__ void task_chunk(const Ctx& c, int task, int lane) {
  const int ch = task >> 1, c0 = ch * c.L, c1 = min(c.N, c0 + c.L);
  if ((task & 1) == 0) {
    for (int e = lane; e < N2; e += 32) c.Fp[(size_t)c0 * N2 + e] = c.Acl[(size_t)c0 * N2 + e];
    __syncwarp();
    for (int t = c0 + 1; t < c1; ++t)
      mm<NX, NX, NX, false, false>(c.Fp + (size_t)t * N2, NX, c.Acl + (size_t)t * N2, NX, c.Fp + (size_t)(t - 1) * N2,
                                   NX, lane);
  } else {
    const double* Al = c.Acl + (size_t)(c1 - 1) * N2;
    for (int e = lane; e < N2; e += 32) c.Ps[(size_t)(c1 - 1) * N2 + e] = Al[(e % NX) * NX + e / NX];
    __syncwarp();
    for (int t = c1 - 2; t >= c0; --t)
      mm<NX, NX, NX, true, false>(c.Ps + (size_t)t * N2, NX, c.Acl + (size_t)t * N2, NX, c.Ps + (size_t)(t + 1) * N2,
                                  NX, lane);
  }
}
