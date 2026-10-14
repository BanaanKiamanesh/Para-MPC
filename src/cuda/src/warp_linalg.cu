// Warp-level dense linear algebra on small matrices in shared memory (warp_linalg.cuh).
#include "warp_linalg.cuh"

__device__ __forceinline__ double nmin(double a, double b) { return (a < b || a != a) ? a : b; }
__device__ __forceinline__ double nmax(double a, double b) { return (a > b || a != a) ? a : b; }
__device__ __forceinline__ double clip01(double v) { return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v); }

__device__ __forceinline__ double combine_op(double a, double b, int op) {
  return op == SUM ? a + b : (op == MIN ? nmin(a, b) : nmax(a, b));
}

__device__ double warp_reduce(double v, int op) {
  for (int o = 16; o > 0; o >>= 1) v = combine_op(v, __shfl_xor_sync(FULL, v, o), op);
  return __shfl_sync(FULL, v, 0);
}

template <int R, int K, int C, bool TA, bool TB, typename F>
__device__ void mm(F* out, int ldo, const F* A, int lda, const F* B, int ldb, int lane, const F* add, int ldadd,
                   typename same_t<F>::type sign) {
  for (int e = lane; e < R * C; e += 32) {
    const int i = e / C, j = e % C;
    F acc = 0;
#pragma unroll
    for (int k = 0; k < K; ++k) acc += (TA ? A[k * lda + i] : A[i * lda + k]) * (TB ? B[j * ldb + k] : B[k * ldb + j]);
    out[i * ldo + j] = add ? add[i * ldadd + j] + sign * acc : sign * acc;
  }
  __syncwarp();
}

// fp64 division and square root cost 300-450 cycles on this GPU (an FMA 50), so each column takes one reciprocal
// square root, and its scaling is fused with the trailing update, which reads the unscaled column from the
// working copy wk: one barrier per column. The trailing updates only touch the entries (i, k) with 1 <= k <= i
// (n(n-1)/2 of them, 28 for n = 8: one pass of the warp); each lane owns fixed entries for all columns, so there
// is no per-column index arithmetic (which the compiler hoisted out of the IPM loop and kept in registers).
template <int Nn, int G, typename F>
__device__ void chol(F* L, F* dinv, const F* A, int lda, F* wk, int lane, unsigned gm) {
  for (int e = lane; e < Nn * Nn; e += G) {           // sym(A) via L, as A may alias wk
    const int i = e / Nn, j = e % Nn;
    L[e] = i >= j ? F(0.5) * (A[i * lda + j] + A[j * lda + i]) : F(0);
  }
  __syncwarp(gm);
  for (int e = lane; e < Nn * Nn; e += G) { wk[e] = L[e]; L[e] = 0; }
  __syncwarp(gm);
  constexpr int TE = Nn * (Nn - 1) / 2, NQ = (TE + G - 1) / G;
  int ii[NQ > 0 ? NQ : 1], kk[NQ > 0 ? NQ : 1];
#pragma unroll
  for (int q = 0; q < NQ; ++q) {                    // entry e of the triangle 1 <= k <= i < Nn, row by row
    const int e = lane + G * q;
    int r = 0;
    while ((r + 1) * (r + 2) / 2 <= e) ++r;
    ii[q] = e < TE ? r + 1 : -1;
    kk[q] = e < TE ? e - r * (r + 1) / 2 + 1 : Nn;  // k = Nn: never updated
  }
#pragma unroll
  for (int j = 0; j < Nn; ++j) {                    // unrolled: the index arithmetic below is compile-time
    const F v = wk[j * Nn + j], rinv = rsqrt(v);
    for (int i = j + lane; i < Nn; i += G) L[i * Nn + j] = i == j ? v * rinv : wk[i * Nn + j] * rinv;
    if (lane == 0) dinv[j] = rinv;
#pragma unroll
    for (int q = 0; q < NQ; ++q) {
      const int i = ii[q], k = kk[q];
      if (k > j) wk[i * Nn + k] -= (wk[i * Nn + j] * rinv) * (wk[k * Nn + j] * rinv);
    }
    __syncwarp(gm);
  }
}

// each lane owns a column and substitutes forward (no barriers inside)
template <int Nn, int G, typename F, typename V>
__device__ void trsm_l(V* X, int ldx, const F* L, const F* dinv, const V* B, int ldb, int ncols, int lane,
                       unsigned gm) {
  for (int cc = lane; cc < ncols; cc += G) {
    V x[Nn];
#pragma unroll
    for (int i = 0; i < Nn; ++i) x[i] = B[i * ldb + cc];
#pragma unroll
    for (int i = 0; i < Nn; ++i) {
      x[i] *= dinv[i];
#pragma unroll
      for (int k = i + 1; k < Nn; ++k) x[k] -= L[k * Nn + i] * x[i];
    }
#pragma unroll
    for (int i = 0; i < Nn; ++i) X[i * ldx + cc] = x[i];
  }
  __syncwarp(gm);
}

// each lane owns a column and substitutes backward
template <int Nn, int G, typename F, typename V>
__device__ void trsm_lt(V* X, int ldx, const F* L, const F* dinv, const V* B, int ldb, int ncols, int lane,
                        unsigned gm) {
  for (int cc = lane; cc < ncols; cc += G) {
    V x[Nn];
#pragma unroll
    for (int i = 0; i < Nn; ++i) x[i] = B[i * ldb + cc];
#pragma unroll
    for (int i = Nn - 1; i >= 0; --i) {
      x[i] *= dinv[i];
#pragma unroll
      for (int k = 0; k < i; ++k) x[k] -= L[i * Nn + k] * x[i];
    }
#pragma unroll
    for (int i = 0; i < Nn; ++i) X[i * ldx + cc] = x[i];
  }
  __syncwarp(gm);
}

template <typename F>
__device__ void lu_solve(F* Am, F* Bm, int ldb, int nc, int lane) {
  for (int k = 0; k < NX; ++k) {
    F v = (lane >= k && lane < NX) ? fabs(Am[lane * NX + k]) : F(-1);
    int idx = lane;
    for (int o = 16; o > 0; o >>= 1) {
      const F v2 = __shfl_xor_sync(FULL, v, o);
      const int i2 = __shfl_xor_sync(FULL, idx, o);
      if (v2 > v || (v2 == v && i2 < idx)) { v = v2; idx = i2; }
    }
    const int piv = __shfl_sync(FULL, idx, 0);
    if (piv != k) {
      for (int j = lane; j < NX; j += 32) {
        const F a = Am[k * NX + j]; Am[k * NX + j] = Am[piv * NX + j]; Am[piv * NX + j] = a;
      }
      for (int j = lane; j < nc; j += 32) {
        const F b = Bm[k * ldb + j]; Bm[k * ldb + j] = Bm[piv * ldb + j]; Bm[piv * ldb + j] = b;
      }
    }
    __syncwarp();
    const F akk = Am[k * NX + k];
    for (int i = k + 1 + lane; i < NX; i += 32) Am[i * NX + k] /= akk;
    __syncwarp();
    const int rest = NX - k - 1;
    for (int e = lane; e < rest * rest; e += 32) {
      const int i = k + 1 + e / rest, j = k + 1 + e % rest;
      Am[i * NX + j] -= Am[i * NX + k] * Am[k * NX + j];
    }
    for (int e = lane; e < rest * nc; e += 32) {
      const int i = k + 1 + e / nc, j = e % nc;
      Bm[i * ldb + j] -= Am[i * NX + k] * Bm[k * ldb + j];
    }
    __syncwarp();
  }
  for (int cc = lane; cc < nc; cc += 32)
    for (int i = NX - 1; i >= 0; --i) {
      F v = Bm[i * ldb + cc];
      for (int k = i + 1; k < NX; ++k) v -= Am[i * NX + k] * Bm[k * ldb + cc];
      Bm[i * ldb + cc] = v / Am[i * NX + i];
    }
  __syncwarp();
}
