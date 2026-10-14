// Warp-level dense linear algebra on small matrices in shared memory (one warp, or a group of its lanes).
// Matrices are row-major with an explicit row stride (ld); every routine ends with a warp barrier. The matrix
// routines are templates over the scalar type F: double, or float for the factorization of the float32 phase.
#pragma once

#include "config.cuh"

// NaN-propagating min and max, clipping to [0, 1], and the reduction operator op (SUM, MIN or MAX)
__device__ __forceinline__ double nmin(double a, double b);
__device__ __forceinline__ double nmax(double a, double b);
__device__ __forceinline__ double clip01(double v);
__device__ __forceinline__ double combine_op(double a, double b, int op);

// warp reduction; the result is lane 0's and broadcast, so every lane holds the same bits
__device__ double warp_reduce(double v, int op);

// a non-deduced context: a literal argument (sign = 1.0) does not take part in deducing F
template <typename T>
struct same_t {
  using type = T;
};

// out (R x C, row stride ldo) = add + sign * op(A) op(B); op(A) is R x K, op(B) is K x C (TA, TB: transposed)
template <int R, int K, int C, bool TA, bool TB, typename F>
__device__ void mm(F* out, int ldo, const F* A, int lda, const F* B, int ldb, int lane, const F* add = nullptr,
                   int ldadd = 0, typename same_t<F>::type sign = 1);

// L = cholesky(sym(A)), lower, and dinv = 1 / diag(L); NaN if A is not positive definite (as jnp.linalg.cholesky).
// wk: Nn x Nn working copy (may be A itself if A is scratch). G lanes do the work: a warp, or a group of a warp
// synchronized by the mask gm.
template <int Nn, int G = 32, typename F>
__device__ void chol(F* L, F* dinv, const F* A, int lda, F* wk, int lane, unsigned gm = FULL);

// X = L^-1 B for ncols columns, with L and dinv from chol; X may alias B. The arithmetic is in the type V of X
// and B (the solves use a float factorization in double arithmetic).
template <int Nn, int G = 32, typename F, typename V>
__device__ void trsm_l(V* X, int ldx, const F* L, const F* dinv, const V* B, int ldb, int ncols, int lane,
                       unsigned gm = FULL);

// X = L^-T B for ncols columns; X may alias B
template <int Nn, int G = 32, typename F, typename V>
__device__ void trsm_lt(V* X, int ldx, const F* L, const F* dinv, const V* B, int ldb, int ncols, int lane,
                        unsigned gm = FULL);

// solve Am X = Bm (Am NX x NX, Bm NX x nc) in place by LU with partial pivoting (as jnp.linalg.solve); Am is
// destroyed
template <typename F>
__device__ void lu_solve(F* Am, F* Bm, int ldb, int nc, int lane);
