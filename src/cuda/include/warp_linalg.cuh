// Warp-level dense linear algebra on small matrices in shared memory (one warp, or a group of its lanes).
// Matrices are row-major with an explicit row stride (ld); every routine ends with a warp barrier.
#pragma once

#include "config.cuh"

// NaN-propagating min and max, clipping to [0, 1], and the reduction operator op (SUM, MIN or MAX)
__device__ __forceinline__ double nmin(double a, double b);
__device__ __forceinline__ double nmax(double a, double b);
__device__ __forceinline__ double clip01(double v);
__device__ __forceinline__ double combine_op(double a, double b, int op);

// warp reduction; the result is lane 0's and broadcast, so every lane holds the same bits
__device__ double warp_reduce(double v, int op);

// out (R x C, row stride ldo) = add + sign * op(A) op(B); op(A) is R x K, op(B) is K x C (TA, TB: transposed)
template <int R, int K, int C, bool TA, bool TB>
__device__ void mm(double* out, int ldo, const double* A, int lda, const double* B, int ldb, int lane,
                   const double* add = nullptr, int ldadd = 0, double sign = 1.0);

// L = cholesky(sym(A)), lower, and dinv = 1 / diag(L); NaN if A is not positive definite (as jnp.linalg.cholesky).
// wk: Nn x Nn working copy (may be A itself if A is scratch). G lanes do the work: a warp, or a group of a warp
// synchronized by the mask gm.
template <int Nn, int G = 32>
__device__ void chol(double* L, double* dinv, const double* A, int lda, double* wk, int lane, unsigned gm = FULL);

// X = L^-1 B for ncols columns, with L and dinv from chol; X may alias B
template <int Nn, int G = 32>
__device__ void trsm_l(double* X, int ldx, const double* L, const double* dinv, const double* B, int ldb, int ncols,
                       int lane, unsigned gm = FULL);

// X = L^-T B for ncols columns; X may alias B
template <int Nn, int G = 32>
__device__ void trsm_lt(double* X, int ldx, const double* L, const double* dinv, const double* B, int ldb, int ncols,
                        int lane, unsigned gm = FULL);

// solve Am X = Bm (Am NX x NX, Bm NX x nc) in place by LU with partial pivoting (as jnp.linalg.solve); Am is
// destroyed
__device__ void lu_solve(double* Am, double* Bm, int ldb, int nc, int lane);
