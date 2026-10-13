// The matrix part of the Newton step: barrier-augmented stage data, the value-function scan for P_t, the
// gains and closed loop, and the chunk transition matrices of the solves.
#pragma once

#include "context.cuh"

// barrier-augmented stage data of stage t (W, Qt, St, Rt) and its element (Abar, M, Qbar) of the
// value-function scan, in c.T
template <int G>
__device__ __forceinline__ void task_build(const Ctx& c, int t, Lanes<G> ln, double* ws);

// P_t for every stage (c.Pm) by the scan over the elements in c.T, and the gains of every stage: the Cholesky
// factor of Sigma_t (c.Ls, c.Ldi), K_t and Acl_t = A_t - B_t K_t. lu: LU combines instead of the Cholesky form;
// bad is set to 1 if a P_t is not finite. Every level of the scan ends with a grid barrier.
__device__ __forceinline__ void factor(const Ctx& c, cg::grid_group& grid, int gw, int TW, int lane, double* ws,
                                       bool lu, double& bad);

// chunk transitions of chunk task / 2: forward Fp_t = Acl_t ... Acl_c0 (even task), backward
// Ps_t = Acl_t' ... Acl_{c1-1}' (odd task)
__device__ void task_chunk(const Ctx& c, int task, int lane);
