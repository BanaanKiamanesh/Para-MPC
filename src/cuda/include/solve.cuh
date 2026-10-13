// One solve with the factorization: right-hand side, the two affine recursions by chunks with a scan over
// the chunk boundaries, and the recovery of the step. A solve is the phases S1 to S6, each ending with a grid
// barrier (S3 and S5 run in block 0).
#pragma once

#include "context.cuh"

struct SolveArgs {
  int kind;             // PRED, CORR, GOND or REFN
  double tau, tp, td;   // tau = sigma mu; tp, td: trial step lengths of the centrality corrector
  bool keep;            // refinement: whether the centrality correction was kept
};

// S1: right-hand side of the solve at stage t (newton_direction's lq_rhs and the g of solve_lq)
template <int G>
__device__ __forceinline__ void task_s1(const Ctx& c, int t, Lanes<G> ln, double* ws, const SolveArgs& a);

// S2: local backward recursion q_t = Acl_t' q_{t+1} + g_t in chunk ch, from q = 0 at its end
__device__ void task_s2(const Ctx& c, int ch, int lane, double* ws);

// S3: costates at the chunk starts, p_c0 = Ps_c0 p_c1 + q_c0, from p_N = rhoxN (one block; buf: 2 P NX doubles)
__device__ __forceinline__ void task_s3(const Ctx& c, double* buf);

// S4a (per stage): the costate p_t inside its chunk, p_t = Ps_t p_c1 + q_t, then the gain
// k_t = Sigma_t^-1 (rhou_t + B_t'(P_{t+1} chat_t + p_{t+1})) and f_t = chat_t - B_t k_t
template <int G>
__device__ __forceinline__ void task_s4a(const Ctx& c, int t, Lanes<G> ln, double* ws);

// S4b: local forward recursion y_{t+1} = Acl_t y_t + f_t in chunk ch, from y = 0 at its start
__device__ void task_s4b(const Ctx& c, int ch, int lane, double* ws);

// S5: states at the chunk ends, x_c1 = Fp_{c1-1} x_c0 + yend, from x_0 = dx0 (one block; buf: 2 P NX doubles)
__device__ __forceinline__ void task_s5(const Ctx& c, double* buf);

// S6: dx_t, du_t, dnu_t, dlam_t, ds_t of stage t; writes the result of this solve (PRED and GOND into the
// direction D*, CORR and REFN into the step S*) and the partials of the step lengths to the boundary (rs, rl)
template <int G>
__device__ __forceinline__ void task_s6(const Ctx& c, int t, Lanes<G> ln, double* ws, const SolveArgs& a, double& rs,
                                        double& rl);

// level l >= 1 of the matrices of the scan over the chunk boundaries (both directions); they depend on the
// factorization only, so they are computed once per iteration, over the whole grid
__device__ __forceinline__ void bnd_level(const Ctx& c, int l, int gw, int TW, int lane);
