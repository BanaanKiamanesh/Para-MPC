// The KKT residual at the current iterate.
#pragma once

#include "context.cuh"

// residual rows of stage t of the KKT system at the current iterate (the tau-independent part) into c.rx, c.ru,
// c.rd, c.rc (c.r0 at t = 0, c.rxN at t = N), and the KKT score's parts: pmax takes the largest residual entry,
// psum adds the products lam_k s_k
template <int G>
__device__ __forceinline__ void task_residual(const Ctx& c, int t, Lanes<G> ln, double& pmax, double& psum);
