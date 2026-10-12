// Grid-wide barriers and reductions, and the profiling marks of -DPM_PROFILE builds.
#pragma once

#include "parampc/context.cuh"
#include "parampc/warp_linalg.cuh"

__device__ __forceinline__ void bar(cg::grid_group& grid) {
  if (gridDim.x == 1) __syncthreads(); else grid.sync();
}

// -DPM_PROFILE: thread 0 adds the cycles since its last mark to slot i (phases end with a barrier)
#ifdef PM_PROFILE
#define MARK(i) do { if (gw == 0 && lane == 0) { const long long t1 = clock64(); c.prof[i] += (double)(t1 - t0); t0 = t1; } } while (0)
#else
#define MARK(i) do { } while (0)
#endif

// reductions: every warp writes its partial, after a barrier every warp reduces all partials alike
__device__ void put(const Ctx& c, int gw, int lane, int k, double v, int op) {
  v = warp_reduce(v, op);
  if (lane == 0) c.red[gw * RK + k] = v;
}
__device__ double get(const Ctx& c, int TW, int lane, int k, int op) {
  double v = op == SUM ? 0.0 : (op == MIN ? INFINITY : -INFINITY);
  for (int w = lane; w < TW; w += 32) v = combine_op(v, c.red[w * RK + k], op);
  return warp_reduce(v, op);
}
