// Grid-wide barrier and reductions (grid.cuh).
#include "grid.cuh"
#include "warp_linalg.cuh"

__device__ __forceinline__ void bar(cg::grid_group& grid) {
  if (gridDim.x == 1) __syncthreads(); else grid.sync();
}

__device__ void put(const Ctx& c, int gw, int lane, int k, double v, int op) {
  v = warp_reduce(v, op);
  if (lane == 0) c.red[gw * RK + k] = v;
}
__device__ double get(const Ctx& c, int TW, int lane, int k, int op) {
  double v = op == SUM ? 0.0 : (op == MIN ? INFINITY : -INFINITY);
  for (int w = lane; w < TW; w += 32) v = combine_op(v, c.red[w * RK + k], op);
  return warp_reduce(v, op);
}
