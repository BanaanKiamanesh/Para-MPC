// Grid-wide barrier and reductions over all warps of the grid.
#pragma once

#include "context.cuh"

// barrier over the whole grid (a block barrier when the grid is one block)
__device__ __forceinline__ void bar(cg::grid_group& grid);

// reductions in two steps: every warp gw stores its partial of v in slot k (k < RK) with put; after a barrier,
// get reduces the partials of all TW warps, alike in every warp, so every lane of the grid holds the same bits
__device__ void put(const Ctx& c, int gw, int lane, int k, double v, int op);
__device__ double get(const Ctx& c, int TW, int lane, int k, int op);
