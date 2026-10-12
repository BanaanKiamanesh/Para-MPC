// Compile-time configuration of the IPM kernel: problem sizes, launch constants, lane groups.
#pragma once

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace cg = cooperative_groups;

#ifndef PM_NX
#error "compile with -DPM_NX=<states> -DPM_NU=<inputs>"
#endif
constexpr int NX = PM_NX, NU = PM_NU, N2 = NX * NX;
#ifndef PM_THREADS
#define PM_THREADS 256                      // threads per block (8 warps: no register spills)
#endif
#ifndef PM_KS_PER_SM
#define PM_KS_PER_SM 3                      // Kogge-Stone over at most PM_KS_PER_SM x multiprocessors elements
#endif
constexpr int KS_ALLOC = 2048;              // at most this many Kogge-Stone elements (size of T2)
constexpr int RK = 8;                       // reduction slots per warp
constexpr int MAXWARPS = 4096;              // reduction partials reserved in the workspace
static_assert(NX <= 32 && NU <= 32, "one warp handles a stage: n, m <= 32");
constexpr unsigned FULL = 0xffffffffu;
// the lanes that handle one stage: a whole warp (G = 32), or a group of G lanes of a warp; sl is the lane within
// the group and gm the group's mask for __syncwarp
template <int G>
struct Lanes {
  int sl;
  unsigned gm;
};
// lanes per stage when a warp handles several stages at once: the vectors of a stage have n or m entries
constexpr int GP = (NX > 16 || NU > 16) ? 32 : (NX > 8 || NU > 8) ? 16 : 8;
constexpr int SPW = 32 / GP;                // stages per warp
enum Kind { PRED = 0, CORR = 1, GOND = 2, REFN = 3 };
enum Op { SUM = 0, MIN = 1, MAX = 2 };
