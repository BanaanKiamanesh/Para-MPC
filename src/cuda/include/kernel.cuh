// The interior-point kernel: the whole solve of parampc.core.ipm in one cooperative launch.
#pragma once

#include "context.cuh"

// Solves the QP in c from the iterate in c.x, c.u, c.nu, c.lam, c.s; writes the best iterate to c.bx, ..., c.bs,
// the iterations to c.kout and (best KKT score, final mu) to c.info. Launch with cudaLaunchCooperativeKernel,
// PM_THREADS threads per block at most and c.WS doubles of dynamic shared memory per warp.
__global__ void __launch_bounds__(PM_THREADS, 1) ipm_kernel(Ctx c);
