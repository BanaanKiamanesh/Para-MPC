// Build entry point: the kernel library (libparampc_ipm) as one translation unit.
//
// parampc/cuda.py compiles this file only (nvcc -DPM_NX=<n> -DPM_NU=<m> -I include). It includes the sources in
// dependency order, so that nvcc sees the whole kernel at once: the kernel relies on every phase being inlined
// into it (see kernel.cu), and nvcc inlines device functions only within a translation unit. Compiling the
// sources separately would need relocatable device code with device link-time optimization, which changes the
// kernel's register allocation and needs a libnvvm that the pip CUDA toolchain does not ship.
//
//   warp_linalg.cu  small dense matrices on one warp: products, Cholesky, triangular solves, LU
//   grid.cu         grid barrier and reductions
//   residual.cu     the KKT residual
//   factor.cu       the factorization: stage data, the value-function scan, gains, chunk transitions
//   solve.cu        one solve with the factorization (the phases S1 to S6)
//   kernel.cu       the interior-point iteration
//   ffi.cu          host side: workspace, launch configuration, the XLA FFI handler
#include "warp_linalg.cu"
#include "grid.cu"
#include "residual.cu"
#include "factor.cu"
#include "solve.cu"
#include "kernel.cu"
#include "ffi.cu"
