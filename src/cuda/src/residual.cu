// The KKT residual at the current iterate (residual.cuh).
#include "residual.cuh"
#include "warp_linalg.cuh"

template <int G>
__device__ __forceinline__ void task_residual(const Ctx& c, int t, Lanes<G> ln, double& pmax, double& psum) {
  const int lane = ln.sl;                                            // the lane within the group
  const int M = c.M;
  if (t == c.N) {
    const double *xN = c.x + (size_t)c.N * NX, *nN = c.nu + (size_t)c.N * NX;
    for (int i = lane; i < NX; i += G) {
      double qx = 0.0;
      for (int j = 0; j < NX; ++j) qx += c.QN[i * NX + j] * xN[j];
      const double v = nN[i] - qx - c.qN[i];
      c.rxN[i] = v;
      pmax = nmax(pmax, fabs(v));
    }
    return;
  }
  const double *A = c.A + (size_t)t * N2, *B = c.B + (size_t)t * NX * NU, *Q = c.Q + (size_t)t * N2;
  const double *S = c.S + (size_t)t * NU * NX, *R = c.R + (size_t)t * NU * NU;
  const double *C = c.C + (size_t)t * M * NX, *D = c.D + (size_t)t * M * NU;
  const double *xt = c.x + (size_t)t * NX, *x1 = xt + NX, *ut = c.u + (size_t)t * NU;
  const double *nt = c.nu + (size_t)t * NX, *n1 = nt + NX, *lt = c.lam + (size_t)t * M, *st = c.s + (size_t)t * M;
  for (int i = lane; i < NX; i += G) {
    double qx = 0.0, su = 0.0, an = 0.0, cl = 0.0, ax = 0.0, bu = 0.0;
    for (int j = 0; j < NX; ++j) {
      qx += Q[i * NX + j] * xt[j]; an += A[j * NX + i] * n1[j]; ax += A[i * NX + j] * xt[j];
    }
    for (int j = 0; j < NU; ++j) { su += S[j * NX + i] * ut[j]; bu += B[i * NU + j] * ut[j]; }
    for (int k = 0; k < M; ++k) cl += C[k * NX + i] * lt[k];
    const double vx = nt[i] - qx - su - c.q[(size_t)t * NX + i] - an - cl;
    const double vd = x1[i] - ax - bu - c.c[(size_t)t * NX + i];
    c.rx[(size_t)t * NX + i] = vx;
    c.rd[(size_t)t * NX + i] = vd;
    pmax = nmax(pmax, nmax(fabs(vx), fabs(vd)));
  }
  for (int i = lane; i < NU; i += G) {
    double ru = 0.0, sx = 0.0, bn = 0.0, dl = 0.0;
    for (int j = 0; j < NU; ++j) ru += R[i * NU + j] * ut[j];
    for (int j = 0; j < NX; ++j) { sx += S[i * NX + j] * xt[j]; bn += B[j * NU + i] * n1[j]; }
    for (int k = 0; k < M; ++k) dl += D[k * NU + i] * lt[k];
    const double v = ru + sx + c.r[(size_t)t * NU + i] + bn + dl;
    c.ru[(size_t)t * NU + i] = v;
    pmax = nmax(pmax, fabs(v));
  }
  for (int k = lane; k < M; k += G) {
    double cx = 0.0, du = 0.0;
    for (int j = 0; j < NX; ++j) cx += C[k * NX + j] * xt[j];
    for (int j = 0; j < NU; ++j) du += D[k * NU + j] * ut[j];
    const double v = cx + du + st[k] - c.d[(size_t)t * M + k];
    c.rc[(size_t)t * M + k] = v;
    pmax = nmax(pmax, fabs(v));
    psum += lt[k] * st[k];
  }
  if (t == 0)
    for (int i = lane; i < NX; i += G) {
      const double v = c.x[i] - c.xbar[i];
      c.r0[i] = v;
      pmax = nmax(pmax, fabs(v));
    }
}
