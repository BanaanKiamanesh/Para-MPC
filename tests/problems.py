"""Named test QPs: each exercises one feature of the problem class."""
import numpy as np

import parampc as pm


def lti(n=6, m=3, N=20, seed=0, unstable=1.05):
    """A random LTI system (spectral radius `unstable`), its equilibrium for a random u_ref, and an initial state."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    A = unstable * A / np.max(np.abs(np.linalg.eigvals(A)))
    B = rng.normal(size=(n, m))
    u_ref = 0.3 * rng.normal(size=m)
    x_ref = np.linalg.solve(np.eye(n) - A, B @ u_ref)
    x0 = x_ref + 3.0 * rng.normal(size=n) / np.sqrt(n)
    return A, B, x_ref, u_ref, x0


def case(name, N=20):
    """A named test QP (see CASES)."""
    if name == 'boxes':                      # input and state boxes, tracking an equilibrium
        A, B, x_ref, u_ref, x0 = lti(N=N)
        qp = pm.make_qp(A, B, Q=np.eye(6), R=0.1 * np.eye(3), N=N, x0=x0, QN=10 * np.eye(6), x_ref=x_ref, u_ref=u_ref)
        return pm.add_bounds(qp, u_min=-0.5, u_max=0.5, x_min=-4.0, x_max=4.0)
    if name == 'terminal':                   # an active terminal box
        A, B, x_ref, u_ref, x0 = lti(N=N, seed=1)
        qp = pm.make_qp(A, B, Q=np.eye(6), R=0.1 * np.eye(3), N=N, x0=x0, QN=0.1 * np.eye(6))
        return pm.add_bounds(qp, u_min=-1.0, u_max=1.0, xN_min=x_ref - 0.02, xN_max=x_ref + 0.02)
    if name == 'ltv':                        # time-varying data, cross terms S, general coupled rows C x + D u <= d
        rng = np.random.default_rng(2)
        n, m, M = 5, 2, 4
        A = 0.95 * np.eye(n) + 0.1 * rng.normal(size=(N, n, n))
        B = rng.normal(size=(N, n, m))
        L = rng.normal(size=(N, n + m, n + m))
        H = L @ np.swapaxes(L, 1, 2) / (n + m) + 0.3 * np.eye(n + m)
        C, D = rng.normal(size=(N, M, n)), rng.normal(size=(N, M, m))
        return pm.make_qp(A, B, Q=H[:, :n, :n], R=H[:, n:, n:], S=H[:, n:, :n], N=N, x0=rng.normal(size=n),
                          q=0.1 * rng.normal(size=(N, n)), r=0.1 * rng.normal(size=(N, m)),
                          c=0.05 * rng.normal(size=(N, n)), C=C, D=D, d=0.5 + rng.random((N, M)))
    if name == 'semidefinite':               # output tracking Q = c c', no terminal cost (the LU fallback)
        A, B, x_ref, u_ref, x0 = lti(N=N, seed=3, unstable=0.98)
        cvec = np.ones((1, 6)) / np.sqrt(6)
        qp = pm.make_qp(A, B, Q=cvec.T @ cvec, R=0.1 * np.eye(3), N=N, x0=x0, QN=np.zeros((6, 6)), q=-cvec[0])
        return pm.add_bounds(qp, u_min=-0.5, u_max=0.5)
    if name == 'wide':                       # more inputs than states (m > n)
        rng = np.random.default_rng(4)
        n, m = 3, 7
        A = 1.02 * np.eye(n) + 0.05 * rng.normal(size=(n, n))
        B = rng.normal(size=(n, m))
        qp = pm.make_qp(A, B, Q=np.eye(n), R=0.5 * np.eye(m), N=N, x0=3 * rng.normal(size=n))
        return pm.add_bounds(qp, u_min=-0.2, u_max=0.2, x_min=-5.0, x_max=5.0)
    raise ValueError(name)


CASES = ('boxes', 'terminal', 'ltv', 'semidefinite', 'wide')
