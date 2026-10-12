"""Linear plants for linear MPC."""
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np


def oscillating_masses(n_masses=6, spring=1.0, dt=0.5, pairs=((0, 1), (2, 3), (4, 5))):
    """A chain of unit masses joined to each other and to two walls by springs; actuator j pulls the masses
    pairs[j] towards each other (Wang and Boyd's fast-MPC example). State (positions, velocities), exact zero-order
    hold discretization; the plant is undamped, its eigenvalues on the unit circle. Returns (A, B) as NumPy
    arrays."""
    p = n_masses
    K = spring * (2 * np.eye(p) - np.eye(p, k=1) - np.eye(p, k=-1))
    F = np.zeros((p, len(pairs)))
    for j, (a, b) in enumerate(pairs):
        F[a, j], F[b, j] = 1.0, -1.0
    Ac = np.block([[np.zeros((p, p)), np.eye(p)], [-K, np.zeros((p, p))]])
    Bc = np.vstack([np.zeros((p, F.shape[1])), F])
    n, m = Bc.shape
    E = np.asarray(jsl.expm(jnp.asarray(np.block([[Ac, Bc], [np.zeros((m, n + m))]]) * dt)))
    return E[:n, :n], E[:n, n:]


def random_lti(n, m, rho=1.05, rng=None, seed=0):
    """A random LTI system (A, B) with spectral radius rho (open-loop unstable for rho > 1), drawn from the
    generator `rng` (default: np.random.default_rng(seed))."""
    rng = np.random.default_rng(seed) if rng is None else rng
    A = rng.normal(size=(n, n))
    A = rho * A / np.max(np.abs(np.linalg.eigvals(A)))
    return A, rng.normal(size=(n, m))
