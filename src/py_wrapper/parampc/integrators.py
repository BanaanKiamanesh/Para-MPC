"""Explicit integrators that turn continuous-time dynamics f(x, u, p) into a discrete-time step F(x, u, p)."""
from jax import lax


def euler(f, dt):
    """x_{t+1} = x + dt f(x, u, p)."""
    return lambda x, u, p: x + dt * f(x, u, p)


def midpoint(f, dt):
    """The explicit midpoint rule (RK2)."""
    def step(x, u, p):
        k1 = f(x, u, p)
        return x + dt * f(x + 0.5 * dt * k1, u, p)
    return step


def rk4(f, dt, substeps=1):
    """The classical Runge-Kutta method, `substeps` steps of dt / substeps."""
    h = dt / substeps

    def step(x, u, p):
        def one(x, _):
            k1 = f(x, u, p)
            k2 = f(x + 0.5 * h * k1, u, p)
            k3 = f(x + 0.5 * h * k2, u, p)
            k4 = f(x + h * k3, u, p)
            return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), None
        return lax.scan(one, x, None, length=substeps)[0]
    return step
