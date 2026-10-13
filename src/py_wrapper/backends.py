"""The solver backends, and which of them can run on this machine.

    'cuda'   the whole interior-point solve as one CUDA kernel (needs a GPU and nvcc)
    'scan'   the associative scans compiled by XLA (CPU or GPU)
    'sweep'  the sequential Riccati recursion inside the same IPM (fastest on a CPU)
    'auto'   'cuda' on a GPU with nvcc, else 'sweep' on a CPU, else 'scan'
"""
import os
import shutil

import jax

BACKENDS = ('cuda', 'scan', 'sweep')


def available_backends():
    """The backends that can run here, fastest first."""
    out = ['scan', 'sweep']
    try:
        gpu = any(dev.platform == 'gpu' for dev in jax.devices())
    except RuntimeError:
        gpu = False
    if gpu:
        from . import cuda
        if shutil.which(cuda.nvcc()) or os.path.exists(cuda.nvcc()):
            out.insert(0, 'cuda')
    return out


def resolve_backend(backend):
    """The backend that 'auto' stands for here; any other name is checked and returned."""
    if backend == 'auto':
        avail = available_backends()
        return 'cuda' if 'cuda' in avail else ('sweep' if jax.default_backend() == 'cpu' else 'scan')
    if backend not in BACKENDS:
        raise ValueError(f"backend must be 'auto' or one of {BACKENDS}, not {backend!r}")
    return backend
