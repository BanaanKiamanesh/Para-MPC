"""Python wrapper of the CUDA kernel (src/cuda): the whole interior-point solve on the GPU in one kernel launch.

    ipm(data, z0, tol, mu_min, max_iter, start)   the contract of parampc.core.ipm with its default options;
                                                  core.ipm(..., method='cuda') calls it
    profile(data, z0, tol)                        where one solve spends its time, phase by phase

The kernel is compiled once per problem size (n states, m inputs) with the nvcc of the environment (the pip
package nvidia-cuda-nvcc, or $NVCC), cached in ~/.cache/parampc ($PARAMPC_CACHE), loaded with ctypes and
registered as an XLA FFI target. The call is an ordinary XLA operation on XLA's stream, without copies: it works
inside jax.jit, lax.scan and lax.while_loop.
"""
import ctypes
import functools
import hashlib
import os
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

_PACKAGE = Path(__file__).resolve().parent
DATA_KEYS = ('A', 'B', 'c', 'Q', 'S', 'R', 'q', 'r', 'QN', 'qN', 'C', 'D', 'd', 'x0')
ITERATE_KEYS = ('x', 'u', 'nu', 'lam', 's')
PHASES = ('residual', 'build', 'scan', 'S4b', 'bnd', 'chunks', 'S1', 'S2', 'S3', 'S4a', 'S5', 'S6', 'mu_aff',
          'update')    # the kernel's profiling slots (-DPM_PROFILE); bnd: scan matrices over the chunk boundaries


def source_dir():
    """The CUDA sources (include/, src/): $PARAMPC_CUDA_DIR, the copy shipped inside an installed package, or the
    repository's src/cuda."""
    for d in (os.environ.get('PARAMPC_CUDA_DIR'), _PACKAGE / '_cuda', _PACKAGE.parents[1] / 'cuda'):
        if d and (Path(d) / 'src' / 'ipm.cu').is_file():
            return Path(d)
    raise FileNotFoundError('parampc: CUDA sources not found (set PARAMPC_CUDA_DIR to the src/cuda directory)')


def nvcc():
    """The CUDA compiler: $NVCC, else the one in the environment's nvidia packages, else the PATH's."""
    if os.environ.get('NVCC'):
        return os.environ['NVCC']
    try:
        import nvidia
        for base in nvidia.__path__:
            for sub in ('cu13', 'cuda_nvcc'):
                path = os.path.join(base, sub, 'bin', 'nvcc')
                if os.path.exists(path):
                    return path
    except ImportError:
        pass
    return 'nvcc'


def compute_capability():
    dev = jax.devices('gpu')[0]
    return str(getattr(dev, 'compute_capability', '8.6')).replace('.', '')


def _cache_dir():
    """Where compiled kernels are cached: $PARAMPC_CACHE, else the user's cache directory."""
    for d in (os.environ.get('PARAMPC_CACHE'),
              os.path.join(os.environ.get('XDG_CACHE_HOME') or os.path.expanduser('~/.cache'), 'parampc')):
        if d:
            try:
                os.makedirs(d, exist_ok=True)
                if os.access(d, os.W_OK):
                    return d
            except OSError:
                pass
    raise RuntimeError('parampc: no writable cache directory for the CUDA kernel (set PARAMPC_CACHE)')


@functools.cache
def target(nx, nu, profile=False):
    """Compile (if needed), load and register the kernel for n = nx states and m = nu inputs. Returns the FFI
    target's name and the library's workspace-size function."""
    import jax.ffi
    root = source_dir()
    sources = sorted((root / 'include').rglob('*.cuh')) + [root / 'src' / 'ipm.cu']
    arch = compute_capability()
    digest = hashlib.sha1(f'{nx},{nu},{arch},{profile}'.encode())
    for path in sources:
        digest.update(path.read_bytes())
    tag = digest.hexdigest()[:10]
    build = _cache_dir()
    stem = f'libparampc_ipm_n{nx}_m{nu}_sm{arch}{"_prof" if profile else ""}'
    lib = os.path.join(build, f'{stem}_{tag}.so')
    if not os.path.exists(lib):
        tmp = f'{lib}.{os.getpid()}.tmp'
        cmd = [nvcc(), '-O3', '-std=c++17', f'-arch=sm_{arch}', '-shared', '-Xcompiler', '-fPIC',
               '-Xcompiler', '-Wno-return-type', '-diag-suppress', '940,2473', f'-DPM_NX={nx}', f'-DPM_NU={nu}',
               '-I', jax.ffi.include_dir(), '-I', str(root / 'include'), '-o', tmp, str(root / 'src' / 'ipm.cu')]
        if profile:
            cmd.append('-DPM_PROFILE')
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f'parampc: nvcc failed for n={nx}, m={nu}:\n{proc.stderr[-4000:]}')
        os.replace(tmp, lib)
        for old in os.listdir(build):                      # builds of earlier versions of the sources
            if old.startswith(stem + '_') and old.endswith('.so') and old != os.path.basename(lib):
                os.remove(os.path.join(build, old))
    name = f'parampc_ipm_n{nx}_m{nu}' + ('_prof' if profile else '')
    so = ctypes.cdll.LoadLibrary(lib)
    jax.ffi.register_ffi_target(name, jax.ffi.pycapsule(so.ParampcIpm), platform='CUDA')
    so.parampc_ipm_workspace.restype = ctypes.c_int64
    so.parampc_ipm_workspace.argtypes = (ctypes.c_int64, ctypes.c_int64)
    return name, so.parampc_ipm_workspace


def call(data, z0, tol, mu_min, max_iter, blocks=0, profile=False, start='given'):
    """The raw FFI call: (x, u, nu, lam, s, iterations, (best KKT score, final mu), workspace)."""
    N, nx, nu = data['B'].shape
    M = data['C'].shape[1]
    f64 = jnp.float64
    name, workspace = target(nx, nu, profile)
    prm = jnp.stack([jnp.asarray(tol, f64), jnp.asarray(mu_min, f64), jnp.asarray(max_iter, f64),
                     jnp.asarray(1.0 if start == 'mehrotra' else 0.0, f64)])
    shapes = (jax.ShapeDtypeStruct((N + 1, nx), f64), jax.ShapeDtypeStruct((N, nu), f64),
              jax.ShapeDtypeStruct((N + 1, nx), f64), jax.ShapeDtypeStruct((N, M), f64),
              jax.ShapeDtypeStruct((N, M), f64), jax.ShapeDtypeStruct((), jnp.int32),
              jax.ShapeDtypeStruct((2,), f64), jax.ShapeDtypeStruct((workspace(N, M),), f64))
    args = [data[k] for k in DATA_KEYS] + [z0[k] for k in ITERATE_KEYS] + [prm]
    return jax.ffi.ffi_call(name, shapes)(*[jnp.asarray(a, f64) for a in args], blocks=np.int64(blocks))


def ipm(data, z0, tol=1e-8, mu_min=1e-12, max_iter=30, start='given', blocks=0):
    """parampc.core.ipm with the default options, on the GPU in one kernel launch.
    Returns (best iterate, iterations, its KKT score, final duality measure).
    start: 'given' (z0) or 'mehrotra' (Mehrotra's starting point from z0, counted as one iteration).
    blocks: thread blocks of the persistent kernel (0: one per stage, up to what fits on the GPU)."""
    if start not in ('given', 'mehrotra'):
        raise ValueError(f"start must be 'given' or 'mehrotra', not {start!r}")
    x, u, nu, lam, s, k, info, _ = call(data, z0, tol, mu_min, max_iter, blocks, False, start)
    return dict(x=x, u=u, nu=nu, lam=lam, s=s), k, info[0], info[1]


def profile(data, z0, tol=1e-8, mu_min=1e-12, max_iter=30, blocks=0):
    """Share of one solve's time in each phase of the kernel (a -DPM_PROFILE build; one thread's clock, so a
    breakdown of the wall time, every phase ending with a grid barrier). Returns
    ({phase: fraction}, cycles per IPM iteration, iterations)."""
    out = jax.block_until_ready(jax.jit(lambda d, z: call(d, z, tol, mu_min, max_iter, blocks, True))(data, z0))
    prof, k = np.asarray(out[-1][-64:-32]), int(out[5])
    total = prof[:len(PHASES)].sum()
    return {p: float(v / total) for p, v in zip(PHASES, prof)}, float(total / max(k, 1)), k
