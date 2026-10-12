"""Shared test setup: the backends to test, and the skip rule for tests marked gpu (they need a CUDA GPU and nvcc;
run them with JAX_PLATFORMS=cuda,cpu)."""
import pytest

import parampc

HAVE_CUDA = 'cuda' in parampc.available_backends()
BACKENDS = ['sweep', 'scan'] + (['cuda'] if HAVE_CUDA else [])


def pytest_collection_modifyitems(config, items):
    if HAVE_CUDA:
        return
    skip = pytest.mark.skip(reason='needs a CUDA GPU and nvcc (backend cuda)')
    for item in items:
        if 'gpu' in item.keywords:
            item.add_marker(skip)


def pytest_generate_tests(metafunc):
    if 'backend' in metafunc.fixturenames:
        metafunc.parametrize('backend', BACKENDS)
