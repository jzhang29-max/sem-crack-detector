"""Redirect hardcoded device="cuda" to an available device.

sam3/model/position_encoding.py:55 constructs a precompute buffer with a literal
device="cuda", ignoring the device argument threaded through build_sam3_image_model.
On Apple Silicon that raises "Torch not compiled with CUDA enabled". This rewrites the
device on torch's tensor factory functions for the duration of a `with` block, so the
build succeeds without editing site-packages.

Deliberately narrow: it only rewrites a CUDA request, and only while active.
"""
import contextlib
import torch

_FACTORIES = ("zeros", "ones", "empty", "full", "arange", "tensor", "linspace", "eye", "rand", "randn")


def _is_cuda(d):
    if d is None:
        return False
    s = str(d if not isinstance(d, torch.device) else d.type)
    return s.startswith("cuda")


@contextlib.contextmanager
def redirect_cuda(to="cpu"):
    orig = {n: getattr(torch, n) for n in _FACTORIES if hasattr(torch, n)}

    def wrap(fn):
        def inner(*a, **k):
            if _is_cuda(k.get("device")):
                k["device"] = to
            return fn(*a, **k)
        return inner

    try:
        for n, fn in orig.items():
            setattr(torch, n, wrap(fn))
        yield
    finally:
        for n, fn in orig.items():
            setattr(torch, n, fn)


@contextlib.contextmanager
def no_pin_memory():
    """Neutralise Tensor.pin_memory().

    sam3/model/geometry_encoders.py:651 calls scale.pin_memory() before a non_blocking
    copy. On Apple Silicon torch resolves the pinning device to mps:0, and pinning a CPU
    tensor to an MPS storage raises. pin_memory is purely a host-to-device transfer
    optimisation, so returning the tensor unchanged is numerically identical.
    """
    orig = torch.Tensor.pin_memory
    torch.Tensor.pin_memory = lambda self, *a, **k: self
    try:
        yield
    finally:
        torch.Tensor.pin_memory = orig
