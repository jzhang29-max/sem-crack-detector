"""Import sam3 on a machine with no triton, without stubbing triton globally.

Stubbing `triton` makes torch._dynamo and torch._inductor believe a real triton exists and
demand progressively more of its API. Instead we pre-register the single module that needs
it — sam3.model.edt, a Euclidean-distance-transform kernel used only by the VIDEO TRACKER —
so the real file is never executed and torch never sees triton at all.

If anything ever calls into it, it raises loudly rather than returning wrong numbers.
"""
import sys, types

_m = types.ModuleType("sam3.model.edt")


def edt_triton(*a, **k):
    raise RuntimeError(
        "edt_triton (video-tracker path) requires a real triton/CUDA build; "
        "this run is image-only and must not reach it."
    )


_m.edt_triton = edt_triton
sys.modules["sam3.model.edt"] = _m
