from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def _torch_act_and_mul(fn, x: torch.Tensor, out: torch.Tensor | None = None):
    a, b = x.chunk(2, dim=-1)
    result = fn(a) * b
    if out is not None:
        out.copy_(result)
        return out
    return result


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    import torch

    from freetoken.kernel.backend import is_flashinfer_installed

    if is_flashinfer_installed():
        from flashinfer import silu_and_mul
    elif getattr(torch.version, "hip", None):
        # AMD ROCm port: the triton activation kernel fails LLVM register
        # allocation on some RDNA3 shapes; plain torch ops stay correct.
        return _torch_act_and_mul(torch.nn.functional.silu, x, out=out)
    else:
        from freetoken.kernel.triton.activation import silu_and_mul

        return silu_and_mul(x, out=out)


def gelu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    import torch

    from freetoken.kernel.backend import is_flashinfer_installed

    if is_flashinfer_installed():
        from flashinfer import gelu_and_mul
    elif getattr(torch.version, "hip", None):
        return _torch_act_and_mul(torch.nn.functional.gelu, x, out=out)
    else:
        from freetoken.kernel.triton.activation import gelu_and_mul

        return gelu_and_mul(x, out=out)


def gelu_tanh_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    """tanh-approximate GELU gate (`gelu_pytorch_tanh`) followed by elementwise mul."""
    from freetoken.kernel.backend import is_flashinfer_installed

    if is_flashinfer_installed():
        from flashinfer import gelu_tanh_and_mul
    else:
        from freetoken.kernel.triton.activation import gelu_tanh_and_mul

    return gelu_tanh_and_mul(x, out=out)


def swigluoai_and_mul(
    x: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    alpha: float = 1.702,
    limit: float = 7.0,
):
    """SwiGLU-OAI (gpt-oss / MiniMax-M3 ``swigluoai``) over UNINTERLEAVED halves
    (gate ``x[..., :d]``, up ``x[..., d:]``): ``clamp(gate, max=limit) *
    sigmoid(alpha * gate) * (clamp(up, +-limit) + 1)``. Always the in-repo Triton
    kernel (flashinfer ships no clamped-swiglu *_and_mul)."""
    from freetoken.kernel.triton.activation import swigluoai_and_mul

    return swigluoai_and_mul(x, out=out, alpha=alpha, limit=limit)


__all__ = ["silu_and_mul", "gelu_and_mul", "gelu_tanh_and_mul", "swigluoai_and_mul"]
