"""qwen4exp hyper-connections: the residual path, in place of a pre-norm stream.

Instead of one residual stream with a norm in front of each sublayer, qwen4exp carries
``hc`` streams (4 here) and wraps every sublayer in a pair of operations:

    mixed, inject = mix(state)      # collapse the streams into one block input
    out           = sublayer(mixed)
    state         = combine(state, out, inject)   # scatter the output back, per stream

There is no ``attn_norm``, no ``ffn_norm`` and no ``output_norm`` in the checkpoint --
``mix`` is where the normalisation lives, and the final ``mix`` before the LM head *is*
the output norm.

Transcribed from llama.cpp ``src/models/qwen4exp.cpp`` (``build_hc_mix`` /
``build_hc_combine``) rather than reconstructed from the tensor shapes. The details that
shape numerics and would be invisible in a shape check:

* the RMS norm reduces over one stream (``n_embd``), not the flattened ``hc * n_embd``,
  while its gamma spans the whole flattened layout -- a grouped norm, not a plain one
* the down projection is scaled by ``1/hc`` *before* the SiLU
* streams are collapsed by their mean, not their sum
* ``combine`` weights are ``2 * sigmoid(inject / hc)``, centred on 1 so that a zero
  injection degenerates to an ordinary residual add
* the gamma stored in the file is already ``1 + w``; it is a plain multiply, not the
  ``(1 + w)`` form Gemma-style norms apply at runtime
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def grouped_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, hc: int
) -> torch.Tensor:
    """RMS-normalise each stream separately, then scale by a gamma over all of them.

    ``x`` is ``[T, hc, D]``; the reduction is over ``D`` (one stream), which is what
    ``ggml_rms_norm`` does to a ``[n_embd, hc, nt]`` tensor. ``weight`` is ``[hc * D]``,
    so the scale is per (stream, channel) and cannot be folded into the reduction.

    Returns ``[T, hc * D]``, the flattened layout the projections consume.
    """
    T, streams, D = x.shape
    assert streams == hc, f"expected {hc} streams, got {streams}"
    # Accumulate in at least float32 -- a bf16 sum of squares over 2560 channels loses
    # enough to matter -- but do not force float64 inputs down to float32, or a
    # double-precision comparison against a reference silently becomes a float32 one.
    acc = x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float32
    xa = x.to(acc)
    var = xa.pow(2).mean(dim=-1, keepdim=True)
    xn = (xa * torch.rsqrt(var + eps)).reshape(T, hc * D)
    return (xn * weight.to(acc)).to(x.dtype)


def hc_mix(
    state: torch.Tensor,
    w_norm: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    w_inject: torch.Tensor | None,
    *,
    eps: float,
    hc: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Collapse the ``hc`` residual streams into one sublayer input.

    ``state`` is ``[T, hc, D]``. Returns ``(mixed, inject)`` where ``mixed`` is ``[T, D]``
    and ``inject`` is ``[T, hc]`` -- the per-stream scatter weights ``combine`` will use.
    ``w_inject`` is None for the final mix before the LM head, which produces no injection.
    """
    T, _, D = state.shape
    # Compute in the weights' dtype. grouped_rms_norm accumulates in at least float32, and
    # letting that choice reach the GEMM both mismatches the weights and would silently
    # pick a slower kernel -- on Turing a batched bf16 GEMM is 12-22x slower than fp16.
    xn = grouped_rms_norm(state, w_norm, eps, hc).to(w_down.dtype)   # [T, hc*D]

    # The 1/hc is applied before the SiLU, so it is not a rescale of the gate.
    lo = F.silu(F.linear(xn, w_down) / hc)                 # [T, low_rank]
    gate = torch.sigmoid(F.linear(lo, w_up))               # [T, hc*D]

    gated = (xn * gate).reshape(T, hc, D)
    mixed = gated.mean(dim=1).to(state.dtype)              # [T, D]

    inject = F.linear(xn, w_inject) if w_inject is not None else None
    return mixed, inject


def hc_combine(
    state: torch.Tensor, block_out: torch.Tensor, inject: torch.Tensor, *, hc: int
) -> torch.Tensor:
    """Scatter a sublayer's output back across the streams.

    ``2 * sigmoid`` centres the weights on 1, so an all-zero ``inject`` reduces this to
    adding ``block_out`` to every stream -- an ordinary residual connection.
    """
    w = 2.0 * torch.sigmoid(inject / hc)                   # [T, hc]
    return state + block_out.unsqueeze(1) * w.unsqueeze(-1)


def hc_init(embed: torch.Tensor, hc: int) -> torch.Tensor:
    """The wide residual starts as ``hc`` identical copies of the token embedding."""
    return embed.unsqueeze(1).expand(-1, hc, -1).contiguous()
