"""qwen4exp gated attention: the query projection carries its own gate.

On full-attention layers ``attn_q`` is ``[hidden, 2 * n_head * head_dim]`` -- twice what
the head geometry calls for -- because each head's queries are followed by that head's
output gate. Those layers carry no separate ``attn_gate`` tensor, while the SSM layers do.

The layout is **interleaved per head**::

    [h0_q (head_dim) | h0_gate (head_dim) | h1_q | h1_gate | ...]

not the two contiguous halves the shape alone suggests. llama.cpp takes both with a
``ggml_view_3d`` of stride ``2 * head_dim``, at offsets 0 and ``head_dim``. Splitting it
down the middle instead would hand every head another head's gate: still the right shape,
still fast, and wrong in a way no structural check would catch.

The gate is applied to the attention output *before* the output projection, as
``out * sigmoid(gate)``.

Transcribed from llama.cpp ``src/models/qwen4exp.cpp`` (``build_layer_attn``).
"""

from __future__ import annotations

import torch


def split_q_and_gate(
    q_full: torch.Tensor, num_heads: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Separate the fused query projection into queries and per-head gates.

    ``q_full`` is ``[T, 2 * num_heads * head_dim]``. Returns ``(q, gate)``, each
    ``[T, num_heads, head_dim]``.
    """
    T, width = q_full.shape
    expect = 2 * num_heads * head_dim
    if width != expect:
        raise ValueError(
            f"gated q projection should be {expect} wide "
            f"(2 * {num_heads} heads * {head_dim}), got {width}"
        )
    # Heads are the slow axis and (q, gate) the fast one, which is what makes this an
    # interleave rather than a split.
    pair = q_full.reshape(T, num_heads, 2, head_dim)
    return pair[:, :, 0, :], pair[:, :, 1, :]


def rms_norm_heads(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Per-head RMS norm over ``head_dim``, the form ``attn_q_norm``/``attn_k_norm`` take.

    ``x`` is ``[T, heads, head_dim]`` and ``weight`` is ``[head_dim]``, shared by every
    head.
    """
    acc = x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float32
    xa = x.to(acc)
    var = xa.pow(2).mean(dim=-1, keepdim=True)
    return ((xa * torch.rsqrt(var + eps)) * weight.to(acc)).to(x.dtype)


def apply_output_gate(attn_out: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """``attn_out * sigmoid(gate)``, before the output projection.

    Both are ``[T, num_heads, head_dim]``; the gate is per (head, channel), not per head.
    """
    if attn_out.shape != gate.shape:
        raise ValueError(
            f"attention output {tuple(attn_out.shape)} and gate {tuple(gate.shape)} "
            f"must have the same shape"
        )
    return attn_out * torch.sigmoid(gate)
