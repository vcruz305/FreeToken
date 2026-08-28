"""qwen4exp lightning indexer: block scoring and top-k key selection.

Full-attention layers were trained to attend to ``indexer_top_k`` keys chosen by a small
indexer, not to the whole context. Dense attention is exactly equivalent while the context
fits inside that budget and diverges above it, so this is what long-context fidelity needs.

This module is the numerically delicate half -- pooling, scoring, selection -- kept pure so
it can be checked against the reference. The remaining piece is plumbing: a raw-key cache
tracking the attention cache cell for cell, and a way to hand the selection to the
attention backend, which currently has no per-key mask parameter.

Transcribed from llama.cpp ``src/models/qwen4exp.cpp`` (``build_qsa_top_k``). The details
that decide whether it is right, none of which a shape check would catch:

* cached indexer keys are **raw**. Pooling precedes norm and rope, so neither is applied
  before caching -- normalising first would change what the mean pools over.
* the score is rectified **per head, before** the reduction. Summing first and rectifying
  after is a different function, and the DeepSeek lightning indexer does the former.
* selection is over whole blocks, so the budget is ``top_k + compress_ratio - 1``: enough
  for the whole blocks plus the incomplete tail.
* every token of a block inherits its block's score.
"""

from __future__ import annotations

import torch


def pool_blocks(keys: torch.Tensor, compress_ratio: int) -> torch.Tensor:
    """Mean-pool raw indexer keys into blocks.

    ``keys`` is ``[n_kv, idx_dim]`` in cell order. Returns ``[n_blocks, idx_dim]`` where
    ``n_blocks = ceil(n_kv / compress_ratio)``. A trailing partial block averages only its
    live members, which is what pooling over its actual cells does.
    """
    n_kv, idx_dim = keys.shape
    r = compress_ratio
    n_blocks = (n_kv + r - 1) // r
    pad = n_blocks * r - n_kv
    if pad:
        keys = torch.cat([keys, keys.new_zeros(pad, idx_dim)], dim=0)
    blocks = keys.reshape(n_blocks, r, idx_dim).sum(dim=1)
    counts = keys.new_full((n_blocks, 1), float(r))
    if pad:
        counts[-1] = r - pad
    return blocks / counts


def block_scores(q: torch.Tensor, pooled: torch.Tensor) -> torch.Tensor:
    """Score every block for every query.

    ``q`` is ``[n_tokens, n_heads, idx_dim]`` and ``pooled`` ``[n_blocks, idx_dim]``.
    Returns ``[n_tokens, n_blocks]``.

    The ReLU lands on each head's dot product **before** the heads are summed. Rectifying
    after the sum is a different function and would silently reweight which blocks win.
    """
    per_head = torch.einsum("thd,bd->tbh", q, pooled)
    return torch.relu(per_head).sum(dim=-1)


def expand_to_tokens(scores: torch.Tensor, cell_block: torch.Tensor) -> torch.Tensor:
    """Give every key its block's score. ``cell_block`` is ``[n_kv]`` block ids."""
    return scores.index_select(1, cell_block)


def select_topk(
    token_scores: torch.Tensor,
    *,
    top_k: int,
    compress_ratio: int,
    causal_len: torch.Tensor | None = None,
) -> torch.Tensor:
    """Boolean keep-mask ``[n_tokens, n_kv]`` over the selected keys.

    The budget is ``top_k + compress_ratio - 1`` because selection cuts on block
    boundaries: whole blocks, plus the incomplete tail. When the budget covers everything
    a query may legally see, the mask is all-True and attention degenerates to dense --
    which is what makes a short-context run exactly equivalent, and is the property worth
    testing against.

    ``causal_len[i]`` is how many keys query ``i`` may attend to; keys at or beyond it are
    never selected.
    """
    n_tokens, n_kv = token_scores.shape
    width = min(n_kv, top_k + compress_ratio - 1)

    scores = token_scores
    if causal_len is not None:
        pos = torch.arange(n_kv, device=scores.device).unsqueeze(0)
        allowed = pos < causal_len.unsqueeze(1)
        scores = scores.masked_fill(~allowed, float("-inf"))

    idx = scores.topk(width, dim=-1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    if causal_len is not None:
        # topk still returns indices when fewer than `width` entries are finite.
        mask &= allowed
    return mask
