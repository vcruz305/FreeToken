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
    """Mean-pool raw indexer keys into COMPLETE blocks.

    ``keys`` is ``[n_kv, idx_dim]`` in cell order. Returns ``[n_kv // r, idx_dim]``.

    Only whole blocks are pooled. llama.cpp is explicit that "an incomplete block cannot be
    pooled; the bias below forces those tail cells in" -- the ragged tail is never scored,
    it is always attended. Averaging a partial block instead would both invent a score for
    it and let it lose the top-k, dropping the most recent tokens, which is the opposite of
    what the model expects.
    """
    n_kv, idx_dim = keys.shape
    n_blocks = n_kv // compress_ratio
    if n_blocks == 0:
        return keys.new_zeros(0, idx_dim)
    whole = keys[: n_blocks * compress_ratio]
    return whole.reshape(n_blocks, compress_ratio, idx_dim).mean(dim=1)


def block_positions(n_blocks: int, compress_ratio: int, device=None) -> torch.Tensor:
    """Rope position of each pooled block: its FIRST token, ``b * compress_ratio``.

    From llama.cpp: "block b covers [b*ratio, (b+1)*ratio), so its first token is at
    b*ratio". Using the last member instead would rotate every pooled key by up to
    ``ratio - 1`` positions too far.
    """
    return torch.arange(n_blocks, device=device) * compress_ratio


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

    # Cells past the last whole block are the ragged tail: never scored, always attended.
    tail_start = (n_kv // compress_ratio) * compress_ratio

    scores = token_scores
    if causal_len is not None:
        pos = torch.arange(n_kv, device=scores.device).unsqueeze(0)
        allowed = pos < causal_len.unsqueeze(1)
        scores = scores.masked_fill(~allowed, float("-inf"))

    idx = scores.topk(width, dim=-1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    if tail_start < n_kv:
        mask[:, tail_start:] = True
    if causal_len is not None:
        # topk still returns indices when fewer than `width` entries are finite, and the
        # forced tail must not reach past what a query may legally see.
        mask &= allowed
    return mask
