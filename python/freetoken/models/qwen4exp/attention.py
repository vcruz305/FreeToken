"""qwen4exp full attention with the lightning indexer.

Extends the qwen35moe gated attention (which qwen4exp shares verbatim) with the key
selection the model was trained under: score pooled blocks of ``compress_ratio`` cells with
a small indexer and attend only the top ``indexer_top_k``.

Two properties make this safe to add to a working model:

* **Below the budget it is a no-op.** Selection engages only once a request holds at least
  ``top_k + compress_ratio - 1`` cells; under that, top-k covers everything a query may see
  and dense attention is exactly equivalent. So a short-context run is bit-identical with
  selection on, which is the acceptance test.
* **Above the budget the width is constant.** ``min(n_kv, top_k + r - 1)`` stops varying
  once ``n_kv`` passes the budget, so the filtered index list has a fixed shape and CUDA
  graph capture is unaffected.

Selection is decode-only: ``indptr`` is per-request, so it cannot express a different key
set for each query of a multi-query request. Prefill stays dense, which is exact below the
budget anyway.

Selection runs entirely on device with static shapes (``select_device.py``), so CUDA
graphs are unaffected -- which matters, since graphs are a measured 2.35x here.

Off unless ``FREETOKEN_QWEN4EXP_INDEXER=1``. The scoring and the selection are each
verified against the reference (``tests/models/test_qwen4exp_indexer.py``,
``test_qwen4exp_select_device.py``), but the end-to-end path has not been checked above
``indexer_top_k`` on real weights, which is the only regime where it changes anything.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from freetoken.attention.base import AttentionSpec
from freetoken.core import get_global_ctx
from freetoken.layers import GemmaRMSNorm, LinearReplicated
from freetoken.models.qwen3_5_moe.attention import Qwen3_5Attention
from freetoken.utils import nvtx_annotate

from .indexer import block_positions
from .select_device import select_cells_device

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


def indexer_enabled() -> bool:
    return os.environ.get("FREETOKEN_QWEN4EXP_INDEXER", "0") == "1"


class Qwen4ExpAttention(Qwen3_5Attention):
    """Gated attention plus the lightning indexer's key selection."""

    def __init__(self, config: "ModelConfig", layer_id: int, geo: dict):
        super().__init__(config, layer_id)
        self.idx_heads = geo["indexer_heads"]
        self.idx_dim = geo["indexer_key_length"]
        self.idx_top_k = geo["indexer_top_k"]
        self.compress_ratio = geo["full_attention_interval"]
        self._enabled = indexer_enabled()
        if self._enabled and layer_id == 0:
            from freetoken.utils import init_logger

            init_logger(__name__).info_rank0(
                "qwen4exp: lightning indexer ENABLED (device-side selection, CUDA-graph "
                "safe). Below indexer_top_k it selects every key, so short contexts are "
                "unchanged."
            )

        # BF16 dense in the checkpoint (indexer.q_proj / k_proj), not block-quantised.
        self.index_q_proj = LinearReplicated(
            config.hidden_size, self.idx_heads * self.idx_dim, has_bias=False
        )
        self.index_k_proj = LinearReplicated(config.hidden_size, self.idx_dim, has_bias=False)
        self.index_q_norm = GemmaRMSNorm(self.idx_dim, eps=config.rms_norm_eps)
        self.index_k_norm = GemmaRMSNorm(self.idx_dim, eps=config.rms_norm_eps)
        # The indexer's own rope. self.rotary is built for the attention head_dim (256);
        # indexer vectors are indexer_key_length (128) wide, and reusing the attention
        # instance reshapes them by the wrong head size.
        from freetoken.layers.rotary import get_rope

        self.index_rotary = get_rope(
            head_dim=self.idx_dim,
            rotary_dim=min(config.rotary_config.rotary_dim, self.idx_dim),
            max_position=config.rotary_config.max_position,
            base=config.rotary_config.base,
            rope_scaling=None,
        )
        # Raw keys, one per KV cell. Bound by the model once the cache geometry is known.
        self.index_k_cache: torch.Tensor | None = None

    def bind_index_cache(self, num_cells: int, device, dtype) -> None:
        self.index_k_cache = torch.zeros(num_cells, self.idx_dim, device=device, dtype=dtype)

    def _select(self, index_q: torch.Tensor, batch) -> AttentionSpec | None:
        """Filtered ``(indptr, indices)`` for this decode step.

        Fully vectorised over the batch and free of data-dependent shapes, so it can run
        inside a captured graph. Requests are padded to one common width; padding is masked
        out of the block scores and never selected.
        """
        md = batch.attn_metadata
        indptr, indices = md.indptr, md.indices
        bs = indptr.numel() - 1
        r = self.compress_ratio

        lens = (indptr[1:] - indptr[:-1]).to(torch.int32)
        width = int(indices.numel() // max(bs, 1))
        width = (width // r) * r
        if width < r:
            return None

        # [bs, width] cells, padded. Out-of-range reads are clamped and then masked.
        pos = torch.arange(width, device=indices.device)
        flat = (indptr[:-1].unsqueeze(1) + pos.unsqueeze(0)).clamp(max=indices.numel() - 1)
        cells = indices.reshape(-1)[flat.long()]                      # [bs, width]
        live = pos.unsqueeze(0) < lens.unsqueeze(1)

        n_blocks = (lens // r).to(torch.int32)
        max_blocks = width // r

        raw = self.index_k_cache.index_select(0, cells.reshape(-1).long())
        raw = raw.view(bs, max_blocks, r, self.idx_dim)
        pooled = raw.mean(dim=2)                                      # [bs, max_blocks, d]

        blk_pos = block_positions(max_blocks, r, device=pooled.device).to(torch.int32)
        pooled = self.index_k_norm.forward(pooled)
        flat_p = pooled.reshape(bs * max_blocks, self.idx_dim)
        rot_pos = blk_pos.repeat(bs)
        flat_p, _ = self.index_rotary.forward(rot_pos, flat_p, flat_p.clone())
        pooled = flat_p.view(bs, max_blocks, self.idx_dim)

        # relu per head, then sum over heads -- the order the reference specifies.
        per_head = torch.einsum("bhd,bkd->bkh", index_q, pooled)
        scores = torch.relu(per_head).sum(dim=-1)                     # [bs, max_blocks]

        budget = self.idx_top_k + r - 1
        out_ptr, out_idx = select_cells_device(
            indptr.to(torch.int32), indices.to(torch.int32), scores, n_blocks,
            compress_ratio=r, budget=budget, capacity=width,
        )
        return AttentionSpec(kv_indptr=out_ptr, kv_indices=out_idx)

    @nvtx_annotate("MHA_QSA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        q, k, v, gate = self._project(x)

        spec = None
        if self._enabled and self.index_k_cache is not None:
            # Raw: pooling precedes norm and rope, so neither is applied before caching.
            raw_k = self.index_k_proj.forward(x)
            self.index_k_cache[ctx.batch.out_loc] = raw_k.to(self.index_k_cache.dtype)
            if ctx.batch.is_decode:
                iq = self.index_q_proj.forward(x).view(-1, self.idx_heads, self.idx_dim)
                iq = self.index_q_norm.forward(iq)
                flat = iq.reshape(-1, self.idx_heads * self.idx_dim)
                flat, _ = self.index_rotary.forward(
                    ctx.batch.positions, flat, flat.clone()
                )
                spec = self._select(flat.view(-1, self.idx_heads, self.idx_dim), ctx.batch)

        o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch, spec)
        return self._combine(o, gate)


__all__ = ["Qwen4ExpAttention", "indexer_enabled"]
