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

Off unless ``FREETOKEN_QWEN4EXP_INDEXER=1``, and it currently requires
``--cuda-graph-max-bs 0``. The selection reads per-request lengths on the host and builds
the filtered list there, which a captured graph cannot contain. Making it capturable means
doing the whole selection on device with static shapes -- MiniMax-M3's block-sparse backend
shows the shape of that (per-request block rows and live lengths staged into persistent
buffers, device-read lengths, shape-fixed grids) and is the template for finishing it.

That constraint matters for whether this is worth switching on: CUDA graphs are a measured
2.35x here, so until the selection is capturable the indexer costs more than it saves at
short context. It earns its keep only above ``indexer_top_k``, where dense attention stops
matching what the model was trained on.

The scoring is verified against the reference (``indexer.py``,
``tests/models/test_qwen4exp_indexer.py``); the end-to-end long-context path is not yet
validated, which is why this is opt-in.
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

from .indexer import block_positions, block_scores, pool_blocks

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
                "qwen4exp: lightning indexer ENABLED. Requires --cuda-graph-max-bs 0: the "
                "selection runs on the host and cannot live inside a captured graph."
            )

        # BF16 dense in the checkpoint (indexer.q_proj / k_proj), not block-quantised.
        self.index_q_proj = LinearReplicated(
            config.hidden_size, self.idx_heads * self.idx_dim, has_bias=False
        )
        self.index_k_proj = LinearReplicated(config.hidden_size, self.idx_dim, has_bias=False)
        self.index_q_norm = GemmaRMSNorm(self.idx_dim, eps=config.rms_norm_eps)
        self.index_k_norm = GemmaRMSNorm(self.idx_dim, eps=config.rms_norm_eps)
        # Raw keys, one per KV cell. Bound by the model once the cache geometry is known.
        self.index_k_cache: torch.Tensor | None = None

    def bind_index_cache(self, num_cells: int, device, dtype) -> None:
        self.index_k_cache = torch.zeros(num_cells, self.idx_dim, device=device, dtype=dtype)

    def _select(self, index_q: torch.Tensor, batch) -> AttentionSpec | None:
        """Filtered ``(indptr, indices)`` for this decode step, or None to stay dense."""
        md = batch.attn_metadata
        indptr, indices = md.indptr, md.indices
        bs = indptr.numel() - 1
        lens = (indptr[1:] - indptr[:-1]).tolist()
        budget = self.idx_top_k + self.compress_ratio - 1
        if min(lens) < budget:
            # Under the budget somewhere: selection would be the identity for that request
            # and the widths would differ across the batch. Dense is exact here.
            return None

        keep_ptr = [0]
        keep = []
        for i, n_kv in enumerate(lens):
            cells = indices[indptr[i] : indptr[i] + n_kv]
            n_blocks = n_kv // self.compress_ratio
            tail = cells[n_blocks * self.compress_ratio :]

            raw = self.index_k_cache.index_select(0, cells[: n_blocks * self.compress_ratio].long())
            pooled = pool_blocks(raw, self.compress_ratio)
            pos = block_positions(n_blocks, self.compress_ratio, device=pooled.device)
            pooled = self.index_k_norm.forward(pooled)
            pooled, _ = self.rotary.forward(pos.to(torch.int32), pooled, pooled.clone())

            scores = block_scores(index_q[i : i + 1], pooled)[0]      # [n_blocks]
            take = min(n_blocks, (budget - tail.numel()) // self.compress_ratio)
            top = scores.topk(take).indices
            # every cell of a chosen block, plus the ragged tail which is never scored
            offs = torch.arange(self.compress_ratio, device=cells.device)
            chosen = (top.unsqueeze(1) * self.compress_ratio + offs).reshape(-1)
            keep.append(torch.cat([cells.index_select(0, chosen), tail]))
            keep_ptr.append(keep_ptr[-1] + keep[-1].numel())

        return AttentionSpec(
            kv_indptr=torch.tensor(keep_ptr, device=indices.device, dtype=torch.int32),
            kv_indices=torch.cat(keep).to(torch.int32),
        )

    @nvtx_annotate("MHA_QSA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        q, k, v, gate = self._project(x)

        spec = None
        if self._enabled and self.index_k_cache is not None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "qwen4exp lightning indexer cannot be CUDA-graph captured: its "
                    "selection reads per-request lengths on the host. Serve with "
                    "--cuda-graph-max-bs 0, or finish the device-side selection "
                    "(see attention/m3_sparse.py for the capturable pattern)."
                )
            # Raw: pooling precedes norm and rope, so neither is applied before caching.
            raw_k = self.index_k_proj.forward(x)
            self.index_k_cache[ctx.batch.out_loc] = raw_k.to(self.index_k_cache.dtype)
            if ctx.batch.is_decode:
                iq = self.index_q_proj.forward(x).view(-1, self.idx_heads, self.idx_dim)
                iq = self.index_q_norm.forward(iq)
                flat = iq.reshape(-1, self.idx_heads * self.idx_dim)
                flat, _ = self.rotary.forward(ctx.batch.positions, flat, flat.clone())
                spec = self._select(flat.view(-1, self.idx_heads, self.idx_dim), ctx.batch)

        o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch, spec)
        return self._combine(o, gate)


__all__ = ["Qwen4ExpAttention", "indexer_enabled"]
