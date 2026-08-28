"""Device-side key selection with static shapes, for CUDA graph capture.

The host version loops the requests and reads their lengths with ``.tolist()``, which a
captured graph cannot contain. This does the same selection entirely from device tensors,
and -- the part that actually matters -- without any data-dependent allocation. Boolean
mask indexing (``indices[keep]``) is the trap: it is device-only and sync-free, but it
allocates according to how many entries are True, so a captured graph cannot hold it
either. Every buffer here is sized from ``bs``, ``capacity`` and the input widths.

Each cell computes its own destination slot instead. Cells that are not kept are scattered
to a single trailing scratch slot that is then dropped, which keeps the write static
without branching.

Selection itself: keep the highest-scoring whole blocks that fit the budget, plus the
ragged tail, which is never scored. A request already under the budget keeps everything --
that falls out of the arithmetic rather than a branch, and it is the property that makes a
short-context run identical with selection on.
"""

from __future__ import annotations

import torch


def select_cells_device(
    indptr: torch.Tensor,
    indices: torch.Tensor,
    scores: torch.Tensor,
    n_blocks: torch.Tensor,
    *,
    compress_ratio: int,
    budget: int,
    capacity: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(out_indptr [bs+1], out_indices [bs*capacity])``, both static.

    Only the VALUES in ``out_indptr`` vary with the data; the kernel reads per-request
    counts from it, so a request may keep fewer cells than ``capacity`` without the buffer
    changing shape.
    """
    bs = indptr.numel() - 1
    device, idt = indptr.device, indptr.dtype
    total, max_blocks = indices.numel(), scores.shape[1]

    lens = indptr[1:] - indptr[:-1]
    tail = torch.clamp(lens - n_blocks * compress_ratio, min=0)
    # A request under the budget keeps all of its cells: cells_to_keep == lens, so
    # blocks_to_keep works out to n_blocks and every block is selected.
    cells_to_keep = torch.minimum(lens, torch.full_like(lens, budget))
    blocks_to_keep = torch.minimum(
        torch.clamp((cells_to_keep - tail) // compress_ratio, min=0), n_blocks
    )

    # Rank blocks by score, with out-of-range blocks pushed to the bottom. Two argsorts
    # give each block its rank without a scatter.
    valid = torch.arange(max_blocks, device=device).unsqueeze(0) < n_blocks.unsqueeze(1)
    masked = torch.where(valid, scores, torch.full_like(scores, float("-inf")))
    ranks = torch.argsort(torch.argsort(masked, dim=1, descending=True), dim=1)
    block_selected = ranks < blocks_to_keep.unsqueeze(1)

    cell = torch.arange(total, device=device, dtype=idt)
    # right=True on the request ENDS: a cell exactly at an offset belongs to the next one.
    req = torch.searchsorted(indptr[1:].contiguous(), cell, right=True).clamp(max=bs - 1)
    pos = cell - indptr[req]
    # A tail cell's nominal block is past the last real one. It is kept via `in_tail`, but
    # the gather still has to stay in range.
    blk = torch.clamp(pos // compress_ratio, max=max_blocks - 1)

    in_tail = pos >= (n_blocks[req] * compress_ratio)
    keep = in_tail | block_selected[req.long(), blk.long()]

    # Per-request kept counts, and each kept cell's rank within its request. Both from
    # prefix sums, so no segment loop and no dynamic shape.
    k = keep.to(torch.int32)
    counts = torch.zeros(bs, device=device, dtype=torch.int32).index_add_(0, req.long(), k)
    out_indptr = torch.cat(
        [torch.zeros(1, device=device, dtype=idt), torch.cumsum(counts, 0).to(idt)]
    )
    prefix = torch.cumsum(k, 0) - k                       # exclusive prefix over all cells
    rank_in_req = prefix - prefix[indptr[:-1].long()][req.long()]
    dest = out_indptr[req.long()] + rank_in_req

    # Unkept cells go to one scratch slot past the end, which is then dropped. This keeps
    # the scatter unconditional and the buffer a fixed size.
    scratch = bs * capacity
    dest = torch.where(keep, dest.to(torch.int64), torch.full_like(dest, scratch, dtype=torch.int64))
    out = torch.zeros(scratch + 1, device=device, dtype=idt)
    out.scatter_(0, dest, indices.to(idt))
    return out_indptr, out[:scratch]
