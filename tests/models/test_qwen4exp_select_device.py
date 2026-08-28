"""Device-side key selection must match the host-side loop it replaces.

The host version is simple and already exercised through the serving path: loop the
requests, take the top blocks, append the ragged tail. It cannot run inside a captured CUDA
graph because it reads per-request lengths on the host. The device version has to produce
identical selections with static output shapes.

The oracle here is that host loop, written plainly. Agreement is the whole test -- a device
version that silently keeps the wrong cells does not crash, it just attends to the wrong
part of the context.
"""

from __future__ import annotations

import pytest
import torch

RATIO = 4


def ref_select(indptr, indices, scores, n_blocks, *, compress_ratio, budget):
    """Host-side reference: per request, best whole blocks plus the ragged tail.

    Returns a list of 1-D tensors, one per request, in the order the cells are attended.
    """
    out = []
    bs = indptr.numel() - 1
    for i in range(bs):
        lo, hi = int(indptr[i]), int(indptr[i + 1])
        cells = indices[lo:hi]
        n = int(n_blocks[i])
        tail = cells[n * compress_ratio :]
        if cells.numel() <= budget:
            out.append(cells.clone())            # everything fits: keep it all, in order
            continue
        take = max(0, (budget - tail.numel()) // compress_ratio)
        take = min(take, n)
        top = torch.topk(scores[i, :n], take).indices.sort().values
        offs = torch.arange(compress_ratio)
        chosen = (top.unsqueeze(1) * compress_ratio + offs).reshape(-1)
        out.append(torch.cat([cells.index_select(0, chosen), tail]))
    return out


def _case(lens, seed=0):
    g = torch.Generator().manual_seed(seed)
    indptr = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32)
    indices = torch.arange(int(indptr[-1]), dtype=torch.int32)
    n_blocks = torch.tensor([n // RATIO for n in lens], dtype=torch.int32)
    max_blocks = max(1, int(n_blocks.max()))
    scores = torch.randn(len(lens), max_blocks, generator=g)
    return indptr, indices, scores, n_blocks


def test_reference_keeps_everything_under_budget():
    """The property the design rests on, asserted on the oracle itself."""
    indptr, indices, scores, n_blocks = _case([12, 8])
    got = ref_select(indptr, indices, scores, n_blocks,
                     compress_ratio=RATIO, budget=64)
    torch.testing.assert_close(got[0], indices[0:12])
    torch.testing.assert_close(got[1], indices[12:20])


def test_reference_keeps_the_ragged_tail():
    indptr, indices, scores, n_blocks = _case([13])
    got = ref_select(indptr, indices, scores, n_blocks,
                     compress_ratio=RATIO, budget=8)[0]
    assert indices[12] in got, "the 1-cell tail is never scored and must be kept"


def test_reference_respects_the_budget():
    indptr, indices, scores, n_blocks = _case([40])
    got = ref_select(indptr, indices, scores, n_blocks,
                     compress_ratio=RATIO, budget=12)[0]
    assert got.numel() <= 12


@pytest.mark.skipif(True, reason="device implementation not landed yet")
def test_device_matches_reference():
    """Enable once select_cells_device exists; this is its acceptance test."""
