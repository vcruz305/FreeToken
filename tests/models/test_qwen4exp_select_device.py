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


@pytest.mark.parametrize(
    "lens,budget",
    [([12, 8], 64), ([13], 8), ([40], 12), ([40, 24, 13], 12), ([4], 8), ([7], 4),
     ([2048, 2048], 260), ([100, 5, 63], 20)],
)
def test_device_matches_reference(lens, budget):
    """The acceptance test: same selection as the host loop, on every shape."""
    from freetoken.models.qwen4exp.select_device import select_cells_device

    indptr, indices, scores, n_blocks = _case(lens, seed=len(lens) + budget)
    want = ref_select(indptr, indices, scores, n_blocks,
                      compress_ratio=RATIO, budget=budget)
    cap = max(budget, max(lens))
    got_ptr, got_idx = select_cells_device(
        indptr, indices, scores, n_blocks,
        compress_ratio=RATIO, budget=budget, capacity=cap,
    )
    for i in range(len(lens)):
        g = got_idx[int(got_ptr[i]) : int(got_ptr[i + 1])]
        torch.testing.assert_close(g.sort().values, want[i].sort().values)


def test_output_shapes_do_not_depend_on_the_data():
    """The whole point: a captured graph cannot hold a data-dependent allocation.

    Two very different selections must produce identically shaped buffers.
    """
    from freetoken.models.qwen4exp.select_device import select_cells_device

    a = _case([40, 40], seed=1)
    b = _case([40, 40], seed=2)
    shapes = []
    for indptr, indices, scores, n_blocks in (a, b):
        ptr, idx = select_cells_device(indptr, indices, scores, n_blocks,
                                       compress_ratio=RATIO, budget=12, capacity=40)
        shapes.append((tuple(ptr.shape), tuple(idx.shape)))
    assert shapes[0] == shapes[1] == ((3,), (80,))


def test_unkept_cells_are_not_addressable():
    """Cells outside a request's indptr span must never be attended, whatever is in the
    padding."""
    from freetoken.models.qwen4exp.select_device import select_cells_device

    indptr, indices, scores, n_blocks = _case([40], seed=3)
    ptr, idx = select_cells_device(indptr, indices, scores, n_blocks,
                                   compress_ratio=RATIO, budget=12, capacity=40)
    kept = idx[int(ptr[0]) : int(ptr[1])]
    assert kept.numel() <= 12
    assert kept.unique().numel() == kept.numel(), "a cell must not be attended twice"
