"""qwen4exp lightning indexer: block scoring and selection.

The oracle is a loop-for-loop NumPy transcription of llama.cpp ``build_qsa_top_k``, written
from the reference rather than from the torch version, so agreement means two independent
readings agree.

The property that matters most is the last one: while the budget covers every key a query
may legally see, the selection must be everything. That is what makes dense attention
*exactly* equivalent below ``indexer_top_k``, and it gives a rigorous way to check the
indexer against the model that already works -- a short-context run must not change at all
when selection is switched on.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from freetoken.models.qwen4exp.indexer import (
    block_positions,
    block_scores,
    expand_to_tokens,
    pool_blocks,
    select_topk,
)

IDX_DIM, HEADS, RATIO = 8, 4, 4


def _rand(*shape, seed=0):
    return torch.from_numpy(np.random.default_rng(seed).standard_normal(shape))


def ref_pool(keys, r):
    """Mean over each COMPLETE block's members. An incomplete block is not pooled at all --
    llama.cpp forces its cells in via the bias instead."""
    n_kv, d = keys.shape
    n_blocks = n_kv // r
    out = np.zeros((n_blocks, d))
    for b in range(n_blocks):
        out[b] = keys[b * r : (b + 1) * r].mean(axis=0)
    return out


def ref_scores(q, pooled):
    """relu per head, then sum over heads -- the order is the point."""
    T, H, D = q.shape
    B = pooled.shape[0]
    out = np.zeros((T, B))
    for t in range(T):
        for b in range(B):
            acc = 0.0
            for h in range(H):
                acc += max(0.0, float(q[t, h] @ pooled[b]))
            out[t, b] = acc
    return out


def test_pooling_matches_the_reference():
    keys = _rand(13, IDX_DIM, seed=1)          # 13 cells -> 3 whole blocks + a 1-cell tail
    got = pool_blocks(keys, RATIO)
    want = ref_pool(keys.numpy(), RATIO)
    assert got.shape == (3, IDX_DIM), "the ragged tail is not a block"
    np.testing.assert_allclose(got.numpy(), want, rtol=1e-12, atol=1e-12)


def test_incomplete_trailing_block_is_not_pooled():
    """llama.cpp: "an incomplete block cannot be pooled; the bias forces those tail cells
    in". Pooling it would invent a score for the newest tokens and let them lose the
    top-k -- the opposite of what the model expects."""
    assert pool_blocks(_rand(13, IDX_DIM), RATIO).shape[0] == 3
    assert pool_blocks(_rand(3, IDX_DIM), RATIO).shape[0] == 0


def test_the_ragged_tail_is_always_selected():
    """Its cells are the most recent tokens and are never scored, so they must be kept
    whatever the block scores say."""
    scores = torch.full((2, 13), -1e9, dtype=torch.float64)
    mask = select_topk(scores, top_k=1, compress_ratio=RATIO)
    assert mask[:, 12].all(), "the 1-cell tail must be forced in"


def test_block_positions_are_the_first_token():
    """"block b covers [b*ratio, (b+1)*ratio), so its first token is at b*ratio". Using the
    last member would rotate every pooled key up to ratio-1 positions too far."""
    np.testing.assert_array_equal(
        block_positions(4, RATIO).numpy(), np.array([0, 4, 8, 12])
    )


def test_scores_match_the_reference():
    q = _rand(6, HEADS, IDX_DIM, seed=2)
    pooled = _rand(5, IDX_DIM, seed=3)
    got = block_scores(q, pooled)
    want = ref_scores(q.numpy(), pooled.numpy())
    np.testing.assert_allclose(got.numpy(), want, rtol=1e-12, atol=1e-12)


def test_relu_before_the_sum_is_not_the_same_as_after():
    """Guards the ordering the DeepSeek lightning indexer specifies."""
    q = _rand(4, HEADS, IDX_DIM, seed=4)
    pooled = _rand(3, IDX_DIM, seed=5)
    before = block_scores(q, pooled)
    per_head = torch.einsum("thd,bd->tbh", q, pooled)
    after = torch.relu(per_head.sum(dim=-1))
    assert not torch.allclose(before, after)


def test_every_token_of_a_block_gets_its_block_score():
    scores = torch.tensor([[10.0, 20.0, 30.0]], dtype=torch.float64)
    cell_block = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    got = expand_to_tokens(scores, cell_block)
    np.testing.assert_array_equal(
        got.numpy()[0], np.array([10, 10, 10, 10, 20, 20, 20, 20, 30, 30], dtype=float)
    )


def test_budget_includes_the_block_tail():
    """min(n_kv, top_k + r - 1): whole blocks plus the incomplete tail."""
    scores = _rand(2, 64, seed=6)
    mask = select_topk(scores, top_k=8, compress_ratio=RATIO)
    assert int(mask[0].sum()) == 8 + RATIO - 1
    # and it never exceeds what exists
    small = select_topk(_rand(2, 5, seed=7), top_k=8, compress_ratio=RATIO)
    assert int(small[0].sum()) == 5


def test_selection_is_everything_below_the_budget():
    """The property that makes dense attention exactly equivalent under indexer_top_k.

    A short-context run must be unchanged when selection is switched on, which is the
    strongest available check against the model that already generates correctly.
    """
    n_kv = 40
    scores = _rand(6, n_kv, seed=8)
    mask = select_topk(scores, top_k=2048, compress_ratio=RATIO)
    assert mask.all(), "with a budget above n_kv nothing may be dropped"


def test_causality_is_respected_and_never_selected():
    n_kv = 20
    scores = _rand(4, n_kv, seed=9)
    lens = torch.tensor([1, 5, 12, 20])
    mask = select_topk(scores, top_k=4, compress_ratio=RATIO, causal_len=lens)
    for i, n in enumerate(lens.tolist()):
        assert not mask[i, n:].any(), f"query {i} selected a future key"
        assert mask[i, :n].sum() == min(n, 4 + RATIO - 1)


def test_causal_short_prefix_selects_all_it_may_see():
    """A query that can legally see fewer keys than the budget keeps all of them."""
    scores = _rand(1, 20, seed=10)
    mask = select_topk(scores, top_k=8, compress_ratio=RATIO,
                       causal_len=torch.tensor([3]))
    assert mask[0, :3].all() and not mask[0, 3:].any()


@pytest.mark.parametrize("n_kv", [1, 4, 5, 16, 17])
def test_block_count_and_mask_width_are_consistent(n_kv):
    keys = _rand(n_kv, IDX_DIM, seed=11)
    pooled = pool_blocks(keys, RATIO)
    n_blocks = n_kv // RATIO
    assert pooled.shape[0] == n_blocks
    if n_blocks:
        q = _rand(2, HEADS, IDX_DIM, seed=12)
        # only cells inside whole blocks carry a block score; the tail is forced in
        cell_block = torch.arange(n_blocks * RATIO) // RATIO
        tok = expand_to_tokens(block_scores(q, pooled), cell_block)
        assert tok.shape == (2, n_blocks * RATIO)
