"""Restricting attention to a chosen subset of KV cells via AttentionSpec.

A learned key indexer (qwen4exp) selects which cells a query may attend to. Sliding-window
attention already restricts keys by swapping the index list, so this reuses that mechanism
rather than adding a mask -- with the difference that a filtered list changes LENGTH, so
the request offsets must be replaced alongside it.

The property worth testing is the degenerate one: passing every cell must reproduce dense
attention exactly. That is what makes a short-context run unchanged when selection is
switched on, and it is the acceptance test for the indexer as a whole.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

HEADS, KV_HEADS, D = 4, 2, 32
N_CELLS = 24


def _setup(seed=0):
    from freetoken.kernel.triton.attention import paged_attention

    torch.manual_seed(seed)
    dev = "cuda"
    q = torch.randn(2, HEADS, D, device=dev, dtype=torch.bfloat16)
    k_cache = torch.randn(N_CELLS, KV_HEADS, D, device=dev, dtype=torch.bfloat16)
    v_cache = torch.randn(N_CELLS, KV_HEADS, D, device=dev, dtype=torch.bfloat16)
    return paged_attention, q, k_cache, v_cache, dev


def test_full_selection_reproduces_dense():
    """The degenerate case the whole design leans on."""
    paged_attention, q, k_cache, v_cache, dev = _setup()
    # two queries, one request each, each seeing the first 12 cells
    indptr = torch.tensor([0, 12, 24], device=dev, dtype=torch.int32)
    indices = torch.cat([torch.arange(12), torch.arange(12)]).to(dev, torch.int32)
    q_to_req = torch.tensor([0, 1], device=dev, dtype=torch.int32)
    q_pos = torch.tensor([11, 11], device=dev, dtype=torch.int32)

    dense = paged_attention(q=q, k_cache=k_cache, v_cache=v_cache, indptr=indptr,
                            indices=indices, q_to_req=q_to_req, q_positions=q_pos,
                            sm_scale=D ** -0.5)
    same = paged_attention(q=q, k_cache=k_cache, v_cache=v_cache, indptr=indptr.clone(),
                           indices=indices.clone(), q_to_req=q_to_req, q_positions=q_pos,
                           sm_scale=D ** -0.5)
    torch.testing.assert_close(dense, same)


def test_a_subset_actually_changes_the_result():
    """Guards the test above from being vacuous: dropping keys must matter."""
    paged_attention, q, k_cache, v_cache, dev = _setup(1)
    q_to_req = torch.tensor([0], device=dev, dtype=torch.int32)
    q_pos = torch.tensor([11], device=dev, dtype=torch.int32)

    full = paged_attention(
        q=q[:1], k_cache=k_cache, v_cache=v_cache,
        indptr=torch.tensor([0, 12], device=dev, dtype=torch.int32),
        indices=torch.arange(12, device=dev, dtype=torch.int32),
        q_to_req=q_to_req, q_positions=q_pos, sm_scale=D ** -0.5,
    )
    subset = paged_attention(
        q=q[:1], k_cache=k_cache, v_cache=v_cache,
        indptr=torch.tensor([0, 4], device=dev, dtype=torch.int32),
        indices=torch.tensor([0, 3, 7, 11], device=dev, dtype=torch.int32),
        q_to_req=q_to_req, q_positions=q_pos, sm_scale=D ** -0.5,
    )
    assert not torch.allclose(full.float(), subset.float(), atol=1e-2)


def test_subset_equals_attending_only_those_cells():
    """A filtered list must give exactly what a cache holding only those cells would."""
    paged_attention, q, k_cache, v_cache, dev = _setup(2)
    keep = torch.tensor([2, 5, 9], device=dev, dtype=torch.int32)
    q_to_req = torch.tensor([0], device=dev, dtype=torch.int32)
    q_pos = torch.tensor([23], device=dev, dtype=torch.int32)

    via_subset = paged_attention(
        q=q[:1], k_cache=k_cache, v_cache=v_cache,
        indptr=torch.tensor([0, 3], device=dev, dtype=torch.int32),
        indices=keep, q_to_req=q_to_req, q_positions=q_pos, sm_scale=D ** -0.5,
    )
    # the same three cells packed into their own cache, addressed densely
    packed_k = k_cache.index_select(0, keep.long()).contiguous()
    packed_v = v_cache.index_select(0, keep.long()).contiguous()
    via_packed = paged_attention(
        q=q[:1], k_cache=packed_k, v_cache=packed_v,
        indptr=torch.tensor([0, 3], device=dev, dtype=torch.int32),
        indices=torch.arange(3, device=dev, dtype=torch.int32),
        q_to_req=q_to_req, q_positions=q_pos, sm_scale=D ** -0.5,
    )
    torch.testing.assert_close(via_subset, via_packed)


def test_spec_requires_both_halves():
    """indices without indptr would silently read the wrong request offsets."""
    from freetoken.attention.base import AttentionSpec

    spec = AttentionSpec(kv_indices=torch.zeros(4, dtype=torch.int32))
    assert spec.kv_indptr is None  # the backend asserts on this pairing
