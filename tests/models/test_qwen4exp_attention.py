"""qwen4exp gated attention: the query/gate interleave.

The whole point of these tests is one detail. ``attn_q`` is twice as wide as the head
geometry calls for because each head's queries are followed by that head's gate, and the
layout interleaves per head rather than splitting into two contiguous halves. Both
readings produce identically shaped tensors, so nothing structural distinguishes them --
the wrong one simply pairs every head with another head's gate and degrades output.

The oracle is a strided read written straight from llama.cpp's two ``ggml_view_3d`` calls
(stride ``2 * head_dim``, offsets 0 and ``head_dim``).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from freetoken.models.qwen4exp.attention import (
    apply_output_gate,
    rms_norm_heads,
    split_q_and_gate,
)

HEADS = 4
HEAD_DIM = 6
T = 3


def ref_views(q_full: np.ndarray):
    """Literal transcription of the ggml views.

    ggml_view_3d(Qcur_full, head_dim, n_head, n_tokens, nb1 = elt*head_dim*2, offset)
    with offset 0 for the queries and elt*head_dim for the gate.
    """
    q = np.empty((T, HEADS, HEAD_DIM))
    g = np.empty((T, HEADS, HEAD_DIM))
    for t in range(T):
        for h in range(HEADS):
            base = h * HEAD_DIM * 2
            q[t, h] = q_full[t, base : base + HEAD_DIM]
            g[t, h] = q_full[t, base + HEAD_DIM : base + 2 * HEAD_DIM]
    return q, g


def test_split_matches_the_strided_views():
    rng = np.random.default_rng(0)
    q_full = rng.standard_normal((T, 2 * HEADS * HEAD_DIM))
    want_q, want_g = ref_views(q_full)
    got_q, got_g = split_q_and_gate(torch.from_numpy(q_full), HEADS, HEAD_DIM)
    np.testing.assert_allclose(got_q.numpy(), want_q, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got_g.numpy(), want_g, rtol=1e-12, atol=1e-12)


def test_interleaved_is_not_the_same_as_split_in_half():
    """The check that this test file exists for.

    A half-split reads the first n_head*head_dim values as queries. Under the real
    interleave those values are heads 0..n/2 alternating query and gate blocks, so the
    two disagree for every head but the first.
    """
    q_full = torch.arange(2 * HEADS * HEAD_DIM, dtype=torch.float64).reshape(1, -1)
    got_q, got_g = split_q_and_gate(q_full, HEADS, HEAD_DIM)

    half_q = q_full[:, : HEADS * HEAD_DIM].reshape(1, HEADS, HEAD_DIM)
    half_g = q_full[:, HEADS * HEAD_DIM :].reshape(1, HEADS, HEAD_DIM)

    # head 0's queries happen to coincide -- both start at 0 -- and nothing else does
    np.testing.assert_array_equal(got_q[:, 0].numpy(), half_q[:, 0].numpy())
    assert not torch.equal(got_q, half_q)
    assert not torch.equal(got_g, half_g)


def test_each_head_gets_its_own_gate():
    """Zero one head's gate slot; only that head's gate may change."""
    q_full = torch.ones(1, 2 * HEADS * HEAD_DIM, dtype=torch.float64)
    target = 2
    base = target * HEAD_DIM * 2 + HEAD_DIM
    q_full[0, base : base + HEAD_DIM] = 0.0

    q, gate = split_q_and_gate(q_full, HEADS, HEAD_DIM)
    assert torch.all(q == 1.0), "queries must be untouched"
    for h in range(HEADS):
        expected = 0.0 if h == target else 1.0
        assert torch.all(gate[0, h] == expected), f"head {h} gate"


def test_width_is_validated():
    with pytest.raises(ValueError, match="should be 48 wide"):
        split_q_and_gate(torch.zeros(T, 47), HEADS, HEAD_DIM)


def test_output_gate_is_sigmoid_and_elementwise():
    rng = np.random.default_rng(1)
    out = rng.standard_normal((T, HEADS, HEAD_DIM))
    gate = rng.standard_normal((T, HEADS, HEAD_DIM))
    want = out * (1.0 / (1.0 + np.exp(-gate)))
    got = apply_output_gate(torch.from_numpy(out), torch.from_numpy(gate))
    np.testing.assert_allclose(got.numpy(), want, rtol=1e-12, atol=1e-12)


def test_output_gate_rejects_a_shape_mismatch():
    with pytest.raises(ValueError, match="must have the same shape"):
        apply_output_gate(torch.zeros(T, HEADS, HEAD_DIM), torch.zeros(T, HEADS, HEAD_DIM + 1))


def test_head_norm_is_per_head_over_head_dim():
    """Scaling one head must not move any other -- the norm does not span heads."""
    rng = np.random.default_rng(2)
    x = torch.from_numpy(rng.standard_normal((T, HEADS, HEAD_DIM)))
    w = torch.ones(HEAD_DIM, dtype=torch.float64)

    base = rms_norm_heads(x, w, 1e-6)
    scaled_in = x.clone()
    scaled_in[:, 1, :] *= 25.0
    scaled = rms_norm_heads(scaled_in, w, 1e-6)

    np.testing.assert_allclose(
        scaled[:, [0, 2, 3]].numpy(), base[:, [0, 2, 3]].numpy(), rtol=1e-12, atol=1e-12
    )
    # the scaled head renormalises back to where it started
    np.testing.assert_allclose(scaled[:, 1].numpy(), base[:, 1].numpy(), rtol=1e-6, atol=1e-6)


def test_head_norm_applies_the_shared_gamma():
    x = torch.ones(1, HEADS, HEAD_DIM, dtype=torch.float64)
    w = torch.arange(1, HEAD_DIM + 1, dtype=torch.float64)
    got = rms_norm_heads(x, w, 1e-12)
    # an all-ones head normalises to ones, so the result is just gamma, per head
    for h in range(HEADS):
        np.testing.assert_allclose(got[0, h].numpy(), w.numpy(), rtol=1e-9, atol=1e-9)
