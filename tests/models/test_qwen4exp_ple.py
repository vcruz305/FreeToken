"""qwen4exp PLE n-gram hashing.

The hash itself is short enough that a second transcription would just be the same code,
so these test its properties instead -- the ones that decide whether a token lands on the
right row out of 320 million, and that a shape check cannot see:

* uint64 wraparound. Python integers are unbounded, so the hash masks to 64 bits. Worth
  being exact about this: with this checkpoint the mask never fires, since the largest
  token id times the largest multiplier is ~5.9e18, under 2**63. It is covered anyway
  because the reference is uint64 and a wider vocabulary would overflow, and an unmasked
  hash would then silently pick different rows.
* the EOS reset, which stops the window reading across a document boundary.
* every head landing inside its own slice of the table.
"""

from __future__ import annotations

import numpy as np
import pytest

from freetoken.models.qwen4exp.ple import context_window, ngram_hash, ple_rows

# The real values from unsloth/Qwen3.8-Flash-Next-GGUF.
MULTIPLIERS = [23703573157769, 20109073645365, 8052911324071]
NGRAM = 3
PER_GRAM = 8
N_HEADS = (NGRAM - 1) * PER_GRAM
EOS = 248044

# The first 16 head ranges, which are the running sum of the vocab sizes.
VOCABS = [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
          20000081, 20000093, 20000107, 20000109, 20000113, 20000117, 20000129, 20000143]
OFFSETS = list(np.cumsum([0] + VOCABS[:-1]))


def _rows(tokens, preds):
    return ple_rows(
        tokens, preds,
        multipliers=MULTIPLIERS, head_offsets=OFFSETS, head_vocab_sizes=VOCABS,
        ngram_size=NGRAM, heads_per_ngram=PER_GRAM, eos_token_id=EOS,
    )


VOCAB_SIZE = 248320  # token_embd rows


def test_this_checkpoint_never_overflows_64_bits():
    """So the mask is defensive here, and this is the check that keeps that claim honest.

    If a future checkpoint raises the vocabulary or the multipliers past this bound, this
    fails and the masking stops being a formality.
    """
    worst = (VOCAB_SIZE - 1) * max(MULTIPLIERS)
    assert worst < 2**63, f"products reach {worst}, wraparound is now load-bearing"

    ctx = [VOCAB_SIZE - 1, VOCAB_SIZE - 2, VOCAB_SIZE - 3]
    unmasked = (ctx[0] * MULTIPLIERS[0]) ^ (ctx[1] * MULTIPLIERS[1]) ^ (ctx[2] * MULTIPLIERS[2])
    assert ngram_hash(ctx, MULTIPLIERS, 3) == unmasked


def test_mask_is_applied_when_it_does_matter():
    """With multipliers big enough to overflow, the mask changes the answer."""
    big = [m * 10**6 for m in MULTIPLIERS]
    ctx = [200000, 150000, 90000]
    unmasked = (ctx[0] * big[0]) ^ (ctx[1] * big[1]) ^ (ctx[2] * big[2])
    assert unmasked > 0xFFFFFFFFFFFFFFFF
    got = ngram_hash(ctx, big, 3)
    assert got <= 0xFFFFFFFFFFFFFFFF and got != unmasked


def test_every_head_lands_inside_its_own_range():
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 248320, 40).tolist()
    preds = [rng.integers(0, 248320, NGRAM - 1).tolist() for _ in tokens]
    rows = _rows(tokens, preds)
    assert rows.shape == (40, N_HEADS)
    for h in range(N_HEADS):
        lo, hi = OFFSETS[h], OFFSETS[h] + VOCABS[h]
        assert rows[:, h].min() >= lo and rows[:, h].max() < hi, f"head {h} out of range"


def test_heads_of_one_order_share_a_hash():
    """Within an n-gram order the heads differ only by ``% vocab + offset``."""
    rows = _rows([1234], [[10, 20]])[0]
    ctx = context_window(1234, [10, 20], EOS, NGRAM)
    for n, base in ((2, 0), (3, PER_GRAM)):
        mixed = ngram_hash(ctx, MULTIPLIERS, n)
        for g in range(PER_GRAM):
            h = base + g
            assert rows[h] == mixed % VOCABS[h] + OFFSETS[h]


def test_bigram_head_block_ignores_the_third_token():
    """Heads 0-7 hash n=2, so changing the token two back must not move them."""
    a = _rows([500], [[7, 42]])[0]
    b = _rows([500], [[9999, 42]])[0]      # only the older predecessor differs
    np.testing.assert_array_equal(a[:PER_GRAM], b[:PER_GRAM])
    assert not np.array_equal(a[PER_GRAM:], b[PER_GRAM:]), "the trigram heads must move"


def test_eos_in_the_window_resets_everything_before_it():
    """A window that crosses a document boundary must read as EOS, not leak across it."""
    after_eos = _rows([500], [[123, EOS]])[0]
    both_eos = _rows([500], [[EOS, EOS]])[0]
    np.testing.assert_array_equal(after_eos, both_eos)

    # And the cut propagates: an EOS two back also silences the position after it.
    older_eos = _rows([500], [[EOS, 77]])[0]
    assert not np.array_equal(older_eos, both_eos), "a non-EOS immediate predecessor stands"


def test_missing_predecessors_read_as_eos():
    """At the start of a sequence there is nothing behind the token."""
    start = _rows([500], [[None, None]])[0]
    np.testing.assert_array_equal(start, _rows([500], [[EOS, EOS]])[0])


def test_a_token_being_eos_does_not_cut_its_own_context():
    """The reference only cuts on predecessors, so an EOS token still sees its history."""
    with_history = _rows([EOS], [[11, 22]])[0]
    without = _rows([EOS], [[EOS, EOS]])[0]
    assert not np.array_equal(with_history, without)


def test_head_count_is_validated():
    with pytest.raises(ValueError, match="needs 16 head ranges"):
        ple_rows(
            [1], [[1, 2]],
            multipliers=MULTIPLIERS, head_offsets=OFFSETS[:4], head_vocab_sizes=VOCABS[:4],
            ngram_size=NGRAM, heads_per_ngram=PER_GRAM, eos_token_id=EOS,
        )


def test_context_window_shape_and_order():
    """ctx[0] is the token; ctx[s] is s positions back, from an oldest-first list."""
    assert context_window(5, [1, 2], EOS, 3) == [5, 2, 1]


# ---------------------------------------------------------------------------------------
# The dilated causal conv, and the invariant its history exists to guarantee.
# ---------------------------------------------------------------------------------------

import torch  # noqa: E402

from freetoken.models.qwen4exp.ple import dilated_causal_conv  # noqa: E402

CONV_K, DILATION, WIDTH = 4, 3, 7
HIST = (CONV_K - 1) * DILATION


def _weights(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(WIDTH, CONV_K, generator=g, dtype=torch.float64)


def test_tap_formula_matches_the_reference_indexing():
    """out[t,c] = sum_k w[c,k] * x[t - (K-1-k)*dilation, c], with zeros before the start."""
    w = _weights()
    g = torch.Generator().manual_seed(1)
    x = torch.randn(9, WIDTH, generator=g, dtype=torch.float64)
    got = dilated_causal_conv(x, w, history=None, dilation=DILATION)

    want = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for k in range(CONV_K):
            src = t - (CONV_K - 1 - k) * DILATION
            if src >= 0:
                want[t] += w[:, k] * x[src]
    torch.testing.assert_close(got, want)


def test_chunked_matches_single_shot():
    """The whole point of carrying a history: a decode step must land where a one-shot
    prefill would. Without it every continuation convolves as if it were at position 0,
    which produces no error and no NaN -- just a quietly missing term."""
    w = _weights(2)
    g = torch.Generator().manual_seed(3)
    x = torch.randn(20, WIDTH, generator=g, dtype=torch.float64)

    one_shot = dilated_causal_conv(x, w, history=None, dilation=DILATION)

    out, hist = [], None
    for lo, hi in ((0, 7), (7, 8), (8, 14), (14, 20)):   # prefill, then decode-sized steps
        seg = x[lo:hi]
        out.append(dilated_causal_conv(seg, w, history=hist, dilation=DILATION))
        prev = hist if hist is not None else x.new_zeros(HIST, WIDTH)
        hist = torch.cat([prev, seg], dim=0)[-HIST:]
    torch.testing.assert_close(torch.cat(out, dim=0), one_shot)


def test_a_cold_history_is_actually_different():
    """Guards the test above from passing vacuously."""
    w = _weights(4)
    g = torch.Generator().manual_seed(5)
    x = torch.randn(20, WIDTH, generator=g, dtype=torch.float64)
    one_shot = dilated_causal_conv(x, w, history=None, dilation=DILATION)
    cold_tail = dilated_causal_conv(x[7:], w, history=None, dilation=DILATION)
    assert not torch.allclose(cold_tail, one_shot[7:])


def test_history_shape_is_validated():
    w = _weights()
    x = torch.randn(4, WIDTH, dtype=torch.float64)
    with pytest.raises(ValueError, match="history"):
        dilated_causal_conv(x, w, history=torch.zeros(HIST + 1, WIDTH, dtype=torch.float64),
                            dilation=DILATION)


def test_weight_orientation_is_validated():
    """[channels, taps], not [taps, channels] -- only a shape error when they differ."""
    x = torch.randn(4, WIDTH, dtype=torch.float64)
    with pytest.raises(ValueError, match="channels"):
        dilated_causal_conv(x, _weights().T.contiguous(), history=None, dilation=DILATION)
