"""One expert slot pool holding layers of more than one ggml type.

llama.cpp's "dynamic" quants -- unsloth's UD-* family especially -- deliberately raise the
precision of a handful of layers, so a checkpoint's expert banks are routinely non-uniform.
Qwen3.8-Flash-Next is mixed in every published variant; UD-Q4_K_XL is Q4_K on 47 layers and
Q5_K on one. FreeToken used to reject those outright, because the slot pool is a single
allocation and ``moe_vec.cuh`` derived each row's address from ``ncols / qk`` -- one stride
for the whole pool.

The kernel now walks rows by an explicit byte stride taken from the weight tensor itself, so
the pool can be padded to the widest layer's row and every type in it still lands on the
right block. A layer quantized more tightly simply leaves the tail of its rows unread: the
kernel consumes ``ncols / qk`` blocks from each row's base, and ``qk`` comes from the type
being run, not from the pool.

The equality assertions are the whole point. Reading a padded pool at the unpadded stride
does not crash and does not produce NaN -- it silently reads every block at the wrong offset
and emits fluent nonsense, which is exactly the failure this class of bug produces.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

E, NROWS, NCOLS, TOKENS, TOP_K = 8, 64, 256, 4, 2
NBLOCKS = NCOLS // 32
# ggml block sizes for a 32-element block: q4_0 is a half scale + 16 packed nibbles,
# q8_0 a half scale + 32 int8s.
GGML_Q4_0, GGML_Q8_0 = 2, 8
RB_Q4_0, RB_Q8_0 = NBLOCKS * 18, NBLOCKS * 34


def _blocks(rng, quant_bytes):
    """A bank of syntactically valid blocks: a sane half scale then packed quants.

    The scale has to be a real half -- random bytes decode to NaN/Inf often enough that
    every comparison would pass vacuously.
    """
    d = rng.uniform(0.02, 0.08, (E, NROWS, NBLOCKS)).astype(np.float16)
    q = rng.integers(0, 256, (E, NROWS, NBLOCKS, quant_bytes), dtype=np.uint8)
    raw = np.concatenate([d.view(np.uint8).reshape(E, NROWS, NBLOCKS, 2), q], axis=3)
    return raw.reshape(E, NROWS, NBLOCKS * (2 + quant_bytes)).copy()


@pytest.fixture(scope="module")
def routing():
    torch.manual_seed(0)
    x = torch.randn(TOKENS, NCOLS, device="cuda", dtype=torch.bfloat16)
    ids = torch.randint(0, E, (TOKENS, TOP_K), device="cuda", dtype=torch.int32)
    return x, ids


@pytest.mark.parametrize("pad_to", [RB_Q4_0 + 16, RB_Q4_0 + 64, RB_Q8_0])
def test_padding_a_pool_does_not_move_any_block(routing, pad_to):
    """Same bytes, wider rows: the result must not change at all."""
    from freetoken.kernel.gguf import ggml_moe_a8_vec

    x, ids = routing
    rng = np.random.default_rng(0)
    natural = torch.from_numpy(_blocks(rng, 16)).cuda()
    padded = torch.zeros(E, NROWS, pad_to, dtype=torch.uint8, device="cuda")
    padded[:, :, :RB_Q4_0] = natural

    want = ggml_moe_a8_vec(x, natural, ids, TOP_K, GGML_Q4_0, NROWS, TOKENS)
    got = ggml_moe_a8_vec(x, padded, ids, TOP_K, GGML_Q4_0, NROWS, TOKENS)
    assert torch.isfinite(want).all(), "test vector decoded to non-finite values"
    assert torch.equal(want, got)


@pytest.mark.parametrize(
    "quant_bytes,row_bytes,ggml_type",
    [(16, RB_Q4_0, GGML_Q4_0), (32, RB_Q8_0, GGML_Q8_0)],
)
def test_two_types_share_one_pool_width(routing, quant_bytes, row_bytes, ggml_type):
    """A pool sized for the widest type serves the narrower one unchanged.

    This is the real mixed-bank case: q4_0 and q8_0 layers cohabiting a pool whose rows are
    272 bytes because q8_0 needs that much, with q4_0 using only the first 144 of each.
    """
    from freetoken.kernel.gguf import ggml_moe_a8_vec

    x, ids = routing
    rng = np.random.default_rng(7)
    natural = torch.from_numpy(_blocks(rng, quant_bytes)).cuda()
    shared = torch.zeros(E, NROWS, max(RB_Q4_0, RB_Q8_0), dtype=torch.uint8, device="cuda")
    shared[:, :, :row_bytes] = natural

    want = ggml_moe_a8_vec(x, natural, ids, TOP_K, ggml_type, NROWS, TOKENS)
    got = ggml_moe_a8_vec(x, shared, ids, TOP_K, ggml_type, NROWS, TOKENS)
    assert torch.isfinite(want).all()
    assert torch.equal(want, got)


def test_garbage_in_the_padding_is_never_read(routing):
    """The tail past a row's own blocks must not reach the result.

    Nothing zeroes the padding on the real load path -- the loader writes each layer's
    native bytes into the prefix and leaves the rest -- so the kernel has to be indifferent
    to whatever is out there.
    """
    from freetoken.kernel.gguf import ggml_moe_a8_vec

    x, ids = routing
    rng = np.random.default_rng(3)
    natural = torch.from_numpy(_blocks(rng, 16)).cuda()
    pad_to = RB_Q8_0
    dirty = torch.randint(0, 256, (E, NROWS, pad_to), dtype=torch.uint8, device="cuda")
    dirty[:, :, :RB_Q4_0] = natural

    want = ggml_moe_a8_vec(x, natural, ids, TOP_K, GGML_Q4_0, NROWS, TOKENS)
    got = ggml_moe_a8_vec(x, dirty, ids, TOP_K, GGML_Q4_0, NROWS, TOKENS)
    assert torch.equal(want, got)
