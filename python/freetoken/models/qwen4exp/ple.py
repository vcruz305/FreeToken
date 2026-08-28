"""qwen4exp PLE: n-gram hashed per-layer embeddings.

Each token gathers ``ple_n_heads`` rows from one shared table -- 320,001,536 rows of 160
values, about 29 GB of the checkpoint -- and the row indices come from hashing the token
together with its predecessors. There is no counterpart to this anywhere else in the tree.

``ple_n_heads = (ngram_size - 1) * heads_per_ngram``: one group of heads per n-gram order.
With ngram_size 3 and 8 heads per gram that is 16 heads, covering the bigram (n=2) and
trigram (n=3) of each position. All heads within one order share a single hash and differ
only in which slice of the table they index.

Transcribed from llama.cpp ``src/models/qwen4exp.cpp`` (``llm_graph_input_ple::set_input``).
The parts that are easy to get wrong and impossible to see in a shape check:

* the mixing is uint64 with wraparound, and Python integers are unbounded, so the multiply
  is masked to 64 bits. For *this* checkpoint the mask never actually fires -- the largest
  token id (248319) times the largest multiplier (2.37e13) is about 5.9e18, comfortably
  under 2**63 -- so it is defensive rather than load-bearing here. It stays because the
  reference is uint64 and a checkpoint with a bigger vocabulary or multipliers would
  overflow, at which point an unmasked hash would silently pick different rows.
* an EOS anywhere in the window resets everything **at or before it** to EOS, but a token
  being EOS itself does not cut its own context.
* a missing predecessor -- before the start of the sequence -- reads as EOS.
* the hash is shared across the heads of one order; only ``% vocab[h] + offset[h]`` differs.
"""

from __future__ import annotations

import numpy as np

_U64 = 0xFFFFFFFFFFFFFFFF


def ngram_hash(ctx: list[int], multipliers: list[int], n: int) -> int:
    """``(t[0]*m[0]) ^ (t[1]*m[1]) ^ ... ^ (t[n-1]*m[n-1])`` in uint64 arithmetic.

    ``ctx[0]`` is the token itself and ``ctx[s]`` its predecessor ``s`` positions back.
    """
    mixed = (ctx[0] * multipliers[0]) & _U64
    for j in range(1, n):
        mixed ^= (ctx[j] * multipliers[j]) & _U64
    return mixed & _U64


def context_window(
    token: int, predecessors: list[int | None], eos_token_id: int, ngram_size: int
) -> list[int]:
    """The ``ngram_size`` tokens the hash sees, with the EOS reset applied.

    ``predecessors`` is oldest-first and ``ngram_size - 1`` long; ``None`` marks a position
    before the start of the sequence. Once the walk backwards hits an EOS or a gap,
    everything from there on reads as EOS -- the window does not see across a document
    boundary. The token's own EOS-ness is not a cut, matching the reference.
    """
    ctx = [token]
    n_prev = ngram_size - 1
    cut = False
    for s in range(1, ngram_size):
        # predecessors is oldest-first, so s positions back is index n_prev - s
        t = None if cut else predecessors[n_prev - s]
        cut = cut or t is None or t < 0 or t == eos_token_id
        ctx.append(eos_token_id if cut else t)
    return ctx


def ple_rows(
    tokens: list[int],
    predecessors: list[list[int | None]],
    *,
    multipliers: list[int],
    head_offsets: list[int],
    head_vocab_sizes: list[int],
    ngram_size: int,
    heads_per_ngram: int,
    eos_token_id: int,
) -> np.ndarray:
    """Row indices into ``per_layer_token_embd`` -- ``[len(tokens), n_heads]`` int32.

    ``predecessors[i]`` holds the ``ngram_size - 1`` tokens before ``tokens[i]``,
    oldest-first, with ``None`` for positions before the sequence start.
    """
    n_heads = (ngram_size - 1) * heads_per_ngram
    if len(head_offsets) < n_heads or len(head_vocab_sizes) < n_heads:
        raise ValueError(
            f"PLE needs {n_heads} head ranges, got {len(head_offsets)} offsets and "
            f"{len(head_vocab_sizes)} vocab sizes"
        )
    out = np.empty((len(tokens), n_heads), dtype=np.int32)
    for i, token in enumerate(tokens):
        ctx = context_window(token, predecessors[i], eos_token_id, ngram_size)
        for n in range(2, ngram_size + 1):
            mixed = ngram_hash(ctx, multipliers, n)
            base = (n - 2) * heads_per_ngram
            for g in range(heads_per_ngram):
                h = base + g
                out[i, h] = mixed % head_vocab_sizes[h] + head_offsets[h]
    return out
