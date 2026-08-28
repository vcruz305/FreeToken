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

import math

import numpy as np
import torch
import torch.nn.functional as F
from freetoken.layers import BaseOP

from .hyper_connections import grouped_rms_norm

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


# ---------------------------------------------------------------------------------------
# The PLE block itself.
#
# Transcribed from llama.cpp ``build_ple``. The gather feeds a key/value pair, the key is
# scored against the current residual state per stream, and the gated value plus a dilated
# causal convolution of it are added back to the state.
# ---------------------------------------------------------------------------------------


def signed_sqrt_gate(scores: torch.Tensor) -> torch.Tensor:
    """``sigmoid(sgn(s) * sqrt(clamp(|s|, 1e-6, inf)))``.

    The signed square root compresses the score before the sigmoid, so a large dot product
    does not saturate the gate. The clamp floor keeps the gradient of sqrt finite at zero;
    it is in the reference and is kept because dropping it changes values near s == 0.
    """
    mag = torch.sqrt(torch.clamp(scores.abs(), min=1e-6))
    return torch.sigmoid(torch.sign(scores) * mag)


def dilated_causal_conv(
    x: torch.Tensor, weight: torch.Tensor, *, history: torch.Tensor | None, dilation: int
) -> torch.Tensor:
    """Depthwise causal conv, dilated, as a sum of shifted copies.

    ``out[t, c] = sum_k w[c, k] * x[t - (K-1-k)*dilation, c]`` -- note the tap index runs
    backwards, so tap ``K-1`` is the current position and tap 0 reaches furthest back. Each
    tap is one weight per channel; there is no mixing across channels or across positions
    within a tap.

    ``weight`` is ``[channels, taps]``. ggml stores the kernel fastest-first as
    ``[taps, channels]`` and the reader hands back the reverse, which is the same
    convention qwen35moe's ``ssm_conv1d`` follows. Indexing it the other way round is a
    shape error only when taps == channels, so it is asserted rather than assumed.

    ``history`` is the ``(K-1)*dilation`` positions preceding ``x``, or None at the start of
    a sequence, where the reference reads zeros. Passing it explicitly (rather than holding
    state here) is what lets a chunked prefill match a single-shot one.
    """
    T, C = x.shape
    if weight.shape[0] != C:
        raise ValueError(
            f"PLE conv weight should be [channels, taps] with {C} channels, got "
            f"{tuple(weight.shape)}"
        )
    K = weight.shape[1]
    hist = (K - 1) * dilation
    if history is None:
        history = x.new_zeros(hist, C)
    elif history.shape != (hist, C):
        raise ValueError(
            f"PLE conv history should be {(hist, C)}, got {tuple(history.shape)}"
        )
    padded = torch.cat([history, x], dim=0)                 # [hist + T, C]

    out = None
    for k in range(K):
        start = hist - (K - 1 - k) * dilation
        term = padded[start : start + T] * weight[:, k]
        out = term if out is None else out + term
    return out


class Qwen4ExpPLE(BaseOP):
    """n-gram hashed per-layer embeddings, applied on one layer.

    ``table`` is the 320-million-row shared embedding (~29 GB of the checkpoint); only the
    ``ple_n_heads`` rows a token hashes to are ever gathered, so it is a GGUFEmbedding
    rather than anything dequantised.
    """

    def __init__(
        self, geo: dict, hidden_size: int, eps: float, quant_types: dict, table_rows: int
    ):
        from freetoken.layers.gguf import GGUFEmbedding, GGUFLinear

        self.hc = geo["hc_count"]
        self.eps = eps
        self.hidden_size = hidden_size
        self.n_heads = (geo["ple_ngram_size"] - 1) * geo["ple_heads_per_ngram"]
        self.head_dim = geo["ple_input_dim"]
        self.ngram_size = geo["ple_ngram_size"]
        self.heads_per_ngram = geo["ple_heads_per_ngram"]
        self.multipliers = list(geo["ple_head_multipliers"])
        self.head_offsets = list(geo["ple_head_offsets"])
        self.head_vocab_sizes = list(geo["ple_head_vocab_sizes"])
        self.eos_token_id = geo["ple_eos_token_id"]
        self.conv_kernel = geo["ple_conv_kernel"]

        width = self.hc * hidden_size
        gathered = self.n_heads * self.head_dim
        if gathered != hidden_size:
            raise ValueError(
                f"PLE gathers {self.n_heads} heads of {self.head_dim} = {gathered}, which "
                f"should equal hidden_size {hidden_size}"
            )

        # NOT a module buffer: see the loader. Held as an mmap-backed host tensor and
        # indexed per token, so the 28.8 GB never has to fit anywhere.
        self._table_rows = table_rows
        self._table_type = quant_types["table"]
        # Matches the norms allocated just below, which the engine sized under its dtype.

        self._table = None  # bound by bind_table()
        self.key = GGUFLinear(gathered, width, quant_type=quant_types["key"])
        self.value = GGUFLinear(gathered, hidden_size, quant_type=quant_types["value"])
        self.norm_key = torch.empty(width)
        self.norm_query = torch.empty(width)
        self.norm_conv = torch.empty(width)
        # [channels, taps]: the reader reverses ggml's fastest-first dims.
        self.conv1d = torch.empty(width, self.conv_kernel)
        # The compute dtype, taken from a tensor allocated under the engine's context.
        self._dtype = self.norm_key.dtype

    def bind_table(self, model_path: str) -> None:
        """Point at the packed PLE table without copying it.

        The reader hands back an mmap-backed tensor, so this costs address space rather
        than memory and the OS pages in only the rows actually hashed to.
        """
        from freetoken.models.gguf.reader import iter_gguf_tensors

        for t in iter_gguf_tensors(model_path):
            if t.name == "per_layer_token_embd.weight":
                self._table = t.packed()
                return
        raise ValueError("qwen4exp: per_layer_token_embd.weight not found")

    def gather(self, rows: torch.Tensor, device) -> torch.Tensor:
        """Gather the hashed rows on the host, then dequantise them on the device.

        Only the gathered bytes cross the bus: 16 rows of 90 packed bytes per token.
        """
        from freetoken.kernel.gguf import ggml_dequantize

        assert self._table is not None, "bind_table() must run before the first forward"
        n_tokens = rows.shape[0]
        idx = rows.reshape(-1).to("cpu", torch.int64)
        packed = self._table.index_select(0, idx).to(device, non_blocking=True)
        flat = ggml_dequantize(
            packed, self._table_type, packed.shape[0], self.head_dim, self._dtype
        )
        # get_rows lays the head dimension out slowest, so the heads of one token are
        # adjacent rows; flatten them into that token's single [n_heads*head_dim] vector,
        # which is what the key/value projections consume.
        return flat.reshape(n_tokens, self.n_heads * self.head_dim)

    def forward(
        self,
        state: torch.Tensor,
        emb: torch.Tensor,
        *,
        history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``state`` [T, hc, D]; ``emb`` [T, n_heads*head_dim] gathered on the host.

        The gather is not done here because it reads a 28.8 GB host-resident table, which
        a captured CUDA graph cannot contain. It happens in prepare_host_inputs and arrives
        through a persistent graph-input buffer.
        """
        T = state.shape[0]
        emb = emb[:T].reshape(T, self.n_heads * self.head_dim)

        key = grouped_rms_norm(
            self.key.forward(emb).reshape(T, self.hc, self.hidden_size),
            self.norm_key, self.eps, self.hc,
        ).reshape(T, self.hc, self.hidden_size)
        query = grouped_rms_norm(state, self.norm_query, self.eps, self.hc).reshape(
            T, self.hc, self.hidden_size
        )

        # Per-stream dot product, scaled, then a signed square root before the sigmoid.
        scores = (key * query).sum(dim=-1) / math.sqrt(self.hidden_size)   # [T, hc]
        gate = signed_sqrt_gate(scores)

        value = self.value.forward(emb)                                    # [T, D]
        gated = value.unsqueeze(1) * gate.unsqueeze(-1)                    # [T, hc, D]

        normalized = grouped_rms_norm(gated, self.norm_conv, self.eps, self.hc)  # [T, hc*D]
        conv_out = dilated_causal_conv(
            normalized, self.conv1d, history=history, dilation=self.ngram_size
        )
        conv_out = F.silu(conv_out).reshape(T, self.hc, self.hidden_size)
        return state + gated + conv_out
