"""qwen4exp decoder: a qwen35moe core wrapped in hyper-connections.

The mixers and the FFN are qwen35moe's, unchanged except for one flag:

* full-attention layers use ``Qwen3_5Attention`` -- Qwen3.5 already has the same gated
  attention, with the query projection twice as wide and each head's gate following its
  own queries
* linear layers use ``Qwen3_5GatedDeltaNet`` with ``gate_activation="sigmoid"``. That flag
  is the single numerical difference llama.cpp calls out between the two GDNs; silu would
  give identical shapes and quietly wrong output
* the FFN is ``Qwen3_5MoE`` -- routed softmax top-k with renormalisation plus a
  sigmoid-gated shared expert, which is what qwen4exp does

What is genuinely different is the residual path. There is no ``attn_norm``, ``ffn_norm``
or ``output_norm`` in the checkpoint: the model carries ``hc`` parallel residual streams
and every sublayer is wrapped in mix/combine, which is where the normalisation lives. The
final mix before the LM head *is* the output norm.

Transcribed from llama.cpp ``src/models/qwen4exp.cpp``; see ``hyper_connections.py`` for
the mix/combine algebra and the details that a shape check cannot see.

Not yet implemented here: the lightning-indexer sparse attention path. llama.cpp only
takes it when an indexer cache exists and the layer's compress ratio is nonzero, and falls
back to dense attention otherwise. Since ``indexer_top_k`` is 2048, dense attention is
*mathematically identical* for prompts shorter than that, which is what makes bringing the
model up dense-first a real validation rather than an approximation. Longer contexts need
the indexer and will differ until it lands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, OPList, ParallelLMHead, VocabParallelEmbedding
from freetoken.models.blocks import BaseLLMModel
from freetoken.models.qwen3_5_moe.attention import Qwen3_5Attention
from freetoken.models.qwen3_5_moe.gdn import Qwen3_5GatedDeltaNet
from freetoken.models.qwen3_5_moe.moe import Qwen3_5MoE
from freetoken.utils import nvtx_annotate

from .hyper_connections import hc_combine, hc_init, hc_mix

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class HyperConnection(BaseOP):
    """One mix/combine pair: the weights for wrapping a single sublayer.

    ``down``/``up`` are a low-rank mixer over the flattened ``hc * hidden`` state,
    ``inject`` emits the per-stream scatter weights, and ``norm`` stands in for the
    pre-norm this architecture does not otherwise have.
    """

    def __init__(self, hidden_size: int, hc: int, low_rank: int, eps: float, *, injects: bool):
        width = hc * hidden_size
        self.hc = hc
        self.eps = eps
        self.norm = torch.empty(width)
        self.down = torch.empty(low_rank, width)
        self.up = torch.empty(width, low_rank)
        # The final mix before the LM head produces no injection: it has no sublayer to
        # scatter back into.
        self.inject = torch.empty(hc, width) if injects else None

    def mix(self, state: torch.Tensor):
        return hc_mix(
            state, self.norm, self.down, self.up, self.inject, eps=self.eps, hc=self.hc
        )

    def combine(self, state: torch.Tensor, block_out: torch.Tensor, inject: torch.Tensor):
        return hc_combine(state, block_out, inject, hc=self.hc)


class Qwen4ExpDecoderLayer(BaseOP):
    """``state -> mix -> mixer -> combine -> mix -> moe -> combine -> state``."""

    def __init__(self, config: ModelConfig, geo: dict, layer_id: int):
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
        hc, low_rank = geo["hc_count"], geo["hc_low_rank"]

        if self._is_linear:
            g = config.linear_attention_group()
            assert g is not None
            self.linear_attn = Qwen3_5GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_k_heads=g.num_key_heads,
                num_v_heads=g.num_value_heads,
                head_k_dim=g.key_head_dim,
                head_v_dim=g.value_head_dim,
                conv_kernel_size=g.conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
                layer_id=layer_id,
                expert_quant=config.expert_quant,
                attn_quant=config.attn_quant,
                # The one numerical difference from Qwen3.5's GDN.
                gate_activation="sigmoid",
            )
        else:
            self.self_attn = Qwen3_5Attention(config, layer_id)

        self.mlp = Qwen3_5MoE(config, layer_id)
        self.hc_attn = HyperConnection(
            config.hidden_size, hc, low_rank, config.rms_norm_eps, injects=True
        )
        self.hc_ffn = HyperConnection(
            config.hidden_size, hc, low_rank, config.rms_norm_eps, injects=True
        )

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # The residual stream carries exactly one dtype. A sublayer can hand back something
        # wider -- the triton attention backend accumulates in float32, and a GGUFLinear
        # returns whatever dtype its activation was, since its weight is packed uint8 and
        # cannot mismatch. Without pinning it here that float32 rides the residual through
        # every later layer until it meets the first module with a real typed weight, which
        # is a confusing failure far from its cause and, on Turing, silently selects a GEMM
        # path many times slower than the one asked for.
        mixed, inject = self.hc_attn.mix(state)
        out = (
            self.linear_attn.forward(mixed)
            if self._is_linear
            else self.self_attn.forward(mixed)
        )
        state = self.hc_attn.combine(state, out.to(state.dtype), inject)

        mixed, inject = self.hc_ffn.mix(state)
        ffn_out = self.mlp.forward(mixed).to(state.dtype)
        state = self.hc_ffn.combine(state, ffn_out, inject)
        return state


class Qwen4ExpModel(BaseOP):
    def __init__(self, config: ModelConfig, geo: dict):
        from .gguf import ple_quant_types, ple_table_rows
        from .ple import Qwen4ExpPLE

        self.hc = geo["hc_count"]
        self.geo = geo
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
        )
        self.layers = OPList(
            [Qwen4ExpDecoderLayer(config, geo, i) for i in range(config.num_layers)]
        )
        # PLE runs on the single layer named by ple.layers, before that layer's block.
        self.ple_layers = set(geo["ple_layers"])
        assert len(self.ple_layers) == 1, "qwen4exp expects exactly one PLE layer"
        self.ple = Qwen4ExpPLE(
            geo,
            config.hidden_size,
            config.rms_norm_eps,
            ple_quant_types(config.gguf_model_path, next(iter(self.ple_layers))),
            ple_table_rows(config.gguf_model_path),
        )
        # Stands in for the output norm this architecture does not have.
        self.hc_head = HyperConnection(
            config.hidden_size,
            geo["hc_count"],
            geo["hc_low_rank"],
            config.rms_norm_eps,
            injects=False,
        )

    def _gather_inline(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Hash and gather straight from the batch's token ids.

        Used only off the captured path. The window here comes from the batch alone, so a
        continuation chunk would see fewer predecessors than it should -- acceptable for
        warmup, which is why the real path routes through prepare_host_inputs instead.
        """
        from .ple import ple_rows

        g = self.geo
        toks = input_ids.tolist()
        n_prev = g["ple_ngram_size"] - 1
        preds = [
            [toks[i - s] if i - s >= 0 else None for s in range(n_prev, 0, -1)]
            for i in range(len(toks))
        ]
        rows = ple_rows(
            toks, preds,
            multipliers=list(g["ple_head_multipliers"]),
            head_offsets=list(g["ple_head_offsets"]),
            head_vocab_sizes=list(g["ple_head_vocab_sizes"]),
            ngram_size=g["ple_ngram_size"],
            heads_per_ngram=g["ple_heads_per_ngram"],
            eos_token_id=g["ple_eos_token_id"],
        )
        return self.ple.gather(torch.from_numpy(rows), input_ids.device)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # The wide residual starts as hc identical copies of the embedding, in the dtype
        # the loaded weights actually have. Reading it off a real parameter here rather than
        # from a construction-time default means the residual cannot drift into a wider
        # dtype than the projections it feeds -- which both mismatches them and, on Turing,
        # would pick a GEMM path 12-22x slower than the one asked for.
        state = hc_init(self.embed_tokens.forward(input_ids), self.hc)
        state = state.to(self.hc_head.norm.dtype)

        # Normally the PLE embedding arrives as a graph input, gathered on the host before
        # the capture boundary (see Qwen4ExpForCausalLM.prepare_host_inputs). Warmup and
        # any other path that calls the model directly has no such buffer; those are never
        # captured, so falling back to an inline gather there is safe. Only a captured
        # graph cannot contain the host work.
        batch = get_global_ctx().batch
        host = batch.host_inputs or {}
        ple_emb = host.get("ple_emb")
        if ple_emb is None:
            ple_emb = self._gather_inline(input_ids)

        for i, layer in enumerate(self.layers.op_list):
            if i in self.ple_layers:
                state = self.ple.forward(state, ple_emb, batch=batch)
            state = layer.forward(state)
        mixed, _ = self.hc_head.mix(state)
        return mixed


class Qwen4ExpForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        from .gguf import geometry_from_path

        assert config.gguf_model_path is not None, (
            "qwen4exp is only served from GGUF; the hyper-connection, indexer and PLE "
            "geometry is read from the checkpoint"
        )
        geo = geometry_from_path(config.gguf_model_path)

        # This adapter runs dense attention on the full-attention layers. The checkpoint
        # was trained with a lightning indexer selecting indexer_top_k keys, so dense is
        # EXACTLY equivalent while the context stays at or below that budget and diverges
        # above it -- silently, since attending to more keys neither crashes nor produces
        # anything obviously wrong. Say so at load rather than let it pass unnoticed.
        from freetoken.utils import init_logger

        top_k = geo["indexer_top_k"]
        ctx_len = getattr(config, "max_position_embeddings", None) or geo["context_length"]
        log = init_logger(__name__)
        if ctx_len > top_k:
            log.info_rank0(
                f"qwen4exp: attention runs dense (no lightning indexer). This is exact up "
                f"to {top_k} tokens of context; beyond that it attends to every key rather "
                f"than the top {top_k} the model was trained for, and output will drift "
                f"from the reference."
            )

        self._hidden_size = config.hidden_size
        self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = Qwen4ExpModel(config, geo)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

        from .gguf import convert_qwen4exp_to_gguf

        convert_qwen4exp_to_gguf(self, config, model_path=config.gguf_model_path)
        self.model.ple.bind_table(config.gguf_model_path)

    def init_state(self, num_slots: int, device, dtype) -> None:
        """Allocate PLE's conv history, one row per linear-state slot.

        Called by the engine once the linear state pool is sized, so PLE's history is
        keyed exactly as the GDN's is and a request's two histories cannot drift apart.
        """
        self.model.ple.init_conv_state(num_slots, device, dtype)

    def host_input_spec(self) -> dict:
        """The PLE embedding is a graph input, not host work inside the capture.

        Its table is 28.8 GB and stays on the host, but the gathered rows depend only on
        the token ids -- not on any hidden state -- so they can be produced before the
        forward and read from a persistent buffer, which is exactly how llama.cpp treats
        the hashed rows. Without this a captured graph would contain an unpinned H2D copy
        and capture fails.
        """
        # The buffer dtype must be the model's. Hardcoding bf16 here made PLE's
        # contribution bf16 under --dtype float16, and float16 + bfloat16 promotes to
        # float32, so the whole residual silently widened from one layer onward.
        return {"ple_emb": (self._hidden_size, self.model.hc_head.norm.dtype)}

    def prepare_host_inputs(self, batch) -> None:
        """Hash each position's n-gram window and gather its PLE rows.

        The window comes from the request's own token history, so a decode step and a
        chunked prefill see the same predecessors a single-shot prefill would. Positions
        before the start of the sequence read as EOS, as in the reference.
        """
        from .ple import ple_rows

        g = self.model.geo
        n_prev = g["ple_ngram_size"] - 1
        toks, preds = [], []
        for req in batch.reqs:
            ids = req.input_ids.tolist()
            for pos in range(req.cached_len, req.device_len):
                if pos >= len(ids):
                    break
                toks.append(ids[pos])
                preds.append(
                    [ids[pos - s] if pos - s >= 0 else None for s in range(n_prev, 0, -1)]
                )
        if not toks:
            return
        rows = ple_rows(
            toks, preds,
            multipliers=list(g["ple_head_multipliers"]),
            head_offsets=list(g["ple_head_offsets"]),
            head_vocab_sizes=list(g["ple_head_vocab_sizes"]),
            ngram_size=g["ple_ngram_size"],
            heads_per_ngram=g["ple_heads_per_ngram"],
            eos_token_id=g["ple_eos_token_id"],
        )
        emb = self.model.ple.gather(torch.from_numpy(rows), self._device)
        batch.host_inputs = dict(batch.host_inputs or {})
        batch.host_inputs["ple_emb"] = emb

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)


__all__ = ["Qwen4ExpForCausalLM"]
