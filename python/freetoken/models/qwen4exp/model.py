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
        mixed, inject = self.hc_attn.mix(state)
        out = (
            self.linear_attn.forward(mixed)
            if self._is_linear
            else self.self_attn.forward(mixed)
        )
        state = self.hc_attn.combine(state, out, inject)

        mixed, inject = self.hc_ffn.mix(state)
        state = self.hc_ffn.combine(state, self.mlp.forward(mixed), inject)
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

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        from .ple import ple_rows

        # The wide residual starts as hc identical copies of the embedding.
        state = hc_init(self.embed_tokens.forward(input_ids), self.hc)

        # The hash needs each token's predecessors. A fresh single-sequence prefill has
        # them all in this batch, and positions before the start read as EOS. Decode and
        # chunked prefill need the preceding tokens from the KV cells and a conv history,
        # which is not wired yet -- refuse rather than quietly hash the wrong window.
        ctx = get_global_ctx()
        toks = input_ids.tolist()
        g = self.geo
        n_prev = g["ple_ngram_size"] - 1
        preds = [[toks[i - s] if i - s >= 0 else None for s in range(n_prev, 0, -1)]
                 for i in range(len(toks))]
        rows = ple_rows(
            toks, preds,
            multipliers=list(g["ple_head_multipliers"]),
            head_offsets=list(g["ple_head_offsets"]),
            head_vocab_sizes=list(g["ple_head_vocab_sizes"]),
            ngram_size=g["ple_ngram_size"],
            heads_per_ngram=g["ple_heads_per_ngram"],
            eos_token_id=g["ple_eos_token_id"],
        )
        rows_t = torch.from_numpy(rows).to(input_ids.device, torch.int32)

        for i, layer in enumerate(self.layers.op_list):
            if i in self.ple_layers:
                state = self.ple.forward(state, rows_t, history=None)
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

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)


__all__ = ["Qwen4ExpForCausalLM"]
