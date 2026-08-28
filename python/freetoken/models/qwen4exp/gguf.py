"""GGUF layout for ``general.architecture == "qwen4exp"`` (Qwen3.8-Flash-Next).

Read off ``unsloth/Qwen3.8-Flash-Next-GGUF`` directly rather than from a model card: the
shard headers carry the KV block and the full tensor table, so a few MB of range requests
gives the ground truth for a 72 GB checkpoint.

The architecture is largely two things FreeToken already serves, bolted together:

* the hybrid GDN/full-attention decoder of ``qwen35moe`` -- the SSM layers here carry the
  same tensors at the same shapes, so ``qwen3_5_moe/gdn.py`` should carry over
* routed MoE experts over the offload bank machinery, with the per-layer quant types the
  UD-* builds require

and three things it does not:

* **gated attention**: on full-attention layers ``attn_q`` is [2560, 12288], twice the
  24*256 the head geometry calls for, because each head's queries are followed by that
  head's gate. The layout is interleaved per head -- [h0_q, h0_gate, h1_q, h1_gate, ...] --
  not two contiguous halves; see ``attention.split_q_and_gate``. Those layers have no
  separate ``attn_gate`` while the SSM layers do, and that presence/absence pattern is
  what makes the fusion unambiguous.
* **low-rank hyper-connections** standing in for the usual pre-norms. There is no
  ``attn_norm`` and no ``ffn_norm`` anywhere in the file, and no ``output_norm`` global;
  ``hc_attn_norm``, ``hc_ffn_norm`` and ``output_hc_norm`` take their place. This is not
  DSV4's base/fn/scale parameterisation, so that module does not transfer.
* **PLE**, an n-gram hashed per-layer embedding on layer 1 whose table is
  [160, 320001536] -- roughly 29 GB of the checkpoint, and without precedent in the tree.

Everything here is layout only. There is no forward pass yet, and the smallest published
variant is 72.5 GB, so none of it has been run against real weights.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
)

if TYPE_CHECKING:
    from freetoken.models.gguf.config import GgufConfigShim

_ARCH = "qwen4exp"


def _kv(shim: "GgufConfigShim", key: str, default: Any = None) -> Any:
    """Read ``qwen4exp.<key>`` from the GGUF metadata.

    Raises rather than returning None for a key with no default: a geometry value that
    silently reads as None becomes a TypeError several frames away from the missing key.
    """
    val = shim.metadata.get(f"{shim.model_type}.{key}", default)
    if val is None and default is None:
        raise ValueError(
            f"GGUF {shim.model_path}: missing required key {shim.model_type}.{key}"
        )
    return val


def full_attention_layers(block_count: int, interval: int) -> tuple[int, ...]:
    """Indices of the full-attention (non-SSM) layers.

    llama.cpp places them at ``i % interval == interval - 1``: for 48 blocks at interval 4
    that is [3, 7, ..., 47], twelve layers. Confirmed against the checkpoint two ways --
    which tensors each layer actually carries, and ``attention.compress_ratios`` being
    nonzero at exactly those indices -- with no layer deviating.
    """
    return tuple(i for i in range(block_count) if i % interval == interval - 1)


def layer_kind(layer: int, interval: int) -> str:
    """``"full"`` for a full-attention layer, ``"ssm"`` for a GDN one."""
    return "full" if layer % interval == interval - 1 else "ssm"


# ---------------------------------------------------------------------------------------
# Tensor name map.
#
# Values are (freetoken path, conversion kind), as in the deepseek_v4 adapter. The paths
# are provisional: no model class exists yet, so they record intent rather than a verified
# attribute. What IS verified is the partition -- which tensors live on which layers, and
# that nothing in the file falls outside these tables.
# ---------------------------------------------------------------------------------------

# Present only on full-attention layers (i % 4 == 3).
_FULL_ONLY: dict[str, tuple[str, str]] = {
    "attn_k.weight": ("self_attn.k_proj.weight", "packed"),
    "attn_v.weight": ("self_attn.v_proj.weight", "packed"),
    "attn_output.weight": ("self_attn.o_proj.weight", "packed"),
    "attn_q_norm.weight": ("self_attn.q_norm.weight", "f32"),
    "attn_k_norm.weight": ("self_attn.k_norm.weight", "f32"),
    # The lightning indexer: a plain q/k pair with norms, top_k 2048. Simpler than DSV4's,
    # which additionally carries its own compressor.
    "indexer.q_proj.weight": ("self_attn.indexer.q_proj.weight", "bf16"),
    "indexer.k_proj.weight": ("self_attn.indexer.k_proj.weight", "bf16"),
    "indexer.q_norm.weight": ("self_attn.indexer.q_norm.weight", "f32"),
    "indexer.k_norm.weight": ("self_attn.indexer.k_norm.weight", "f32"),
}

# Present only on GDN/SSM layers. Same shapes and roles as qwen35moe's GDN block.
_SSM_ONLY: dict[str, tuple[str, str]] = {
    "ssm_conv1d.weight": ("linear_attn.conv1d.weight", "f32"),
    "ssm_norm.weight": ("linear_attn.norm.weight", "f32"),
    "ssm_out.weight": ("linear_attn.out_proj.weight", "packed"),
    "ssm_a": ("linear_attn.A_log", "f32"),
    "ssm_dt.bias": ("linear_attn.dt_bias", "f32"),
}

# Present on every layer.
_EVERY_LAYER: dict[str, tuple[str, str]] = {
    "ffn_gate_inp.weight": ("mlp.gate.weight", "f32"),
    # The shared expert is gated: a [hidden] vector scoring it per token, as in qwen35moe.
    "ffn_gate_inp_shexp.weight": ("mlp.shared_expert_gate.weight", "f32"),
    "ffn_down_shexp.weight": ("mlp.shared_expert.down_proj.weight", "packed"),
    # Hyper-connections. 10240 == 4 * 2560: the four residual streams flattened. down/up
    # are a low-rank (320) mixer, inject emits the four per-stream weights, and norm stands
    # in for the pre-norm this architecture does not otherwise have.
    "hc_attn_down.weight": ("hc_attn.down.weight", "packed"),
    "hc_attn_up.weight": ("hc_attn.up.weight", "packed"),
    "hc_attn_norm.weight": ("hc_attn.norm.weight", "f32"),
    "hc_attn_inject.weight": ("hc_attn.inject.weight", "f32"),
    "hc_ffn_down.weight": ("hc_ffn.down.weight", "packed"),
    "hc_ffn_up.weight": ("hc_ffn.up.weight", "packed"),
    "hc_ffn_norm.weight": ("hc_ffn.norm.weight", "f32"),
    "hc_ffn_inject.weight": ("hc_ffn.inject.weight", "f32"),
}

# PLE, on the layers named by ``ple.layers`` (just [1] in this release).
_PLE_LAYER: dict[str, tuple[str, str]] = {
    "ple_key.weight": ("ple.key.weight", "packed"),
    "ple_value.weight": ("ple.value.weight", "packed"),
    "ple_conv1d.weight": ("ple.conv1d.weight", "f32"),
    "ple_norm_conv.weight": ("ple.norm_conv.weight", "f32"),
    "ple_norm_key.weight": ("ple.norm_key.weight", "f32"),
    "ple_norm_query.weight": ("ple.norm_query.weight", "f32"),
}

# Parts of a merged projection: combined by the weight iterator, which is the only place
# that knows the concat order, so they are never renamed 1:1.
#
#   self_attn.qkv_proj             <- attn_q (queries AND gate, interleaved per head),
#                                     attn_k, attn_v
#   linear_attn.in_proj            <- attn_qkv, attn_gate, ssm_beta, ssm_alpha
#   mlp.shared_expert.gate_up_proj <- ffn_gate_shexp, ffn_up_shexp
_MERGED_PARTS: frozenset[str] = frozenset({
    "attn_q.weight",
    "attn_qkv.weight",
    "attn_gate.weight",
    "ssm_beta.weight",
    "ssm_alpha.weight",
    "ffn_gate_shexp.weight",
    "ffn_up_shexp.weight",
})

# Which layer kind each merged part belongs to, so a part appearing on the wrong kind is
# an error rather than a silent skip.
_MERGED_KIND: dict[str, str] = {
    "attn_q.weight": "full",
    "attn_qkv.weight": "ssm",
    "attn_gate.weight": "ssm",
    "ssm_beta.weight": "ssm",
    "ssm_alpha.weight": "ssm",
}

# Routed-expert stacks, read by the offload bank loader rather than yielded as parameters.
_EXPERT_SUFFIXES: frozenset[str] = frozenset({
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_down_exps.weight",
})

_GLOBAL_MAP: dict[str, tuple[str, str]] = {
    "token_embd.weight": ("model.embed_tokens.weight", "packed"),
    # Untied: output.weight is present and distinct from token_embd, so the prefill-slicing
    # GGUFLMHead applies here rather than a tied embedding read.
    "output.weight": ("lm_head.weight", "packed"),
    # The n-gram hashed per-layer embedding table, [160, 320001536] IQ4_NL.
    "per_layer_token_embd.weight": ("model.per_layer_embed.weight", "packed"),
    # Stands in for the output_norm this architecture does not have.
    "output_hc_down.weight": ("model.hc_head.down.weight", "packed"),
    "output_hc_up.weight": ("model.hc_head.up.weight", "packed"),
    "output_hc_norm.weight": ("model.hc_head.norm.weight", "f32"),
}


class Qwen4ExpLayoutError(ValueError):
    """A checkpoint that does not match the layout this adapter was written against."""


def classify(
    name: str,
    *,
    block_count: int,
    interval: int,
    ple_layers: tuple[int, ...] = (1,),
) -> tuple[str, str | None]:
    """Classify one llama.cpp tensor name.

    Returns ``(disposition, freetoken_path)``. Disposition is one of:

    ``"param"``   renamed 1:1; the path is where it lands
    ``"merged"``  part of a fused projection, combined by the weight iterator
    ``"expert"``  a routed-expert stack, read by the offload bank loader
    ``"global"``  a non-layer tensor

    Raises :class:`Qwen4ExpLayoutError` for anything unrecognised, and for a tensor that
    exists but sits on the wrong kind of layer. An ``attn_q`` on an SSM layer would mean
    the schedule is not what this adapter assumes, and quietly ignoring it is how a model
    loads at full speed and emits fluent nonsense.
    """
    if not name.startswith("blk."):
        if name in _GLOBAL_MAP:
            return "global", _GLOBAL_MAP[name][0]
        raise Qwen4ExpLayoutError(f"qwen4exp GGUF: unrecognised global tensor {name!r}")

    _, idx, suffix = name.split(".", 2)
    layer = int(idx)
    if layer >= block_count:
        raise Qwen4ExpLayoutError(
            f"qwen4exp GGUF: tensor {name!r} is on layer {layer}, past block_count "
            f"{block_count}"
        )
    kind = layer_kind(layer, interval)

    def _require(want: str) -> None:
        if kind != want:
            raise Qwen4ExpLayoutError(
                f"qwen4exp GGUF: {name!r} is a {want}-layer tensor but layer {layer} is "
                f"{kind} under full_attention_interval {interval}"
            )

    if suffix in _EXPERT_SUFFIXES:
        return "expert", None
    if suffix in _MERGED_PARTS:
        # The shared-expert halves are merged too but live on every layer, so only the
        # kind-specific parts get a schedule check.
        if suffix in _MERGED_KIND:
            _require(_MERGED_KIND[suffix])
        return "merged", None
    if suffix in _PLE_LAYER:
        if layer not in ple_layers:
            raise Qwen4ExpLayoutError(
                f"qwen4exp GGUF: PLE tensor {name!r} on layer {layer}, which is not in "
                f"ple.layers {list(ple_layers)}"
            )
        return "param", f"layers.{layer}.{_PLE_LAYER[suffix][0]}"
    for table, want in ((_FULL_ONLY, "full"), (_SSM_ONLY, "ssm")):
        if suffix in table:
            _require(want)
            return "param", f"layers.{layer}.{table[suffix][0]}"
    if suffix in _EVERY_LAYER:
        return "param", f"layers.{layer}.{_EVERY_LAYER[suffix][0]}"

    raise Qwen4ExpLayoutError(
        f"qwen4exp GGUF: unmapped tensor {name!r}; this checkpoint does not match the "
        f"layout this adapter expects"
    )


def expected_tensor_names(
    block_count: int,
    interval: int,
    ple_layers: tuple[int, ...] = (1,),
) -> set[str]:
    """Every tensor name this layout requires, for the other direction of the check.

    ``classify`` answers "does the file contain anything we cannot place"; this answers
    "does the file contain everything we need". A checkpoint missing one ``hc_ffn_inject``
    would otherwise load with an uninitialised parameter.
    """
    names = set(_GLOBAL_MAP)
    for layer in range(block_count):
        kind = layer_kind(layer, interval)
        suffixes = set(_EVERY_LAYER) | set(_EXPERT_SUFFIXES)
        if kind == "full":
            suffixes |= set(_FULL_ONLY) | {"attn_q.weight"}
        else:
            suffixes |= set(_SSM_ONLY) | {
                s for s, k in _MERGED_KIND.items() if k == "ssm"
            }
        suffixes |= {"ffn_gate_shexp.weight", "ffn_up_shexp.weight"}
        if layer in ple_layers:
            suffixes |= set(_PLE_LAYER)
        names |= {f"blk.{layer}.{s}" for s in suffixes}
    return names


def parse_gguf_geometry(shim: "GgufConfigShim") -> dict[str, Any]:
    """The geometry this architecture needs, read from the KV block and cross-checked.

    Kept separate from building a ``ModelConfig`` because there is no model class to
    configure yet. The checks are the point: each is a place where assuming rather than
    reading would give a model that loads and is wrong.
    """
    block_count = int(_kv(shim, "block_count"))
    interval = int(_kv(shim, "full_attention_interval"))
    hidden = int(_kv(shim, "embedding_length"))
    heads = int(_kv(shim, "attention.head_count"))
    kv_heads = int(_kv(shim, "attention.head_count_kv"))
    key_len = int(_kv(shim, "attention.key_length"))
    ssm_inner = int(_kv(shim, "ssm.inner_size"))
    ssm_state = int(_kv(shim, "ssm.state_size"))
    ssm_groups = int(_kv(shim, "ssm.group_count"))
    hc_count = int(_kv(shim, "hyper_connection.count"))

    full = full_attention_layers(block_count, interval)
    ratios = _kv(shim, "attention.compress_ratios") or []
    if ratios:
        # An independent statement of the same schedule. If the two disagree the layer
        # kinds are not what we think, and every per-kind tensor lands in the wrong module.
        from_ratios = tuple(i for i, v in enumerate(ratios) if v)
        if from_ratios != full:
            raise Qwen4ExpLayoutError(
                f"qwen4exp GGUF: full-attention layers derived from interval {interval} "
                f"are {list(full)} but compress_ratios marks {list(from_ratios)}"
            )

    if ssm_inner % ssm_state:
        raise Qwen4ExpLayoutError(
            f"qwen4exp GGUF: ssm inner_size {ssm_inner} is not a multiple of state_size "
            f"{ssm_state}, so the per-head vectors have no consistent length"
        )

    return {
        "block_count": block_count,
        "full_attention_interval": interval,
        "full_attention_layers": full,
        "hidden_size": hidden,
        "num_heads": heads,
        "num_kv_heads": kv_heads,
        "key_length": key_len,
        "value_length": int(_kv(shim, "attention.value_length")),
        "num_experts": int(_kv(shim, "expert_count")),
        "experts_used": int(_kv(shim, "expert_used_count")),
        "expert_ffn": int(_kv(shim, "expert_feed_forward_length")),
        "shared_expert_ffn": int(_kv(shim, "expert_shared_feed_forward_length")),
        # attn_q is [hidden, 2 * heads * key_length]: queries, then the fused gate.
        "attn_q_out": 2 * heads * key_len,
        "attn_kv_out": kv_heads * key_len,
        # The GDN fused input is q + k over the SSM groups, then v at the inner width.
        "gdn_qkv": 2 * ssm_groups * ssm_state + ssm_inner,
        "ssm_inner": ssm_inner,
        "ssm_state": ssm_state,
        "ssm_groups": ssm_groups,
        # One A/dt/alpha/beta scalar per v-head.
        "ssm_heads": ssm_inner // ssm_state,
        "hc_count": hc_count,
        "hc_low_rank": int(_kv(shim, "hyper_connection.low_rank")),
        "hc_width": hc_count * hidden,
        "indexer_heads": int(_kv(shim, "attention.indexer.head_count")),
        "indexer_key_length": int(_kv(shim, "attention.indexer.key_length")),
        "indexer_top_k": int(_kv(shim, "attention.indexer.top_k")),
        "ple_layers": tuple(int(x) for x in (_kv(shim, "ple.layers") or ())),
        "ple_ngram_size": int(_kv(shim, "ple.ngram_size", 0)),
        "ple_heads_per_ngram": int(_kv(shim, "ple.heads_per_ngram", 0)),
        "ple_input_dim": int(_kv(shim, "embedding_length_per_layer_input", 0)),
        "ple_head_offsets": tuple(int(x) for x in (_kv(shim, "ple.head_offsets") or ())),
        "ple_head_vocab_sizes": tuple(
            int(x) for x in (_kv(shim, "ple.head_vocab_sizes") or ())
        ),
        "ple_head_multipliers": tuple(
            int(x) for x in (_kv(shim, "ple.layer_multipliers") or ())
        ),
        "ple_conv_kernel": int(_kv(shim, "ple.conv_kernel", 0)),
        # The image token stands in when a position carries no text token; the reference
        # falls back to EOS for files written before that key existed.
        "ple_eos_token_id": int(_kv(shim, "ple.eos_token_id", 0)),
        "ple_image_token_id": int(_kv(shim, "ple.image_token_id", 0)),
        "rope_dimension_sections": tuple(_kv(shim, "rope.dimension_sections") or ()),
        "rope_freq_base": float(_kv(shim, "rope.freq_base", 0.0)),
        "context_length": int(_kv(shim, "context_length", 0)),
    }


def check_ple_tables(geo: dict[str, Any], table_rows: int) -> None:
    """The PLE hash-head table must line up with its offset/size metadata.

    ``head_offsets`` should be the running sum of ``head_vocab_sizes``; the physical table
    is padded past their total to a 64-row boundary (90 rows in this release). A mismatch
    means every hashed lookup lands in the wrong head's range -- a silent quality failure
    rather than a crash, so it is worth asserting rather than trusting.
    """
    offsets, sizes = geo["ple_head_offsets"], geo["ple_head_vocab_sizes"]
    if len(offsets) != len(sizes):
        raise Qwen4ExpLayoutError(
            f"qwen4exp GGUF: {len(offsets)} ple head offsets but {len(sizes)} vocab sizes"
        )
    running = 0
    for i, (off, size) in enumerate(zip(offsets, sizes)):
        if off != running:
            raise Qwen4ExpLayoutError(
                f"qwen4exp GGUF: ple head {i} offset {off} != running sum {running}"
            )
        running += size
    if table_rows < running:
        raise Qwen4ExpLayoutError(
            f"qwen4exp GGUF: per_layer_token_embd has {table_rows} rows but the hash heads "
            f"span {running}"
        )


def _expert_types_per_layer(
    model_path: str, num_layers: int
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """``(gate_up, down)`` ggml types of the routed-expert banks, one entry per layer.

    Every published qwen4exp build is a dynamic quant, so both banks vary by layer; the
    slot pool pads to the widest type and strides by it. Returns None only if the file
    cannot be read, since ``parse_gguf_config`` also runs for metadata-only inspection.
    """
    from .gguf_experts import gguf_expert_types

    try:
        types = gguf_expert_types(model_path, num_layers)
    except Exception:
        return None
    return (tuple(int(t) for t in types["gate_up"]), tuple(int(t) for t in types["down"]))


def parse_gguf_config(shim: "GgufConfigShim") -> ModelConfig:
    """Build a ModelConfig from a qwen4exp checkpoint.

    Every block is a decoder layer -- unlike qwen35moe there is no NextN/MTP block to drop.

    Two decisions worth stating because they are choices, not readings of the file:

    * ``rope.dimension_sections`` is ignored. qwen4exp applies ggml_rope_multi, but all
      four position axes carry the same value for a text-only batch, and ggml advances
      every per-axis theta by the same theta_scale with no per-section reset (the reset is
      the vision path). So which section a dimension falls in stops mattering and the
      result is exactly NeoX RoPE. If image input is ever added this stops being true.
    * ``use_qk_norm`` is True: attn_q_norm / attn_k_norm are per-head RMS norms over
      key_length, present on every full-attention layer.
    """
    geo = parse_gguf_geometry(shim)
    num_layers = geo["block_count"]

    full_ids = geo["full_attention_layers"]
    linear_ids = tuple(i for i in range(num_layers) if i not in set(full_ids))

    full_rotary = RotaryConfig(
        head_dim=geo["key_length"],
        rotary_dim=int(_kv(shim, "rope.dimension_count")),
        max_position=geo["context_length"],
        base=geo["rope_freq_base"],
        scaling=None,
    )
    groups = tuple(
        sorted(
            (
                FullAttentionGroupConfig(
                    name="full",
                    layer_ids=full_ids,
                    num_kv_heads=geo["num_kv_heads"],
                    head_dim=geo["key_length"],
                    rotary_config=full_rotary,
                ),
                LinearGatedDeltaGroupConfig(
                    name="linear",
                    layer_ids=linear_ids,
                    num_key_heads=geo["ssm_groups"],
                    num_value_heads=geo["ssm_heads"],
                    key_head_dim=geo["ssm_state"],
                    value_head_dim=geo["ssm_state"],
                    conv_kernel_dim=int(_kv(shim, "ssm.conv_kernel")),
                    output_gate=True,
                ),
            ),
            key=lambda g: g.layer_ids[0] if g.layer_ids else 1 << 30,
        )
    )

    config = ModelConfig(
        num_layers=num_layers,
        num_qo_heads=geo["num_heads"],
        num_kv_heads=geo["num_kv_heads"],
        head_dim=geo["key_length"],
        hidden_size=geo["hidden_size"],
        vocab_size=shim.vocab_size,
        intermediate_size=0,
        hidden_act="silu",
        rms_norm_eps=float(_kv(shim, "attention.layer_norm_rms_epsilon")),
        tie_word_embeddings=shim.tie_word_embeddings,
        rotary_config=full_rotary,
        num_experts=geo["num_experts"],
        num_experts_per_tok=geo["experts_used"],
        moe_intermediate_size=geo["expert_ffn"],
        shared_expert_intermediate_size=geo["shared_expert_ffn"],
        norm_topk_prob=True,
        moe_enabled=True,
        use_qk_norm=True,
        model_type=shim.model_type,
        architectures=list(shim.architectures),
        vision_config=None,
        image_token_id=None,
        attention_groups=groups,
        expert_quant="gguf",
        gguf_expert_types=_expert_types_per_layer(shim.model_path, num_layers),
        gguf_model_path=shim.model_path,
        weight_block_size=None,
        attn_quant="gguf",
        dense_quant="gguf",
        lm_head_quant="gguf",
    )

    return config


def geometry_from_path(model_path: str) -> dict[str, Any]:
    """The qwen4exp-only geometry, re-read from the checkpoint.

    ModelConfig is a frozen dataclass with no field for hyper-connection, indexer or PLE
    geometry, and it is the only object a module constructor receives. Rather than smuggle
    a dict onto it, the model re-derives these from ``config.gguf_model_path`` -- which it
    must hold anyway, because the per-tensor ggml types are only in the file. Reading the
    KV block is a metadata-only load, not a weight load.
    """
    from freetoken.models.gguf.config import GgufConfigShim
    from freetoken.models.gguf.reader import gguf_architecture, load_gguf_metadata

    shim = GgufConfigShim(
        architectures=[],
        model_path=model_path,
        model_type=gguf_architecture(model_path),
        metadata=load_gguf_metadata(model_path),
        vocab_size=0,
        tie_word_embeddings=False,
    )
    return parse_gguf_geometry(shim)


# --------------------------------------------------------------------------------------
# Weight loading
#
# The GDN tensors are V-reordered exactly as qwen35moe's are: llama.cpp's converter has
# qwen4exp inherit ``_LinearAttentionVReorderBase`` (conversion/qwen4exp.py), the same base
# qwen35moe uses. So the un-tiling helpers there apply unchanged and are imported rather
# than re-derived.
# --------------------------------------------------------------------------------------


def _q35():
    """The qwen35moe GGUF helpers, imported lazily to keep import order simple."""
    from freetoken.models.qwen3_5_moe import gguf as m

    return m


def _scan_types(model_path: str) -> dict[tuple[int, str], int]:
    return _q35()._scan_quant_types(model_path)


def convert_qwen4exp_to_gguf(model, config, *, model_path: str) -> None:
    """Swap dense ops for GGUF-quant ops so the packed buffers have somewhere to land.

    Only the projections become GGUF ops. Everything the checkpoint stores small and dense
    -- the hyper-connection weights, the norms, the routers -- stays an ordinary tensor and
    is dequantised at load time, matching how qwen35moe treats its norms and gates.
    """
    from freetoken.layers.gguf import GGUFEmbedding, GGUFLinear, GGUFLMHead, gguf_merged_or_plain

    q35 = _q35()
    types = _scan_types(model_path)
    geo = geometry_from_path(model_path)
    H = config.hidden_size

    def qt(layer: int, suffix: str) -> int:
        key = (layer, suffix)
        if key not in types:
            raise Qwen4ExpLayoutError(
                f"qwen4exp GGUF: no quant type for blk.{layer}.{suffix}; the checkpoint "
                f"does not match the layout this adapter expects"
            )
        return types[key]

    # Read the compute dtype off a tensor the engine allocated under its own
    # torch_dtype(config.dtype) context, rather than trusting the ambient default here.
    mdtype = model.model.hc_head.norm.dtype
    gt = types[(-1, "token_embd.weight")]
    model.model.embed_tokens = GGUFEmbedding(
        config.vocab_size, H, quant_type=gt, dtype=mdtype
    )
    if not config.tie_word_embeddings:
        # Untied: slice to the last prefill position before the projection, or a long
        # prompt materialises [prompt_len, vocab] logits for no reason.
        model.lm_head = GGUFLMHead(
            H, config.vocab_size, quant_type=types[(-1, "output.weight")]
        )

    full_ids = set(geo["full_attention_layers"])
    n_head, n_kv, hd = config.num_qo_heads, config.num_kv_heads, config.head_dim
    g = config.linear_attention_group()

    for i, layer in enumerate(model.model.layers.op_list):
        if i in full_ids:
            # q carries its own gate, so the q half is twice the head width.
            layer.self_attn.qkv_proj = gguf_merged_or_plain(
                H,
                [2 * n_head * hd, n_kv * hd, n_kv * hd],
                [qt(i, "attn_q.weight"), qt(i, "attn_k.weight"), qt(i, "attn_v.weight")],
            )
            layer.self_attn.o_proj = GGUFLinear(
                n_head * hd, H, quant_type=qt(i, "attn_output.weight")
            )
        else:
            qkv = g.num_key_heads * g.key_head_dim * 2 + g.num_value_heads * g.value_head_dim
            layer.linear_attn.in_proj = gguf_merged_or_plain(
                H,
                [qkv, g.num_value_heads * g.value_head_dim, g.num_value_heads, g.num_value_heads],
                [
                    qt(i, "attn_qkv.weight"),
                    qt(i, "attn_gate.weight"),
                    qt(i, "ssm_beta.weight"),
                    qt(i, "ssm_alpha.weight"),
                ],
            )
            # out_proj stays dense: its COLUMNS index the V-head dimension, and a column
            # permutation on packed blocks is unsafe -- a 128-wide head straddles quant
            # block boundaries. Same reason qwen35moe leaves it dense.

        layer.mlp.shared_expert.gate_up_proj = gguf_merged_or_plain(
            H,
            [config.shared_expert_intermediate_size] * 2,
            [qt(i, "ffn_gate_shexp.weight"), qt(i, "ffn_up_shexp.weight")],
        )
        layer.mlp.shared_expert.down_proj = GGUFLinear(
            config.shared_expert_intermediate_size, H, quant_type=qt(i, "ffn_down_shexp.weight")
        )


def _kv_meta(model_path: str, key: str):
    from freetoken.models.gguf.reader import gguf_architecture, load_gguf_metadata

    return load_gguf_metadata(model_path)[f"{gguf_architecture(model_path)}.{key}"]


def iter_gguf_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
):
    """Yield ``(freetoken_param_name, tensor)`` for every non-expert qwen4exp tensor.

    Routed experts are served from the offload cache and are skipped here, as in qwen35moe.

    The GDN tensors need the same V-head un-tiling qwen35moe applies: llama.cpp's converter
    has qwen4exp inherit ``_LinearAttentionVReorderBase``, so the reorder is identical and
    the helpers are imported rather than re-derived. Row permutations are safe on packed
    blocks (each row is its own run of blocks); ``ssm_out`` is the exception, because its
    COLUMNS index the V heads and a 128-wide head straddles quant block boundaries -- it is
    dequantised, un-tiled dense, and yielded as a plain ``.weight``.

    Indexer tensors are skipped deliberately: this adapter runs dense attention, which is
    exactly equivalent below ``indexer_top_k`` (2048) tokens. A silent skip would be
    indistinguishable from having forgotten them, so it is counted and logged.
    """
    import torch
    from freetoken.models.gguf.reader import iter_gguf_tensors

    q35 = _q35()
    _to_bf16, _to_f32 = q35._to_bf16, q35._to_f32
    _dequant_any, _ungroup_v = q35._dequant_any, q35._ungroup_v
    _ungroup_packed_rows = q35._ungroup_packed_rows

    assert not include_moe_experts, (
        "qwen4exp keeps its routed experts in the offload cache; they are loaded by the "
        "expert-bank loader, not by iter_gguf_weights."
    )
    assert include_non_moe

    geo = geometry_from_path(model_path)
    quant_map = _scan_types(model_path)
    block_count = geo["block_count"]
    full_ids = set(geo["full_attention_layers"])
    ple_layers = geo["ple_layers"]

    vK, vN, vD = geo["ssm_groups"], geo["ssm_heads"], geo["ssm_state"]
    vR = vN // vK
    untile = vK != vN
    qk_rows = 2 * vK * vD
    qkv_size = geo["gdn_qkv"]
    conv_kernel = int(_kv_meta(model_path, "ssm.conv_kernel"))

    qkv_buf: dict = {}
    in_proj_buf: dict = {}
    gate_up_buf: dict = {}
    skipped_indexer = 0

    for t in iter_gguf_tensors(model_path):
        name = t.name

        if not name.startswith("blk."):
            if name == "token_embd.weight":
                yield "model.embed_tokens.qweight", t.packed()
            elif name == "output.weight":
                yield "lm_head.qweight", t.packed()
            elif name == "output_hc_norm.weight":
                yield "model.hc_head.norm", _to_bf16(t)
            elif name == "output_hc_down.weight":
                yield "model.hc_head.down", _dequant_any(t)
            elif name == "output_hc_up.weight":
                yield "model.hc_head.up", _dequant_any(t)
            elif name == "per_layer_token_embd.weight":
                # 28.8 GB of packed rows (320001536 x 90 B). It is never resident on the
                # GPU -- a token gathers 16 rows of 160 values, about 1.4 kB -- so it is
                # deliberately kept out of the state dict and mmapped by the PLE block
                # instead. Yielding it here would try to move 28.8 GB onto a 22 GB card.
                continue
            else:
                raise Qwen4ExpLayoutError(f"qwen4exp GGUF: unrecognised global {name!r}")
            continue

        _, idx, suffix = name.split(".", 2)
        layer = int(idx)
        if layer >= block_count:
            raise Qwen4ExpLayoutError(
                f"qwen4exp GGUF: {name!r} is past block_count {block_count}"
            )
        base = f"model.layers.{layer}"

        if suffix in _EXPERT_SUFFIXES:
            continue
        if suffix.startswith("indexer."):
            skipped_indexer += 1
            continue

        # Hyper-connections, on every layer. Small and dense in the module, so dequantised.
        if suffix.startswith("hc_"):
            which, part = suffix[3:].split("_", 1)
            part = part.replace(".weight", "")
            dest = f"{base}.hc_{which}.{part}"
            yield dest, (_to_bf16(t) if part in ("norm", "inject") else _dequant_any(t))
            continue

        if suffix in _PLE_LAYER:
            if layer not in ple_layers:
                raise Qwen4ExpLayoutError(
                    f"qwen4exp GGUF: PLE tensor {name!r} on a layer that is not in "
                    f"ple.layers {list(ple_layers)}"
                )
            part = suffix.replace(".weight", "")[4:]
            if part in ("key", "value"):
                yield f"model.ple.{part}.qweight", t.packed()
            else:
                yield f"model.ple.{part}", _to_bf16(t)
            continue

        # MoE router and shared expert, on every layer.
        if suffix == "ffn_gate_inp.weight":
            yield f"{base}.mlp.gate.weight", _to_bf16(t)
            continue
        if suffix == "ffn_gate_inp_shexp.weight":
            # [hidden] in the file; the module is Linear(hidden, 1).
            yield f"{base}.mlp.shared_expert_gate.weight", _to_bf16(t).reshape(1, -1)
            continue
        if suffix == "ffn_down_shexp.weight":
            yield f"{base}.mlp.shared_expert.down_proj.qweight", t.packed()
            continue
        if suffix in ("ffn_gate_shexp.weight", "ffn_up_shexp.weight"):
            slot = "gate" if suffix.startswith("ffn_gate") else "up"
            gate_up_buf.setdefault(layer, {})[slot] = t.packed()
            slots = gate_up_buf.get(layer)
            if slots and "gate" in slots and "up" in slots:
                types = [
                    quant_map.get((layer, "ffn_gate_shexp.weight")),
                    quant_map.get((layer, "ffn_up_shexp.weight")),
                ]
                dest = f"{base}.mlp.shared_expert.gate_up_proj"
                if len(set(types)) == 1:
                    yield f"{dest}.qweight", torch.cat([slots["gate"], slots["up"]], dim=0)
                else:
                    yield f"{dest}.qweight_0", slots["gate"]
                    yield f"{dest}.qweight_1", slots["up"]
                del gate_up_buf[layer]
            continue

        if layer in full_ids:
            if suffix in ("attn_q.weight", "attn_k.weight", "attn_v.weight"):
                part = suffix.split("_")[1][0]
                qkv_buf.setdefault(layer, {})[part] = t.packed()
                slots = qkv_buf.get(layer)
                if slots and {"q", "k", "v"} <= set(slots):
                    types = [
                        quant_map.get((layer, f"attn_{p}.weight")) for p in ("q", "k", "v")
                    ]
                    dest = f"{base}.self_attn.qkv_proj"
                    if len(set(types)) == 1:
                        yield f"{dest}.qweight", torch.cat(
                            [slots["q"], slots["k"], slots["v"]], dim=0
                        )
                    else:
                        for i, p in enumerate(("q", "k", "v")):
                            yield f"{dest}.qweight_{i}", slots[p]
                    del qkv_buf[layer]
            elif suffix == "attn_output.weight":
                yield f"{base}.self_attn.o_proj.qweight", t.packed()
            elif suffix == "attn_q_norm.weight":
                yield f"{base}.self_attn.q_norm.weight", _to_bf16(t)
            elif suffix == "attn_k_norm.weight":
                yield f"{base}.self_attn.k_norm.weight", _to_bf16(t)
            else:
                raise Qwen4ExpLayoutError(
                    f"qwen4exp GGUF: unmapped full-attention tensor {name!r}"
                )
            continue

        # GDN layers.
        if suffix == "attn_qkv.weight":
            w = t.packed()
            if untile:
                w = torch.cat(
                    [w[:qk_rows], _ungroup_packed_rows(w[qk_rows:], vK, vR, vD)], dim=0
                )
            in_proj_buf.setdefault(layer, {})["qkv"] = w
        elif suffix == "attn_gate.weight":
            w = t.packed()
            if untile:
                w = _ungroup_packed_rows(w, vK, vR, vD)
            in_proj_buf.setdefault(layer, {})["gate"] = w
        elif suffix in ("ssm_beta.weight", "ssm_alpha.weight"):
            w = t.packed()
            if untile:
                w = _ungroup_packed_rows(w, vK, vR, 1)  # one row per V head
            in_proj_buf.setdefault(layer, {})[
                "beta" if suffix.startswith("ssm_beta") else "alpha"
            ] = w
        elif suffix == "ssm_out.weight":
            w = _dequant_any(t)
            if untile:
                w = _ungroup_v(w, 1, vK, vR, vD)
            yield f"{base}.linear_attn.out_proj.weight", w
            continue
        elif suffix == "ssm_a":
            a = _to_f32(t)
            if untile:
                a = _ungroup_v(a, 0, vK, vR, 1)
            if torch.any(a >= 0):
                raise Qwen4ExpLayoutError(
                    f"qwen4exp GGUF: blk.{layer}.ssm_a has non-negative entries, which "
                    f"would make log(-A) NaN"
                )
            yield f"{base}.linear_attn.A_log", torch.log(-a)
            continue
        elif suffix == "ssm_dt.bias":
            dt = _to_f32(t)
            if untile:
                dt = _ungroup_v(dt, 0, vK, vR, 1)
            yield f"{base}.linear_attn.dt_bias", dt
            continue
        elif suffix == "ssm_norm.weight":
            yield f"{base}.linear_attn.norm.weight", _to_bf16(t)
            continue
        elif suffix == "ssm_conv1d.weight":
            w = _to_bf16(t).reshape(qkv_size, conv_kernel)
            if untile:
                qk, v = w[:qk_rows], w[qk_rows:]
                w = torch.cat([qk, _ungroup_v(v, 0, vK, vR, vD)], dim=0)
            yield f"{base}.linear_attn.conv1d.weight", w.reshape(qkv_size, 1, conv_kernel)
            continue
        else:
            raise Qwen4ExpLayoutError(f"qwen4exp GGUF: unmapped GDN tensor {name!r}")

        slots = in_proj_buf.get(layer)
        if slots and {"qkv", "gate", "beta", "alpha"} <= set(slots):
            parts = ("qkv", "gate", "beta", "alpha")
            src = ("attn_qkv", "attn_gate", "ssm_beta", "ssm_alpha")
            types = [quant_map.get((layer, f"{s}.weight")) for s in src]
            dest = f"{base}.linear_attn.in_proj"
            if len(set(types)) == 1:
                yield f"{dest}.qweight", torch.cat([slots[p] for p in parts], dim=0)
            else:
                for i, p in enumerate(parts):
                    yield f"{dest}.qweight_{i}", slots[p]
            del in_proj_buf[layer]

    if qkv_buf or in_proj_buf or gate_up_buf:
        raise Qwen4ExpLayoutError(
            f"qwen4exp GGUF: incomplete fused projections left over -- "
            f"qkv {sorted(qkv_buf)}, in_proj {sorted(in_proj_buf)}, "
            f"shared_expert {sorted(gate_up_buf)}"
        )
    if skipped_indexer:
        from freetoken.utils import init_logger

        init_logger(__name__).info_rank0(
            f"qwen4exp: skipped {skipped_indexer} indexer tensors; attention runs dense, "
            f"which is exact below indexer_top_k ({geo['indexer_top_k']}) tokens"
        )


def ple_table_rows(model_path: str) -> int:
    """Rows in ``per_layer_token_embd``, read from the tensor table.

    Not derivable from the metadata: the head vocab sizes sum to 320001446 while the table
    is 320001536 rows, padded to a 64-row boundary. Sizing the embedding from the sum would
    leave the last 90 rows unaddressable and, worse, mis-size the packed buffer.
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors

    for t in iter_gguf_tensors(model_path):
        if t.name == "per_layer_token_embd.weight":
            # GGUF dims are fastest-first; the row count is the slow axis.
            return int(t.shape[-1]) if len(t.shape) > 1 else int(t.shape[0])
    raise Qwen4ExpLayoutError("qwen4exp GGUF: per_layer_token_embd.weight not found")


def ple_quant_types(model_path: str, ple_layer: int) -> dict:
    """ggml types of the three PLE tensors that stay packed."""
    types = _scan_types(model_path)
    out = {}
    for key, src in (
        ("table", (-1, "per_layer_token_embd.weight")),
        ("key", (ple_layer, "ple_key.weight")),
        ("value", (ple_layer, "ple_value.weight")),
    ):
        if src not in types:
            raise Qwen4ExpLayoutError(f"qwen4exp GGUF: no quant type for {src[1]!r}")
        out[key] = types[src]
    return out
