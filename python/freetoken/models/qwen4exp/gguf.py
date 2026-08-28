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
  24*256 the head geometry calls for, because it carries its own gate concatenated after
  the queries. Those layers have no separate ``attn_gate`` while the SSM layers do, and
  that presence/absence pattern is what makes the fusion unambiguous.
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

if TYPE_CHECKING:
    from freetoken.models.gguf.config import GgufConfigShim

_ARCH = "qwen4exp"


def _kv(shim: "GgufConfigShim", key: str, default: Any = None) -> Any:
    """One ``qwen4exp.*`` metadata value, or ``default`` when absent."""
    return shim.get(f"{_ARCH}.{key}", default)


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
#   self_attn.qkv_proj             <- attn_q (queries AND gate), attn_k, attn_v
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
