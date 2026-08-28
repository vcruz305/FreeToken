"""qwen4exp layout, reconciled both ways against a real checkpoint's tensor table.

The fixture is the complete tensor listing of ``unsloth/Qwen3.8-Flash-Next-GGUF``
UD-IQ1_S, read from the shard headers over HTTP range requests. Names only -- no weights,
so this runs anywhere.

Both directions matter, and they fail differently:

* every name in the file must be placed. An unplaced tensor is a weight that never reaches
  the model, which loads fine and degrades quality silently.
* every name the layout requires must be in the file. A missing one is a parameter left
  uninitialised, which also loads fine.

Neither shows up as a crash, which is why this is asserted rather than assumed. Nothing
here has been run against real weights: the smallest published variant is 72.5 GB.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from freetoken.models.qwen4exp.gguf import (
    Qwen4ExpLayoutError,
    parse_gguf_geometry,
    check_ple_tables,
    classify,
    expected_tensor_names,
    full_attention_layers,
    layer_kind,
)

BLOCK_COUNT = 48
INTERVAL = 4
PLE_LAYERS = (1,)
FIXTURE = Path(__file__).parent / "data" / "qwen4exp_tensor_names.txt"


@pytest.fixture(scope="module")
def real_table() -> dict[str, list[int]]:
    """``{tensor name: dims}`` for the whole checkpoint."""
    table = {}
    for line in FIXTURE.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, dims, _type = line.split("	")
        table[name] = [int(x) for x in dims.split(",")]
    # The checkpoint's own split.tensors.count.
    assert len(table) == 1224, f"fixture has {len(table)} tensors, expected 1224"
    return table


@pytest.fixture(scope="module")
def real_names(real_table) -> list[str]:
    return list(real_table)


# The KV block this checkpoint declares, for the geometry cross-check.
KV = {
    "qwen4exp.block_count": 48,
    "qwen4exp.full_attention_interval": 4,
    "qwen4exp.embedding_length": 2560,
    "qwen4exp.attention.head_count": 24,
    "qwen4exp.attention.head_count_kv": 2,
    "qwen4exp.attention.key_length": 256,
    "qwen4exp.attention.value_length": 256,
    "qwen4exp.expert_count": 512,
    "qwen4exp.expert_used_count": 10,
    "qwen4exp.expert_feed_forward_length": 640,
    "qwen4exp.expert_shared_feed_forward_length": 640,
    "qwen4exp.ssm.inner_size": 6144,
    "qwen4exp.ssm.state_size": 128,
    "qwen4exp.ssm.group_count": 16,
    "qwen4exp.hyper_connection.count": 4,
    "qwen4exp.hyper_connection.low_rank": 320,
    "qwen4exp.attention.indexer.head_count": 4,
    "qwen4exp.attention.indexer.key_length": 128,
    "qwen4exp.attention.indexer.top_k": 2048,
    "qwen4exp.attention.compress_ratios": [
        0 if i % 4 != 3 else 4 for i in range(48)
    ],
    "qwen4exp.ple.layers": [1],
}


class _Shim:
    def __init__(self, kv):
        self._kv = kv

    def get(self, key, default=None):
        return self._kv.get(key, default)


def test_every_tensor_in_the_file_is_placed(real_names):
    unplaced = []
    for name in real_names:
        try:
            classify(name, block_count=BLOCK_COUNT, interval=INTERVAL, ple_layers=PLE_LAYERS)
        except Qwen4ExpLayoutError as exc:
            unplaced.append(f"{name}: {exc}")
    assert not unplaced, "unplaced tensors:\n" + "\n".join(unplaced[:20])


def test_layout_requires_nothing_the_file_lacks(real_names):
    missing = expected_tensor_names(BLOCK_COUNT, INTERVAL, PLE_LAYERS) - set(real_names)
    assert not missing, f"layout expects {len(missing)} absent tensors: {sorted(missing)[:20]}"


def test_layout_invents_nothing_the_file_does_not_have(real_names):
    extra = set(real_names) - expected_tensor_names(BLOCK_COUNT, INTERVAL, PLE_LAYERS)
    assert not extra, f"file has {len(extra)} tensors outside the layout: {sorted(extra)[:20]}"


def test_dispositions_partition_the_checkpoint(real_names):
    """The counts are the arithmetic that closes: 12*10 + 36*9 + 48*16 + 6 + 6 == 1224."""
    counts: dict[str, int] = {}
    for name in real_names:
        kind, _ = classify(
            name, block_count=BLOCK_COUNT, interval=INTERVAL, ple_layers=PLE_LAYERS
        )
        counts[kind] = counts.get(kind, 0) + 1
    assert sum(counts.values()) == 1224
    # 3 stacks on each of 48 layers.
    assert counts["expert"] == 144
    assert counts["global"] == 6
    # attn_q on 12 full layers; attn_qkv/attn_gate/ssm_beta/ssm_alpha on 36 SSM layers;
    # ffn_gate_shexp/ffn_up_shexp on all 48.
    assert counts["merged"] == 12 + 4 * 36 + 2 * 48
    assert counts["param"] == 1224 - counts["expert"] - counts["global"] - counts["merged"]


def test_full_attention_schedule():
    full = full_attention_layers(BLOCK_COUNT, INTERVAL)
    assert full == (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47)
    assert len(full) == 12
    assert all(layer_kind(i, INTERVAL) == "full" for i in full)
    assert sum(layer_kind(i, INTERVAL) == "ssm" for i in range(BLOCK_COUNT)) == 36


def test_full_layers_carry_no_gate_tensor_and_ssm_layers_do(real_names):
    """The gate is fused into attn_q on full layers, which is why they lack attn_gate."""
    have_gate = {int(n.split(".")[1]) for n in real_names if n.endswith("attn_gate.weight")}
    have_q = {int(n.split(".")[1]) for n in real_names if n.endswith("attn_q.weight")}
    assert not (have_gate & have_q)
    assert have_q == set(full_attention_layers(BLOCK_COUNT, INTERVAL))
    assert have_gate | have_q == set(range(BLOCK_COUNT))


def test_no_standalone_norms_anywhere(real_names):
    """Hyper-connection norms stand in for attn_norm/ffn_norm/output_norm.

    If a future build reintroduces them, the hyper-connection block is not doing what this
    adapter assumes and the residual path needs rechecking.
    """
    def suffix_of(name: str) -> str:
        # endswith is wrong here: hc_attn_norm.weight ends with attn_norm.weight.
        return name.split(".", 2)[2] if name.startswith("blk.") else name

    for absent in ("attn_norm.weight", "ffn_norm.weight", "output_norm.weight"):
        assert not [n for n in real_names if suffix_of(n) == absent], (
            f"{absent} unexpectedly present"
        )
    assert len([n for n in real_names if n.endswith("hc_attn_norm.weight")]) == BLOCK_COUNT
    assert "output_hc_norm.weight" in real_names


def test_a_tensor_on_the_wrong_layer_kind_is_an_error():
    """Silently skipping it would load at full speed and emit fluent nonsense."""
    # layer 0 is SSM under interval 4, so a full-attention tensor there is a contradiction.
    with pytest.raises(Qwen4ExpLayoutError, match="full-layer tensor"):
        classify("blk.0.attn_q.weight", block_count=BLOCK_COUNT, interval=INTERVAL)
    # and the converse.
    with pytest.raises(Qwen4ExpLayoutError, match="ssm-layer tensor"):
        classify("blk.3.ssm_out.weight", block_count=BLOCK_COUNT, interval=INTERVAL)


def test_unknown_and_out_of_range_tensors_are_errors():
    with pytest.raises(Qwen4ExpLayoutError, match="unmapped"):
        classify("blk.0.attn_wat.weight", block_count=BLOCK_COUNT, interval=INTERVAL)
    with pytest.raises(Qwen4ExpLayoutError, match="unrecognised global"):
        classify("something_else.weight", block_count=BLOCK_COUNT, interval=INTERVAL)
    with pytest.raises(Qwen4ExpLayoutError, match="past block_count"):
        classify("blk.48.ffn_gate_inp.weight", block_count=BLOCK_COUNT, interval=INTERVAL)


def test_ple_tensors_only_on_the_declared_layer():
    kind, path = classify("blk.1.ple_key.weight", block_count=BLOCK_COUNT, interval=INTERVAL)
    assert kind == "param" and path == "layers.1.ple.key.weight"
    with pytest.raises(Qwen4ExpLayoutError, match="not in ple.layers"):
        classify("blk.2.ple_key.weight", block_count=BLOCK_COUNT, interval=INTERVAL)


def test_ple_head_tables_are_consistent():
    """Offsets are the running sum of sizes; the table is padded to a 64-row boundary."""
    sizes = (20000003, 20000023, 20000033)
    offsets = (0, 20000003, 40000026)
    geo = {"ple_head_offsets": offsets, "ple_head_vocab_sizes": sizes}
    check_ple_tables(geo, table_rows=sum(sizes) + 90)

    with pytest.raises(Qwen4ExpLayoutError, match="running sum"):
        check_ple_tables(
            {"ple_head_offsets": (0, 1, 2), "ple_head_vocab_sizes": sizes}, table_rows=10**9
        )
    with pytest.raises(Qwen4ExpLayoutError, match="offsets but"):
        check_ple_tables(
            {"ple_head_offsets": (0,), "ple_head_vocab_sizes": sizes}, table_rows=10**9
        )
    with pytest.raises(Qwen4ExpLayoutError, match="rows but the hash heads span"):
        check_ple_tables(geo, table_rows=sum(sizes) - 1)


def test_arch_is_not_registered_yet():
    """No model class exists, so claiming support would turn a clear refusal into a crash."""
    from freetoken.models.gguf.config import GGUF_ARCH_TO_REGISTRY

    assert "qwen4exp" not in GGUF_ARCH_TO_REGISTRY


def test_derived_widths_match_the_real_tensor_shapes(real_table):
    """Arithmetic from the KV block must reproduce the shapes actually in the file.

    This is what confirms the structural readings rather than leaving them as plausible
    stories: the gate fused into attn_q (2 * 24 * 256 = 12288), the GDN input decomposition
    (2 * 16 * 128 + 6144 = 10240), and the four residual streams flattened into the
    hyper-connection width (4 * 2560 = 10240). Each is a place where a wrong reading gives
    a model that loads and is quietly wrong.
    """
    geo = parse_gguf_geometry(_Shim(KV))
    cases = [
        ("blk.3.attn_q.weight", geo["attn_q_out"]),
        ("blk.3.attn_k.weight", geo["attn_kv_out"]),
        ("blk.3.attn_v.weight", geo["attn_kv_out"]),
        ("blk.0.attn_qkv.weight", geo["gdn_qkv"]),
        ("blk.0.attn_gate.weight", geo["ssm_inner"]),
        ("blk.0.hc_attn_norm.weight", geo["hc_width"]),
        ("blk.0.hc_ffn_norm.weight", geo["hc_width"]),
        ("blk.0.ssm_a", geo["ssm_heads"]),
        ("blk.0.ssm_dt.bias", geo["ssm_heads"]),
        ("blk.3.indexer.q_proj.weight", geo["indexer_heads"] * geo["indexer_key_length"]),
        ("blk.3.indexer.k_proj.weight", geo["indexer_key_length"]),
    ]
    for name, derived in cases:
        assert real_table[name][-1] == derived, (
            f"{name}: file says {real_table[name]}, geometry derives {derived}"
        )
    # hc down/up are the low-rank mixer between hc_width and low_rank.
    assert real_table["blk.0.hc_attn_down.weight"] == [geo["hc_width"], geo["hc_low_rank"]]
    assert real_table["blk.0.hc_attn_up.weight"] == [geo["hc_low_rank"], geo["hc_width"]]
    # inject emits one weight per residual stream.
    assert real_table["blk.0.hc_attn_inject.weight"] == [geo["hc_width"], geo["hc_count"]]


def test_schedule_disagreement_is_rejected(real_table):
    """compress_ratios is an independent statement of the layer schedule.

    If it disagrees with the interval, the layer kinds are not what we think and every
    per-kind tensor lands in the wrong module, so it must fail loudly rather than pick one.
    """
    bad = dict(KV)
    bad["qwen4exp.attention.compress_ratios"] = [
        0 if i % 4 != 2 else 4 for i in range(48)
    ]
    with pytest.raises(Qwen4ExpLayoutError, match="compress_ratios marks"):
        parse_gguf_geometry(_Shim(bad))


def test_untied_lm_head(real_table):
    """output.weight is separate from token_embd, so the prefill-slicing head applies."""
    assert real_table["output.weight"] == real_table["token_embd.weight"]
    assert "output.weight" in real_table and "token_embd.weight" in real_table
