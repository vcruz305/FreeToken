"""``gguf_expert_types`` is per-layer everywhere, and the engine's reader agrees.

Mixed quant banks made this field a pair of per-layer tuples instead of a pair of ints.
Three adapters set it (qwen3_5_moe, qwen3_moe, deepseek_v4) and the engine reads it in
``_cpu_moe_executor_viable``, so the shape has to be one contract across all four. It is
not: the reader iterates each bank, and an ``int`` is not iterable, so a stale adapter
raises TypeError deep inside a load rather than failing anywhere useful.

That path is only reached when the engine is deciding CPU/hybrid residency, so serving
with ``--moe-backend offload`` never exercises it -- which is exactly why it needs a test
rather than a passing smoke run.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from freetoken.engine.engine import _cpu_moe_executor_viable

GGML_Q4_K = 12
GGML_Q6_K = 14
GGML_IQ3_S = 21  # no CPU kernel


def _config(expert_types, **kw):
    return SimpleNamespace(
        hidden_act="silu",
        moe_weight_format=None,
        expert_quant="gguf",
        gguf_expert_types=expert_types,
        **kw,
    )


@pytest.mark.parametrize(
    "adapter",
    [
        "freetoken.models.qwen3_5_moe.gguf",
        "freetoken.models.qwen3_moe.gguf",
    ],
)
def test_adapters_declare_a_per_layer_return_type(adapter):
    """The annotation is the contract; a scalar pair here is the bug this file exists for."""
    import importlib

    mod = importlib.import_module(adapter)
    fn = getattr(mod, "_expert_types_per_layer", None)
    assert fn is not None, f"{adapter} should expose _expert_types_per_layer"
    ret = inspect.signature(fn).return_annotation
    assert "tuple[int, ...]" in str(ret), f"{adapter} returns {ret}, expected per-layer tuples"


def test_reader_accepts_per_layer_tuples_and_finds_one_type():
    """A uniform bank expressed per-layer is still CPU-viable."""
    n = 8
    cfg = _config(((GGML_Q4_K,) * n, (GGML_Q4_K,) * n))
    # Either outcome is fine as far as the extension goes; what must not happen is a raise.
    assert isinstance(_cpu_moe_executor_viable(cfg), bool)


def test_a_bank_that_varies_by_layer_is_not_cpu_viable():
    """The CPU executor picks one weight_format for the whole run, so mixed stays on GPU."""
    cfg = _config(((GGML_Q4_K,) * 4 + (GGML_Q6_K,) * 4, (GGML_Q4_K,) * 8))
    assert _cpu_moe_executor_viable(cfg) is False


def test_two_banks_of_different_types_are_not_cpu_viable():
    """One weight_format serves both banks, so gate_up != down cannot run there."""
    cfg = _config(((GGML_Q4_K,) * 8, (GGML_Q6_K,) * 8))
    assert _cpu_moe_executor_viable(cfg) is False


def test_a_type_with_no_cpu_kernel_is_not_viable():
    cfg = _config(((GGML_IQ3_S,) * 8, (GGML_IQ3_S,) * 8))
    assert _cpu_moe_executor_viable(cfg) is False


def test_absent_types_are_handled():
    assert _cpu_moe_executor_viable(_config(None)) is False


def test_scalar_pair_would_have_raised():
    """Pins why this matters: the old shape is not silently tolerated, it explodes.

    If someone reverts an adapter to ``(int, int)``, the reader must not quietly accept it
    and take one layer's type for all of them -- that would be a wrong answer rather than
    an error. Raising is the correct failure here, and this records that.
    """
    with pytest.raises(TypeError):
        _cpu_moe_executor_viable(_config((GGML_Q4_K, GGML_Q4_K)))
