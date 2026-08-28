"""qwen4exp -- the architecture behind Qwen3.8-Flash-Next.

A qwen35moe core -- gated delta net on the linear layers, gated attention on the full
ones, routed MoE with a gated shared expert -- wrapped in a hyper-connection residual path,
with n-gram hashed per-layer embeddings on one layer.

Served from GGUF only; the hyper-connection, indexer and PLE geometry is read from the
checkpoint rather than from a config file.
"""

from .gguf import iter_gguf_weights, parse_gguf_config
from .gguf_experts import (
    gguf_expert_specs,
    gguf_expert_types,
    load_gguf_expert_sources,
)
from .model import Qwen4ExpForCausalLM

__all__ = [
    "Qwen4ExpForCausalLM",
    "parse_gguf_config",
    "iter_gguf_weights",
    # The offload expert-bank loader resolves these off the package named by the model
    # spec, not off the submodule, so they have to be re-exported here.
    "gguf_expert_types",
    "gguf_expert_specs",
    "load_gguf_expert_sources",
]
