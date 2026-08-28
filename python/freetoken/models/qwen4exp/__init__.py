"""qwen4exp -- the architecture behind Qwen3.8-Flash-Next.

A qwen35moe core -- gated delta net on the linear layers, gated attention on the full
ones, routed MoE with a gated shared expert -- wrapped in a hyper-connection residual path,
with n-gram hashed per-layer embeddings on one layer.

Served from GGUF only; the hyper-connection, indexer and PLE geometry is read from the
checkpoint rather than from a config file.
"""

from .gguf import iter_gguf_weights, parse_gguf_config
from .model import Qwen4ExpForCausalLM

__all__ = ["Qwen4ExpForCausalLM", "parse_gguf_config", "iter_gguf_weights"]
