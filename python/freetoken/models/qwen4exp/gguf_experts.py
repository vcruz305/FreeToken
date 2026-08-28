"""Routed-expert banks for qwen4exp -- the qwen35moe loader, unchanged.

The three expert stacks carry the same names and the same axis order as qwen35moe's, so
there is nothing architecture-specific to do here:

    ffn_gate_exps / ffn_up_exps   ggml [H, I, E] -> torch [E, I, H]   (2560, 640, 512)
    ffn_down_exps                 ggml [I, H, E] -> torch [E, H, I]   ( 640, 2560, 512)

with E=512 experts, I=640 expert_feed_forward_length and H=2560 hidden. The gate_up
fusion (each expert's gate rows then its own up rows, concatenated on the row axis) and
the per-layer quant types are handled identically.

The one qwen35moe behaviour that does not apply is skipping a trailing NextN/MTP block.
qwen4exp has no MTP block: block_count is 48 and all 48 are decoder layers, so the
``layer >= num_layers`` guard never fires and re-exporting is safe rather than merely
convenient.

Re-exported rather than copied so a fix to the expert loader cannot land for one
architecture and silently miss the other.
"""

from __future__ import annotations

from freetoken.models.qwen3_5_moe.gguf_experts import (
    gguf_expert_specs,
    gguf_expert_types,
    load_gguf_expert_sources,
)

__all__ = ["gguf_expert_types", "gguf_expert_specs", "load_gguf_expert_sources"]
