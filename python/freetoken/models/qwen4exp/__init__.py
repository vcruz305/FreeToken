"""qwen4exp -- the architecture behind Qwen3.8-Flash-Next.

Only the GGUF layout layer exists so far: config parsing, the tensor name map, and the
layer schedule. There is no model class yet, so the arch is deliberately NOT registered in
``GGUF_ARCH_TO_REGISTRY`` -- a registry entry pointing at a class that does not exist would
turn "unsupported architecture" into an import error.
"""
