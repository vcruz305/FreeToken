<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/freetoken-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/freetoken-logo-light.svg">
    <img alt="FreeToken" src="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/freetoken-logo.svg" width=65%>
  </picture>
</div>

<p align="center">
| <a href="https://www.flashml.ai/"><b>Download</b></a> | <a href="https://arxiv.org/abs/2608.16157"><b>Paper</b></a> | <a href="https://join.slack.com/t/flashml/shared_invite/zt-3zpdh5j10-9dwTXrgLiqpVxizhA9KVbA"><b>Developer Slack</b></a> | <a href="https://discord.gg/xzwSnMdsX"><b>Community Discord</b></a> | <a href="https://github.com/FlashML-org/FreeToken/blob/main/assets/freetoken-wechatgroup.png"><b>Community WeChat</b></a> |
</p>


> ### This fork: GGUF support for more quant types and Qwen3.5 models
>
> Upstream FreeToken loads GGUF for Gemma-4 only, and only Q4_0, Q8_0 and Q6_K. This fork
> widens that. Proposed upstream as [FlashML-org/FreeToken#131](https://github.com/FlashML-org/FreeToken/pull/131).
>
> **What works now**
>
> | Model | Arch | Quant tested | Result |
> |---|---|---|---|
> | [Ornith-1.5-35B-A3B](https://huggingface.co/vcruz305/Ornith-1.5-35B-A3B-GGUF) | qwen35moe | IQ3_S | 8/8 factual, 47-50 tok/s |
> | [Ornith-1.5-35B-A3B](https://huggingface.co/vcruz305/Ornith-1.5-35B-A3B-GGUF) | qwen35moe | IQ3_XXS | 5/6 factual, 50-52 tok/s |
> | [Ornith-1.0-35B-AEON](https://huggingface.co/vcruz305/Ornith-1.0-35B-AEON-Ultimate-Uncensored-GGUF) | qwen35moe | needs a uniform-bank quant | covered by the same adapter |
> | [Qwen3.8-27B](https://huggingface.co/vcruz305/Qwen3.8-27B-GGUF) | qwen35 (dense) | Q4_K_M | support added, test pending |
>
> Measured on an RTX 4060 Laptop, 8GB VRAM, with the routed experts offloaded to host RAM.
> More quants at [huggingface.co/vcruz305](https://huggingface.co/vcruz305/models).
>
> **What changed**
>
> - All 21 ggml types are now described on the Python side. `csrc/gguf/` already dispatched
>   19 of them but the tables only listed 6, so K-quants and I-quants were unreachable for
>   every architecture, not just this one.
> - Fixed a silent data-corruption bug: none of the five `switch (type)` blocks in
>   `gguf_kernel.cu` had a `default:`, and the output tensor is allocated with
>   `torch::empty`. An unsupported quant type returned uninitialized memory instead of
>   raising. That one is worth having on its own.
> - Added the `qwen35moe` and `qwen35` architectures, including the four transforms
>   llama.cpp's converter applies that a loader has to undo (`ssm_a` holds `A` rather than
>   `A_log`, the `(1+w)` norm shift is already folded in, V heads are stored tiled rather
>   than grouped, and merged projections are not uniformly typed).
> - I-quants have no MMQ kernel, so prefill falls back to `ggml_dequantize` plus a torch
>   matmul. That branch existed already but was unreachable.
>
> **Known limits**
>
> - MoE expert banks must use one ggml type across all layers. llama.cpp's `_M` and `_XXS`
>   levels raise the precision of the first few layers' `ffn_down_exps`, so those refuse to
>   load with an error naming the offending layers. Quantize with `llama-quantize --pure`
>   to avoid it. Dense models have no expert banks and are unaffected.
> - TP=1 only, the NextN/MTP block is dropped, and bank loading is serial.
> - Tested with the flashinfer attention backend. The triton fallback is unexercised here.


Unlock datacenter-class intelligence on the hardware you already own — Run 290B+ frontier MoE models locally on your gaming PC at blistering interactive speeds.

## About

FreeToken is an edge-native Mixture-of-Experts (MoE) serving engine designed for running frontier-scale open-weight models on personal and consumer hardware. It treats heterogeneous edge resources—GPUs, CPUs, host memory, and interconnects—as a unified, elastic inference platform. Its core features include:  

- **Fast Edge-Native Runtime**: Provides efficient MoE serving with bandwidth-adaptive CPU–GPU co-execution ($q^\star$ policy), full-layer double-buffered prefill streaming, global LRU expert caching, graph-compatible execution, and the FTW fast weight format.  
- **Semantic-Aware Caching**: Features semantic anchor checkpoints for recurrent state and KV caches, allowing agentic context edits (e.g., tool calls, thinking blocks) to avoid redundant context recomputation.  
- **Elastic Memory Management**: Supports dynamic, runtime VRAM re-allocation between expert caches and KV memory without engine restarts or weight reloading.  
- **Broad MoE & Ecosystem Support**: Supports frontier open-weight MoE models (e.g., DeepSeek-V4-Flash, Qwen3.6-35B-A3B, GLM-5.2) across various parameter scales and quantization formats (e.g., MXFP4, NVFP4, FP8, BF16), with Anthropic/OpenAI-compatible APIs for seamless integration with real-world coding and tool-calling agents (e.g., Codex, Claude Code, OpenCode, OpenClaw, DeepSeek Harness). 
- **Diverse Consumer Hardware**: Scales across consumer laptops, gaming desktops, and workstation GPUs, with native support for NVIDIA RTX 30, RTX 40, and RTX 50 series GPUs.  

## Getting Started

### Desktop app

Download FreeToken for Windows or Linux at [flashml.ai](https://www.flashml.ai/). It sets the engine up for you and gives you a GUI for running models, chatting, and tuning the engine.

<div align="center">
  <img alt="FreeToken Desktop" src="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/desktop-console.png" width=92%>
</div>

### CLI

Install FreeToken with [uv](https://docs.astral.sh/uv/) (recommended) or pip:

```bash
uv pip install "freetoken[accel]"
```

Or build from source:

```bash
git clone https://github.com/FlashML-org/FreeToken.git && cd FreeToken
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"
```

For More details:

- [Install FreeToken](https://github.com/FlashML-org/FreeToken/blob/main/docs/install.md)
- [Quick start](https://github.com/FlashML-org/FreeToken/blob/main/docs/quickstart.md)
- [Supported models](https://github.com/FlashML-org/FreeToken/blob/main/docs/models.md)
- [CLI reference](https://github.com/FlashML-org/FreeToken/blob/main/docs/cli.md)

## Citation

If you use FreeToken for your research, please cite our [paper](https://arxiv.org/abs/2608.16157):

```bibtex
@article{yang2026freetoken,
  title={FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution},
  author={Yang, Shuo and Fan, Xiaoze and Pan, Melissa and Xi, Haocheng and Wang, Zhe and Sun, Shanlin and Keutzer, Kurt and Han, Song and Zaharia, Matei and Xu, Chenfeng and Stoica, Ion},
  journal={arXiv preprint arXiv:2608.16157},
  year={2026}
}
```

## Acknowledgment

FreeToken was deeply inspired by [mini-sglang](https://github.com/sgl-project/mini-sglang), and
learned the design and reused code from the following projects:
[SGLang](https://github.com/sgl-project/sglang),
[vLLM](https://github.com/vllm-project/vllm),
[FlashInfer](https://github.com/flashinfer-ai/flashinfer),
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention),
[LightLLM](https://github.com/ModelTC/lightllm) and [llama.cpp](https://github.com/ggml-org/llama.cpp).

## License

[Apache License 2.0](https://github.com/FlashML-org/FreeToken/blob/main/LICENSE).
