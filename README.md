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


> ### This fork: multi-shard GGUF, DeepSeek-V4, the Qwen3 / Qwen3.5 / Qwen3.6 families, and every ggml quant type
>
> Upstream FreeToken reads GGUF for Gemma-4 only, only Q4_0/Q8_0/Q6_K, and only as a single
> file. This fork widens all three: any ggml quant type the vendored kernels already handle,
> the `qwen3moe` / `qwen35moe` / `qwen35` architectures covering the Qwen3, Qwen3.5 and
> Qwen3.6 families, the `deepseek4` architecture (DeepSeek-V4-Flash), and split
> `-00001-of-000NN` checkpoints.
>
> Multi-shard matters more than it sounds. Every large GGUF ships split, so without it the
> quant-type work was unreachable for exactly the models it was meant to serve. Point
> `--model` at any shard or at the folder holding them.
>
> Proposed upstream as [FlashML-org/FreeToken#131](https://github.com/FlashML-org/FreeToken/pull/131)
> (quant types and the Qwen architectures),
> [#154](https://github.com/FlashML-org/FreeToken/pull/154) (multi-shard),
> [#138](https://github.com/FlashML-org/FreeToken/pull/138) (kernel guards) and
> [#210](https://github.com/FlashML-org/FreeToken/pull/210) (DeepSeek-V4).
>
> #### Verified on my hardware
>
> RTX 4060 Laptop, 8GB VRAM, 64GB system RAM, routed experts offloaded to host memory.
>
> | Model | Arch | Quant | Result |
> |---|---|---|---|
> | [Ornith-1.5-35B-A3B](https://huggingface.co/vcruz305/Ornith-1.5-35B-A3B-GGUF) | qwen35moe | IQ3_S | 8/8 factual, 47 to 50 tok/s |
> | [Ornith-1.5-35B-A3B](https://huggingface.co/vcruz305/Ornith-1.5-35B-A3B-GGUF) | qwen35moe | IQ3_XXS | 6/6 factual, 50 to 52 tok/s |
> | Ornith-1.5-35B-A3B split into 3 shards | qwen35moe | IQ3_S | 6/6 factual, 44 to 46 tok/s |
> | [Qwen3-30B-A3B](https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF) | qwen3moe | IQ4_XS | 6/6 factual, 45 to 47 tok/s |
>
> A 35B model with 3B active, in 16GB, decoding at 50 tok/s on a laptop GPU with 8GB of
> VRAM. For reference, llama.cpp on the same file on CPU does 11 tok/s.
>
> #### Supported by architecture
>
> These carry a `general.architecture` this fork now handles. I have not run all of them,
> so this is "the loader covers it", not "I benchmarked it". Bank column is from reading
> each file's tensor table.
>
> | Model | Arch | Expert banks | Notes |
> |---|---|---|---|
> | [Qwen3-235B-A22B](https://huggingface.co/unsloth/Qwen3-235B-A22B-GGUF) | qwen3moe | check per quant | split, 3 to 10 shards |
> | [Qwen3.5-122B-A10B](https://huggingface.co/unsloth/Qwen3.5-122B-A10B-GGUF) | qwen35moe | uniform at IQ3_S | should load as is |
> | [Qwen3.6-35B-A3B](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) | qwen35moe | mixed at IQ3_S | needs a `--pure` quant |
> | [Ornith-1.0-35B-AEON](https://huggingface.co/vcruz305/Ornith-1.0-35B-AEON-Ultimate-Uncensored-GGUF) | qwen35moe | mixed at Q4_K_M | needs a `--pure` quant |
> | [Qwen3.8-27B](https://huggingface.co/vcruz305/Qwen3.8-27B-GGUF) | qwen35 dense | none | Q4_K_M, test in progress |
> | [Qwen3.6-27B](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF) | qwen35 dense | none | Q4_K_M |
> | [Qwen3.5-27B](https://huggingface.co/unsloth/Qwen3.5-27B-GGUF) | qwen35 dense | none | Q4_K_M |
> | [Qwen3.5-9B](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) | qwen35 dense | none | Q4_K_M |
> | [Qwen3.5-4B](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF) | qwen35 dense | none | Q4_K_M |
>
> Dense models have no expert banks, so the uniform-bank rule below does not apply to them
> and ordinary `_M` quants are fine.
>
> More of my quants at [huggingface.co/vcruz305](https://huggingface.co/vcruz305/models).
>
> #### Quant types
>
> All 21 ggml types are now described on the Python side. `csrc/gguf/` already dispatched 19
> of them, but the tables only listed 6, so K-quants and I-quants were unreachable for every
> architecture rather than just this one.
>
> | Family | Types | Prefill | Decode |
> |---|---|---|---|
> | Standard | Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 | MMQ | MMVQ |
> | K-quants | Q2_K, Q3_K, Q4_K, Q5_K, Q6_K | MMQ | MMVQ |
> | I-quants | IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS | dequant plus matmul | MMVQ |
>
> I-quants have no MMQ kernel upstream, so prefill falls back to `ggml_dequantize` and a
> torch matmul. That branch already existed but was unreachable, because `_MMQ` was tested
> before `_DEQUANT` and the two sets were identical.
>
> #### What else changed
>
> - Fixed a silent data corruption bug. None of the five `switch (type)` blocks in
>   `gguf_kernel.cu` had a `default:`, and the output tensor is allocated with
>   `torch::empty`, so an unsupported quant type returned uninitialized memory instead of
>   raising. `ggml_moe_get_block_size` returned 0. That one is worth having on its own,
>   independent of the rest of this fork.
> - Documented the four transforms llama.cpp's converter applies that a loader has to undo:
>   `ssm_a` holds `A` rather than `A_log`, the `(1+w)` norm shift is already folded in, V
>   heads are stored tiled rather than grouped when there are fewer K heads than V heads,
>   and merged projections are not uniformly typed. None of these are visible to shape,
>   dtype or byte identity checks. The model loads, runs at full speed, and produces fluent
>   nonsense.
> - Tests derive the block sizes from `ggml-common.h` and extract the `case` labels from
>   `gguf_kernel.cu` at runtime, so the Python tables cannot drift from the kernels without
>   a test failing.
>
> #### Known limits
>
> - MoE expert banks have to use one ggml type across every layer. The GPU slot pool is a
>   single allocation and `moe_vec.cuh` indexes it as `expert * nrows * (ncols / qk)` with no
>   padding allowance, so two row strides in one pool would read every block at the wrong
>   offset. llama.cpp's `_M` and `_XXS` levels raise the precision of the first few layers'
>   `ffn_down_exps`, which trips this. Those refuse to load with an error naming the layers.
>   Quantize with `llama-quantize --pure` to get one type throughout. Dense models are
>   unaffected.
> #### DeepSeek-V4-Flash
>
> A 284B MoE with MLA, DSA sparse attention, hyper-connections and hash routing. Verified on
> a Quadro RTX 6000 (Turing, sm_75, 24GB) with 204 GiB of RAM available to WSL, serving the
> 164GB [`antirez/deepseek-v4-gguf`](https://huggingface.co/antirez/deepseek-v4-gguf)
> Q4KExperts build: 43 layers, 256 experts, 145 GiB of expert banks split 24 layers
> GPU-pinned and 19 OS-locked for CPU decode, about 18GB VRAM. Correct at temperature 0
> (`"The capital city of France is"` -> `" Paris."`).
>
> Two caveats worth reading before you try it.
>
> **Most published checkpoints will not load.** Of the thirteen
> `unsloth/DeepSeek-V4-Flash-0731-GGUF` variants, eleven mix ggml types across layers and the
> two uniform ones are MXFP4, which has no `BLOCK_SHAPE` entry and no vendored kernel. The
> `antirez` builds are uniform and do load. Check before downloading 90GB.
>
> **CUDA graph capture crashes on this configuration**, so it currently needs
> `--cuda-graph-max-bs 0` and decode is slower than it should be. Being worked on.
>
> Host RAM is the binding constraint, not VRAM: the full expert set is held pinned, and WSL
> caps CUDA pinning near 40% of RAM (measured 81.78 GiB of 204), which is why the residency
> split moves layers to the CPU executor.
>
> - TP=1 only, same as the existing Gemma-4 GGUF path.
> - The NextN/MTP block is dropped, so no speculative decoding.
> - Expert bank loading is serial, so first load of a 16GB checkpoint takes about a minute.
>   Converting to FTW with `ft checkpoint` avoids paying it every time.
> - Tested with the flashinfer attention backend. The triton fallback is unexercised here.
> - First token argmax matches llama.cpp CPU on 2 of 6 single token prompts. Every
>   disagreement is a plausible near tie, and this runs CUDA W4A8 MMVQ against CPU AVX, so I
>   do not think exact agreement is reachable. Stating the number rather than implying it is
>   bit exact.


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
