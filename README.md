# AKVSgl

AKVSGLang brings cache engineering to SGLang, letting agents drop selected KV states and reposition retained ones through a stateless API while reusing corresponding cache to reduce long-horizon reasoning costs.

## Environment Setup

AKVSgl 0.1.0 applies source patches to a fixed SGLang v0.5.20 revision. Use Linux with an NVIDIA GPU and the upstream CUDA 13 software stack. In addition to SGLang's requirements, Context kernels need a C++20 compiler (GCC 13 is validated).

Install Rust and Cargo if unavailable:

```bash
curl https://sh.rustup.rs -sSf | sh -s -- -y
source "$HOME/.cargo/env"
```

With uv installed, clone this repository and install all packages in one command:

```bash
git clone https://github.com/thu-nics/AKV.git
cd AKV
uv venv akv --python 3.12
source akv/bin/activate
uv pip install -e .
```

This installs AKVSgl, the patched SGLang and gateway, and Mooncake (`mooncake-transfer-engine-cuda13==0.3.13`). Mooncake is included for PD disaggregation; installing it does not enable PD. Git and the native build toolchain (including Cargo, make and Perl for the gateway) must be available. The build environment supplies protoc and builds vendored OpenSSL automatically.

Use uv for this installation: plain pip does not read the local dependency sources in `pyproject.toml`. No manual patch command or activation import is required. Generated upstream sources stay in `AKVSgl/.sglang/`; keep this directory while using the editable installation. Re-run the same install command after changing patches or native extensions. Clean generated trees are rebuilt when inputs change; the previous tree is preserved. Edits made directly inside the generated upstream tree are reported rather than overwritten.

## Quick Start

AKVSGLang extends SGLang's OpenAI-compatible chat completion API.

1. Launch the server:

```bash
python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B
```

For the Drop/Repos example below, launch with page size 1 and a supported attention backend:

```bash
python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --page-size 1 --attention-backend flashinfer
```

Both commands use SGLang's native entry point. Ordinary requests need no AKVSgl-specific launch flag.

| Server argument | Usage |
| --- | --- |
| `--page-size 1` | Required for Drop/Repos. |
| `--attention-backend` | Use `flashinfer`, `triton`, or `fa3`, subject to the model and GPU's native support. Both prefill and decode backends must be supported. |
| `--disable-drop-aware-eviction` | New optional flag. Drop-aware eviction is enabled by default; pass this flag to disable it. |

2. Send a request with cache operations:

```python
import requests

response = requests.post(
    "http://localhost:30000/v1/chat/completions",
    json={
        "model": "Qwen/Qwen3-0.6B",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Answer the latest user question accurately."},
            {"role": "user", "content": "The capital of France is Paris."},
            {"role": "assistant", "content": "I have read the note."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        "drop_message": {"3": [1, 2]},
        "reposition": [3],
        "temperature": 0,
        "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False},
    },
    timeout=120,
)
response.raise_for_status()
print(response.json())
```

Message indices are **zero-based**. Here, after message `3`, the server drops the KV of messages `1` and `2` and compacts the retained positions before generating the response. Drop alone preserves the surviving positions; `reposition` explicitly removes the gaps.

| Request field | Usage |
| --- | --- |
| `drop_message` | Map a trigger message index to the message indices whose KV should be dropped **after** that trigger. |
| `reposition` | List message indices after which to compact retained KV positions. Optional; independent of `drop_message`. |
| `drop_rule` | Alternative structured rule: `message_drop` or `thinking_drop`. Do not combine it with `drop_message`. (We will further release more drop rules)|

For example, `"drop_rule": {"type": "message_drop", "drop_messages": {"3": [1, 2]}}` can replace `drop_message` above.

Context requests report `cached_tokens`, `drop_skipped_tokens`, and `repos_tokens` in `response.usage.prompt_tokens_details`, including zero values.

For PD deployment, follow [SGLang's PD guide](https://docs.sglang.io/docs/advanced_features/pd_disaggregation) and use the same system revision on both workers, with `--page-size 1` and `--disaggregation-transfer-backend mooncake`. Wait for the gateway's `/readiness` endpoint before sending requests. Context PD currently supports only Mooncake and excludes staging buffers, KV offload, and KV checksum mode.

## Supported Models

Drop/Reposition is currently enabled for these text-model architectures:

| Model family | Architecture |
| --- | --- |
| Qwen | `QWenLMHeadModel` |
| Qwen1.5, Qwen2, Qwen2.5 | `Qwen2ForCausalLM`, `Qwen2MoeForCausalLM` |
| Qwen3, including MoE variants, AgenticQwen | `Qwen3ForCausalLM`, `Qwen3MoeForCausalLM` |
| GPT-OSS | `GptOssForCausalLM` |
| MiniMax-M2 family, including M2.7 | `MiniMaxM2ForCausalLM` |

TP is supported. Context requests currently exclude PP/CP/DCP, speculative decoding, LoRA, HiCache, external/session caches, and beam search. Unsupported Context requests are rejected at the API; model startup is not restricted by this Context model list.
