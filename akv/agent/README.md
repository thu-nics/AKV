# AKV Agent

See the [root README](../../README.md) for the Quick Start, harness integration, and policy design.

## Installation

From the repository root, create a separate client environment. The base package uses only the Python standard library and does not include the GPU runtime.

```bash
uv venv --python 3.12 .venv-algo
source .venv-algo/bin/activate
uv pip install -e ./akv/agent
```

Token-based policies additionally require the optional token counting dependencies:

```bash
uv pip install -e "./akv/agent[tokenization]"
```

## Code Structure

The main modules are organized into policy composition, shared helpers, and HTTP transport:

```text
akv/agent/
├── policy.py                 # CachePolicy.process: compose decisions and prepare requests
├── types.py                  # Shared request, action, cache operation history, and token-layout types
├── policies/                 # Composable policy components
│   ├── triggers.py           # When to select messages for deletion
│   ├── selectors.py          # Which messages to delete
│   ├── reposition.py         # When to compact retained positions
│   ├── semantic.py           # Verdict-labelled drops, path ledger, and semantic selection
│   ├── tools.py              # CacheTool interface and tool implementations
│   └── utils/                # Request, message, and policy-context helpers
│       ├── tokenization.py   # ModelTokenCounter entry point
│       └── models/           # Chat-template rendering and model-specific counters
└── backends/                 # Native HTTP/SSE transport and request captures
```

[CachePolicy.process](policy.py) combines components from [policies/](policies/) and uses [shared helpers](policies/utils/) to prepare the request. The optional [backend](backends/) sends requests independently of policy evaluation. The [demo](../../examples/demo.py) connects these parts in an agent loop.

## Token Counting

Token-based policies count the prompt rendered for the model. For a new model, the counter must match AKV System's tokenizer, chat template, tool formatting, and thinking options.

Load the model's local assets and pass the counter to token-based components:

```python
from akv.agent.policies.utils.tokenization import ModelTokenCounter

counter = ModelTokenCounter("/path/to/model")
```

To adapt a new model:

1. **Use the model's own chat template.** For Jinja models, supply `tokenizer.json`, the special-token settings in `tokenizer_config.json`, and `chat_template.jinja` (or the `chat_template` string in `tokenizer_config.json`). These must match the assets used by the server. The counter loads this configured template; it does not support replacing it through per-request `chat_template_kwargs["chat_template"]`.
2. **Adapt rendering where needed.** The [counter loader](policies/utils/models/__init__.py) uses the generic [Jinja counter](policies/utils/models/jinja.py) for model types without a registered counter. If the model needs special rendering or message-boundary handling, add a counter in [models/](policies/utils/models/) and register its `config.json` `model_type` in `MODEL_COUNTERS`. Implement the [ChatTokenCounter](types.py) interface: `message_layout(...)` returns message boundaries, per-message token counts, and the generation-prefix length. See [GPT-OSS](policies/utils/models/gpt_oss.py) and [MiniMax](policies/utils/models/minimax.py) for existing adaptations.
3. **Verify the rendered prompt and message boundaries.** Compare with the server's prompt rendering and tokenization for plain messages, tool calls/results, and reasoning modes supported by the model. Check the total prompt count, each message's token ownership, and the final generation prefix so token-based selection drops the intended content.

## License

AKV is licensed under [Apache-2.0](../../LICENSE).
