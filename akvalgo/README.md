# akvalgo

akvalgo provides drop/reposition policies and stateless request adaptation for agent workflows. It derives cumulative cache operations from the full conversation history while leaving history management, tool execution, and the agent loop to the caller.

## Environment Setup

The package uses Python 3.12 and has no runtime dependencies beyond the standard library. With uv installed, run these commands from the repository root:

```bash
UV_PROJECT_ENVIRONMENT="$PWD/akvalgo/.venv-algo" uv sync --project akvalgo --no-dev
akvalgo/.venv-algo/bin/python -c 'import akvalgo; print(akvalgo.__file__)'
```

Keep the algorithm environment separate from the server environment. For server installation and configuration, see [akvsgl](../README.md).

## Quick Start

Prepare a Chat Completions JSON request, then apply the policy before sending it:

```python
from akvalgo import AKVPolicy, adapt_request, apply_context

policy = AKVPolicy(
    keep_tool_responses=12,
    reposition_after_call=4,
    reposition_interval_calls=4,
)


def prepare_request(payload):
    policy_input = adapt_request(payload, api_style="sglang")
    decision = policy.decide(policy_input)
    return apply_context(payload, decision)
```

`payload` contains the full message history, tools, and generation options. The default `kv_drop` mode preserves the request's messages and adds cumulative `drop_message` and `reposition` instructions. `apply_context` returns a copy without modifying the caller's history.

Build each request from the full history without existing cache instructions. Message indices are **zero-based**; send the prepared request without inserting, merging, or reordering messages afterward.

Use your own HTTP client or the optional native HTTP/SSE backend:

```python
from akvalgo.backends import send_chat_request

result = send_chat_request("http://127.0.0.1:30000/v1", prepare_request(payload))
```

The backend collects streamed text and tool calls, requires `[DONE]`, and supports request capture through `trace_context`. Interrupted responses retain partial captures, and missing usage remains unknown. The caller appends the response, executes tools, and adds their results before the next request.

## Policy Options

| Option | Usage |
| --- | --- |
| `keep_tool_responses` | Number of recent individual tool responses to retain. |
| `mode` | `kv_drop` adds cache instructions; `text_drop` removes selected messages from the request copy. |
| `rolling_drop_target` | Select `tool`, `assistant`, or `both` for rolling deletion. |
| `reposition_after_call` | First complete tool batch to trigger repositioning. |
| `reposition_interval_calls` | Repeat interval for repositioning. |
| `reposition_trigger_tokens` | Trigger repositioning by token position length instead of batch count. |
| `drop_after_call`, `drop_interval_calls` | Schedule drops by complete tool batches. |
| `drop_trigger_tokens`, `drop_trigger_responses` | Trigger drops by visible prompt length or tool-response count. |
| `drop_target_tokens` | Target visible prompt length after a triggered drop. |
| `token_counter` | Counter used by token-based policies. |

Without an explicit drop trigger, the policy uses a rolling window. Drop trigger modes are mutually exclusive. Repositioning requires `kv_drop`; token thresholds and targets require a matching counter. See [AKVPolicy](policy.py) for defaults and configuration rules.

## Token Counting

Install the optional tokenization dependencies when using token-based policies:

```bash
UV_PROJECT_ENVIRONMENT="$PWD/akvalgo/.venv-algo" uv sync --project akvalgo --no-dev --extra tokenization
```

[LocalChatTemplateTokenCounter](tokenization.py) loads a local model's tokenization resources. Adapters are provided for GPT-OSS, MiniMax, and MiroThinker. The tokenizer, template, and rendering settings must match the actual request. Custom counters implement the `ChatTokenCounter` contract in [types.py](types.py).

## License

AKV is licensed under [Apache-2.0](../LICENSE). Individual files retain their applicable license notices.
