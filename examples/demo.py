"""A minimal HTTP/SSE agent loop driven by three policy components.

Configure trigger, selector, and reposition in main. The default composition
uses drop_context to select old tool results and repositions every third tool batch.
Pass --cache-tool semantic_drop for semantic awareness: the model labels each
dropped tool result as failed, irrelevant or resolved with a reason, and PathMemory
answers repeats of judged paths from those notes.
"""

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from akv.agent import CacheOperationHistory, CachePolicy
from akv.agent.backends import send_chat_request
from akv.agent.policies import (
    CacheTool,
    DropContextTool,
    ModelSelection,
    ModelTrigger,
    PathMemory,
    SemanticDropTool,
    SemanticSelection,
    TurnReposition,
    TurnTrigger,
)
from akv.agent.types import BackendTraceContext, JsonDict, ToolCall

ToolHandler = Callable[[str, Sequence[JsonDict]], JsonDict]
ResultFormatter = Callable[[int, str], str]

RECORDS = {
    "parcel-104": {"note": "Parcel handed to the regional carrier.", "next_record": "route-B7"},
    "route-B7": {"note": "Carrier transferred the parcel to the city depot.", "next_record": "depot-C2"},
    "depot-C2": {"note": "The parcel is out for final delivery.", "next_record": "receipt-F9"},
    "receipt-F9": {"delivery_location": "North Library, locker 17", "next_record": None},
}
TOOLS = (
    {
        "type": "function",
        "function": {
            "name": "read_record",
            "description": "Read one delivery record. Follow next_record until it is null.",
            "parameters": {
                "type": "object",
                "properties": {"record_id": {"type": "string"}},
                "required": ["record_id"],
                "additionalProperties": False,
            },
        },
    },
)


def read_record(arguments: str, messages: Sequence[JsonDict]) -> JsonDict:
    """Business tool; this handler does not need the supplied call-time history."""
    args = json.loads(arguments)
    if not isinstance(args, dict) or set(args) != {"record_id"}:
        raise ValueError("Expected exactly one argument: record_id")
    record_id = args["record_id"]
    if not isinstance(record_id, str) or record_id not in RECORDS:
        raise ValueError("Unknown record_id")
    return RECORDS[record_id]


def register_tool(
    schema: JsonDict,
    handler: ToolHandler,
    *,
    tools: list[JsonDict],
    handlers: dict[str, ToolHandler],
) -> None:
    """Example harness registration: pair a model-facing schema with executable code."""
    name = schema["function"]["name"]
    if name in handlers:
        raise ValueError(f"Tool name is already registered: {name}")
    tools.append(schema)
    handlers[name] = handler


def label_result(message_id: int, content: str) -> str:
    """Expose a stable ID when the result enters canonical history."""
    return json.dumps({"message_id": message_id, "result": json.loads(content)})


def execute_tool(
    call: ToolCall,
    *,
    handlers: Mapping[str, ToolHandler] | None = None,
    messages: Sequence[JsonDict] = (),
) -> str:
    """Tool dispatch belongs to this demo, not to AKV or the model server."""
    try:
        handlers = {"read_record": read_record} if handlers is None else handlers
        if call.name not in handlers:
            raise ValueError(f"Unknown tool: {call.name}")
        return json.dumps(handlers[call.name](call.arguments, messages))
    except (ValueError, TypeError) as error:
        return json.dumps({"error": str(error)})


def run_agent(
    base_url: str,
    policy: CachePolicy,
    output: Path,
    *,
    model: str = "gpt-oss-120b",
    max_steps: int = 12,
    max_tokens: int = 2048,
    cache_tool: CacheTool | None = None,
    path_memory: PathMemory | None = None,
) -> str:
    if max_steps <= 0 or max_tokens <= 0:
        raise ValueError("max_steps and max_tokens must be positive")
    cache_operation_history = CacheOperationHistory()
    tools: list[JsonDict] = []
    handlers: dict[str, ToolHandler] = {}
    result_formatters: dict[str, ResultFormatter] = {}
    # Configure tools once. Descriptions go to the model; handlers stay in the harness.
    register_tool(TOOLS[0], read_record, tools=tools, handlers=handlers)
    if cache_tool:
        register_tool(
            cache_tool.schema,
            lambda arguments, messages: cache_tool.execute(
                arguments, messages=messages, cache_operation_history=cache_operation_history
            ),
            tools=tools,
            handlers=handlers,
        )
        result_formatters["read_record"] = label_result
    output.mkdir(parents=True, exist_ok=False)
    payloads = output / "payloads"
    payloads.mkdir()
    history = [
        {"role": "system", "content": "Use read_record to trace the parcel. Read one record at a time."},
        {
            "role": "user",
            "content": "Start at parcel-104. Follow next_record until null, then report the final delivery location.",
        },
    ]
    if isinstance(cache_tool, SemanticDropTool):
        history[0]["content"] += (
            " Each record result includes its canonical message_id. Keep the newest record result "
            "visible until you have read its exact next_record and received the successor result. "
            f"Then call {cache_tool.name} for older record results that are no longer needed, giving "
            "each a verdict and a reason. For 'resolved', the reason must keep the facts you still "
            "need. Never guess a record ID or delete the newest result. "
            "When next_record is null, report the delivery location immediately without another "
            "cleanup call. "
            "A scheduled result takes effect on the next model request."
        )
    elif cache_tool:
        history[0]["content"] += (
            " Each record result includes its canonical message_id. Keep the newest record result "
            "visible until you have read its exact next_record and received the successor result. "
            f"Then call {cache_tool.name} for older record results that are no longer needed, using "
            "their message_id values. Never guess a record ID or delete the newest result. "
            "When next_record is null, report the delivery location immediately without another "
            "cleanup call. "
            "A scheduled result takes effect on the next model request."
        )
    trace = {"status": "incomplete", "messages": history, "steps": []}
    completed_batches = 0
    try:
        for step in range(1, max_steps + 1):
            # 1. The harness converts its history/tools to the final chat payload.
            payload = {
                "model": model,
                "messages": history,
                "tools": tools,
                "tool_choice": "auto",
                "stream": True,
                "stream_options": {"include_usage": True},
                "max_completion_tokens": max_tokens,
                "temperature": 1.0,
                "reasoning_effort": "medium",
                "separate_reasoning": True,
            }
            # 2. Apply this request's rules using the cache operation history.
            prepared, next_cache_operation_history = policy.process(
                payload, cache_operation_history=cache_operation_history, api_style="sglang"
            )
            for label, body in (("before", payload), ("after", prepared)):
                (payloads / f"{step:02d}-{label}.json").write_text(
                    json.dumps(body, ensure_ascii=False, indent=2) + "\n"
                )
            cache_operation_history_snapshot = next_cache_operation_history.to_dict()
            new_operations = cache_operation_history_snapshot["operations"][
                len(cache_operation_history.operations) :
            ]
            event = {
                "step": step,
                "completed_tool_batches": completed_batches,
                "canonical_message_ids": list(range(len(history))),
                "drop_message": prepared.get("drop_message"),
                "reposition": prepared.get("reposition"),
                "new_operations": new_operations,
                "cache_operation_history": cache_operation_history_snapshot,
            }
            trace["steps"].append(event)
            print(
                f"\nStep {step} (completed batches={completed_batches}): "
                f"drop={prepared.get('drop_message')}, reposition={prepared.get('reposition')}",
                flush=True,
            )
            print(f"  New cache operations: {json.dumps(new_operations)}", flush=True)
            # 3. Send unchanged; no message conversion is allowed after AKV.
            result = send_chat_request(
                base_url,
                prepared,
                trace_context=BackendTraceContext(output / "calls" / f"{step:02d}", {"step": step}),
            )
            cache_operation_history = next_cache_operation_history
            event.update(
                finish_reason=result.finish_reason, usage=result.usage, backend_metadata=result.metadata
            )
            history.append(result.message.as_openai(include_reasoning_content=True))
            if result.finish_reason == "length":
                raise RuntimeError(
                    "Demo stopped at the generation limit; use a new output and more max_tokens"
                )
            calls = result.message.tool_calls
            if not calls:
                if result.finish_reason != "stop" or not (result.message.content or "").strip():
                    raise RuntimeError("The model returned neither a complete answer nor tool calls")
                trace["status"] = "completed"
                print(f"Answer: {result.message.content}", flush=True)
                return result.message.content
            # 4. Execute every call and append every result before the next generation.
            call_history = history[:-1]
            for call in calls:
                # 4a. A repeat of a judged path is answered from its recorded note.
                guard = path_memory.guard(call, call_history) if path_memory else None
                if guard is not None:
                    content = json.dumps(guard)
                else:
                    content = execute_tool(call, handlers=handlers, messages=call_history)
                    formatter = result_formatters.get(call.name)
                    if formatter is not None:
                        content = formatter(len(history), content)
                history.append(
                    {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": content}
                )
                print(f"  message {len(history) - 1}: {call.name}({call.arguments}) -> {content}", flush=True)
            completed_batches += 1
        raise RuntimeError("Demo reached max_steps without a final answer")
    except Exception as error:
        trace["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        # Includes original history even when old tool results are hidden from the model.
        (output / "trajectory.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", default="gpt-oss-120b")
    parser.add_argument("--output", type=Path, default=Path("local/demo-01"))
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--cache-tool", choices=("drop_context", "semantic_drop"), default="drop_context")
    args = parser.parse_args()
    path_memory = None
    if args.cache_tool == "semantic_drop":
        cache_tool = SemanticDropTool()
        path_memory = PathMemory(cache_tool)
        policy = CachePolicy(
            trigger=TurnTrigger(),
            selector=SemanticSelection(cache_tool),
            reposition=TurnReposition(after=3, every=3),
        )
    else:
        cache_tool = DropContextTool()
        policy = CachePolicy(
            trigger=ModelTrigger(cache_tool),
            selector=ModelSelection(cache_tool),
            reposition=TurnReposition(after=3, every=3),
        )
    run_agent(
        args.base_url,
        policy,
        args.output,
        model=args.model,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        cache_tool=cache_tool,
        path_memory=path_memory,
    )


if __name__ == "__main__":
    main()
