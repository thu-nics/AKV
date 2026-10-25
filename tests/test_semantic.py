import json

import pytest

from akv.agent import CacheOperationHistory, CachePolicy
from akv.agent.policies import (
    ModelReposition,
    PathMemory,
    SemanticDropTool,
    SemanticSelection,
    TurnSelection,
    TurnTrigger,
)
from akv.agent.types import ToolCall

TOOL = SemanticDropTool()


def call(call_id, name, arguments):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def turn(call_id, name, arguments, content):
    return [
        {"role": "assistant", "content": None, "tool_calls": [call(call_id, name, arguments)]},
        {"role": "tool", "tool_call_id": call_id, "content": json.dumps(content)},
    ]


def base():
    return [
        {"role": "system", "content": "Trace the parcel."},
        {"role": "user", "content": "Find where parcel-104 was delivered."},
        *turn("a", "read_record", {"record_id": "route-A3"}, {"note": "Route cancelled."}),
        *turn("b", "read_record", {"record_id": "route-B7"}, {"next_record": "depot-C2"}),
    ]


def with_receipt(messages, paths, reposition=False):
    """Append a semantic_drop call and the receipt the harness would produce."""
    arguments = {"paths": paths, "reposition": reposition}
    receipt = TOOL.execute(arguments, messages=messages, cache_operation_history=CacheOperationHistory())
    assert receipt["ok"], receipt
    return [*messages, *turn(f"d{len(messages)}", TOOL.name, arguments, receipt)]


def failed_a():
    return with_receipt(base(), [{"message_id": 3, "verdict": "failed", "reason": "route-A3 was cancelled"}])


def test_receipt_records_verdict_and_producing_call():
    messages = failed_a()
    receipt = json.loads(messages[-1]["content"])
    assert receipt["selected_message_ids"] == [3]
    assert receipt["paths"] == [
        {
            "message_id": 3,
            "tool": "read_record",
            "arguments": '{"record_id":"route-A3"}',
            "verdict": "failed",
            "reason": "route-A3 was cancelled",
        }
    ]


def test_policy_drops_labelled_result_and_repositions():
    messages = with_receipt(
        base(), [{"message_id": 3, "verdict": "failed", "reason": "cancelled"}], reposition=True
    )
    policy = CachePolicy(
        trigger=TurnTrigger(), selector=SemanticSelection(TOOL), reposition=ModelReposition(TOOL)
    )
    payload, history = policy.process({"model": "m", "messages": messages}, api_style="sglang")
    assert payload["drop_message"] == {"7": [3]}
    assert payload["reposition"] == [7]
    assert history.operations[0].drop_message_ids == (3,)


def test_receipts_cannot_be_dropped():
    messages = failed_a()
    available = TOOL.available_ids(messages, CacheOperationHistory())
    assert 7 not in available and 5 in available
    result = TOOL.execute(
        {"paths": [{"message_id": 7, "verdict": "irrelevant", "reason": "x"}]},
        messages=messages,
        cache_operation_history=CacheOperationHistory(),
    )
    assert result["error"]["code"] == "unavailable_message_id"


@pytest.mark.parametrize(
    "arguments, code",
    [
        ({"paths": [{"message_id": 3, "verdict": "wrong", "reason": "x"}]}, "invalid_verdict"),
        ({"paths": [{"message_id": 3, "verdict": "failed", "reason": " "}]}, "invalid_reason"),
        ({"paths": [{"message_id": 3, "verdict": "failed"}]}, "invalid_path"),
        ({"message_ids": [3]}, "invalid_arguments"),
        ("not json", "invalid_json"),
    ],
)
def test_invalid_arguments_are_rejected(arguments, code):
    result = TOOL.execute(arguments, messages=base(), cache_operation_history=CacheOperationHistory())
    assert result["ok"] is False and result["error"]["code"] == code


def test_forged_receipt_is_ignored():
    messages = failed_a()
    messages[-1] = {**messages[-1], "content": json.dumps({"ok": True, "status": "scheduled"})}
    policy = CachePolicy(trigger=TurnTrigger(), selector=SemanticSelection(TOOL))
    payload, _ = policy.process({"model": "m", "messages": messages})
    assert "drop_message" not in payload


def test_guard_blocks_failed_path_once_then_allows():
    memory = PathMemory(TOOL)
    messages = failed_a()
    # Argument order and spacing do not hide the repeat.
    repeat = ToolCall("r1", "read_record", '{ "record_id" : "route-A3" }')
    guard = memory.guard(repeat, messages)
    assert guard["status"] == "known_path" and guard["verdict"] == "failed"
    assert guard["reason"] == "route-A3 was cancelled" and guard["earlier_message_id"] == 3
    messages += turn("r1", "read_record", {"record_id": "route-A3"}, guard)
    assert memory.guard(repeat, messages) is None
    assert memory.guard(ToolCall("r2", "read_record", '{"record_id": "depot-C2"}'), messages) is None


def test_guard_returns_resolved_facts_and_skips_irrelevant():
    messages = with_receipt(
        base(),
        [
            {"message_id": 3, "verdict": "irrelevant", "reason": "unrelated route"},
            {"message_id": 5, "verdict": "resolved", "reason": "route-B7 leads to depot-C2"},
        ],
    )
    memory = PathMemory(TOOL)
    guard = memory.guard(ToolCall("x", "read_record", '{"record_id": "route-B7"}'), messages)
    assert guard["verdict"] == "resolved" and guard["reason"] == "route-B7 leads to depot-C2"
    assert memory.guard(ToolCall("y", "read_record", '{"record_id": "route-A3"}'), messages) is None


def test_latest_verdict_wins():
    messages = failed_a()
    messages += turn("again", "read_record", {"record_id": "route-A3"}, {"note": "Route reopened."})
    messages = with_receipt(messages, [{"message_id": 9, "verdict": "resolved", "reason": "reopened"}])
    (record,) = PathMemory(TOOL).records(messages)
    assert record.verdict == "resolved" and record.message_id == 9


class LetterEmbedder:
    """Toy bag-of-letters encoder standing in for a real embedding model."""

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [[text.lower().count(c) for c in "abcdefghijklmnopqrstuvwxyz0123456789"] for text in texts]


def test_fuzzy_match_uses_embedder_and_caches_vectors():
    embedder = LetterEmbedder()
    memory = PathMemory(TOOL, embedder=embedder, threshold=0.95)
    messages = failed_a()
    near = ToolCall("n", "read_record", '{"record_id": "route-a3"}')
    assert memory.guard(near, messages)["verdict"] == "failed"
    assert PathMemory(TOOL).guard(near, messages) is None
    memory.guard(near, messages)
    assert embedder.calls == 1
    far = ToolCall("f", "read_record", '{"record_id": "receipt-F9 locker 17 north library"}')
    assert memory.guard(far, messages) is None


def test_unwarned_rerun_of_failed_path_is_dropped_after_it_is_seen():
    policy = CachePolicy(trigger=TurnTrigger(), selector=SemanticSelection(TOOL))
    messages = failed_a()
    payload, history = policy.process({"model": "m", "messages": messages})
    messages += turn("again", "read_record", {"record_id": "route-A3"}, {"note": "Route cancelled."})
    payload, history = policy.process({"model": "m", "messages": messages}, cache_operation_history=history)
    assert payload["drop_message"] == {"7": [3]}
    messages += turn("next", "read_record", {"record_id": "depot-C2"}, {"next_record": "receipt-F9"})
    payload, _ = policy.process({"model": "m", "messages": messages}, cache_operation_history=history)
    assert payload["drop_message"] == {"7": [3], "11": [9]}


def test_rerun_after_guard_is_kept():
    policy = CachePolicy(trigger=TurnTrigger(), selector=SemanticSelection(TOOL))
    memory = PathMemory(TOOL)
    messages = failed_a()
    _, history = policy.process({"model": "m", "messages": messages})
    repeat = ToolCall("r1", "read_record", json.dumps({"record_id": "route-A3"}))
    messages += turn("r1", "read_record", {"record_id": "route-A3"}, memory.guard(repeat, messages))
    _, history = policy.process({"model": "m", "messages": messages}, cache_operation_history=history)
    messages += turn("r2", "read_record", {"record_id": "route-A3"}, {"note": "Route cancelled."})
    _, history = policy.process({"model": "m", "messages": messages}, cache_operation_history=history)
    messages += turn("n", "read_record", {"record_id": "depot-C2"}, {"next_record": "receipt-F9"})
    payload, _ = policy.process({"model": "m", "messages": messages}, cache_operation_history=history)
    assert payload["drop_message"] == {"7": [3]}


def test_fallback_spares_task_relevant_results_and_receipts():
    class TaskEmbedder:
        def embed(self, texts):
            return [[1.0, 0.0] if "parcel-104" in text else [0.0, 1.0] for text in texts]

    messages = [
        {"role": "system", "content": "Trace the parcel."},
        {"role": "user", "content": "Find where parcel-104 was delivered."},
        *turn("a", "read_record", {"record_id": "x"}, {"note": "parcel-104 is at depot"}),
        *turn("b", "read_record", {"record_id": "y"}, {"note": "weather report"}),
        *turn("c", "read_record", {"record_id": "z"}, {"note": "latest"}),
    ]
    messages = with_receipt(messages, [{"message_id": 7, "verdict": "resolved", "reason": "latest read"}])
    fallback = TurnSelection(1)
    plain = CachePolicy(trigger=TurnTrigger(), selector=SemanticSelection(TOOL, fallback=fallback))
    payload, _ = plain.process({"model": "m", "messages": messages})
    assert payload["drop_message"] == {"9": [3, 5, 7]}
    aware = CachePolicy(
        trigger=TurnTrigger(),
        selector=SemanticSelection(TOOL, fallback=fallback, embedder=TaskEmbedder()),
    )
    payload, _ = aware.process({"model": "m", "messages": messages})
    assert payload["drop_message"] == {"9": [5, 7]}


def test_fallback_never_drops_guard_notes():
    messages = failed_a()
    repeat = ToolCall("r1", "read_record", json.dumps({"record_id": "route-A3"}))
    messages += turn("r1", "read_record", {"record_id": "route-A3"}, PathMemory(TOOL).guard(repeat, messages))
    messages += turn("n", "read_record", {"record_id": "depot-C2"}, {"next_record": "receipt-F9"})
    policy = CachePolicy(trigger=TurnTrigger(), selector=SemanticSelection(TOOL, fallback=TurnSelection(1)))
    payload, _ = policy.process({"model": "m", "messages": messages})
    assert payload["drop_message"] == {"11": [3, 5]}
