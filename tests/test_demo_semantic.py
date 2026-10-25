"""Run the demo agent loop offline with a scripted model in place of AKV System."""

import importlib.util
import json
from pathlib import Path

from akv.agent import CachePolicy, GeneratedMessage, GenerationResult, ToolCall
from akv.agent.policies import PathMemory, SemanticDropTool, SemanticSelection, TurnReposition, TurnTrigger

spec = importlib.util.spec_from_file_location("demo", Path(__file__).parents[1] / "examples" / "demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def read(call_id, record_id):
    return (ToolCall(call_id, "read_record", json.dumps({"record_id": record_id})),)


SCRIPT = [
    read("c1", "parcel-104"),
    read("c2", "route-B7"),
    (
        ToolCall(
            "c3",
            "semantic_drop",
            json.dumps(
                {"paths": [{"message_id": 3, "verdict": "resolved", "reason": "parcel-104 leads to route-B7"}]}
            ),
        ),
    ),
    read("c4", "parcel-104"),  # A repeat of the resolved path.
    read("c5", "depot-C2"),
    read("c6", "receipt-F9"),
    "North Library, locker 17",
]


def test_demo_records_notes_guards_repeats_and_drops(tmp_path, monkeypatch):
    sent = []

    def scripted(base_url, payload, *, trace_context):
        sent.append(payload)
        step = SCRIPT[len(sent) - 1]
        if isinstance(step, str):
            return GenerationResult(GeneratedMessage(content=step), "stop", 0.0)
        return GenerationResult(GeneratedMessage(tool_calls=step), "tool_calls", 0.0)

    monkeypatch.setattr(demo, "send_chat_request", scripted)
    tool = SemanticDropTool()
    policy = CachePolicy(
        trigger=TurnTrigger(), selector=SemanticSelection(tool), reposition=TurnReposition(after=3, every=3)
    )
    answer = demo.run_agent(
        "http://unused", policy, tmp_path / "run", cache_tool=tool, path_memory=PathMemory(tool)
    )
    assert answer == "North Library, locker 17"
    messages = sent[-1]["messages"]
    receipt = json.loads(messages[7]["content"])
    assert receipt["paths"][0]["verdict"] == "resolved"
    guard = json.loads(messages[9]["content"])
    assert guard["status"] == "known_path" and guard["reason"] == "parcel-104 leads to route-B7"
    assert sent[-1]["drop_message"] == {"7": [3]}
    assert 9 not in sum(sent[-1]["drop_message"].values(), [])
