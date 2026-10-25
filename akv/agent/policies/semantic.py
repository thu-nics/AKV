"""Semantic awareness: the model labels what it drops, and the labels outlive the KV entries.

AKV System only appends, drops and repositions KV entries; it never reads text. This
module adds the missing judgement on the agent side:

- ``SemanticDropTool`` is a cache tool whose arguments carry a verdict and reason for
  every dropped result. Its receipt stays in history, so the lesson remains visible to
  the model after the result's KV entries are gone.
- ``PathMemory`` rebuilds the ledger of judged paths from history alone, then lets the
  harness answer a repeated tool call from the ledger instead of executing it again.
- ``SemanticSelection`` drops model-labelled paths and rerun duplicates of discarded
  paths, and can protect task-relevant results from a fallback selector.
"""

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..types import CacheAction, CacheOperationHistory, JsonDict, ToolCall
from .tools import ToolArgumentError, tool_actions
from .utils.context import PolicyContext

VERDICTS = ("failed", "irrelevant", "resolved")
GUARD_STATUS = "known_path"
MAX_REASON_CHARS = 500


class Embedder(Protocol):
    """Optional text encoder for fuzzy matching; AKV ships no model of its own."""

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


@dataclass(frozen=True)
class PathRecord:
    """One judged reasoning path: the tool call that produced a dropped result."""

    message_id: int
    tool: str
    arguments: str
    verdict: str
    reason: str

    @property
    def key(self) -> str:
        return path_key(self.tool, self.arguments)

    def as_json(self) -> JsonDict:
        return {
            "message_id": self.message_id,
            "tool": self.tool,
            "arguments": self.arguments,
            "verdict": self.verdict,
            "reason": self.reason,
        }


def canonical_arguments(arguments: str | Mapping[str, Any] | None) -> str:
    """Normalize JSON arguments so key order and spacing do not hide a repeat."""
    try:
        value = json.loads(arguments) if isinstance(arguments, str) else arguments
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (ValueError, TypeError):
        return str(arguments or "").strip()


def path_key(tool: str, arguments: str | Mapping[str, Any] | None) -> str:
    return f"{tool}:{canonical_arguments(arguments)}"


def producing_call(messages: Sequence[JsonDict], message_id: int) -> JsonDict | None:
    """Return the assistant tool call whose result is ``message_id``."""
    result = messages[message_id]
    if result.get("role") != "tool":
        return None
    for i in range(message_id - 1, -1, -1):
        message = messages[i]
        if message.get("role") == "tool":
            continue
        for call in message.get("tool_calls") or ():
            if call.get("id") == result.get("tool_call_id"):
                return call
        return None
    return None


@dataclass(frozen=True)
class SemanticDropTool:
    """Schedule Drop for earlier tool results, recording why each one is no longer needed."""

    name: str = "semantic_drop"
    protected_prefix_messages: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Tool name must be nonempty")
        if type(self.protected_prefix_messages) is not int or self.protected_prefix_messages < 0:
            raise ValueError("protected_prefix_messages must be a nonnegative integer")

    @property
    def schema(self) -> JsonDict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Schedule deletion of visible EARLIER tool results from the KV cache on the next "
                    "model request, and record why each one is no longer needed. Copy the canonical "
                    "message_id supplied with each result. For every result give a verdict: 'failed' "
                    "when the path was a dead end or wrong, 'irrelevant' when it does not bear on the "
                    "task, or 'resolved' when its information has been used; then give a short reason. "
                    "For 'resolved', the reason must state the facts you still need, because the result "
                    "itself will be gone. The receipt keeps these notes visible, and repeating a judged "
                    "tool call returns the note instead of running it again. Any invalid entry rejects "
                    "the entire call; use available_message_ids in the error to correct it. Continue "
                    "the original task after a successful call."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "paths": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 64,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "message_id": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 9223372036854775807,
                                    },
                                    "verdict": {"type": "string", "enum": list(VERDICTS)},
                                    "reason": {"type": "string", "minLength": 1, "maxLength": MAX_REASON_CHARS},
                                },
                                "required": ["message_id", "verdict", "reason"],
                                "additionalProperties": False,
                            },
                            "description": "Visible earlier tool results to delete, each with a verdict.",
                        },
                        "reposition": {
                            "type": "boolean",
                            "description": "Also request position compaction, if enabled by the policy.",
                        },
                    },
                    "required": ["paths"],
                    "additionalProperties": False,
                },
            },
        }

    def available_ids(
        self, messages: Sequence[JsonDict], cache_operation_history: CacheOperationHistory
    ) -> list[int]:
        """Visible earlier tool results, excluding this tool's receipts, which hold the notes."""
        if cache_operation_history.message_count > len(messages):
            raise ValueError("Cache operation history extends beyond the history before this tool call")
        hidden = {i for op in cache_operation_history.operations for i in op.drop_message_ids}
        available = []
        for i, message in enumerate(messages):
            if i < self.protected_prefix_messages or message["role"] != "tool" or i in hidden:
                continue
            call = producing_call(messages, i)
            if call is not None and call["function"]["name"] != self.name:
                available.append(i)
        return available

    def execute(
        self,
        arguments: str | Mapping[str, Any],
        *,
        messages: Sequence[JsonDict],
        cache_operation_history: CacheOperationHistory,
    ) -> JsonDict:
        """Return a receipt; do not contact the server or claim deletion has occurred."""
        available = self.available_ids(messages, cache_operation_history)
        try:
            action, records = self.action(arguments, messages, available)
            return scheduled_result(action, records)
        except ToolArgumentError as error:
            return {
                "ok": False,
                "status": "error",
                "error": error.details,
                "available_message_ids": available,
            }

    @staticmethod
    def action(
        arguments: str | Mapping[str, Any], messages: Sequence[JsonDict], available: Sequence[int]
    ) -> tuple[CacheAction, tuple[PathRecord, ...]]:
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
        except ValueError as error:
            raise ToolArgumentError("invalid_json", "Arguments must be a JSON object") from error
        if not isinstance(args, Mapping) or "paths" not in args or set(args) - {"paths", "reposition"}:
            raise ToolArgumentError("invalid_arguments", "Expected paths and optional reposition only")
        paths = args["paths"]
        if not isinstance(paths, list) or not 1 <= len(paths) <= 64:
            raise ToolArgumentError("invalid_paths", "paths must be an array of 1 to 64 entries")
        reposition = args.get("reposition", False)
        if type(reposition) is not bool:
            raise ToolArgumentError("invalid_reposition", "reposition must be boolean")
        records, seen = [], set()
        for index, entry in enumerate(paths):
            if not isinstance(entry, Mapping) or set(entry) != {"message_id", "verdict", "reason"}:
                raise ToolArgumentError(
                    "invalid_path", "Each path needs exactly message_id, verdict and reason", item_index=index
                )
            message_id, verdict, reason = entry["message_id"], entry["verdict"], entry["reason"]
            if type(message_id) is not int or message_id not in available:
                raise ToolArgumentError(
                    "unavailable_message_id",
                    "ID must identify a visible, unprotected earlier tool result",
                    item_index=index,
                    message_id=message_id,
                )
            if message_id in seen:
                raise ToolArgumentError("duplicate_message_id", "Each ID may appear once", item_index=index)
            if verdict not in VERDICTS:
                raise ToolArgumentError(
                    "invalid_verdict", f"verdict must be one of {', '.join(VERDICTS)}", item_index=index
                )
            if not isinstance(reason, str) or not reason.strip() or len(reason) > MAX_REASON_CHARS:
                raise ToolArgumentError(
                    "invalid_reason", f"reason must be 1 to {MAX_REASON_CHARS} characters", item_index=index
                )
            seen.add(message_id)
            call = producing_call(messages, message_id)
            records.append(
                PathRecord(
                    message_id,
                    call["function"]["name"],
                    canonical_arguments(call["function"].get("arguments")),
                    verdict,
                    reason.strip(),
                )
            )
        records.sort(key=lambda record: record.message_id)
        return CacheAction(tuple(seen), reposition), tuple(records)

    def read_result(
        self,
        arguments: str | Mapping[str, Any],
        content: Any,
        *,
        messages: Sequence[JsonDict],
        cache_operation_history: CacheOperationHistory,
    ) -> CacheAction | None:
        """Interpret a matching success receipt without executing the tool."""
        try:
            outcome = json.loads(content)
            action, records = self.action(
                arguments, messages, self.available_ids(messages, cache_operation_history)
            )
        except (ValueError, TypeError):
            return None
        return action if outcome == scheduled_result(action, records) else None


def scheduled_result(action: CacheAction, records: Sequence[PathRecord]) -> JsonDict:
    return {
        "ok": True,
        "status": "scheduled",
        "selected_message_ids": list(action.drop_message_ids),
        "reposition": action.reposition,
        "paths": [record.as_json() for record in records],
        "next_turn_instruction": (
            "Continue the original task. Cache changes take effect on the next model request; "
            "these notes remain visible after the results are gone."
        ),
    }


def receipt_records(messages: Sequence[JsonDict], tool_name: str) -> list[PathRecord]:
    """Read judged paths from this tool's successful receipts, oldest first."""
    records = []
    for i, message in enumerate(messages):
        if message.get("role") != "tool":
            continue
        call = producing_call(messages, i)
        if call is None or call["function"]["name"] != tool_name:
            continue
        try:
            outcome = json.loads(message.get("content") or "")
        except (ValueError, TypeError):
            continue
        if not isinstance(outcome, dict) or outcome.get("ok") is not True:
            continue
        for item in outcome.get("paths") or ():
            try:
                records.append(PathRecord(**item))
            except TypeError:
                continue
    return records


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b)) / norm if norm else 0.0


@dataclass
class PathMemory:
    """Ledger of judged paths, rebuilt from history on every query.

    History is the only state, so each conversation branch gets the ledger its own
    receipts imply. The optional embedder adds fuzzy matching between calls of the
    same tool; vectors are cached by text.
    """

    tool: SemanticDropTool
    embedder: Embedder | None = None
    threshold: float = 0.9
    guard_verdicts: tuple[str, ...] = ("failed", "resolved")
    max_guards: int = 1
    vectors: dict[str, tuple[float, ...]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not 0 < self.threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        if set(self.guard_verdicts) - set(VERDICTS):
            raise ValueError(f"guard_verdicts must be drawn from {', '.join(VERDICTS)}")
        if type(self.max_guards) is not int or self.max_guards < 0:
            raise ValueError("max_guards must be a nonnegative integer")

    def records(self, messages: Sequence[JsonDict]) -> list[PathRecord]:
        """The latest verdict for each distinct path."""
        latest = {record.key: record for record in receipt_records(messages, self.tool.name)}
        return list(latest.values())

    def match(
        self, name: str, arguments: str | Mapping[str, Any], messages: Sequence[JsonDict]
    ) -> PathRecord | None:
        records = [record for record in self.records(messages) if record.tool == name]
        key = path_key(name, arguments)
        for record in records:
            if record.key == key:
                return record
        if self.embedder is None or not records:
            return None
        query, *candidates = self.embed([key, *(record.key for record in records)])
        score, best = max(zip((cosine(query, c) for c in candidates), records), key=lambda pair: pair[0])
        return best if score >= self.threshold else None

    def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        missing = [text for text in dict.fromkeys(texts) if text not in self.vectors]
        if missing:
            vectors = self.embedder.embed(missing)
            if len(vectors) != len(missing):
                raise ValueError("Embedder must return one vector per text")
            self.vectors.update((text, tuple(vector)) for text, vector in zip(missing, vectors))
        return [self.vectors[text] for text in texts]

    def guard(self, call: ToolCall, messages: Sequence[JsonDict]) -> JsonDict | None:
        """Return a tool result to use instead of executing ``call``, or None to execute it.

        A path is answered from the ledger at most ``max_guards`` times; after that the
        call runs, so the model can still recover data it deliberately asks for again.
        """
        if call.name == self.tool.name:
            return None
        record = self.match(call.name, call.arguments, messages)
        if record is None or record.verdict not in self.guard_verdicts:
            return None
        if guard_count(messages, record.key) >= self.max_guards:
            return None
        instruction = (
            "This path was already tried and judged a dead end. Choose a different approach."
            if record.verdict == "failed"
            else "This path was already resolved; use the recorded facts above. Repeat the identical "
            "call only if you need the original result again."
        )
        return {
            "ok": False,
            "status": GUARD_STATUS,
            "path": record.key,
            "verdict": record.verdict,
            "reason": record.reason,
            "earlier_message_id": record.message_id,
            "instruction": instruction,
        }


def guard_outcome(message: JsonDict) -> JsonDict | None:
    """Return the parsed result if ``message`` is a PathMemory guard result."""
    if message.get("role") != "tool":
        return None
    try:
        outcome = json.loads(message.get("content") or "")
    except (ValueError, TypeError):
        return None
    return outcome if isinstance(outcome, dict) and outcome.get("status") == GUARD_STATUS else None


def guard_count(messages: Sequence[JsonDict], key: str) -> int:
    outcomes = (guard_outcome(message) for message in messages)
    return sum(1 for outcome in outcomes if outcome is not None and outcome.get("path") == key)


@dataclass(frozen=True)
class SemanticSelection:
    """Select labelled paths, rerun duplicates of discarded paths, and fallback picks.

    1. Results the model just labelled with ``SemanticDropTool``.
    2. Visible results whose call repeats a path judged ``failed`` or ``irrelevant``,
       unless the model reran it after a ``PathMemory`` guard warned it.
    3. Results chosen by ``fallback``, minus those whose content an embedder finds
       relevant to the user's task (cosine at least ``protect_threshold``).

    This tool's receipts and guard results are never selected, so the notes stay visible.
    """

    tool: SemanticDropTool
    fallback: Callable[[PolicyContext], set[int]] | None = None
    embedder: Embedder | None = None
    protect_threshold: float = 0.5
    discard_verdicts: tuple[str, ...] = ("failed", "irrelevant")

    def __post_init__(self) -> None:
        if not -1 <= self.protect_threshold <= 1:
            raise ValueError("protect_threshold must be in [-1, 1]")
        if set(self.discard_verdicts) - set(VERDICTS):
            raise ValueError(f"discard_verdicts must be drawn from {', '.join(VERDICTS)}")

    def __call__(self, context: PolicyContext) -> set[int]:
        messages = context.request.messages
        hidden = context.hidden
        notes = {i for i in self.tool_results(messages) if self.is_note(messages, i)}
        selected = set(tool_actions(context, self.tool).drop_message_ids)
        discarded = {
            record.key
            for record in receipt_records(messages, self.tool.name)
            if record.verdict in self.discard_verdicts
        }
        # New results are still unread; a rerun the model insisted on after a guard is kept.
        for i in self.tool_results(messages):
            call = producing_call(messages, i)
            if i >= context.cache_operation_history.message_count or call is None:
                continue
            key = path_key(call["function"]["name"], call["function"].get("arguments"))
            if key in discarded and not guard_count(messages[:i], key):
                selected.add(i)
        if self.fallback is not None:
            picked = self.fallback(context)
            selected |= picked - self.relevant(messages, picked - selected)
        return selected - notes - hidden

    def tool_results(self, messages: Sequence[JsonDict]) -> list[int]:
        return [i for i, message in enumerate(messages) if message.get("role") == "tool"]

    def is_note(self, messages: Sequence[JsonDict], i: int) -> bool:
        """This tool's receipts and PathMemory guard results carry the recorded reasons."""
        call = producing_call(messages, i)
        if call is not None and call["function"]["name"] == self.tool.name:
            return True
        return guard_outcome(messages[i]) is not None

    def relevant(self, messages: Sequence[JsonDict], ids: set[int]) -> set[int]:
        if self.embedder is None or not ids:
            return set()
        task = "\n".join(
            str(m.get("content") or "") for m in messages if m.get("role") == "user"
        ).strip()
        if not task:
            return set()
        ordered = sorted(ids)
        query, *vectors = self.embedder.embed([task, *(str(messages[i].get("content") or "") for i in ordered)])
        return {i for i, vector in zip(ordered, vectors) if cosine(query, vector) >= self.protect_threshold}
