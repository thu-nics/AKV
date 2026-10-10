"""Sliding or triggered result windows with optional assistant deletion."""

import itertools
from copy import deepcopy
from dataclasses import dataclass

from akvalgo.messages import atomic_message_units, history_tool_calls

from .reposition import RepositionConfig, message_tokens, reposition_events
from .types import ChatTokenCounter, ContextDecision, PolicyRequest


@dataclass(frozen=True)
class AKVPolicy:
    keep_tool_responses: int
    mode: str = "kv_drop"
    rolling_drop_target: str = "tool"
    reposition_after_call: int | None = None
    reposition_interval_calls: int | None = None
    reposition_trigger_tokens: int | None = None
    token_counter: ChatTokenCounter | None = None
    drop_after_call: int | None = None
    drop_interval_calls: int | None = None
    drop_trigger_tokens: int | None = None
    drop_trigger_responses: int | None = None
    drop_target_tokens: int | None = None

    def __post_init__(self):
        if type(self.keep_tool_responses) is not int or self.keep_tool_responses <= 0:
            raise ValueError("keep_tool_responses must be a positive integer")
        if self.mode not in ("text_drop", "kv_drop"):
            raise ValueError("mode must be text_drop or kv_drop")
        if self.rolling_drop_target not in ("tool", "assistant", "both"):
            raise ValueError("rolling_drop_target must be tool, assistant, or both")
        for name in (
            "drop_after_call",
            "drop_interval_calls",
            "drop_trigger_tokens",
            "drop_trigger_responses",
            "drop_target_tokens",
        ):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer or null")
        if (
            sum(
                value is not None
                for value in (self.drop_after_call, self.drop_trigger_tokens, self.drop_trigger_responses)
            )
            > 1
        ):
            raise ValueError("Drop trigger modes are mutually exclusive")
        if self.drop_interval_calls is not None and self.drop_after_call is None:
            raise ValueError("drop_interval_calls requires drop_after_call")
        if self.triggered and (self.mode != "kv_drop" or self.rolling_drop_target != "tool"):
            raise ValueError("Triggered deletion requires tool-only kv_drop")
        if self.drop_target_tokens is not None:
            if not self.triggered:
                raise ValueError("drop_target_tokens requires a drop trigger")
            if self.drop_trigger_tokens is not None and self.drop_target_tokens >= self.drop_trigger_tokens:
                raise ValueError("drop_target_tokens must be below drop_trigger_tokens")
        if (
            self.drop_trigger_responses is not None
            and self.drop_target_tokens is None
            and self.drop_trigger_responses <= self.keep_tool_responses
        ):
            raise ValueError("Response trigger must exceed the retained response count")
        if (
            self.drop_trigger_tokens is not None or self.drop_target_tokens is not None
        ) and self.token_counter is None:
            raise ValueError("Token drop requires a token counter")
        RepositionConfig(
            self.reposition_after_call,
            self.reposition_interval_calls,
            self.reposition_trigger_tokens,
            self.token_counter,
        )
        if self.mode != "kv_drop" and (
            self.reposition_after_call is not None or self.reposition_trigger_tokens is not None
        ):
            raise ValueError("Reposition requires kv_drop")

    @property
    def triggered(self):
        return any(
            value is not None
            for value in (self.drop_after_call, self.drop_trigger_tokens, self.drop_trigger_responses)
        )

    def decide(self, request: PolicyRequest) -> ContextDecision:
        units = atomic_message_units(request.messages, 0)
        ids = tuple(i for i, message in enumerate(request.messages) if message["role"] == "tool")
        schedule = {trigger: (old,) for old, trigger in zip(ids, ids[self.keep_tool_responses :])}
        drop_metadata = {}
        if self.triggered:
            schedule, drop_metadata = triggered_drop_schedule(
                request,
                keep_tool_responses=self.keep_tool_responses,
                after_call=self.drop_after_call,
                interval_calls=self.drop_interval_calls,
                trigger_tokens=self.drop_trigger_tokens,
                trigger_responses=self.drop_trigger_responses,
                token_counter=self.token_counter,
                target_tokens=self.drop_target_tokens,
            )
        if self.rolling_drop_target != "tool":
            schedule = rolling_message_schedule(request, schedule, self.rolling_drop_target)
        dropped = {i for values in schedule.values() for i in values}
        reposition, tokens = reposition_events(
            request,
            schedule,
            after_call=self.reposition_after_call,
            interval_calls=self.reposition_interval_calls,
            trigger_tokens=self.reposition_trigger_tokens,
            token_counter=self.token_counter,
            prompt_end_message_ids_by_boundary=request.prompt_end_message_ids,
        )
        metadata = {
            "policy": "akv",
            "implementation_version": "rolling-tool-drop-v1",
            "mode": self.mode,
            "keep_tool_responses": self.keep_tool_responses,
            "rolling_drop_target": self.rolling_drop_target,
            "tool_response_count": len(ids),
            "retained_tool_response_count": sum(i not in dropped for i in ids),
            "dropped_tool_response_count": sum(i in dropped for i in ids),
            "dropped_assistant_message_count": sum(
                request.messages[i]["role"] == "assistant" for i in dropped
            ),
            "dropped_message_ids": sorted(dropped),
            "drop_message": schedule,
            "drop_event_count": len(schedule),
            "rolling_window_tool_response_ids": list(ids[-self.keep_tool_responses :]),
            "reposition_after_call": self.reposition_after_call,
            "reposition_interval_calls": self.reposition_interval_calls,
            "reposition_requested": bool(reposition),
            "reposition_message_ids": list(reposition or ()),
            "reposition_event_count": len(reposition or ()),
            **tokens,
            **drop_metadata,
        }
        messages = None
        if self.mode == "text_drop" and dropped:
            visible = []
            for unit in units:
                calls = {
                    call["id"]: call["function"]["name"]
                    for call in history_tool_calls(request.messages[unit[0]])
                }
                for i in unit:
                    if i in dropped:
                        continue
                    message = deepcopy(request.messages[i])
                    # A retained result still needs its tool author when its assistant is gone.
                    if message["role"] == "tool" and unit[0] in dropped and not message.get("name"):
                        message["name"] = calls[message["tool_call_id"]]
                    visible.append(message)
            messages = tuple(visible)
        return ContextDecision(
            messages=messages,
            drop_message=(schedule or None) if self.mode == "kv_drop" else None,
            reposition=reposition,
            metadata=metadata,
        )


def rolling_message_schedule(request, tool_schedule, target):
    """Delete an assistant once all results from its batch have expired."""
    expired = {i: trigger for trigger, ids in tool_schedule.items() for i in ids}
    schedule = {trigger: list(ids) for trigger, ids in tool_schedule.items()} if target == "both" else {}
    pending = []
    for unit in atomic_message_units(request.messages, 0):
        assistant, *results = unit
        if request.messages[assistant]["role"] != "assistant":
            continue
        pending.append(assistant)
        if results:
            if all(i in expired for i in results):
                schedule.setdefault(max(expired[i] for i in results), []).extend(pending)
            pending = []
    return {trigger: tuple(sorted(ids)) for trigger, ids in sorted(schedule.items())}


def triggered_drop_schedule(
    request: PolicyRequest,
    *,
    keep_tool_responses: int,
    after_call=None,
    interval_calls=None,
    trigger_tokens=None,
    trigger_responses=None,
    token_counter=None,
    target_tokens=None,
) -> tuple[dict, dict]:
    """Keep K newest results when an independent call/token trigger fires.

    Length is the exact prompt span minus all hidden tool-result spans, including
    system, assistant reasoning, tool declarations and the generation header.
    This attention-length metric is independent of reposition's position metric.
    A result batch is atomic for triggering, but each result is a deletion unit.
    Rebuilding from recorded boundaries makes cold resumes and retries identical.
    """
    boundaries = tuple(request.completed_call_message_ids)
    if (
        len(boundaries) != request.call_index
        or any(
            type(b) is not int
            or b <= 0
            or b >= len(request.messages)
            or request.messages[b].get("role") != "tool"
            for b in boundaries
        )
        or any(a >= b for a, b in itertools.pairwise(boundaries))
    ):
        raise ValueError("triggered drop requires complete ordered logical-call boundaries")
    layout = None
    if trigger_tokens is not None or target_tokens is not None:
        if token_counter is None:
            raise ValueError("token drop requires a token counter")
        layout = token_counter.message_layout(
            request.messages,
            request.tools,
            enable_thinking=request.enable_thinking,
            reasoning_effort=request.reasoning_effort,
            api_style=request.api_style,
        )
    schedule, events, hidden = {}, [], set()
    tool_ids = [i for i, message in enumerate(request.messages) if message.get("role") == "tool"]
    for call, boundary in enumerate(boundaries, 1):
        visible = [i for i in tool_ids if i <= boundary and i not in hidden]
        visible_tokens = None
        if layout is not None:
            visible_tokens = (
                layout.message_ends[boundary]
                + layout.generation_tokens
                - sum(message_tokens(layout, i) for i in hidden)
            )
        if trigger_tokens is not None:
            triggered = visible_tokens > trigger_tokens
        elif trigger_responses is not None:
            triggered = len(visible) >= trigger_responses
        else:
            triggered = (
                after_call is not None
                and call >= after_call
                and (
                    call == after_call
                    or interval_calls is not None
                    and (call - after_call) % interval_calls == 0
                )
            )
        selected = visible[:-keep_tool_responses] if triggered else []
        if triggered and target_tokens is not None:
            selected = []
            remaining = visible_tokens
            for message_id in visible:
                if remaining <= target_tokens:
                    break
                selected.append(message_id)
                remaining -= message_tokens(layout, message_id)
        if selected:
            schedule[boundary] = tuple(selected)
            hidden.update(selected)
            events.append(
                {
                    "call": call,
                    "message_id": boundary,
                    "dropped_message_ids": selected,
                    "retained_message_ids": [i for i in visible if i not in hidden],
                    "visible_response_count_before": len(visible),
                    "visible_response_count_after": len(visible) - len(selected),
                    "visible_tokens_before": visible_tokens,
                    "visible_tokens_after": (
                        visible_tokens - sum(message_tokens(layout, i) for i in selected)
                    )
                    if layout is not None
                    else None,
                    "target_tokens": target_tokens,
                    "target_reached": remaining <= target_tokens if target_tokens is not None else True,
                }
            )
    return schedule, {
        "drop_after_call": after_call,
        "drop_interval_calls": interval_calls,
        "drop_trigger_tokens": trigger_tokens,
        "drop_trigger_responses": trigger_responses,
        "drop_token_metric": "exact_prompt_minus_hidden_tool_spans" if layout is not None else None,
        "drop_trigger_events": events,
    }
