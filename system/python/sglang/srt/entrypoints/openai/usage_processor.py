from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, final

from sglang.srt.entrypoints.openai.protocol import PromptTokensDetails, UsageInfo


@final
class UsageProcessor:
    """Stateless helpers that turn raw token counts into a UsageInfo."""

    @staticmethod
    def _details_if_cached(count: int) -> Optional[PromptTokensDetails]:
        """Return PromptTokensDetails only when count > 0 (keeps JSON slim)."""
        return PromptTokensDetails(cached_tokens=count) if count > 0 else None

    @staticmethod
    def prompt_tokens_details(
        meta_infos: Iterable[Mapping[str, Any]], enable_cache_report: bool = False
    ) -> Optional[PromptTokensDetails]:
        """Count each prompt once; Context counters retain P's accounting."""
        cached = repos = skipped = 0
        has_context = False
        for meta in meta_infos:
            context = meta.get("context_usage")
            if context is not None:
                has_context = True
                cached += context["cached_tokens"]
                repos += context["repos_tokens"]
                skipped += context["drop_skipped_tokens"]
            elif enable_cache_report:
                cached += meta.get("cached_tokens", 0)
        if has_context:
            return PromptTokensDetails(
                cached_tokens=cached, repos_tokens=repos, drop_skipped_tokens=skipped
            )
        return UsageProcessor._details_if_cached(cached)

    @staticmethod
    def calculate_response_usage(
        responses: List[Dict[str, Any]],
        n_choices: int = 1,
        enable_cache_report: bool = False,
        image_tokens: int = 0,
        audio_tokens: int = 0,
        video_tokens: int = 0,
    ) -> UsageInfo:
        completion_tokens = sum(
            r["meta_info"].get("completion_tokens", 0) for r in responses
        )
        prompt_tokens = sum(
            responses[i]["meta_info"].get("prompt_tokens", 0)
            for i in range(0, len(responses), n_choices)
        )

        # some API don't have reasoning_tokens semantics
        reasoning_tokens = sum(
            r["meta_info"].get("reasoning_tokens", 0) for r in responses
        )

        cached_details = UsageProcessor.prompt_tokens_details(
            (responses[i]["meta_info"] for i in range(0, len(responses), n_choices)),
            enable_cache_report,
        )

        return UsageProcessor.calculate_token_usage(
            prompt_tokens=prompt_tokens,
            reasoning_tokens=reasoning_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_details,
            image_tokens=image_tokens,
            audio_tokens=audio_tokens,
            video_tokens=video_tokens,
        )

    @staticmethod
    def calculate_streaming_usage(
        prompt_tokens: Mapping[int, int],
        reasoning_tokens: Mapping[int, int],
        completion_tokens: Mapping[int, int],
        cached_tokens: Mapping[int, int],
        n_choices: int,
        enable_cache_report: bool = False,
        image_tokens: int = 0,
        audio_tokens: int = 0,
        video_tokens: int = 0,
        context_usage: Optional[Mapping[int, Mapping[str, int]]] = None,
    ) -> UsageInfo:
        # index % n_choices == 0 marks the first choice of a prompt
        total_prompt_tokens = sum(
            tok for idx, tok in prompt_tokens.items() if idx % n_choices == 0
        )
        total_reasoning_tokens = sum(reasoning_tokens.values())
        total_completion_tokens = sum(completion_tokens.values())

        cached_details = UsageProcessor.prompt_tokens_details(
            (
                {
                    "cached_tokens": cached_tokens.get(idx, 0),
                    "context_usage": (context_usage or {}).get(idx),
                }
                for idx in prompt_tokens
                if idx % n_choices == 0
            ),
            enable_cache_report,
        )

        return UsageProcessor.calculate_token_usage(
            prompt_tokens=total_prompt_tokens,
            reasoning_tokens=total_reasoning_tokens,
            completion_tokens=total_completion_tokens,
            cached_tokens=cached_details,
            image_tokens=image_tokens,
            audio_tokens=audio_tokens,
            video_tokens=video_tokens,
        )

    @staticmethod
    def calculate_token_usage(
        prompt_tokens: int,
        completion_tokens: int,
        reasoning_tokens: Optional[int] = 0,
        cached_tokens: Optional[PromptTokensDetails] = None,
        image_tokens: int = 0,
        audio_tokens: int = 0,
        video_tokens: int = 0,
    ) -> UsageInfo:
        """Calculate token usage information"""
        # `cached_tokens` is already a PromptTokensDetails (or None) carrying the
        # cached count. Attach multimodal counts to the same object, creating one
        # only when there is something to report so plain-text requests keep
        # prompt_tokens_details=None (backward compatible).
        details = cached_tokens
        if image_tokens or audio_tokens or video_tokens:
            if details is None:
                details = PromptTokensDetails()
            if image_tokens:
                details.image_tokens = image_tokens
            if audio_tokens:
                details.audio_tokens = audio_tokens
            if video_tokens:
                details.video_tokens = video_tokens

        return UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=details,
            reasoning_tokens=reasoning_tokens,
        )
