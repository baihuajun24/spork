"""Interfaces that keep SPORK orchestration model- and benchmark-agnostic."""
from __future__ import annotations

from typing import Any, Protocol


class ModelAdapter(Protocol):
    """Model-specific prompt, parsing, and token accounting hooks."""

    name: str
    main_stop: tuple[str, ...]
    probe_stop: tuple[str, ...]
    uses_chat_api: bool

    def render_main_prompt(self, messages: list[dict], *, enable_thinking: bool) -> str:
        ...

    def build_probe_prompt(
        self,
        messages: list[dict],
        *,
        enable_thinking: bool,
        observed_main_prefix: str | None = None,
    ) -> str:
        ...

    def parse_main_tool_call(self, text: str) -> dict | None:
        ...

    def parse_probe_tool_call(self, text: str) -> dict | None:
        ...

    def extract_probe_logprobs(self, logprobs_obj: dict | None) -> dict:
        ...

    def first_token_top1_prob(self, logprobs: dict) -> float | None:
        ...

    def span_confidence(self, logprobs: dict, *, skip_first: bool = True) -> float | None:
        ...

    def token_metrics(self, text: str) -> dict[str, Any]:
        ...

    def preview(self, text: str | None, limit: int = 200) -> str:
        ...


def append_assistant_tool_messages(
    messages: list[dict],
    assistant_text: str,
    tool_result: str,
) -> None:
    """Default OpenAI-style history update used by current benchmark adapters."""
    messages.append({"role": "assistant", "content": assistant_text})
    messages.append({"role": "tool", "content": tool_result})
