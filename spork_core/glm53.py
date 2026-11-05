"""GLM-5.3-Flash XML tool-call helpers for the SPORK chat path."""

from __future__ import annotations

import json
import re
from typing import Any

from . import tool_calls


TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
TOOL_CALL_OPEN_TOKEN_ID = 154843
TOOL_CALL_CLOSE_TOKEN_ID = 154844
ARG_RE = re.compile(
    r"<arg_key>(?P<key>.*?)</arg_key>\s*"
    r"<arg_value>(?P<value>.*?)</arg_value>",
    re.DOTALL,
)
TOOL_CALL_RE = re.compile(
    r"<tool_call>(?P<body>.*?)</tool_call>",
    re.DOTALL,
)


def parse_tool_call(text: str) -> dict | None:
    """Parse the first complete GLM XML call, matching vLLM's glm47 parser."""
    match = TOOL_CALL_RE.search(text or "")
    if match is None:
        return None
    body = match.group("body")
    first_arg = body.find("<arg_key>")
    name = (body if first_arg < 0 else body[:first_arg]).strip()
    if not name:
        return None
    arguments = {
        arg.group("key").strip(): arg.group("value")
        for arg in ARG_RE.finditer(body)
        if arg.group("key").strip()
    }
    return {"name": name, "arguments": arguments}


def parse_probe_tool_call(
    text: str, forced_prefix: str = TOOL_CALL_OPEN
) -> dict | None:
    """Parse text generated after a prefilled ``<tool_call>`` token."""
    text = text or ""
    reconstructed = text if text.startswith(forced_prefix) else forced_prefix + text
    return parse_tool_call(reconstructed)


class Glm53ToolCallAdapter:
    """Chat-API adapter for GLM-5.3-Flash's GLM-4.7-compatible XML grammar."""

    name = "glm5.3_xml_tool_call"
    main_stop = (TOOL_CALL_CLOSE,)
    probe_stop = (TOOL_CALL_CLOSE,)
    uses_chat_api = True
    reasoning_message_field = "reasoning"
    tool_call_open = TOOL_CALL_OPEN
    tool_call_close = TOOL_CALL_CLOSE
    main_stop_token_ids = (TOOL_CALL_CLOSE_TOKEN_ID,)

    def __init__(self, tokenizer: Any = None, tools: list[dict] | None = None):
        self.tokenizer = tokenizer
        self.tools = tools or []

    def render_main_prompt(self, messages: list[dict], *, enable_thinking: bool) -> str:
        raise NotImplementedError("GLM-5.3 prompts are rendered by the chat API")

    def build_probe_prompt(
        self,
        messages: list[dict],
        *,
        enable_thinking: bool,
        observed_main_prefix: str | None = None,
    ) -> str:
        raise NotImplementedError("GLM-5.3 probes continue a chat assistant message")

    def build_probe_prefix(self, enable_thinking: bool) -> str:
        return ("</think>\n\n" if enable_thinking else "") + TOOL_CALL_OPEN

    def build_chat_probe_message(
        self, observed_prefix: str, *, enable_thinking: bool
    ) -> tuple[dict, str]:
        """Return the final assistant prefill expected by GLM's chat template."""
        if enable_thinking:
            reasoning = observed_prefix or ""
            if reasoning.startswith("<think>"):
                reasoning = reasoning[len("<think>") :]
            if reasoning.endswith("</think>"):
                reasoning = reasoning[: -len("</think>")]
            return {
                "role": "assistant",
                "reasoning": reasoning,
                "content": TOOL_CALL_OPEN,
            }, TOOL_CALL_OPEN
        return {
            "role": "assistant",
            "content": (observed_prefix or "") + TOOL_CALL_OPEN,
        }, TOOL_CALL_OPEN

    def build_probe_constraint_extra(self) -> None:
        return None

    def parse_tool_call(self, text: str) -> dict | None:
        return parse_tool_call(text)

    def parse_main_tool_call(self, text: str) -> dict | None:
        return parse_tool_call(text)

    def parse_probe_tool_call(
        self, text: str, forced_prefix: str = TOOL_CALL_OPEN
    ) -> dict | None:
        return parse_probe_tool_call(text, forced_prefix)

    def normalize_call(self, call: dict | None) -> dict | None:
        """Coerce GLM XML values to the declared tau2 JSON-schema types."""
        if call is None:
            return None
        properties = {}
        for tool in self.tools:
            fn = tool.get("function", tool)
            if fn.get("name") == call.get("name"):
                properties = fn.get("parameters", {}).get("properties", {})
                break
        arguments = {}
        for key, value in (call.get("arguments") or {}).items():
            declared_type = properties.get(key, {}).get("type")
            if isinstance(declared_type, list):
                declared_type = next(
                    (kind for kind in declared_type if kind != "null"), "string"
                )
            if isinstance(value, str) and declared_type in (
                "array", "object", "integer", "number", "boolean"
            ):
                try:
                    converted = json.loads(value)
                    expected_type = {
                        "array": list,
                        "object": dict,
                        "integer": int,
                        "number": (int, float),
                        "boolean": bool,
                    }[declared_type]
                    if isinstance(converted, expected_type) and not (
                        declared_type in ("integer", "number")
                        and isinstance(converted, bool)
                    ):
                        value = converted
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
            arguments[key] = value
        return {"name": call["name"], "arguments": arguments}

    def extract_probe_logprobs(self, logprobs_obj: dict | None) -> dict:
        return tool_calls.extract_full_logprobs(logprobs_obj)

    def first_token_top1_prob(self, logprobs: dict) -> float | None:
        return tool_calls.first_token_top1_prob(logprobs)

    def span_confidence(self, logprobs: dict, *, skip_first: bool = True) -> float | None:
        return tool_calls.span_min_prob(logprobs, skip_first=skip_first)

    def token_metrics(self, text: str) -> dict[str, Any]:
        if self.tokenizer is None:
            return {
                "cot_tokens": None,
                "tool_call_tokens": None,
                "post_tool_call_tokens": None,
                "total_decode_tokens": None,
                "token_count_approx": True,
            }
        ids = self.tokenizer.encode(text or "", add_special_tokens=False)
        total = len(ids)
        try:
            open_idx = ids.index(TOOL_CALL_OPEN_TOKEN_ID)
            close_idx = ids.index(TOOL_CALL_CLOSE_TOKEN_ID, open_idx + 1)
        except ValueError:
            return {
                "cot_tokens": total,
                "tool_call_tokens": 0,
                "post_tool_call_tokens": 0,
                "total_decode_tokens": total,
                "token_count_approx": False,
            }
        return {
            "cot_tokens": open_idx,
            "tool_call_tokens": close_idx - open_idx + 1,
            "post_tool_call_tokens": total - close_idx - 1,
            "total_decode_tokens": total,
            "token_count_approx": False,
        }

    def preview(self, text: str | None, limit: int = 200) -> str:
        return tool_calls.preview(text, limit=limit)
