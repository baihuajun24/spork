"""Qwen3 tool-call rendering/parsing helpers."""
from __future__ import annotations

import json
import re
from typing import Any

from . import tool_calls


TOOL_CALL_OPEN_TOKEN_ID = 151657
TOOL_CALL_CLOSE_TOKEN_ID = 151658
FORK_PREFIX = '<tool_call>\n{"name": "'
THINK_END_PREFIX = '</think>\n\n<tool_call>\n{"name": "'
EXPECTED_NO_THINK_PROBE_TAIL = '<think>\n\n</think>\n\n<tool_call>\n{"name": "'
TOOLCALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

# Qwen3.5 XML-style tool calls
QWEN35_TOOL_CALL_OPEN_TOKEN_ID = 248058
QWEN35_TOOL_CALL_CLOSE_TOKEN_ID = 248059
QWEN35_FORK_PREFIX = '<tool_call>\n<function='
QWEN35_THINK_END_PREFIX = '</think>\n\n<tool_call>\n<function='
QWEN35_TOOLCALL_RE = re.compile(
    r"<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>", re.DOTALL
)
QWEN35_PARAM_RE = re.compile(r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", re.DOTALL)


def render_prompt(tokenizer: Any, messages: list[dict], tools: list[dict],
                  enable_thinking: bool) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def build_probe_prompt(
    tokenizer: Any,
    messages: list[dict],
    tools: list[dict],
    *,
    enable_thinking: bool = False,
    observed_main_prefix: str | None = None,
) -> str:
    prompt = render_prompt(tokenizer, messages, tools, enable_thinking=enable_thinking)
    if enable_thinking:
        prefix = observed_main_prefix or "<think>\n"
        if "<think" in prefix and "</think>" not in prefix:
            return prompt + prefix + "\n" + THINK_END_PREFIX
        return prompt + "<think>\n\n" + THINK_END_PREFIX
    return prompt + FORK_PREFIX


def normalize_args(args: Any) -> dict:
    return tool_calls.normalize_args(args)


def parse_baseline_tool_call(text: str) -> dict | None:
    m = TOOLCALL_RE.search(text or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and obj.get("name"):
        return {"name": obj["name"], "arguments": normalize_args(obj.get("arguments", {}))}
    return None


def parse_qwen35_baseline_tool_call(text: str) -> dict | None:
    """Parse Qwen3.5 XML-style tool calls, with JSON fallback."""
    m = QWEN35_TOOLCALL_RE.search(text or "")
    if m:
        func_name = m.group(1)
        params_text = m.group(2)
        arguments = {}
        for pm in QWEN35_PARAM_RE.finditer(params_text):
            arguments[pm.group(1)] = pm.group(2).strip()
        return {"name": func_name, "arguments": normalize_args(arguments)}
    return parse_baseline_tool_call(text)


def parse_probe_tool_call(text: str) -> dict | None:
    """Parse text generated after '<tool_call>\n{"name": "'."""
    reconstructed = '{"name": "' + (text or "")
    depth = 0
    end = None
    in_str = False
    esc = False
    for i, ch in enumerate(reconstructed):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return None
    try:
        obj = json.loads(reconstructed[:end])
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and obj.get("name"):
        return {"name": obj["name"], "arguments": normalize_args(obj.get("arguments", {}))}
    return None


def _extract_json_object(text: str) -> str | None:
    depth = 0
    end = None
    in_str = False
    esc = False
    for i, ch in enumerate(text or ""):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return None
    body = text[:end]
    try:
        json.loads(body)
    except json.JSONDecodeError:
        return None
    return body


def probe_body_text(text: str, *, xml_mode: bool = False) -> str | None:
    """Return the structured tool-call body generated after the fork prefix."""
    if not text:
        return None
    if xml_mode:
        parsed = parse_qwen35_probe_tool_call(text)
        if not parsed:
            return None
        body = f"<function={parsed['name']}>"
        for key, value in parsed.get("arguments", {}).items():
            body += f"\n<parameter={key}>\n{value}\n</parameter>"
        body += "\n</function>"
        return body
    return _extract_json_object('{"name": "' + text)


def tool_body_text(text: str, *, xml_mode: bool = False) -> str | None:
    """Extract the main output's tool-call body for D3 prefix diagnostics."""
    if xml_mode:
        m = QWEN35_TOOLCALL_RE.search(text or "")
        if m:
            return f"<function={m.group(1)}>{m.group(2)}</function>"
    m = TOOLCALL_RE.search(text or "")
    if not m:
        return None
    body = m.group(1).strip()
    try:
        json.loads(body)
    except json.JSONDecodeError:
        return None
    return body


def parse_qwen35_probe_tool_call(text: str) -> dict | None:
    """Parse text generated after '<tool_call>\n<function='.

    Probe output starts AFTER '<function=', so text looks like:
      "get_user_details>\n<parameter=user_id>\nraj_sanchez_7340\n</parameter>\n</function>"
    """
    if not text:
        return None
    # Extract function name: everything before first ">"
    gt_idx = text.find(">")
    if gt_idx < 0:
        return None
    func_name = text[:gt_idx].strip()
    if not func_name or not re.match(r"^[\w.-]+$", func_name):
        return None
    # Extract parameters from the remainder
    remainder = text[gt_idx + 1:]
    arguments = {}
    for pm in QWEN35_PARAM_RE.finditer(remainder):
        arguments[pm.group(1)] = pm.group(2).strip()
    return {"name": func_name, "arguments": normalize_args(arguments)}


def build_qwen35_probe_prompt(
    tokenizer: Any,
    messages: list[dict],
    tools: list[dict],
    *,
    enable_thinking: bool = False,
    observed_main_prefix: str | None = None,
) -> str:
    """Build probe prompt for Qwen3.5 with XML-style fork prefix.

    render_prompt with enable_thinking=True already ends with '<think>\n'.
    We append observed CoT (if any) then close with '</think>\\n\\n<tool_call>\\n<function='.
    """
    prompt = render_prompt(tokenizer, messages, tools, enable_thinking=enable_thinking)
    if enable_thinking:
        # prompt already ends with <think>\n from the chat template
        # Just append any observed CoT content + the closing prefix
        if observed_main_prefix:
            # Strip leading <think> if present (it's already in prompt)
            cot = observed_main_prefix
            if cot.startswith("<think>"):
                cot = cot[len("<think>"):]
            if cot.startswith("\n"):
                cot = cot[1:]
            return prompt + cot + "\n" + QWEN35_THINK_END_PREFIX
        # No observed prefix: empty CoT, just close immediately
        return prompt + "\n" + QWEN35_THINK_END_PREFIX
    return prompt + QWEN35_FORK_PREFIX


def args_exact_match(a: dict | None, b: dict | None) -> bool:
    return tool_calls.args_exact_match(a, b)


def args_partial_overlap(a: dict | None, b: dict | None) -> float:
    return tool_calls.args_partial_overlap(a, b)


def extract_full_logprobs(logprobs_obj: dict | None, max_positions: int = 16) -> dict:
    return tool_calls.extract_full_logprobs(logprobs_obj, max_positions=max_positions)


def first_token_top1_prob(logprobs: dict) -> float | None:
    return tool_calls.first_token_top1_prob(logprobs)


def span_min_prob(logprobs: dict, *, skip_first: bool = True) -> float | None:
    return tool_calls.span_min_prob(logprobs, skip_first=skip_first)


def token_split(text: str, tokenizer: Any) -> dict:
    ids = tokenizer.encode(text or "", add_special_tokens=False)
    total = len(ids)
    try:
        open_idx = ids.index(TOOL_CALL_OPEN_TOKEN_ID)
        close_idx = ids.index(TOOL_CALL_CLOSE_TOKEN_ID)
    except ValueError:
        return {
            "cot_tokens": total,
            "tool_call_tokens": 0,
            "post_tool_call_tokens": 0,
            "total_decode_tokens": total,
        }
    if close_idx < open_idx:
        return {
            "cot_tokens": total,
            "tool_call_tokens": 0,
            "post_tool_call_tokens": 0,
            "total_decode_tokens": total,
        }
    return {
        "cot_tokens": open_idx,
        "tool_call_tokens": close_idx - open_idx + 1,
        "post_tool_call_tokens": total - close_idx - 1,
        "total_decode_tokens": total,
    }


def preview(text: str | None, limit: int = 200) -> str:
    return tool_calls.preview(text, limit=limit)


class QwenToolCallAdapter:
    """Qwen chat-template adapter for the generic SPORK turn runner."""

    name = "qwen3_tool_call"
    main_stop = ("</tool_call>",)
    probe_stop = ("</tool_call>",)
    uses_chat_api = False

    def __init__(self, tokenizer: Any, tools: list[dict]):
        self.tokenizer = tokenizer
        self.tools = tools

    def render_main_prompt(self, messages: list[dict], *, enable_thinking: bool) -> str:
        return render_prompt(self.tokenizer, messages, self.tools, enable_thinking)

    def build_probe_prompt(
        self,
        messages: list[dict],
        *,
        enable_thinking: bool,
        observed_main_prefix: str | None = None,
    ) -> str:
        return build_probe_prompt(
            self.tokenizer,
            messages,
            self.tools,
            enable_thinking=enable_thinking,
            observed_main_prefix=observed_main_prefix,
        )

    def parse_main_tool_call(self, text: str) -> dict | None:
        return parse_baseline_tool_call(text)

    def parse_probe_tool_call(self, text: str) -> dict | None:
        return parse_probe_tool_call(text)

    def extract_probe_logprobs(self, logprobs_obj: dict | None) -> dict:
        return extract_full_logprobs(logprobs_obj)

    def first_token_top1_prob(self, logprobs: dict) -> float | None:
        return first_token_top1_prob(logprobs)

    def span_confidence(self, logprobs: dict, *, skip_first: bool = True) -> float | None:
        return span_min_prob(logprobs, skip_first=skip_first)

    def token_metrics(self, text: str) -> dict[str, Any]:
        return token_split(text, self.tokenizer)

    def preview(self, text: str | None, limit: int = 200) -> str:
        return preview(text, limit=limit)


class Qwen35ToolCallAdapter(QwenToolCallAdapter):
    """Adapter for Qwen3.5-27B which uses XML-style tool calls."""

    name = "qwen3.5_tool_call"
    main_stop_token_ids = (QWEN35_TOOL_CALL_OPEN_TOKEN_ID,)

    def build_probe_prompt(
        self,
        messages: list[dict],
        *,
        enable_thinking: bool,
        observed_main_prefix: str | None = None,
    ) -> str:
        return build_qwen35_probe_prompt(
            self.tokenizer,
            messages,
            self.tools,
            enable_thinking=enable_thinking,
            observed_main_prefix=observed_main_prefix,
        )

    def parse_main_tool_call(self, text: str) -> dict | None:
        return parse_qwen35_baseline_tool_call(text)

    def parse_probe_tool_call(self, text: str) -> dict | None:
        return parse_qwen35_probe_tool_call(text)

    def token_metrics(self, text: str) -> dict[str, Any]:
        ids = self.tokenizer.encode(text or "", add_special_tokens=False)
        total = len(ids)
        try:
            open_idx = ids.index(QWEN35_TOOL_CALL_OPEN_TOKEN_ID)
            close_idx = ids.index(QWEN35_TOOL_CALL_CLOSE_TOKEN_ID)
        except ValueError:
            return {
                "cot_tokens": total,
                "tool_call_tokens": 0,
                "post_tool_call_tokens": 0,
                "total_decode_tokens": total,
            }
        if close_idx < open_idx:
            return {
                "cot_tokens": total,
                "tool_call_tokens": 0,
                "post_tool_call_tokens": 0,
                "total_decode_tokens": total,
            }
        return {
            "cot_tokens": open_idx,
            "tool_call_tokens": close_idx - open_idx + 1,
            "post_tool_call_tokens": total - close_idx - 1,
            "total_decode_tokens": total,
        }
