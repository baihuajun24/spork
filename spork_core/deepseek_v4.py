"""DeepSeek-V4-Flash DSML tool-call rendering/parsing helpers.

Mirrors the QwenToolCallAdapter interface (see spork_core/adapters.py and
spork_core/qwen.py) but targets V4's DSML tool-call grammar instead of Qwen's
JSON / XML <tool_call> format.

V4 DSML tool-call grammar (the bar is U+FF5C FULLWIDTH VERTICAL BAR):

    <｜DSML｜tool_calls>
    <｜DSML｜invoke name="TOOL">
    <｜DSML｜parameter name="KEY" string="true|false">VALUE</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>

Param typing: when ``string="true"`` the value is a literal string; when
``string="false"`` the value is JSON (json.loads). We additionally coerce to the
tool's JSON-schema type when a schema is available, porting the logic from
vLLM's ``tool_parsers/deepseekv32_tool_parser.py`` (V4 subclasses that parser,
overriding only the block token from ``function_calls`` to ``tool_calls``).

This module is intentionally import-light (re, json, typing only). It does NOT
import torch / vllm / transformers at module top so it can be imported and unit
tested standalone. ``token_metrics`` falls back to an approximate whitespace /
character count when no tokenizer is supplied; V4 exposes no integer tool-call
token ids, so ``token_split`` / D3 prefix diagnostics are unsupported (the D2
confidence signal comes from API logprobs, not from this counter).
"""
from __future__ import annotations

import json
import math
import os
import re
from typing import Any

from . import tool_calls


# --- DSML literal tokens (｜ == U+FF5C FULLWIDTH VERTICAL BAR) -----------------
TOOL_CALL_OPEN = "<｜DSML｜tool_calls>"
TOOL_CALL_CLOSE = "</｜DSML｜tool_calls>"
INVOKE_OPEN_PREFIX = '<｜DSML｜invoke name="'

# V4 has no dedicated integer tool-call token ids (unlike Qwen's 151657/151658),
# so token_split / D3 prefix diagnostics are unsupported.
TOOL_CALL_OPEN_TOKEN_ID = None
TOOL_CALL_CLOSE_TOKEN_ID = None

# --- Probe prefill suffixes ---------------------------------------------------
# The probe prefills an assistant message that ends right at the invoke-name
# opener, so the continuation generates "TOOL"...>...</invoke>.
PROBE_PREFIX_THINK = '</think>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name="'
PROBE_PREFIX_NO_THINK = '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="'

# --- DSML regexes ported from deepseekv32_tool_parser.py ----------------------
INVOKE_RE = re.compile(
    r'<｜DSML｜invoke\s+name="([^"]+)"\s*>(.*?)</｜DSML｜invoke>', re.DOTALL
)
PARAMETER_TYPED_RE = re.compile(
    r'<｜DSML｜parameter\s+name="([^"]+)"(?:\s+string="(true|false)")?\s*>(.*?)</｜DSML｜parameter>',
    re.DOTALL,
)


# --- Schema-aware param typing (ported from deepseekv32_tool_parser.py) -------
def _convert_param_value_checked(value: str, param_type: str) -> Any:
    """Convert a parameter value to the correct type. Mirrors upstream."""
    param_type = str(param_type).lower()
    if param_type in ("string", "str", "text"):
        return value
    if value.lower() == "null":
        return None
    if param_type in ("integer", "int"):
        return int(value)
    if param_type in ("number", "float"):
        val = float(value)
        return val if val != int(val) else int(val)
    if param_type in ("boolean", "bool"):
        value = value.strip()
        if value.lower() not in ("false", "0", "true", "1"):
            raise ValueError("Invalid boolean value")
        return value.lower() in ("true", "1")
    if param_type in ("object", "array"):
        return json.loads(value)
    return json.loads(value)


def _convert_param_value(value: str, param_type: Any) -> Any:
    """Try each candidate type; fall back to the raw string. Mirrors upstream."""
    if not isinstance(param_type, list):
        param_type = [param_type]
    for current_type in param_type:
        try:
            return _convert_param_value_checked(value, current_type)
        except Exception:  # noqa: BLE001
            continue
    return value


def _tool_properties(tools: list | None, function_name: str) -> dict:
    """Return the JSON-schema ``properties`` map for ``function_name``.

    Tolerates both OpenAI-style tool dicts (the shape the SPORK eval passes:
    ``{"type":"function","function":{"name":..., "parameters":{"properties":...}}}``)
    and vLLM ``Tool`` objects (``tool.function.name`` / ``tool.function.parameters``).
    """
    if not tools:
        return {}
    for tool in tools:
        fn = None
        if isinstance(tool, dict):
            fn = tool.get("function", tool)
        elif hasattr(tool, "function"):
            fn = tool.function
        if fn is None:
            continue
        name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
        if name != function_name:
            continue
        params = (
            fn.get("parameters") if isinstance(fn, dict)
            else getattr(fn, "parameters", None)
        )
        if isinstance(params, dict) and isinstance(params.get("properties"), dict):
            return params["properties"]
        return {}
    return {}


def _iter_tool_defs(tools: list | None) -> list[tuple[str, dict]]:
    """Return ``(name, properties)`` for available tools."""
    out: list[tuple[str, dict]] = []
    if not tools:
        return out
    for tool in tools:
        fn = None
        if isinstance(tool, dict):
            fn = tool.get("function", tool)
        elif hasattr(tool, "function"):
            fn = tool.function
        if fn is None:
            continue
        name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
        if not name:
            continue
        params = (
            fn.get("parameters") if isinstance(fn, dict)
            else getattr(fn, "parameters", None)
        )
        properties = {}
        if isinstance(params, dict) and isinstance(params.get("properties"), dict):
            properties = params["properties"]
        out.append((name, properties))
    return out


def _canonical_tool_name_hint(tools: list | None, variant: str) -> str:
    names = [name for name, _ in _iter_tool_defs(tools)]
    if not names:
        return ""
    joined = ", ".join(f"`{name}`" for name in names)
    if variant == "none":
        return ""
    if variant == "next_tool":
        return (
            "Next I should use this tool. The tool function name must be copied "
            f"exactly from this list: {joined}. Do not output a number, menu "
            "label, or description.\n"
        )
    if variant == "dsml_contract":
        return (
            "I will now write only one DSML tool call. The DSML invoke name must "
            f"be exactly one function name from: {joined}. No prose, no menu "
            "labels, no numbering.\n"
        )
    if variant == "name_only":
        return (
            "The next characters after name=\" must be copied exactly from this "
            f"function-name list: {joined}. Immediately after the copied name, "
            "write \"> and then the DSML parameter tags. Do not write any other "
            "word before the function name.\n"
        )
    if variant == "no_placeholders":
        return (
            "Choose the real tool now. Valid function names are: "
            f"{joined}. Never write placeholder/internal strings such as "
            "PENDING, PREFIX_WITH_UNDERSCORE_IF_NEEDED, Cursor, UNKNOWN, a "
            "number, or a menu label.\n"
        )
    if variant == "fewshot":
        first = names[0]
        return (
            "Use canonical DSML exactly like this pattern: "
            f"<｜DSML｜invoke name=\"{first}\">. For this request, replace "
            f"{first} with exactly one of: {joined}. No prose or numbering.\n"
        )
    return (
        "Use exactly one canonical function name from this list for the invoke "
        f"name: {joined}. Do not output a number, menu label, or description.\n"
    )


def _normalize_invoke_name(
    invoke_name: str,
    typed_params: list[tuple[str, str, str]],
    tools: list | None,
) -> str:
    """Map parser-mode probe labels back to function names when unambiguous.

    In vLLM parser mode, a V4 probe can continue the prefilled DSML opener with
    a rendered menu label such as ``"1. Look up user by email"`` instead of the
    canonical function name. If the parameter names identify exactly one tool,
    use that tool name so D1/D2 can compare against structured main calls.
    """
    tool_defs = _iter_tool_defs(tools)
    if not tool_defs:
        return invoke_name
    known_names = {name for name, _ in tool_defs}
    if invoke_name in known_names:
        return invoke_name

    arg_names = {name for name, _, _ in typed_params}
    if not arg_names:
        return invoke_name
    candidates = [
        name for name, properties in tool_defs
        if properties and arg_names.issubset(set(properties))
    ]
    if len(candidates) == 1:
        return candidates[0]
    return invoke_name


def _convert_params(
    function_name: str,
    typed_params: list[tuple[str, str, str]],
    tools: list | None,
) -> dict:
    """Convert raw DSML param values to typed Python values.

    ``typed_params`` is a list of ``(name, string_flag, raw_value)`` tuples where
    ``string_flag`` is the literal "true"/"false" of the DSML ``string="..."``
    attribute. Schema type (when a tool schema is available) takes precedence;
    otherwise the string="" heuristic decides literal-string vs json.loads.
    """
    properties = _tool_properties(tools, function_name)
    converted: dict[str, Any] = {}
    for name, string_flag, raw in typed_params:
        if name in properties and isinstance(properties[name], dict):
            # Schema-aware: coerce to the declared JSON-schema type.
            param_type = properties[name].get("type", "string")
            converted[name] = _convert_param_value(raw, param_type)
        elif string_flag == "true":
            # No schema: string="true" => literal string.
            converted[name] = raw
        else:
            # No schema: string="false" => JSON value (fall back to raw on error).
            try:
                converted[name] = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                converted[name] = raw
    return converted


def _parse_invoke(invoke_name: str, invoke_body: str, tools: list | None) -> dict:
    typed_params: list[tuple[str, str, str]] = [
        (m.group(1), m.group(2) or "true", m.group(3))
        for m in PARAMETER_TYPED_RE.finditer(invoke_body)
    ]
    invoke_name = _normalize_invoke_name(invoke_name, typed_params, tools)
    arguments = _convert_params(invoke_name, typed_params, tools)
    return {"name": invoke_name, "arguments": arguments}


def parse_tool_call(text: str, tools: list | None = None) -> dict | None:
    """Parse the FIRST complete DSML invoke from ``text``.

    Tolerates a missing outer ``<｜DSML｜tool_calls>`` wrapper. Returns
    ``{"name": str, "arguments": dict}`` with typed args, or None if no complete
    ``<｜DSML｜invoke ...>...</｜DSML｜invoke>`` is present.
    """
    if not text:
        return None
    m = INVOKE_RE.search(text)
    if not m:
        return None
    return _parse_invoke(m.group(1), m.group(2), tools)


def parse_probe_tool_call(
    text: str,
    forced_prefix: str = INVOKE_OPEN_PREFIX,
    tools: list | None = None,
) -> dict | None:
    """Parse a probe continuation generated after a prefilled invoke opener.

    The probe is generated AFTER an assistant message that was prefilled to end
    in ``forced_prefix`` (e.g. ``<｜DSML｜invoke name="`` or the full opener with
    the ``<｜DSML｜tool_calls>`` wrapper). We reconstruct the full invoke by
    prepending the cut prefix and parse with ``parse_tool_call``.
    """
    text = text or ""
    if text.startswith(forced_prefix):
        reconstructed = text
    else:
        # forced_prefix ends with the invoke-name opener; only re-prepend the
        # piece that is not already present in the generated text.
        reconstructed = forced_prefix + text
    parsed = parse_tool_call(reconstructed, tools)
    if parsed is not None:
        known_names = {name for name, _ in _iter_tool_defs(tools)}
        if not known_names or parsed.get("name") in known_names:
            return parsed

    # Raw-mode probes can sometimes continue directly with parameter tags after
    # the prefilled invoke opener. If the parameters identify exactly one tool,
    # recover that function name from the schema.
    typed_params: list[tuple[str, str, str]] = [
        (m.group(1), m.group(2) or "true", m.group(3))
        for m in PARAMETER_TYPED_RE.finditer(text)
    ]
    if not typed_params:
        return None
    inferred_name = _normalize_invoke_name("", typed_params, tools)
    if not inferred_name:
        return None
    known_names = {name for name, _ in _iter_tool_defs(tools)}
    if inferred_name not in known_names:
        return None
    return {
        "name": inferred_name,
        "arguments": _convert_params(inferred_name, typed_params, tools),
    }


def build_probe_prefix(enable_thinking: bool, tools: list | None = None) -> str:
    """Return the exact assistant-suffix string to prefill for the probe.

    With thinking on the suffix closes the think block first; with thinking off
    it goes straight to the tool-call opener. When tool schemas are available,
    include a short canonical-name hint immediately before the DSML opener. Both
    variants end at the invoke-name opener so the continuation generates the
    tool name.
    """
    variant = os.environ.get("SPORK_V4_PROBE_PROMPT_VARIANT", "none")
    prefix = PROBE_PREFIX_THINK if enable_thinking else PROBE_PREFIX_NO_THINK
    hint = _canonical_tool_name_hint(tools, variant)
    if not hint:
        return prefix
    if enable_thinking:
        return '</think>\n\n' + hint + '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="'
    return hint + '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="'


def build_probe_constraint_extra(tools: list | None = None) -> dict | None:
    """Return optional vLLM structured-output constraints for the probe.

    ``SPORK_V4_PROBE_CONSTRAINT=regex`` constrains the continuation immediately
    after ``<｜DSML｜invoke name="`` to one of the active canonical tool names,
    then releases the rest of the DSML invoke body to arbitrary text.
    """
    mode = os.environ.get("SPORK_V4_PROBE_CONSTRAINT", "").strip().lower()
    if mode in ("", "none", "off", "0", "false"):
        return None
    if mode != "regex":
        raise ValueError(f"Unsupported SPORK_V4_PROBE_CONSTRAINT={mode!r}")

    names = sorted({name for name, _ in _iter_tool_defs(tools)})
    if not names:
        return None
    name_alt = "|".join(re.escape(name) for name in names)
    # The forced assistant suffix already ends with '<｜DSML｜invoke name="'.
    # The generated continuation should start with TOOL"> then free-form params.
    regex = f'(?:{name_alt})"\\s*>[\\s\\S]*'
    return {"structured_outputs": {"regex": regex}}


def format_d3_draft_after_boundary(probe_call: dict, tools: list | None = None) -> str:
    """Return DSML draft text after ``<...invoke name="`` for V4 HTTP D3.

    The V4 proposer watches for the token sequence of:
    ``<｜DSML｜tool_calls>\n<｜DSML｜invoke name="``.
    Once that boundary is generated, the draft should begin with the function
    name and include the rest of the DSML block.
    """
    name = probe_call.get("name") or ""
    args = probe_call.get("arguments") or {}
    properties = _tool_properties(tools, name)
    parts = [f'{name}">']
    for key, value in args.items():
        prop = properties.get(key, {}) if isinstance(properties, dict) else {}
        typ = prop.get("type") if isinstance(prop, dict) else None
        is_string = typ in (None, "string") and isinstance(value, str)
        if is_string:
            string_flag = "true"
            raw_value = value
        else:
            string_flag = "false"
            raw_value = json.dumps(value, ensure_ascii=False)
        parts.append(
            f'<｜DSML｜parameter name="{key}" string="{string_flag}">'
            f'{raw_value}</｜DSML｜parameter>'
        )
    parts.append("</｜DSML｜invoke>")
    parts.append(TOOL_CALL_CLOSE)
    return "\n".join(parts)


def preview(text: str | None, limit: int = 200) -> str:
    return tool_calls.preview(text, limit=limit)


class DeepSeekV4ToolCallAdapter:
    """DeepSeek-V4-Flash DSML adapter for the generic SPORK turn runner.

    Conforms to the same ModelAdapter Protocol as QwenToolCallAdapter (same
    method names / signatures / return types). Constructor mirrors Qwen's
    ``(tokenizer, tools)`` shape and tolerates ``tokenizer=None`` — V4 has no
    client-side jinja chat_template, so prompts are rendered server-side and the
    tokenizer is only used (best-effort) for token counting.
    """

    name = "deepseek_v4_dsml_tool_call"
    main_stop = (TOOL_CALL_CLOSE,)
    probe_stop = (TOOL_CALL_CLOSE,)
    uses_chat_api = True

    # String constants for callers that introspect the grammar.
    tool_call_open = TOOL_CALL_OPEN
    tool_call_close = TOOL_CALL_CLOSE
    # No integer tool-call token ids on V4 -> token_split / D3 unsupported.
    main_stop_token_ids: tuple[int, ...] = ()

    def __init__(self, tokenizer: Any = None, tools: list[dict] | None = None):
        self.tokenizer = tokenizer
        self.tools = tools or []

    # --- prompt rendering -----------------------------------------------------
    def render_main_prompt(self, messages: list[dict], *, enable_thinking: bool) -> str:
        """V4 has no client-side jinja chat_template (apply_chat_template fails).

        Prompts are rendered SERVER-SIDE by vLLM from the OpenAI ``messages`` list
        via /v1/chat/completions. This adapter therefore does not produce a flat
        prompt string. The V4 integration drives /v1/chat/completions directly
        with ``messages`` and never calls this method; it raises to make the
        unsupported path explicit if a /v1/completions code path reaches here.
        """
        raise NotImplementedError(
            "V4 renders prompts server-side via /v1/chat/completions; "
            "the V4 turn runner should pass `messages` to the chat endpoint "
            "rather than calling render_main_prompt()."
        )

    def build_probe_prompt(
        self,
        messages: list[dict],
        *,
        enable_thinking: bool,
        observed_main_prefix: str | None = None,
    ) -> str:
        """Same rationale as render_main_prompt: the probe is driven via
        /v1/chat/completions with ``continue_final_message=true`` and a prefilled
        assistant message whose suffix is ``build_probe_prefix(enable_thinking)``.
        Use ``build_probe_prefix`` to obtain that suffix.
        """
        raise NotImplementedError(
            "V4 probe is driven via /v1/chat/completions with "
            "continue_final_message=true; prefill the assistant message with "
            "build_probe_prefix(enable_thinking) instead of calling "
            "build_probe_prompt()."
        )

    def build_probe_prefix(self, enable_thinking: bool) -> str:
        return build_probe_prefix(enable_thinking, self.tools)

    def build_probe_constraint_extra(self) -> dict | None:
        return build_probe_constraint_extra(self.tools)

    def format_d3_draft(self, probe_call: dict) -> str:
        return format_d3_draft_after_boundary(probe_call, self.tools)

    # --- parsing --------------------------------------------------------------
    def parse_tool_call(self, text: str) -> dict | None:
        return parse_tool_call(text, self.tools)

    def parse_main_tool_call(self, text: str) -> dict | None:
        return parse_tool_call(text, self.tools)

    def parse_probe_tool_call(
        self, text: str, forced_prefix: str = INVOKE_OPEN_PREFIX
    ) -> dict | None:
        return parse_probe_tool_call(text, forced_prefix, self.tools)

    # --- logprob helpers (identical to Qwen; API logprobs are model-agnostic) -
    def extract_probe_logprobs(self, logprobs_obj: dict | None) -> dict:
        return tool_calls.extract_full_logprobs(logprobs_obj)

    def first_token_top1_prob(self, logprobs: dict) -> float | None:
        return tool_calls.first_token_top1_prob(logprobs)

    def span_confidence(self, logprobs: dict, *, skip_first: bool = True) -> float | None:
        return tool_calls.span_min_prob(logprobs, skip_first=skip_first)

    # --- token accounting (best-effort; no V4 tool-call token ids) ------------
    def token_metrics(self, text: str) -> dict[str, Any]:
        """Best-effort token counting.

        V4 exposes no integer tool-call token ids, so we cannot split decode
        tokens into cot / tool_call / post_tool_call segments the way Qwen does.
        We report total decode tokens only (exact when a tokenizer is supplied,
        otherwise an approximate whitespace/char-based estimate flagged with
        ``token_count_approx=True``). The D2 confidence signal does NOT depend on
        this — it comes from /v1/chat/completions logprobs.
        """
        text = text or ""
        if self.tokenizer is not None:
            try:
                total = len(self.tokenizer.encode(text, add_special_tokens=False))
                approx = False
            except Exception:  # noqa: BLE001
                total, approx = self._approx_tokens(text), True
        else:
            total, approx = self._approx_tokens(text), True
        return {
            "cot_tokens": None,
            "tool_call_tokens": None,
            "post_tool_call_tokens": None,
            "total_decode_tokens": total,
            "token_count_approx": approx,
        }

    @staticmethod
    def _approx_tokens(text: str) -> int:
        """~4 chars/token heuristic, floored by whitespace word count."""
        if not text:
            return 0
        words = len(text.split())
        chars = math.ceil(len(text) / 4)
        return max(words, chars)

    def preview(self, text: str | None, limit: int = 200) -> str:
        return preview(text, limit=limit)
