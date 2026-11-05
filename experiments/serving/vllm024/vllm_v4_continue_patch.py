"""Compatibility patch for DeepSeek-V4 assistant-message continuation.

vLLM 0.24 forwards ``continue_final_message`` to the DeepSeek-V4 tokenizer,
but that tokenizer wrapper ignores the flag.  The underlying V4 encoder
already supports the equivalent per-message ``wo_eos`` field.  This module
bridges the two without modifying the installed vLLM package.
"""
from __future__ import annotations

import copy
import importlib.metadata
from typing import Any


def install_v4_continue_final_message_patch() -> bool:
    """Install the vLLM 0.24 V4 continuation patch once.

    Returns ``True`` when this call installs the patch and ``False`` when it
    was already installed.
    """
    if importlib.metadata.version('vllm') != '0.24.0':
        raise RuntimeError('the V4 continuation patch targets vLLM 0.24.0')
    import vllm.tokenizers.deepseek_v4 as deepseek_v4

    current = deepseek_v4.get_deepseek_v4_tokenizer
    if getattr(current, "_spork_continue_final_message_patch", False):
        return False

    original_get_tokenizer = current

    def get_patched_deepseek_v4_tokenizer(tokenizer: Any) -> Any:
        wrapped = original_get_tokenizer(tokenizer)
        tokenizer_class = wrapped.__class__
        original_apply_chat_template = tokenizer_class.apply_chat_template

        def apply_chat_template(
            self: Any,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> str | list[int]:
            if kwargs.get("continue_final_message"):
                conversation = copy.deepcopy(kwargs.get("conversation", messages))
                if not conversation or conversation[-1].get("role") != "assistant":
                    raise ValueError(
                        "continue_final_message requires a final assistant message"
                    )
                conversation[-1]["wo_eos"] = True
                kwargs["conversation"] = conversation
            return original_apply_chat_template(
                self, messages, tools=tools, **kwargs
            )

        tokenizer_class.apply_chat_template = apply_chat_template
        return wrapped

    get_patched_deepseek_v4_tokenizer._spork_continue_final_message_patch = True
    deepseek_v4.get_deepseek_v4_tokenizer = get_patched_deepseek_v4_tokenizer
    return True
