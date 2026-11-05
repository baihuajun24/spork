"""Reusable baseline/SPORK turn runner."""
from __future__ import annotations

import asyncio
import re
import json
import time
from dataclasses import dataclass
from typing import Any

import aiohttp

from . import adapters, gates, tool_calls
from .adapters import ModelAdapter
from .executors import SpeculativeToolExecution, ToolExecutor
from .executors import commit_speculative, execute_speculative
from .vllm_client import VllmClient, chat_logprobs_to_flat


@dataclass
class TurnConfig:
    gate: str = "args_exact_strict"
    confidence_threshold: float = 0.90
    enable_thinking: bool = False
    first_token_timeout_s: float = 5.0
    baseline_max_tokens: int = 1024
    context_window_tokens: int = 32768
    probe_max_tokens: int = 160
    seed: int = 42
    main_stop: tuple[str, ...] = ("</tool_call>",)
    probe_after_tokens: int = 1


def _fork_point(turn: int) -> str:
    return "post_user_prefill" if turn == 1 else "post_tool_result_prefill"


def _normalize_openai_tool_call(tc: dict | None) -> dict | None:
    if not isinstance(tc, dict):
        return None
    fn = tc.get("function") or {}
    if not isinstance(fn, dict) or not fn.get("name"):
        return None
    raw_args = fn.get("arguments") or {}
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args) if raw_args.strip() else {}
        except json.JSONDecodeError:
            args = {"_raw": raw_args}
    elif isinstance(raw_args, dict):
        args = raw_args
    else:
        args = {}
    return {"name": fn["name"], "arguments": args}


def _first_structured_tool_call(res: dict) -> dict | None:
    for tc in res.get("tool_calls") or []:
        call = _normalize_openai_tool_call(tc)
        if call is not None:
            return call
    return None


def _probe_call_from_chat_choice(
    choice: dict,
    model_adapter: ModelAdapter,
    probe_prefix: str,
) -> tuple[dict | None, str]:
    """Extract a forced V4 probe across vLLM chat response variants.

    vLLM 0.24 can classify a continued assistant prefill as ``reasoning`` even
    when the generated suffix is DSML. Parser-mode deployments may instead
    return a structured tool call, while older raw deployments return content.
    """
    message = choice.get("message") or {}
    structured = _first_structured_tool_call(message)
    if structured is not None:
        return structured, "structured_tool_calls"
    for field in ("content", "reasoning", "reasoning_content"):
        generated = message.get(field)
        if generated:
            return model_adapter.parse_probe_tool_call(generated, probe_prefix), field
    return None, "empty"


def _append_tool_messages(
    messages: list[dict],
    assistant_text: str,
    tool_result: str,
    *,
    tool_calls_structured: list[dict] | None = None,
    tool_results: list[str] | None = None,
    reasoning: str | None = None,
    reasoning_field: str = "reasoning",
) -> None:
    """Append history after a tool call, preserving OpenAI tool_calls when present."""
    if assistant_text:
        assistant_text = re.sub(r"^\s*<think>.*?</think>\s*", "", assistant_text, count=1, flags=re.DOTALL)
    if not tool_calls_structured:
        adapters.append_assistant_tool_messages(messages, assistant_text, tool_result)
        return
    results = tool_results if tool_results is not None else [tool_result]
    if len(results) != len(tool_calls_structured):
        raise ValueError("structured tool calls and results must have equal lengths")
    normalized_calls = []
    for index, raw_call in enumerate(tool_calls_structured):
        tc = dict(raw_call)
        fn = dict(tc.get("function") or {})
        raw_args = fn.get("arguments")
        if isinstance(raw_args, str) and raw_args.strip():
            try:
                json.loads(raw_args)
            except json.JSONDecodeError:
                # Preserve the exact malformed arguments as transport JSON.
                fn["arguments"] = json.dumps({"_raw": raw_args}, ensure_ascii=False)
        if not tc.get("id"):
            tc["id"] = f"spork-tool-call-{index}"
        tc["type"] = tc.get("type") or "function"
        tc["function"] = fn
        normalized_calls.append((tc, fn, results[index]))
    assistant_message = {
        "role": "assistant",
        "content": assistant_text or "",
        "tool_calls": [tc for tc, _, _ in normalized_calls],
    }
    if reasoning:
        assistant_message[reasoning_field] = reasoning
    messages.append(assistant_message)
    for tc, fn, result in normalized_calls:
        messages.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "name": fn.get("name") or "",
            "content": result,
        })


def _build_chat_probe_message(
    model_adapter: ModelAdapter,
    observed_prefix: str,
    *,
    enable_thinking: bool,
) -> tuple[dict, str]:
    """Build a DeepSeek-V4 assistant prefill for a forced DSML probe.

    In vLLM 0.24 thinking mode, previous chain-of-thought belongs in the
    assistant ``reasoning`` field. The renderer itself closes ``</think>``
    before rendering ``content``, so the content prefill must not repeat that
    closer. The returned prefix is the exact string used to reconstruct the
    probe continuation for the repository DSML parser.
    """
    if hasattr(model_adapter, "build_chat_probe_message"):
        return model_adapter.build_chat_probe_message(
            observed_prefix, enable_thinking=enable_thinking,
        )
    probe_prefix = model_adapter.build_probe_prefix(enable_thinking)
    if enable_thinking:
        think_close = "</think>\n\n"
        if not probe_prefix.startswith(think_close):
            raise ValueError("thinking probe prefix must start with </think>")
        content_prefix = probe_prefix[len(think_close):]
        return {
            "role": "assistant",
            "reasoning": observed_prefix,
            "content": content_prefix,
        }, content_prefix
    return {
        "role": "assistant",
        "content": observed_prefix + probe_prefix,
    }, probe_prefix


async def _bounded_chat_max_tokens(
    session: aiohttp.ClientSession,
    client: VllmClient,
    model_adapter: ModelAdapter,
    messages: list[dict],
    requested_max_tokens: int,
    context_window_tokens: int,
) -> int:
    """Keep OpenAI chat requests inside the shared prompt+output window.

    V4 prompts are rendered server-side, so use vLLM's render endpoint instead
    of approximating the chat template locally. The full message history is
    preserved; only the per-request output allowance is reduced when tool
    outputs have consumed part of the 32k context window.
    """
    if context_window_tokens <= 0:
        return max(1, requested_max_tokens)
    try:
        prompt_tokens = await client.render_chat_token_count(
            session,
            messages,
            tools=getattr(model_adapter, "tools", None) or None,
        )
    except Exception:
        return max(1, requested_max_tokens)
    remaining = max(1, context_window_tokens - prompt_tokens)
    return max(1, min(requested_max_tokens, remaining))


async def run_baseline_turn(
    session: aiohttp.ClientSession,
    client: VllmClient,
    model_adapter: ModelAdapter,
    messages: list[dict],
    executor: ToolExecutor,
    turn: int,
    cfg: TurnConfig,
) -> dict:
    uses_chat = getattr(model_adapter, "uses_chat_api", False)
    if not uses_chat:
        prompt = model_adapter.render_main_prompt(messages, enable_thinking=cfg.enable_thinking)
    t0 = time.time()
    first_delay = {"s": None}
    main_max_tokens = cfg.baseline_max_tokens

    async def on_first(delay_s: float, _text: str) -> None:
        first_delay["s"] = delay_s

    if uses_chat:
        main_max_tokens = await _bounded_chat_max_tokens(
            session, client, model_adapter, messages,
            cfg.baseline_max_tokens, cfg.context_window_tokens,
        )
        # V4 chat-API path renders server-side; vLLM 0.24 normally consumes
        # the DSML body and returns structured OpenAI tool-call deltas.
        res = await client.stream_chat(
            session,
            messages,
            max_tokens=main_max_tokens,
            stop=list(model_adapter.main_stop),
            seed=cfg.seed,
            tools=getattr(model_adapter, "tools", None) or None,
            capture_stream_events=True,
            on_first_token=on_first,
        )
    else:
        res = await client.stream(
            session,
            prompt,
            max_tokens=cfg.baseline_max_tokens,
            stop=list(cfg.main_stop or model_adapter.main_stop),
            seed=cfg.seed,
            on_first_token=on_first,
        )
    main_end_ms = (time.time() - t0) * 1000.0
    text = res["text"]
    structured_calls = []
    structured_calls_raw = []
    if uses_chat:
        for tc in res.get("tool_calls") or []:
            call = _normalize_openai_tool_call(tc)
            if call is not None:
                structured_calls.append(call)
                structured_calls_raw.append(tc)
    if structured_calls:
        main_calls = structured_calls
    else:
        main_call = model_adapter.parse_main_tool_call(text)
        main_calls = [main_call] if main_call is not None else []
    if hasattr(model_adapter, "normalize_call"):
        main_calls = [model_adapter.normalize_call(call) for call in main_calls]
    main_call = main_calls[0] if main_calls else None
    has_tool = bool(main_calls)
    canonical_dispatch_ms = None
    canonical_end_ms = None
    tool_result = ""
    tool_results = []
    tool_wall_s = 0.0
    if has_tool:
        canonical_dispatch_ms = (time.time() - t0) * 1000.0
        if len(main_calls) > 1:
            if not hasattr(executor, "execute_many"):
                raise RuntimeError("executor does not support native multi-tool turns")
            executed = await asyncio.to_thread(executor.execute_many, main_calls)
            tool_results = [item[0] for item in executed]
            tool_wall_s = sum(item[1] for item in executed)
            tool_result = tool_results[0]
        else:
            tool_result, tool_wall_s = await asyncio.to_thread(executor.execute, main_call)
            tool_results = [tool_result]
        canonical_end_ms = (time.time() - t0) * 1000.0
        _append_tool_messages(
            messages,
            text,
            tool_result,
            tool_calls_structured=structured_calls_raw if structured_calls else None,
            tool_results=tool_results if uses_chat else None,
            reasoning=res.get("reasoning") if uses_chat else None,
            reasoning_field=getattr(model_adapter, "reasoning_message_field", "reasoning"),
        )
    turn_end_ms = (time.time() - t0) * 1000.0
    entry = {
        "turn": turn,
        "fork_point": _fork_point(turn),
        "mode": "baseline",
        "main_first_token_ms": (
            res["first_token_s"] * 1000.0 if res["first_token_s"] is not None else None
        ),
        "main_end_ms": main_end_ms,
        "canonical_tool_dispatch_ms": canonical_dispatch_ms,
        "canonical_tool_end_ms": canonical_end_ms,
        "canonical_tool_wall_s": round(tool_wall_s, 3),
        "real_tool_wall_s": round(executor.last_real_wall_s, 3),
        "floor_sleep_s": round(executor.last_floor_sleep_s, 3),
        "turn_end_ms": turn_end_ms,
        "main_max_tokens": main_max_tokens,
        "baseline_has_tool_call": has_tool,
        "baseline_tool_name": main_call["name"] if has_tool else None,
        "baseline_tool_args": main_call.get("arguments") if has_tool else None,
        "baseline_tool_calls": main_calls,
        "baseline_text": text,
        "main_usage": res.get("usage"),
        "main_reasoning": res.get("reasoning", "") if uses_chat else "",
        "main_structured_tool_call_count": len(res.get("tool_calls") or []) if uses_chat else 0,
        "main_structured_tool_calls": res.get("tool_calls") if uses_chat else None,
        "main_stream_events": res.get("stream_events") if uses_chat else None,
        **model_adapter.token_metrics(text),
    }
    if has_tool:
        entry["tool_result_len"] = len(tool_result)
        entry["tool_result_preview"] = model_adapter.preview(tool_result)
    else:
        entry["final_answer"] = text.strip()
    return entry


async def run_spork_turn(
    session: aiohttp.ClientSession,
    client: VllmClient,
    model_adapter: ModelAdapter,
    messages: list[dict],
    executor: ToolExecutor,
    turn: int,
    cfg: TurnConfig,
) -> dict:
    uses_chat = getattr(model_adapter, "uses_chat_api", False)
    if not uses_chat:
        main_prompt = model_adapter.render_main_prompt(messages, enable_thinking=cfg.enable_thinking)
    main_max_tokens = cfg.baseline_max_tokens
    t0 = time.time()

    first_token_event = asyncio.Event()
    first_token_s: float | None = None
    first_token_abs: float | None = None
    first_token_text: str | None = None
    observed_main_parts: list[str] = []
    observed_main_token_count = 0
    probe_observed_main_token_count: int | None = None

    async def on_first(delay_s: float, text: str) -> None:
        nonlocal first_token_s, first_token_abs, first_token_text
        first_token_s = delay_s
        first_token_abs = time.time()
        first_token_text = text
        first_token_event.set()

    async def on_token(text: str, token_index: int) -> None:
        nonlocal observed_main_token_count
        observed_main_parts.append(text)
        observed_main_token_count = token_index

    if uses_chat:
        main_max_tokens = await _bounded_chat_max_tokens(
            session, client, model_adapter, messages,
            cfg.baseline_max_tokens, cfg.context_window_tokens,
        )
        main_task = asyncio.create_task(
            client.stream_chat(
                session,
                messages,
                max_tokens=main_max_tokens,
                stop=list(model_adapter.main_stop),
                seed=cfg.seed,
                tools=getattr(model_adapter, "tools", None) or None,
                capture_stream_events=True,
                on_first_token=on_first,
                on_token=on_token,
            )
        )
    else:
        main_task = asyncio.create_task(
            client.stream(
                session,
                main_prompt,
                max_tokens=cfg.baseline_max_tokens,
                stop=list(cfg.main_stop or model_adapter.main_stop),
                seed=cfg.seed,
                on_first_token=on_first,
                on_token=on_token,
            )
        )

    probe_dispatch_ms = None
    probe_end_ms = None
    probe_wall_s = 0.0
    probe_logprobs = dict(tool_calls.EMPTY_LOGPROBS)
    probe_call = None
    first_token_timeout = False
    spec_task: asyncio.Task[SpeculativeToolExecution] | None = None
    spec_dispatch_ms = None
    spec_end_ms = None
    spec_wall_s = 0.0
    spec_result = ""
    spec_execution: SpeculativeToolExecution | None = None
    spec_worker_error: str | None = None
    probe_dispatch_after_first_token_ms = None
    probe_source = None
    probe_response = None

    try:
        await asyncio.wait_for(first_token_event.wait(), timeout=cfg.first_token_timeout_s)
    except asyncio.TimeoutError:
        first_token_timeout = True

    if not first_token_timeout:
        target_tokens = max(1, cfg.probe_after_tokens)
        while observed_main_token_count < target_tokens and not main_task.done():
            await asyncio.sleep(0.005)
        observed_main_prefix = "".join(observed_main_parts) or (first_token_text or "")
        probe_observed_main_token_count = observed_main_token_count
        probe_dispatch_abs = time.time()
        probe_dispatch_ms = (probe_dispatch_abs - t0) * 1000.0
        probe_dispatch_after_first_token_ms = (
            (probe_dispatch_abs - first_token_abs) * 1000.0
            if first_token_abs is not None
            else None
        )
        if uses_chat:
            # V4 probe: prefill an assistant message ending in the DSML opener and
            # continue it. cot_so_far is the main prefix observed at the configured
            # token position, matching the Qwen observed_main_prefix semantics.
            cot_so_far = observed_main_prefix
            if hasattr(model_adapter, "build_chat_probe_message"):
                probe_message, probe_prefix = model_adapter.build_chat_probe_message(
                    cot_so_far, enable_thinking=cfg.enable_thinking,
                )
            else:
                probe_message, probe_prefix = _build_chat_probe_message(
                    model_adapter, cot_so_far, enable_thinking=cfg.enable_thinking,
                )
            probe_messages = list(messages) + [probe_message]
            probe_max_tokens = await _bounded_chat_max_tokens(
                session, client, model_adapter, probe_messages,
                cfg.probe_max_tokens, cfg.context_window_tokens,
            )
            choice, probe_wall_s = await client.complete_chat(
                session,
                probe_messages,
                max_tokens=probe_max_tokens,
                temperature=0.0,
                stop=list(model_adapter.probe_stop),
                seed=cfg.seed,
                logprobs=True,
                top_logprobs=5,
                tools=getattr(model_adapter, "tools", None) or None,
                continue_final_message=True,
                add_generation_prompt=False,
                extra=model_adapter.build_probe_constraint_extra()
                if hasattr(model_adapter, "build_probe_constraint_extra")
                else None,
            )
            probe_end_ms = (time.time() - t0) * 1000.0
            flat_lp = chat_logprobs_to_flat(choice)
            probe_logprobs = model_adapter.extract_probe_logprobs(flat_lp)
            probe_call, probe_source = _probe_call_from_chat_choice(
                choice, model_adapter, probe_prefix,
            )
            if hasattr(model_adapter, "normalize_call"):
                probe_call = model_adapter.normalize_call(probe_call)
            probe_response = choice
        else:
            probe_prompt = model_adapter.build_probe_prompt(
                messages,
                enable_thinking=cfg.enable_thinking,
                observed_main_prefix=observed_main_prefix,
            )
            choice, probe_wall_s = await client.complete(
                session,
                probe_prompt,
                max_tokens=cfg.probe_max_tokens,
                stop=list(model_adapter.probe_stop),
                logprobs=5,
                seed=cfg.seed,
            )
            probe_end_ms = (time.time() - t0) * 1000.0
            probe_logprobs = model_adapter.extract_probe_logprobs(choice.get("logprobs"))
            probe_call = model_adapter.parse_probe_tool_call(choice.get("text", ""))
            if hasattr(model_adapter, "normalize_call"):
                probe_call = model_adapter.normalize_call(probe_call)
            probe_source = "completion_text"
            probe_response = choice

    # For confidence-gated variants, the dispatch decision is known after probe but before
    # main finishes. For args_exact_strict we dispatch optimistically and verify later.
    dispatch_by_confidence = True
    if cfg.gate in ("confidence_strict", "confidence_name_loose"):
        conf = model_adapter.span_confidence(probe_logprobs, skip_first=True)
        dispatch_by_confidence = conf is not None and conf >= cfg.confidence_threshold
    if probe_call and dispatch_by_confidence:
        spec_dispatch_ms = (time.time() - t0) * 1000.0
        spec_task = asyncio.create_task(asyncio.to_thread(execute_speculative, executor, probe_call))

    main_res = await main_task
    main_end_ms = (time.time() - t0) * 1000.0
    text = main_res["text"]
    main_call = _first_structured_tool_call(main_res) if uses_chat else None
    if main_call is None:
        main_call = model_adapter.parse_main_tool_call(text)
    if hasattr(model_adapter, "normalize_call"):
        main_call = model_adapter.normalize_call(main_call)
    has_tool = main_call is not None
    decision = gates.decide_gate(
        cfg.gate,
        probe_call,
        main_call,
        probe_logprobs,
        confidence_threshold=cfg.confidence_threshold,
    )

    canonical_dispatch_ms = None
    canonical_end_ms = None
    spec_used = False
    spec_wasted = False
    tool_result = ""
    tool_real_wall_s = 0.0
    tool_artificial_sleep_s = 0.0
    if has_tool:
        if decision.accept_after_main and spec_task is not None:
            try:
                spec_execution = await spec_task
                commit_speculative(executor, spec_execution)
                tool_result = spec_execution.text
                spec_wall_s = spec_execution.wall_s
                tool_real_wall_s = spec_execution.real_wall_s
                tool_artificial_sleep_s = spec_execution.artificial_sleep_s
                spec_end_ms = (time.time() - t0) * 1000.0
                spec_used = True
            except Exception as exc:  # noqa: BLE001
                spec_worker_error = f"{type(exc).__name__}: {exc}"
        else:
            if spec_task is not None:
                spec_wasted = True
        if not spec_used:
            canonical_dispatch_ms = (time.time() - t0) * 1000.0
            tool_result, _ = await asyncio.to_thread(executor.execute, main_call)
            tool_real_wall_s = executor.last_real_wall_s
            tool_artificial_sleep_s = executor.last_floor_sleep_s
            canonical_end_ms = (time.time() - t0) * 1000.0
        _append_tool_messages(
            messages,
            text,
            tool_result,
            tool_calls_structured=main_res.get("tool_calls") if uses_chat else None,
            reasoning=main_res.get("reasoning") if uses_chat else None,
            reasoning_field=getattr(model_adapter, "reasoning_message_field", "reasoning"),
        )

    observed_turn_end_ms = (time.time() - t0) * 1000.0
    reconcile_ready_ms = main_end_ms
    if spec_task is not None and not spec_used:
        spec_wasted = True
    cleanup_start = time.time()
    if spec_task is not None and spec_execution is None:
        try:
            spec_execution = await spec_task
            spec_wall_s = spec_execution.wall_s
        except Exception as exc:  # noqa: BLE001
            spec_worker_error = f"{type(exc).__name__}: {exc}"
    cleanup_wait_ms = (time.time() - cleanup_start) * 1000.0

    spec_trace = None
    if spec_task is not None:
        raw_trace = spec_execution.trace if spec_execution is not None else None
        def relative_ms(key: str) -> float | None:
            value = (raw_trace or {}).get(key)
            return round((value - t0) * 1000.0, 3) if value is not None else None

        worker_start_ms = relative_ms("worker_start_s")
        worker_end_ms = relative_ms("worker_end_s")
        if spec_used and worker_start_ms is not None and worker_end_ms is not None:
            worker_duration_ms = max(0.0, worker_end_ms - worker_start_ms)
            real_overlap_ms = min(
                worker_duration_ms,
                max(0.0, reconcile_ready_ms - worker_start_ms),
            )
        else:
            real_overlap_ms = 0.0
        spec_actual_end_ms = worker_end_ms
        spec_trace = {
            "launched": True,
            "committed": spec_used,
            "wasted": spec_wasted,
            "exact_match": decision.args_exact,
            "predicted_call": probe_call,
            "snapshot_start_ms": relative_ms("snapshot_start_s"),
            "snapshot_end_ms": relative_ms("snapshot_end_s"),
            "worker_start_ms": worker_start_ms,
            "worker_end_ms": worker_end_ms,
            "commit_start_ms": relative_ms("commit_start_s"),
            "commit_end_ms": relative_ms("commit_end_s"),
            "tool_wall_s": round(spec_wall_s, 3) if spec_execution is not None else None,
            "real_tool_wall_s": (
                round(spec_execution.real_wall_s, 3) if spec_execution is not None else None
            ),
            "floor_sleep_s": (
                round(spec_execution.artificial_sleep_s, 3)
                if spec_execution is not None else None
            ),
            "worker_error": spec_worker_error,
        }
    else:
        real_overlap_ms = 0.0
        spec_actual_end_ms = None

    turn_end_ms = observed_turn_end_ms
    if spec_used:
        tool_result_ready_ms = max(reconcile_ready_ms, spec_actual_end_ms or reconcile_ready_ms)
    elif has_tool:
        tool_result_ready_ms = canonical_end_ms
    else:
        tool_result_ready_ms = None
    entry = {
        "turn": turn,
        "fork_point": _fork_point(turn),
        "mode": "spork",
        "gate_variant": cfg.gate,
        "confidence_threshold": cfg.confidence_threshold,
        "main_max_tokens": main_max_tokens,
        "main_first_token_ms": (
            main_res["first_token_s"] * 1000.0
            if main_res["first_token_s"] is not None
            else None
        ),
        "main_end_ms": main_end_ms,
        "main_dispatch_ms": 0.0,
        "main_tool_call_ready_ms": main_end_ms,
        "reconcile_ready_ms": reconcile_ready_ms,
        "probe_dispatch_ms": probe_dispatch_ms,
        "probe_dispatch_after_first_token_ms": probe_dispatch_after_first_token_ms,
        "probe_after_tokens": max(1, cfg.probe_after_tokens),
        "probe_observed_main_token_count": probe_observed_main_token_count,
        "probe_observed_main_prefix": (
            observed_main_prefix if not first_token_timeout else None
        ),
        "probe_end_ms": probe_end_ms,
        "probe_wall_s": round(probe_wall_s, 3),
        "spec_tool_dispatch_ms": spec_dispatch_ms,
        "spec_tool_end_ms": spec_end_ms,
        "spec_tool_actual_end_ms": spec_actual_end_ms,
        "spec_tool_wall_s": round(spec_wall_s, 3),
        "canonical_tool_dispatch_ms": canonical_dispatch_ms,
        "canonical_tool_end_ms": canonical_end_ms,
        "real_tool_wall_s": round(tool_real_wall_s, 3),
        "floor_sleep_s": round(tool_artificial_sleep_s, 3),
        "turn_end_ms": turn_end_ms,
        "real_overlap_ms": round(real_overlap_ms, 3),
        "first_token_timeout": first_token_timeout,
        "baseline_has_tool_call": has_tool,
        "baseline_tool_name": main_call["name"] if has_tool else None,
        "baseline_tool_args": main_call.get("arguments") if has_tool else None,
        "baseline_text": text,
        "main_usage": main_res.get("usage"),
        "probe_usage": (probe_response or {}).get("usage"),
        "main_reasoning": main_res.get("reasoning", "") if uses_chat else "",
        "main_structured_tool_call_count": (
            len(main_res.get("tool_calls") or []) if uses_chat else 0
        ),
        "main_structured_tool_calls": main_res.get("tool_calls") if uses_chat else None,
        "main_stream_events": main_res.get("stream_events") if uses_chat else None,
        **model_adapter.token_metrics(text),
        "probe_tool_name": probe_call["name"] if probe_call else None,
        "probe_tool_args": probe_call.get("arguments") if probe_call else None,
        "probe_parse_ok": probe_call is not None,
        "probe_response_source": probe_source,
        "probe_response": probe_response,
        "probe_logprobs": probe_logprobs,
        "probe_first_token_top1_prob": model_adapter.first_token_top1_prob(probe_logprobs),
        "probe_span_min_prob_skip0": decision.confidence,
        "hit_name": decision.name_match,
        "hit_args_exact": decision.args_exact,
        "args_partial_overlap": round(decision.args_partial, 3),
        "gate_match": decision.accept_after_main,
        "spec_dispatched": spec_task is not None,
        "spec_used": spec_used,
        "spec_wasted": spec_wasted,
        "spec_cleanup_wait_ms_excluded_from_e2e": round(cleanup_wait_ms, 3),
        "probe_trace": ([{
            "probe_id": 1,
            "dispatch_ms": probe_dispatch_ms,
            "completion_ms": probe_end_ms,
            "parsed_call": probe_call,
            "response_source": probe_source,
        }] if probe_dispatch_ms is not None else []),
        "speculative_tool": spec_trace,
        "critical_path_hidden_ms": round(real_overlap_ms, 3),
        "tool_result_ready_ms": (
            round(tool_result_ready_ms, 3) if tool_result_ready_ms is not None else None
        ),
    }
    if has_tool:
        entry["tool_result_len"] = len(tool_result)
        entry["tool_result_preview"] = model_adapter.preview(tool_result)
    else:
        entry["final_answer"] = text.strip()
    return entry


async def run_spork_cross_turn(
    session: aiohttp.ClientSession,
    client: VllmClient,
    model_adapter: ModelAdapter,
    probe_client: VllmClient,
    probe_adapter: ModelAdapter,
    messages: list[dict],
    executor: ToolExecutor,
    turn: int,
    cfg: TurnConfig,
) -> dict:
    """Cross-model SPORK turn: main decode on `client`/`model_adapter`,
    speculative probe on a SEPARATE `probe_client`/`probe_adapter`.

    Mirrors run_spork_turn exactly, except the probe completion is rendered with
    the probe model's tokenizer/template and dispatched to the probe endpoint.
    The observed main CoT prefix still comes from the main stream; the gate still
    compares the probe's parsed tool call against the MAIN model's eventual call.
    """
    uses_chat = getattr(model_adapter, "uses_chat_api", False)
    main_prompt = model_adapter.render_main_prompt(messages, enable_thinking=cfg.enable_thinking)
    t0 = time.time()

    first_token_event = asyncio.Event()
    first_token_s: float | None = None
    first_token_abs: float | None = None
    first_token_text: str | None = None

    async def on_first(delay_s: float, text: str) -> None:
        nonlocal first_token_s, first_token_abs, first_token_text
        first_token_s = delay_s
        first_token_abs = time.time()
        first_token_text = text
        first_token_event.set()

    main_task = asyncio.create_task(
        client.stream(
            session,
            main_prompt,
            max_tokens=cfg.baseline_max_tokens,
            stop=list(cfg.main_stop or model_adapter.main_stop),
            seed=cfg.seed,
            on_first_token=on_first,
        )
    )

    probe_dispatch_ms = None
    probe_end_ms = None
    probe_wall_s = 0.0
    probe_logprobs = dict(tool_calls.EMPTY_LOGPROBS)
    probe_call = None
    first_token_timeout = False
    spec_task: asyncio.Task[SpeculativeToolExecution] | None = None
    spec_dispatch_ms = None
    spec_end_ms = None
    spec_wall_s = 0.0
    probe_dispatch_after_first_token_ms = None

    try:
        await asyncio.wait_for(first_token_event.wait(), timeout=cfg.first_token_timeout_s)
    except asyncio.TimeoutError:
        first_token_timeout = True

    if not first_token_timeout:
        # Probe prompt rendered with the PROBE model's tokenizer/template, using
        # the observed main CoT prefix.
        probe_prompt = probe_adapter.build_probe_prompt(
            messages,
            enable_thinking=cfg.enable_thinking,
            observed_main_prefix=first_token_text,
        )
        probe_dispatch_abs = time.time()
        probe_dispatch_ms = (probe_dispatch_abs - t0) * 1000.0
        probe_dispatch_after_first_token_ms = (
            (probe_dispatch_abs - first_token_abs) * 1000.0
            if first_token_abs is not None
            else None
        )
        choice, probe_wall_s = await probe_client.complete(
            session,
            probe_prompt,
            max_tokens=cfg.probe_max_tokens,
            stop=list(probe_adapter.probe_stop),
            logprobs=5,
            seed=cfg.seed,
        )
        probe_end_ms = (time.time() - t0) * 1000.0
        probe_logprobs = probe_adapter.extract_probe_logprobs(choice.get("logprobs"))
        probe_call = probe_adapter.parse_probe_tool_call(choice.get("text", ""))

    dispatch_by_confidence = True
    if cfg.gate in ("confidence_strict", "confidence_name_loose"):
        conf = probe_adapter.span_confidence(probe_logprobs, skip_first=True)
        dispatch_by_confidence = conf is not None and conf >= cfg.confidence_threshold
    if probe_call and dispatch_by_confidence:
        spec_dispatch_ms = (time.time() - t0) * 1000.0
        spec_task = asyncio.create_task(asyncio.to_thread(execute_speculative, executor, probe_call))

    main_res = await main_task
    main_end_ms = (time.time() - t0) * 1000.0
    text = main_res["text"]
    main_call = _first_structured_tool_call(main_res) if uses_chat else None
    if main_call is None:
        main_call = model_adapter.parse_main_tool_call(text)
    has_tool = main_call is not None
    decision = gates.decide_gate(
        cfg.gate,
        probe_call,
        main_call,
        probe_logprobs,
        confidence_threshold=cfg.confidence_threshold,
    )

    canonical_dispatch_ms = None
    canonical_end_ms = None
    spec_used = False
    spec_wasted = False
    tool_result = ""
    tool_real_wall_s = 0.0
    tool_artificial_sleep_s = 0.0
    if has_tool:
        if decision.accept_after_main and spec_task is not None:
            spec_execution = await spec_task
            commit_speculative(executor, spec_execution)
            tool_result = spec_execution.text
            spec_wall_s = spec_execution.wall_s
            tool_real_wall_s = spec_execution.real_wall_s
            tool_artificial_sleep_s = spec_execution.artificial_sleep_s
            spec_end_ms = (time.time() - t0) * 1000.0
            spec_used = True
        else:
            if spec_task is not None:
                spec_wasted = True
                if spec_task.done():
                    try:
                        spec_execution = spec_task.result()
                        spec_wall_s = spec_execution.wall_s
                    except Exception:
                        pass
                else:
                    spec_task.cancel()
            canonical_dispatch_ms = (time.time() - t0) * 1000.0
            tool_result, _ = await asyncio.to_thread(executor.execute, main_call)
            tool_real_wall_s = executor.last_real_wall_s
            tool_artificial_sleep_s = executor.last_floor_sleep_s
            canonical_end_ms = (time.time() - t0) * 1000.0
        adapters.append_assistant_tool_messages(messages, text, tool_result)

    if spec_used and spec_dispatch_ms is not None and spec_wall_s > 0:
        real_overlap_ms = min(spec_wall_s * 1000.0, max(0.0, main_end_ms - spec_dispatch_ms))
        spec_actual_end_ms = spec_dispatch_ms + spec_wall_s * 1000.0
    else:
        real_overlap_ms = 0.0
        spec_actual_end_ms = None

    turn_end_ms = (time.time() - t0) * 1000.0
    entry = {
        "turn": turn,
        "fork_point": _fork_point(turn),
        "mode": "spork_cross",
        "gate_variant": cfg.gate,
        "confidence_threshold": cfg.confidence_threshold,
        "main_first_token_ms": (
            main_res["first_token_s"] * 1000.0
            if main_res["first_token_s"] is not None
            else None
        ),
        "main_end_ms": main_end_ms,
        "probe_dispatch_ms": probe_dispatch_ms,
        "probe_dispatch_after_first_token_ms": probe_dispatch_after_first_token_ms,
        "probe_end_ms": probe_end_ms,
        "probe_wall_s": round(probe_wall_s, 3),
        "spec_tool_dispatch_ms": spec_dispatch_ms,
        "spec_tool_end_ms": spec_end_ms,
        "spec_tool_actual_end_ms": spec_actual_end_ms,
        "spec_tool_wall_s": round(spec_wall_s, 3),
        "canonical_tool_dispatch_ms": canonical_dispatch_ms,
        "canonical_tool_end_ms": canonical_end_ms,
        "real_tool_wall_s": round(tool_real_wall_s, 3),
        "floor_sleep_s": round(tool_artificial_sleep_s, 3),
        "turn_end_ms": turn_end_ms,
        "real_overlap_ms": round(real_overlap_ms, 3),
        "first_token_timeout": first_token_timeout,
        "baseline_has_tool_call": has_tool,
        "baseline_tool_name": main_call["name"] if has_tool else None,
        "baseline_tool_args": main_call.get("arguments") if has_tool else None,
        "baseline_text": text,
        "main_usage": main_res.get("usage"),
        "probe_usage": (probe_response or {}).get("usage"),
        "main_reasoning": main_res.get("reasoning", "") if uses_chat else "",
        **model_adapter.token_metrics(text),
        "probe_tool_name": probe_call["name"] if probe_call else None,
        "probe_tool_args": probe_call.get("arguments") if probe_call else None,
        "probe_parse_ok": probe_call is not None,
        "probe_logprobs": probe_logprobs,
        "probe_first_token_top1_prob": probe_adapter.first_token_top1_prob(probe_logprobs),
        "probe_span_min_prob_skip0": decision.confidence,
        "hit_name": decision.name_match,
        "hit_args_exact": decision.args_exact,
        "args_partial_overlap": round(decision.args_partial, 3),
        "gate_match": decision.accept_after_main,
        "spec_dispatched": spec_task is not None,
        "spec_used": spec_used,
        "spec_wasted": spec_wasted,
    }
    if has_tool:
        entry["tool_result_len"] = len(tool_result)
        entry["tool_result_preview"] = model_adapter.preview(tool_result)
    else:
        entry["final_answer"] = text.strip()
    return entry


async def run_spork_replay_turn(
    session: aiohttp.ClientSession,
    client: VllmClient,
    model_adapter: ModelAdapter,
    messages: list[dict],
    turn: int,
    cfg: TurnConfig,
    *,
    baseline_tool_name: str | None,
    baseline_tool_args: dict | None,
    baseline_tool_result: str,
    baseline_tool_wall_ms: float,
) -> dict:
    """Run one replay turn: main + probe dispatch normally, NO tool execution.

    State advances via baseline's recorded tool_result (caller must append to messages).
    T_tool is read from baseline's recorded wall; T_overlap is computed as if the tool
    had been speculatively executed for baseline_tool_wall_ms starting at probe dispatch.
    Produces per-turn metrics for EQ1 validation: α_name, α_args_exact, T_dec, T_tool,
    T_overlap, plus multi-metric span scores for Adaptive gate calibration.
    """
    uses_chat = getattr(model_adapter, "uses_chat_api", False)
    main_prompt = model_adapter.render_main_prompt(messages, enable_thinking=cfg.enable_thinking)
    t0 = time.time()

    first_token_event = asyncio.Event()
    first_token_s: float | None = None
    first_token_abs: float | None = None
    first_token_text: str | None = None

    async def on_first(delay_s: float, text: str) -> None:
        nonlocal first_token_s, first_token_abs, first_token_text
        first_token_s = delay_s
        first_token_abs = time.time()
        first_token_text = text
        first_token_event.set()

    main_task = asyncio.create_task(
        client.stream(
            session,
            main_prompt,
            max_tokens=cfg.baseline_max_tokens,
            stop=list(cfg.main_stop or model_adapter.main_stop),
            seed=cfg.seed,
            on_first_token=on_first,
        )
    )

    probe_dispatch_ms = None
    probe_end_ms = None
    probe_wall_s = 0.0
    probe_logprobs = dict(tool_calls.EMPTY_LOGPROBS)
    probe_call = None
    first_token_timeout = False
    probe_dispatch_after_first_token_ms = None

    try:
        await asyncio.wait_for(first_token_event.wait(), timeout=cfg.first_token_timeout_s)
    except asyncio.TimeoutError:
        first_token_timeout = True

    if not first_token_timeout:
        probe_prompt = model_adapter.build_probe_prompt(
            messages,
            enable_thinking=cfg.enable_thinking,
            observed_main_prefix=first_token_text,
        )
        probe_dispatch_abs = time.time()
        probe_dispatch_ms = (probe_dispatch_abs - t0) * 1000.0
        probe_dispatch_after_first_token_ms = (
            (probe_dispatch_abs - first_token_abs) * 1000.0
            if first_token_abs is not None
            else None
        )
        choice, probe_wall_s = await client.complete(
            session,
            probe_prompt,
            max_tokens=cfg.probe_max_tokens,
            stop=list(model_adapter.probe_stop),
            logprobs=5,
            seed=cfg.seed,
        )
        probe_end_ms = (time.time() - t0) * 1000.0
        probe_logprobs = model_adapter.extract_probe_logprobs(choice.get("logprobs"))
        probe_call = model_adapter.parse_probe_tool_call(choice.get("text", ""))

    main_res = await main_task
    main_end_ms = (time.time() - t0) * 1000.0
    text = main_res["text"]
    main_call = _first_structured_tool_call(main_res) if uses_chat else None
    if main_call is None:
        main_call = model_adapter.parse_main_tool_call(text)

    # Compare against BASELINE's actual tool call (foresight measurement)
    baseline_call = (
        {"name": baseline_tool_name, "arguments": baseline_tool_args or {}}
        if baseline_tool_name else None
    )
    hit_name = bool(
        probe_call and baseline_call and probe_call.get("name") == baseline_call.get("name")
    )
    hit_args_exact = bool(
        hit_name and tool_calls.args_exact_match(
            probe_call.get("arguments"), baseline_call.get("arguments")
        )
    )
    args_partial = (
        tool_calls.args_partial_overlap(
            probe_call.get("arguments"), baseline_call.get("arguments")
        ) if hit_name and probe_call and baseline_call else 0.0
    )
    # Also compare main's re-decoded call to baseline (measures main-thread drift under
    # SPORK-induced concurrent-batch nondeterminism, informational — not used in EQ1).
    main_hit_name = bool(
        main_call and baseline_call and main_call.get("name") == baseline_call.get("name")
    )
    main_hit_args_exact = bool(
        main_hit_name and tool_calls.args_exact_match(
            main_call.get("arguments"), baseline_call.get("arguments")
        )
    )

    # Multi-metric span scores for Adaptive gate calibration (PI decision: log all 4)
    span_scores = tool_calls.span_scores(probe_logprobs, span_len=5, skip_first=True)

    # Simulated speculative overlap: if probe dispatched successfully, the speculative
    # tool would have started at probe_dispatch and finished at probe_dispatch +
    # baseline_tool_wall_ms. Overlap with main decode window [first_token, main_end]:
    t_overlap_ms = 0.0
    spec_tool_simulated_end_ms = None
    if probe_dispatch_ms is not None and baseline_tool_wall_ms > 0:
        spec_tool_simulated_end_ms = probe_dispatch_ms + baseline_tool_wall_ms
        # Overlap = min(spec_end, main_end) - max(spec_start, main_first_token)
        first_token_ms = (first_token_s or 0.0) * 1000.0
        overlap_start = max(probe_dispatch_ms, first_token_ms)
        overlap_end = min(spec_tool_simulated_end_ms, main_end_ms)
        t_overlap_ms = max(0.0, overlap_end - overlap_start)

    # Compute canonical per-turn times for EQ1
    t_dec_ms = main_end_ms
    t_tool_ms = baseline_tool_wall_ms
    t_base_ms = t_dec_ms + t_tool_ms  # what baseline would spend this turn
    # Replay per-turn: if spec accepted + tool ready before main_end → tool "free" (overlapped);
    # otherwise serial tool after main. We report the IDEAL T_spork under the acceptance rule.
    # Decision rule here is strict args_exact (the canonical EQ1 α).
    alpha_accepted = 1.0 if hit_args_exact else 0.0
    t_spork_ideal_ms = t_base_ms - alpha_accepted * t_overlap_ms  # no T_oh in replay
    t_oh_ms = 0.0  # by construction, replay has no extra overhead beyond probe decode wall

    entry: dict[str, Any] = {
        "turn": turn,
        "mode": "spork_replay",
        "fork_point": _fork_point(turn),
        "gate_variant": cfg.gate,
        "confidence_threshold": cfg.confidence_threshold,
        "main_first_token_ms": (first_token_s * 1000.0) if first_token_s is not None else None,
        "main_end_ms": main_end_ms,
        "probe_dispatch_ms": probe_dispatch_ms,
        "probe_dispatch_after_first_token_ms": probe_dispatch_after_first_token_ms,
        "probe_end_ms": probe_end_ms,
        "probe_wall_s": round(probe_wall_s, 3),
        "first_token_timeout": first_token_timeout,
        "turn_end_ms": main_end_ms,  # replay: no tool wall added
        # Baseline ground truth (for RQ1 foresight)
        "baseline_has_tool_call": baseline_tool_name is not None,
        "baseline_tool_name": baseline_tool_name,
        "baseline_tool_args": baseline_tool_args,
        "baseline_tool_wall_ms": baseline_tool_wall_ms,
        # Probe prediction
        "probe_tool_name": probe_call["name"] if probe_call else None,
        "probe_tool_args": probe_call.get("arguments") if probe_call else None,
        "probe_parse_ok": probe_call is not None,
        "probe_logprobs": probe_logprobs,
        "probe_first_token_top1_prob": model_adapter.first_token_top1_prob(probe_logprobs),
        "probe_span_scores": span_scores,  # multi-metric (PI Q1: log all 4)
        # Accuracy against baseline
        "hit_name": hit_name,
        "hit_args_exact": hit_args_exact,
        "args_partial_overlap": round(args_partial, 3),
        # Main re-decode vs baseline (trajectory-drift under concurrent batch)
        "main_redecoded_tool_name": main_call["name"] if main_call else None,
        "main_hit_name": main_hit_name,
        "main_hit_args_exact": main_hit_args_exact,
        # EQ1 inputs
        "T_dec_ms": round(t_dec_ms, 3),
        "T_tool_ms": round(t_tool_ms, 3),
        "T_base_ms": round(t_base_ms, 3),
        "T_overlap_ms": round(t_overlap_ms, 3),
        "T_spork_ideal_ms": round(t_spork_ideal_ms, 3),
        "T_oh_ms": round(t_oh_ms, 3),
        "alpha_accepted": alpha_accepted,
        # Main text for debugging
        "main_text_len": len(text),
        **model_adapter.token_metrics(text),
    }
    return entry


async def run_spork_adaptive_replay_turn(
    session: aiohttp.ClientSession,
    client: VllmClient,
    model_adapter: ModelAdapter,
    messages: list[dict],
    turn: int,
    cfg: TurnConfig,
    *,
    baseline_tool_name: str | None,
    baseline_tool_args: dict | None,
    baseline_tool_result: str,
    baseline_tool_wall_ms: float,
    adaptive_cfg,   # AdaptiveConfig (imported by caller to avoid circular import)
) -> dict:
    """Adaptive-replay turn: live main + N-token-cadence probe scheduler, NO tool execution.

    Per-turn metrics record the full retry history (including aborts) plus commit
    position, T_dec, T_tool_baseline, T_overlap (simulated against committed probe),
    and α vs baseline's actual tool call. Used for EQ1 validation + RQ1 foresight
    at mid-CoT fork positions.
    """
    from .adaptive_scheduler import run_adaptive_probe_schedule

    outcome = await run_adaptive_probe_schedule(
        session, client, model_adapter, messages, adaptive_cfg,
    )

    baseline_call = (
        {"name": baseline_tool_name, "arguments": baseline_tool_args or {}}
        if baseline_tool_name else None
    )
    probe_call = outcome.committed_call
    hit_name = bool(
        probe_call and baseline_call and probe_call.get("name") == baseline_call.get("name")
    )
    hit_args_exact = bool(
        hit_name and tool_calls.args_exact_match(
            probe_call.get("arguments"), baseline_call.get("arguments")
        )
    )
    args_partial = (
        tool_calls.args_partial_overlap(
            probe_call.get("arguments"), baseline_call.get("arguments")
        ) if hit_name and probe_call and baseline_call else 0.0
    )

    # Also re-parse main's output (main is running live in replay; its text may drift
    # from baseline under concurrent-scheduling nondeterminism).
    main_call = model_adapter.parse_main_tool_call(outcome.main_text)
    main_hit_name = bool(
        main_call and baseline_call and main_call.get("name") == baseline_call.get("name")
    )
    main_hit_args_exact = bool(
        main_hit_name and tool_calls.args_exact_match(
            main_call.get("arguments"), baseline_call.get("arguments")
        )
    )

    # Simulated overlap: if a probe committed, the speculative tool would have
    # dispatched at that probe's phase2_end_ms and run for baseline_tool_wall_ms.
    # Overlap with main decode window [first_token, main_end].
    t_overlap_ms = 0.0
    spec_tool_simulated_end_ms = None
    first_token_ms = outcome.first_token_ms or 0.0
    committed = next((r for r in outcome.retries if r.retry_idx == outcome.committed_retry_idx), None) \
        if outcome.committed_retry_idx is not None else None
    if committed and baseline_tool_wall_ms > 0:
        spec_dispatch_ms = committed.probe_phase2_end_ms or committed.probe_phase1_end_ms
        spec_tool_simulated_end_ms = spec_dispatch_ms + baseline_tool_wall_ms
        overlap_start = max(spec_dispatch_ms, first_token_ms)
        overlap_end = min(spec_tool_simulated_end_ms, outcome.main_end_ms)
        t_overlap_ms = max(0.0, overlap_end - overlap_start)

    t_dec_ms = outcome.main_end_ms
    t_tool_ms = baseline_tool_wall_ms
    t_base_ms = t_dec_ms + t_tool_ms
    alpha_accepted = 1.0 if hit_args_exact else 0.0

    # T_oh = sum of probe phase1 walls for aborted retries (wasted decode)
    aborted_phase1_wall = sum(
        (r.probe_phase1_end_ms - r.probe_dispatch_ms) for r in outcome.retries
        if r.decision == "abort"
    )
    # Plus committed probe's phase1+phase2 wall (this overlaps main so it's NOT pure overhead;
    # but track separately).
    committed_probe_wall = (
        (committed.probe_phase2_end_ms or committed.probe_phase1_end_ms) - committed.probe_dispatch_ms
        if committed else 0.0
    )

    t_spork_ideal_ms = t_base_ms - alpha_accepted * t_overlap_ms  # EQ1 in replay = no T_oh
    t_oh_ms = aborted_phase1_wall  # aborted retries are pure wasted decode

    retries_list = []
    for r in outcome.retries:
        retries_list.append({
            "retry_idx": r.retry_idx,
            "main_tokens_at_dispatch": r.main_tokens_at_dispatch,
            "probe_dispatch_ms": r.probe_dispatch_ms,
            "probe_phase1_end_ms": r.probe_phase1_end_ms,
            "probe_phase2_end_ms": r.probe_phase2_end_ms,
            "phase1_span_scores": r.phase1_span_scores,
            "phase1_metric_score": r.phase1_metric_score,
            "decision": r.decision,
            "probe_phase1_text": r.probe_phase1_text[:200],
            "probe_phase2_text_preview": r.probe_phase2_text[:200] if r.probe_phase2_text else "",
            "probe_parsed_call": r.probe_parsed_call,
            # Keep raw logprobs too, for post-hoc multi-threshold sweep
            "phase1_logprobs": r.phase1_logprobs,
        })

    entry: dict[str, Any] = {
        "turn": turn,
        "mode": "spork_adaptive_replay",
        "fork_point": _fork_point(turn),
        "gate_variant": cfg.gate,
        "main_first_token_ms": outcome.first_token_ms,
        "main_end_ms": outcome.main_end_ms,
        "turn_end_ms": outcome.main_end_ms,
        "first_token_timeout": outcome.first_token_timeout,
        # Baseline ground truth
        "baseline_has_tool_call": baseline_tool_name is not None,
        "baseline_tool_name": baseline_tool_name,
        "baseline_tool_args": baseline_tool_args,
        "baseline_tool_wall_ms": baseline_tool_wall_ms,
        # Committed probe (Adaptive's answer)
        "probe_tool_name": probe_call["name"] if probe_call else None,
        "probe_tool_args": probe_call.get("arguments") if probe_call else None,
        "probe_parse_ok": probe_call is not None,
        "committed_retry_idx": outcome.committed_retry_idx,
        "committed_main_token_count": outcome.committed_main_token_count,
        "total_retries_attempted": len(outcome.retries),
        # Accuracy vs baseline
        "hit_name": hit_name,
        "hit_args_exact": hit_args_exact,
        "args_partial_overlap": round(args_partial, 3),
        # Main-redecode comparison
        "main_redecoded_tool_name": main_call["name"] if main_call else None,
        "main_hit_name": main_hit_name,
        "main_hit_args_exact": main_hit_args_exact,
        # EQ1 inputs
        "T_dec_ms": round(t_dec_ms, 3),
        "T_tool_ms": round(t_tool_ms, 3),
        "T_base_ms": round(t_base_ms, 3),
        "T_overlap_ms": round(t_overlap_ms, 3),
        "T_spork_ideal_ms": round(t_spork_ideal_ms, 3),
        "T_oh_ms": round(t_oh_ms, 3),
        "alpha_accepted": alpha_accepted,
        # Retry detail
        "retries": retries_list,
        # Main text for debugging
        "main_text_len": len(outcome.main_text),
        **model_adapter.token_metrics(outcome.main_text),
    }
    return entry


@dataclass
class RetryTurnConfig:
    max_retries: int = 5
    confidence_threshold: float = 0.90
    retry_token_step: int = 50
    snap_to_sentence: bool = True
    probe_max_tokens: int = 100
    probe_timeout_s: float = 0.0
    baseline_max_tokens: int = 3072
    context_window_tokens: int = 32768
    enable_thinking: bool = True
    first_token_timeout_s: float = 10.0
    seed: int = 42
    main_stop: tuple[str, ...] = ("</tool_call>",)
    theta_decay_per_retry: float = 0.0
    min_tokens_first_probe: int = 0
    hybrid_name_loose_first: bool = False
    hybrid_name_threshold: float = 0.60
    continue_after_dispatch: bool = False
    d3_set: Any = None
    d3_clear: Any = None
    retain_candidate_pool: bool = False
    dispatch_all_parsed: bool = False
    probe_token_targets: tuple[int, ...] = ()


def _portfolio_calls_exact(candidate: dict | None, main_call: dict | None) -> bool:
    return bool(
        candidate
        and main_call
        and candidate.get("name") == main_call.get("name")
        and tool_calls.args_exact_match(
            candidate.get("arguments"), main_call.get("arguments")
        )
    )


def _select_portfolio_candidate(
    candidates: list[dict[str, Any]], main_call: dict | None
) -> dict[str, Any] | None:
    """Select the earliest-dispatched strict-exact portfolio candidate."""
    matches = [
        candidate
        for candidate in candidates
        if _portfolio_calls_exact(candidate.get("call"), main_call)
    ]
    return min(matches, key=lambda candidate: candidate["dispatch_ms"]) if matches else None


def _resolve_retry_main_stop(model_adapter: ModelAdapter, cfg: RetryTurnConfig) -> list[str]:
    """Resolve the D2 main decode stop strings.

    D2 retry must use the adapter's main stop for chat-model adapters such as
    DeepSeek-V4; the config default is the Qwen XML closer and is intentionally
    not a fallback for V4.
    """
    return list(model_adapter.main_stop)


async def _capture_decode_completion(coro, completion: dict, started_at: float):
    """Stamp the stream's return inside its task, before any probe join."""
    result = await coro
    completion["end_ms"] = (time.time() - started_at) * 1000.0
    return result


async def run_spork_retry_turn(
    session: aiohttp.ClientSession,
    client: VllmClient,
    model_adapter: ModelAdapter,
    messages: list[dict],
    executor: ToolExecutor,
    turn: int,
    cfg: RetryTurnConfig,
) -> dict:
    """D2 multi-probe retry turn: probe up to N times with growing observed_prefix.

    Each probe fires after main has decoded more CoT tokens. If any probe's
    span_min_prob >= theta, dispatch spec tool. Final gate is always strict
    (name + args_exact). If gate rejects, fallback to serial.
    """
    import threading

    uses_chat = getattr(model_adapter, "uses_chat_api", False)
    main_prompt = None
    if not uses_chat:
        main_prompt = model_adapter.render_main_prompt(messages, enable_thinking=cfg.enable_thinking)
    main_max_tokens = cfg.baseline_max_tokens
    t0 = time.time()
    main_completion: dict[str, float] = {}

    # Streaming state: accumulate main's decoded text via on_token callback
    main_text_lock = threading.Lock()
    main_accumulated = {"text": "", "token_count": 0}
    first_token_event = asyncio.Event()
    first_token_s: float | None = None
    first_token_abs: float | None = None

    async def on_first(delay_s: float, text: str) -> None:
        nonlocal first_token_s, first_token_abs
        first_token_s = delay_s
        first_token_abs = time.time()
        first_token_event.set()

    async def on_token(delta_text: str, token_idx: int) -> None:
        with main_text_lock:
            main_accumulated["text"] += delta_text
            main_accumulated["token_count"] = token_idx

    if uses_chat:
        main_max_tokens = await _bounded_chat_max_tokens(
            session, client, model_adapter, messages,
            cfg.baseline_max_tokens, cfg.context_window_tokens,
        )
        # V4 chat-API main decode: render server-side from messages, stop at the
        # DSML closer; DSML body streams as content text (parsed below). on_token
        # cadence drives the multi-probe retry schedule exactly as in the Qwen path.
        main_task = asyncio.create_task(_capture_decode_completion(
            client.stream_chat(
                session,
                messages,
                max_tokens=main_max_tokens,
                stop=_resolve_retry_main_stop(model_adapter, cfg),
                seed=cfg.seed,
                tools=getattr(model_adapter, "tools", None) or None,
                on_first_token=on_first,
                on_token=on_token,
            ), main_completion, t0
        ))
    else:
        main_task = asyncio.create_task(_capture_decode_completion(
            client.stream(
                session,
                main_prompt,
                max_tokens=cfg.baseline_max_tokens,
                stop=_resolve_retry_main_stop(model_adapter, cfg),
                seed=cfg.seed,
                on_first_token=on_first,
                on_token=on_token,
            ), main_completion, t0
        ))

    # Retry loop state
    probes: list[dict] = []
    spec_task: asyncio.Task[SpeculativeToolExecution] | None = None
    spec_dispatch_ms: float | None = None
    committed_probe_call: dict | None = None
    committed_retry_idx: int | None = None
    superseded_specs = 0
    first_token_timeout = False
    d3_set_count = 0
    portfolio_candidates: list[dict[str, Any]] = []
    selected_portfolio_candidate: dict[str, Any] | None = None

    try:
        await asyncio.wait_for(first_token_event.wait(), timeout=cfg.first_token_timeout_s)
    except asyncio.TimeoutError:
        first_token_timeout = True

    def _at_sentence_boundary(text: str) -> bool:
        """Check if text ends at a coherent thought unit."""
        if not text:
            return False
        return text.endswith("\n") or text.endswith(". ") or text.endswith(".\n")

    if not first_token_timeout:
        last_probe_tokens = 0
        # For late-start: wait for min tokens before even starting the probe loop
        if cfg.min_tokens_first_probe > 0:
            deadline = time.time() + 12.0
            while time.time() < deadline:
                if main_task.done():
                    break
                with main_text_lock:
                    if main_accumulated["token_count"] >= cfg.min_tokens_first_probe:
                        break
                await asyncio.sleep(0.03)

        # Hybrid state: track if we already dispatched on name-only
        hybrid_name_dispatched = False

        for retry_idx in range(cfg.max_retries):
            # Retry 0: fire after first token (or after min_tokens_first_probe)
            # Retry N>0: wait for (last_probe_tokens + step) tokens AND sentence boundary
            if retry_idx > 0:
                target_tokens = (
                    cfg.probe_token_targets[retry_idx]
                    if retry_idx < len(cfg.probe_token_targets)
                    else last_probe_tokens + cfg.retry_token_step
                )
                deadline = time.time() + 10.0
                reached_target = False
                while time.time() < deadline:
                    if main_task.done():
                        break
                    with main_text_lock:
                        current_tokens = main_accumulated["token_count"]
                        current_text = main_accumulated["text"]
                    if current_tokens >= target_tokens:
                        if (
                            cfg.probe_token_targets
                            or not cfg.snap_to_sentence
                            or _at_sentence_boundary(current_text)
                        ):
                            reached_target = True
                            break
                        if current_tokens >= target_tokens + 30:
                            reached_target = True
                            break
                    await asyncio.sleep(0.03)
                if not reached_target and main_task.done():
                    break

            if main_task.done():
                break

            with main_text_lock:
                observed_prefix = main_accumulated["text"]
                tokens_at_probe = main_accumulated["token_count"]
            last_probe_tokens = tokens_at_probe

            probe_dispatch_abs = time.time()
            probe_dispatch_ms = (probe_dispatch_abs - t0) * 1000.0

            probe_error = None
            probe_logprobs = {}
            probe_call = None
            probe_wall_s = 0.0
            probe_response = None
            try:
                if uses_chat:
                    # V4 probe: prefill assistant message ending in the DSML opener,
                    # continuing the observed CoT (observed_prefix == cot_so_far).
                    probe_message, probe_prefix = _build_chat_probe_message(
                        model_adapter,
                        observed_prefix,
                        enable_thinking=cfg.enable_thinking,
                    )
                    probe_messages = list(messages) + [probe_message]
                    probe_max_tokens = await _bounded_chat_max_tokens(
                        session, client, model_adapter, probe_messages,
                        cfg.probe_max_tokens, cfg.context_window_tokens,
                    )
                    probe_coro = client.complete_chat(
                        session,
                        probe_messages,
                        max_tokens=probe_max_tokens,
                        temperature=0.0,
                        stop=list(model_adapter.probe_stop),
                        seed=cfg.seed,
                        logprobs=True,
                        top_logprobs=5,
                        tools=getattr(model_adapter, "tools", None) or None,
                        continue_final_message=True,
                        add_generation_prompt=False,
                        extra=model_adapter.build_probe_constraint_extra()
                        if hasattr(model_adapter, "build_probe_constraint_extra")
                        else None,
                    )
                    if cfg.probe_timeout_s and cfg.probe_timeout_s > 0:
                        choice, probe_wall_s = await asyncio.wait_for(
                            probe_coro, timeout=cfg.probe_timeout_s
                        )
                    else:
                        choice, probe_wall_s = await probe_coro
                    probe_response = choice
                    flat_lp = chat_logprobs_to_flat(choice)
                    probe_logprobs = model_adapter.extract_probe_logprobs(flat_lp)
                    probe_call, probe_source = _probe_call_from_chat_choice(
                        choice, model_adapter, probe_prefix,
                    )
                else:
                    probe_prompt = model_adapter.build_probe_prompt(
                        messages,
                        enable_thinking=cfg.enable_thinking,
                        observed_main_prefix=observed_prefix,
                    )
                    probe_coro = client.complete(
                        session,
                        probe_prompt,
                        max_tokens=cfg.probe_max_tokens,
                        stop=list(model_adapter.probe_stop),
                        logprobs=5,
                        seed=cfg.seed,
                    )
                    if cfg.probe_timeout_s and cfg.probe_timeout_s > 0:
                        choice, probe_wall_s = await asyncio.wait_for(
                            probe_coro, timeout=cfg.probe_timeout_s
                        )
                    else:
                        choice, probe_wall_s = await probe_coro
                    probe_response = choice
                    probe_logprobs = model_adapter.extract_probe_logprobs(choice.get("logprobs"))
                    probe_call = model_adapter.parse_probe_tool_call(choice.get("text", ""))
                if hasattr(model_adapter, "normalize_call"):
                    probe_call = model_adapter.normalize_call(probe_call)
            except asyncio.TimeoutError:
                probe_wall_s = time.time() - probe_dispatch_abs
                probe_error = f"timeout_after_{cfg.probe_timeout_s:.1f}s"
            except Exception as exc:
                probe_wall_s = time.time() - probe_dispatch_abs
                probe_error = f"{type(exc).__name__}: {exc}"
            probe_end_ms = (time.time() - t0) * 1000.0
            confidence = model_adapter.span_confidence(probe_logprobs, skip_first=True)

            # Compute effective threshold for this retry (adaptive decay)
            effective_theta = cfg.confidence_threshold - (retry_idx * cfg.theta_decay_per_retry)
            effective_theta = max(effective_theta, 0.3)

            probe_entry = {
                "retry_idx": retry_idx,
                "tokens_at_dispatch": tokens_at_probe,
                "observed_prefix_len": len(observed_prefix),
                "probe_dispatch_ms": probe_dispatch_ms,
                "probe_end_ms": probe_end_ms,
                "probe_wall_s": round(probe_wall_s, 3),
                "confidence": confidence,
                "effective_theta": round(effective_theta, 3),
                "probe_tool_name": probe_call["name"] if probe_call else None,
                "probe_tool_args": probe_call.get("arguments") if probe_call else None,
                "probe_parse_ok": probe_call is not None,
                "probe_logprobs": probe_logprobs,
                "probe_error": probe_error,
                "probe_response_source": probe_source if uses_chat and not probe_error else None,
                "probe_response": probe_response,
                "usage": (probe_response or {}).get("usage"),
                "dispatched": False,
            }
            probes.append(probe_entry)

            # D3-in-HTTP: register this probe's predicted body as draft tokens in
            # the sidecar so the main decode can accept them at <tool_call>. Update
            # to the latest parseable probe (mirrors engine-mode set_probe_tokens
            # refresh). Acceptance metrics below are unaffected by this.
            if cfg.d3_set is not None and probe_call is not None:
                try:
                    d3_info = await cfg.d3_set(
                        probe_call,
                        messages if uses_chat else main_prompt,
                    )
                    if d3_info is not None:
                        probe_entry["d3_set"] = d3_info
                        d3_set_count += 1
                except Exception as _e:
                    probe_entry["d3_set_error"] = str(_e)

            # Dispatch decision
            if cfg.retain_candidate_pool:
                should_dispatch = bool(
                    probe_call
                    and (
                        cfg.dispatch_all_parsed
                        or (
                            confidence is not None
                            and confidence >= effective_theta
                        )
                    )
                )
                if should_dispatch:
                    duplicate = next(
                        (
                            candidate
                            for candidate in portfolio_candidates
                            if _portfolio_calls_exact(candidate["call"], probe_call)
                        ),
                        None,
                    )
                    if duplicate is not None:
                        probe_entry["candidate_id"] = duplicate["candidate_id"]
                        probe_entry["duplicate_candidate"] = True
                    else:
                        candidate_dispatch_ms = (time.time() - t0) * 1000.0
                        candidate = {
                            "candidate_id": len(portfolio_candidates),
                            "retry_idx": retry_idx,
                            "call": probe_call,
                            "dispatch_ms": candidate_dispatch_ms,
                            "task": asyncio.create_task(
                                asyncio.to_thread(
                                    execute_speculative, executor, probe_call
                                )
                            ),
                            "execution": None,
                            "worker_error": None,
                        }
                        portfolio_candidates.append(candidate)
                        probe_entry["candidate_id"] = candidate["candidate_id"]
                        probe_entry["dispatched"] = True
                continue

            if probe_call and confidence is not None and confidence >= effective_theta:
                # Hybrid mode: first probe dispatches on name-only (loose) at lower theta
                if cfg.hybrid_name_loose_first and retry_idx == 0:
                    if confidence >= cfg.hybrid_name_threshold:
                        spec_dispatch_ms = (time.time() - t0) * 1000.0
                        spec_task = asyncio.create_task(
                            asyncio.to_thread(execute_speculative, executor, probe_call)
                        )
                        committed_probe_call = probe_call
                        committed_retry_idx = retry_idx
                        probe_entry["dispatched"] = True
                        hybrid_name_dispatched = True
                        # Don't break — continue probing for args upgrade
                    continue
                # Standard: dispatch on confidence threshold
                if spec_task is None:
                    spec_dispatch_ms = (time.time() - t0) * 1000.0
                    spec_task = asyncio.create_task(
                        asyncio.to_thread(execute_speculative, executor, probe_call)
                    )
                    committed_probe_call = probe_call
                    committed_retry_idx = retry_idx
                    probe_entry["dispatched"] = True
                    if not cfg.continue_after_dispatch:
                        break
                    continue

                if cfg.continue_after_dispatch:
                    same_candidate = (
                        committed_probe_call is not None
                        and committed_probe_call.get("name") == probe_call.get("name")
                        and tool_calls.args_exact_match(
                            committed_probe_call.get("arguments"),
                            probe_call.get("arguments"),
                        )
                    )
                    if same_candidate:
                        probe_entry["kept_existing_dispatch"] = True
                        continue
                    if not spec_task.done():
                        spec_task.cancel()
                    superseded_specs += 1
                    spec_dispatch_ms = (time.time() - t0) * 1000.0
                    spec_task = asyncio.create_task(
                        asyncio.to_thread(execute_speculative, executor, probe_call)
                    )
                    committed_probe_call = probe_call
                    committed_retry_idx = retry_idx
                    probe_entry["dispatched"] = True
                    probe_entry["superseded_previous"] = True
                    continue

                break

            # Hybrid: if we already dispatched on name-only and a later probe
            # gives high-confidence with (potentially) better args, upgrade
            if cfg.hybrid_name_loose_first and hybrid_name_dispatched and probe_call:
                if confidence is not None and confidence >= cfg.confidence_threshold:
                    # Cancel old spec, dispatch new with better args
                    if spec_task is not None and not spec_task.done():
                        spec_task.cancel()
                    spec_dispatch_ms = (time.time() - t0) * 1000.0
                    spec_task = asyncio.create_task(
                        asyncio.to_thread(execute_speculative, executor, probe_call)
                    )
                    committed_probe_call = probe_call
                    committed_retry_idx = retry_idx
                    probe_entry["dispatched"] = True
                    break

    # main_end_ms remains the reconciliation barrier for the fixed-trace
    # counterfactual. The stream completion timestamp excludes probe-join wait.
    main_res = await main_task
    main_end_ms = (time.time() - t0) * 1000.0
    main_decode_complete_ms = main_completion["end_ms"]
    text = main_res["text"]
    # D3-in-HTTP: clear sidecar draft-token state so the NEXT request/turn starts
    # clean (state hygiene — verified E2). Safe no-op if D3 disabled.
    d3_status = None
    if cfg.d3_clear is not None:
        try:
            d3_status = await cfg.d3_clear()
        except Exception:
            d3_status = None
    main_call = _first_structured_tool_call(main_res) if uses_chat else None
    if main_call is None:
        main_call = model_adapter.parse_main_tool_call(text)
    if hasattr(model_adapter, "normalize_call"):
        main_call = model_adapter.normalize_call(main_call)
    has_tool = main_call is not None

    # Strict gate: name + args_exact required
    gate_match = False
    spec_used = False
    spec_wasted = False
    spec_end_ms: float | None = None
    spec_wall_s = 0.0
    tool_result = ""
    tool_real_wall_s = 0.0
    real_overlap_ms = 0.0

    if has_tool and cfg.retain_candidate_pool:
        selected_portfolio_candidate = _select_portfolio_candidate(
            portfolio_candidates, main_call
        )
        if selected_portfolio_candidate is not None:
            spec_task = selected_portfolio_candidate["task"]
            spec_dispatch_ms = selected_portfolio_candidate["dispatch_ms"]
            committed_probe_call = selected_portfolio_candidate["call"]
            committed_retry_idx = selected_portfolio_candidate["retry_idx"]
            gate_match = True
            try:
                spec_execution = await spec_task
                selected_portfolio_candidate["execution"] = spec_execution
                commit_speculative(executor, spec_execution)
                tool_result = spec_execution.text
                spec_wall_s = spec_execution.wall_s
                tool_real_wall_s = spec_execution.real_wall_s
                spec_end_ms = (time.time() - t0) * 1000.0
                spec_used = True
                raw_trace = spec_execution.trace or {}
                worker_start_s = raw_trace.get("worker_start_s")
                worker_end_s = raw_trace.get("worker_end_s")
                if worker_start_s is not None and worker_end_s is not None:
                    worker_start_ms = (worker_start_s - t0) * 1000.0
                    worker_duration_ms = max(
                        0.0, (worker_end_s - worker_start_s) * 1000.0
                    )
                    real_overlap_ms = min(
                        worker_duration_ms,
                        max(0.0, main_end_ms - worker_start_ms),
                    )
            except Exception as exc:  # noqa: BLE001
                selected_portfolio_candidate["worker_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
        spec_wasted = any(
            candidate is not selected_portfolio_candidate
            for candidate in portfolio_candidates
        )
    elif cfg.retain_candidate_pool:
        # A probe can false-positive on a final-answer turn. Its isolated work is
        # discarded just like any non-selected candidate and must be accounted as
        # wasted, while the canonical world remains untouched.
        spec_wasted = bool(portfolio_candidates)
    elif has_tool and spec_task is not None and committed_probe_call is not None:
        name_match = committed_probe_call.get("name") == main_call.get("name")
        args_match = tool_calls.args_exact_match(
            committed_probe_call.get("arguments"), main_call.get("arguments")
        )
        gate_match = name_match and args_match

        if gate_match:
            spec_execution = await spec_task
            commit_speculative(executor, spec_execution)
            tool_result = spec_execution.text
            spec_wall_s = spec_execution.wall_s
            tool_real_wall_s = spec_execution.real_wall_s
            spec_end_ms = (time.time() - t0) * 1000.0
            spec_used = True
            if spec_dispatch_ms is not None:
                real_overlap_ms = min(spec_wall_s * 1000.0, max(0.0, main_end_ms - spec_dispatch_ms))
        else:
            spec_wasted = True
            if spec_task.done():
                try:
                    _ = spec_task.result()
                except Exception:
                    pass
            else:
                spec_task.cancel()

    # If no spec used, execute tool serially
    canonical_dispatch_ms: float | None = None
    canonical_end_ms: float | None = None
    if has_tool and not spec_used:
        canonical_dispatch_ms = (time.time() - t0) * 1000.0
        tool_result, _ = await asyncio.to_thread(executor.execute, main_call)
        tool_real_wall_s = executor.last_real_wall_s
        canonical_end_ms = (time.time() - t0) * 1000.0

    if has_tool:
        _append_tool_messages(
            messages,
            text,
            tool_result,
            tool_calls_structured=main_res.get("tool_calls") if uses_chat else None,
            reasoning=main_res.get("reasoning") if uses_chat else None,
            reasoning_field=getattr(model_adapter, "reasoning_message_field", "reasoning"),
        )

    if cfg.retain_candidate_pool:
        for candidate in portfolio_candidates:
            if candidate is selected_portfolio_candidate:
                continue
            task = candidate["task"]
            if task.done():
                try:
                    candidate["execution"] = task.result()
                except Exception as exc:  # noqa: BLE001
                    candidate["worker_error"] = f"{type(exc).__name__}: {exc}"
            else:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    turn_end_ms = (time.time() - t0) * 1000.0

    representative_probe = None
    if probes:
        if selected_portfolio_candidate is not None:
            representative_probe = next(
                (
                    probe
                    for probe in probes
                    if probe.get("retry_idx")
                    == selected_portfolio_candidate["retry_idx"]
                ),
                probes[-1],
            )
        else:
            representative_probe = probes[-1]
    representative_call = (
        {
            "name": representative_probe.get("probe_tool_name"),
            "arguments": representative_probe.get("probe_tool_args"),
        }
        if representative_probe and representative_probe.get("probe_parse_ok")
        else committed_probe_call
    )
    representative_logprobs = (
        representative_probe.get("probe_logprobs") or {}
        if representative_probe
        else {}
    )

    # Compute strict portfolio accuracy against the main call. Name-only hits are
    # diagnostic; they never authorize a state commit.
    if cfg.retain_candidate_pool:
        hit_name = bool(
            main_call
            and any(
                candidate["call"].get("name") == main_call.get("name")
                for candidate in portfolio_candidates
            )
        )
        hit_args_exact = selected_portfolio_candidate is not None
    else:
        hit_name = bool(committed_probe_call and main_call and committed_probe_call.get("name") == main_call.get("name"))
        hit_args_exact = bool(hit_name and tool_calls.args_exact_match(
            committed_probe_call.get("arguments") if committed_probe_call else None,
            main_call.get("arguments") if main_call else None,
        ))
    representative_decision = gates.decide_gate(
        "args_exact_strict",
        representative_call,
        main_call,
        representative_logprobs,
        confidence_threshold=cfg.confidence_threshold,
    )
    portfolio_trace = [
        {
            "candidate_id": candidate["candidate_id"],
            "retry_idx": candidate["retry_idx"],
            "call": candidate["call"],
            "dispatch_ms": round(candidate["dispatch_ms"], 3),
            "selected": candidate is selected_portfolio_candidate,
            "exact_match": _portfolio_calls_exact(candidate["call"], main_call),
            "worker_error": candidate.get("worker_error"),
            "execution_trace": (
                candidate["execution"].trace if candidate.get("execution") is not None else None
            ),
        }
        for candidate in portfolio_candidates
    ]

    entry = {
        "turn": turn,
        "fork_point": _fork_point(turn),
        "mode": "spork_portfolio" if cfg.retain_candidate_pool else "spork_retry",
        "turn_started_at_s": t0,
        "gate_variant": "args_exact_strict",
        "max_retries": cfg.max_retries,
        "confidence_threshold": cfg.confidence_threshold,
        "continue_after_dispatch": cfg.continue_after_dispatch,
        "retain_candidate_pool": cfg.retain_candidate_pool,
        "probe_token_targets": list(cfg.probe_token_targets),
        "main_max_tokens": main_max_tokens,
        "main_first_token_ms": (first_token_s * 1000.0) if first_token_s is not None else None,
        "main_end_ms": main_end_ms,
        "main_decode_complete_ms": main_decode_complete_ms,
        "reconciliation_ready_ms": main_end_ms,
        "probe_join_wait_ms": max(0.0, main_end_ms - main_decode_complete_ms),
        "counterfactual_tool_serialization_boundary": "reconciliation_ready_after_probe_join",
        "turn_end_ms": turn_end_ms,
        "first_token_timeout": first_token_timeout,
        "baseline_has_tool_call": has_tool,
        "baseline_tool_name": main_call["name"] if has_tool else None,
        "baseline_tool_args": main_call.get("arguments") if has_tool else None,
        "baseline_text": text,
        "main_usage": main_res.get("usage"),
        "main_reasoning": main_res.get("reasoning", "") if uses_chat else "",
        "main_structured_tool_call_count": len(main_res.get("tool_calls") or []) if uses_chat else 0,
        "main_structured_tool_calls": main_res.get("tool_calls") if uses_chat else None,
        **model_adapter.token_metrics(text),
        # Probe/retry info
        "num_probes": len(probes),
        "committed_retry_idx": committed_retry_idx,
        "committed_probe_name": committed_probe_call["name"] if committed_probe_call else None,
        "committed_probe_args": committed_probe_call.get("arguments") if committed_probe_call else None,
        "selected_candidate_id": (
            selected_portfolio_candidate["candidate_id"]
            if selected_portfolio_candidate is not None
            else None
        ),
        "portfolio_distinct_candidates": len(portfolio_candidates),
        "portfolio_candidates": portfolio_trace,
        # Gate decision
        "hit_name": hit_name,
        "hit_args_exact": hit_args_exact,
        "gate_match": gate_match,
        "spec_dispatched": bool(portfolio_candidates) if cfg.retain_candidate_pool else spec_task is not None,
        "spec_used": spec_used,
        "spec_wasted": spec_wasted,
        "superseded_specs": superseded_specs,
        # D3-in-HTTP injection accounting (Phase 5)
        "d3_set_count": d3_set_count,
        "d3_status": d3_status,
        # Timing
        "spec_dispatch_ms": spec_dispatch_ms,
        "spec_end_ms": spec_end_ms,
        "spec_wall_s": round(spec_wall_s, 3),
        "canonical_tool_dispatch_ms": canonical_dispatch_ms,
        "canonical_tool_end_ms": canonical_end_ms,
        "real_tool_wall_s": round(tool_real_wall_s, 3),
        "real_overlap_ms": round(real_overlap_ms, 3),
        "critical_path_hidden_ms": round(real_overlap_ms, 3),
        "probe_parse_ok": any(probe.get("probe_parse_ok") for probe in probes),
        "probe_tool_name": representative_call.get("name") if representative_call else None,
        "probe_tool_args": representative_call.get("arguments") if representative_call else None,
        "probe_response_source": (
            representative_probe.get("probe_response_source")
            if representative_probe
            else None
        ),
        "probe_response": (
            representative_probe.get("probe_response")
            if representative_probe
            else None
        ),
        "probe_usage": representative_probe.get("usage") if representative_probe else None,
        "probe_logprobs": representative_logprobs,
        "probe_first_token_top1_prob": model_adapter.first_token_top1_prob(
            representative_logprobs
        ),
        "probe_span_min_prob_skip0": representative_decision.confidence,
        "args_partial_overlap": round(representative_decision.args_partial, 3),
        # Per-probe detail
        "probes": probes,
    }
    if has_tool:
        entry["tool_result_len"] = len(tool_result)
        entry["tool_result_preview"] = model_adapter.preview(tool_result)
    else:
        entry["final_answer"] = text.strip()
    return entry
