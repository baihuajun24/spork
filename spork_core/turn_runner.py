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


def _append_tool_messages(
    messages: list[dict],
    assistant_text: str,
    tool_result: str,
    *,
    tool_calls_structured: list[dict] | None = None,
) -> None:
    """Append history after a tool call, preserving OpenAI tool_calls when present."""
    if assistant_text:
        assistant_text = re.sub(r"^\s*<think>.*?</think>\s*", "", assistant_text, count=1, flags=re.DOTALL)
    if not tool_calls_structured:
        adapters.append_assistant_tool_messages(messages, assistant_text, tool_result)
        return
    tc = dict(tool_calls_structured[0])
    fn = dict(tc.get("function") or {})
    if not tc.get("id"):
        tc["id"] = "spork-tool-call-0"
    tc["type"] = tc.get("type") or "function"
    tc["function"] = fn
    messages.append({"role": "assistant", "content": assistant_text or "", "tool_calls": [tc]})
    messages.append({
        "role": "tool",
        "tool_call_id": tc["id"],
        "name": fn.get("name") or "",
        "content": tool_result,
    })


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
        # V4 chat-API path: render server-side from OpenAI messages; main decode
        # stops at the DSML tool-call closer. The DSML opener/body streams as plain
        # content text and is parsed by parse_main_tool_call below.
        res = await client.stream_chat(
            session,
            messages,
            max_tokens=main_max_tokens,
            stop=list(model_adapter.main_stop),
            seed=cfg.seed,
            tools=getattr(model_adapter, "tools", None) or None,
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
    main_call = _first_structured_tool_call(res) if uses_chat else None
    if main_call is None:
        main_call = model_adapter.parse_main_tool_call(text)
    has_tool = main_call is not None
    canonical_dispatch_ms = None
    canonical_end_ms = None
    tool_result = ""
    tool_wall_s = 0.0
    if has_tool:
        canonical_dispatch_ms = (time.time() - t0) * 1000.0
        tool_result, tool_wall_s = await asyncio.to_thread(executor.execute, main_call)
        canonical_end_ms = (time.time() - t0) * 1000.0
        _append_tool_messages(
            messages,
            text,
            tool_result,
            tool_calls_structured=res.get("tool_calls") if uses_chat else None,
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
        "baseline_text": text,
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

    async def on_first(delay_s: float, text: str) -> None:
        nonlocal first_token_s, first_token_abs, first_token_text
        first_token_s = delay_s
        first_token_abs = time.time()
        first_token_text = text
        first_token_event.set()

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
                on_first_token=on_first,
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
    probe_dispatch_after_first_token_ms = None

    try:
        await asyncio.wait_for(first_token_event.wait(), timeout=cfg.first_token_timeout_s)
    except asyncio.TimeoutError:
        first_token_timeout = True

    if not first_token_timeout:
        probe_dispatch_abs = time.time()
        probe_dispatch_ms = (probe_dispatch_abs - t0) * 1000.0
        probe_dispatch_after_first_token_ms = (
            (probe_dispatch_abs - first_token_abs) * 1000.0
            if first_token_abs is not None
            else None
        )
        if uses_chat:
            # V4 probe: prefill an assistant message ending in the DSML opener and
            # continue it. cot_so_far is the observed main prefix (first streamed
            # chunk here, matching the Qwen observed_main_prefix semantics).
            probe_prefix = model_adapter.build_probe_prefix(cfg.enable_thinking)
            cot_so_far = first_token_text or ""
            probe_messages = list(messages) + [
                {"role": "assistant", "content": cot_so_far + probe_prefix}
            ]
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
                continue_final_message=True,
                add_generation_prompt=False,
                extra=model_adapter.build_probe_constraint_extra()
                if hasattr(model_adapter, "build_probe_constraint_extra")
                else None,
            )
            probe_end_ms = (time.time() - t0) * 1000.0
            choice_content = (choice.get("message") or {}).get("content", "") or ""
            flat_lp = chat_logprobs_to_flat(choice)
            probe_logprobs = model_adapter.extract_probe_logprobs(flat_lp)
            probe_call = model_adapter.parse_probe_tool_call(choice_content, probe_prefix)
        else:
            probe_prompt = model_adapter.build_probe_prompt(
                messages,
                enable_thinking=cfg.enable_thinking,
                observed_main_prefix=first_token_text,
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
        _append_tool_messages(
            messages,
            text,
            tool_result,
            tool_calls_structured=main_res.get("tool_calls") if uses_chat else None,
        )

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
        **model_adapter.token_metrics(text),
        "probe_tool_name": probe_call["name"] if probe_call else None,
        "probe_tool_args": probe_call.get("arguments") if probe_call else None,
        "probe_parse_ok": probe_call is not None,
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
    """Config for D2 multi-probe retry."""
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
    # Extended strategy knobs
    theta_decay_per_retry: float = 0.0
    min_tokens_first_probe: int = 0
    hybrid_name_loose_first: bool = False
    hybrid_name_threshold: float = 0.60
    continue_after_dispatch: bool = False
    # D3-in-HTTP (Phase 5, 2026-06-14): optional async hooks that inject probe
    # draft tokens into the SPORK sidecar so the main decode accepts them at the
    # <tool_call> boundary. d3_set(probe_call_dict, prompt_len) registers draft
    # tokens; d3_clear() resets sidecar state after the turn. None = D3 disabled
    # (pure d1_d2). These do NOT affect acceptance metrics (gate_match/hit_name);
    # they only accelerate the main decode wall-clock.
    d3_set: Any = None
    d3_clear: Any = None


def _resolve_retry_main_stop(model_adapter: ModelAdapter, cfg: RetryTurnConfig) -> list[str]:
    """Resolve the D2 main decode stop strings.

    D2 retry must use the adapter's main stop for chat-model adapters such as
    DeepSeek-V4; the config default is the Qwen XML closer and is intentionally
    not a fallback for V4.
    """
    return list(model_adapter.main_stop)


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
        with main_text_lock:
            main_accumulated["text"] += text
            main_accumulated["token_count"] += 1
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
        main_task = asyncio.create_task(
            client.stream_chat(
                session,
                messages,
                max_tokens=main_max_tokens,
                stop=_resolve_retry_main_stop(model_adapter, cfg),
                seed=cfg.seed,
                tools=getattr(model_adapter, "tools", None) or None,
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
                stop=_resolve_retry_main_stop(model_adapter, cfg),
                seed=cfg.seed,
                on_first_token=on_first,
                on_token=on_token,
            )
        )

    # Retry loop state
    probes: list[dict] = []
    spec_task: asyncio.Task[SpeculativeToolExecution] | None = None
    spec_dispatch_ms: float | None = None
    committed_probe_call: dict | None = None
    committed_retry_idx: int | None = None
    superseded_specs = 0
    first_token_timeout = False
    d3_set_count = 0

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
                target_tokens = last_probe_tokens + cfg.retry_token_step
                deadline = time.time() + 10.0
                reached_target = False
                while time.time() < deadline:
                    if main_task.done():
                        break
                    with main_text_lock:
                        current_tokens = main_accumulated["token_count"]
                        current_text = main_accumulated["text"]
                    if current_tokens >= target_tokens:
                        if not cfg.snap_to_sentence or _at_sentence_boundary(current_text):
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
            try:
                if uses_chat:
                    # V4 probe: prefill assistant message ending in the DSML opener,
                    # continuing the observed CoT (observed_prefix == cot_so_far).
                    probe_prefix = model_adapter.build_probe_prefix(cfg.enable_thinking)
                    probe_messages = list(messages) + [
                        {"role": "assistant", "content": observed_prefix + probe_prefix}
                    ]
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
                    choice_content = (choice.get("message") or {}).get("content", "") or ""
                    flat_lp = chat_logprobs_to_flat(choice)
                    probe_logprobs = model_adapter.extract_probe_logprobs(flat_lp)
                    probe_call = model_adapter.parse_probe_tool_call(choice_content, probe_prefix)
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
                    probe_logprobs = model_adapter.extract_probe_logprobs(choice.get("logprobs"))
                    probe_call = model_adapter.parse_probe_tool_call(choice.get("text", ""))
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

    # Wait for main to finish
    main_res = await main_task
    main_end_ms = (time.time() - t0) * 1000.0
    text = main_res["text"]
    # D3-in-HTTP: clear sidecar draft-token state so the NEXT request/turn starts
    # clean (state hygiene — verified E2). Safe no-op if D3 disabled.
    d3_status = None
    if cfg.d3_clear is not None:
        try:
            d3_status = await cfg.d3_clear()
        except Exception:
            d3_status = None
    main_call = model_adapter.parse_main_tool_call(text)
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

    if has_tool and spec_task is not None and committed_probe_call is not None:
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
        )

    turn_end_ms = (time.time() - t0) * 1000.0

    # Compute accuracy metrics against main's call
    hit_name = bool(committed_probe_call and main_call and committed_probe_call.get("name") == main_call.get("name"))
    hit_args_exact = bool(hit_name and tool_calls.args_exact_match(
        committed_probe_call.get("arguments") if committed_probe_call else None,
        main_call.get("arguments") if main_call else None,
    ))

    entry = {
        "turn": turn,
        "fork_point": _fork_point(turn),
        "mode": "spork_retry",
        "gate_variant": "args_exact_strict",
        "max_retries": cfg.max_retries,
        "confidence_threshold": cfg.confidence_threshold,
        "continue_after_dispatch": cfg.continue_after_dispatch,
        "main_max_tokens": main_max_tokens,
        "main_first_token_ms": (first_token_s * 1000.0) if first_token_s is not None else None,
        "main_end_ms": main_end_ms,
        "turn_end_ms": turn_end_ms,
        "first_token_timeout": first_token_timeout,
        "baseline_has_tool_call": has_tool,
        "baseline_tool_name": main_call["name"] if has_tool else None,
        "baseline_tool_args": main_call.get("arguments") if has_tool else None,
        "baseline_text": text,
        **model_adapter.token_metrics(text),
        # Probe/retry info
        "num_probes": len(probes),
        "committed_retry_idx": committed_retry_idx,
        "committed_probe_name": committed_probe_call["name"] if committed_probe_call else None,
        "committed_probe_args": committed_probe_call.get("arguments") if committed_probe_call else None,
        # Gate decision
        "hit_name": hit_name,
        "hit_args_exact": hit_args_exact,
        "gate_match": gate_match,
        "spec_dispatched": spec_task is not None,
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
        # Per-probe detail
        "probes": probes,
    }
    if has_tool:
        entry["tool_result_len"] = len(tool_result)
        entry["tool_result_preview"] = model_adapter.preview(tool_result)
    else:
        entry["final_answer"] = text.strip()
    return entry
