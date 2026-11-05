"""Small async client for vLLM /v1/completions."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import aiohttp


@dataclass
class VllmClient:
    base_url: str = "http://localhost:8000/v1"
    model: str = ""
    timeout_s: int = 600
    main_stop_token_ids: tuple[int, ...] = (151657,)
    tool_choice: str | dict | None = None
    parallel_tool_calls: bool | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    include_reasoning: bool | None = None

    @property
    def completions_url(self) -> str:
        return self.base_url.rstrip("/") + "/completions"

    @property
    def models_url(self) -> str:
        return self.base_url.rstrip("/") + "/models"

    @property
    def chat_completions_url(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    @property
    def root_url(self) -> str:
        root = self.base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
        return root

    async def render_chat(
        self,
        session: aiohttp.ClientSession,
        body: dict[str, Any],
        *,
        timeout_s: float = 60,
    ) -> tuple[dict, str]:
        """Use the patched render route when present, otherwise native tokenize."""
        for endpoint in ("/v1/chat/completions/render", "/tokenize"):
            async with session.post(
                self.root_url + endpoint,
                json=body,
                timeout=aiohttp.ClientTimeout(total=timeout_s),
            ) as resp:
                if getattr(resp, "status", None) == 404 and endpoint.endswith("/render"):
                    continue
                resp.raise_for_status()
                return await resp.json(), endpoint
        raise RuntimeError("neither chat render nor tokenize endpoint is available")

    async def wait_ready(self, deadline_s: float = 120.0) -> dict:
        end = time.time() + deadline_s
        last: Exception | None = None
        async with aiohttp.ClientSession(trust_env=False) as session:
            while time.time() < end:
                try:
                    async with session.get(self.models_url, timeout=5) as resp:
                        resp.raise_for_status()
                        return await resp.json()
                except Exception as exc:  # noqa: BLE001
                    last = exc
                    await asyncio.sleep(2)
        raise RuntimeError(f"vLLM not ready after {deadline_s}s: {last}")

    async def render_chat_token_count(
        self,
        session: aiohttp.ClientSession,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
    ) -> int:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools
            if self.tool_choice is not None:
                payload["tool_choice"] = self.tool_choice
            if self.parallel_tool_calls is not None:
                payload["parallel_tool_calls"] = self.parallel_tool_calls
        if self.chat_template_kwargs:
            payload["chat_template_kwargs"] = dict(self.chat_template_kwargs)
        rendered, _ = await self.render_chat(session, payload)
        token_ids = rendered.get("token_ids", rendered.get("tokens"))
        if token_ids is None:
            raise ValueError("chat render/tokenize response lacks token IDs")
        return len(token_ids)

    def request_body(
        self,
        prompt: str,
        *,
        max_tokens: int,
        stop: list[str] | None = None,
        logprobs: int | None = None,
        temperature: float = 0.0,
        stream: bool = False,
        seed: int = 42,
    ) -> dict:
        body: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": stop or [],
            "include_stop_str_in_output": True,
            "stream": stream,
            "seed": seed,
        }
        if stream:
            body["stream_options"] = {"include_usage": True}
        if stop and any("<tool_call>" in s for s in stop):
            body["stop_token_ids"] = list(self.main_stop_token_ids)
        if logprobs is not None:
            body["logprobs"] = logprobs
        return body

    async def complete(
        self,
        session: aiohttp.ClientSession,
        prompt: str,
        *,
        max_tokens: int,
        stop: list[str] | None = None,
        logprobs: int | None = None,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> tuple[dict, float]:
        body = self.request_body(
            prompt,
            max_tokens=max_tokens,
            stop=stop,
            logprobs=logprobs,
            temperature=temperature,
            stream=False,
            seed=seed,
        )
        last: Exception | None = None
        for attempt in range(3):
            try:
                t0 = time.time()
                async with session.post(
                    self.completions_url,
                    json=body,
                    timeout=aiohttp.ClientTimeout(total=self.timeout_s),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                choice = dict(data["choices"][0])
                choice["usage"] = data.get("usage")
                return choice, time.time() - t0
            except Exception as exc:  # noqa: BLE001
                last = exc
                await asyncio.sleep(min(2 ** attempt, 5))
        raise RuntimeError(f"vLLM complete failed after 3 attempts: {last}")

    async def stream(
        self,
        session: aiohttp.ClientSession,
        prompt: str,
        *,
        max_tokens: int,
        stop: list[str] | None = None,
        temperature: float = 0.0,
        seed: int = 42,
        on_first_token: Callable[[float, str], Awaitable[None]] | None = None,
        on_token: Callable[[str, int], Awaitable[None]] | None = None,
    ) -> dict:
        body = self.request_body(
            prompt,
            max_tokens=max_tokens,
            stop=stop,
            temperature=temperature,
            stream=True,
            seed=seed,
        )
        t0 = time.time()
        full_text = ""
        usage: dict[str, Any] | None = None
        first_token_ts: float | None = None
        last: Exception | None = None
        delta_count = 0
        for attempt in range(3):
            finished = False
            try:
                async with session.post(
                    self.completions_url,
                    json=body,
                    timeout=aiohttp.ClientTimeout(total=self.timeout_s),
                ) as resp:
                    resp.raise_for_status()
                    async for raw_line in resp.content:
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line or not line.startswith("data:"):
                            continue
                        payload = line[len("data:"):].strip()
                        if payload == "[DONE]":
                            finished = True
                            break
                        try:
                            chunk = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        if chunk.get("usage") is not None:
                            usage = chunk["usage"]
                        choices = chunk.get("choices") or []
                        delta_text = choices[0].get("text", "") if choices else ""
                        if delta_text:
                            if first_token_ts is None:
                                first_token_ts = time.time()
                                if on_first_token is not None:
                                    await on_first_token(first_token_ts - t0, delta_text)
                                    await asyncio.sleep(0)
                            full_text += delta_text
                            delta_count += 1
                            if on_token is not None:
                                # vLLM SSE chunks usually ≈ 1 token each. Pass delta_count as
                                # a token-proxy so adaptive schedulers can trigger at intervals.
                                try:
                                    await on_token(delta_text, delta_count)
                                except Exception:
                                    pass
                if finished:
                    return {
                        "text": full_text,
                        "first_token_s": (
                            first_token_ts - t0 if first_token_ts is not None else None
                        ),
                        "wall_s": time.time() - t0,
                        "usage": usage,
                    }
            except Exception as exc:  # noqa: BLE001
                last = exc
                await asyncio.sleep(min(2 ** attempt, 5))
                full_text = ""
                first_token_ts = None
                usage = None
        raise RuntimeError(f"vLLM stream failed after 3 attempts: {last}")

    # --- chat-completions API (additive; used by chat-API adapters, e.g. V4) ---
    def chat_request_body(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float = 0.0,
        stop: list[str] | None = None,
        seed: int = 42,
        stream: bool = False,
        logprobs: bool = True,
        top_logprobs: int = 5,
        tools: list[dict] | None = None,
        continue_final_message: bool = False,
        add_generation_prompt: bool = True,
        extra: dict | None = None,
    ) -> dict:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": stop or [],
            "include_stop_str_in_output": True,
            "seed": seed,
            "stream": stream,
            "logprobs": bool(logprobs),
        }
        if logprobs:
            body["top_logprobs"] = top_logprobs
        if tools:
            body["tools"] = tools
            if self.tool_choice is not None:
                body["tool_choice"] = self.tool_choice
            if self.parallel_tool_calls is not None:
                body["parallel_tool_calls"] = self.parallel_tool_calls
        if self.chat_template_kwargs:
            body["chat_template_kwargs"] = dict(self.chat_template_kwargs)
        if self.include_reasoning is not None:
            body["include_reasoning"] = self.include_reasoning
        if stream:
            body["stream_options"] = {"include_usage": True}
        # vLLM continue_final_message requires add_generation_prompt=False.
        if continue_final_message:
            body["continue_final_message"] = True
        body["add_generation_prompt"] = add_generation_prompt
        if extra:
            body.update(extra)
        return body

    async def complete_chat(
        self,
        session: aiohttp.ClientSession,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float = 0.0,
        stop: list[str] | None = None,
        seed: int = 42,
        logprobs: bool = True,
        top_logprobs: int = 5,
        tools: list[dict] | None = None,
        continue_final_message: bool = False,
        add_generation_prompt: bool = True,
        extra: dict | None = None,
    ) -> tuple[dict, float]:
        """Non-streaming /v1/chat/completions. Returns (choice_dict, wall_s).

        The returned ``choice`` is the OpenAI chat choice dict
        ``{"message": {...}, "logprobs": {"content": [...]}, ...}``. Callers can
        read ``choice["message"]["content"]`` for the generated text and pass the
        whole choice through ``chat_logprobs_to_flat`` for the D2 confidence gate.
        """
        body = self.chat_request_body(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            seed=seed,
            stream=False,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            tools=tools,
            continue_final_message=continue_final_message,
            add_generation_prompt=add_generation_prompt,
            extra=extra,
        )
        last: Exception | None = None
        for attempt in range(3):
            try:
                t0 = time.time()
                async with session.post(
                    self.chat_completions_url,
                    json=body,
                    timeout=aiohttp.ClientTimeout(total=self.timeout_s),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                choice = dict(data["choices"][0])
                choice["usage"] = data.get("usage")
                return choice, time.time() - t0
            except Exception as exc:  # noqa: BLE001
                last = exc
                await asyncio.sleep(min(2 ** attempt, 5))
        raise RuntimeError(f"vLLM complete_chat failed after 3 attempts: {last}")

    async def stream_chat(
        self,
        session: aiohttp.ClientSession,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float = 0.0,
        stop: list[str] | None = None,
        seed: int = 42,
        tools: list[dict] | None = None,
        continue_final_message: bool = False,
        add_generation_prompt: bool = True,
        extra: dict | None = None,
        capture_stream_events: bool = False,
        on_first_token: Callable[[float, str], Awaitable[None]] | None = None,
        on_token: Callable[[str, int], Awaitable[None]] | None = None,
    ) -> dict:
        """Streaming /v1/chat/completions. Returns the same shape as ``stream``:
        ``{"text": str, "first_token_s": float|None, "wall_s": float}``.

        Reads content, reasoning/reasoning_content, and structured tool-call deltas.
        vLLM 0.24 exposes DeepSeek-V4 thinking as ``delta.reasoning``; older
        deployments may use ``reasoning_content``. Reasoning tokens count as main
        decode tokens and trigger the callbacks used by SPORK's early probes.
        """
        _think_prefix = getattr(self, "main_think_prefix", None)
        _did_think = bool(_think_prefix) and not continue_final_message
        if _did_think:
            messages = list(messages) + [{"role": "assistant", "content": _think_prefix}]
            continue_final_message = True
            add_generation_prompt = False
        body = self.chat_request_body(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            seed=seed,
            stream=True,
            logprobs=False,
            tools=tools,
            continue_final_message=continue_final_message,
            add_generation_prompt=add_generation_prompt,
            extra=extra,
        )
        t0 = time.time()
        full_text = ""
        full_reasoning = ""
        usage: dict[str, Any] | None = None
        tool_call_parts: dict[int, dict[str, Any]] = {}
        stream_events: list[dict[str, Any]] = []
        first_token_ts: float | None = None
        last: Exception | None = None
        delta_count = 0
        for attempt in range(3):
            finished = False
            try:
                async with session.post(
                    self.chat_completions_url,
                    json=body,
                    timeout=aiohttp.ClientTimeout(total=self.timeout_s),
                ) as resp:
                    if resp.status >= 400:
                        detail = (await resp.text())[:2048]
                        raise RuntimeError(
                            f"HTTP {resp.status} {resp.reason}: {detail}"
                        )
                    async for raw_line in resp.content:
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line or not line.startswith("data:"):
                            continue
                        payload = line[len("data:"):].strip()
                        if payload == "[DONE]":
                            finished = True
                            break
                        try:
                            chunk = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        if chunk.get("usage") is not None:
                            usage = chunk["usage"]
                        choices = chunk.get("choices") or []
                        delta = choices[0].get("delta", {}) if choices else {}
                        delta_text = delta.get("content") or "" if isinstance(delta, dict) else ""
                        reasoning_delta = ""
                        if isinstance(delta, dict):
                            reasoning_delta = (
                                delta.get("reasoning")
                                or delta.get("reasoning_content")
                                or ""
                            )
                            if capture_stream_events:
                                stream_events.append(chunk)
                            for tc in delta.get("tool_calls") or []:
                                if not isinstance(tc, dict):
                                    continue
                                idx = int(tc.get("index", len(tool_call_parts)))
                                cur = tool_call_parts.setdefault(
                                    idx,
                                    {
                                        "id": None,
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    },
                                )
                                if tc.get("id"):
                                    cur["id"] = tc["id"]
                                if tc.get("type"):
                                    cur["type"] = tc["type"]
                                fn_delta = tc.get("function") or {}
                                if isinstance(fn_delta, dict):
                                    fn = cur.setdefault("function", {"name": "", "arguments": ""})
                                    if fn_delta.get("name"):
                                        fn["name"] = (fn.get("name") or "") + fn_delta["name"]
                                    if fn_delta.get("arguments"):
                                        fn["arguments"] = (
                                            fn.get("arguments") or ""
                                        ) + fn_delta["arguments"]
                                if first_token_ts is None:
                                    first_token_ts = time.time()
                                    if on_first_token is not None:
                                        await on_first_token(first_token_ts - t0, full_text)
                                        await asyncio.sleep(0)
                        for emitted_text, is_reasoning in (
                            (reasoning_delta, True),
                            (delta_text, False),
                        ):
                            if not emitted_text:
                                continue
                            if first_token_ts is None:
                                first_token_ts = time.time()
                                if on_first_token is not None:
                                    await on_first_token(first_token_ts - t0, emitted_text)
                                    await asyncio.sleep(0)
                            if is_reasoning:
                                full_reasoning += emitted_text
                            else:
                                full_text += emitted_text
                            delta_count += 1
                            if on_token is not None:
                                try:
                                    await on_token(emitted_text, delta_count)
                                except Exception:
                                    pass
                if finished:
                    tool_calls = [
                        tc for _, tc in sorted(tool_call_parts.items())
                        if (tc.get("function") or {}).get("name")
                    ]
                    if _did_think:
                        full_text = _think_prefix + full_text
                    return {
                        "text": full_text,
                        "reasoning": full_reasoning,
                        "usage": usage,
                        "tool_calls": tool_calls,
                        "stream_events": stream_events,
                        "first_token_s": (
                            first_token_ts - t0 if first_token_ts is not None else None
                        ),
                        "wall_s": time.time() - t0,
                    }
            except Exception as exc:  # noqa: BLE001
                last = exc
                await asyncio.sleep(min(2 ** attempt, 5))
                full_text = ""
                full_reasoning = ""
                usage = None
                tool_call_parts = {}
                stream_events = []
                first_token_ts = None
                delta_count = 0
        raise RuntimeError(f"vLLM stream_chat failed after 3 attempts: {last}")


def chat_logprobs_to_flat(choice: dict | None) -> dict:
    """Convert an OpenAI chat ``choice`` into the flat logprobs schema that
    ``tool_calls.extract_full_logprobs`` / ``span_min_prob`` expect.

    Input: ``choice["logprobs"]["content"]`` is a list of per-token dicts
    ``{"token": str, "logprob": float, "top_logprobs": [{"token","logprob"}, ...]}``.
    Output: ``{"tokens": [...], "token_logprobs": [...],
               "top_logprobs": [{tok: lp, ...}, ...]}`` (the /v1/completions shape).
    Returns the EMPTY_LOGPROBS-shaped dict when logprobs are absent.
    """
    empty = {"tokens": [], "token_logprobs": [], "top_logprobs": []}
    if not choice:
        return dict(empty)
    lp = choice.get("logprobs") or {}
    content = lp.get("content")
    if not content:
        return dict(empty)
    tokens: list = []
    token_logprobs: list = []
    top_logprobs: list = []
    for pos in content:
        if not isinstance(pos, dict):
            continue
        tokens.append(pos.get("token"))
        token_logprobs.append(pos.get("logprob"))
        top_map: dict = {}
        for alt in pos.get("top_logprobs") or []:
            if isinstance(alt, dict) and "token" in alt:
                top_map[alt["token"]] = alt.get("logprob")
        top_logprobs.append(top_map)
    return {
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
    }
