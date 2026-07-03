"""SPORK-Adaptive (D2) probe scheduler: N-token retry loop + span-gate abort.

Spec (PI-decided 2026-04-29, see reports/adaptive_gaia_replay_plan_20260429.md §3.1):
  - Fire probe every N main-decoded tokens (cadence)
  - Each probe decodes `span_len` tokens (default 5)
  - Compute span confidence via `aggregation` over top-1 logprobs (mean_top1 default)
  - If span_conf < threshold → ABORT this probe, wait cadence more tokens, retry
  - If span_conf ≥ threshold → COMMIT: continue probe to tool_call close, parse
  - Max `max_retries` probe attempts per turn
  - Max 2 concurrent vLLM requests (main + ≤1 probe) — never dispatch new probe
    while a prior probe is in flight

Per-retry records capture all 4 span metrics (mean_top1, geo_mean, min_top1,
mean_top1_margin) so threshold / aggregation can be tuned post-hoc.

Not intended for E2E (tool execution) — replay variant only. E2E extension
comes next week per `reports/spork_adaptive_plan_20260429.md` timeline.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from . import tool_calls
from .adapters import ModelAdapter
from .vllm_client import VllmClient


@dataclass
class AdaptiveConfig:
    retry_cadence_tokens: int = 50
    probe_span_tokens: int = 20           # decode span: must include arg-content tokens past JSON boilerplate.
                                           # 5 tokens = mostly boilerplate (name + `", "arguments":`), can't discriminate.
                                           # 20 tokens = boilerplate + first ~15 arg-content tokens. See smoke
                                           # discussion 2026-04-29 in reports/adaptive_gaia_replay_plan_20260429.md §8.
    confidence_threshold: float = 0.9
    confidence_metric: str = "min_top1"    # URL characters have low per-token prob; min catches hallucinated args.
                                           # mean_top1 was too lenient (boilerplate tokens inflate the mean).
    max_retries_per_turn: int = 10
    commit_max_tokens: int = 100
    probe_max_tokens_phase1: int = None    # = probe_span_tokens; set in __post_init__
    seed: int = 42
    enable_thinking: bool = False
    first_token_timeout_s: float = 10.0

    def __post_init__(self):
        if self.probe_max_tokens_phase1 is None:
            self.probe_max_tokens_phase1 = self.probe_span_tokens


@dataclass
class AdaptiveRetryRecord:
    """One probe attempt (may or may not commit)."""
    retry_idx: int
    main_tokens_at_dispatch: int
    probe_dispatch_ms: float
    probe_phase1_end_ms: float
    phase1_span_scores: dict          # {mean_top1, geo_mean, min_top1, mean_top1_margin, span_len_used}
    phase1_metric_score: float | None # the metric used by the gate
    decision: str                     # "abort" | "commit"
    probe_phase2_end_ms: float | None = None  # only set if commit
    probe_parsed_call: dict | None = None
    probe_phase1_text: str = ""
    probe_phase2_text: str = ""
    phase1_logprobs: dict = field(default_factory=dict)


@dataclass
class AdaptiveOutcome:
    retries: list[AdaptiveRetryRecord]
    committed_retry_idx: int | None
    committed_call: dict | None
    committed_main_token_count: int | None    # main tokens decoded when commit fired
    first_token_ms: float | None
    main_end_ms: float
    main_text: str
    first_token_timeout: bool


def _select_metric(scores: dict, metric: str) -> float | None:
    v = scores.get(metric)
    if v is None:
        return None
    return float(v)


async def run_adaptive_probe_schedule(
    session: aiohttp.ClientSession,
    client: VllmClient,
    model_adapter: ModelAdapter,
    messages: list[dict],
    cfg: AdaptiveConfig,
) -> AdaptiveOutcome:
    """Drive main stream + concurrent probe scheduler.

    Coordination:
      - Main stream runs in one asyncio task, counting decoded tokens.
      - Scheduler loop in the main coroutine: on every N-token boundary,
        IF no probe is in flight AND retries_used < max_retries, dispatch probe.
      - Probe Phase-1 (span_len tokens + logprobs) is awaited inline.
        If span_conf >= threshold, continue to Phase-2 (up to commit_max_tokens).
        Otherwise abort; loop to wait for next boundary.
      - Once committed, no further probes fire; main stream continues to its own stop.
    """
    t0 = time.time()
    main_prompt = model_adapter.render_main_prompt(messages, enable_thinking=cfg.enable_thinking)

    first_token_event = asyncio.Event()
    first_token_s: float | None = None
    first_token_abs: float | None = None
    # Running accumulated main text + token count. Must grow across probe retries
    # so each probe sees the actual decoded CoT so far (not just first token).
    main_state = {"text": "", "token_count": 0}
    main_state_lock = asyncio.Lock()

    async def on_first(delay_s: float, _text: str) -> None:
        # on_first_token fires, then on_token fires for the same delta immediately after.
        # Don't accumulate here (on_token handles it) — just record timestamp + signal event.
        nonlocal first_token_s, first_token_abs
        first_token_s = delay_s
        first_token_abs = time.time()
        first_token_event.set()

    async def on_token(delta_text: str, token_count: int) -> None:
        async with main_state_lock:
            main_state["text"] += delta_text
            main_state["token_count"] = token_count

    # Start main stream. We rely on vllm_client to fire on_first when first delta
    # arrives, and on_token on each subsequent delta (if supported).
    # NOTE: the current vllm_client may only expose on_first_token. We tolerate
    # missing on_token — we'll use elapsed wall as a proxy for "scheduler tick".
    kwargs = dict(
        max_tokens=4096,  # think-mode headroom; main decodes until </tool_call> or final-answer
        stop=list(model_adapter.main_stop),
        seed=cfg.seed,
        on_first_token=on_first,
    )
    # Try to pass on_token if the client supports it; ignore otherwise.
    try:
        kwargs["on_token"] = on_token
    except Exception:
        pass

    main_task = asyncio.create_task(client.stream(session, main_prompt, **kwargs))

    # Wait for first token (fork-after-prefill constraint per VISION §11 Constraint 1)
    first_token_timeout = False
    try:
        await asyncio.wait_for(first_token_event.wait(), timeout=cfg.first_token_timeout_s)
    except asyncio.TimeoutError:
        first_token_timeout = True

    retries: list[AdaptiveRetryRecord] = []
    committed: AdaptiveRetryRecord | None = None
    committed_main_token_count: int | None = None
    committed_call: dict | None = None
    retries_used = 0
    next_cadence_threshold = cfg.retry_cadence_tokens  # next main-token count that triggers a probe
    stop_event = asyncio.Event()  # set when main task completes

    async def _wait_main_done():
        await main_task
        stop_event.set()
    wait_main = asyncio.create_task(_wait_main_done())

    if not first_token_timeout:
        # Scheduling loop: tick periodically to check whether main has crossed next cadence threshold
        while (retries_used < cfg.max_retries_per_turn
               and committed is None
               and not main_task.done()):
            async with main_state_lock:
                cur = main_state["token_count"]
                live_prefix = main_state["text"]
            if cur >= next_cadence_threshold:
                # Dispatch probe
                retry_idx = retries_used
                retries_used += 1
                main_tokens_at_dispatch = cur
                probe_dispatch_abs = time.time()
                probe_dispatch_ms = (probe_dispatch_abs - t0) * 1000.0
                observed_prefix = live_prefix
                import os, sys
                if os.environ.get("ADAPTIVE_DEBUG"):
                    print(f"[ADAPTIVE_DEBUG] retry {retry_idx}: main_tokens={cur} "
                          f"observed_len={len(observed_prefix)} "
                          f"prefix_head={observed_prefix[:80]!r}", file=sys.stderr, flush=True)

                # Phase 1: decode span_len tokens with logprobs
                probe_prompt = model_adapter.build_probe_prompt(
                    messages,
                    enable_thinking=cfg.enable_thinking,
                    observed_main_prefix=observed_prefix,
                )
                try:
                    choice, phase1_wall_s = await client.complete(
                        session,
                        probe_prompt,
                        max_tokens=cfg.probe_max_tokens_phase1,
                        stop=list(model_adapter.probe_stop),
                        logprobs=5,
                        seed=cfg.seed,
                    )
                except Exception as e:
                    retries.append(AdaptiveRetryRecord(
                        retry_idx=retry_idx, main_tokens_at_dispatch=main_tokens_at_dispatch,
                        probe_dispatch_ms=probe_dispatch_ms,
                        probe_phase1_end_ms=(time.time() - t0) * 1000.0,
                        phase1_span_scores={}, phase1_metric_score=None,
                        decision="abort", probe_phase1_text=f"[ERROR:{e!r}]",
                    ))
                    next_cadence_threshold += cfg.retry_cadence_tokens
                    continue
                phase1_end_ms = (time.time() - t0) * 1000.0
                phase1_text = choice.get("text", "") or ""
                lp = model_adapter.extract_probe_logprobs(choice.get("logprobs"))
                scores = tool_calls.span_scores(
                    lp, span_len=cfg.probe_span_tokens, skip_first=True
                )
                metric = _select_metric(scores, cfg.confidence_metric)

                # Gate
                if metric is None or metric < cfg.confidence_threshold:
                    retries.append(AdaptiveRetryRecord(
                        retry_idx=retry_idx, main_tokens_at_dispatch=main_tokens_at_dispatch,
                        probe_dispatch_ms=probe_dispatch_ms,
                        probe_phase1_end_ms=phase1_end_ms,
                        phase1_span_scores=scores, phase1_metric_score=metric,
                        decision="abort", probe_phase1_text=phase1_text,
                        phase1_logprobs=lp,
                    ))
                    next_cadence_threshold += cfg.retry_cadence_tokens
                    continue

                # Commit: continue probe to tool_call close via Phase-2
                # We build a Phase-2 prompt = probe_prompt + phase1_text (continue
                # from where Phase-1 left off). Uses prefix cache naturally.
                phase2_prompt = probe_prompt + phase1_text
                try:
                    choice2, phase2_wall_s = await client.complete(
                        session,
                        phase2_prompt,
                        max_tokens=cfg.commit_max_tokens,
                        stop=list(model_adapter.probe_stop),
                        logprobs=None,
                        seed=cfg.seed,
                    )
                except Exception as e:
                    retries.append(AdaptiveRetryRecord(
                        retry_idx=retry_idx, main_tokens_at_dispatch=main_tokens_at_dispatch,
                        probe_dispatch_ms=probe_dispatch_ms,
                        probe_phase1_end_ms=phase1_end_ms,
                        phase1_span_scores=scores, phase1_metric_score=metric,
                        decision="abort", probe_phase1_text=phase1_text,
                        phase1_logprobs=lp, probe_phase2_text=f"[PHASE2_ERROR:{e!r}]",
                    ))
                    next_cadence_threshold += cfg.retry_cadence_tokens
                    continue
                phase2_end_ms = (time.time() - t0) * 1000.0
                phase2_text = choice2.get("text", "") or ""
                # Full probe text includes both phases; parser works on phase1+phase2 combined
                full_probe_text = phase1_text + phase2_text
                parsed_call = model_adapter.parse_probe_tool_call(full_probe_text)

                record = AdaptiveRetryRecord(
                    retry_idx=retry_idx, main_tokens_at_dispatch=main_tokens_at_dispatch,
                    probe_dispatch_ms=probe_dispatch_ms,
                    probe_phase1_end_ms=phase1_end_ms,
                    phase1_span_scores=scores, phase1_metric_score=metric,
                    decision="commit", probe_phase2_end_ms=phase2_end_ms,
                    probe_parsed_call=parsed_call,
                    probe_phase1_text=phase1_text, probe_phase2_text=phase2_text,
                    phase1_logprobs=lp,
                )
                retries.append(record)
                if parsed_call is not None:
                    committed = record
                    committed_call = parsed_call
                    committed_main_token_count = main_tokens_at_dispatch
                    break
                else:
                    # Span accepted but parse failed — advance cadence, continue
                    next_cadence_threshold += cfg.retry_cadence_tokens
                    continue
            else:
                # Not yet time for next probe. Sleep briefly. vLLM scheduling scale is ~10-30ms/token.
                await asyncio.sleep(0.02)

    # Ensure main completes
    await wait_main
    main_end_ms = (time.time() - t0) * 1000.0
    main_res = main_task.result()
    main_text = main_res.get("text", "") if isinstance(main_res, dict) else ""

    return AdaptiveOutcome(
        retries=retries,
        committed_retry_idx=(committed.retry_idx if committed else None),
        committed_call=committed_call,
        committed_main_token_count=committed_main_token_count,
        first_token_ms=(first_token_s * 1000.0) if first_token_s is not None else None,
        main_end_ms=main_end_ms,
        main_text=main_text,
        first_token_timeout=first_token_timeout,
    )
